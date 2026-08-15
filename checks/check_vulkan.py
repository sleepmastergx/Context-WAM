"""Gate: can this machine render for sim eval, and on what?

Run this FIRST on any new pod -- before the ~3 min of imports it takes sapien
to reach the same conclusion, and with a far more useful error than its
"failed to find a rendering device".

    python checks/check_vulkan.py

Exit codes:  0 = GPU rendering available
             1 = only CPU rendering (lavapipe) available -- eval works, slowly
             2 = no Vulkan at all

Background: NVIDIA_DRIVER_CAPABILITIES must include `graphics`, but that alone
is not enough. The NVIDIA ICD also has to open a DRM render node, and some
RunPod pods bind-mount /dev/dri/renderD* from the host with an owner outside
the container's user namespace. Root then gets EACCES -- cap_dac_override does
not cross a userns boundary -- and the ICD fails to initialise. There is no fix
from inside such a container: mknod is barred in a userns, and the nodes are
busy bind-mounts so they cannot be replaced.
"""
import ctypes
import glob
import os
import sys

NVIDIA_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"
LAVAPIPE_ICD = "/usr/share/vulkan/icd.d/lvp_icd.json"


class _AppInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("pApplicationName", ctypes.c_char_p),
                ("applicationVersion", ctypes.c_uint32),
                ("pEngineName", ctypes.c_char_p),
                ("engineVersion", ctypes.c_uint32),
                ("apiVersion", ctypes.c_uint32)]


class _InstInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(_AppInfo)),
                ("enabledLayerCount", ctypes.c_uint32),
                ("ppEnabledLayerNames", ctypes.c_void_p),
                ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p)]


DEVICE_TYPES = {0: "OTHER", 1: "INTEGRATED_GPU", 2: "DISCRETE_GPU",
                3: "VIRTUAL_GPU", 4: "CPU"}


def enumerate_devices(icd_path):
    """Return (devices, error) for one ICD, without importing sapien/torch."""
    if not os.path.exists(icd_path):
        return [], f"no such ICD manifest: {icd_path}"
    os.environ["VK_ICD_FILENAMES"] = icd_path
    os.environ["VK_DRIVER_FILES"] = icd_path   # newer loaders prefer this name
    try:
        lib = ctypes.CDLL("libvulkan.so.1")
    except OSError as e:
        return [], f"libvulkan.so.1 not loadable: {e}"

    app = _AppInfo(0, None, b"gate", 1, b"gate", 1, (1 << 22) | (2 << 12))
    ci = _InstInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
    inst = ctypes.c_void_p()
    if lib.vkCreateInstance(ctypes.byref(ci), None, ctypes.byref(inst)) != 0:
        return [], "vkCreateInstance failed (VK_ERROR_INCOMPATIBLE_DRIVER)"

    count = ctypes.c_uint32(0)
    lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(count), None)
    if count.value == 0:
        return [], "ICD loaded but exposes no physical device"
    handles = (ctypes.c_void_p * count.value)()
    lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(count), handles)

    out = []
    for i in range(count.value):
        buf = (ctypes.c_ubyte * 1024)()
        lib.vkGetPhysicalDeviceProperties(handles[i], ctypes.byref(buf))
        dtype = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint32))[4]
        name = bytes(buf)[20:276].split(b"\0")[0].decode(errors="replace")
        out.append((name, DEVICE_TYPES.get(dtype, str(dtype))))
    return out, None


def diagnose_render_nodes():
    nodes = sorted(glob.glob("/dev/dri/renderD*"))
    if not nodes:
        print("  /dev/dri: no render node present at all")
        return
    for n in nodes:
        st = os.stat(n)
        ok = os.access(n, os.R_OK | os.W_OK)
        print(f"  {n}: uid={st.st_uid} gid={st.st_gid} "
              f"mode={oct(st.st_mode & 0o777)} openable={ok}")
        if not ok:
            print("    ^ EACCES even as root => owner is outside this "
                  "container's user namespace.")
            print("      This pod cannot do GPU rendering. Recreate the pod "
                  "so /dev/dri is mapped, or fall back to lavapipe.")


def main():
    caps = os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "<unset>")
    print(f"NVIDIA_DRIVER_CAPABILITIES = {caps}")
    if "graphics" not in caps and "all" not in caps:
        print("  ^ missing `graphics` -- recreate the pod with "
              "NVIDIA_DRIVER_CAPABILITIES=all")

    print(f"\n[1] NVIDIA ICD  ({NVIDIA_ICD})")
    devs, err = enumerate_devices(NVIDIA_ICD)
    if err:
        print(f"  UNUSABLE: {err}")
        diagnose_render_nodes()
    else:
        for name, dtype in devs:
            print(f"  {name!r}  [{dtype}]")
        if any(d != "CPU" for _, d in devs):
            print("\nRESULT: GPU rendering available. Use:")
            print(f"  export VK_ICD_FILENAMES={NVIDIA_ICD}")
            return 0

    print(f"\n[2] lavapipe ICD  ({LAVAPIPE_ICD})")
    devs, err = enumerate_devices(LAVAPIPE_ICD)
    if err:
        print(f"  UNUSABLE: {err}")
        print("  install it with: apt-get update && "
              "apt-get install -y mesa-vulkan-drivers")
        print("\nRESULT: no Vulkan renderer at all -- sim eval cannot run.")
        return 2
    for name, dtype in devs:
        print(f"  {name!r}  [{dtype}]")
    print("\nRESULT: CPU rendering only. Sim eval runs but is very slow;")
    print("        ManiSkill also needs render_backend='cpu' (see")
    print("        dp/eval_dp.py --render-backend).")
    print(f"  export VK_ICD_FILENAMES={LAVAPIPE_ICD}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
