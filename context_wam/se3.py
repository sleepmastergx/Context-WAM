"""Minimal quaternion (wxyz) / rotation-vector helpers, pure numpy.

Used by gpu_cache (window-relative EEF deltas) and the eval client (composing
absolute EEF targets from deltas). Deltas are rotation VECTORS of the relative
rotation R_0^T R_k -- continuous, no Euler wrap (roll sits at +-pi in this data).
"""
import numpy as np


def quat_mul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw*bw - ax*bx - ay*by - az*bz,
                     aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw], axis=-1)


def quat_conj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def quat_to_rotvec(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., :1] < 0, -q, q)                     # shortest arc
    w = np.clip(q[..., 0], -1.0, 1.0)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2.0 * np.arctan2(s[..., 0], w)
    scale = np.where(s[..., 0] > 1e-12, ang / np.maximum(s[..., 0], 1e-12), 2.0)
    return v * scale[..., None]


def rotvec_to_quat(r):
    ang = np.linalg.norm(r, axis=-1, keepdims=True)
    half = 0.5 * ang
    k = np.where(ang > 1e-12, np.sin(half) / np.maximum(ang, 1e-12), 0.5)
    return np.concatenate([np.cos(half), r * k], axis=-1)


def relative_rotvec(q0, qk):
    """rotvec of R_0^T R_k, broadcasting q0 [...,4] against qk [...,4]."""
    return quat_to_rotvec(quat_mul(quat_conj(q0), qk))


def compose(p0, q0, dp, rv):
    """Absolute (p, q) from anchor (p0, q0) and deltas (dp, rotvec)."""
    return p0 + dp, quat_mul(q0, rotvec_to_quat(rv))
