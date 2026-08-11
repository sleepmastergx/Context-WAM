"""Gate: the per-layer memory must advance EXACTLY once per window.

Fast-WAM trains with one forward pass per window and infers with 20 denoising
steps. If the fast weights moved per forward pass, deploy would advance the
state 20x faster than training — the train/deploy operator mismatch that cost
92% divergence on RMBench and a silently-zeroed memory on MoveCube.

Asserts:
  1. read_layer() is PURE — 20 reads leave the state bit-identical to 0 reads.
  2. A "training" window (1 read) and an "inference" window (20 reads) end in
     the SAME state, so the two regimes stay the same operator.
  3. advance() does move the state (the test is not vacuously passing).
  4. At init the gate is ~0, so a memory-equipped model starts equal to stock.
"""
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "context_wam"))
from per_layer_memory import PerLayerEpisodeMemory  # noqa: E402


def snapshot(mem):
    return [tuple(t.clone() for t in s) if s is not None else None
            for s in mem._states]


def same(a, b):
    for sa, sb in zip(a, b):
        if (sa is None) != (sb is None):
            return False
        if sa is None:
            continue
        for ta, tb in zip(sa, sb):
            if not torch.equal(ta, tb):
                return False
    return True


def main():
    torch.manual_seed(0)
    L, B, D = 5, 2, 1024          # fastwam_decoupled: 5 action layers
    mem = PerLayerEpisodeMemory(n_layers=L, hidden_dim=D, latent_channels=48,
                                proprio_dim=14)
    lat = torch.randn(B, 48, 3, 24, 20)
    pro = torch.randn(B, 14)

    # --- 4. init parity ----------------------------------------------------
    mem.advance(lat, pro)
    add = mem.read_layer(0, 32, torch.float32)
    gate_norm = float(add.norm())
    assert gate_norm < 1.0, f"gate not near zero at init: {gate_norm}"

    # --- 1 + 2. read purity across denoising -------------------------------
    before = snapshot(mem)
    m_before = [t.clone() for t in mem._m]
    for step in range(20):                      # a full inference denoise loop
        for l in range(L):
            mem.read_layer(l, 32, torch.float32)
    after = snapshot(mem)
    assert same(before, after), "FAIL: reads mutated the fast weights"
    assert all(torch.equal(a, b) for a, b in zip(m_before, mem._m)), \
        "FAIL: reads mutated the readout"

    # training regime = 1 read; inference regime = 20 reads; same end state
    mem2 = PerLayerEpisodeMemory(n_layers=L, hidden_dim=D, latent_channels=48,
                                 proprio_dim=14)
    mem2.load_state_dict(mem.state_dict())
    mem2.reset()
    mem2.advance(lat, pro)
    for l in range(L):
        mem2.read_layer(l, 32, torch.float32)   # single training forward
    assert same(snapshot(mem), snapshot(mem2)), \
        "FAIL: 1-step and 20-step regimes diverge"

    # --- 3. advance() actually moves the state -----------------------------
    mem.advance(lat, pro)
    assert not same(before, snapshot(mem)), "FAIL: advance() is a no-op"

    # multi-window carry, states stay finite
    mem.reset()
    for w in range(6):
        s = mem.advance(torch.randn(B, 48, 3, 24, 20), pro)
        assert all(torch.isfinite(t).all() for st in mem._states for t in st), \
            f"non-finite fast weights at window {w}"
    print(f"PASS  write-once verified: 20 reads == 0 reads == 1 read; "
          f"advance() moves state; 6-window carry finite; "
          f"init gate norm {gate_norm:.2e}")
    print(f"      params: {sum(p.numel() for p in mem.parameters())/1e6:.2f}M "
          f"for {L} layers")


if __name__ == "__main__":
    main()
