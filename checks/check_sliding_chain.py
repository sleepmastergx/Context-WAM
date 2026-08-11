"""CPU gate for the sliding chain — run before ANY GPU spend.

    1. bookkeeping     j_max matches the settled strictly-before convention
    2. streaming       deploy-style step-by-step detached advance() ends in the
                       same readout as the batched in-graph training chain
                       (the check_chunk_equiv analog — a mismatch here is the
                       silent 92% train/deploy bug)
    3. gradient reach  loss on the readouts produces gradient on the cell
                       params AND the learned init (deep credit through the
                       whole chain); a no_grad chain produces none
    4. checkpointing   segmented torch.utils.checkpoint chain == plain chain,
                       gradients and readouts alike

Runs on CPU in a few seconds:  python checks/check_sliding_chain.py
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "context_wam"))

from per_layer_memory import PerLayerEpisodeMemory   # noqa: E402
from sliding_chain import SlidingChain               # noqa: E402


class FakeCache:
    """Stride-1 contiguous per-episode rows, like the real converter output."""

    def __init__(self, lengths, seed=0):
        g = torch.Generator().manual_seed(seed)
        lat, sta, ep, start = [], [], [], []
        for e, T in enumerate(lengths):
            n = T - 33 + 1
            lat.append(torch.randn(n, 48, 3, 16, 32, generator=g) * 0.5)
            sta.append(torch.randn(n, 8, generator=g))
            ep.append(torch.full((n,), e, dtype=torch.int32))
            start.append(torch.arange(n, dtype=torch.int32))
        self.latents = torch.cat(lat)
        self.states = torch.cat(sta)
        self.ep = torch.cat(ep)
        self.start = torch.cat(start)
        self.row0 = {}
        off = 0
        for e, T in enumerate(lengths):
            self.row0[e] = off
            off += T - 33 + 1


def build_memory():
    torch.manual_seed(1)
    return PerLayerEpisodeMemory(
        n_layers=2, hidden_dim=64, latent_channels=48, proprio_dim=8,
        d_k=16, d_v=16, d_hidden=32, d_out=24, chunk=1).float()


def main():
    lengths = [300, 260]
    cache = FakeCache(lengths)
    mem = build_memory()
    chain = SlidingChain(cache, mem, w=8)

    # ---- 1: bookkeeping --------------------------------------------------
    assert chain.j_max(32) == 0, chain.j_max(32)
    assert chain.j_max(33) == 1, chain.j_max(33)
    assert chain.j_max(43) == 2, chain.j_max(43)   # writes s=0, s=8
    assert chain.j_max(47) == 2, chain.j_max(47)   # settled: 47 still reads 2
    assert chain.j_max(49) == 3, chain.j_max(49)   # s=16 ends at 48 <= 48
    print("1/4 bookkeeping: strictly-before convention holds "
          "(43->2, 47->2, 49->3)")

    # ---- 2: streaming equivalence ---------------------------------------
    windows = [(0, 230), (1, 205)]
    idx = torch.tensor([cache.row0[e] + t for e, t in windows])
    stats = chain.load_states(idx)
    train_m = [mem._m[l].detach().clone() for l in range(mem.n_layers)]

    for b, (e, t) in enumerate(windows):
        mem.reset()
        for j in range(1, chain.j_max(t) + 1):
            row = cache.row0[e] + 8 * (j - 1)
            mem.advance(cache.latents[row:row + 1].float(),
                        cache.states[row:row + 1].float(),
                        detach_carry=True)          # deploy operator
        for l in range(mem.n_layers):
            d = (mem._m[l][0] - train_m[l][b]).abs().max().item()
            assert d < 1e-5, f"stream/train mismatch ep{e} t{t} layer{l}: {d}"
    print(f"2/4 streaming == training chain (J={int(stats['chain_J'])}, "
          f"max |diff| < 1e-5)")

    # ---- 3: gradient reach ----------------------------------------------
    mem.zero_grad(set_to_none=True)
    chain.load_states(idx)
    loss = sum(mem._m[l].sum() for l in range(mem.n_layers))
    loss.backward()
    for l, cell in enumerate(mem.cells):
        for name in ("W1_0", "to_k"):
            p = cell.W1_0 if name == "W1_0" else cell.to_k.weight
            g = p.grad
            assert g is not None and float(g.abs().sum()) > 0, \
                f"no gradient on cells[{l}].{name} — chain not in-graph"
    plain_grads = {n: p.grad.detach().clone()
                   for n, p in mem.named_parameters() if p.grad is not None}
    with torch.no_grad():
        chain.load_states(idx)
        assert mem._m[0].grad_fn is None, "no_grad chain still built a graph"
    print("3/4 gradient reaches cell params AND the learned init; "
          "no_grad chain builds no graph")

    # ---- 4: checkpointed == plain ---------------------------------------
    mem.zero_grad(set_to_none=True)
    ck = SlidingChain(cache, mem, w=8, checkpoint_every=3)
    ck.load_states(idx)
    for l in range(mem.n_layers):
        d = (mem._m[l] - train_m[l]).abs().max().item()
        assert d < 1e-6, f"checkpointed readout differs at layer {l}: {d}"
    loss = sum(mem._m[l].sum() for l in range(mem.n_layers))
    loss.backward()
    for n, p in mem.named_parameters():
        if n in plain_grads:
            d = (p.grad - plain_grads[n]).abs().max().item()
            assert d < 1e-5, f"checkpointed grad differs on {n}: {d}"
    print("4/4 checkpoint_every=3 chain == plain chain (readouts and grads)")

    print("ALL PASS — sliding chain is safe to train with")


if __name__ == "__main__":
    main()
