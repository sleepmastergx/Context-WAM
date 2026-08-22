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
import os
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
    mode = os.environ.get("CHAIN_WRITE_INPUT", "pooled")   # pooled | tokens
    return PerLayerEpisodeMemory(
        n_layers=2, hidden_dim=64, latent_channels=48, proprio_dim=8,
        d_k=16, d_v=16, d_hidden=32, d_out=24,
        chunk=(384 if mode == "tokens" else 1), write_input=mode).float()


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
    print("1/6 bookkeeping: strictly-before convention holds "
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
    print(f"2/6 streaming == training chain (J={int(stats['chain_J'])}, "
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
    print("3/6 gradient reaches cell params AND the learned init; "
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
    print("4/6 checkpoint_every=3 chain == plain chain (readouts and grads)")

    # ---- 5: stacked cells == looped cells --------------------------------
    from memory import ttt_with_state, stack_cells, stacked_init_state, \
        ttt_with_state_stacked
    mem.zero_grad(set_to_none=True)
    d_in = mem.cells[0].ln_in.normalized_shape[0]   # 56 pooled / 200 tokens
    x = torch.randn(3, 4, d_in)
    mask = torch.ones(3, 4)
    ref, grads_ref = [], {}
    for cell in mem.cells:
        st = None
        for rep in range(2):                       # two chained windows
            m, _, st = ttt_with_state(cell, x * (0.8 ** rep), mask, st)
        ref.append(m)
    torch.stack(ref).sum().backward()
    grads_ref = {n: p.grad.detach().clone()
                 for n, p in mem.named_parameters() if p.grad is not None}
    mem.zero_grad(set_to_none=True)
    P = stack_cells(list(mem.cells))
    st = stacked_init_state(P, 3)
    for rep in range(2):
        m, _, st = ttt_with_state_stacked(P, x * (0.8 ** rep), mask, st)
    d = (m - torch.stack(ref)).abs().max().item()
    assert d < 1e-5, f"stacked readout differs from looped: {d}"
    m.sum().backward()
    for n, p in mem.named_parameters():
        if n in grads_ref:
            d = (p.grad - grads_ref[n]).abs().max().item()
            assert d < 1e-5, f"stacked grad differs on {n}: {d}"
    print("5/6 stacked-over-cells chain == per-cell loop (readouts and grads)")

    # ---- 6: chain-once accumulation protocol == fused backward -----------
    # d(sum_k loss_k)/dtheta via detached leaves + one chain backward must
    # equal backprop of the summed loss straight through readouts().
    proj = torch.nn.Linear(24, 1)                  # stand-in for the 5B DiT
    losses = lambda mlist, sl: sum(                # noqa: E731
        proj(mlist[l][sl]).pow(2).mean() for l in range(mem.n_layers))
    mem.zero_grad(set_to_none=True)
    ms, _ = chain.readouts(idx)
    (losses(ms, slice(0, 1)) + losses(ms, slice(1, 2))).backward()
    fused = {n: p.grad.detach().clone()
             for n, p in mem.named_parameters() if p.grad is not None}
    mem.zero_grad(set_to_none=True)
    proj.zero_grad(set_to_none=True)
    ms, _ = chain.readouts(idx)
    leaves = [m.detach().requires_grad_(True) for m in ms]
    for k in range(2):                             # two "micro-batches"
        losses(leaves, slice(k, k + 1)).backward()
    torch.autograd.backward(ms, [leaf.grad for leaf in leaves])
    for n, p in mem.named_parameters():
        if n in fused:
            d = (p.grad - fused[n]).abs().max().item()
            assert d < 1e-6, f"chain-once grad differs on {n}: {d}"
    print("6/6 chain-once accumulation protocol == fused backward")

    print("ALL PASS — sliding chain is safe to train with")


if __name__ == "__main__":
    main()
