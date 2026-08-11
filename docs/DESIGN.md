# Design record — context-wam

Compact version of the project's design review (2026-08-10/11). The module
docstrings in `context_wam/sliding_chain.py` and `train.py` carry the
operational versions of these rules.

## The mechanism

Per-layer TTT fast weights in Fast-WAM's **action expert** (RoboTTT-style):
each of the 5 layers owns a state M_ℓ, fused at the MoT seam

    O_ℓ = O_attn,ℓ + tanh(α_ℓ) · O_TTT,ℓ        α init 1e-3

so at step 0 the ttt arm IS the control (gate ≈ 0.001). The video expert and
its KV-cache inference path are untouched — the patch rebinds
`MoT._apply_expert_post_block` on the instance only. Cell: MLP fast weights +
Titans gating (learned per-input η, forget gate, momentum), fp32 always.

Why per-layer readout and not a bigger state: in the predecessor study the
memory *stored* the answer at 94–98% probe decode and still didn't change
behaviour — readout bandwidth, not capacity, was binding (the 32× capacity
arm was the only significantly *worse* one).

## The write policy (sliding-w, settled 2026-08-11)

- Writes come from the **video stream only — never actions** — over the whole
  episode, one write per `w=8` raw steps, chained from a learned init.
- **Strictly-before convention**: a window starting at raw step t reads the
  state after the last COMPLETED write before t:
  `j_max(t) = (t − 33)//8 + 1`. A window at t=47 reads writes {s=0, s=8}
  (content through raw 40); t=49 additionally reads s=16.
- **w=8 realization**: the Wan VAE only encodes clips of T%4==1 subsampled
  frames, so disjoint 8-step chunks have no latent representation. Each write
  consumes the cached latents of the trailing 33-step window ending at the
  write time; the overlap is de-duplicated by the surprise gate.
- w is frame-clocked → the write operator is identical at train and deploy by
  construction, and the replan rate R never touches the memory (free knob for
  both arms).

## The in-graph chain (why training looks the way it does)

One batched chain per episode per optimizer step, rolled only to the furthest
sampled window, **kept entirely on the autograd tape**. The write parameters
(k/v projections, η-net, forget gate, learned init) receive gradient *only*
through: action loss → read(M_j) → M_j → … → M_1 → params. That outer loop is
what teaches the memory *what to store* — the inner TTT loss alone is a
generic compressor (the shape of RoboMME's own ~22% TTT baselines).

Consequences:
- **Cache inputs forever, never cache states.** Chain inputs are frozen-VAE
  latents (constant); states must be recomputed each step because the write
  parameters just changed. Affordable precisely because writes read VAE
  latents, not DiT features — the 5B never runs on a write.
- **Never detach / precompute the chain.** Not an approximation — a different
  objective in which the write path silently stops training. The only legal
  VRAM lever is `chain_checkpoint_every` (exact gradients, recompute in
  backward).
- BPTT depth at w=8 is ~62 writes ≈ 2× the deepest validated run (~33);
  `mem_gnorm` is logged every step, and `sliding_w: 16` is the fallback.

## The comparison design

- Both arms draw the SAME uniform-random exec windows with the same seed —
  the chain is the *only* difference, so the McNemar pair is clean.
- `checks/check_arms_match.py` asserts the model configs differ in exactly
  `memory.enabled` (115.5M vs 117.8M, +1.9%).
- Fixed, most-validated settings everywhere else: M5 action expert (depth =
  injection points), `fused_mlp` fusion, both experts train (their recipe),
  from-scratch action expert, EMA added.

## Ablation ladder (flags to add later)

control (no memory) → **sliding-w (this repo, primary)** → frozen-M* (writes
stop at the video→exec boundary; isolates pure episodic recall and is immune
to deploy-time drift) → write-everywhere (adds exec-time adaptation, RoboTTT
territory). Sliding−control = the headline; frozen vs sliding = the
write-policy ablation; expected on MoveCube: sliding ≈ frozen, since the
manner is fixed at the boundary and recent exec content is redundant with the
33-frame attention window.

## Known open risks (monitored, not solved)

1. **Deploy drift**: exec-phase writes at deploy read the policy's own
   (drifting) frames, and timeout episodes run ~3× training length → forget
   gate compounds beyond the trained regime. Diagnostic: manner-decode from
   the state vs chunk count; fallback: cap writes at the trained chunk count.
2. **Retention is learned, not structural**: the forget gate must hold the
   video content through ~20+ exec writes (predecessor evidence says it can:
   94–98% decode with no decay, but only up to training length).
3. **Statistics**: 50-episode evals gave ±8–10 pt CIs and left a +10 pt
   effect at p=0.18 — budget 150+ eval episodes or 3 seeds/arm up front.
