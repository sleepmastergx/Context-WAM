"""Encode the goal string(s) into T5 contexts, once.

`load_text_encoder: false` in the model config means the model never builds T5,
so `sample['context']`/`['context_mask']` must be precomputed — job 1516 died on
exactly that. MoveCube has ONE fixed goal, so this is a single forward pass and
every window shares the result.

VideoUnmask does NOT: its goal string names the QUERY ("...the container hiding
the **blue** cube"), 9 distinct goals over the 100 episodes. A single shared
context would leave the model blind to the query and the run would measure
nothing — the same failure mode the DP pipeline guards with `--with-text`.
`--lerobot-root` therefore encodes every distinct goal in the dataset and
records the episode -> goal map; gpu_cache.py indexes it per window.

Payload (single-goal):  {context [L,4096], mask [L], goal}
Payload (per-episode):  {contexts [K,L,4096], masks [K,L], goals [K],
                         ep_task {ep: k}}

Mirrors scripts/precompute_text_embeds.py:246-338 — same loader, same
HuggingfaceTokenizer(clean="whitespace"), same `text_encoder(ids, mask)` call,
same bf16 context / bool mask payload.
"""
import argparse
import json
import os
import sys

GOAL = ("watch the video carefully, then move the cube to the target "
        "in the same manner as before")
FW = os.environ.get(
    "FASTWAM_ROOT",
    str(__import__("pathlib").Path(__file__).resolve().parents[1]
        / "third_party" / "fastwam"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/movecube_fastwam/text_context.pt")
    ap.add_argument("--context-len", type=int, default=128)  # tokenizer_max_len
    ap.add_argument("--goal", default=GOAL)
    ap.add_argument("--lerobot-root", default=None,
                    help="encode EVERY distinct goal in this LeRobot dataset "
                         "and record the episode->goal map (VideoUnmask: the "
                         "query color lives in the goal string)")
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)   # we chdir below; keep out anchored

    goals, ep_task = [args.goal], None
    if args.lerobot_root:
        root = os.path.abspath(args.lerobot_root)
        goals, ep_task = [], {}
        for line in open(os.path.join(root, "meta", "episodes.jsonl")):
            rec = json.loads(line)
            tasks = rec["tasks"]
            if len(tasks) != 1:
                sys.exit(f"episode {rec['episode_index']} has {len(tasks)} "
                         "goals — one goal per episode is assumed")
            if tasks[0] not in goals:
                goals.append(tasks[0])
            ep_task[int(rec["episode_index"])] = goals.index(tasks[0])
        print(f"{len(ep_task)} episodes -> {len(goals)} distinct goals")

    if not os.environ.get("SLURM_JOB_ID"):
        sys.exit("T5-XXL needs a GPU: run inside a Slurm job")

    os.chdir(FW)                       # their config paths are relative
    sys.path.insert(0, f"{FW}/src")
    import torch
    from fastwam.models.wan22.helpers.loader import (_load_registered_model,
                                                     _resolve_configs)
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        redirect_common_files=True)
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path, "wan_video_text_encoder",
        torch_dtype=torch.bfloat16, device="cuda").eval()
    tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path,
                                     seq_len=args.context_len,
                                     clean="whitespace")

    ctxs, masks = [], []
    with torch.inference_mode():
        # one goal per forward, exactly as the single-goal path did — padding a
        # batch of unequal-length goals would change the encoder's inputs.
        for g in goals:
            ids, mask = tokenizer([g], return_mask=True, add_special_tokens=True)
            ids = ids.to("cuda")
            mask = mask.to(device="cuda", dtype=torch.bool)
            context = text_encoder(ids, mask)
            ctxs.append(context[0].detach().to("cpu", torch.bfloat16).contiguous())
            masks.append(mask[0].detach().to("cpu", torch.bool).contiguous())

    if ep_task is None:
        payload = {"context": ctxs[0], "mask": masks[0], "goal": goals[0]}
    else:
        payload = {"contexts": torch.stack(ctxs), "masks": torch.stack(masks),
                   "goals": goals, "ep_task": ep_task}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(payload, args.out)
    print(f"wrote {args.out}: {len(ctxs)} context(s) {tuple(ctxs[0].shape)} "
          f"{ctxs[0].dtype}, real tokens "
          f"{[int(m.sum()) for m in masks]}", flush=True)


if __name__ == "__main__":
    main()
