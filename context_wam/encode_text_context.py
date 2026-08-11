"""Encode MoveCube's single goal string into a T5 context, once.

`load_text_encoder: false` in the model config means the model never builds T5,
so `sample['context']`/`['context_mask']` must be precomputed — job 1516 died on
exactly that. MoveCube has ONE fixed goal, so this is a single forward pass and
every window shares the result.

Mirrors scripts/precompute_text_embeds.py:246-338 — same loader, same
HuggingfaceTokenizer(clean="whitespace"), same `text_encoder(ids, mask)` call,
same bf16 context / bool mask payload.
"""
import argparse
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
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)   # we chdir below; keep out anchored

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

    with torch.inference_mode():
        ids, mask = tokenizer([args.goal], return_mask=True,
                              add_special_tokens=True)
        ids = ids.to("cuda")
        mask = mask.to(device="cuda", dtype=torch.bool)
        context = text_encoder(ids, mask)

    payload = {"context": context[0].detach().to("cpu", torch.bfloat16).contiguous(),
               "mask": mask[0].detach().to("cpu", torch.bool).contiguous(),
               "goal": args.goal}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(payload, args.out)
    print(f"wrote {args.out}: context {tuple(payload['context'].shape)} "
          f"{payload['context'].dtype}, mask {tuple(payload['mask'].shape)} "
          f"({int(payload['mask'].sum())} real tokens)", flush=True)


if __name__ == "__main__":
    main()
