"""Stream a training run's log.jsonl into wandb without touching the trainer.

Backfills everything already in log.jsonl (dedup by step, keeps the last record
per step, so the pre-crash history is included), then follows the file and logs
every new record as it appears. Resumable: re-running reuses the same wandb run id.
"""
import argparse, json, os, time
from pathlib import Path
import yaml, wandb

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True, type=Path)
ap.add_argument("--project", default="wam-original")
ap.add_argument("--entity", default=None)
ap.add_argument("--group", default="movecube-fastwam-30x30")
ap.add_argument("--name", default=None)
ap.add_argument("--final-step", type=int, default=64260)
ap.add_argument("--poll-seconds", type=float, default=15)
args = ap.parse_args()

run_dir = args.run_dir.resolve()
log_path = run_dir / "log.jsonl"
cfg_path = run_dir / "run_config.yaml"
id_path = run_dir / ".wandb_run_id"
cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
run_id = id_path.read_text().strip() if id_path.exists() else __import__("secrets").token_hex(4)
id_path.write_text(run_id)

run = wandb.init(project=args.project, entity=args.entity, group=args.group,
                 name=args.name or run_dir.name, id=run_id, resume="allow",
                 config=cfg, tags=[*cfg.get("wandb", {}).get("tags", []), "log-sync"],
                 settings=wandb.Settings(x_disable_stats=True, x_disable_meta=True))
print(f"wandb: {run.url}", flush=True)
last_step = run.summary.get("step", -1) if run.resumed else -1
print(f"resuming from step {last_step}", flush=True)

def records():
    """Yield parsed records appended to log.jsonl, following the file forever."""
    pos = 0
    while True:
        if log_path.exists():
            with open(log_path) as fh:
                fh.seek(pos)
                for line in fh:
                    if not line.endswith("\n"):
                        break  # partial line, re-read next time
                    pos += len(line)
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass
        yield None

pending = {}
done = False
for rec in records():
    if rec is not None:
        if rec["step"] > last_step:
            pending[rec["step"]] = rec  # later duplicates (resumes) win
        continue
    # end of currently available data: flush in step order
    for step in sorted(pending):
        r = pending[step]
        run.log({k: v for k, v in r.items() if k != "step"}, step=step)
        last_step = step
    if pending:
        print(f"synced through step {last_step}", flush=True)
        pending = {}
    if last_step >= args.final_step:
        print("final step reached; finishing", flush=True)
        run.finish()
        break
    time.sleep(args.poll_seconds)
