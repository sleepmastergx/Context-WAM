# Vendored Fast-WAM

- Source: git@github.com:rhyang6/fastwam.git
- Commit: 7ca5e2f ("Write the training launch log inside the run output dir")
- Vendored: 2026-08-11, from /shared_work/physical_intelligence/policies/Fast-WAM/fastwam
- License: MIT (see LICENSE, preserved verbatim)
- Contents: src/fastwam (python package), scripts/{accelerate_configs,ds_configs},
  pyproject.toml. Checkpoints, data, outputs, docs and paper material are NOT vendored.
- Local modifications: none. All context-wam changes live outside this directory and
  integrate by subclassing/patching at runtime (the repo treats this copy as read-only).
