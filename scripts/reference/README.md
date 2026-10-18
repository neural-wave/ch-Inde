# Reference: the original full-data refit runner

`full_data_refit.py` is the script that produced the shipped checkpoints (job
`full-data-harvest-winner-s42-e30` on the GPU host): it wrote the `full_data_refit` date manifest,
generated the resolved per-component configurations from `config/full_data_harvest_winner.yaml`,
launched both fits through `src.context.full_refit_cli` and exported the bundle
(`ensemble.json`, gate, checkpoints, diagnostics).

It is kept here for provenance only. It expects the project repository layout — it imports
`scripts.agent.context_rope_experiments` and writes under `<repo>/artifacts/runs` — so run it from
the project root, not from this delivery folder.

To replay the refit from the delivery folder itself, use `../../train.py`, which reads the resolved
configurations in `../../configs/` and the full-data manifest directly.
