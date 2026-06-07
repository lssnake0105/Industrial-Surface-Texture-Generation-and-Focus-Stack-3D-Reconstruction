# Source Code

Selected public-facing scripts from the SRTP workspace are collected here.

Main entry points:

- `surface_sample_generator.py`: generates synthetic 3D surface samples.
- `simulate_antiglare_prototype.py`: runs a compact anti-glare focus-stack simulation and model prototype.
- `dff_depth_direction.py`: centralizes the focus-stack height convention.
- `glare_aware_dff.py` and `run_real_focus_measure_eval.py`: evaluate glare-aware DFF and real focus-stack samples when local data is available.
- `train_*.py`: training experiments for learning-based correction models.

Some scripts retain local-output defaults inherited from the project workspace. For public use, pass explicit output paths when available and consult `data/README.md` for missing local data.
