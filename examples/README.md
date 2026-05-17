# Ves2Dpy examples (N=32)

MLARM / vesnet runs at **N=32** via `newtorch/entry_vesnet.py`.

| Directory | Case |
|-----------|------|
| `ex1_one_ves_parabolic/` | 1 vesicle (off-center), parabolic flow |
| `ex2_two_ves_shear/` | 2 vesicles from `shear_N32.npy`, shear flow |
| `ex3_multi_ves_vortex/` | 8 vesicles from `48vesTG_N32.npy`, vortex flow |


## Run

```bash
export VES2D_TRAINED_ROOT=/path/to/Ves2Dpy_N32/trained
export VES2D_INNER_NEAR_ROOT=/path/to/inner_downsample32

./examples/ex1_one_ves_parabolic/run.sh
./examples/ex2_two_ves_shear/run.sh
./examples/ex3_multi_ves_vortex/run.sh
```

- **ex1** builds `initial.npy` before running.
- **ex2** / **ex3** load external `.npy` files (no generated ICs). **ex3** uses columns `[0, 6, 12, 18, 24, 30, 36, 42]` from the 48-vesicle file; edit `input_indices` in `config.json` to change the subset.

```bash
./examples/ex1_one_ves_parabolic/run.sh --num-steps 100
```

## Outputs

- `output/<outfile>.bin` — trajectory
- `<outfile>.log` — run log

