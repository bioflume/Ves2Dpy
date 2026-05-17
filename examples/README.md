# Ves2Dpy examples (N=32)

MLARM / vesnet runs at **N=32** via `newtorch/entry_vesnet.py`.

| Directory | Case |
|-----------|------|
| `ex1_one_ves_parabolic/` | 1 vesicle (off-center), parabolic flow |
| `ex2_two_ves_shear/` | 2 vesicles from `shear_N32.npy`, shear flow |
| `ex3_multi_ves_vortex/` | 8 of 48 vesicles from `48vesTG_N32.npy`, vortex flow |

## Initial-condition `.npy` layout

All IC arrays use shape **`(2N, nv)`** with **N = 32** (so **`(64, nv)`**):

| Rows | Content |
|------|---------|
| `0 : N-1` | x-coordinates of each vesicle |
| `N : 2N-1` | y-coordinates of each vesicle |
| columns | one vesicle per column (`nv` vesicles) |

Examples: `shear_N32.npy` is `(64, 2)`; `48vesTG_N32.npy` is `(64, 48)`.  
`input_indices` in `config.json` lists **column** indices to keep (ex3: 8 columns from 48).

## Run

```bash
export VES2D_TRAINED_ROOT=/path/to/Ves2Dpy_N32/trained
export VES2D_INNER_NEAR_ROOT=/path/to/inner_downsample32

./examples/ex1_one_ves_parabolic/run.sh
./examples/ex2_two_ves_shear/run.sh      # needs shear_N32.npy
./examples/ex3_multi_ves_vortex/run.sh   # needs 48vesTG_N32.npy
```

Copy data files into the example folders, or set `VES2D_SHEAR_IC` / `VES2D_TG_IC` to their paths.

- **ex1** writes `initial.npy` as `(64, 1)` before running.
- **ex2** loads all columns from `shear_N32.npy`.
- **ex3** loads columns `[0, 6, 12, 18, 24, 30, 36, 42]` from `48vesTG_N32.npy` (edit `input_indices` to change).

```bash
./examples/ex1_one_ves_parabolic/run.sh --num-steps 100
```

## Outputs

- `output/<outfile>.bin` — trajectory
- `<outfile>.log` — run log

## Regenerate ex1 initial conditions

```bash
python examples/make_initial_conditions.py
```
