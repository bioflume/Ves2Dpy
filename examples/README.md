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

./examples/ex1_one_ves_parabolic/run.sh
./examples/ex2_two_ves_shear/run.sh      # needs shear_N32.npy
./examples/ex3_multi_ves_vortex/run.sh   # needs 48vesTG_N32.npy
```

- **ex1** writes `initial.npy` as `(64, 1)` before running.
- **ex2** loads from `shear_N32.npy`.
- **ex3** loads vesicle `[0, 6, 12, 18, 24, 30, 36, 42]` from `48vesTG_N32.npy` (edit `input_indices` to change).

```bash
./examples/ex1_one_ves_parabolic/run.sh --num-steps 100
```

### Optional: plot frames and video

After a successful run, generate PNG frames (`output/<outfile>/1.png`, …) and `output/<outfile>.mp4` using `newtorch/TG_postprocess.py` and `newtorch/create_video.py`:

```bash
./examples/ex1_one_ves_parabolic/run.sh --postprocess
# or
VES2D_POSTPROCESS=1 ./examples/ex1_one_ves_parabolic/run.sh
```

Requires **matplotlib**, **opencv-python** (`cv2`), and **tqdm**.

## Outputs

- `output/<outfile>.bin` — trajectory
- `<outfile>.log` — run log
- With `--postprocess`: `output/<outfile>/*.png` and `output/<outfile>.mp4`

