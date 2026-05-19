# Trained model assets (Hugging Face)

Neural surrogates for Ves2Dpy are too large for git. They are published as a **Hugging Face model repo** and downloaded automatically at runtime.

## Default repo

Set your published repo with environment variable `VES2D_HF_REPO` (default from `manifest.json`: `ves2d/ves2d-trained`).

Cache directory: `~/.cache/ves2d` (override with `VES2D_CACHE_DIR`).

## Upload from the cluster

1. Create a model repo on [huggingface.co](https://huggingface.co/new) (type: Model).
2. `pip install huggingface_hub` and `export HF_TOKEN=hf_...`
3. From the repo root on a node that has the trained files:

```bash
# N=32 (adjust paths to your allocation)
python scripts/upload_trained_to_hf.py \
  --repo-id YOUR_ORG/ves2d-trained \
  --resolution 32 \
  --n32-trained-root /work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/trained \
  --n32-inner-near-root /work/09452/alberto47/ls6/vesicle_nearF2024/trained_disth_nocoords/inner_downsample32 \
  --n32-inner-near-weight /work/09452/alberto47/vista/Ves2Dpy/trained/2025ves_merged_disth_innerNearFourier.pth \
  --n32-ten-self /work/09452/alberto47/ls6/vesicle_selften/save_models/Ves_2025Feb_downsample_selften_zerolevel_12blks_loss_0.01105_2242401_cuda2.pth \
  --upload

# N=128 (TorchScript .pt under trained/torch_script_models/)
python scripts/upload_trained_to_hf.py \
  --repo-id YOUR_ORG/ves2d-trained \
  --resolution 128 \
  --n128-trained-root /work/09452/alberto47/vista/Ves2Dpy/trained \
  --upload
```

4. Set `VES2D_HF_REPO=YOUR_ORG/ves2d-trained` on laptops and CI.

## Use in Python

```python
from tools.model_hub import ensure_resolution_assets, apply_layout_to_params

layout = ensure_resolution_assets(32)
params = apply_layout_to_params({"resolution": 32, ...}, layout)
```

Or enable in `vesnet` config:

```json
{
  "resolution": 32,
  "use_hf_hub": true,
  "hf_repo": "YOUR_ORG/ves2d-trained"
}
```

Leave `trained_root` unset to download automatically. Set `use_hf_hub: false` and `trained_root` to use a local tree only.

## File layout on the Hub

See `manifest.json` for the canonical list. Summary:

| Resolution | Contents |
|------------|----------|
| N=32 | `N32/trained/` — adv, near, ten-adv norms + `.pth` weights; `N32/inner_near/` — inner near norms |
| N=128 | `N128/trained/` — norms + `torch_script_models/*.pt` |
