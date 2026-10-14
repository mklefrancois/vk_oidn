# Packed weights

Trained Open Image Denoise `RT` filter weights converted to the vk_oidn packed format
(`*.vkoidnw`, see [`src/weights_format.h`](../src/weights_format.h)) with
[`tools/pack_weights.py`](../tools/pack_weights.py).

Source: [RenderKit/oidn-weights](https://github.com/RenderKit/oidn-weights) commit
`28883d1769d5930e13cf7f1676dd852bd81ed9e7` (Open Image Denoise 2.5.1), Apache License 2.0,
Copyright Intel Corporation.

| File                          | Inputs                          | Quality  |
|-------------------------------|---------------------------------|----------|
| `rt_hdr`                      | HDR color                       | balanced/high |
| `rt_hdr_small`                | HDR color                       | fast     |
| `rt_hdr_alb`                  | HDR color + albedo              | balanced/high |
| `rt_hdr_alb_small`            | HDR color + albedo              | fast     |
| `rt_hdr_alb_nrm`              | HDR color + albedo + normal     | balanced/high |
| `rt_hdr_alb_nrm_small`        | HDR color + albedo + normal     | fast     |
| `rt_hdr_calb_cnrm`            | HDR color + clean albedo/normal | balanced |
| `rt_hdr_calb_cnrm_small`      | HDR color + clean albedo/normal | fast     |
| `rt_hdr_calb_cnrm_large`      | HDR color + clean albedo/normal | high     |
| `rt_ldr_alb_nrm`              | LDR color + albedo + normal     | balanced/high |
| `rt_ldr_alb_nrm_small`        | LDR color + albedo + normal     | fast     |

To pack other models (e.g. `rt_ldr`, `rt_alb`, `rt_nrm`), run the tool on the `.tza` files of
oidn-weights and either add them to this folder (they are embedded automatically) or load them at
runtime:

```
python tools/pack_weights.py path/to/oidn-weights/rt_ldr.tza -o weights
python tools/pack_weights.py --dump weights/rt_ldr.vkoidnw
```

## Updating to new Open Image Denoise weights

[`tools/update_weights.py`](../tools/update_weights.py) downloads the models of this folder from a
commit of oidn-weights (verified against their Git LFS checksums), packs them, and replaces the
files only if all of them were packed. It also updates the source commit above and, with
`--oidn-version`, the Open Image Denoise release used as the reference by the CI.

```
python tools/update_weights.py <oidn-weights commit or branch> --oidn-version 2.6.0 --dry-run
python tools/update_weights.py <oidn-weights commit or branch> --oidn-version 2.6.0
```

Use the oidn-weights commit of the Open Image Denoise release (the `weights` submodule of
[RenderKit/oidn](https://github.com/RenderKit/oidn)). Then rebuild, validate on the GPU with
`tests/scripts/run_validation.sh` against that release, and push: the CI compares the results with
the same release on Linux.

- **Retrained weights** (same networks, possibly different channel counts): nothing else to change.
- **New network structure**: the update stops with "unknown tensors" or "missing tensors". Add the
  topology and its receptive field to `tools/pack_weights.py` (from `core/unet_filter.cpp` of Open
  Image Denoise); the kernels support any channel counts.
- **New operations** (other than 3x3 convolutions, ReLU, 2x2 max pooling and 2x nearest
  upsampling), or changes to the input/output transfer functions: the Slang kernels need to be
  extended.
