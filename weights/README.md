# Packed weights

Trained Open Image Denoise `RT` filter weights converted to the vk_oidn packed format
(`*.vkoidnw`, see [`src/weights_format.h`](../src/weights_format.h)) with
[`tools/pack_weights.py`](../tools/pack_weights.py).

Source: [RenderKit/oidn-weights](https://github.com/RenderKit/oidn-weights) commit
`28883d1769d5930e13cf7f1676dd852bd81ed9e7` (Open Image Denoise 2.5), Apache License 2.0,
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
