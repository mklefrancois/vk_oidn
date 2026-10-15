# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""Performance chart for the vk_oidn README: GPU time per image, vk_oidn 0.1.3 vs OIDN 2.5.1 CUDA.
Writes perf_light.png and perf_dark.png next to this script. Requires matplotlib."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = {  # resolution -> quality -> (vk_oidn ms, CUDA ms)
  '1920x1080': {'High quality': (5.01, 8.10), 'Fast quality': (2.89, 5.55)},
  '3840x2160': {'High quality': (22.77, 34.65), 'Fast quality': (12.55, 23.41)},
}
THEMES = {
  'light': dict(surface='#fcfcfb', text='#0b0b0b', text2='#52514e', grid='#e4e3df', s1='#2a78d6', s2='#eb6834'),
  'dark': dict(surface='#1a1a19', text='#f0efec', text2='#c3c2b7', grid='#3a3936', s1='#3987e5', s2='#d95926'),
}
SERIES = ['vk_oidn 0.1.3 (Vulkan)', 'Open Image Denoise 2.5.1 (CUDA)']


def draw(mode):
  t = THEMES[mode]
  plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), dpi=150, facecolor=t['surface'])
  for ax, (res, qualities) in zip(axes, DATA.items()):
    ax.set_facecolor(t['surface'])
    xmax = max(v for pair in qualities.values() for v in pair) * 1.32
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(qualities) - 0.4)
    ax.invert_yaxis()
    bar_h = 0.34
    for gi, (q, (vk, cuda)) in enumerate(qualities.items()):
      for si, (val, col) in enumerate(((vk, t['s1']), (cuda, t['s2']))):
        y = gi + (si - 0.5) * (bar_h + 0.03)  # small surface gap between adjacent bars
        ax.barh(y, val, height=bar_h, color=col, linewidth=0)
        label = f'{val:.2f} ms'
        if si == 0:
          label += f'   {cuda / vk:.1f}x faster'
        ax.text(val + xmax * 0.015, y, label, va='center', ha='left', fontsize=10,
                color=t['text'] if si == 0 else t['text2'], fontweight='bold' if si == 0 else 'normal')
    ax.set_yticks(range(len(qualities)))
    ax.set_yticklabels(list(qualities.keys()), color=t['text'])
    ax.tick_params(axis='y', length=0, pad=8)
    ax.tick_params(axis='x', colors=t['text2'], length=0)
    ax.set_xlabel('GPU time per image (ms, lower is better)', color=t['text2'], fontsize=10)
    ax.set_title(res, color=t['text'], fontsize=12, loc='left', pad=10, fontweight='bold')
    ax.grid(axis='x', color=t['grid'], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
      ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(t['grid'])
  handles = [plt.Rectangle((0, 0), 1, 1, color=t['s1']), plt.Rectangle((0, 0), 1, 1, color=t['s2'])]
  leg = fig.legend(handles, SERIES, loc='upper center', ncol=2, frameon=False, fontsize=10.5,
                   bbox_to_anchor=(0.5, 1.0), handlelength=1.2, handleheight=0.9)
  for txt in leg.get_texts():
    txt.set_color(t['text'])
  fig.text(0.5, 0.005, 'NVIDIA RTX PRO 4500 Blackwell, HDR color + albedo + normal',
           ha='center', va='bottom', fontsize=9, color=t['text2'])
  fig.tight_layout(rect=(0, 0.04, 1, 0.9), w_pad=3)
  fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'perf_{mode}.png'), facecolor=t['surface'])
  plt.close(fig)


for mode in THEMES:
  draw(mode)
print('ok')
