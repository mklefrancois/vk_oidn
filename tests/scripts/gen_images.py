#!/usr/bin/env python3
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generates synthetic noisy test images (PFM) for validating and benchmarking the denoiser.

The scene is a checkerboard with a sphere lit by a directional light and a bright spot. The noisy
color is an average of a few "samples" with exponentially distributed noise and rare fireflies,
which resembles Monte Carlo rendering noise. Requires numpy.

Outputs in the directory: color.pfm (HDR), ldr.pfm (tone mapped), albedo.pfm, normal.pfm,
clean.pfm (noise-free color).

Usage:
  gen_images.py output_dir width height [--seed N] [--spp N]
"""

import argparse
import os
import numpy as np


def write_pfm(path, img):
  h, w, c = img.shape
  with open(path, 'wb') as f:
    f.write(b'PF\n' if c == 3 else b'Pf\n')
    f.write(f'{w} {h}\n-1.0\n'.encode())
    f.write(np.flipud(img).astype('<f4').tobytes())


def generate(out, W, H, seed=1, spp=4):
  rng = np.random.default_rng(seed)
  y, x = np.mgrid[0:H, 0:W].astype(np.float32)
  u, v = x / W, y / H

  # Albedo: checkerboard and a sphere
  alb = np.zeros((H, W, 3), np.float32)
  checker = ((np.floor(u * 12) + np.floor(v * 8)) % 2)[..., None]
  alb[:] = 0.2 + 0.6 * checker * np.array([0.9, 0.5, 0.3], np.float32) + \
           0.2 * (1 - checker) * np.array([0.2, 0.6, 0.9], np.float32)
  cx, cy, r = 0.6, 0.45, 0.22
  d = np.sqrt(((u - cx) * W / H) ** 2 + (v - cy) ** 2)
  sphere = d < r
  alb[sphere] = np.array([0.8, 0.8, 0.75], np.float32)

  # Normals
  nrm = np.zeros((H, W, 3), np.float32)
  nrm[..., 2] = 1
  nx = ((u - cx) * W / H) / r
  ny = (v - cy) / r
  nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0, 1))
  nrm[sphere] = np.stack([nx, -ny, nz], -1)[sphere]

  # Lighting
  L = np.array([0.4, 0.6, 0.7], np.float32)
  L /= np.linalg.norm(L)
  diffuse = np.clip((nrm * L).sum(-1), 0, 1)[..., None]
  light = 0.3 + 2.0 * diffuse + 8.0 * np.exp(-(((u - 0.2) ** 2 + (v - 0.2) ** 2) / 0.002))[..., None]
  clean = alb * light

  # Monte Carlo-like noise: average of spp samples with heavy-tailed fireflies
  samples = clean[None] * rng.exponential(1.0, (spp, H, W, 1)).astype(np.float32)
  fire = (rng.random((spp, H, W, 1)) < 0.002) * rng.exponential(30.0, (spp, H, W, 1))
  noisy = (samples + fire * clean[None]).mean(0).astype(np.float32)

  os.makedirs(out, exist_ok=True)
  write_pfm(os.path.join(out, 'color.pfm'), noisy)
  write_pfm(os.path.join(out, 'albedo.pfm'), alb)
  write_pfm(os.path.join(out, 'normal.pfm'), nrm)
  write_pfm(os.path.join(out, 'ldr.pfm'), np.clip(noisy / (1 + noisy), 0, 1).astype(np.float32))
  write_pfm(os.path.join(out, 'clean.pfm'), clean.astype(np.float32))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generates synthetic noisy test images.')
  parser.add_argument('output_dir')
  parser.add_argument('width', type=int)
  parser.add_argument('height', type=int)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--spp', type=int, default=4)
  args = parser.parse_args()
  generate(args.output_dir, args.width, args.height, args.seed, args.spp)
