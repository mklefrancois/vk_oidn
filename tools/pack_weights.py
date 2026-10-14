#!/usr/bin/env python3
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""Converts Open Image Denoise weights (.tza) into packed vk_oidn weights (.vkoidnw).

The packed format (see src/weights_format.h) contains the layer graph of the U-Net and the
weights/biases already in the GPU layout used by the convolution kernels, so loading them is a
single upload.

Usage:
  pack_weights.py input.tza [-o output.vkoidnw]
  pack_weights.py --dump file.vkoidnw
"""

import argparse
import os
import struct
import sys

MAGIC = b'VKOIDNW1'
VERSION = 1
ALIGNMENT = 256

SRC_INPUT = -1
SRC_NONE = -2

FLAG_RELU = 1 << 0
FLAG_POOL_DST = 1 << 1
FLAG_UPSAMPLE_SRC0 = 1 << 2

LARGE_NETWORK = 1 << 0

RECEPTIVE_FIELD_BASE = 174
RECEPTIVE_FIELD_LARGE = 202

# U-Net topologies (must match Open Image Denoise, core/unet_filter.cpp)
# (layer name, first source, second source, flags); sources are layer names or 'input'
UNET_BASE = [
  ('enc_conv0',  'input',      None,         FLAG_RELU),
  ('enc_conv1',  'enc_conv0',  None,         FLAG_RELU | FLAG_POOL_DST),
  ('enc_conv2',  'enc_conv1',  None,         FLAG_RELU | FLAG_POOL_DST),
  ('enc_conv3',  'enc_conv2',  None,         FLAG_RELU | FLAG_POOL_DST),
  ('enc_conv4',  'enc_conv3',  None,         FLAG_RELU | FLAG_POOL_DST),
  ('enc_conv5a', 'enc_conv4',  None,         FLAG_RELU),
  ('enc_conv5b', 'enc_conv5a', None,         FLAG_RELU),
  ('dec_conv4a', 'enc_conv5b', 'enc_conv3',  FLAG_RELU | FLAG_UPSAMPLE_SRC0),
  ('dec_conv4b', 'dec_conv4a', None,         FLAG_RELU),
  ('dec_conv3a', 'dec_conv4b', 'enc_conv2',  FLAG_RELU | FLAG_UPSAMPLE_SRC0),
  ('dec_conv3b', 'dec_conv3a', None,         FLAG_RELU),
  ('dec_conv2a', 'dec_conv3b', 'enc_conv1',  FLAG_RELU | FLAG_UPSAMPLE_SRC0),
  ('dec_conv2b', 'dec_conv2a', None,         FLAG_RELU),
  ('dec_conv1a', 'dec_conv2b', 'input',      FLAG_RELU | FLAG_UPSAMPLE_SRC0),
  ('dec_conv1b', 'dec_conv1a', None,         FLAG_RELU),
  ('dec_conv0',  'dec_conv1b', None,         FLAG_RELU),
]

UNET_LARGE = [
  ('enc_conv1a', 'input',      None,         FLAG_RELU),
  ('enc_conv1b', 'enc_conv1a', None,         FLAG_RELU | FLAG_POOL_DST),
  ('enc_conv2a', 'enc_conv1b', None,         FLAG_RELU),
  ('enc_conv2b', 'enc_conv2a', None,         FLAG_RELU | FLAG_POOL_DST),
  ('enc_conv3a', 'enc_conv2b', None,         FLAG_RELU),
  ('enc_conv3b', 'enc_conv3a', None,         FLAG_RELU | FLAG_POOL_DST),
  ('enc_conv4a', 'enc_conv3b', None,         FLAG_RELU),
  ('enc_conv4b', 'enc_conv4a', None,         FLAG_RELU | FLAG_POOL_DST),
  ('enc_conv5a', 'enc_conv4b', None,         FLAG_RELU),
  ('enc_conv5b', 'enc_conv5a', None,         FLAG_RELU),
  ('dec_conv4a', 'enc_conv5b', 'enc_conv3b', FLAG_RELU | FLAG_UPSAMPLE_SRC0),
  ('dec_conv4b', 'dec_conv4a', None,         FLAG_RELU),
  ('dec_conv3a', 'dec_conv4b', 'enc_conv2b', FLAG_RELU | FLAG_UPSAMPLE_SRC0),
  ('dec_conv3b', 'dec_conv3a', None,         FLAG_RELU),
  ('dec_conv2a', 'dec_conv3b', 'enc_conv1b', FLAG_RELU | FLAG_UPSAMPLE_SRC0),
  ('dec_conv2b', 'dec_conv2a', None,         FLAG_RELU),
  ('dec_conv1a', 'dec_conv2b', 'input',      FLAG_RELU | FLAG_UPSAMPLE_SRC0),
  ('dec_conv1b', 'dec_conv1a', None,         FLAG_RELU),
  ('dec_conv1c', 'dec_conv1b', None,         FLAG_RELU),
]

HEADER_FORMAT = '<8sIIIIIII3IQQ'
LAYER_FORMAT = '<24siiIIIIIIII'
assert struct.calcsize(HEADER_FORMAT) == 64
assert struct.calcsize(LAYER_FORMAT) == 64


def round_up(x, a):
  return (x + a - 1) // a * a


# -------------------------------------------------------------------------------------------------
# TZA parsing
# -------------------------------------------------------------------------------------------------

class Tensor:
  def __init__(self, dims, layout, values):
    self.dims = dims       # tuple of ints
    self.layout = layout   # 'x' or 'oihw'
    self.values = values   # flat list of floats


def parse_tza(path):
  data = open(path, 'rb').read()
  magic, major, _minor, table_offset = struct.unpack_from('<HBBQ', data, 0)
  if magic != 0x41D7 or major != 2:
    raise ValueError('%s: not a version 2 TZA file' % path)

  num_tensors = struct.unpack_from('<I', data, table_offset)[0]
  p = table_offset + 4
  tensors = {}
  for _ in range(num_tensors):
    name_len = struct.unpack_from('<H', data, p)[0]; p += 2
    name = data[p:p+name_len].decode(); p += name_len
    ndims = data[p]; p += 1
    dims = struct.unpack_from('<%dI' % ndims, data, p); p += 4 * ndims
    layout = data[p:p+ndims].decode(); p += ndims
    data_type = chr(data[p]); p += 1
    offset = struct.unpack_from('<Q', data, p)[0]; p += 8

    count = 1
    for d in dims:
      count *= d
    if data_type == 'h':
      values = struct.unpack_from('<%de' % count, data, offset)
    elif data_type == 'f':
      values = struct.unpack_from('<%df' % count, data, offset)
    else:
      raise ValueError('%s: unsupported data type %s' % (path, data_type))
    tensors[name] = Tensor(dims, layout, values)

  return tensors


# -------------------------------------------------------------------------------------------------
# Packing
# -------------------------------------------------------------------------------------------------

def pack(tensors):
  large = 'enc_conv1b.weight' in tensors
  topology = UNET_LARGE if large else UNET_BASE

  first_weight = tensors[topology[0][0] + '.weight']
  input_channels = first_weight.dims[1]
  input_padded = round_up(input_channels, 16)

  # Logical and padded output channels of each layer (and of the input)
  logical = {'input': input_channels}
  padded = {'input': input_padded}
  index = {'input': SRC_INPUT}

  layers = []
  data = bytearray()

  def align_data():
    data.extend(b'\0' * (round_up(len(data), ALIGNMENT) - len(data)))

  for layer_index, (name, src0, src1, flags) in enumerate(topology):
    weight = tensors.get(name + '.weight')
    bias = tensors.get(name + '.bias')
    if weight is None or bias is None:
      raise ValueError('missing tensors for layer %s' % name)
    if weight.layout != 'oihw' or len(weight.dims) != 4 or weight.dims[2:] != (3, 3):
      raise ValueError('layer %s: unsupported weight shape %s' % (name, weight.dims))

    O, I = weight.dims[0], weight.dims[1]
    if bias.dims != (O,):
      raise ValueError('layer %s: bias shape mismatch' % name)

    # Input channels: first source, then second source, each padded to 16
    C0_logical, C0 = logical[src0], padded[src0]
    C1_logical, C1 = (logical[src1], padded[src1]) if src1 else (0, 0)
    if I != C0_logical + C1_logical:
      raise ValueError('layer %s: %d input channels, expected %d + %d' % (name, I, C0_logical, C1_logical))

    Opad = round_up(O, 16)
    num_blocks = Opad // 16
    num_chunks = (C0 + C1) // 16

    # Maps a padded input channel to a logical input channel (-1 for padding)
    def logical_input(i):
      if i < C0:
        return i if i < C0_logical else -1
      j = i - C0
      return C0_logical + j if j < C1_logical else -1

    input_map = [logical_input(i) for i in range(C0 + C1)]
    w = weight.values

    # Weights: [K chunk][tap][output block][16 input][16 output]
    align_data()
    weight_offset = len(data)
    packed = []
    for kc in range(num_chunks):
      for tap in range(9):
        ky, kx = divmod(tap, 3)
        for nb in range(num_blocks):
          for ic in range(16):
            i = input_map[kc * 16 + ic]
            for oc in range(16):
              o = nb * 16 + oc
              if i < 0 or o >= O:
                packed.append(0.)
              else:
                packed.append(w[((o * I + i) * 3 + ky) * 3 + kx])
    data.extend(struct.pack('<%de' % len(packed), *packed))

    # Bias replicated in 16 rows, FP32 and FP16
    bias_rows = []
    for nb in range(num_blocks):
      row = [bias.values[nb * 16 + oc] if nb * 16 + oc < O else 0. for oc in range(16)]
      bias_rows.extend(row * 16)

    align_data()
    bias_f32_offset = len(data)
    data.extend(struct.pack('<%df' % len(bias_rows), *bias_rows))

    align_data()
    bias_f16_offset = len(data)
    data.extend(struct.pack('<%de' % len(bias_rows), *bias_rows))

    layers.append(struct.pack(LAYER_FORMAT, name.encode(), index[src0],
                              index[src1] if src1 else SRC_NONE, flags,
                              C0, C1, Opad, O,
                              weight_offset, bias_f32_offset, bias_f16_offset))

    logical[name] = O
    padded[name] = Opad
    index[name] = layer_index

  align_data()

  output_channels = logical[topology[-1][0]]
  data_offset = round_up(64 + 64 * len(layers), ALIGNMENT)
  header = struct.pack(HEADER_FORMAT, MAGIC, VERSION, 64, len(layers),
                       input_channels, output_channels,
                       RECEPTIVE_FIELD_LARGE if large else RECEPTIVE_FIELD_BASE,
                       LARGE_NETWORK if large else 0,
                       0, 0, 0, data_offset, len(data))

  blob = bytearray(header)
  for layer in layers:
    blob.extend(layer)
  blob.extend(b'\0' * (data_offset - len(blob)))
  blob.extend(data)
  return bytes(blob)


# -------------------------------------------------------------------------------------------------
# Dump
# -------------------------------------------------------------------------------------------------

def dump(path):
  blob = open(path, 'rb').read()
  (magic, version, header_size, num_layers, input_channels, output_channels, receptive_field,
   flags, _r0, _r1, _r2, data_offset, data_size) = struct.unpack_from(HEADER_FORMAT, blob, 0)
  if magic != MAGIC:
    raise ValueError('%s: not a packed vk_oidn weights file' % path)

  print('%s: version %d, %d layers, %d -> %d channels, receptive field %d, %s network, data %d bytes'
        % (path, version, num_layers, input_channels, output_channels, receptive_field,
           'large' if flags & LARGE_NETWORK else 'base', data_size))

  def src_name(s):
    return {SRC_INPUT: 'input', SRC_NONE: '-'}.get(s, str(s))

  for i in range(num_layers):
    (name, src0, src1, lflags, c0, c1, cout, cout_logical,
     w_off, b32_off, b16_off) = struct.unpack_from(LAYER_FORMAT, blob, header_size + 64 * i)
    ops = []
    if lflags & FLAG_UPSAMPLE_SRC0: ops.append('upsample0')
    if lflags & FLAG_RELU: ops.append('relu')
    if lflags & FLAG_POOL_DST: ops.append('pool')
    print('  %2d %-12s src0=%-5s src1=%-5s in=%3d+%-3d out=%3d (%3d)  %s'
          % (i, name.rstrip(b'\0').decode(), src_name(src0), src_name(src1), c0, c1, cout,
             cout_logical, ' '.join(ops)))


def main():
  parser = argparse.ArgumentParser(description='Converts Open Image Denoise weights to vk_oidn packed weights.')
  parser.add_argument('input', nargs='+', help='input .tza files (or .vkoidnw files with --dump)')
  parser.add_argument('-o', '--output', help='output file (single input) or directory')
  parser.add_argument('--dump', action='store_true', help='print the contents of packed weights')
  args = parser.parse_args()

  if args.dump:
    for path in args.input:
      dump(path)
    return

  for path in args.input:
    name = os.path.splitext(os.path.basename(path))[0] + '.vkoidnw'
    if args.output and len(args.input) == 1 and not os.path.isdir(args.output):
      out_path = args.output
    else:
      out_path = os.path.join(args.output or '.', name)

    blob = pack(parse_tza(path))
    with open(out_path, 'wb') as f:
      f.write(blob)
    print('%s -> %s (%d bytes)' % (path, out_path, len(blob)))


if __name__ == '__main__':
  main()
