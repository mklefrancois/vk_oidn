#!/usr/bin/env python3
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""Adds the RoundingModeRTE execution mode for 16-bit floats to all entry points of a SPIR-V module.

Slang has no option for the floating-point rounding mode, but some drivers default to
round-toward-zero for FP32 -> FP16 conversions, which biases the denoised output. The kernels
therefore request round-to-nearest-even explicitly (requires SPIR-V 1.4 or SPV_KHR_float_controls
and the shaderRoundingModeRTEFloat16 device property).

Usage:
  spirv_float_controls.py input.spv output.spv
"""

import struct
import sys

OP_CAPABILITY = 17
OP_ENTRY_POINT = 15
OP_EXECUTION_MODE = 16
CAPABILITY_ROUNDING_MODE_RTE = 4467
EXECUTION_MODE_ROUNDING_MODE_RTE = 4462


def instruction(opcode, operands):
  return [((len(operands) + 1) << 16) | opcode] + operands


def main():
  if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)

  data = open(sys.argv[1], 'rb').read()
  words = list(struct.unpack('<%dI' % (len(data) // 4), data))
  if words[0] != 0x07230203:
    sys.exit('error: %s is not a SPIR-V module' % sys.argv[1])

  header, body = words[:5], words[5:]

  # Split the instructions
  insts = []
  i = 0
  while i < len(body):
    count = body[i] >> 16
    if count == 0:
      sys.exit('error: invalid SPIR-V instruction')
    insts.append(body[i:i+count])
    i += count

  has_capability = any(inst[0] & 0xFFFF == OP_CAPABILITY and inst[1] == CAPABILITY_ROUNDING_MODE_RTE
                       for inst in insts)
  entry_points = [inst[2] for inst in insts if inst[0] & 0xFFFF == OP_ENTRY_POINT]

  out = []
  capability_added = has_capability
  modes_added = False
  for idx, inst in enumerate(insts):
    opcode = inst[0] & 0xFFFF
    out.append(inst)
    # Add the capability after the last existing capability
    if not capability_added and opcode == OP_CAPABILITY and \
       (idx + 1 == len(insts) or insts[idx + 1][0] & 0xFFFF != OP_CAPABILITY):
      out.append(instruction(OP_CAPABILITY, [CAPABILITY_ROUNDING_MODE_RTE]))
      capability_added = True
    # Add the execution modes after the last entry point
    if not modes_added and opcode == OP_ENTRY_POINT and \
       (idx + 1 == len(insts) or insts[idx + 1][0] & 0xFFFF != OP_ENTRY_POINT):
      for ep in entry_points:
        out.append(instruction(OP_EXECUTION_MODE, [ep, EXECUTION_MODE_ROUNDING_MODE_RTE, 16]))
      modes_added = True

  if not capability_added or not modes_added:
    sys.exit('error: could not patch the SPIR-V module')

  words = header + [w for inst in out for w in inst]
  with open(sys.argv[2], 'wb') as f:
    f.write(struct.pack('<%dI' % len(words), *words))


if __name__ == '__main__':
  main()
