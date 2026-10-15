#!/bin/bash
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
# Runs the tensor core micro-benchmarks (see README.md). Usage: run_all.sh [path to bench executable]
cd "$(dirname "$0")"
GLSL="${VULKAN_SDK}/Bin/glslangValidator"
[ -x "$GLSL" ] || [ -x "$GLSL.exe" ] || GLSL="${VULKAN_SDK}/bin/glslangValidator"
BENCH="${1:-./build/Release/bench}"
mkdir -p spv
compile() { "$GLSL" -V --target-env vulkan1.3 --quiet "$@"; }
median() { sed -n 's/.*median \([0-9.]*\) ms.*/\1/p'; }

echo "== Cooperative matrix peak (register-only multiply-add chains, 8192 workgroups of 4 subgroups) =="
for cfg in "FP16 x FP16 -> FP16|-DTYPE_F16 -DACC_F16|16" "FP16 x FP16 -> FP32|-DTYPE_F16 -DACC_F32|16" \
           "FP8 x FP8 -> FP16|-DTYPE_FP8 -DACC_F16|32" "FP8 x FP8 -> FP32|-DTYPE_FP8 -DACC_F32|32" \
           "INT8 x INT8 -> INT32|-DTYPE_INT8 -DACC_I32|32"; do
  IFS='|' read -r name defs kk <<< "$cfg"
  best=0
  for nacc in 1 2 4 8; do
    compile $defs -DNACC=$nacc -o spv/peak.spv peak.comp || continue
    ms=$("$BENCH" spv/peak.spv 8192 1 1 --buf 1 --buf 4 --int 256 --iters 10 | median)
    t=$(python -c "print('%.0f' % (2*16*16*$kk*256*$nacc*8192*4/$ms/1e9))")
    [ "$t" -gt "$best" ] && best=$t
  done
  printf "  %-22s %4d TFLOPS (best of 1-8 independent chains)\n" "$name" "$best"
done

echo "== Cooperative vector matrix-vector chain (FP16, inferencing-optimal matrix, 1M invocations) =="
for n in 32 64 128; do
  compile -DN=$n -o spv/cvp.spv coopvec_peak.comp
  ms=$("$BENCH" spv/cvp.spv 16384 1 1 --buf 2 --buf 0.1 --buf 2 --int 64 --coopvec-matrix 1 $n $n 1 --iters 10 | median)
  python -c "print('  %3dx%-3d %.0f TFLOPS' % ($n, $n, 2.0*$n*$n*64*16384*64/$ms/1e9))"
done

echo "== 3x3 convolution with cooperative vectors, 1920x1088, 64 -> 64 channels (154 GFLOP) =="
for mode in "TAPS|64 64 9|" "ROWS|64 192 3|"; do
  IFS='|' read -r m mat _ <<< "$mode"
  variants='"" "-DNO_LOAD"'
  [ "$m" = "TAPS" ] && variants='"" "-DNO_LOAD" "-DMEM_ONLY"' # MEM_ONLY needs equal product input and output sizes
  eval "set -- $variants"
  for v in "$@"; do
    compile -D$m $v -o spv/conv.spv conv_coopvec.comp || continue
    ms=$("$BENCH" spv/conv.spv 60 272 1 --buf 260 --buf 0.1 --buf 256 --buf 0.01 --int 1920 --int 1088 \
         --coopvec-matrix 1 $mat --iters 10 | median)
    python -c "print('  %-5s %-11s %7.3f ms  %5.1f TFLOPS' % ('$m', '$v' or 'full', $ms, 2*1920*1088*576*64/$ms/1e9))"
  done
done
