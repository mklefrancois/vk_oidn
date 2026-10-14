// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// vk_oidn compute kernels (Slang)
//
// Derived from the Vulkan backend of Intel(R) Open Image Denoise. All memory is accessed through
// buffer device addresses passed as push constants (entry point uniform parameters), so the
// kernels need no descriptor sets.
//
// Entry points:
//   inputProcess             images (color/albedo/normal) -> f16 HWC tensor
//   outputProcess            f16 HWC tensor -> image
//   imageCopy                image -> image
//   autoexposureDownsample   image -> average luminance bins
//   autoexposureReduce       bins -> partial log2 sums
//   autoexposureReduceFinal  partial sums -> exposure scale
//   convCoopMat              3x3 convolution with cooperative matrices (tensor cores)
//   convCoopMatUp            same, with the first source 2x upsampled, using a sub-pixel
//                            decomposition (2.25x fewer multiply-adds for that source)
//   convSimple               3x3 convolution without cooperative matrices (fallback)
//
// Compile-time options (preprocessor):
//   MX       number of 16-pixel column fragments per subgroup row in convCoopMat (default 2)
//   N_FRAGS  number of 16-channel output blocks per workgroup in convCoopMat (default 2)
//   ACC_F16  FP16 accumulation in convCoopMat (default FP32)
//   IMAGE_IO inputProcess, outputProcess and autoexposureDownsample access storage images
//            (descriptor set 0 with push descriptors: binding 0 = color, 1 = albedo,
//            2 = normal, 3 = output) instead of buffers; the presence of albedo and normal is
//            indicated by bits 4 and 5 of the input processing flags; bit 6 of the output
//            processing flags indicates a 16-bit float output image
//
// Specialization constants:
//   0: WG_SIZE   workgroup size of convCoopMat (4 * subgroup size)
//   1: UPSAMPLE  2x nearest upsampling of the first convolution source
//   2: POOL      2x2 max pooling of the convolution output
//   3: RELU      ReLU activation
//
// Tensors are in f16 HWC layout with channels padded to multiples of 16.
// Packed weights: [K chunk][tap][16-channel output block][16 input][16 output] f16 (see
// src/weights_format.h).

#if !defined(MX)
  #define MX 2
#endif
#if !defined(N_FRAGS)
  #define N_FRAGS 2
#endif

[vk::constant_id(0)] const uint WG_SIZE = 128;
[vk::constant_id(1)] const bool UPSAMPLE = false;
[vk::constant_id(2)] const bool POOL = false;
[vk::constant_id(3)] const bool RELU = true;

static const float FLT_MAX = 3.402823466e+38;

// -------------------------------------------------------------------------------------------------
// Built-ins and helpers
// -------------------------------------------------------------------------------------------------

uint getSubgroupID()
{
  return spirv_asm { result:$$uint = OpLoad builtin(SubgroupId:uint); };
}

uint3 getNumWorkGroups()
{
  return spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
}

// Barrier for shared memory accesses within a subgroup
void subgroupSharedBarrier()
{
  GroupMemoryBarrierWithWaveSync();
}

uint packHalf2(float2 v)  { return reinterpret<uint>(half2(v)); }
float2 unpackHalf2(uint v) { return float2(reinterpret<half2>(v)); }

uint4 max8(uint4 a, uint4 b)
{
  uint4 r;
  [unroll]
  for (int i = 0; i < 4; ++i)
    r[i] = reinterpret<uint>(max(reinterpret<half2>(a[i]), reinterpret<half2>(b[i])));
  return r;
}

float3 nanToZero(float3 v)
{
  return float3(isnan(v.x) ? 0. : v.x, isnan(v.y) ? 0. : v.y, isnan(v.z) ? 0. : v.z);
}

float luminance(float3 c)
{
  return 0.212671 * c.x + 0.715160 * c.y + 0.072169 * c.z;
}

float16_t toHalf(float x)
{
  return float16_t(clamp(x, -65504., 65504.)); // avoid infinities
}

// -------------------------------------------------------------------------------------------------
// Image access
// -------------------------------------------------------------------------------------------------

// Packed image format (must match the host code)
//   bits  0-23: pixel byte stride
//   bits 24-25: number of channels (1-3)
//   bit     28: half precision
uint imageWByteStride(uint fmt) { return fmt & 0xFFFFFFu; }
uint imageC(uint fmt)           { return (fmt >> 24) & 0x3u; }
bool imageIsHalf(uint fmt)      { return (fmt & (1u << 28)) != 0u; }

uint64_t imagePixelAddress(uint64_t ptr, uint hByteStride, uint fmt, int h, int w)
{
  return ptr + uint64_t(h) * uint64_t(hByteStride) + uint64_t(w) * uint64_t(imageWByteStride(fmt));
}

float3 imageLoad3(uint64_t ptr, uint hByteStride, uint fmt, int h, int w)
{
  const uint64_t addr = imagePixelAddress(ptr, hByteStride, fmt, h, w);
  const uint C = imageC(fmt);
  float3 c;
  if (imageIsHalf(fmt))
  {
    // Load the pixel with aligned 32-bit loads because 16-bit loads from addresses which are not
    // 4-byte aligned are not handled correctly by some drivers
    const uint64_t addr4 = addr & ~3ull;
    uint* p = (uint*)addr4;
    const uint w0 = p[0];
    const uint w1 = (C >= 2u || (addr & 2ull) != 0ull) ? p[1] : 0u;
    if ((addr & 2ull) == 0ull)
      c = float3(unpackHalf2(w0), unpackHalf2(w1).x);
    else
      c = float3(unpackHalf2(w0).y, unpackHalf2(w1));
  }
  else
  {
    float* p = (float*)addr;
    c = float3(p[0], C >= 2u ? p[1] : 0., C >= 3u ? p[2] : 0.);
  }

  if (C == 3u)
    return c;
  else if (C == 2u)
    return c.xyy;
  else
    return c.xxx;
}

void imageStore3(uint64_t ptr, uint hByteStride, uint fmt, int h, int w, float3 v)
{
  const uint64_t addr = imagePixelAddress(ptr, hByteStride, fmt, h, w);
  const uint C = imageC(fmt);
  if (imageIsHalf(fmt))
  {
    half* p = (half*)addr;
    p[0] = half(v.x);
    if (C >= 2u) p[1] = half(v.y);
    if (C >= 3u) p[2] = half(v.z);
  }
  else
  {
    float* p = (float*)addr;
    p[0] = v.x;
    if (C >= 2u) p[1] = v.y;
    if (C >= 3u) p[2] = v.z;
  }
}

// -------------------------------------------------------------------------------------------------
// Image I/O: buffers or storage images (IMAGE_IO)
// -------------------------------------------------------------------------------------------------

#if defined(IMAGE_IO)
[[vk::binding(0, 0)]] RWTexture2D<float4> colorImage;
[[vk::binding(1, 0)]] RWTexture2D<float4> albedoImage;
[[vk::binding(2, 0)]] RWTexture2D<float4> normalImage;
[[vk::binding(3, 0)]] RWTexture2D<float4> outputImage;
#endif

static const int IMAGE_COLOR  = 0;
static const int IMAGE_ALBEDO = 1;
static const int IMAGE_NORMAL = 2;

// Loads the RGB value of a pixel of an input image (the buffer parameters are ignored for images)
float3 loadInput(int image, uint64_t ptr, uint hByteStride, uint fmt, int h, int w)
{
#if defined(IMAGE_IO)
  const int2 pos = int2(w, h);
  if (image == IMAGE_ALBEDO)
    return albedoImage[pos].xyz;
  else if (image == IMAGE_NORMAL)
    return normalImage[pos].xyz;
  else
    return colorImage[pos].xyz;
#else
  return imageLoad3(ptr, hByteStride, fmt, h, w);
#endif
}

// Stores the RGB value of a pixel of the output image (the alpha of output images is preserved)
void storeOutput(uint64_t ptr, uint hByteStride, uint fmt, int flags, int h, int w, float3 v)
{
#if defined(IMAGE_IO)
  // Round to FP16 with round-to-nearest-even for 16-bit float images (the rounding of the
  // storage image conversion is implementation-defined)
  if ((flags & 64) != 0)
    v = float3(half3(v));
  const int2 pos = int2(w, h);
  outputImage[pos] = float4(v, outputImage[pos].w);
#else
  imageStore3(ptr, hByteStride, fmt, h, w, v);
#endif
}

// Returns whether an auxiliary input image is present
bool hasInput(uint64_t ptr, int flags, int flag)
{
#if defined(IMAGE_IO)
  return (flags & flag) != 0;
#else
  return ptr != 0ull;
#endif
}

// -------------------------------------------------------------------------------------------------
// Transfer function (must match core/color.h of Open Image Denoise)
// -------------------------------------------------------------------------------------------------

static const int TRANSFER_LINEAR = 0;
static const int TRANSFER_SRGB   = 1;
static const int TRANSFER_PU     = 2;
static const int TRANSFER_LOG    = 3;

static const float SRGB_A  =  12.92;
static const float SRGB_B  =  1.055;
static const float SRGB_C  =  1. / 2.4;
static const float SRGB_D  = -0.055;
static const float SRGB_Y0 =  0.0031308;
static const float SRGB_X0 =  0.04045;

float srgbForward(float y) { return (y <= SRGB_Y0) ? (SRGB_A * y) : (SRGB_B * pow(y, SRGB_C) + SRGB_D); }
float srgbInverse(float x) { return (x <= SRGB_X0) ? (x / SRGB_A) : pow((x - SRGB_D) / SRGB_B, 1. / SRGB_C); }

static const float PU_A  =  1.41283765e+03;
static const float PU_B  =  1.64593172e+00;
static const float PU_C  =  4.31384981e-01;
static const float PU_D  = -2.94139609e-03;
static const float PU_E  =  1.92653254e-01;
static const float PU_F  =  6.26026094e-03;
static const float PU_G  =  9.98620152e-01;
static const float PU_Y0 =  1.57945760e-06;
static const float PU_Y1 =  3.22087631e-02;
static const float PU_X0 =  2.23151711e-03;
static const float PU_X1 =  3.70974749e-01;

float puForward(float y)
{
  if (y <= PU_Y0)
    return PU_A * y;
  else if (y <= PU_Y1)
    return PU_B * pow(y, PU_C) + PU_D;
  else
    return PU_E * log(y + PU_F) + PU_G;
}

float puInverse(float x)
{
  if (x <= PU_X0)
    return x / PU_A;
  else if (x <= PU_X1)
    return pow((x - PU_D) / PU_B, 1. / PU_C);
  else
    return exp((x - PU_G) / PU_E) - PU_F;
}

float3 transferForward(int type, float normScale, float3 y)
{
  switch (type)
  {
  case TRANSFER_SRGB: return float3(srgbForward(y.x), srgbForward(y.y), srgbForward(y.z));
  case TRANSFER_PU:   return float3(puForward(y.x), puForward(y.y), puForward(y.z)) * normScale;
  case TRANSFER_LOG:  return log(y + 1.) * normScale;
  default:            return y;
  }
}

float3 transferInverse(int type, float rcpNormScale, float3 x)
{
  switch (type)
  {
  case TRANSFER_SRGB: return float3(srgbInverse(x.x), srgbInverse(x.y), srgbInverse(x.z));
  case TRANSFER_PU:   return float3(puInverse(x.x * rcpNormScale), puInverse(x.y * rcpNormScale), puInverse(x.z * rcpNormScale));
  case TRANSFER_LOG:  return exp(x * rcpNormScale) - 1.;
  default:            return x;
  }
}

// -------------------------------------------------------------------------------------------------
// Input processing: images -> f16 HWC tensor (zero-padded outside the tile)
// -------------------------------------------------------------------------------------------------

struct InputProcessParams
{
  uint64_t dst;           // f16 hwc tensor
  uint64_t inputScalePtr; // optional autoexposure result
  uint64_t inputPtr;      // color, albedo or normal
  uint64_t albedoPtr;     // auxiliary albedo (optional)
  uint64_t normalPtr;     // auxiliary normal (optional)
  uint inputHStride;
  uint inputFmt;
  uint albedoHStride;
  uint albedoFmt;
  uint normalHStride;
  uint normalFmt;
  int dstH;
  int dstW;
  int dstC;
  int flags; // bits 0-1: transfer function, bit 2: hdr, bit 3: snorm
  int hSrcBegin;
  int wSrcBegin;
  int hDstBegin;
  int wDstBegin;
  int tileH;
  int tileW;
  float inputScale;
  float normScale;
};

[shader("compute")]
[numthreads(64, 4, 1)]
void inputProcess(uint3 globalID : SV_DispatchThreadID, uniform InputProcessParams p)
{
  const int hDst = int(globalID.y);
  const int wDst = int(globalID.x);
  if (hDst >= p.dstH || wDst >= p.dstW)
    return;

  const int h = hDst - p.hDstBegin;
  const int w = wDst - p.wDstBegin;

  const int transferType = p.flags & 3;
  const bool hdr   = (p.flags & 4) != 0;
  const bool snorm = (p.flags & 8) != 0;

  float values[9] = { 0., 0., 0., 0., 0., 0., 0., 0., 0. };

  if (h >= 0 && h < p.tileH && w >= 0 && w < p.tileW)
  {
    const int hSrc = h + p.hSrcBegin;
    const int wSrc = w + p.wSrcBegin;

    // Input
    // The scale is read from a valid address even without autoexposure (inputScalePtr = 0), then
    // selected: some implementations (e.g. lavapipe) perform loads from null pointers inside
    // uniform branches that are not taken. The value read from dst is not used.
    const bool scaleFromPtr = p.inputScalePtr != 0ull;
    const float loadedScale = *(float*)(scaleFromPtr ? p.inputScalePtr : p.dst);
    const float inputScale = scaleFromPtr ? loadedScale : p.inputScale;
    float3 v = loadInput(IMAGE_COLOR, p.inputPtr, p.inputHStride, p.inputFmt, hSrc, wSrc) * inputScale;
    v = clamp(nanToZero(v), snorm ? -1. : 0., hdr ? FLT_MAX : 1.);
    if (snorm)
      v = v * 0.5 + 0.5;
    v = transferForward(transferType, p.normScale, v);
    values[0] = v.x;
    values[1] = v.y;
    values[2] = v.z;

    // Albedo
    if (hasInput(p.albedoPtr, p.flags, 16))
    {
      const float3 a = clamp(nanToZero(loadInput(IMAGE_ALBEDO, p.albedoPtr, p.albedoHStride, p.albedoFmt, hSrc, wSrc)), 0., 1.);
      values[3] = a.x;
      values[4] = a.y;
      values[5] = a.z;

      // Normal
      if (hasInput(p.normalPtr, p.flags, 32))
      {
        float3 n = clamp(nanToZero(loadInput(IMAGE_NORMAL, p.normalPtr, p.normalHStride, p.normalFmt, hSrc, wSrc)), -1., 1.);
        n = n * 0.5 + 0.5;
        values[6] = n.x;
        values[7] = n.y;
        values[8] = n.z;
      }
    }
  }

  // Store the padded channels as 16-byte blocks
  uint4* dst = (uint4*)(p.dst + uint64_t((hDst * p.dstW + wDst) * p.dstC) * 2ull);
  for (int c = 0; c < p.dstC; c += 8)
  {
    uint4 block = uint4(0);
    if (c < 9)
    {
      float v[8];
      [unroll]
      for (int i = 0; i < 8; ++i)
        v[i] = (c + i < 9) ? values[min(c + i, 8)] : 0.;
      block = uint4(packHalf2(float2(v[0], v[1])), packHalf2(float2(v[2], v[3])),
                    packHalf2(float2(v[4], v[5])), packHalf2(float2(v[6], v[7])));
    }
    dst[c / 8] = block;
  }
}

// -------------------------------------------------------------------------------------------------
// Output processing: f16 HWC tensor -> image
// -------------------------------------------------------------------------------------------------

struct OutputProcessParams
{
  uint64_t src;           // f16 hwc tensor
  uint64_t dst;           // image
  uint64_t inputScalePtr; // optional autoexposure result
  uint dstHStride;
  uint dstFmt;
  int srcW;
  int srcC;
  int hSrcBegin;
  int wSrcBegin;
  int hDstBegin;
  int wDstBegin;
  int tileH;
  int tileW;
  int flags; // bits 0-1: transfer function, bit 2: hdr, bit 3: snorm
  float outputScale;
  float rcpNormScale;
};

[shader("compute")]
[numthreads(64, 4, 1)]
void outputProcess(uint3 globalID : SV_DispatchThreadID, uniform OutputProcessParams p)
{
  const int h = int(globalID.y);
  const int w = int(globalID.x);
  if (h >= p.tileH || w >= p.tileW)
    return;

  const int hSrc = h + p.hSrcBegin;
  const int wSrc = w + p.wSrcBegin;
  const int hDst = h + p.hDstBegin;
  const int wDst = w + p.wDstBegin;

  const int transferType = p.flags & 3;
  const bool hdr   = (p.flags & 4) != 0;
  const bool snorm = (p.flags & 8) != 0;

  // Load the first 4 channels with a single 8-byte load
  const uint2 srcBlock = *(uint2*)(p.src + uint64_t((hSrc * p.srcW + wSrc) * p.srcC) * 2ull);
  const float2 v01 = unpackHalf2(srcBlock.x);
  const float2 v23 = unpackHalf2(srcBlock.y);
  float3 v = float3(v01.x, v01.y, v23.x);

  v = clamp(nanToZero(v), 0., FLT_MAX);
  v = transferInverse(transferType, p.rcpNormScale, v);

  if (imageC(p.dstFmt) == 1u)
    v = float3((v.x + v.y + v.z) * (1. / 3.));

  if (snorm)
    v = max(v * 2. - 1., -1.);

  if (!hdr)
    v = min(v, 1.);

  // Read from a valid address even without autoexposure (see inputProcess), then select
  float outputScale = p.outputScale;
  const bool scaleFromPtr = p.inputScalePtr != 0ull;
  const float inputScale = *(float*)(scaleFromPtr ? p.inputScalePtr : p.src);
  if (scaleFromPtr)
    outputScale = (inputScale != 0.) ? (1. / inputScale) : 0.;

  storeOutput(p.dst, p.dstHStride, p.dstFmt, p.flags, hDst, wDst, v * outputScale);
}

// -------------------------------------------------------------------------------------------------
// Image copy
// -------------------------------------------------------------------------------------------------

struct ImageCopyParams
{
  uint64_t src;
  uint64_t dst;
  uint srcHStride;
  uint srcFmt;
  uint dstHStride;
  uint dstFmt;
  int H;
  int W;
};

[shader("compute")]
[numthreads(64, 4, 1)]
void imageCopy(uint3 globalID : SV_DispatchThreadID, uniform ImageCopyParams p)
{
  const int h = int(globalID.y);
  const int w = int(globalID.x);
  if (h >= p.H || w >= p.W)
    return;

  imageStore3(p.dst, p.dstHStride, p.dstFmt, h, w, imageLoad3(p.src, p.srcHStride, p.srcFmt, h, w));
}

// -------------------------------------------------------------------------------------------------
// Autoexposure
// -------------------------------------------------------------------------------------------------

static const int AE_MAX_BIN_SIZE = 16;
static const int AE_DOWNSAMPLE_GROUP_SIZE = AE_MAX_BIN_SIZE * AE_MAX_BIN_SIZE;
static const int AE_REDUCE_GROUP_SIZE = 256;
static const float AE_EPS = 1e-8;
static const float AE_KEY = 0.18;

struct AutoexposureDownsampleParams
{
  uint64_t src;
  uint64_t bins;
  uint srcHStride;
  uint srcFmt;
  int H;
  int W;
};

groupshared float aeSums[AE_REDUCE_GROUP_SIZE];
groupshared int aeCounts[AE_REDUCE_GROUP_SIZE];

[shader("compute")]
[numthreads(AE_MAX_BIN_SIZE, AE_MAX_BIN_SIZE, 1)]
void autoexposureDownsample(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID,
                            uint localIndex : SV_GroupIndex,
                            uniform AutoexposureDownsampleParams p)
{
  const uint3 numGroups = getNumWorkGroups();
  const int numGroupsH = int(numGroups.y);
  const int numGroupsW = int(numGroups.x);
  const int groupH = int(groupID.y);
  const int groupW = int(groupID.x);

  const int beginH = groupH * p.H / numGroupsH;
  const int beginW = groupW * p.W / numGroupsW;
  const int endH = (groupH + 1) * p.H / numGroupsH;
  const int endW = (groupW + 1) * p.W / numGroupsW;

  const int h = beginH + int(localID.y);
  const int w = beginW + int(localID.x);

  float L = 0.;
  if (h < endH && w < endW)
  {
    const float3 c = clamp(nanToZero(loadInput(IMAGE_COLOR, p.src, p.srcHStride, p.srcFmt, h, w)), 0., FLT_MAX);
    L = luminance(c);
  }

  aeSums[localIndex] = L;

  for (uint i = AE_DOWNSAMPLE_GROUP_SIZE / 2; i > 0; i >>= 1)
  {
    GroupMemoryBarrierWithGroupSync();
    if (localIndex < i)
      aeSums[localIndex] += aeSums[localIndex + i];
  }

  if (localIndex == 0)
  {
    const float avgL = aeSums[0] / float((endH - beginH) * (endW - beginW));
    ((float*)p.bins)[groupH * numGroupsW + groupW] = avgL;
  }
}

struct AutoexposureReduceParams
{
  uint64_t bins;
  uint64_t sums;
  uint64_t counts;
  int size;
};

[shader("compute")]
[numthreads(AE_REDUCE_GROUP_SIZE, 1, 1)]
void autoexposureReduce(uint3 globalID : SV_DispatchThreadID, uint3 groupID : SV_GroupID,
                        uint localIndex : SV_GroupIndex,
                        uniform AutoexposureReduceParams p)
{
  float sum = 0.;
  int count = 0;
  const int globalSize = int(getNumWorkGroups().x) * AE_REDUCE_GROUP_SIZE;
  for (int i = int(globalID.x); i < p.size; i += globalSize)
  {
    const float L = ((float*)p.bins)[i];
    if (L > AE_EPS)
    {
      sum += log2(L);
      ++count;
    }
  }

  aeSums[localIndex] = sum;
  aeCounts[localIndex] = count;

  for (uint i = AE_REDUCE_GROUP_SIZE / 2; i > 0; i >>= 1)
  {
    GroupMemoryBarrierWithGroupSync();
    if (localIndex < i)
    {
      aeSums[localIndex] += aeSums[localIndex + i];
      aeCounts[localIndex] += aeCounts[localIndex + i];
    }
  }

  if (localIndex == 0)
  {
    ((float*)p.sums)[groupID.x] = aeSums[0];
    ((int*)p.counts)[groupID.x] = aeCounts[0];
  }
}

struct AutoexposureReduceFinalParams
{
  uint64_t sums;
  uint64_t counts;
  uint64_t result;
  int size;
};

[shader("compute")]
[numthreads(AE_REDUCE_GROUP_SIZE, 1, 1)]
void autoexposureReduceFinal(uint localIndex : SV_GroupIndex,
                             uniform AutoexposureReduceFinalParams p)
{
  if (int(localIndex) < p.size)
  {
    aeSums[localIndex] = ((float*)p.sums)[localIndex];
    aeCounts[localIndex] = ((int*)p.counts)[localIndex];
  }
  else
  {
    aeSums[localIndex] = 0.;
    aeCounts[localIndex] = 0;
  }

  for (uint i = AE_REDUCE_GROUP_SIZE / 2; i > 0; i >>= 1)
  {
    GroupMemoryBarrierWithGroupSync();
    if (localIndex < i)
    {
      aeSums[localIndex] += aeSums[localIndex + i];
      aeCounts[localIndex] += aeCounts[localIndex + i];
    }
  }

  if (localIndex == 0)
  {
    *(float*)p.result = (aeCounts[0] > 0) ? (AE_KEY / exp2(aeSums[0] / float(aeCounts[0]))) : 1.;
  }
}

// -------------------------------------------------------------------------------------------------
// 3x3 convolution: common
// -------------------------------------------------------------------------------------------------
//
// The convolution is computed as an implicit GEMM:
//   M = output pixels, N = output channels, K = 3x3 taps * input channels
// The input channels are processed in chunks of 16. For each chunk, the input tile with a 1-pixel
// halo is loaded into shared memory (zero-padded at the image borders), optionally upsampling the
// first source (nearest) and concatenating the second source along the channel dimension.

struct ConvParams
{
  uint64_t src0;   // first source (half resolution if upsampled)
  uint64_t src1;   // second source (concatenated, optional)
  uint64_t weight; // packed weights
  uint64_t bias;   // packed bias (replicated in 16 rows)
  uint64_t dst;    // destination (half resolution if pooled)
  int H;           // convolution height
  int W;           // convolution width
  int C0;          // padded channels of the first source
  int C1;          // padded channels of the second source (0 if none)
  int NB;          // number of 16-channel output blocks
};

static const int KSTR = 3; // uint4 elements per pixel in shared memory (16 halfs + 8 halfs padding)

// Loads 8 channels of an input tile pixel (zero outside the image)
uint4 loadConvInput(ConvParams p, int kc, int y, int x, int part)
{
  const int numChunks0 = p.C0 / 16;
  const bool fromSrc0 = kc < numChunks0;
  if (y < 0 || y >= p.H || x < 0 || x >= p.W)
    return uint4(0);

  const uint64_t base = fromSrc0 ? p.src0 : p.src1;
  const int C = fromSrc0 ? p.C0 : p.C1;
  const int c = (fromSrc0 ? kc : kc - numChunks0) * 16;
  const bool up = fromSrc0 && UPSAMPLE;
  const int srcW = up ? (p.W >> 1) : p.W;
  const int sy = up ? (y >> 1) : y;
  const int sx = up ? (x >> 1) : x;
  return *(uint4*)(base + (uint64_t(sy * srcW + sx) * uint64_t(C) + uint64_t(c + part * 8)) * 2ull);
}

// Returns the byte offset of a packed 16x16 weight block
uint64_t getWeightOffset(ConvParams p, int kc, int tap, int nb)
{
  return uint64_t((kc * 9 + tap) * p.NB + nb) * 512ull;
}

// Returns the address of 8 channels of a destination pixel
uint64_t getDstAddress(ConvParams p, int y, int x, int dstW, int c)
{
  return p.dst + (uint64_t(y * dstW + x) * uint64_t(p.NB * 16) + uint64_t(c)) * 2ull;
}

// -------------------------------------------------------------------------------------------------
// 3x3 convolution with cooperative matrices
// -------------------------------------------------------------------------------------------------
//
// The workgroup computes an output tile of TH x TW pixels and N_FRAGS * 16 output channels.
// Each of the NUM_SG subgroups computes 2 rows of MX * 16 pixels (2 * MX M fragments) for all
// N_FRAGS output channel blocks. The two rows of a subgroup form the rows of the 2x2 pooling
// windows, so pooling can be fused without exchanging data between subgroups.
// The input tiles are double-buffered in shared memory: the next chunk is prefetched into
// registers while the current one is being multiplied, so there is one barrier per chunk.

using namespace linalg;

#if defined(ACC_F16)
  typealias AccT = half;
#else
  typealias AccT = float;
#endif

typealias MatA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias MatC = CoopMat<AccT, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
typealias MatO = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

static const int CM_NUM_SG = 4;                       // number of subgroups
static const int CM_TH = 2 * CM_NUM_SG;               // tile height
static const int CM_TW = 16 * MX;                     // tile width
static const int CM_IN_W = CM_TW + 2;                 // input tile width with halo
static const int CM_IN_H = CM_TH + 2;                 // input tile height with halo
static const int CM_IN_TILE_SIZE = CM_IN_H * CM_IN_W * KSTR; // input tile size in uint4 units
static const int CM_NUM_LOADS = CM_IN_H * CM_IN_W * 2;       // uint4 global loads per input tile
// Maximum number of global loads per invocation (assuming at least 32 invocations per subgroup)
static const int CM_MAX_LOADS = (CM_NUM_LOADS + CM_NUM_SG * 32 - 1) / (CM_NUM_SG * 32);

groupshared uint4 cmIn[2 * CM_IN_TILE_SIZE];

// Stores an output fragment (16 pixels x 16 channels) of a row
void storeFragment(ConvParams p, MatO o, int sg, uint lane, uint numLanes, int y, int x0, int nb)
{
  if (x0 + 16 <= p.W)
  {
    // Store directly to the destination
    o.Store<CoopMatMatrixLayout.RowMajor>((half*)getDstAddress(p, y, x0, p.W, nb * 16), 0, uint(p.NB * 16));
  }
  else if (x0 < p.W)
  {
    // Partial fragment: store through shared memory (the staging area of the subgroup)
    o.Store<CoopMatMatrixLayout.RowMajor>(cmIn, uint(sg * 32), 2);
    subgroupSharedBarrier();

    for (uint i = lane; i < 32u; i += numLanes)
    {
      const int px = int(i >> 1);
      const int part = int(i & 1);
      if (x0 + px < p.W)
        *(uint4*)getDstAddress(p, y, x0 + px, p.W, nb * 16 + part * 8) = cmIn[sg * 32 + int(i)];
    }

    subgroupSharedBarrier();
  }
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void convCoopMat(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex, uniform ConvParams p)
{
  const int sg = int(getSubgroupID());
  const uint lane = WaveGetLaneIndex();
  const uint numLanes = WaveGetLaneCount();
  const int x0 = int(groupID.x) * CM_TW;
  const int y0 = int(groupID.y) * CM_TH;
  const int nb0 = int(groupID.z) * N_FRAGS;

  // Start loading the first input chunk
  uint4 prefetch[CM_MAX_LOADS];
  [unroll]
  for (int j = 0; j < CM_MAX_LOADS; ++j)
  {
    const int i = int(localIndex) + j * int(WG_SIZE);
    const int pix = i >> 1;
    const int ty = pix / CM_IN_W;
    const int tx = pix - ty * CM_IN_W;
    prefetch[j] = (i < CM_NUM_LOADS) ? loadConvInput(p, 0, y0 + ty - 1, x0 + tx - 1, i & 1) : uint4(0);
  }

  // Initialize the accumulators with the bias
  MatC acc[2][MX][N_FRAGS];
  [unroll]
  for (int n = 0; n < N_FRAGS; ++n)
  {
    MatC b = MatC(AccT(0.));
    if (nb0 + n < p.NB)
    {
    #if defined(ACC_F16)
      b = MatC.Load<CoopMatMatrixLayout.RowMajor>((half*)(p.bias + uint64_t(nb0 + n) * 512ull), 0, 16);
    #else
      b = MatC.Load<CoopMatMatrixLayout.RowMajor>((float*)(p.bias + uint64_t(nb0 + n) * 1024ull), 0, 16);
    #endif
    }
    [unroll]
    for (int m = 0; m < 2; ++m)
      [unroll]
      for (int mx = 0; mx < MX; ++mx)
        acc[m][mx][n] = b;
  }

  [unroll]
  for (int j = 0; j < CM_MAX_LOADS; ++j)
  {
    const int i = int(localIndex) + j * int(WG_SIZE);
    if (i < CM_NUM_LOADS)
      cmIn[(i >> 1) * KSTR + (i & 1)] = prefetch[j];
  }
  GroupMemoryBarrierWithGroupSync();

  // Iterate over the input channel chunks
  const int numChunks = (p.C0 + p.C1) / 16;
  for (int kc = 0; kc < numChunks; ++kc)
  {
    const int buf = kc & 1;

    // Prefetch the next chunk
    if (kc + 1 < numChunks)
    {
      [unroll]
      for (int j = 0; j < CM_MAX_LOADS; ++j)
      {
        const int i = int(localIndex) + j * int(WG_SIZE);
        const int pix = i >> 1;
        const int ty = pix / CM_IN_W;
        const int tx = pix - ty * CM_IN_W;
        prefetch[j] = (i < CM_NUM_LOADS) ? loadConvInput(p, kc + 1, y0 + ty - 1, x0 + tx - 1, i & 1) : uint4(0);
      }
    }

    [unroll]
    for (int tap = 0; tap < 9; ++tap)
    {
      const int ky = tap / 3;
      const int kx = tap - ky * 3;

      MatB b[N_FRAGS];
      [unroll]
      for (int n = 0; n < N_FRAGS; ++n)
      {
        if (nb0 + n < p.NB)
          b[n] = MatB.Load<CoopMatMatrixLayout.RowMajor>((half*)(p.weight + getWeightOffset(p, kc, tap, nb0 + n)), 0, 16);
      }

      [unroll]
      for (int m = 0; m < 2; ++m)
      {
        [unroll]
        for (int mx = 0; mx < MX; ++mx)
        {
          const int row = sg * 2 + m;
          const MatA a = MatA.Load<CoopMatMatrixLayout.RowMajor>(
            cmIn, uint(buf * CM_IN_TILE_SIZE + ((row + ky) * CM_IN_W + mx * 16 + kx) * KSTR), KSTR);

          [unroll]
          for (int n = 0; n < N_FRAGS; ++n)
          {
            if (nb0 + n < p.NB)
              acc[m][mx][n] = coopMatMulAdd<AccT, false>(a, b[n], acc[m][mx][n]);
          }
        }
      }
    }

    // Store the next chunk into the other buffer, which is no longer being read
    if (kc + 1 < numChunks)
    {
      [unroll]
      for (int j = 0; j < CM_MAX_LOADS; ++j)
      {
        const int i = int(localIndex) + j * int(WG_SIZE);
        if (i < CM_NUM_LOADS)
          cmIn[(buf ^ 1) * CM_IN_TILE_SIZE + (i >> 1) * KSTR + (i & 1)] = prefetch[j];
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }

  // Epilogue (the input tile buffers are reused as staging areas)
  [unroll]
  for (int n = 0; n < N_FRAGS; ++n)
  {
    const int nb = nb0 + n;
    if (nb >= p.NB)
      continue;

    if (RELU)
    {
      [unroll]
      for (int m = 0; m < 2; ++m)
        [unroll]
        for (int mx = 0; mx < MX; ++mx)
          for (int i = 0; i < acc[m][mx][n].GetLength(); ++i)
            acc[m][mx][n][i] = max(acc[m][mx][n][i], AccT(0.));
    }

    if (POOL)
    {
      const int yo = (y0 >> 1) + sg;
      const int Wo = p.W >> 1;

      [unroll]
      for (int mx = 0; mx < MX; ++mx)
      {
        // Vertical max
        MatC pooled = acc[0][mx][n];
        for (int i = 0; i < pooled.GetLength(); ++i)
          pooled[i] = max(acc[0][mx][n][i], acc[1][mx][n][i]);

        const MatO o = MatO(pooled);
        o.Store<CoopMatMatrixLayout.RowMajor>(cmIn, uint(sg * 32), 2);
        subgroupSharedBarrier();

        // Horizontal max and store
        for (uint i = lane; i < 16u; i += numLanes)
        {
          const int px = int(i >> 1);
          const int part = int(i & 1);
          const int xo = ((x0 + mx * 16) >> 1) + px;
          if (yo < (p.H >> 1) && xo < Wo)
          {
            const uint4 v = max8(cmIn[sg * 32 + (px * 2) * 2 + part], cmIn[sg * 32 + (px * 2 + 1) * 2 + part]);
            *(uint4*)getDstAddress(p, yo, xo, Wo, nb * 16 + part * 8) = v;
          }
        }

        subgroupSharedBarrier();
      }
    }
    else
    {
      [unroll]
      for (int m = 0; m < 2; ++m)
      {
        const int y = y0 + sg * 2 + m;
        if (y < p.H)
        {
          [unroll]
          for (int mx = 0; mx < MX; ++mx)
            storeFragment(p, MatO(acc[m][mx][n]), sg, lane, numLanes, y, x0 + mx * 16, nb);
        }
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------
// 3x3 convolution of a 2x upsampled source with cooperative matrices (sub-pixel decomposition)
// -------------------------------------------------------------------------------------------------
//
// A 3x3 convolution of a 2x nearest-upsampled tensor U(y, x) = S(y / 2, x / 2) is equivalent to
// four 2x2 convolutions of S, one per output phase (py, px) = (y % 2, x % 2), with combined
// weights: along each axis, phase 0 uses the source offsets {-1, 0} with the weights {w0, w1 + w2}
// and phase 1 the offsets {0, +1} with {w0 + w1, w2}. This reduces the multiply-adds for the
// upsampled source from 9 to 4 taps per output pixel. The zero padding is also equivalent, since
// U is outside the image exactly when S is.
//
// The output tile is 8 rows x 32 pixels. Each matrix fragment contains the 16 pixels of one output
// row with the same horizontal phase (x = x0 + px + 2 * j), so all its rows use the same weights.
// Each subgroup computes one phase (py, px) = (sg / 2, sg % 2): the 4 output rows y0 + py + 2 * i,
// so every weight tile loaded is used by 4 fragments. The second source (concatenated, full
// resolution) is convolved as usual, with stride-2 pixel rows.
//
// Packed weights: first source chunks [kc][phase = py * 2 + px][tap = (dy + 1 - py) * 2 + (dx + 1 - px)]
// [block][16][16], then second source chunks [kc][tap][block][16][16] (see the host code).

static const int UP_TW = 32;
static const int UP_IN0_W = UP_TW / 2 + 2; // first source tile (half resolution) with halo
static const int UP_IN0_H = CM_TH / 2 + 2;
static const int UP_IN1_W = UP_TW + 2;     // second source tile with halo
static const int UP_IN1_H = CM_TH + 2;
static const int UP_LOADS0 = UP_IN0_H * UP_IN0_W * 2;
static const int UP_LOADS1 = UP_IN1_H * UP_IN1_W * 2;
static const int UP_TILE_SIZE = UP_IN1_H * UP_IN1_W * KSTR; // in uint4 units (largest tile)
static const int UP_MAX_LOADS = (UP_LOADS1 + CM_NUM_SG * 32 - 1) / (CM_NUM_SG * 32);

groupshared uint4 upIn[2 * UP_TILE_SIZE];

// Loads the i-th uint4 of the input tile of the specified chunk (zero outside the image)
uint4 loadUpInput(ConvParams p, int kc, int i, int y0, int x0)
{
  const int numChunks0 = p.C0 / 16;
  const int pix = i >> 1;
  const int part = i & 1;
  if (kc < numChunks0)
  {
    // First source: half resolution tile
    if (i >= UP_LOADS0)
      return uint4(0);
    const int hy = pix / UP_IN0_W;
    const int hx = pix - hy * UP_IN0_W;
    const int sy = (y0 >> 1) - 1 + hy;
    const int sx = (x0 >> 1) - 1 + hx;
    const int H0 = p.H >> 1;
    const int W0 = p.W >> 1;
    if (sy < 0 || sy >= H0 || sx < 0 || sx >= W0)
      return uint4(0);
    return *(uint4*)(p.src0 + (uint64_t(sy * W0 + sx) * uint64_t(p.C0) + uint64_t(kc * 16 + part * 8)) * 2ull);
  }
  else
  {
    // Second source: full resolution tile
    if (i >= UP_LOADS1)
      return uint4(0);
    const int ty = pix / UP_IN1_W;
    const int tx = pix - ty * UP_IN1_W;
    const int y = y0 - 1 + ty;
    const int x = x0 - 1 + tx;
    if (y < 0 || y >= p.H || x < 0 || x >= p.W)
      return uint4(0);
    return *(uint4*)(p.src1 + (uint64_t(y * p.W + x) * uint64_t(p.C1) + uint64_t((kc - numChunks0) * 16 + part * 8)) * 2ull);
  }
}

// Returns the byte offset of a weight block of the first source (phase, 2x2 tap)
uint64_t getUpWeightOffset0(ConvParams p, int kc, int phase, int tap, int nb)
{
  return uint64_t(((kc * 4 + phase) * 4 + tap) * p.NB + nb) * 512ull;
}

// Returns the byte offset of a weight block of the second source (3x3 tap)
uint64_t getUpWeightOffset1(ConvParams p, int kc1, int tap, int nb)
{
  const int base = (p.C0 / 16) * 16 * p.NB;
  return uint64_t(base + (kc1 * 9 + tap) * p.NB + nb) * 512ull;
}

// Stores an output fragment containing the pixels x0 + 2 * j of a row
void storeFragmentStride2(ConvParams p, MatO o, int sg, uint lane, uint numLanes, int y, int x0, int nb)
{
  if (x0 + 30 < p.W)
  {
    // Store directly to the destination (every other pixel)
    o.Store<CoopMatMatrixLayout.RowMajor>((half*)getDstAddress(p, y, x0, p.W, nb * 16), 0, uint(p.NB * 32));
  }
  else
  {
    // Partial fragment: store through shared memory (the staging area of the subgroup)
    o.Store<CoopMatMatrixLayout.RowMajor>(upIn, uint(sg * 32), 2);
    subgroupSharedBarrier();

    for (uint i = lane; i < 32u; i += numLanes)
    {
      const int px = int(i >> 1);
      const int part = int(i & 1);
      const int x = x0 + px * 2;
      if (x < p.W)
        *(uint4*)getDstAddress(p, y, x, p.W, nb * 16 + part * 8) = upIn[sg * 32 + int(i)];
    }

    subgroupSharedBarrier();
  }
}

[shader("compute")]
[numthreads(WG_SIZE, 1, 1)]
void convCoopMatUp(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex, uniform ConvParams p)
{
  const int sg = int(getSubgroupID());
  const uint lane = WaveGetLaneIndex();
  const uint numLanes = WaveGetLaneCount();
  const int x0 = int(groupID.x) * UP_TW;
  const int y0 = int(groupID.y) * CM_TH;
  const int nb0 = int(groupID.z) * N_FRAGS;
  const int numChunks0 = p.C0 / 16;
  const int numChunks = (p.C0 + p.C1) / 16;
  const int py = sg >> 1; // output phase of the subgroup
  const int px = sg & 1;

  // Start loading the first input chunk
  uint4 prefetch[UP_MAX_LOADS];
  [unroll]
  for (int j = 0; j < UP_MAX_LOADS; ++j)
    prefetch[j] = loadUpInput(p, 0, int(localIndex) + j * int(WG_SIZE), y0, x0);

  // Initialize the accumulators with the bias: [output row of the phase][output block]
  MatC acc[4][N_FRAGS];
  [unroll]
  for (int n = 0; n < N_FRAGS; ++n)
  {
    MatC b = MatC(AccT(0.));
    if (nb0 + n < p.NB)
    {
    #if defined(ACC_F16)
      b = MatC.Load<CoopMatMatrixLayout.RowMajor>((half*)(p.bias + uint64_t(nb0 + n) * 512ull), 0, 16);
    #else
      b = MatC.Load<CoopMatMatrixLayout.RowMajor>((float*)(p.bias + uint64_t(nb0 + n) * 1024ull), 0, 16);
    #endif
    }
    [unroll]
    for (int i = 0; i < 4; ++i)
      acc[i][n] = b;
  }

  [unroll]
  for (int j = 0; j < UP_MAX_LOADS; ++j)
  {
    const int i = int(localIndex) + j * int(WG_SIZE);
    if (i < UP_LOADS1)
      upIn[(i >> 1) * KSTR + (i & 1)] = prefetch[j];
  }
  GroupMemoryBarrierWithGroupSync();

  for (int kc = 0; kc < numChunks; ++kc)
  {
    const int base = (kc & 1) * UP_TILE_SIZE;

    // Prefetch the next chunk
    if (kc + 1 < numChunks)
    {
      [unroll]
      for (int j = 0; j < UP_MAX_LOADS; ++j)
        prefetch[j] = loadUpInput(p, kc + 1, int(localIndex) + j * int(WG_SIZE), y0, x0);
    }

    if (kc < numChunks0)
    {
      // First source (upsampled): 2x2 convolution of the half resolution tile for the phase
      [unroll]
      for (int tap = 0; tap < 4; ++tap)
      {
        const int dy = py - 1 + (tap >> 1); // {-1, 0} for phase 0, {0, +1} for phase 1
        const int dx = px - 1 + (tap & 1);

        MatB b[N_FRAGS];
        [unroll]
        for (int n = 0; n < N_FRAGS; ++n)
        {
          if (nb0 + n < p.NB)
            b[n] = MatB.Load<CoopMatMatrixLayout.RowMajor>(
              (half*)(p.weight + getUpWeightOffset0(p, kc, py * 2 + px, tap, nb0 + n)), 0, 16);
        }

        [unroll]
        for (int i = 0; i < 4; ++i)
        {
          // Output row y0 + py + 2 * i maps to the half resolution row i of the tile
          const MatA a = MatA.Load<CoopMatMatrixLayout.RowMajor>(
            upIn, uint(base + ((i + 1 + dy) * UP_IN0_W + 1 + dx) * KSTR), KSTR);

          [unroll]
          for (int n = 0; n < N_FRAGS; ++n)
          {
            if (nb0 + n < p.NB)
              acc[i][n] = coopMatMulAdd<AccT, false>(a, b[n], acc[i][n]);
          }
        }
      }
    }
    else
    {
      // Second source (full resolution): 3x3 convolution with stride-2 pixel rows
      const int kc1 = kc - numChunks0;
      [unroll]
      for (int tap = 0; tap < 9; ++tap)
      {
        const int ky = tap / 3;
        const int kx = tap - ky * 3;

        MatB b[N_FRAGS];
        [unroll]
        for (int n = 0; n < N_FRAGS; ++n)
        {
          if (nb0 + n < p.NB)
            b[n] = MatB.Load<CoopMatMatrixLayout.RowMajor>(
              (half*)(p.weight + getUpWeightOffset1(p, kc1, tap, nb0 + n)), 0, 16);
        }

        [unroll]
        for (int i = 0; i < 4; ++i)
        {
          const int row = py + 2 * i;
          const MatA a = MatA.Load<CoopMatMatrixLayout.RowMajor>(
            upIn, uint(base + ((row + ky) * UP_IN1_W + px + kx) * KSTR), 2 * KSTR);

          [unroll]
          for (int n = 0; n < N_FRAGS; ++n)
          {
            if (nb0 + n < p.NB)
              acc[i][n] = coopMatMulAdd<AccT, false>(a, b[n], acc[i][n]);
          }
        }
      }
    }

    // Store the next chunk into the other buffer, which is no longer being read
    if (kc + 1 < numChunks)
    {
      const int nextBase = ((kc + 1) & 1) * UP_TILE_SIZE;
      [unroll]
      for (int j = 0; j < UP_MAX_LOADS; ++j)
      {
        const int i = int(localIndex) + j * int(WG_SIZE);
        if (i < UP_LOADS1)
          upIn[nextBase + (i >> 1) * KSTR + (i & 1)] = prefetch[j];
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }

  // Epilogue (the input tile buffers are reused as staging areas)
  [unroll]
  for (int n = 0; n < N_FRAGS; ++n)
  {
    const int nb = nb0 + n;
    if (nb >= p.NB)
      continue;

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
      const int y = y0 + py + 2 * i;
      if (RELU)
      {
        for (int e = 0; e < acc[i][n].GetLength(); ++e)
          acc[i][n][e] = max(acc[i][n][e], AccT(0.));
      }
      if (y < p.H)
        storeFragmentStride2(p, MatO(acc[i][n]), sg, lane, numLanes, y, x0 + px, nb);
    }
  }
}

// -------------------------------------------------------------------------------------------------
// 3x3 convolution without cooperative matrices (fallback)
// -------------------------------------------------------------------------------------------------
//
// The workgroup computes an output tile of 8 x 16 pixels and 16 output channels; each invocation
// computes the 16 output channels of one pixel.

static const int CS_TH = 8;
static const int CS_TW = 16;
static const int CS_IN_W = CS_TW + 2;
static const int CS_IN_H = CS_TH + 2;
static const int CS_GROUP_SIZE = CS_TH * CS_TW;

groupshared uint4 csIn[CS_IN_H * CS_IN_W * KSTR];
groupshared uint4 csW[9 * 16 * 2];          // weights of the current chunk: [tap][16 input][16 output]
groupshared uint4 csOut[CS_TH * CS_TW * 2]; // output tile for pooling

[shader("compute")]
[numthreads(CS_GROUP_SIZE, 1, 1)]
void convSimple(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex, uniform ConvParams p)
{
  const int tid = int(localIndex);
  const int ty = tid / CS_TW;
  const int tx = tid - ty * CS_TW;
  const int x0 = int(groupID.x) * CS_TW;
  const int y0 = int(groupID.y) * CS_TH;
  const int nb = int(groupID.z);

  float acc[16];
  [unroll]
  for (int i = 0; i < 16; ++i)
    acc[i] = ((float*)p.bias)[nb * 256 + i]; // first row of the replicated bias

  const int numChunks = (p.C0 + p.C1) / 16;
  for (int kc = 0; kc < numChunks; ++kc)
  {
    GroupMemoryBarrierWithGroupSync();
    for (int i = tid; i < CS_IN_H * CS_IN_W * 2; i += CS_GROUP_SIZE)
    {
      const int pix = i >> 1;
      const int iy = pix / CS_IN_W;
      const int ix = pix - iy * CS_IN_W;
      csIn[pix * KSTR + (i & 1)] = loadConvInput(p, kc, y0 + iy - 1, x0 + ix - 1, i & 1);
    }
    for (int i = tid; i < 9 * 32; i += CS_GROUP_SIZE)
    {
      const int tap = i / 32;
      csW[i] = ((uint4*)(p.weight + getWeightOffset(p, kc, tap, nb)))[i - tap * 32];
    }
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (int tap = 0; tap < 9; ++tap)
    {
      const int ky = tap / 3;
      const int kx = tap - ky * 3;
      const int pix = (ty + ky) * CS_IN_W + (tx + kx);

      const uint4 in0 = csIn[pix * KSTR];
      const uint4 in1 = csIn[pix * KSTR + 1];
      float x[16];
      [unroll]
      for (int i = 0; i < 4; ++i)
      {
        const float2 a = unpackHalf2(in0[i]);
        const float2 b = unpackHalf2(in1[i]);
        x[i*2] = a.x; x[i*2+1] = a.y;
        x[8+i*2] = b.x; x[8+i*2+1] = b.y;
      }

      [unroll]
      for (int ic = 0; ic < 16; ++ic)
      {
        const uint4 w0 = csW[(tap * 16 + ic) * 2];
        const uint4 w1 = csW[(tap * 16 + ic) * 2 + 1];
        [unroll]
        for (int i = 0; i < 4; ++i)
        {
          const float2 a = unpackHalf2(w0[i]);
          const float2 b = unpackHalf2(w1[i]);
          acc[i*2]     += x[ic] * a.x;
          acc[i*2+1]   += x[ic] * a.y;
          acc[8+i*2]   += x[ic] * b.x;
          acc[8+i*2+1] += x[ic] * b.y;
        }
      }
    }
  }

  if (RELU)
  {
    [unroll]
    for (int i = 0; i < 16; ++i)
      acc[i] = max(acc[i], 0.);
  }

  uint4 out0, out1;
  [unroll]
  for (int i = 0; i < 4; ++i)
  {
    out0[i] = reinterpret<uint>(half2(toHalf(acc[i*2]),   toHalf(acc[i*2+1])));
    out1[i] = reinterpret<uint>(half2(toHalf(acc[8+i*2]), toHalf(acc[8+i*2+1])));
  }

  const int y = y0 + ty;
  const int x = x0 + tx;

  if (POOL)
  {
    csOut[tid * 2]     = out0;
    csOut[tid * 2 + 1] = out1;
    GroupMemoryBarrierWithGroupSync();

    // 2x2 max pooling: the first (TH/2)*(TW/2)*2 invocations each compute 8 channels
    const int Wo = p.W >> 1;
    const int Ho = p.H >> 1;
    if (tid < (CS_TH / 2) * (CS_TW / 2) * 2)
    {
      const int part = tid & 1;
      const int po = tid >> 1;
      const int pyo = po / (CS_TW / 2);
      const int pxo = po - pyo * (CS_TW / 2);
      const int i00 = ((pyo * 2) * CS_TW + pxo * 2) * 2 + part;
      const int i10 = ((pyo * 2 + 1) * CS_TW + pxo * 2) * 2 + part;
      const uint4 v = max8(max8(csOut[i00], csOut[i00 + 2]), max8(csOut[i10], csOut[i10 + 2]));
      const int yo = (y0 >> 1) + pyo;
      const int xo = (x0 >> 1) + pxo;
      if (yo < Ho && xo < Wo)
        *(uint4*)getDstAddress(p, yo, xo, Wo, nb * 16 + part * 8) = v;
    }
  }
  else if (y < p.H && x < p.W)
  {
    uint4* dst = (uint4*)getDstAddress(p, y, x, p.W, nb * 16);
    dst[0] = out0;
    dst[1] = out1;
  }
}
