# vk_oidn architecture

This document explains how vk_oidn works: how an application integrates it, what happens on the
GPU when an image is denoised, and how the network and its kernels are organized. Measured
results are in [validation_and_performance.md](validation_and_performance.md).

vk_oidn runs the `RT` filter of [Intel® Open Image Denoise](https://github.com/RenderKit/oidn)
(OIDN): a U-Net convolutional network trained to remove Monte Carlo noise from rendered images,
optionally guided by albedo and normal images. The network weights are OIDN's; the GPU kernels
and the runtime are vk_oidn's (derived from the Vulkan backend of the
[OIDN fork](https://github.com/mklefrancois/oidn-vulkan)).

## 1. Integration in a Vulkan application

The library never creates its own device, queue or command buffers. Like DLSS, it records work
into command buffers owned by the application, which submits them whenever and wherever it wants.

```mermaid
sequenceDiagram
    autonumber
    participant App as Application
    participant SDK as vk_oidn
    participant CB as Command buffer
    participant GPU as Vulkan queue

    Note over App,SDK: Device creation
    App->>SDK: getRequiredDeviceExtensions()
    App->>SDK: enableRequiredFeatures()
    App->>App: vkCreateDevice()

    Note over App,SDK: Initialization (once per device)
    App->>SDK: Context::create(device, vkGetDeviceProcAddr, enabled extensions)
    SDK->>SDK: detect cooperative matrices, float controls, push descriptors
    SDK->>SDK: create the pipeline layouts

    Note over App,GPU: Per resolution / configuration
    App->>SDK: Denoiser::create(context, cmd, width, height, inputs, quality)
    SDK->>SDK: select the weights, plan memory and tiles, create the pipelines
    SDK->>CB: record the weight upload (staging copy)
    App->>GPU: vkQueueSubmit(cmd)

    Note over App,GPU: Every frame
    App->>CB: render the noisy color, albedo, normal
    App->>CB: barrier (writes visible to compute shader reads)
    App->>SDK: cmdDenoise(cmd, inputs)
    SDK->>CB: record the dispatches (autoexposure, tiles, convolutions)
    App->>CB: barrier (compute shader writes visible to the next use)
    App->>CB: display / post-process the output
    App->>GPU: vkQueueSubmit(cmd)
```

All memory is accessed by device address (`VK_KHR_buffer_device_address`), passed in push
constants, so the kernels need no descriptor sets. Image views, when used, are bound with push
descriptors (`VK_KHR_push_descriptor`) right before the three kernels that touch them.

```mermaid
flowchart LR
    subgraph "Application resources"
        CB[(Color)]
        AB[(Albedo)]
        NB[(Normal)]
        OB[(Output)]
    end

    subgraph "vk_oidn resources (owned by the Denoiser)"
        W[(Weights<br/>FP16 16x16 tiles)]
        S[(Scratch<br/>intermediate tensors)]
        AE[(Autoexposure<br/>bins and result)]
    end

    CB -->|"device address<br/>or storage image"| K1[Input processing]
    AB --> K1
    NB --> K1
    CB --> K0[Autoexposure]
    K0 --> AE
    AE --> K1
    K1 --> S
    S <--> K2[Convolutions]
    W --> K2
    S --> K3[Output processing]
    AE --> K3
    K3 -->|"device address<br/>or storage image"| OB
```

## 2. The denoising pipeline (one `cmdDenoise` call)

Large images are split into overlapping tiles (by default at most 2160x2160 pixels each, like
OIDN; `maxMemoryMB` lowers this). Each tile runs the whole network, and only the central part of
each tile, without the overlap, is written to the output. Every dispatch is followed by a
compute-to-compute memory barrier.

```mermaid
flowchart TD
    Start([cmdDenoise]) --> Barrier[Barrier: previous executions done with the scratch memory]
    Barrier --> HDR{HDR and automatic<br/>input scale?}
    HDR -->|yes| AE1[Autoexposure downsample<br/>16x16 luminance bins]
    AE1 --> AE2[Autoexposure reduce<br/>log2 sums per workgroup]
    AE2 --> AE3[Autoexposure final<br/>scale = 0.18 / exp2 mean]
    AE3 --> Tiles
    HDR -->|no| Tiles

    subgraph Tiles ["For each tile"]
        direction TB
        IP["Input processing<br/>scale, sanitize, transfer function (PU / sRGB)<br/>color + albedo + normal → FP16 HWC tensor, 16 channels"]
        UNET["U-Net: 16 or 19 fused 3x3 convolutions<br/>(see below)"]
        OP["Output processing<br/>inverse transfer function, 1 / scale<br/>central part of the tile → output image"]
        IP --> UNET --> OP
    end

    Tiles --> Done([Commands recorded])
```

The **transfer function** maps the input to the range the network was trained on: the
perceptual-uniform (PU) curve for HDR, sRGB for linear LDR, nothing for sRGB-encoded LDR. The
output processing applies the inverse.

## 3. The U-Net

The base network (16 layers) for the `hdr + albedo + normal` model is below. Each box is a single
kernel dispatch: pooling, upsampling and concatenation are fused into the convolutions, so no
intermediate pooled/upsampled/concatenated tensors are ever written. The numbers are the output
channels; resolutions are relative to the tile.

```mermaid
flowchart TD
    IN["input<br/>9 → 16 ch, 1/1"]

    subgraph Encoder
        E0["enc_conv0<br/>32 ch, 1/1"]
        E1["enc_conv1 + pool<br/>32 ch, 1/2"]
        E2["enc_conv2 + pool<br/>48 ch, 1/4"]
        E3["enc_conv3 + pool<br/>64 ch, 1/8"]
        E4["enc_conv4 + pool<br/>80 ch, 1/16"]
        E5A["enc_conv5a<br/>96 ch, 1/16"]
        E5B["enc_conv5b<br/>96 ch, 1/16"]
    end

    subgraph Decoder
        D4A["upsample + concat + dec_conv4a<br/>112 ch, 1/8"]
        D4B["dec_conv4b<br/>112 ch, 1/8"]
        D3A["upsample + concat + dec_conv3a<br/>96 ch, 1/4"]
        D3B["dec_conv3b<br/>96 ch, 1/4"]
        D2A["upsample + concat + dec_conv2a<br/>64 ch, 1/2"]
        D2B["dec_conv2b<br/>64 ch, 1/2"]
        D1A["upsample + concat + dec_conv1a<br/>64 ch, 1/1"]
        D1B["dec_conv1b<br/>32 ch, 1/1"]
        D0["dec_conv0<br/>3 → 16 ch, 1/1"]
    end

    OUT["output"]

    IN --> E0 --> E1 --> E2 --> E3 --> E4 --> E5A --> E5B
    E5B --> D4A --> D4B --> D3A --> D3B --> D2A --> D2B --> D1A --> D1B --> D0 --> OUT
    E3 -. skip .-> D4A
    E2 -. skip .-> D3A
    E1 -. skip .-> D2A
    IN -. skip .-> D1A
```

All convolutions are 3x3 with bias and ReLU. The large network (`rt_hdr_calb_cnrm_large`, used
in high quality with clean auxiliary images) has two convolutions per encoder level and wider
layers (64 to 256 channels). The graph is not hard-coded in the runtime: each packed weight file
contains its layer table (sources, fused operations, channel counts), and the runtime executes it
in order.

The intermediate tensors (FP16, channels-last, padded to multiples of 16 channels) are placed in
one scratch buffer. The planner assigns offsets from the tensor lifetimes, largest first, so
tensors that are not alive at the same time share memory (1920x1088 tile: 384 MB for the base
network).

## 4. The convolution kernel

The 3x3 convolutions are computed as implicit GEMMs on tensor cores, with
`VK_KHR_cooperative_matrix` 16x16x16 FP16 matrices: M = output pixels, N = output channels,
K = 3x3 taps x input channels. Accumulation is FP32 in high quality and FP16 in the balanced and
fast modes, like OIDN's CUDA backend.

```mermaid
flowchart TD
    subgraph WG ["Workgroup: 4 subgroups, tile of 8 rows x 32 (or 48) pixels x 32 (or 16) output channels"]
        direction TB
        P0["Prefetch input chunk 0 into registers<br/>(16 channels, tile + 1-pixel halo, zero outside the image,<br/>nearest upsampling of the first source, concatenated second source)"]
        B0["Load the bias into the accumulators"]
        S0["Store chunk 0 into shared memory buffer 0, barrier"]
        P0 --> B0 --> S0 --> LOOP

        subgraph LOOP ["For each chunk of 16 input channels"]
            direction TB
            PF["Prefetch the next chunk into registers"]
            TAPS["For each of the 9 taps:<br/>load weight tiles (global, 16x16)<br/>load pixel tiles from shared memory (16x16)<br/>MMA into 2 rows x MX columns x NF channel blocks"]
            ST["Store the next chunk into the other shared memory buffer<br/>one barrier per chunk"]
            PF --> TAPS --> ST
        end

        LOOP --> EPI["Epilogue: ReLU, FP32 → FP16<br/>pooling: vertical max in registers, horizontal max through shared memory<br/>store to the destination tensor"]
    end
```

Design points:

- **Double buffering**: the global loads of the next input chunk overlap the matrix
  multiplications of the current one, with a single barrier per chunk.
- **Pooling fusion**: each subgroup owns two consecutive output rows, which are exactly the rows
  of the 2x2 pooling windows, so pooling needs no exchange between subgroups.
- **Upsampling and concatenation fusion**: the concatenation is done while loading the input
  tile into shared memory (the concatenated tensor never exists in memory). For the upsampled
  source, a dedicated kernel (`convCoopMatUp`) uses a **sub-pixel decomposition**: a 3x3
  convolution of a 2x nearest-upsampled tensor equals four 2x2 convolutions of the
  low-resolution tensor (one per output phase) with combined weights, which needs 56% fewer
  multiply-adds. Each subgroup computes one phase, so each combined weight tile is reused by
  4 matrix fragments (section below).
- **Per-layer tile selection**: 32x8 pixels x 32 channels by default, 48 pixels wide for layers
  with more than 64 input channels (more reuse of the weights), and 16 channels per workgroup for
  layers with an odd number of 16-channel blocks or too few workgroups to fill the GPU.
- **Fallback**: GPUs without cooperative matrices (and non-NVIDIA GPUs by default) use a plain
  FMA kernel with the same tiling of the input.

### Sub-pixel decomposition of the upsampling layers

```mermaid
flowchart LR
    subgraph Before ["Fused upsampling (generic kernel)"]
        direction TB
        S1["low-res source S<br/>H/2 x W/2"] -->|"nearest 2x<br/>(on load)"| U1["U, H x W"]
        U1 -->|"3x3 convolution<br/>9 taps per pixel"| O1["output H x W"]
    end

    subgraph After ["Sub-pixel decomposition (convCoopMatUp)"]
        direction TB
        S2["low-res source S<br/>H/2 x W/2"] --> P00["phase (0,0)<br/>2x2 taps, combined weights"]
        S2 --> P01["phase (0,1)<br/>2x2 taps"]
        S2 --> P10["phase (1,0)<br/>2x2 taps"]
        S2 --> P11["phase (1,1)<br/>2x2 taps"]
        P00 --> O2["output H x W<br/>(interleaved phases)"]
        P01 --> O2
        P10 --> O2
        P11 --> O2
    end
```

Along each axis, output position 2m + p reads the upsampled positions 2m + p - 1, 2m + p, 2m + p + 1,
which are the source pixels {m - 1, m, m} for p = 0 and {m, m, m + 1} for p = 1. The weights of
the taps reading the same source pixel are summed (on the host, when the denoiser is created):
p = 0 uses {w0, w1 + w2} at offsets {-1, 0}, p = 1 uses {w0 + w1, w2} at offsets {0, +1}. The
skip connection (second source, full resolution) is convolved normally. Disable with
`VK_OIDN_SUBPIXEL=0`.

## 5. Build pipeline

```mermaid
flowchart LR
    subgraph Weights
        TZA["OIDN weights<br/>*.tza"] -->|tools/pack_weights.py| VKW["packed weights<br/>weights/*.vkoidnw<br/>layer graph + GPU tiles"]
        VKW -->|tools/embed_files.py| WCPP[weights_embedded.cpp]
    end

    subgraph Kernels
        SL["shaders/vk_oidn.slang"] -->|"slangc -O1<br/>per entry point and variant"| SPV1[SPIR-V]
        SPV1 -->|"tools/spirv_float_controls.py<br/>FP16 round-to-nearest-even"| SPV2[SPIR-V]
        SPV2 -->|tools/embed_files.py| KCPP[kernels_embedded.cpp]
    end

    WCPP --> LIB["vk_oidn.lib"]
    KCPP --> LIB
    SRC["src/*.cpp"] --> LIB
```

The Slang file is compiled into 20 kernels: input processing, output processing and the
autoexposure downsampling (each with buffer and storage image I/O), the two autoexposure
reductions, image copy, the fallback convolution, 6 cooperative matrix convolution variants
(3 tile shapes x FP32/FP16 accumulation) and 4 sub-pixel upsampling convolution variants
(2 tile shapes x FP32/FP16 accumulation). Each is compiled with and without FP16 float controls;
the variant without them is used on devices which do not support round-to-nearest-even and
denormal preservation for FP16. The experimental `VK_NV_cooperative_matrix2` convolution
(`shaders/experimental/conv_cm2.comp`) is written in GLSL and compiled with glslang.
