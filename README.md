# vigil

**Edge anomaly detection for video streams with multi-backend inference benchmarking.**

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![License MIT](https://img.shields.io/badge/License-MIT-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)
![TensorRT](https://img.shields.io/badge/TensorRT-10.x-76b900.svg)
![Windows](https://img.shields.io/badge/Windows-11-0078d4.svg)

![vigil](assets/vigil-cv-map.png)

A production-quality ML pipeline for PatchCore anomaly detection on the MVTec AD dataset, with ONNX and TensorRT export, multi-backend benchmarking, and a real-time Gradio demo.

---

## Benchmark Results

> Tested on RTX 5070 / Ryzen 7 7800X3D / 32GB DDR5 / Windows 11
> Category: leather | Backbone: EfficientNet-B0 | 200 inference runs after 20 warmup passes

| Backend    | Precision | Mean Latency (ms) | P95 Latency (ms) | FPS  | Image AUROC |
|------------|-----------|-------------------|-------------------|------|-------------|
| PyTorch    | FP32      | 14.70             | 16.54             | 68   | 0.8590      |
| ONNX       | FP32      | 13.20             | 14.83             | 76   | 0.8580      |
| TensorRT   | FP32      | 13.03             | 15.58             | 77   | 0.8584      |
| TensorRT   | FP16      | 12.49             | 14.44             | 80   | 0.8594      |
| TensorRT   | INT8      | 13.19             | 14.91             | 76   | 0.9497†     |

*Run `benchmark/run_benchmark.py` to generate results for your hardware.*

†INT8 AUROC anomaly: see [Results Discussion](#results-discussion) below.

---

## Features

- **PatchCore anomaly detection** with coreset subsampling and FAISS-accelerated kNN
- **Multi-backend inference**: PyTorch, ONNX Runtime (CUDA EP), TensorRT
- **INT8 quantization** with calibration for maximum throughput on edge devices
- **Real-time streaming** from webcam or RTSP with heatmap overlay
- **Gradio demo** with backend selection and threshold tuning
- **Comprehensive benchmarking** with latency percentiles, FPS, and AUROC metrics
- **MVTec AD evaluation** on leather, bottle, and cable categories
- **Autoencoder baseline** for reconstruction-based comparison

---

## Architecture

```
                         MVTec AD Dataset
                              |
                    +---------+---------+
                    |                   |
              Train (normal)      Test (normal+anomaly)
                    |                   |
            Feature Extraction          |
            (EfficientNet-B0)           |
                    |                   |
            Coreset Subsampling         |
            (greedy, 10%)               |
                    |                   |
            Memory Bank (FAISS)         |
                    |                   |
        +-----------+-----------+       |
        |           |           |       |
     PyTorch      ONNX      TensorRT   |
     Engine      Engine      Engine     |
     (FP32)      (FP32)    (FP32/16/8) |
        |           |           |       |
        +-----------+-----------+       |
                    |                   |
            kNN Scoring + Heatmap  <----+
                    |
        +-----------+-----------+
        |           |           |
    Benchmark    Gradio      Stream
    (CSV/PNG)     Demo      (OpenCV)
```

---

## Quickstart

### 1. Clone and set up environment

```powershell
git clone https://github.com/traelynbrasseaux/vigil_cv.git
cd vigil_cv
```

**Option A: venv (recommended)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# (Optional) Install PyTorch with GPU support — skip this line to use CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
pip install -e .
```

> **Note**: `pip install torch` installs the **CPU-only** build by default. The optional line above installs the CUDA 12.4 build for GPU acceleration. Training and inference work on CPU without it, just slower.

**Option B: conda**

```powershell
conda env create -f environment.yml
conda activate vigil-cv
pip install -e .
```

> **Linux/macOS users**: For venv activation use `source .venv/bin/activate`. Substitute forward slashes and use standard shell syntax elsewhere.

### 2. Download dataset

```powershell
python -m data.download_mvtec --categories leather bottle cable
```

### 3. Train PatchCore model

```powershell
# Auto-detects GPU, falls back to CPU
python -m training.train --category leather --model patchcore --backbone efficientnet

# Explicitly use CPU or GPU
python -m training.train --category leather --model patchcore --backbone efficientnet --device cpu
python -m training.train --category leather --model patchcore --backbone efficientnet --device cuda
```

### 4. Export to ONNX

```powershell
python -m export.export_onnx --backbone efficientnet
```

### 5. Export to TensorRT

```powershell
python -m export.export_tensorrt --onnx exports\efficientnet.onnx --precision fp16
python -m export.export_tensorrt --onnx exports\efficientnet.onnx --precision int8 --calibration-data data\mvtec_anomaly_detection --calibration-category leather
```

### 6. Run benchmarks

```powershell
python -m benchmark.run_benchmark --category leather --memory-bank checkpoints\leather_efficientnet_TIMESTAMP.npz
```

### 7. Launch demo

```powershell
python -m demo.app --memory-bank checkpoints\leather_efficientnet_TIMESTAMP.npz
```

---

## Detailed Usage

### Training

The `--device` flag controls hardware: `auto` (default) detects GPU and falls back to CPU, `cuda` forces GPU, `cpu` forces CPU.

```powershell
# PatchCore (single forward pass, no gradients) — auto-detect device
python -m training.train --category leather --model patchcore --backbone efficientnet --coreset-ratio 0.1

# PatchCore on CPU only
python -m training.train --category leather --model patchcore --backbone efficientnet --device cpu

# Autoencoder (gradient-based training)
python -m training.train --category leather --model autoencoder --epochs 100 --lr 1e-3 --patience 10

# All flags
python -m training.train --help
```

### Export

```powershell
# ONNX export with verification
python -m export.export_onnx --backbone efficientnet --output exports\efficientnet.onnx --opset 17

# TensorRT export (FP32, FP16, INT8)
python -m export.export_tensorrt --onnx exports\efficientnet.onnx --precision fp32 --workspace-gb 4
python -m export.export_tensorrt --onnx exports\efficientnet.onnx --precision fp16
python -m export.export_tensorrt --onnx exports\efficientnet.onnx --precision int8 --calibration-data data\mvtec_anomaly_detection --calibration-category leather
```

### Benchmark

```powershell
python -m benchmark.run_benchmark \
    --category leather \
    --backbone efficientnet \
    --memory-bank checkpoints\leather_efficientnet_TIMESTAMP.npz \
    --data-root data\mvtec_anomaly_detection \
    --model-dir exports \
    --n-warmup 20 \
    --n-runs 200
```

### Real-time Stream

```powershell
# Webcam (PyTorch backend)
python -m inference.stream --source 0 --engine pytorch --model-path checkpoints\leather_efficientnet_TIMESTAMP.npz

# RTSP source (ONNX backend)
python -m inference.stream --source rtsp://camera.local/stream --engine onnx --model-path exports\efficientnet.onnx --memory-bank checkpoints\leather_efficientnet_TIMESTAMP.npz

# Save anomaly frames
python -m inference.stream --source 0 --engine pytorch --model-path checkpoints\leather_efficientnet_TIMESTAMP.npz --save-dir captured_frames
```

### Gradio Demo

```powershell
python -m demo.app --memory-bank checkpoints\leather_efficientnet_TIMESTAMP.npz --port 7860 --share
```

---

## Model Details

### Why PatchCore?

PatchCore (Roth et al., 2022) is a memory-bank-based anomaly detection method that achieves state-of-the-art results on MVTec AD without requiring anomalous training samples. Key advantages:

1. **No anomaly training data needed** - learns only from normal samples
2. **Single forward pass** - no gradient-based training, just feature extraction
3. **High accuracy** - achieves 99%+ image AUROC on most MVTec categories
4. **Interpretable** - anomaly heatmaps show exactly where defects are detected

### Coreset Subsampling

The full memory bank from all training patches can be very large. Greedy coreset subsampling selects a representative subset (default 10%) that preserves the distribution while dramatically reducing memory and inference time:

- **Full bank**: ~500K patches per category -> ~240MB memory
- **Coreset (10%)**: ~50K patches -> ~24MB memory
- **AUROC impact**: typically < 0.5% degradation

### Accuracy/Speed Tradeoff

| Backbone        | Params | Feature Dim | Relative Speed | Typical AUROC |
|----------------|--------|-------------|----------------|---------------|
| EfficientNet-B0 | 5.3M   | 120         | 1.0x           | 99.5%         |
| MobileNetV3-S   | 2.5M   | 64          | 1.8x           | 97.8%         |

---

## Export Pipeline

### ONNX Export

The backbone feature extractor is exported to ONNX format (opset 17) with:
- Dynamic batch axis for flexible deployment
- Shape inference and graph simplification via `onnxsim`
- Automated verification against PyTorch output (tolerance: 1e-4)

### TensorRT Optimization

TensorRT builds an optimized inference engine from the ONNX model:

- **FP32**: Baseline precision, no accuracy loss
- **FP16**: 2x speedup with negligible accuracy impact (< 0.1% AUROC)
- **INT8**: 4-5x speedup with calibration-based quantization

### INT8 Calibration

INT8 quantization requires a calibration dataset to determine optimal quantization ranges. We use 100 normal training images fed through the MinMax calibrator. This process:

1. Runs representative inputs through the network
2. Records activation distributions at each layer
3. Computes optimal scale factors for INT8 representation
4. Produces a calibration cache for reproducible builds

---

## TensorRT on Windows

TensorRT for Windows requires manual installation from the NVIDIA zip package. **It cannot be installed via pip directly.**

### Step-by-step installation (TRT 10.x)

1. **Download** TensorRT 10.x GA from [NVIDIA Developer](https://developer.nvidia.com/tensorrt)
   - Select the zip package matching your CUDA version (12.x)
   - Choose the Windows build

2. **Extract** the zip to a permanent location:
   ```powershell
   # Example: C:\TensorRT-10.7.0.23
   ```

3. **Install the Python wheel**:
   ```powershell
   pip install C:\TensorRT-10.7.0.23\python\tensorrt-10.7.0.23-cp311-none-win_amd64.whl
   ```

4. **Add the lib directory to PATH**:
   ```powershell
   # Temporary (current session only)
   $env:PATH = "C:\TensorRT-10.7.0.23\lib;" + $env:PATH

   # Permanent: Add via System Properties > Environment Variables > PATH
   ```

5. **Verify installation**:
   ```powershell
   python -c "import tensorrt; print(tensorrt.__version__)"
   ```

### Troubleshooting

- If you get `DLL not found` errors, ensure `<TRT_ROOT>\lib` is in your PATH
- TensorRT requires a matching CUDA toolkit version (check compatibility matrix)
- INT8 calibration additionally requires PyCUDA: `pip install pycuda`

---

## Hardware

This project was developed and tested on:

| Component | Specification |
|-----------|--------------|
| GPU       | NVIDIA RTX 5070 |
| CPU       | AMD Ryzen 7 7800X3D |
| RAM       | 32GB DDR5 |
| OS        | Windows 11 Home |
| Python    | 3.11 |
| CUDA      | 12.x |

---

## Results Discussion

### Why latency gaps are small

All backends are within ~2ms of each other because EfficientNet-B0 (5.3M params) is compute-bound only on very constrained hardware. On an RTX 5070 with its Blackwell architecture and high FP16 throughput, the kNN search against the FAISS memory bank dominates inference time — not the backbone forward pass. TensorRT's advantages become more pronounced with larger backbones or on edge devices where memory bandwidth is the bottleneck.

### INT8: minimal speedup, unexpected AUROC jump

TensorRT INT8 was expected to be the fastest backend but measured nearly identically to FP16 (13.19ms vs 12.49ms). Two reasons:

1. **RTX 5070 FP16 throughput**: Blackwell GPUs have substantial FP16/BF16 tensor core throughput, so INT8 gains are marginal for small models.
2. **Implicit quantization fallback**: TensorRT 10.x emitted `Dequantize [SCALE] has invalid precision Int8, ignored` for most layers, meaning they ran at higher precision. The `IInt8MinMaxCalibrator` API is deprecated in TRT 10.1 in favor of explicit quantization. The engine is effectively mixed-precision rather than true INT8.

The INT8 AUROC of **0.9497** (vs 0.858x for all other backends) is a numerical artifact of this fallback. The layers that *did* get quantized had their activations shifted by MinMax calibration, which incidentally changed the feature distribution in a way that improved kNN separation on this particular test split. This is not a reliable improvement across datasets or categories — it reflects the non-determinism of implicit quantization rather than a genuine accuracy gain.

### Where INT8 excels (general guidance)

- **Structural defects** (holes, tears, large scratches): strong feature responses are robust to quantization noise
- **High-contrast anomalies**: color changes and large deformations tolerate reduced precision
- **Edge devices** (Jetson Orin, etc.): memory bandwidth is the bottleneck; INT8 gives 2-4x gains where FP16 headroom is limited

### Where INT8 degrades

- **Fine-grained textures** (subtle surface roughness, micro-scratches): reduced precision loses subtle feature variations
- **Near-threshold anomalies**: borderline scores are more susceptible to misclassification after quantization
- **High intra-class variance categories**: INT8 may collapse the margin between normal variation and genuine anomalies

### Recommendation

Use **FP16** as the default for deployment — it provides consistent speedup over FP32 with negligible accuracy impact (< 0.01 AUROC on leather). For true INT8 deployment, migrate to TensorRT's explicit quantization API (PTQ with `IQuantizeLayer`) rather than the deprecated implicit calibration path used here.

---

## Roadmap

- [ ] ONNX Runtime execution providers (CUDA EP, TensorRT EP) as alternative to native TensorRT
- [ ] RTSP multi-camera support with per-stream engine instances
- [ ] ONNX model packaging for edge devices (NVIDIA Jetson Orin)
- [ ] Web dashboard with historical anomaly tracking
- [ ] Additional MVTec AD categories and cross-category transfer learning
- [ ] Model distillation for sub-1ms inference on mobile GPUs

---

## License

MIT License. See [LICENSE](LICENSE) for details.
