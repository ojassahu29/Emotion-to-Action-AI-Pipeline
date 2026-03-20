# Edge Deployment Plan

## Overview

Plan for deploying the Emotion-to-Action pipeline on edge devices (mobile phones, tablets, wearables).

---

## Model Specifications

| Component | Size (approx.) | Format |
|-----------|----------------|--------|
| Emotion LinearSVC | ~1 MB | Pickle / ONNX |
| Intensity GradientBoosting | ~3–5 MB | Pickle / ONNX |
| TF-IDF Vectorizers (Word+Char) | ~5 MB | Pickle |
| Decision Engine (rules) | <1 KB | Code logic |
| **Total** | **~9–11 MB** | — |

## Latency Estimates

| Stage | Desktop | Mobile (ARM) |
|-------|---------|--------------|
| Text cleaning | <1 ms | ~2 ms |
| TF-IDF transform | ~5 ms | ~15 ms |
| Feature encoding | ~2 ms | ~5 ms |
| Emotion prediction | ~10 ms | ~30 ms |
| Intensity prediction | ~10 ms | ~30 ms |
| Uncertainty + Decision | <1 ms | ~2 ms |
| **Total** | **~30 ms** | **~85 ms** |

Both estimates are well within real-time requirements (<100ms for mobile).

## Mobile Feasibility

### ✅ Feasible
- **Small model size**: 9–11 MB is extremely lightweight for standard apps
- **Fast inference**: <100ms end-to-end on target architectures
- **No GPU required**: LinearSVC and GradientBoosting are highly CPU efficient
- **Offline capable**: no external API dependencies
- **Low memory footprint**: strictly scalar/vector operations

### Deployment Options

1. **ONNX Runtime Mobile**
   - Convert sklearn models to ONNX format
   - Run via ONNX Runtime for iOS/Android
   - Best cross-platform performance

2. **CoreML / TensorFlow Lite**
   - Convert via sklearn-to-coreml or custom export
   - Native mobile framework integration

3. **Embedded Python (Kivy/BeeWare)**
   - Run sklearn directly in embedded Python
   - Simpler but larger app bundle

### Recommended: ONNX Runtime

- Convert models: `skl2onnx` package
- Deploy: ONNX Runtime Mobile SDK
- Supported: iOS 12+, Android 8+
- Size overhead: ~5 MB for ONNX Runtime

## Optimization Strategies

1. **Model Condensation**: Tune hyperparameters for smaller trees in the GradientBoosting config (e.g. fewer estimators).
2. **Prune features**: Cap TF-IDF char n-grams back to ~500 if space is strictly bounded.
3. **Quantization**: 32-bit → 16-bit float mappings via ONNX.
4. **Caching**: Maintain TF-IDF memory instances continuously.

## Monitoring & Updates

- **On-device logging**: track confidence scores, uncertain_flag rates
- **Model updates**: bundle new models via app updates or OTA
- **A/B testing**: compare model versions using confidence distributions
- **Privacy**: all processing on-device, no data leaves the phone

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model drift | Accuracy degrades over time | Periodic retraining with new data |
| Device fragmentation | Performance varies | Test on min-spec devices |
| Battery drain | Continuous inference | Batch processing, infer only on journal save |
| Storage constraints | Limited model size | ONNX quantization, feature pruning |
