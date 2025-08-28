# Multi-Vendor ML Inference Benchmark - Usage Examples

This document provides comprehensive examples of how to use the multi-vendor benchmark system for fair performance comparison across different ML inference frameworks.

## Quick Start Examples

### 1. Basic Multi-Vendor Comparison

Compare all available frameworks on standard vision models:

```bash
# Run complete multi-vendor benchmark
python multi_vendor_benchmark.py --generate-report

# Output: Comprehensive HTML report with performance rankings
```

### 2. CPU-Only Framework Comparison

Compare CPU-optimized frameworks for deployment scenarios without GPU:

```bash
# CPU frameworks only
python multi_vendor_benchmark.py --device cpu --generate-report

# Expected frameworks tested:
# - ONNX Runtime (CPU)
# - PyTorch (CPU)
# - TensorFlow (CPU)
# - Intel OpenVINO
# - TVM (LLVM)
```

### 3. GPU Acceleration Benchmark

Compare GPU-accelerated frameworks for high-performance scenarios:

```bash
# GPU acceleration comparison
python multi_vendor_benchmark.py --device cuda --generate-report

# Expected frameworks tested:
# - ONNX Runtime (CUDA)
# - PyTorch (CUDA)
# - TensorRT (NVIDIA)
# - TVM (CUDA)
```

## Advanced Usage Scenarios

### 4. Framework-Specific Testing

Test specific frameworks for targeted evaluation:

```bash
# Compare Microsoft vs Google frameworks
python multi_vendor_benchmark.py \
  --frameworks onnxruntime_cpu tensorflow_cpu onnxruntime_cuda \
  --generate-report

# Compare optimization impact
python multi_vendor_benchmark.py \
  --frameworks tvm_llvm tvm_llvm_tuned tensorrt onnxruntime_cpu \
  --models resnet18_onnx \
  --generate-report
```

### 5. Model-Specific Analysis

Focus on specific models for detailed analysis:

```bash
# ResNet-18 performance across all frameworks
python multi_vendor_benchmark.py \
  --models resnet18_onnx \
  --warmup-iterations 10 \
  --measurement-iterations 50 \
  --generate-report

# Lightweight model comparison
python multi_vendor_benchmark.py \
  --models squeezenet_onnx mobilenetv2_onnx \
  --frameworks onnxruntime_cpu pytorch_cpu openvino_cpu \
  --generate-report
```

### 6. Batch Size Scaling Analysis

Analyze framework performance across different batch sizes:

```bash
# Single sample inference (latency-critical)
python multi_vendor_benchmark.py --batch-size 1 --generate-report

# Throughput optimization
python multi_vendor_benchmark.py --batch-size 8 --generate-report

# High-throughput scenarios
python multi_vendor_benchmark.py --batch-size 32 --generate-report
```

### 7. High-Precision Measurement

For research and detailed analysis requiring statistical rigor:

```bash
# High-precision benchmarking
python multi_vendor_benchmark.py \
  --warmup-iterations 20 \
  --measurement-iterations 100 \
  --generate-report \
  --output-dir high_precision_results
```

## Integration with Existing Baseline Benchmark

### 8. Combined TVM + Multi-Vendor Analysis

Run both TVM Meta Schedule tuning and multi-vendor comparison:

```bash
# Step 1: Run TVM Meta Schedule baseline benchmark
python baseline_benchmark.py \
  --benchmark-suite vision \
  --enable-tuning \
  --generate-html

# Step 2: Run multi-vendor comparison
python multi_vendor_benchmark.py \
  --models resnet18_onnx mobilenetv2_onnx squeezenet_onnx \
  --generate-report

# Step 3: Compare results for comprehensive analysis
```

### 9. Framework Selection Workflow

Complete workflow for selecting optimal framework:

```bash
# Phase 1: Quick overview
python multi_vendor_benchmark.py \
  --measurement-iterations 10 \
  --frameworks onnxruntime_cpu pytorch_cpu tensorflow_cpu

# Phase 2: Detailed analysis of top performers
python multi_vendor_benchmark.py \
  --frameworks onnxruntime_cpu openvino_cpu \
  --warmup-iterations 15 \
  --measurement-iterations 50 \
  --batch-size 1 \
  --generate-report

# Phase 3: Production validation
python multi_vendor_benchmark.py \
  --frameworks onnxruntime_cpu \
  --batch-size 1 4 8 16 \
  --generate-report
```

## Specialized Use Cases

### 10. Edge/Mobile Deployment Analysis

Optimize for resource-constrained environments:

```bash
# Lightweight frameworks for edge deployment
python multi_vendor_benchmark.py \
  --frameworks onnxruntime_cpu tensorflow_lite openvino_cpu \
  --models squeezenet_onnx mobilenetv2_onnx \
  --batch-size 1 \
  --generate-report
```

### 11. Cloud/Server Deployment Analysis

Optimize for high-throughput server deployment:

```bash
# Server-optimized frameworks
python multi_vendor_benchmark.py \
  --frameworks onnxruntime_cuda tensorrt pytorch_cuda \
  --batch-size 8 16 32 \
  --device cuda \
  --generate-report
```

### 12. Research Reproducibility

Standardized benchmarking for academic research:

```bash
# Research-grade reproducible benchmark
python multi_vendor_benchmark.py \
  --warmup-iterations 20 \
  --measurement-iterations 100 \
  --models resnet18_onnx \
  --frameworks onnxruntime_cpu pytorch_cpu tensorflow_cpu tvm_llvm \
  --output-dir research_results_$(date +%Y%m%d) \
  --generate-report
```

## Output Analysis Examples

### 13. JSON Results Processing

Process benchmark results programmatically:

```python
import json

# Load results
with open('multi_vendor_benchmark_results/multi_vendor_results.json', 'r') as f:
    results = json.load(f)

# Find fastest framework for each model
models = {}
for result in results:
    model = result['model']
    if model not in models or result['runtime_ms_mean'] < models[model]['runtime_ms_mean']:
        models[model] = result

for model, best_result in models.items():
    print(f"{model}: {best_result['framework']} ({best_result['runtime_ms_mean']:.1f}ms)")
```

### 14. Performance Regression Detection

Compare benchmark runs for regression testing:

```python
# Compare two benchmark runs
def compare_benchmarks(baseline_file, current_file):
    with open(baseline_file) as f:
        baseline = {r['framework'] + '_' + r['model']: r for r in json.load(f)}
    
    with open(current_file) as f:
        current = {r['framework'] + '_' + r['model']: r for r in json.load(f)}
    
    regressions = []
    improvements = []
    
    for key in baseline:
        if key in current:
            baseline_time = baseline[key]['runtime_ms_mean']
            current_time = current[key]['runtime_ms_mean']
            change = (current_time - baseline_time) / baseline_time
            
            if change > 0.05:  # 5% regression threshold
                regressions.append((key, change))
            elif change < -0.05:  # 5% improvement threshold
                improvements.append((key, -change))
    
    return regressions, improvements
```

### 15. Hardware Optimization Recommendations

Generate optimization recommendations based on results:

```python
def analyze_hardware_efficiency(results_file):
    with open(results_file) as f:
        results = json.load(f)
    
    # Analyze memory efficiency
    memory_efficient = sorted(results, key=lambda x: x['memory_mb_peak'])[:3]
    
    # Analyze throughput leaders
    throughput_leaders = sorted(results, key=lambda x: x['throughput_samples_per_sec'], reverse=True)[:3]
    
    # Analyze load time efficiency
    fast_load = sorted(results, key=lambda x: x['model_load_time_ms'])[:3]
    
    print("Memory Efficient:", [r['framework'] for r in memory_efficient])
    print("Throughput Leaders:", [r['framework'] for r in throughput_leaders])
    print("Fast Loading:", [r['framework'] for r in fast_load])
```

## CI/CD Integration Examples

### 16. Automated Performance Testing

Integrate with CI/CD pipelines:

```yaml
# .github/workflows/performance_benchmark.yml
name: Performance Benchmark
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        pip install onnxruntime tensorflow torch
        
    - name: Run benchmark
      run: |
        cd meta
        python multi_vendor_benchmark.py \
          --measurement-iterations 20 \
          --output-dir benchmark_results
          
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: meta/benchmark_results/
```

### 17. Performance Regression Alerts

Set up automated alerts for performance regressions:

```bash
#!/bin/bash
# performance_check.sh

# Run current benchmark
python multi_vendor_benchmark.py --output-dir current_results

# Compare with baseline
python -c "
import json
import sys

with open('baseline_results/multi_vendor_results.json') as f:
    baseline = json.load(f)
with open('current_results/multi_vendor_results.json') as f:
    current = json.load(f)

regressions = 0
for b, c in zip(baseline, current):
    if b['framework'] == c['framework'] and b['model'] == c['model']:
        change = (c['runtime_ms_mean'] - b['runtime_ms_mean']) / b['runtime_ms_mean']
        if change > 0.10:  # 10% regression threshold
            print(f'REGRESSION: {c[\"framework\"]} on {c[\"model\"]}: {change*100:.1f}% slower')
            regressions += 1

if regressions > 0:
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Performance regressions detected!"
    exit 1
else
    echo "No performance regressions found."
fi
```

## Custom Extensions

### 18. Adding Custom Frameworks

Extend the benchmark with proprietary or custom frameworks:

```python
from multi_vendor_benchmark import InferenceFramework, VendorBenchmarkResult

class CustomFramework(InferenceFramework):
    def __init__(self, device="cpu"):
        super().__init__("custom_framework", device)
    
    def load_model(self, model_path, input_shapes):
        # Load your custom framework model
        return CustomModel(model_path)
    
    def run_inference(self, model, input_data):
        # Run inference with your framework
        return model.predict(input_data)
    
    # ... implement other required methods

# Use in benchmark
benchmark = MultiVendorBenchmark()
benchmark.frameworks["custom_framework"] = CustomFramework()
results = benchmark.benchmark_all_frameworks()
```

### 19. Custom Metrics Collection

Add domain-specific metrics:

```python
from dataclasses import dataclass
from multi_vendor_benchmark import VendorBenchmarkResult

@dataclass
class ExtendedBenchmarkResult(VendorBenchmarkResult):
    accuracy_top1: float = 0.0
    energy_consumption_mj: float = 0.0
    carbon_footprint_g: float = 0.0

# Collect additional metrics during benchmarking
def extended_benchmark(framework, model, input_data):
    # ... run standard benchmark
    
    # Add custom metrics
    accuracy = measure_accuracy(model, validation_data)
    energy = measure_energy_consumption()
    carbon = calculate_carbon_footprint(energy)
    
    return ExtendedBenchmarkResult(
        # ... standard fields
        accuracy_top1=accuracy,
        energy_consumption_mj=energy,
        carbon_footprint_g=carbon
    )
```

## Best Practices Summary

### 20. Complete Evaluation Workflow

Comprehensive evaluation following best practices:

```bash
# 1. Quick survey of available frameworks
python multi_vendor_benchmark.py --measurement-iterations 5

# 2. Detailed analysis of top candidates
python multi_vendor_benchmark.py \
  --frameworks onnxruntime_cpu openvino_cpu tvm_llvm_tuned \
  --warmup-iterations 10 \
  --measurement-iterations 30 \
  --generate-report

# 3. Production validation with multiple batch sizes
for batch_size in 1 4 8 16; do
  python multi_vendor_benchmark.py \
    --frameworks onnxruntime_cpu \
    --batch-size $batch_size \
    --output-dir production_validation_b${batch_size}
done

# 4. Generate comprehensive report
python multi_vendor_benchmark.py \
  --frameworks onnxruntime_cpu openvino_cpu \
  --generate-report \
  --output-dir final_evaluation
```

This multi-vendor benchmark provides the foundation for making informed decisions about ML inference framework selection based on comprehensive, fair, and reproducible performance evaluation.