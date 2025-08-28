# TVM Meta Scheduler End-to-End Benchmark

This directory contains a comprehensive end-to-end benchmark system for the TVM Meta Scheduler. The benchmark measures the performance of both baseline (untuned) schedules and Meta Schedule tuned schedules on common ML workloads and complete models to provide performance evaluation and comparison.

## Overview

The end-to-end benchmark provides:

- **Standardized workloads**: Common ML operations like matrix multiplication, convolution, and dense layers
- **Complete model benchmarks**: End-to-end inference on real ML models (ResNet, MobileNet, SqueezeNet)
- **Meta Schedule integration**: Automatic tuning and baseline vs tuned performance comparison
- **Consistent measurement**: Following the benchmark format specification in `docs/arch/benchmark.rst`
- **Hardware detection**: Automatic detection of CPU, memory, and GPU configurations
- **Model downloading**: Automatic download and caching of ONNX models
- **Comprehensive reporting**: JSON output, HTML reports, and performance analysis
- **Batch processing**: Run multiple configurations for A/B testing and regression detection
- **Extensible design**: Easy to add new workloads, models, and measurement configurations

## Usage

### Basic Usage

Run the baseline benchmark with default settings:

```bash
cd meta/
python baseline_benchmark.py
```

### End-to-End Model Benchmarking

Run complete model inference benchmarks:

```bash
# Run vision model benchmarks
python baseline_benchmark.py --benchmark-suite vision

# Run with tuning enabled for comparison
python baseline_benchmark.py --benchmark-suite vision --enable-tuning

# Generate HTML report
python baseline_benchmark.py --benchmark-suite vision --enable-tuning --generate-html
```

### Advanced Usage

Specify target hardware, workloads, and features:

```bash
# CPU targeting with tuning
python baseline_benchmark.py --target "llvm -mcpu=core-avx2" --enable-tuning

# GPU targeting (if available)
python baseline_benchmark.py --target "cuda -arch=sm_75" --enable-tuning

# Specific workloads only
python baseline_benchmark.py --workloads matmul_1024x1024 resnet18_onnx --enable-tuning

# Custom output directory and model cache
python baseline_benchmark.py --output-dir ./my_benchmark_results --model-cache-dir ./model_cache

# Operators only (no models)
python baseline_benchmark.py --benchmark-suite operators --enable-tuning
```

### Command Line Options

- `--target`: Target device specification (default: "llvm -mcpu=core-avx2")
- `--output-dir`: Output directory for results (default: "baseline_benchmark_results")
- `--workloads`: Specific workloads to run (default: all workloads)
- `--benchmark-suite`: Benchmark suite to run ("operators", "vision", "all")
- `--builder-timeout`: Builder timeout in seconds (default: 30)
- `--runner-timeout`: Runner timeout in seconds (default: 10)
- `--enable-tuning`: Enable Meta Schedule tuning for comparison
- `--model-cache-dir`: Directory to cache downloaded models
- `--generate-html`: Generate HTML report with performance analysis

## End-to-End Model Benchmarks

The benchmark now includes complete ML model inference benchmarks:

### Vision Models
- **ResNet-18**: Image classification model (224x224 input)
- **MobileNetV2**: Efficient mobile vision model 
- **SqueezeNet**: Lightweight CNN architecture

Models are automatically downloaded from the ONNX Model Zoo and cached locally for reuse.

## Workloads

The benchmark includes both operator-level and model-level workloads:

### Operator Benchmarks
- **matmul_1024x1024**: 1024x1024 matrix multiplication
- **conv2d_resnet_c1**: ResNet-18 first convolution layer  
- **dense_512x512**: Fully connected layer 512x512

### Model Benchmarks
- **resnet18_onnx**: Complete ResNet-18 inference
- **mobilenetv2_onnx**: Complete MobileNetV2 inference
- **squeezenet_onnx**: Complete SqueezeNet inference

## Benchmark Suites

Predefined benchmark suites for different use cases:

- **operators**: Run only operator-level benchmarks
- **vision**: Run only vision model benchmarks  
- **all**: Run all available benchmarks (default)

## Meta Schedule Integration

When `--enable-tuning` is specified, the benchmark:

1. Measures baseline performance with default schedules
2. Runs Meta Schedule tuning with limited trials (50 for quick evaluation)
3. Measures tuned performance with optimized schedules
4. Calculates speedup and performance improvements
5. Reports both baseline and tuned results for comparison

## Output Formats

### JSON Output

Results are saved in structured JSON format following the benchmark specification:

```json
[
  {
    "workload": "resnet18_onnx",
    "engine": "tvm_meta_schedule_baseline_with_tuning",
    "hardware": "Intel_Core_i7_8core", 
    "runtime_ms_mean": 45.2,
    "runtime_ms_std": 2.1,
    "tuned_runtime_ms_mean": 28.7,
    "tuned_runtime_ms_std": 1.8,
    "speedup": 1.57,
    "workload_metadata": {
      "description": "ResNet-18 ONNX Model",
      "type": "model",
      "input_shapes": [[1, 3, 224, 224]],
      "data_type": "float32",
      "model_url": "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v1-7.onnx"
    },
    "engine_version": "baseline_v1.1",
    "engine_config": {
      "target": "llvm -mcpu=core-avx2",
      "opt_level": 3,
      "enable_tuning": true
    },
    "hardware_config": {
      "cpu_count": 8,
      "memory_gb": 16.0,
      "cpu_platform": "Intel Core i7"
    },
    "execution_config": {
      "number": 1,
      "repeat": 10,
      "min_repeat_ms": 0
    },
    "metrics": {
      "tuning_enabled": true,
      "speedup": 1.57
    },
    "timestamp": "2024-01-15T10:30:00Z"
  }
]
```

### HTML Report

Generate comprehensive HTML reports with:

```bash
python baseline_benchmark.py --enable-tuning --generate-html
```

The HTML report includes:
- Executive summary with key metrics
- Operator benchmark results table
- End-to-end model benchmark results table  
- Performance analysis and speedup calculations
- Configuration details and hardware information
- Color-coded speedup indicators (green=good, orange=neutral, red=regression)

## Integration with Existing Tools

The end-to-end benchmark integrates with the existing Meta Scheduler infrastructure:

- Uses patterns from `measure_programs.py` for measurement consistency
- Compatible with `meta_common.py` utilities for data handling
- Follows the same target specification and hardware detection as other meta tools
- Output format matches the specification in `docs/arch/benchmark.rst`
- Integrates with Meta Schedule tuning pipeline for automated optimization

## Batch Processing and Comparison

### Batch Benchmarking

Run multiple configurations for A/B testing:

```python
from baseline_benchmark import BaselineBenchmark

benchmark = BaselineBenchmark()

configurations = [
    {"name": "baseline", "enable_tuning": False, "target": "llvm"},
    {"name": "tuned", "enable_tuning": True, "target": "llvm"},
    {"name": "gpu_baseline", "enable_tuning": False, "target": "cuda"},
    {"name": "gpu_tuned", "enable_tuning": True, "target": "cuda"}
]

all_results = benchmark.run_batch_benchmark(configurations)
```

### Result Comparison

Compare benchmark results for regression testing:

```python
comparison = benchmark.compare_results(baseline_results, new_results)
print(f"Improvements: {comparison['summary']['improvements']}")
print(f"Regressions: {comparison['summary']['regressions']}")
print(f"Geometric Mean Speedup: {comparison['summary']['geometric_mean_speedup']:.2f}x")
```

## Testing

Run the comprehensive test suite:

```bash
cd tests/python/unittest/
python -m pytest test_meta_schedule_baseline_benchmark.py -v
```

The tests cover:
- Hardware detection and configuration
- Workload creation for both operators and models
- Model downloading and caching
- Benchmark execution with and without tuning
- Result saving and formatting (JSON and HTML)
- Error handling and mock mode
- Integration testing for complete workflows

## Extending the Benchmark

### Adding New Models

1. Add model definition to `_get_end_to_end_models()`:

```python
ModelBenchmark(
    name="my_model_onnx",
    description="My Custom Model",
    model_url="https://example.com/model.onnx", 
    model_format="onnx",
    input_shapes={"data": [1, 3, 224, 224]},
    category="vision"
)
```

2. Model will be automatically included in benchmark suites

### Adding New Operators

1. Define a new `BenchmarkWorkload` in `_get_baseline_workloads()`
2. Implement the workload creation in `_create_tvm_workload()`
3. Add corresponding test cases

Example:
```python
BenchmarkWorkload(
    name="my_custom_op",
    description="My custom operation",
    workload_type="custom",
    input_shapes=[[100, 200], [200, 300]]
)
```

### Custom Benchmark Suites

Add new benchmark suite categories by modifying the `run_benchmark()` method:

```python
elif benchmark_suite == "nlp":
    nlp_models = [m.name for m in models if m.category == "nlp"]
    selected_workloads = [w for w in all_workloads if w.name in nlp_models]
```

## Dependencies

- **Required**: Python 3.7+, psutil, numpy
- **Optional**: TVM (for actual measurement), ONNX (for model loading), GPUtil (for GPU detection)
- **Development**: pytest (for testing)

When TVM or ONNX are not available, the benchmark runs in mock mode for testing and development.

## Performance Considerations

- **Baseline measurement**: Uses TVM's default schedules without Meta Schedule optimization
- **Meta Schedule tuning**: Limited to 50 trials per workload for quick evaluation (adjustable)
- **Model caching**: Downloaded models are cached locally to avoid repeated downloads
- **Hardware detection**: Ensures consistent environment reporting across runs
- **Statistical reliability**: Multiple measurement repeats with mean and standard deviation reporting

## End-to-End Benefits

This end-to-end benchmark enables:

1. **Complete performance evaluation** from model loading to inference measurement
2. **Real-world performance assessment** on complete ML models, not just isolated operators
3. **Meta Schedule effectiveness quantification** through baseline vs tuned comparisons
4. **Regression testing** to ensure optimizations don't degrade performance
5. **Cross-platform comparison** across different hardware targets
6. **Research reproducibility** through standardized models and measurement methodology
7. **Automated performance tracking** for CI/CD integration

## Troubleshooting

### Common Issues

1. **TVM Import Error**: The benchmark will run in mock mode
2. **ONNX Import Error**: Model benchmarks will be skipped, operators will still run
3. **Model Download Failure**: Check network connectivity and model URLs
4. **Target Not Supported**: Check your TVM installation and target string
5. **Timeout Errors**: Increase `--builder-timeout` and `--runner-timeout`
6. **Memory Issues**: Use smaller workloads, reduce model batch sizes, or increase system memory
7. **Tuning Failures**: Check tuning logs in the output directory

### Debug Mode

Enable detailed logging by modifying the logging level:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Performance Troubleshooting

- Monitor system resources during benchmarking
- Check that target specifications match your hardware
- Verify model inputs are correctly sized
- Use smaller trial counts for faster tuning iterations

## Examples

### Quick Performance Check

```bash
# Run operators only for quick check
python baseline_benchmark.py --benchmark-suite operators --enable-tuning
```

### Complete Model Evaluation

```bash
# Full end-to-end evaluation with HTML report
python baseline_benchmark.py --benchmark-suite all --enable-tuning --generate-html
```

### CI/Regression Testing

```bash
# Focused operator testing for CI
python baseline_benchmark.py --benchmark-suite operators --timeout 60
```

### Research Evaluation

```bash
# Comprehensive research evaluation
python baseline_benchmark.py --benchmark-suite all --enable-tuning --generate-html --model-cache-dir ./models
```