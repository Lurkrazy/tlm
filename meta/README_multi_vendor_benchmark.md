# Multi-Vendor ML Inference Benchmark

A comprehensive benchmarking system that provides fair performance comparison across multiple ML inference frameworks and vendors. This benchmark ensures consistent measurement methodology, hardware utilization, and model inputs across all tested frameworks.

## Overview

The multi-vendor benchmark addresses the critical need for **fair and standardized performance evaluation** across different ML inference solutions. It implements industry best practices for benchmarking to ensure meaningful and reproducible comparisons.

## Fair Benchmarking Methodology

### Core Principles

1. **Identical Models**: All frameworks test the same ONNX models with identical architectures
2. **Consistent Inputs**: Same input shapes, data types, batch sizes, and random seed across all tests  
3. **Same Hardware**: All frameworks run on identical hardware configuration
4. **Standardized Measurement**: Consistent timing methodology, warmup procedures, and statistical analysis
5. **Realistic Workloads**: Real-world models from ONNX Model Zoo, not synthetic benchmarks
6. **Optimized Comparison**: Compare optimized versions where available (TensorRT, OpenVINO, etc.)
7. **Transparent Reporting**: Full configuration details and raw measurements included

### Measurement Methodology

#### Warmup Protocol
- **Minimum 5 warmup iterations** to ensure JIT compilation and memory allocation
- **First inference timing** captured separately (important for deployment scenarios)
- **Model loading time** measured independently from inference time

#### Statistical Rigor  
- **20+ measurement iterations** for statistical reliability
- **Mean and standard deviation** reported for all metrics
- **Raw measurements** preserved for post-analysis
- **Memory peak usage** tracked during inference

#### Performance Metrics
- **Latency**: Mean inference time with confidence intervals
- **Throughput**: Samples processed per second
- **Memory Usage**: Peak memory consumption during inference
- **Model Loading**: Time to load and initialize model
- **First Inference**: Cold start performance

### Hardware Consistency

All frameworks tested on:
- **Same CPU/GPU**: Identical processing units
- **Same Memory**: Same RAM configuration and availability
- **Same OS/Drivers**: Consistent system environment
- **Same Power Mode**: Fixed CPU frequency and power settings
- **Process Isolation**: Each framework tested in separate process

## Supported Frameworks

### Industry Leaders
- **ONNX Runtime** (Microsoft): CPU and CUDA execution providers
- **TensorRT** (NVIDIA): GPU-optimized inference engine
- **OpenVINO** (Intel): CPU and Intel GPU optimization
- **PyTorch** (Meta): Native PyTorch and TorchScript
- **TensorFlow** (Google): TensorFlow and TensorFlow Lite
- **TVM** (Apache): Tensor compiler with auto-tuning

### Framework-Specific Optimizations

Each framework is configured for optimal performance:

#### ONNX Runtime
- **CPU**: Uses all available CPU cores with optimal thread settings
- **CUDA**: GPU memory optimization and kernel fusion
- **Execution Providers**: Automatic fallback from GPU to CPU when needed

#### TensorRT  
- **Precision**: FP32, FP16, and INT8 optimization paths
- **Dynamic Shapes**: Optimized for specific input dimensions
- **Layer Fusion**: Aggressive optimization for inference

#### OpenVINO
- **CPU Optimization**: Vectorization and multi-threading
- **Intel Hardware**: Optimized for Intel CPUs and integrated GPUs
- **Model Optimization**: Graph optimization and quantization

#### PyTorch
- **TorchScript**: JIT compilation for production deployment
- **CUDA**: GPU acceleration with optimized memory management
- **Threading**: Optimal CPU thread configuration

#### TensorFlow
- **Graph Optimization**: TensorFlow's graph optimization passes
- **XLA**: Accelerated Linear Algebra compilation
- **TensorFlow Lite**: Mobile/edge optimization

#### TVM
- **Auto-Tuning**: Meta Schedule optimization with configurable trials
- **Target-Specific**: Hardware-specific code generation
- **Operator Fusion**: Advanced graph optimization

## Benchmark Models

All models sourced from the **ONNX Model Zoo** for consistency and reproducibility:

### Vision Models
- **ResNet-18**: Standard CNN architecture (11MB model, 224x224 input)
- **MobileNetV2**: Efficient mobile architecture (14MB model, 224x224 input)  
- **SqueezeNet**: Lightweight CNN (5MB model, 224x224 input)

### Model Characteristics
- **Input Format**: NCHW (batch, channels, height, width)
- **Data Type**: Float32 (FP32) for baseline comparison
- **Batch Sizes**: 1, 4, 8, 16 for throughput analysis
- **Pre-processing**: Consistent normalization and scaling

## Usage

### Quick Start

```bash
# Benchmark all available frameworks on all models
python multi_vendor_benchmark.py --generate-report

# CPU-only comparison
python multi_vendor_benchmark.py --device cpu --generate-report

# Specific frameworks and models
python multi_vendor_benchmark.py --frameworks onnxruntime_cpu pytorch_cpu tvm_llvm --models resnet18_onnx

# High-precision measurement
python multi_vendor_benchmark.py --warmup-iterations 10 --measurement-iterations 50
```

### Advanced Configuration

```bash
# GPU acceleration where available
python multi_vendor_benchmark.py --device cuda --batch-size 8

# Custom output directory and model cache
python multi_vendor_benchmark.py --output-dir ./benchmark_results --model-cache-dir ./models

# Framework-specific testing
python multi_vendor_benchmark.py --frameworks onnxruntime_cpu onnxruntime_cuda pytorch_cpu pytorch_cuda
```

## Output and Reporting

### JSON Results Format

Detailed results follow a standardized format:

```json
{
  "vendor": "Microsoft",
  "framework": "onnxruntime_cpu", 
  "model": "resnet18_onnx",
  "hardware": "Intel_x86_64",
  "runtime_ms_mean": 23.45,
  "runtime_ms_std": 1.2,
  "memory_mb_peak": 245.6,
  "throughput_samples_per_sec": 42.6,
  "model_load_time_ms": 156.3,
  "first_inference_ms": 89.2,
  "warmup_iterations": 5,
  "measurement_iterations": 20,
  "batch_size": 1,
  "execution_provider": "CPUExecutionProvider",
  "optimization_level": "default",
  "precision": "fp32",
  "raw_measurements": [23.1, 23.8, 22.9, ...]
}
```

### HTML Comparison Report

Interactive report includes:
- **Performance Rankings**: Best to worst across all metrics
- **Per-Model Comparison**: Side-by-side framework performance
- **Framework Summary**: Aggregate statistics and rankings
- **Configuration Details**: Hardware and software specifications
- **Visual Indicators**: Color-coded performance levels

### Key Insights Provided

1. **Absolute Performance**: Raw inference times for deployment planning
2. **Relative Performance**: Framework rankings for technology selection
3. **Optimization Impact**: Baseline vs optimized performance gains
4. **Memory Efficiency**: Memory usage patterns across frameworks
5. **Loading Overhead**: Model initialization costs
6. **Scalability**: Batch size performance characteristics

## Extending the Benchmark

### Adding New Frameworks

Implement the `InferenceFramework` abstract class:

```python
class CustomFramework(InferenceFramework):
    def load_model(self, model_path: str, input_shapes: Dict) -> Any:
        # Load and initialize your framework's model
        pass
    
    def run_inference(self, model: Any, input_data: Dict) -> Dict:
        # Execute inference and return outputs
        pass
    
    def warm_up(self, model: Any, input_data: Dict, iterations: int) -> float:
        # Warm up model and return average time
        pass
    
    def get_model_info(self, model: Any) -> Dict:
        # Return framework-specific model information
        pass
```

### Adding New Models

Add models to the `_get_benchmark_models()` method:

```python
BenchmarkModel(
    name="custom_model_onnx",
    description="Custom Model Description", 
    model_url="https://example.com/model.onnx",
    input_shapes={"input": [1, 3, 224, 224]},
    category="vision"
)
```

### Custom Metrics

Extend `VendorBenchmarkResult` with additional metrics:

```python
@dataclass
class ExtendedBenchmarkResult(VendorBenchmarkResult):
    accuracy_top1: float = 0.0
    accuracy_top5: float = 0.0
    flops_per_inference: int = 0
    energy_consumption_mj: float = 0.0
```

## Best Practices for Fair Comparison

### Environment Setup
1. **System Isolation**: Close unnecessary applications
2. **CPU Governor**: Set to 'performance' mode for consistent clocking
3. **Memory**: Ensure sufficient RAM to avoid swapping
4. **Temperature**: Monitor CPU/GPU temperature for thermal throttling
5. **Process Priority**: Run benchmark with elevated priority

### Statistical Validity
1. **Multiple Runs**: Execute benchmark multiple times across different days
2. **Outlier Detection**: Remove statistical outliers from final analysis
3. **Confidence Intervals**: Report confidence intervals with means
4. **Sample Size**: Ensure sufficient measurements for statistical power

### Framework Fairness
1. **Latest Versions**: Use most recent stable releases
2. **Optimal Configuration**: Apply framework-specific optimizations
3. **Model Format**: Use each framework's preferred model format when available
4. **Documentation**: Follow each framework's performance tuning guidelines

## Integration with Existing Tools

The multi-vendor benchmark integrates with:
- **Meta Scheduler Baseline Benchmark**: Shares model downloading and caching
- **TVM Auto-Tuning**: Leverages Meta Schedule tuning infrastructure  
- **ONNX Ecosystem**: Uses standard ONNX models for broad compatibility
- **CI/CD Pipelines**: JSON output suitable for automated performance tracking

## Research and Development Use Cases

### Algorithm Development
- **Optimization Impact**: Quantify benefits of new optimization techniques
- **Hardware Evaluation**: Compare performance across different hardware
- **Model Architecture**: Evaluate inference performance of new architectures

### Product Deployment
- **Framework Selection**: Data-driven framework choice for production
- **Hardware Sizing**: Capacity planning based on performance requirements
- **SLA Definition**: Performance guarantees based on benchmark results

### Academic Research
- **Reproducible Results**: Standardized methodology for paper comparisons
- **Baseline Establishment**: Reference performance for new techniques
- **Cross-Platform Analysis**: Performance portability studies

## Troubleshooting

### Common Issues

1. **Framework Import Errors**: Install missing frameworks or run subset
2. **Model Download Failures**: Check network connectivity and retry
3. **CUDA Out of Memory**: Reduce batch size or use CPU fallback  
4. **Inconsistent Results**: Check system load and thermal throttling

### Performance Validation

1. **Sanity Checks**: Verify frameworks produce similar outputs
2. **Baseline Comparison**: Compare against known reference results
3. **System Monitoring**: Watch CPU/memory/GPU utilization during tests

### Debug Mode

Enable detailed logging:

```python
logging.getLogger().setLevel(logging.DEBUG)
```

## Dependencies

### Required
- **Python 3.7+**: Core runtime
- **NumPy**: Numerical operations
- **psutil**: System monitoring

### Optional (Auto-detected)
- **onnxruntime**: ONNX Runtime framework
- **torch**: PyTorch framework
- **tensorflow**: TensorFlow framework  
- **tensorrt**: TensorRT optimization
- **openvino**: Intel OpenVINO toolkit
- **tvm**: Apache TVM compiler

Install all dependencies:

```bash
pip install onnxruntime torch tensorflow tensorrt openvino-dev tvm psutil numpy
```

## Future Enhancements

### Planned Features
- **Accuracy Benchmarking**: Model accuracy validation across frameworks
- **Energy Measurement**: Power consumption profiling
- **Quantization Comparison**: INT8/FP16 precision analysis
- **Edge Device Support**: ARM/mobile processor benchmarking
- **Distributed Inference**: Multi-GPU and cluster performance

### Extensibility
- **Plugin Architecture**: Framework plugins for easy extension
- **Custom Metrics**: User-defined performance metrics
- **Benchmark Suites**: Domain-specific benchmark collections
- **Automated Reporting**: Integration with performance monitoring systems

This multi-vendor benchmark provides the foundation for fair, comprehensive, and reproducible ML inference performance evaluation across the entire ecosystem of available frameworks and optimization tools.