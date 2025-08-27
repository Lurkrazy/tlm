# Meta Scheduler Baseline Benchmark

This directory contains a standardized baseline benchmark for the TVM Meta Scheduler. The benchmark measures the performance of baseline (untuned) schedules on common ML workloads to provide a performance reference point.

## Overview

The baseline benchmark provides:

- **Standardized workloads**: Common ML operations like matrix multiplication, convolution, and dense layers
- **Consistent measurement**: Following the benchmark format specification in `docs/arch/benchmark.rst`
- **Hardware detection**: Automatic detection of CPU, memory, and GPU configurations
- **JSON output**: Structured results compatible with analysis tools
- **Extensible design**: Easy to add new workloads and measurement configurations

## Usage

### Basic Usage

Run the baseline benchmark with default settings:

```bash
cd meta/
python baseline_benchmark.py
```

### Advanced Usage

Specify target hardware and workloads:

```bash
# CPU targeting
python baseline_benchmark.py --target "llvm -mcpu=core-avx2"

# GPU targeting (if available)
python baseline_benchmark.py --target "cuda -arch=sm_75"

# Specific workloads only
python baseline_benchmark.py --workloads matmul_1024x1024 dense_512x512

# Custom output directory
python baseline_benchmark.py --output-dir ./my_benchmark_results
```

### Command Line Options

- `--target`: Target device specification (default: "llvm -mcpu=core-avx2")
- `--output-dir`: Output directory for results (default: "baseline_benchmark_results")
- `--workloads`: Specific workloads to run (default: all workloads)
- `--builder-timeout`: Builder timeout in seconds (default: 30)
- `--runner-timeout`: Runner timeout in seconds (default: 10)

## Workloads

The benchmark includes the following baseline workloads:

### Matrix Multiplication
- **matmul_1024x1024**: 1024x1024 matrix multiplication
- Tests basic linear algebra performance

### Convolution
- **conv2d_resnet_c1**: ResNet-18 first convolution layer
- Tests 2D convolution with realistic CNN parameters

### Dense Layer
- **dense_512x512**: Fully connected layer 512x512
- Tests dense matrix multiplication patterns

## Output Format

Results are saved in JSON format following the benchmark specification:

```json
[
  {
    "workload": "matmul_1024x1024",
    "engine": "tvm_meta_schedule_baseline",
    "hardware": "Intel_Core_i7_8core", 
    "runtime_ms_mean": 45.2,
    "runtime_ms_std": 2.1,
    "workload_metadata": {
      "description": "Matrix multiplication 1024x1024",
      "type": "matmul",
      "input_shapes": [[1024, 1024], [1024, 1024]],
      "data_type": "float32"
    },
    "engine_version": "baseline_v1.0",
    "engine_config": {
      "target": "llvm -mcpu=core-avx2",
      "opt_level": 3
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
    "metrics": {},
    "runtime_raw": [
      {"runtime_ms": 44.8},
      {"runtime_ms": 45.6},
      ...
    ],
    "timestamp": "2024-01-15T10:30:00Z"
  }
]
```

## Integration with Existing Tools

The baseline benchmark integrates with the existing Meta Scheduler infrastructure:

- Uses `measure_programs.py` patterns for measurement
- Compatible with `meta_common.py` utilities
- Follows the same target and hardware detection as other meta tools
- Output format matches the specification in `docs/arch/benchmark.rst`

## Testing

Run the test suite:

```bash
cd tests/python/unittest/
python -m pytest test_meta_schedule_baseline_benchmark.py -v
```

The tests cover:
- Hardware detection
- Workload creation
- Benchmark execution 
- Result saving and formatting
- Error handling and mock mode

## Extending the Benchmark

### Adding New Workloads

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

### Custom Measurement Logic

Override `_measure_baseline_performance()` to implement custom measurement:

```python
def _measure_baseline_performance(self, workload, mod, exec_config):
    # Custom measurement logic
    results = my_custom_measurement(mod, exec_config)
    return {
        "runtime_ms_mean": results.mean,
        "runtime_ms_std": results.std,
        "runtime_raw": results.raw_data
    }
```

## Dependencies

- **Required**: Python 3.7+, psutil
- **Optional**: TVM (for actual measurement), GPUtil (for GPU detection)
- **Development**: pytest (for testing)

When TVM is not available, the benchmark runs in mock mode for testing and development.

## Performance Considerations

- The benchmark uses baseline schedules without Meta Schedule tuning
- Results provide a reference point for measuring tuning effectiveness
- Hardware detection ensures consistent environment reporting
- Multiple measurement repeats improve statistical reliability

## Troubleshooting

### Common Issues

1. **TVM Import Error**: The benchmark will run in mock mode
2. **Target Not Supported**: Check your TVM installation and target string
3. **Timeout Errors**: Increase `--builder-timeout` and `--runner-timeout`
4. **Memory Issues**: Use smaller workloads or increase system memory

### Debug Mode

Enable detailed logging by modifying the logging level:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When contributing new workloads or features:

1. Follow the existing code style and patterns
2. Add comprehensive test cases
3. Update this documentation
4. Ensure compatibility with the benchmark format specification

## License

Licensed under the Apache License, Version 2.0. See the main repository LICENSE file for details.