#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Multi-Vendor ML Inference Benchmark Demo

Demonstrates the multi-vendor benchmarking system with simulated results
when actual ML frameworks are not available.
"""

import json
import os
import time
import platform
import tempfile
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class BenchmarkModel:
    """Defines a model for multi-vendor benchmarking"""
    name: str
    description: str
    model_url: str
    input_shapes: Dict[str, List[int]]
    data_type: str = "float32"
    category: str = "vision"


@dataclass 
class VendorBenchmarkResult:
    """Benchmark result for a specific vendor/framework"""
    vendor: str
    framework: str
    model: str
    hardware: str
    runtime_ms_mean: float
    runtime_ms_std: float
    memory_mb_peak: float
    throughput_samples_per_sec: float
    model_load_time_ms: float
    first_inference_ms: float
    warmup_iterations: int
    measurement_iterations: int
    batch_size: int
    execution_provider: Optional[str] = None
    optimization_level: Optional[str] = None
    precision: str = "fp32"
    timestamp: str = ""


class DemoMultiVendorBenchmark:
    """Demo multi-vendor benchmark with simulated results"""
    
    def __init__(self, output_dir: str = "demo_benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Simulate hardware detection
        self.hardware = f"{platform.processor()}_{platform.machine()}"
        
        # Define demo frameworks with simulated performance characteristics
        self.framework_configs = {
            "onnxruntime_cpu": {
                "vendor": "Microsoft",
                "execution_provider": "CPUExecutionProvider",
                "base_latency": 45.0,  # ms
                "variance": 0.15,
                "memory_multiplier": 1.0
            },
            "onnxruntime_cuda": {
                "vendor": "Microsoft", 
                "execution_provider": "CUDAExecutionProvider",
                "base_latency": 12.0,  # ms
                "variance": 0.10,
                "memory_multiplier": 1.2
            },
            "pytorch_cpu": {
                "vendor": "Meta",
                "execution_provider": None,
                "base_latency": 52.0,  # ms
                "variance": 0.20,
                "memory_multiplier": 1.1
            },
            "pytorch_cuda": {
                "vendor": "Meta",
                "execution_provider": None, 
                "base_latency": 15.0,  # ms
                "variance": 0.12,
                "memory_multiplier": 1.3
            },
            "tensorflow_cpu": {
                "vendor": "Google",
                "execution_provider": None,
                "base_latency": 48.0,  # ms
                "variance": 0.18,
                "memory_multiplier": 1.05
            },
            "tensorrt": {
                "vendor": "NVIDIA",
                "execution_provider": None,
                "base_latency": 8.0,   # ms (optimized)
                "variance": 0.08,
                "memory_multiplier": 0.9
            },
            "openvino_cpu": {
                "vendor": "Intel",
                "execution_provider": None,
                "base_latency": 35.0,  # ms (optimized for Intel)
                "variance": 0.12,
                "memory_multiplier": 0.85
            },
            "tvm_llvm": {
                "vendor": "Apache",
                "execution_provider": None,
                "base_latency": 40.0,  # ms
                "variance": 0.16,
                "memory_multiplier": 0.95
            },
            "tvm_llvm_tuned": {
                "vendor": "Apache",
                "execution_provider": None,
                "base_latency": 28.0,  # ms (after auto-tuning)
                "variance": 0.14,
                "memory_multiplier": 0.90
            }
        }
    
    def get_benchmark_models(self) -> List[BenchmarkModel]:
        """Get list of benchmark models"""
        return [
            BenchmarkModel(
                name="resnet18_onnx",
                description="ResNet-18 Image Classification",
                model_url="https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v1-7.onnx",
                input_shapes={"data": [1, 3, 224, 224]},
                category="vision"
            ),
            BenchmarkModel(
                name="mobilenetv2_onnx", 
                description="MobileNetV2 Efficient Vision Model",
                model_url="https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
                input_shapes={"data": [1, 3, 224, 224]},
                category="vision"
            ),
            BenchmarkModel(
                name="squeezenet_onnx",
                description="SqueezeNet Lightweight CNN", 
                model_url="https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.0-7.onnx",
                input_shapes={"data": [1, 3, 224, 224]},
                category="vision"
            )
        ]
    
    def simulate_benchmark(self, framework_name: str, model: BenchmarkModel, 
                          warmup_iterations: int = 5, measurement_iterations: int = 20,
                          batch_size: int = 1) -> VendorBenchmarkResult:
        """Simulate benchmarking a model with a framework"""
        
        if framework_name not in self.framework_configs:
            raise ValueError(f"Unknown framework: {framework_name}")
        
        config = self.framework_configs[framework_name]
        
        # Simulate model complexity factor
        model_complexity = {
            "resnet18_onnx": 1.0,
            "mobilenetv2_onnx": 0.7,  # More efficient
            "squeezenet_onnx": 0.4    # Lightweight
        }
        complexity_factor = model_complexity.get(model.name, 1.0)
        
        # Simulate batch size scaling (sub-linear for most frameworks)
        batch_factor = batch_size ** 0.8
        
        # Calculate simulated performance metrics
        base_latency = config["base_latency"] * complexity_factor * batch_factor
        variance = config["variance"]
        
        # Simulate measurement variance
        measurements = []
        random.seed(42)  # For reproducible demo results
        for _ in range(measurement_iterations):
            # Add realistic variance to measurements
            measurement = base_latency * (1 + random.gauss(0, variance))
            measurements.append(max(0.1, measurement))  # Minimum 0.1ms
        
        runtime_mean = sum(measurements) / len(measurements)
        runtime_std = (sum((x - runtime_mean) ** 2 for x in measurements) / len(measurements)) ** 0.5
        
        # Calculate derived metrics
        throughput = (batch_size * 1000) / runtime_mean  # samples per second
        memory_peak = 200 * config["memory_multiplier"] * complexity_factor  # MB
        model_load_time = random.uniform(50, 200)  # ms
        first_inference_time = runtime_mean * random.uniform(1.5, 2.5)  # First inference slower
        
        return VendorBenchmarkResult(
            vendor=config["vendor"],
            framework=framework_name,
            model=model.name,
            hardware=self.hardware,
            runtime_ms_mean=runtime_mean,
            runtime_ms_std=runtime_std,
            memory_mb_peak=memory_peak,
            throughput_samples_per_sec=throughput,
            model_load_time_ms=model_load_time,
            first_inference_ms=first_inference_time,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            batch_size=batch_size,
            execution_provider=config["execution_provider"],
            optimization_level="default",
            precision="fp32",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
    
    def run_demo_benchmark(self, models: List[str] = None, 
                          frameworks: List[str] = None,
                          batch_size: int = 1) -> List[VendorBenchmarkResult]:
        """Run demo benchmark with simulated results"""
        
        benchmark_models = self.get_benchmark_models()
        if models:
            benchmark_models = [m for m in benchmark_models if m.name in models]
        
        selected_frameworks = list(self.framework_configs.keys())
        if frameworks:
            selected_frameworks = [f for f in selected_frameworks if f in frameworks]
        
        results = []
        
        print(f"Running demo benchmark on {len(benchmark_models)} models with {len(selected_frameworks)} frameworks")
        print("=" * 80)
        
        for model in benchmark_models:
            print(f"\nBenchmarking {model.name} ({model.description})")
            
            for framework_name in selected_frameworks:
                print(f"  Testing {framework_name}...")
                
                try:
                    result = self.simulate_benchmark(framework_name, model, batch_size=batch_size)
                    results.append(result)
                    print(f"    Runtime: {result.runtime_ms_mean:.1f}ms, Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    continue
        
        return results
    
    def save_results(self, results: List[VendorBenchmarkResult], filename: str = "demo_results.json") -> str:
        """Save demo results to JSON"""
        output_path = os.path.join(self.output_dir, filename)
        
        results_data = [asdict(result) for result in results]
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        return output_path
    
    def generate_summary_report(self, results: List[VendorBenchmarkResult]) -> str:
        """Generate summary report"""
        
        # Group results by model
        model_results = {}
        for result in results:
            if result.model not in model_results:
                model_results[result.model] = []
            model_results[result.model].append(result)
        
        print("\n" + "=" * 80)
        print("MULTI-VENDOR BENCHMARK SUMMARY REPORT")
        print("=" * 80)
        
        for model_name, model_results_list in model_results.items():
            print(f"\n{model_name.upper()}")
            print("-" * 60)
            
            # Sort by performance (lowest latency first) 
            sorted_results = sorted(model_results_list, key=lambda x: x.runtime_ms_mean)
            
            print(f"{'Framework':<20} {'Vendor':<12} {'Runtime (ms)':<15} {'Throughput':<12} {'Rank'}")
            print("-" * 70)
            
            for i, result in enumerate(sorted_results):
                rank_indicator = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
                print(f"{result.framework:<20} {result.vendor:<12} {result.runtime_ms_mean:>8.1f} ± {result.runtime_ms_std:>4.1f} {result.throughput_samples_per_sec:>8.1f} {rank_indicator}")
        
        # Overall framework ranking
        print(f"\nOVERALL FRAMEWORK RANKING")
        print("-" * 40)
        
        framework_stats = {}
        for result in results:
            if result.framework not in framework_stats:
                framework_stats[result.framework] = {
                    "total_throughput": 0,
                    "count": 0,
                    "vendor": result.vendor
                }
            framework_stats[result.framework]["total_throughput"] += result.throughput_samples_per_sec
            framework_stats[result.framework]["count"] += 1
        
        # Calculate average throughput per framework
        framework_rankings = []
        for framework, stats in framework_stats.items():
            avg_throughput = stats["total_throughput"] / stats["count"]
            framework_rankings.append((framework, stats["vendor"], avg_throughput))
        
        framework_rankings.sort(key=lambda x: x[2], reverse=True)
        
        print(f"{'Framework':<20} {'Vendor':<12} {'Avg Throughput':<15} {'Overall Rank'}")
        print("-" * 60)
        for i, (framework, vendor, avg_throughput) in enumerate(framework_rankings):
            rank_indicator = "🏆" if i == 0 else f"#{i+1}"
            print(f"{framework:<20} {vendor:<12} {avg_throughput:>10.1f} {rank_indicator}")
        
        print("\n" + "=" * 80)
        
        # Save text report
        report_path = os.path.join(self.output_dir, "demo_summary_report.txt")
        with open(report_path, 'w') as f:
            f.write("Multi-Vendor ML Inference Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")
            for framework, vendor, avg_throughput in framework_rankings:
                f.write(f"{framework}: {avg_throughput:.1f} samples/sec ({vendor})\n")
        
        return report_path


def main():
    """Demo main function"""
    print("Multi-Vendor ML Inference Benchmark Demo")
    print("This demo simulates benchmark results when actual ML frameworks are not available")
    print("")
    
    # Create demo benchmark
    benchmark = DemoMultiVendorBenchmark()
    
    # Run demo with different scenarios
    scenarios = [
        {
            "name": "CPU Frameworks Comparison",
            "frameworks": ["onnxruntime_cpu", "pytorch_cpu", "tensorflow_cpu", "openvino_cpu", "tvm_llvm"],
            "batch_size": 1
        },
        {
            "name": "GPU Acceleration Comparison", 
            "frameworks": ["onnxruntime_cuda", "pytorch_cuda", "tensorrt"],
            "batch_size": 1
        },
        {
            "name": "TVM Auto-Tuning Impact",
            "frameworks": ["tvm_llvm", "tvm_llvm_tuned"],
            "batch_size": 1
        }
    ]
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\n{'='*20} {scenario['name']} {'='*20}")
        
        results = benchmark.run_demo_benchmark(
            frameworks=scenario["frameworks"],
            batch_size=scenario["batch_size"]
        )
        
        all_results.extend(results)
        
        # Save scenario results
        scenario_file = f"demo_{scenario['name'].lower().replace(' ', '_')}_results.json"
        benchmark.save_results(results, scenario_file)
    
    # Generate overall summary
    benchmark.save_results(all_results, "demo_all_results.json")
    benchmark.generate_summary_report(all_results)
    
    print(f"\nDemo completed! Results saved to {benchmark.output_dir}")
    print(f"Total tests run: {len(all_results)}")


if __name__ == "__main__":
    main()