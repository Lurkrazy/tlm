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
Integrated Multi-Vendor and TVM Meta Schedule Benchmark

This script provides a unified interface to run both TVM Meta Schedule benchmarks
and multi-vendor framework comparisons, enabling comprehensive performance evaluation
across the entire ML inference ecosystem.
"""

import argparse
import json
import os
import sys
import time
import logging
from typing import Dict, List, Optional

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from baseline_benchmark import BaselineBenchmark, BenchmarkResult
    TVM_BASELINE_AVAILABLE = True
except ImportError:
    TVM_BASELINE_AVAILABLE = False
    print("Warning: TVM baseline benchmark not available")

try:
    from multi_vendor_benchmark import MultiVendorBenchmark, VendorBenchmarkResult
    MULTI_VENDOR_AVAILABLE = True
except ImportError:
    MULTI_VENDOR_AVAILABLE = False
    print("Warning: Multi-vendor benchmark not available")

try:
    from demo_multi_vendor_benchmark import DemoMultiVendorBenchmark
    DEMO_AVAILABLE = True
except ImportError:
    DEMO_AVAILABLE = False


class IntegratedBenchmark:
    """Integrated benchmark coordinator for TVM and multi-vendor comparison"""
    
    def __init__(self, output_dir: str = "integrated_benchmark_results",
                 device: str = "cpu", target: str = "llvm"):
        self.output_dir = output_dir
        self.device = device
        self.target = target
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'integrated_benchmark.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize benchmarks
        self.tvm_benchmark = None
        self.multi_vendor_benchmark = None
        self.demo_benchmark = None
        
        self._initialize_benchmarks()
    
    def _initialize_benchmarks(self):
        """Initialize available benchmark systems"""
        
        # TVM Meta Schedule Baseline Benchmark
        if TVM_BASELINE_AVAILABLE:
            try:
                self.tvm_benchmark = BaselineBenchmark(
                    target=self.target,
                    output_dir=os.path.join(self.output_dir, "tvm_results"),
                    enable_tuning=True
                )
                self.logger.info("TVM Meta Schedule benchmark initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TVM benchmark: {e}")
        
        # Multi-Vendor Benchmark
        if MULTI_VENDOR_AVAILABLE:
            try:
                self.multi_vendor_benchmark = MultiVendorBenchmark(
                    output_dir=os.path.join(self.output_dir, "multi_vendor_results"),
                    device=self.device
                )
                self.logger.info("Multi-vendor benchmark initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize multi-vendor benchmark: {e}")
        
        # Demo Benchmark (fallback)
        if DEMO_AVAILABLE:
            try:
                self.demo_benchmark = DemoMultiVendorBenchmark(
                    output_dir=os.path.join(self.output_dir, "demo_results")
                )
                self.logger.info("Demo benchmark initialized as fallback")
            except Exception as e:
                self.logger.warning(f"Failed to initialize demo benchmark: {e}")
    
    def run_comprehensive_benchmark(self, models: List[str] = None,
                                   frameworks: List[str] = None,
                                   enable_tvm_tuning: bool = True,
                                   enable_multi_vendor: bool = True,
                                   benchmark_suite: str = "vision",
                                   warmup_iterations: int = 5,
                                   measurement_iterations: int = 20) -> Dict:
        """Run comprehensive benchmark across all available systems"""
        
        results = {
            "tvm_results": None,
            "multi_vendor_results": None,
            "demo_results": None,
            "comparison_analysis": None
        }
        
        self.logger.info("Starting comprehensive benchmark")
        self.logger.info(f"Models: {models or 'all'}")
        self.logger.info(f"Frameworks: {frameworks or 'all available'}")
        
        # Run TVM Meta Schedule Benchmark
        if enable_tvm_tuning and self.tvm_benchmark:
            self.logger.info("Running TVM Meta Schedule benchmark...")
            try:
                tvm_results = self.tvm_benchmark.run_benchmark(
                    workloads=models,
                    benchmark_suite=benchmark_suite
                )
                
                # Save TVM results
                tvm_output = self.tvm_benchmark.save_results(tvm_results, "tvm_baseline_results.json")
                results["tvm_results"] = tvm_results
                
                # Generate TVM HTML report
                self.tvm_benchmark.generate_html_report(tvm_results)
                
                self.logger.info(f"TVM benchmark completed: {len(tvm_results)} workloads")
                
            except Exception as e:
                self.logger.error(f"TVM benchmark failed: {e}")
        
        # Run Multi-Vendor Benchmark
        if enable_multi_vendor and self.multi_vendor_benchmark:
            self.logger.info("Running multi-vendor benchmark...")
            try:
                multi_vendor_results = self.multi_vendor_benchmark.benchmark_all_frameworks(
                    models=models,
                    frameworks=frameworks,
                    warmup_iterations=warmup_iterations,
                    measurement_iterations=measurement_iterations
                )
                
                # Save multi-vendor results
                self.multi_vendor_benchmark.save_results(multi_vendor_results, "multi_vendor_results.json")
                results["multi_vendor_results"] = multi_vendor_results
                
                # Generate multi-vendor HTML report
                self.multi_vendor_benchmark.generate_comparison_report(multi_vendor_results)
                
                self.logger.info(f"Multi-vendor benchmark completed: {len(multi_vendor_results)} tests")
                
            except Exception as e:
                self.logger.error(f"Multi-vendor benchmark failed: {e}")
        
        # Run Demo Benchmark (if real frameworks unavailable)
        if not results["multi_vendor_results"] and self.demo_benchmark:
            self.logger.info("Running demo benchmark as fallback...")
            try:
                demo_results = self.demo_benchmark.run_demo_benchmark(
                    models=models,
                    frameworks=frameworks
                )
                
                # Save demo results
                self.demo_benchmark.save_results(demo_results, "demo_results.json")
                results["demo_results"] = demo_results
                
                # Generate demo summary
                self.demo_benchmark.generate_summary_report(demo_results)
                
                self.logger.info(f"Demo benchmark completed: {len(demo_results)} tests")
                
            except Exception as e:
                self.logger.error(f"Demo benchmark failed: {e}")
        
        # Generate comprehensive comparison analysis
        results["comparison_analysis"] = self._generate_comparison_analysis(results)
        
        return results
    
    def _generate_comparison_analysis(self, results: Dict) -> Dict:
        """Generate comprehensive comparison analysis across all benchmark results"""
        
        analysis = {
            "summary": {},
            "tvm_analysis": {},
            "framework_rankings": {},
            "recommendations": []
        }
        
        # Analyze TVM results
        if results["tvm_results"]:
            tvm_results = results["tvm_results"]
            tuned_results = [r for r in tvm_results if r.speedup is not None]
            
            if tuned_results:
                avg_speedup = sum(r.speedup for r in tuned_results) / len(tuned_results)
                max_speedup = max(r.speedup for r in tuned_results)
                analysis["tvm_analysis"] = {
                    "total_workloads": len(tvm_results),
                    "tuned_workloads": len(tuned_results),
                    "average_speedup": avg_speedup,
                    "max_speedup": max_speedup,
                    "speedup_range": [min(r.speedup for r in tuned_results), max_speedup]
                }
        
        # Analyze multi-vendor or demo results
        vendor_results = results["multi_vendor_results"] or results["demo_results"]
        if vendor_results:
            # Group by framework
            framework_performance = {}
            for result in vendor_results:
                framework = result.framework if hasattr(result, 'framework') else getattr(result, 'framework', 'unknown')
                if framework not in framework_performance:
                    framework_performance[framework] = {
                        "runtimes": [],
                        "throughputs": [],
                        "memories": [],
                        "vendor": getattr(result, 'vendor', 'Unknown')
                    }
                
                framework_performance[framework]["runtimes"].append(result.runtime_ms_mean)
                framework_performance[framework]["throughputs"].append(result.throughput_samples_per_sec)
                framework_performance[framework]["memories"].append(result.memory_mb_peak)
            
            # Calculate framework rankings
            framework_rankings = []
            for framework, perf in framework_performance.items():
                avg_runtime = sum(perf["runtimes"]) / len(perf["runtimes"])
                avg_throughput = sum(perf["throughputs"]) / len(perf["throughputs"])
                avg_memory = sum(perf["memories"]) / len(perf["memories"])
                
                framework_rankings.append({
                    "framework": framework,
                    "vendor": perf["vendor"],
                    "avg_runtime_ms": avg_runtime,
                    "avg_throughput_samples_per_sec": avg_throughput,
                    "avg_memory_mb": avg_memory,
                    "tests_count": len(perf["runtimes"])
                })
            
            # Sort by throughput (higher is better)
            framework_rankings.sort(key=lambda x: x["avg_throughput_samples_per_sec"], reverse=True)
            analysis["framework_rankings"] = framework_rankings
        
        # Generate recommendations
        recommendations = []
        
        if analysis["tvm_analysis"]:
            tvm_analysis = analysis["tvm_analysis"]
            if tvm_analysis["average_speedup"] > 1.5:
                recommendations.append(
                    f"TVM Meta Schedule tuning provides significant speedup (avg {tvm_analysis['average_speedup']:.2f}x). "
                    "Consider using TVM for production deployment."
                )
            elif tvm_analysis["average_speedup"] > 1.2:
                recommendations.append(
                    f"TVM Meta Schedule tuning provides moderate speedup (avg {tvm_analysis['average_speedup']:.2f}x). "
                    "Evaluate against framework overhead."
                )
        
        if analysis["framework_rankings"]:
            top_framework = analysis["framework_rankings"][0]
            recommendations.append(
                f"Top performing framework: {top_framework['framework']} ({top_framework['vendor']}) "
                f"with {top_framework['avg_throughput_samples_per_sec']:.1f} samples/sec average throughput."
            )
            
            # Memory efficiency recommendation
            memory_efficient = min(analysis["framework_rankings"], key=lambda x: x["avg_memory_mb"])
            if memory_efficient != top_framework:
                recommendations.append(
                    f"Most memory efficient: {memory_efficient['framework']} "
                    f"({memory_efficient['avg_memory_mb']:.1f} MB average). "
                    "Consider for resource-constrained deployment."
                )
        
        analysis["recommendations"] = recommendations
        
        # Save analysis
        analysis_path = os.path.join(self.output_dir, "comprehensive_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def generate_integrated_report(self, results: Dict) -> str:
        """Generate integrated HTML report combining all benchmark results"""
        
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive ML Inference Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; margin-bottom: 30px; }
        .section { margin-bottom: 40px; }
        .summary { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #e9ecef; }
        .best { background-color: #d4edda; font-weight: bold; }
        .good { background-color: #d1ecf1; }
        .neutral { background-color: #fff3cd; }
        .recommendation { background-color: #e2e3e5; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .metric { display: inline-block; margin-right: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Comprehensive ML Inference Benchmark Report</h1>
        <p>Complete performance evaluation across TVM Meta Schedule and multi-vendor frameworks</p>
    </div>
"""
        
        # Add executive summary
        analysis = results.get("comparison_analysis", {})
        
        html_content += """
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="summary">
"""
        
        if analysis.get("tvm_analysis"):
            tvm_analysis = analysis["tvm_analysis"]
            html_content += f"""
            <h3>TVM Meta Schedule Performance</h3>
            <div class="metric"><strong>Workloads Tested:</strong> {tvm_analysis.get('total_workloads', 0)}</div>
            <div class="metric"><strong>Tuned Workloads:</strong> {tvm_analysis.get('tuned_workloads', 0)}</div>
            <div class="metric"><strong>Average Speedup:</strong> {tvm_analysis.get('average_speedup', 0):.2f}x</div>
            <div class="metric"><strong>Max Speedup:</strong> {tvm_analysis.get('max_speedup', 0):.2f}x</div>
"""
        
        if analysis.get("framework_rankings"):
            framework_count = len(analysis["framework_rankings"])
            top_framework = analysis["framework_rankings"][0]
            html_content += f"""
            <h3>Multi-Vendor Framework Comparison</h3>
            <div class="metric"><strong>Frameworks Tested:</strong> {framework_count}</div>
            <div class="metric"><strong>Top Performer:</strong> {top_framework['framework']} ({top_framework['vendor']})</div>
            <div class="metric"><strong>Best Throughput:</strong> {top_framework['avg_throughput_samples_per_sec']:.1f} samples/sec</div>
"""
        
        html_content += """
        </div>
    </div>
"""
        
        # Add recommendations
        if analysis.get("recommendations"):
            html_content += """
    <div class="section">
        <h2>Recommendations</h2>
"""
            for rec in analysis["recommendations"]:
                html_content += f'        <div class="recommendation">{rec}</div>\n'
            
            html_content += "    </div>\n"
        
        # Add TVM results table if available
        if results.get("tvm_results"):
            html_content += """
    <div class="section">
        <h2>TVM Meta Schedule Results</h2>
        <table>
            <tr>
                <th>Workload</th>
                <th>Type</th>
                <th>Baseline (ms)</th>
                <th>Tuned (ms)</th>
                <th>Speedup</th>
            </tr>
"""
            
            for result in results["tvm_results"]:
                speedup_class = ""
                speedup_text = "N/A"
                tuned_text = "N/A"
                
                if result.speedup:
                    speedup_text = f"{result.speedup:.2f}x"
                    tuned_text = f"{result.tuned_runtime_ms_mean:.2f}"
                    if result.speedup >= 2.0:
                        speedup_class = "best"
                    elif result.speedup >= 1.5:
                        speedup_class = "good"
                    elif result.speedup >= 1.2:
                        speedup_class = "neutral"
                
                html_content += f"""
            <tr>
                <td>{result.workload}</td>
                <td>{result.workload_metadata.get('type', 'unknown')}</td>
                <td>{result.runtime_ms_mean:.2f}</td>
                <td>{tuned_text}</td>
                <td class="{speedup_class}">{speedup_text}</td>
            </tr>
"""
            
            html_content += "        </table>\n    </div>\n"
        
        # Add framework comparison table
        if analysis.get("framework_rankings"):
            html_content += """
    <div class="section">
        <h2>Framework Performance Ranking</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Framework</th>
                <th>Vendor</th>
                <th>Avg Runtime (ms)</th>
                <th>Avg Throughput (samples/sec)</th>
                <th>Avg Memory (MB)</th>
                <th>Tests</th>
            </tr>
"""
            
            for i, framework in enumerate(analysis["framework_rankings"]):
                row_class = "best" if i == 0 else ("good" if i == 1 else ("neutral" if i == 2 else ""))
                
                html_content += f"""
            <tr class="{row_class}">
                <td>#{i + 1}</td>
                <td>{framework['framework']}</td>
                <td>{framework['vendor']}</td>
                <td>{framework['avg_runtime_ms']:.2f}</td>
                <td>{framework['avg_throughput_samples_per_sec']:.1f}</td>
                <td>{framework['avg_memory_mb']:.1f}</td>
                <td>{framework['tests_count']}</td>
            </tr>
"""
            
            html_content += "        </table>\n    </div>\n"
        
        html_content += """
</body>
</html>
"""
        
        # Save integrated report
        report_path = os.path.join(self.output_dir, "integrated_benchmark_report.html")
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Integrated report saved to {report_path}")
        return report_path
    
    def print_summary(self, results: Dict):
        """Print comprehensive benchmark summary"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE ML INFERENCE BENCHMARK SUMMARY")
        print("="*80)
        
        analysis = results.get("comparison_analysis", {})
        
        # TVM Summary
        if analysis.get("tvm_analysis"):
            tvm_analysis = analysis["tvm_analysis"]
            print(f"\nTVM Meta Schedule Results:")
            print(f"  Workloads tested: {tvm_analysis.get('total_workloads', 0)}")
            print(f"  Tuned workloads: {tvm_analysis.get('tuned_workloads', 0)}")
            print(f"  Average speedup: {tvm_analysis.get('average_speedup', 0):.2f}x")
            print(f"  Max speedup: {tvm_analysis.get('max_speedup', 0):.2f}x")
        
        # Framework Rankings
        if analysis.get("framework_rankings"):
            print(f"\nFramework Performance Rankings:")
            print(f"{'Rank':<6} {'Framework':<20} {'Vendor':<12} {'Throughput (samples/sec)':<20}")
            print("-" * 70)
            
            for i, framework in enumerate(analysis["framework_rankings"][:5]):  # Top 5
                rank_indicator = "🏆" if i == 0 else f"#{i+1}"
                print(f"{rank_indicator:<6} {framework['framework']:<20} {framework['vendor']:<12} {framework['avg_throughput_samples_per_sec']:<20.1f}")
        
        # Recommendations
        if analysis.get("recommendations"):
            print(f"\nKey Recommendations:")
            for i, rec in enumerate(analysis["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        print("="*80)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Integrated Multi-Vendor and TVM Meta Schedule Benchmark")
    
    # General options
    parser.add_argument("--output-dir", type=str, default="integrated_benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to use for benchmarking")
    parser.add_argument("--target", type=str, default="llvm",
                       help="TVM target specification")
    
    # Benchmark control
    parser.add_argument("--models", nargs="+",
                       help="Specific models to benchmark")
    parser.add_argument("--frameworks", nargs="+",
                       help="Specific frameworks to benchmark")
    parser.add_argument("--benchmark-suite", type=str, choices=["operators", "vision", "all"], default="vision",
                       help="TVM benchmark suite to run")
    
    # Measurement parameters
    parser.add_argument("--warmup-iterations", type=int, default=5,
                       help="Number of warmup iterations")
    parser.add_argument("--measurement-iterations", type=int, default=20,
                       help="Number of measurement iterations")
    
    # Feature flags
    parser.add_argument("--enable-tvm-tuning", action="store_true", default=True,
                       help="Enable TVM Meta Schedule tuning")
    parser.add_argument("--disable-tvm-tuning", action="store_false", dest="enable_tvm_tuning",
                       help="Disable TVM Meta Schedule tuning")
    parser.add_argument("--enable-multi-vendor", action="store_true", default=True,
                       help="Enable multi-vendor benchmarking")
    parser.add_argument("--disable-multi-vendor", action="store_false", dest="enable_multi_vendor",
                       help="Disable multi-vendor benchmarking")
    parser.add_argument("--generate-report", action="store_true",
                       help="Generate integrated HTML report")
    
    return parser.parse_args()


def main():
    """Main integrated benchmark execution"""
    args = parse_args()
    
    print("Integrated Multi-Vendor and TVM Meta Schedule Benchmark")
    print("=" * 60)
    
    # Create integrated benchmark
    benchmark = IntegratedBenchmark(
        output_dir=args.output_dir,
        device=args.device,
        target=args.target
    )
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(
        models=args.models,
        frameworks=args.frameworks,
        enable_tvm_tuning=args.enable_tvm_tuning,
        enable_multi_vendor=args.enable_multi_vendor,
        benchmark_suite=args.benchmark_suite,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations
    )
    
    # Generate integrated report if requested
    if args.generate_report:
        benchmark.generate_integrated_report(results)
    
    # Print summary
    benchmark.print_summary(results)
    
    print(f"\nBenchmark completed! Results saved to {args.output_dir}")
    
    return results


if __name__ == "__main__":
    main()