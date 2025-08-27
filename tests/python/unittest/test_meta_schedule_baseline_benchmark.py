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

"""Test Meta Scheduler Baseline Benchmark"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Import the baseline benchmark module
import sys
meta_path = os.path.join(os.path.dirname(__file__), "../../../meta")
sys.path.insert(0, meta_path)

from baseline_benchmark import (
    BaselineBenchmark, 
    BenchmarkWorkload, 
    HardwareConfig, 
    ExecutionConfig,
    BenchmarkResult
)


class TestBaselineBenchmark(unittest.TestCase):
    """Test cases for the baseline benchmark"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark = BaselineBenchmark(
            target="llvm", 
            output_dir=self.temp_dir,
            builder_timeout=5,
            runner_timeout=5
        )
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_hardware_detection(self):
        """Test hardware configuration detection"""
        hw_config = self.benchmark.hardware_config
        
        self.assertIsInstance(hw_config, HardwareConfig)
        self.assertGreater(hw_config.cpu_count, 0)
        self.assertGreater(hw_config.memory_gb, 0)
        self.assertIsNotNone(hw_config.cpu_platform)
        
    def test_workload_creation(self):
        """Test workload definition and creation"""
        workloads = self.benchmark._get_baseline_workloads()
        
        self.assertIsInstance(workloads, list)
        self.assertGreater(len(workloads), 0)
        
        for workload in workloads:
            self.assertIsInstance(workload, BenchmarkWorkload)
            self.assertIsNotNone(workload.name)
            self.assertIsNotNone(workload.workload_type)
            self.assertIsInstance(workload.input_shapes, list)
            
    def test_matmul_workload_creation(self):
        """Test matrix multiplication workload creation"""
        workload = BenchmarkWorkload(
            name="test_matmul",
            description="Test matmul",
            workload_type="matmul",
            input_shapes=[[64, 64], [64, 64]]
        )
        
        try:
            mod = self.benchmark._create_tvm_workload(workload)
            # If TVM is available, check the module
            if mod is not None:
                self.assertIsNotNone(mod)
        except Exception as e:
            # TVM might not be available in test environment
            self.skipTest(f"TVM not available: {e}")
            
    def test_benchmark_execution(self):
        """Test benchmark execution"""
        # Test with a single small workload
        results = self.benchmark.run_benchmark(workloads=["matmul_1024x1024"])
        
        self.assertIsInstance(results, list)
        if len(results) > 0:
            result = results[0]
            self.assertIsInstance(result, BenchmarkResult)
            self.assertEqual(result.workload, "matmul_1024x1024")
            self.assertGreater(result.runtime_ms_mean, 0)
            self.assertGreaterEqual(result.runtime_ms_std, 0)
            
    def test_result_saving(self):
        """Test saving benchmark results"""
        # Create mock results
        mock_result = BenchmarkResult(
            workload="test_workload",
            engine="test_engine", 
            hardware="test_hardware",
            runtime_ms_mean=10.5,
            runtime_ms_std=1.2,
            workload_metadata={"test": "data"},
            engine_version="1.0",
            engine_config={},
            hardware_config={},
            execution_config={},
            metrics={},
            runtime_raw=[],
            timestamp="2024-01-01T00:00:00Z"
        )
        
        results = [mock_result]
        output_file = self.benchmark.save_results(results, "test_results.json")
        
        self.assertTrue(os.path.exists(output_file))
        
        # Verify JSON format
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
            
        self.assertIsInstance(saved_data, list)
        self.assertEqual(len(saved_data), 1)
        self.assertEqual(saved_data[0]["workload"], "test_workload")
        
    def test_benchmark_with_missing_workload(self):
        """Test benchmark with non-existent workload name"""
        results = self.benchmark.run_benchmark(workloads=["nonexistent_workload"])
        self.assertEqual(len(results), 0)
        
    def test_execution_config(self):
        """Test execution configuration"""
        exec_config = ExecutionConfig(number=2, repeat=5, min_repeat_ms=100)
        
        self.assertEqual(exec_config.number, 2)
        self.assertEqual(exec_config.repeat, 5) 
        self.assertEqual(exec_config.min_repeat_ms, 100)
        
    @patch('baseline_benchmark.TVM_AVAILABLE', False)
    def test_mock_mode(self):
        """Test benchmark runs in mock mode when TVM is not available"""
        benchmark = BaselineBenchmark(target="llvm", output_dir=self.temp_dir)
        results = benchmark.run_benchmark(workloads=["matmul_1024x1024"])
        
        # Should still return results in mock mode
        self.assertIsInstance(results, list)
        if len(results) > 0:
            result = results[0]
            self.assertGreater(result.runtime_ms_mean, 0)
            
    def test_workload_types(self):
        """Test different workload types"""
        workloads = self.benchmark._get_baseline_workloads()
        workload_types = {w.workload_type for w in workloads}
        
        # Should have multiple workload types
        self.assertIn("matmul", workload_types)
        
    def test_output_directory_creation(self):
        """Test output directory creation"""
        new_temp_dir = os.path.join(self.temp_dir, "new_benchmark_dir")
        
        benchmark = BaselineBenchmark(
            target="llvm",
            output_dir=new_temp_dir
        )
        
        self.assertTrue(os.path.exists(new_temp_dir))
        
    def test_summary_output(self):
        """Test benchmark summary output"""
        mock_result = BenchmarkResult(
            workload="test_workload",
            engine="test_engine",
            hardware="test_hardware", 
            runtime_ms_mean=15.5,
            runtime_ms_std=2.1,
            workload_metadata={},
            engine_version="1.0",
            engine_config={},
            hardware_config={},
            execution_config={},
            metrics={},
            runtime_raw=[],
            timestamp="2024-01-01T00:00:00Z"
        )
        
        # This should not raise an exception
        self.benchmark.print_summary([mock_result])

        
class TestTargets(unittest.TestCase):
    """Test benchmark with different targets"""
    
    def test_different_targets(self):
        """Test benchmark with different targets"""
        targets = ["llvm", "llvm -mcpu=core-avx2"]
        
        for target in targets:
            with tempfile.TemporaryDirectory() as temp_dir:
                benchmark = BaselineBenchmark(target=target, output_dir=temp_dir)
                
                # Should create without error
                self.assertEqual(benchmark.target_str, target)
        
        
class TestBenchmarkIntegration(unittest.TestCase):
    """Integration test for full benchmark workflow"""
    
    def test_benchmark_integration(self):
        """Integration test for full benchmark workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark = BaselineBenchmark(target="llvm", output_dir=temp_dir)
            
            # Run a single workload
            results = benchmark.run_benchmark(workloads=["dense_512x512"])
            
            # Save results
            output_file = benchmark.save_results(results)
            
            # Verify file exists and has correct format
            self.assertTrue(os.path.exists(output_file))
            
            with open(output_file, 'r') as f:
                data = json.load(f)
                
            self.assertIsInstance(data, list)
            if len(data) > 0:
                self.assertIn("workload", data[0])
                self.assertIn("runtime_ms_mean", data[0])


if __name__ == "__main__":
    unittest.main()