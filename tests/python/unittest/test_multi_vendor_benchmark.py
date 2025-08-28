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
Tests for Multi-Vendor ML Inference Benchmark

Comprehensive test suite covering framework implementations, benchmarking logic,
result generation, and error handling scenarios.
"""

import pytest
import tempfile
import os
import json
import shutil
import sys
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Add meta directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'meta'))

try:
    from multi_vendor_benchmark import (
        BenchmarkModel, VendorBenchmarkResult, InferenceFramework,
        ONNXRuntimeFramework, PyTorchFramework, TensorFlowFramework,
        OpenVINOFramework, TVMFramework, MultiVendorBenchmark
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


class MockInferenceFramework(InferenceFramework):
    """Mock framework for testing"""
    
    def __init__(self, name="mock", device="cpu", simulate_error=False):
        super().__init__(name, device)
        self.simulate_error = simulate_error
        self.load_time = 100.0  # ms
        self.inference_time = 50.0  # ms
        self.memory_usage = 200.0  # MB
    
    def load_model(self, model_path, input_shapes):
        if self.simulate_error:
            raise RuntimeError("Simulated model loading error")
        return {"model_path": model_path, "input_shapes": input_shapes}
    
    def warm_up(self, model, input_data, iterations=3):
        if self.simulate_error:
            raise RuntimeError("Simulated warmup error")
        return self.inference_time  # Return warmup time
    
    def run_inference(self, model, input_data):
        if self.simulate_error:
            raise RuntimeError("Simulated inference error")
        
        # Simulate inference with random outputs
        outputs = {}
        for name, data in input_data.items():
            output_shape = [data.shape[0], 1000]  # Classification output
            outputs[f"output_{name}"] = np.random.randn(*output_shape).astype(np.float32)
        return outputs
    
    def get_model_info(self, model):
        return {"framework": self.name, "device": self.device}


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_model():
    """Create sample benchmark model"""
    return BenchmarkModel(
        name="test_model",
        description="Test Model for Unit Tests",
        model_url="https://example.com/test_model.onnx",
        input_shapes={"data": [1, 3, 224, 224]},
        data_type="float32",
        category="vision"
    )


@pytest.fixture
def mock_model_file(temp_dir):
    """Create mock ONNX model file"""
    model_path = os.path.join(temp_dir, "test_model.onnx")
    # Create a simple binary file to simulate ONNX model
    with open(model_path, 'wb') as f:
        f.write(b"mock onnx model data" * 100)
    return model_path


class TestBenchmarkModel:
    """Test BenchmarkModel dataclass"""
    
    def test_benchmark_model_creation(self):
        """Test creating BenchmarkModel instance"""
        model = BenchmarkModel(
            name="resnet18",
            description="ResNet-18 Classification Model",
            model_url="https://example.com/resnet18.onnx",
            input_shapes={"input": [1, 3, 224, 224]},
            data_type="float32",
            category="vision"
        )
        
        assert model.name == "resnet18"
        assert model.description == "ResNet-18 Classification Model"
        assert model.input_shapes["input"] == [1, 3, 224, 224]
        assert model.data_type == "float32"
        assert model.category == "vision"


class TestVendorBenchmarkResult:
    """Test VendorBenchmarkResult dataclass"""
    
    def test_result_creation(self):
        """Test creating benchmark result"""
        result = VendorBenchmarkResult(
            vendor="Test",
            framework="test_framework",
            model="test_model",
            hardware="test_hardware",
            runtime_ms_mean=50.0,
            runtime_ms_std=2.5,
            memory_mb_peak=200.0,
            throughput_samples_per_sec=20.0,
            model_load_time_ms=100.0,
            first_inference_ms=75.0,
            warmup_iterations=5,
            measurement_iterations=20,
            batch_size=1
        )
        
        assert result.vendor == "Test"
        assert result.runtime_ms_mean == 50.0
        assert result.throughput_samples_per_sec == 20.0
        assert result.batch_size == 1
    
    def test_result_serialization(self):
        """Test result serialization to dict"""
        result = VendorBenchmarkResult(
            vendor="Test",
            framework="test_framework", 
            model="test_model",
            hardware="test_hardware",
            runtime_ms_mean=50.0,
            runtime_ms_std=2.5,
            memory_mb_peak=200.0,
            throughput_samples_per_sec=20.0,
            model_load_time_ms=100.0,
            first_inference_ms=75.0,
            warmup_iterations=5,
            measurement_iterations=20,
            batch_size=1
        )
        
        result_dict = asdict(result)
        assert isinstance(result_dict, dict)
        assert result_dict["vendor"] == "Test"
        assert result_dict["runtime_ms_mean"] == 50.0


class TestMockInferenceFramework:
    """Test mock inference framework"""
    
    def test_mock_framework_creation(self):
        """Test creating mock framework"""
        framework = MockInferenceFramework("test_mock", "cpu")
        assert framework.name == "test_mock"
        assert framework.device == "cpu"
    
    def test_mock_framework_operations(self, sample_model, mock_model_file):
        """Test mock framework operations"""
        framework = MockInferenceFramework("test_mock", "cpu")
        
        # Test model loading
        model = framework.load_model(mock_model_file, sample_model.input_shapes)
        assert model["model_path"] == mock_model_file
        
        # Test inference
        input_data = {"data": np.random.randn(1, 3, 224, 224).astype(np.float32)}
        outputs = framework.run_inference(model, input_data)
        assert "output_data" in outputs
        assert outputs["output_data"].shape == (1, 1000)
        
        # Test model info
        info = framework.get_model_info(model)
        assert info["framework"] == "test_mock"
    
    def test_mock_framework_error_simulation(self, sample_model, mock_model_file):
        """Test error simulation in mock framework"""
        framework = MockInferenceFramework("test_mock", "cpu", simulate_error=True)
        
        with pytest.raises(RuntimeError, match="Simulated model loading error"):
            framework.load_model(mock_model_file, sample_model.input_shapes)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason=f"Import failed: {IMPORT_ERROR if not IMPORT_SUCCESS else ''}")
class TestInferenceFrameworkIntegration:
    """Test actual inference framework implementations"""
    
    def test_onnxruntime_framework_creation(self):
        """Test ONNX Runtime framework creation"""
        framework = ONNXRuntimeFramework(device="cpu")
        assert framework.name == "onnxruntime"
        assert framework.device == "cpu"
        assert framework.execution_provider == "CPUExecutionProvider"
    
    def test_pytorch_framework_creation(self):
        """Test PyTorch framework creation"""
        framework = PyTorchFramework(device="cpu")
        assert framework.name == "pytorch"
        assert framework.device == "cpu"
    
    def test_tensorflow_framework_creation(self):
        """Test TensorFlow framework creation"""
        framework = TensorFlowFramework(device="cpu")
        assert framework.name == "tensorflow"
        assert framework.device == "cpu"
    
    def test_openvino_framework_creation(self):
        """Test OpenVINO framework creation"""
        framework = OpenVINOFramework(device="cpu")
        assert framework.name == "openvino"
        assert framework.device == "cpu"
    
    def test_tvm_framework_creation(self):
        """Test TVM framework creation"""
        framework = TVMFramework(device="cpu", target="llvm")
        assert framework.name == "tvm"
        assert framework.device == "cpu"
        assert framework.target_str == "llvm"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason=f"Import failed: {IMPORT_ERROR if not IMPORT_SUCCESS else ''}")
class TestMultiVendorBenchmark:
    """Test MultiVendorBenchmark class"""
    
    def test_benchmark_initialization(self, temp_dir):
        """Test benchmark initialization"""
        benchmark = MultiVendorBenchmark(output_dir=temp_dir, device="cpu")
        
        assert benchmark.output_dir == temp_dir
        assert benchmark.device == "cpu"
        assert os.path.exists(benchmark.model_cache_dir)
        assert isinstance(benchmark.frameworks, dict)
    
    def test_get_benchmark_models(self, temp_dir):
        """Test getting benchmark models"""
        benchmark = MultiVendorBenchmark(output_dir=temp_dir, device="cpu")
        models = benchmark._get_benchmark_models()
        
        assert len(models) >= 3  # At least ResNet, MobileNet, SqueezeNet
        assert all(isinstance(model, BenchmarkModel) for model in models)
        assert any(model.name == "resnet18_onnx" for model in models)
        assert any(model.name == "mobilenetv2_onnx" for model in models)
    
    @patch('urllib.request.urlretrieve')
    def test_model_download(self, mock_download, temp_dir, sample_model):
        """Test model downloading with mocked download"""
        benchmark = MultiVendorBenchmark(output_dir=temp_dir, device="cpu")
        
        # Create mock model file after download
        def create_mock_file(url, path):
            with open(path, 'wb') as f:
                f.write(b"mock onnx model")
        
        mock_download.side_effect = create_mock_file
        
        model_path = benchmark._download_model(sample_model)
        
        assert os.path.exists(model_path)
        assert mock_download.called
        mock_download.assert_called_once()
    
    def test_model_cache_usage(self, temp_dir, sample_model, mock_model_file):
        """Test using cached model instead of downloading"""
        benchmark = MultiVendorBenchmark(output_dir=temp_dir, device="cpu")
        
        # Copy mock file to cache location
        cached_path = os.path.join(benchmark.model_cache_dir, f"{sample_model.name}.onnx")
        shutil.copy2(mock_model_file, cached_path)
        
        with patch('urllib.request.urlretrieve') as mock_download:
            model_path = benchmark._download_model(sample_model)
            
            assert model_path == cached_path
            assert not mock_download.called  # Should not download if cached
    
    def test_framework_initialization(self, temp_dir):
        """Test framework initialization with mocked dependencies"""
        with patch.multiple(
            'multi_vendor_benchmark',
            ONNXRUNTIME_AVAILABLE=True,
            TORCH_AVAILABLE=True,
            TENSORFLOW_AVAILABLE=True,
            TVM_AVAILABLE=True,
            OPENVINO_AVAILABLE=True
        ):
            benchmark = MultiVendorBenchmark(output_dir=temp_dir, device="cpu")
            frameworks = benchmark._initialize_frameworks()
            
            # Should have multiple frameworks when all are available
            expected_frameworks = [
                "onnxruntime_cpu", "pytorch_cpu", "tensorflow_cpu", 
                "tensorflow_lite", "openvino_cpu", "tvm_llvm", "tvm_llvm_tuned"
            ]
            
            for framework_name in expected_frameworks:
                assert framework_name in frameworks
    
    def test_save_results(self, temp_dir):
        """Test saving benchmark results"""
        benchmark = MultiVendorBenchmark(output_dir=temp_dir, device="cpu")
        
        # Create sample results
        results = [
            VendorBenchmarkResult(
                vendor="Test1",
                framework="test_framework_1",
                model="test_model",
                hardware="test_hardware",
                runtime_ms_mean=50.0,
                runtime_ms_std=2.5,
                memory_mb_peak=200.0,
                throughput_samples_per_sec=20.0,
                model_load_time_ms=100.0,
                first_inference_ms=75.0,
                warmup_iterations=5,
                measurement_iterations=20,
                batch_size=1
            ),
            VendorBenchmarkResult(
                vendor="Test2",
                framework="test_framework_2",
                model="test_model",
                hardware="test_hardware",
                runtime_ms_mean=45.0,
                runtime_ms_std=1.8,
                memory_mb_peak=180.0,
                throughput_samples_per_sec=22.2,
                model_load_time_ms=120.0,
                first_inference_ms=65.0,
                warmup_iterations=5,
                measurement_iterations=20,
                batch_size=1
            )
        ]
        
        output_path = benchmark.save_results(results, "test_results.json")
        
        assert os.path.exists(output_path)
        
        # Verify saved content
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
        
        assert len(saved_data) == 2
        assert saved_data[0]["vendor"] == "Test1"
        assert saved_data[1]["vendor"] == "Test2"
        assert saved_data[0]["runtime_ms_mean"] == 50.0
    
    def test_generate_comparison_report(self, temp_dir):
        """Test generating HTML comparison report"""
        benchmark = MultiVendorBenchmark(output_dir=temp_dir, device="cpu")
        
        # Create sample results with different performance
        results = [
            VendorBenchmarkResult(
                vendor="Fast",
                framework="fast_framework",
                model="resnet18_onnx",
                hardware="test_hardware",
                runtime_ms_mean=30.0,
                runtime_ms_std=1.5,
                memory_mb_peak=150.0,
                throughput_samples_per_sec=33.3,
                model_load_time_ms=80.0,
                first_inference_ms=45.0,
                warmup_iterations=5,
                measurement_iterations=20,
                batch_size=1
            ),
            VendorBenchmarkResult(
                vendor="Slow",
                framework="slow_framework", 
                model="resnet18_onnx",
                hardware="test_hardware",
                runtime_ms_mean=60.0,
                runtime_ms_std=3.0,
                memory_mb_peak=250.0,
                throughput_samples_per_sec=16.7,
                model_load_time_ms=150.0,
                first_inference_ms=90.0,
                warmup_iterations=5,
                measurement_iterations=20,
                batch_size=1
            )
        ]
        
        report_path = benchmark.generate_comparison_report(results)
        
        assert os.path.exists(report_path)
        assert report_path.endswith(".html")
        
        # Verify HTML content
        with open(report_path, 'r') as f:
            html_content = f.read()
        
        assert "Multi-Vendor ML Inference Benchmark Report" in html_content
        assert "fast_framework" in html_content
        assert "slow_framework" in html_content
        assert "30.00" in html_content  # Fast framework runtime
        assert "60.00" in html_content  # Slow framework runtime


class TestBenchmarkingIntegration:
    """Integration tests for full benchmarking workflow"""
    
    def test_mock_framework_benchmarking(self, temp_dir, sample_model, mock_model_file):
        """Test full benchmarking with mock framework"""
        # Create mock benchmark that uses our mock framework
        benchmark = MultiVendorBenchmark(output_dir=temp_dir, device="cpu")
        
        # Replace frameworks with mock
        mock_framework = MockInferenceFramework("mock_test", "cpu")
        benchmark.frameworks = {"mock_test": mock_framework}
        
        # Mock model download to use our test file
        with patch.object(benchmark, '_download_model', return_value=mock_model_file):
            with patch.object(benchmark, '_get_benchmark_models', return_value=[sample_model]):
                results = benchmark.benchmark_all_frameworks(
                    models=[sample_model.name],
                    frameworks=["mock_test"],
                    warmup_iterations=2,
                    measurement_iterations=5,
                    batch_size=1
                )
        
        assert len(results) == 1
        result = results[0]
        
        assert result.framework == "mock_test"
        assert result.model == sample_model.name
        assert result.runtime_ms_mean > 0
        assert result.throughput_samples_per_sec > 0
        assert result.warmup_iterations == 2
        assert result.measurement_iterations == 5
    
    def test_error_handling_in_benchmarking(self, temp_dir, sample_model, mock_model_file):
        """Test error handling during benchmarking"""
        benchmark = MultiVendorBenchmark(output_dir=temp_dir, device="cpu")
        
        # Create mock framework that simulates errors
        error_framework = MockInferenceFramework("error_framework", "cpu", simulate_error=True)
        working_framework = MockInferenceFramework("working_framework", "cpu", simulate_error=False)
        
        benchmark.frameworks = {
            "error_framework": error_framework,
            "working_framework": working_framework
        }
        
        with patch.object(benchmark, '_download_model', return_value=mock_model_file):
            with patch.object(benchmark, '_get_benchmark_models', return_value=[sample_model]):
                results = benchmark.benchmark_all_frameworks(
                    models=[sample_model.name],
                    frameworks=["error_framework", "working_framework"],
                    warmup_iterations=2,
                    measurement_iterations=5
                )
        
        # Should only have results from working framework
        assert len(results) == 1
        assert results[0].framework == "working_framework"
    
    def test_batch_size_handling(self, temp_dir, sample_model, mock_model_file):
        """Test different batch sizes"""
        benchmark = MultiVendorBenchmark(output_dir=temp_dir, device="cpu")
        mock_framework = MockInferenceFramework("mock_test", "cpu")
        benchmark.frameworks = {"mock_test": mock_framework}
        
        batch_sizes = [1, 4, 8]
        all_results = []
        
        for batch_size in batch_sizes:
            with patch.object(benchmark, '_download_model', return_value=mock_model_file):
                with patch.object(benchmark, '_get_benchmark_models', return_value=[sample_model]):
                    results = benchmark.benchmark_all_frameworks(
                        models=[sample_model.name],
                        frameworks=["mock_test"],
                        warmup_iterations=2,
                        measurement_iterations=5,
                        batch_size=batch_size
                    )
            
            assert len(results) == 1
            assert results[0].batch_size == batch_size
            all_results.extend(results)
        
        assert len(all_results) == len(batch_sizes)
        
        # Throughput should generally increase with batch size
        throughputs = [r.throughput_samples_per_sec for r in all_results]
        assert throughputs[1] > throughputs[0]  # batch_size 4 > batch_size 1
        assert throughputs[2] > throughputs[1]  # batch_size 8 > batch_size 4


class TestCommandLineInterface:
    """Test command line interface"""
    
    @patch('multi_vendor_benchmark.MultiVendorBenchmark')
    def test_main_function_basic(self, mock_benchmark_class):
        """Test main function with basic arguments"""
        mock_benchmark = Mock()
        mock_benchmark.benchmark_all_frameworks.return_value = []
        mock_benchmark_class.return_value = mock_benchmark
        
        # Mock sys.argv
        test_args = ["multi_vendor_benchmark.py", "--device", "cpu"]
        
        with patch('sys.argv', test_args):
            from multi_vendor_benchmark import parse_args, main
            
            args = parse_args()
            assert args.device == "cpu"
            assert args.batch_size == 1
            assert args.warmup_iterations == 5
            assert args.measurement_iterations == 20
    
    @patch('multi_vendor_benchmark.MultiVendorBenchmark')
    def test_main_function_with_options(self, mock_benchmark_class):
        """Test main function with various options"""
        mock_benchmark = Mock()
        mock_benchmark.benchmark_all_frameworks.return_value = []
        mock_benchmark_class.return_value = mock_benchmark
        
        test_args = [
            "multi_vendor_benchmark.py",
            "--device", "cuda",
            "--batch-size", "8", 
            "--warmup-iterations", "10",
            "--measurement-iterations", "50",
            "--models", "resnet18_onnx", "mobilenetv2_onnx",
            "--frameworks", "onnxruntime_cpu", "pytorch_cpu",
            "--generate-report"
        ]
        
        with patch('sys.argv', test_args):
            from multi_vendor_benchmark import parse_args
            
            args = parse_args()
            assert args.device == "cuda"
            assert args.batch_size == 8
            assert args.warmup_iterations == 10
            assert args.measurement_iterations == 50
            assert args.models == ["resnet18_onnx", "mobilenetv2_onnx"]
            assert args.frameworks == ["onnxruntime_cpu", "pytorch_cpu"]
            assert args.generate_report == True


# Mock mode tests for CI environments
class TestMockMode:
    """Test benchmark in mock mode (when dependencies unavailable)"""
    
    def test_mock_mode_functionality(self, temp_dir):
        """Test that benchmark works in mock mode"""
        # Simulate missing dependencies
        with patch.multiple(
            'multi_vendor_benchmark',
            ONNXRUNTIME_AVAILABLE=False,
            TORCH_AVAILABLE=False,
            TENSORFLOW_AVAILABLE=False,
            TVM_AVAILABLE=False,
            OPENVINO_AVAILABLE=False
        ):
            benchmark = MultiVendorBenchmark(output_dir=temp_dir, device="cpu")
            frameworks = benchmark._initialize_frameworks()
            
            # Should have no frameworks when none are available
            assert len(frameworks) == 0
    
    def test_partial_availability(self, temp_dir):
        """Test with partial framework availability"""
        with patch.multiple(
            'multi_vendor_benchmark',
            ONNXRUNTIME_AVAILABLE=True,
            TORCH_AVAILABLE=False,
            TENSORFLOW_AVAILABLE=True,
            TVM_AVAILABLE=False,
            OPENVINO_AVAILABLE=False
        ):
            benchmark = MultiVendorBenchmark(output_dir=temp_dir, device="cpu")
            frameworks = benchmark._initialize_frameworks()
            
            # Should only have ONNX Runtime and TensorFlow frameworks
            expected_frameworks = ["onnxruntime_cpu", "tensorflow_cpu", "tensorflow_lite"]
            assert len(frameworks) >= 2
            for fw_name in expected_frameworks:
                assert fw_name in frameworks


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])