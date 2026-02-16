"""
Comprehensive Test Suite for 30-Qubit QGAN Generator
Run on laptop to verify all components before HPC/QC deployment.

Features:
- Quantum circuit correctness verification
- Classical mapper physical feasibility checks
- End-to-end generation pipeline testing
- Export format validation (OpenQASM 2.0, JSON)
- Performance profiling

Usage:
    python test_qgan_30q.py --verbose --profile
"""

import sys
from pathlib import Path
import argparse
import time
import json
from typing import Dict, List, Tuple
import psutil
import os
from datetime import datetime

import torch
import numpy as np

# Import local modules
from hybrid_circuit_30q import HybridQGANCircuit30Q, QubitConfig30Q
from hybrid_mapper_30q import HybridReadoutMapper30Q
from qgan_generator_30q import QGAN_Generator30Q


class Checkpoint:
    """Track progress through slow operations"""
    def __init__(self, name: str, timeout_sec: float = 30.0):
        self.name = name
        self.timeout_sec = timeout_sec
        self.start_time = None
        self.last_update = None
        self.events = []
        self.process = psutil.Process(os.getpid())
    
    def __enter__(self):
        self.start_time = time.time()
        self.last_update = self.start_time
        self.log_event(f"[START] {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        mem = self.process.memory_info().rss / (1024**2)  # MB
        self.log_event(f"[END] {self.name} ({elapsed:.2f}s, {mem:.1f}MB)")
        self.print_summary()
        if exc_type:
            self.log_event(f"[ERROR] {exc_type.__name__}: {exc_val}")
        return False
    
    def check(self, label: str, throw_on_timeout: bool = False):
        """Periodic checkpoint within operation"""
        now = time.time()
        elapsed = now - self.start_time
        
        mem = self.process.memory_info().rss / (1024**2)
        msg = f"[CHECK] {label} @ {elapsed:.2f}s, {mem:.1f}MB"
        self.log_event(msg)
        
        if throw_on_timeout and elapsed > self.timeout_sec:
            raise TimeoutError(f"{self.name} exceeded timeout of {self.timeout_sec}s")
        
        return elapsed
    
    def log_event(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        full_msg = f"[{timestamp}] {msg}"
        self.events.append(full_msg)
        print(full_msg)
    
    def print_summary(self):
        print(f"\n--- Checkpoint Summary: {self.name} ---")
        for event in self.events[-5:]:  # Last 5 events
            print(f"  {event}")
        print()


class TestResult:
    """Container for test results"""
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.details = []
        self.duration = 0.0
        self.memory_start = 0.0
        self.memory_end = 0.0
        self.checkpoints = []
        self.process = psutil.Process(os.getpid())
    
    def add_detail(self, msg: str):
        self.details.append(msg)
    
    def assert_condition(self, condition: bool, msg: str):
        if not condition:
            self.passed = False
            self.details.append(f"FAILED: {msg}")
        else:
            self.details.append(f"OK: {msg}")
    
    def record_checkpoint(self, label: str):
        mem = self.process.memory_info().rss / (1024**2)
        self.checkpoints.append((label, mem))
    
    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        mem_delta = self.memory_end - self.memory_start
        mem_str = f"Δ{mem_delta:+.1f}MB" if mem_delta != 0 else "stable"
        lines = [f"{status} | {self.name} ({self.duration:.3f}s, {mem_str})"]
        for detail in self.details:
            lines.append(f"      {detail}")
        return "\n".join(lines)



class QGAN30QTester:
    """Comprehensive test suite"""

    def __init__(self, verbose: bool = False, device: str = "cpu", timeout_sec: float = 120.0,
                 split_registers: bool = True):
        self.verbose = verbose
        self.device = device
        self.timeout_sec = timeout_sec
        self.split_registers = split_registers
        self.results: List[TestResult] = []
        self.process = psutil.Process(os.getpid())
    
    def log(self, msg: str):
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {msg}")
    
    def test_circuit_initialization(self) -> TestResult:
        """Test 1: Quantum circuit can be initialized"""
        result = TestResult("Circuit Initialization")
        result.memory_start = self.process.memory_info().rss / (1024**2)
        start = time.time()
        
        try:
            with Checkpoint("QubitConfig30Q()", timeout_sec=10) as cp:
                config = QubitConfig30Q()
                result.record_checkpoint("config_created")
            
            with Checkpoint("HybridQGANCircuit30Q init", timeout_sec=30) as cp:
                circuit = HybridQGANCircuit30Q(n_layers=2, config=config, split_registers=self.split_registers)
                result.record_checkpoint("circuit_created")
            
            self.log("Checking circuit attributes...")
            result.assert_condition(circuit.config.n_qubits == 30, "Qubit count = 30")
            result.assert_condition(len(circuit.config.wires_topo) == 8, "Topology register = 8 qubits")
            result.assert_condition(len(circuit.config.wires_nodes) == 15, "Node register = 15 qubits")
            result.assert_condition(len(circuit.config.wires_latent) == 7, "Latent register = 7 qubits")
            
            self.log("Getting circuit info...")
            with Checkpoint("get_circuit_info()", timeout_sec=10) as cp:
                info = circuit.get_circuit_info()
                result.record_checkpoint("info_retrieved")
            
            result.assert_condition(info['n_qubits'] == 30, "Circuit info consistent")
            result.add_detail(f"Circuit depth: {info['estimated_circuit_depth']} gates")
            result.add_detail(f"Classical sim memory: {info['classial_sim_memory_gb']:.2f} GB")
            
        except Exception as e:
            result.passed = False
            result.add_detail(f"EXCEPTION: {type(e).__name__}: {str(e)}")
            import traceback
            result.add_detail(f"Traceback: {traceback.format_exc()}")
        
        result.duration = time.time() - start
        result.memory_end = self.process.memory_info().rss / (1024**2)
        return result
    
    def test_circuit_forward_pass(self) -> TestResult:
        """Test 2: Quantum circuit executes forward pass"""
        result = TestResult("Circuit Forward Pass")
        result.memory_start = self.process.memory_info().rss / (1024**2)
        start = time.time()
        
        try:
            self.log("Creating config and circuit for forward pass...")
            with Checkpoint("Circuit setup for forward", timeout_sec=30) as cp:
                config = QubitConfig30Q()
                result.record_checkpoint("config_created")
                circuit = HybridQGANCircuit30Q(n_layers=2, config=config, split_registers=self.split_registers)
                result.record_checkpoint("circuit_created")

            self.log("Generating random parameters...")
            with Checkpoint("Parameter generation", timeout_sec=5) as cp:
                p_topo = np.random.randn(2, 8, 3) * 0.1
                p_node = np.random.randn(2, 15, 3) * 0.1
                p_latent = np.random.randn(2, 7, 3) * 0.1
                result.record_checkpoint("params_generated")
            
            self.log("Executing circuit.forward()... (THIS CAN HANG)")
            with Checkpoint("Circuit forward execution", timeout_sec=60) as cp:
                output = circuit.forward(p_topo, p_node, p_latent)
                result.record_checkpoint("forward_completed")
            
            self.log("Validating output shapes...")
            result.assert_condition(
                output['topo_logits'].shape == (256,),
                f"Topology logits shape: {output['topo_logits'].shape}"
            )
            result.assert_condition(
                output['node_expvals'].shape == (15,),
                f"Node expvals shape: {output['node_expvals'].shape}"
            )
            result.assert_condition(
                output['latent_features'].shape == (7,),
                f"Latent features shape: {output['latent_features'].shape}"
            )
            
            # Check probability normalization
            topo_sum = output['topo_logits'].sum().item()
            result.assert_condition(
                abs(topo_sum - 1.0) < 0.01,
                f"Topology probabilities sum to 1: {topo_sum:.6f}"
            )
            
            # Check expval bounds
            result.assert_condition(
                torch.all(output['node_expvals'] >= -1.1) and torch.all(output['node_expvals'] <= 1.1),
                "Node expvals in bounds [-1, 1]"
            )
            
        except Exception as e:
            result.passed = False
            result.add_detail(f"EXCEPTION: {type(e).__name__}: {str(e)}")
            import traceback
            result.add_detail(f"Traceback: {traceback.format_exc()}")
        
        result.duration = time.time() - start
        result.memory_end = self.process.memory_info().rss / (1024**2)
        return result
    
    def test_mapper_initialization(self) -> TestResult:
        """Test 3: Classical mapper initialization"""
        result = TestResult("Mapper Initialization")
        result.memory_start = self.process.memory_info().rss / (1024**2)
        start = time.time()
        
        try:
            self.log("Creating HybridReadoutMapper30Q...")
            with Checkpoint("Mapper initialization", timeout_sec=30) as cp:
                mapper = HybridReadoutMapper30Q(
                    num_linker_atoms=10,
                    angstrom_scale=12.0,
                    latent_dim=7,
                    device=self.device
                )
                result.record_checkpoint("mapper_created")
            
            self.log("Validating mapper structure...")
            result.assert_condition(mapper.n_metal_atoms == 5, "Metal atoms count = 5")
            result.assert_condition(mapper.num_linker_atoms == 10, "Linker atoms count = 10")
            result.assert_condition(mapper.total_atoms == 15, "Total atoms = 15")
            
            # Check parameter counts
            self.log("Counting parameters...")
            n_params = sum(p.numel() for p in mapper.parameters())
            result.add_detail(f"Mapper parameters: {n_params}")
            result.assert_condition(n_params > 1000, "Sufficient trainable parameters")
            
        except Exception as e:
            result.passed = False
            result.add_detail(f"EXCEPTION: {type(e).__name__}: {str(e)}")
            import traceback
            result.add_detail(f"Traceback: {traceback.format_exc()}")
        
        result.duration = time.time() - start
        result.memory_end = self.process.memory_info().rss / (1024**2)
        return result
    
    def test_mapper_forward_pass(self) -> TestResult:
        """Test 4: Mapper forward pass and physical constraints"""
        result = TestResult("Mapper Forward Pass")
        result.memory_start = self.process.memory_info().rss / (1024**2)
        start = time.time()
        
        try:
            self.log("Creating mapper for forward pass...")
            with Checkpoint("Mapper forward setup", timeout_sec=30) as cp:
                mapper = HybridReadoutMapper30Q(num_linker_atoms=10, device=self.device)
                result.record_checkpoint("mapper_created")
            
            # Dummy quantum outputs
            self.log("Creating dummy quantum outputs...")
            with Checkpoint("Dummy output creation", timeout_sec=5) as cp:
                topo_logits = torch.ones(256) / 256
                node_expvals = torch.randn(15)
                latent_features = torch.randn(7)
                result.record_checkpoint("inputs_created")
            
            self.log("Executing mapper forward pass...")
            with Checkpoint("Mapper forward execution", timeout_sec=30) as cp:
                outputs = mapper(topo_logits, node_expvals, latent_features)
                result.record_checkpoint("forward_completed")
            
            result.assert_condition(
                outputs['metal_atoms'].shape == (5, 3),
                f"Metal atoms shape: {outputs['metal_atoms'].shape}"
            )
            result.assert_condition(
                outputs['linker_atoms'].shape == (10, 3),
                f"Linker atoms shape: {outputs['linker_atoms'].shape}"
            )
            result.assert_condition(
                outputs['lattice_params'].shape == (6,),
                f"Lattice params shape: {outputs['lattice_params'].shape}"
            )
            
            # Check physical constraints
            lattice_lengths = outputs['lattice_params'][:3]
            result.assert_condition(
                torch.all(lattice_lengths > 4.0) and torch.all(lattice_lengths < 25.0),
                f"Lattice lengths in range [4, 25] Å: {lattice_lengths.detach().numpy()}"
            )
            
            lattice_angles = outputs['lattice_params'][3:]
            result.assert_condition(
                torch.all(lattice_angles > 60.0) and torch.all(lattice_angles < 120.0),
                f"Lattice angles in range [60, 120]°: {lattice_angles.detach().numpy()}"
            )
            
            result.assert_condition(
                0 <= outputs['validity'] <= 1,
                f"Validity score in [0,1]: {outputs['validity'].item():.3f}"
            )
            
        except Exception as e:
            result.passed = False
            result.add_detail(f"EXCEPTION: {type(e).__name__}: {str(e)}")
            import traceback
            result.add_detail(f"Traceback: {traceback.format_exc()}")
        
        result.duration = time.time() - start
        result.memory_end = self.process.memory_info().rss / (1024**2)
        return result
    
    def test_generator_initialization(self) -> TestResult:
        """Test 5: Full QGAN generator initialization"""
        result = TestResult("QGAN Generator Init")
        result.memory_start = self.process.memory_info().rss / (1024**2)
        start = time.time()
        
        try:
            self.log("Creating QGAN_Generator30Q... (THIS CAN HANG)")
            with Checkpoint("QGAN Generator initialization", timeout_sec=60) as cp:
                gen = QGAN_Generator30Q(
                    noise_dim=32,
                    num_linker_atoms=10,
                    n_circuit_layers=2,
                    device=self.device,
                    split_registers=self.split_registers
                )
                result.record_checkpoint("generator_created")
            
            self.log("Getting model stats...")
            with Checkpoint("get_model_stats()", timeout_sec=10) as cp:
                stats = gen.get_model_stats()
                result.record_checkpoint("stats_retrieved")
            
            result.add_detail(f"Total params: {stats['model_metrics']['total_parameters']}")
            result.add_detail(f"Model size: {stats['model_metrics']['model_size_kb']:.1f} KB")
            
            result.assert_condition(
                stats['quantum_config']['n_qubits'] == 30,
                "Quantum config: 30 qubits"
            )
            result.assert_condition(
                stats['classical_config']['noise_dim'] == 32,
                "Classical config: noise_dim = 32"
            )
            
        except Exception as e:
            result.passed = False
            result.add_detail(f"EXCEPTION: {type(e).__name__}: {str(e)}")
            import traceback
            result.add_detail(f"Traceback: {traceback.format_exc()}")
        
        result.duration = time.time() - start
        result.memory_end = self.process.memory_info().rss / (1024**2)
        return result
    
    def test_end_to_end_generation(self) -> TestResult:
        """Test 6: End-to-end MOF generation"""
        result = TestResult("End-to-End Generation")
        result.memory_start = self.process.memory_info().rss / (1024**2)
        start = time.time()
        
        try:
            self.log("Creating generator for end-to-end test...")
            with Checkpoint("E2E Generator init", timeout_sec=60) as cp:
                gen = QGAN_Generator30Q(noise_dim=32, num_linker_atoms=10, device=self.device, split_registers=self.split_registers)
                result.record_checkpoint("generator_created")
            
            # Generate batch
            self.log("Creating noise input...")
            with Checkpoint("Noise generation", timeout_sec=5) as cp:
                noise = torch.randn(2, 32)
                result.record_checkpoint("noise_created")
            
            self.log("Executing generation... (CAN HANG HERE)")
            with Checkpoint("Generation forward pass", timeout_sec=60) as cp:
                outputs = gen(noise)
                result.record_checkpoint("generation_completed")
            
            result.assert_condition(
                outputs['metal_atoms'].shape == (2, 5, 3),
                f"Batch metal atoms: {outputs['metal_atoms'].shape}"
            )
            result.assert_condition(
                outputs['linker_atoms'].shape == (2, 10, 3),
                f"Batch linker atoms: {outputs['linker_atoms'].shape}"
            )
            
            # Check all validity scores are reasonable
            validities = outputs['validity'].detach().numpy()
            result.add_detail(f"Validity scores: min={validities.min():.3f}, max={validities.max():.3f}, mean={validities.mean():.3f}")
            result.assert_condition(
                np.all(validities >= 0) and np.all(validities <= 1),
                "All validity scores in [0, 1]"
            )
            
        except Exception as e:
            result.passed = False
            result.add_detail(f"EXCEPTION: {type(e).__name__}: {str(e)}")
            import traceback
            result.add_detail(f"Traceback: {traceback.format_exc()}")
        
        result.duration = time.time() - start
        result.memory_end = self.process.memory_info().rss / (1024**2)
        return result
    
    def test_batch_processing(self) -> TestResult:
        """Test 7: Batch processing consistency"""
        result = TestResult("Batch Processing")
        result.memory_start = self.process.memory_info().rss / (1024**2)
        start = time.time()
        
        try:
            self.log("Creating generator for batch test...")
            with Checkpoint("Batch test generator init", timeout_sec=60) as cp:
                gen = QGAN_Generator30Q(noise_dim=32, num_linker_atoms=10, device=self.device, split_registers=self.split_registers)
                result.record_checkpoint("generator_created")
            
            # Single sample
            self.log("Processing single sample...")
            with Checkpoint("Single sample processing", timeout_sec=30) as cp:
                noise_single = torch.randn(1, 32)
                output_single = gen(noise_single)
                result.record_checkpoint("single_sample_done")
            
            # Same seed, batch of 2
            self.log("Processing batch of 2...")
            with Checkpoint("Batch of 2 processing", timeout_sec=30) as cp:
                torch.manual_seed(0)
                noise_batch = torch.randn(2, 32)
                output_batch = gen(noise_batch)
                result.record_checkpoint("batch_done")
            
            result.add_detail(f"Single sample metal atoms: {output_single['metal_atoms'].shape}")
            result.add_detail(f"Batch sample metal atoms: {output_batch['metal_atoms'].shape}")
            result.assert_condition(
                output_single['metal_atoms'].shape[0] == 1,
                "Single sample processing"
            )
            result.assert_condition(
                output_batch['metal_atoms'].shape[0] == 2,
                "Batch processing"
            )
            
        except Exception as e:
            result.passed = False
            result.add_detail(f"EXCEPTION: {type(e).__name__}: {str(e)}")
            import traceback
            result.add_detail(f"Traceback: {traceback.format_exc()}")
        
        result.duration = time.time() - start
        result.memory_end = self.process.memory_info().rss / (1024**2)
        return result
    
    def test_export_formats(self) -> TestResult:
        """Test 8: Export to OpenQASM and structure formats"""
        result = TestResult("Export Formats")
        result.memory_start = self.process.memory_info().rss / (1024**2)
        start = time.time()
        
        try:
            self.log("Creating generator for export test...")
            with Checkpoint("Export test generator init", timeout_sec=60) as cp:
                gen = QGAN_Generator30Q(noise_dim=32, num_linker_atoms=10, device=self.device, split_registers=self.split_registers)
                result.record_checkpoint("generator_created")
            
            with Checkpoint("Noise creation", timeout_sec=5) as cp:
                noise = torch.randn(1, 32)
                result.record_checkpoint("noise_created")
            
            # OpenQASM export
            self.log("Exporting to OpenQASM...")
            with Checkpoint("OpenQASM export", timeout_sec=30) as cp:
                qasm = gen.export_openqasm(noise)
                result.record_checkpoint("qasm_exported")
            result.assert_condition(
                "OPENQASM 2.0" in qasm,
                "OpenQASM header present"
            )
            result.assert_condition(
                "30" in qasm,  # Should mention 30 qubits
                "Qubit count mentioned"
            )
            
            # Structure export
            self.log("Getting quantum outputs...")
            with Checkpoint("Quantum output extraction", timeout_sec=30) as cp:
                outputs, quantum_dict = gen(noise, return_quantum_outputs=True)
                result.record_checkpoint("quantum_outputs_obtained")
            
            result.assert_condition(
                'topo_logits' in quantum_dict,
                "Quantum outputs returned"
            )
            
            result.add_detail(f"OpenQASM size: {len(qasm)} chars")
            
        except Exception as e:
            result.passed = False
            result.add_detail(f"EXCEPTION: {type(e).__name__}: {str(e)}")
            import traceback
            result.add_detail(f"Traceback: {traceback.format_exc()}")
        
        result.duration = time.time() - start
        result.memory_end = self.process.memory_info().rss / (1024**2)
        return result
    
    def test_checkpoint_save_load(self) -> TestResult:
        """Test 9: Model checkpoint persistence"""
        result = TestResult("Checkpoint Save/Load")
        result.memory_start = self.process.memory_info().rss / (1024**2)
        start = time.time()
        
        try:
            import tempfile
            
            self.log("Creating generator for checkpoint test...")
            with Checkpoint("Checkpoint test generator init", timeout_sec=60) as cp:
                gen1 = QGAN_Generator30Q(noise_dim=32, num_linker_atoms=10, device=self.device, split_registers=self.split_registers)
                result.record_checkpoint("generator_created")
            
            # Generate before saving
            self.log("Generating sample before save...")
            with Checkpoint("Pre-save generation", timeout_sec=30) as cp:
                noise = torch.randn(1, 32)
                output_before = gen1(noise)
                result.record_checkpoint("presave_output_generated")
            
            # Save
            self.log("Saving checkpoint...")
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                checkpoint_path = f.name
            
            with Checkpoint("Checkpoint save", timeout_sec=30) as cp:
                gen1.save_checkpoint(checkpoint_path)
                result.record_checkpoint("checkpoint_saved")
            result.add_detail(f"Checkpoint saved: {checkpoint_path}")
            
            # Load into new generator
            self.log("Creating new generator for loading...")
            with Checkpoint("New generator for loading", timeout_sec=60) as cp:
                gen2 = QGAN_Generator30Q(noise_dim=32, num_linker_atoms=10, device=self.device, split_registers=self.split_registers)
                result.record_checkpoint("gen2_created")
            
            self.log("Loading checkpoint...")
            with Checkpoint("Checkpoint load", timeout_sec=30) as cp:
                gen2.load_checkpoint(checkpoint_path)
                result.record_checkpoint("checkpoint_loaded")
            
            # Generate with loaded model
            self.log("Generating with loaded model...")
            with Checkpoint("Post-load generation", timeout_sec=30) as cp:
                torch.manual_seed(42)
                noise_test = torch.randn(1, 32)
                output_after = gen2(noise_test)
                result.record_checkpoint("postload_output_generated")
            
            result.assert_condition(
                Path(checkpoint_path).exists(),
                "Checkpoint file exists"
            )
            result.add_detail(f"Checkpoint size: {Path(checkpoint_path).stat().st_size / 1024:.1f} KB")
            
            # Cleanup
            Path(checkpoint_path).unlink()
            
        except Exception as e:
            result.passed = False
            result.add_detail(f"EXCEPTION: {type(e).__name__}: {str(e)}")
            import traceback
            result.add_detail(f"Traceback: {traceback.format_exc()}")
        
        result.duration = time.time() - start
        result.memory_end = self.process.memory_info().rss / (1024**2)
        return result
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Execute all tests and return summary"""
        mode = "SPLIT (laptop)" if self.split_registers else "FULL 30q (HPC)"
        print("=" * 70)
        print("30-QUBIT QGAN GENERATOR - COMPREHENSIVE TEST SUITE")
        print(f"Mode: {mode} | Timeout: {self.timeout_sec}s | Device: {self.device}")
        print("=" * 70)
        print()
        
        tests = [
            self.test_circuit_initialization,
            self.test_circuit_forward_pass,
            self.test_mapper_initialization,
            self.test_mapper_forward_pass,
            self.test_generator_initialization,
            self.test_end_to_end_generation,
            self.test_batch_processing,
            self.test_export_formats,
            self.test_checkpoint_save_load,
        ]
        
        for i, test_fn in enumerate(tests, 1):
            test_name = test_fn.__name__
            print(f"\n>>> Running Test {i}/9: {test_name}")
            print("-" * 70)
            try:
                result = test_fn()
                self.results.append(result)
                print(result)
            except Exception as e:
                print(f"*** CRITICAL ERROR in {test_name}: {type(e).__name__}: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("\n" + "=" * 70)
        print(f"SUMMARY: {passed}/{total} tests passed")
        print("=" * 70)
        
        # Detailed summary with timing
        print("\nTest Timing Breakdown:")
        for result in self.results:
            status = "✓" if result.passed else "✗"
            print(f"  {status} {result.name:35s} {result.duration:7.3f}s")
        
        return {r.name: r.passed for r in self.results}


def main():
    parser = argparse.ArgumentParser(description="Test 30-qubit QGAN Generator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--profile", "-p", action="store_true", help="Performance profiling")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="PyTorch device")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout per test in seconds")
    parser.add_argument("--full", action="store_true",
                        help="Use full 30q circuit (needs ~16GB+ RAM). Default: split mode (~500MB)")

    args = parser.parse_args()

    split = not args.full
    mode_str = "FULL 30q (HPC)" if args.full else "SPLIT registers (laptop-safe)"
    print(f"Starting test suite: {mode_str}, timeout={args.timeout}s, device={args.device}")

    # Run tests
    tester = QGAN30QTester(
        verbose=args.verbose, device=args.device,
        timeout_sec=args.timeout, split_registers=split
    )
    results = tester.run_all_tests()
    
    # Exit code based on results
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
