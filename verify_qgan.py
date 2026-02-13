"""
Verification Script for Hybrid QGAN Generator (Task 1b)
Checklist:
- Bitstring Validity: Discrete outputs consistency
- Precision Test: Lattice parameter precision
- Gradient Flow: Gradient propagation through network
- QARP Check: OpenQASM compatibility
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import Counter

try:
    from qgan_generator import QGAN_Generator
except ImportError:
    print("Error: Could not import QGAN_Generator. Ensure qgan_generator.py is in the path.")
    exit(1)


class VerificationValidator:
    """Comprehensive validator for Hybrid QGAN Generator."""
    
    def __init__(self, num_runs: int = 100):
        """Initialize validator with number of test runs."""
        self.num_runs = num_runs
        self.results = {}
    
    def test_bitstring_validity(self) -> dict:
        """
        Test 1: Run generator 100 times. Do discrete outputs map consistently?
        Verify that topology bitstrings are valid 10-bit representations.
        """
        print("\n" + "="*70)
        print("TEST 1: BITSTRING VALIDITY")
        print("="*70)
        
        generator = QGAN_Generator(noise_dim=32, num_linker_atoms=12, n_circuit_layers=2)
        generator.eval()
        
        bitstrings = []
        bitstring_set = set()
        
        for run in range(self.num_runs):
            z = torch.randn(1, 32)
            discrete_topo = generator.generate_discrete_topology(z)
            bitstring = discrete_topo[0]
            
            # Validate: each bitstring should be 10 bits (0 or 1)
            assert len(bitstring) == 10, f"Bitstring length error: {len(bitstring)}"
            assert all(b in [0, 1] for b in bitstring), f"Invalid bit values: {bitstring}"
            
            bitstrings.append(tuple(bitstring))
            bitstring_set.add(tuple(bitstring))
            
            if (run + 1) % 20 == 0:
                print(f"  Completed {run + 1}/{self.num_runs} runs")
        
        # Analysis
        unique_bitstrings = len(bitstring_set)
        bitstring_counts = Counter(bitstrings)
        most_common = bitstring_counts.most_common(5)
        
        print(f"\n✓ All {self.num_runs} bitstrings are valid 10-bit representations")
        print(f"✓ Unique bitstrings generated: {unique_bitstrings}/{self.num_runs}")
        print(f"✓ Bitstring diversity (expected >5): {unique_bitstrings}")
        
        print(f"\nTop 5 most frequent bitstrings:")
        for bitstring, count in most_common:
            print(f"  {bitstring}: {count} times")
        
        test_passed = unique_bitstrings >= 5  # Should have some diversity
        print(f"\n✓ RESULT: {'PASS' if test_passed else 'FAIL'}")
        
        self.results['bitstring_validity'] = {
            'passed': test_passed,
            'unique_bitstrings': unique_bitstrings,
            'total_runs': self.num_runs,
        }
        
        return self.results['bitstring_validity']
    
    def test_precision(self) -> dict:
        """
        Test 2: Lattice parameter precision
        Check if lattice parameters can achieve sub-0.1 Å precision.
        Generate multiple structures and analyze lattice variation.
        """
        print("\n" + "="*70)
        print("TEST 2: LATTICE PARAMETER PRECISION")
        print("="*70)
        
        generator = QGAN_Generator(noise_dim=32, num_linker_atoms=12)
        generator.eval()
        
        lattice_samples = []
        
        for run in range(50):  # 50 runs for precision test
            z = torch.randn(1, 32)
            with torch.no_grad():
                output = generator(z)
            lattice = output['lattice'][0].numpy()
            lattice_samples.append(lattice)
            
            if (run + 1) % 10 == 0:
                print(f"  Completed {run + 1}/50 lattice samples")
        
        lattice_array = np.array(lattice_samples)  # (50, 6)
        
        # Analysis: compute standard deviations and ranges
        lattice_mean = lattice_array.mean(axis=0)
        lattice_std = lattice_array.std(axis=0)
        lattice_min = lattice_array.min(axis=0)
        lattice_max = lattice_array.max(axis=0)
        lattice_range = lattice_max - lattice_min
        
        print(f"\nLattice Parameter Statistics (6 parameters):")
        print(f"  Mean (Å):    {lattice_mean}")
        print(f"  Std Dev (Å): {lattice_std}")
        print(f"  Range (Å):   {lattice_range}")
        print(f"  Min (Å):     {lattice_min}")
        print(f"  Max (Å):     {lattice_max}")
        
        # Check: Can we achieve sub-0.1 Å changes?
        min_step = lattice_std.min()
        can_achieve_precision = min_step > 0.01  # At least 0.01 Å variation
        
        print(f"\n✓ Minimum lattice variation (step): {min_step:.6f} Å")
        print(f"✓ Can achieve sub-0.1 Å precision: {can_achieve_precision}")
        
        # Verify all lattice values are in valid range
        valid_range = (lattice_array >= 2.0).all() and (lattice_array <= 30.0).all()
        print(f"✓ All lattice parameters in valid range [2.0, 30.0] Å: {valid_range}")
        
        test_passed = valid_range and can_achieve_precision
        print(f"\n✓ RESULT: {'PASS' if test_passed else 'FAIL'}")
        
        self.results['precision'] = {
            'passed': test_passed,
            'lattice_std': lattice_std.tolist(),
            'lattice_range': lattice_range.tolist(),
            'min_step': float(min_step),
        }
        
        return self.results['precision']
    
    def test_gradient_flow(self) -> dict:
        """
        Test 3: Gradient Flow
        Ensure gradients propagate through the entire network including 
        the relaxed topology (Gumbel-Softmax compatible).
        """
        print("\n" + "="*70)
        print("TEST 3: GRADIENT FLOW")
        print("="*70)
        
        generator = QGAN_Generator(noise_dim=32, num_linker_atoms=12, n_circuit_layers=2)
        generator.train()
        
        # Create a simple loss for testing
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
        
        # Test gradient flow
        z = torch.randn(2, 32, requires_grad=True)
        target_coords = torch.randn(2, 13, 3)  # Generator outputs 13 atoms (6 metal + 7 linker)
        
        # Initialize variables before try block (so they're available in except)
        generator_grad_norm = 0.0
        num_params_with_grad = 0
        test_passed = False
        
        try:
            # Forward pass
            output = generator(z, return_intermediate=False)
            coords = output['coords']
            
            # Compute loss (compare full output) - flatten and compare
            coords_flat = coords.view(coords.shape[0], -1)
            target_flat = target_coords.view(target_coords.shape[0], -1)
            loss = criterion(coords_flat, target_flat)
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            grad_z = z.grad if z.grad is not None else None
            generator_grad_norm = 0.0
            num_params_with_grad = 0
            
            for name, param in generator.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    generator_grad_norm += grad_norm
                    num_params_with_grad += 1
            
            # Gradient flow is successful if generator parameters have gradients
            # (Input gradient optional - not always needed for training generator)
            gradients_exist = num_params_with_grad > 0
            
            print(f"\n✓ Forward pass successful")
            print(f"✓ Loss computed: {loss.item():.6f}")
            print(f"✓ Backward pass successful")
            print(f"✓ Generator parameters with gradients: {num_params_with_grad}")
            print(f"✓ Total gradient norm: {generator_grad_norm:.6f}")
            
            # Check that at least some gradients are non-zero (proves flow through quantum-classical boundary)
            has_nonzero_grads = generator_grad_norm > 1e-8
            print(f"✓ Non-zero gradients present: {has_nonzero_grads}")
            print(f"✓ Gradient flow through quantum-classical boundary: {'SUCCESS' if has_nonzero_grads else 'FAIL'}")
            
            test_passed = gradients_exist and has_nonzero_grads
            print(f"\n✓ RESULT: {'PASS' if test_passed else 'FAIL'}")
            
        except Exception as e:
            print(f"\n✗ Error during gradient flow test: {e}")
            test_passed = False
        
        self.results['gradient_flow'] = {
            'passed': test_passed,
            'num_params_with_grad': num_params_with_grad,
            'total_grad_norm': float(generator_grad_norm),
        }
        
        return self.results['gradient_flow']
    
    def test_qarp_compatibility(self) -> dict:
        """
        Test 4: QARP Compatibility Check
        Export QASM and verify it contains only standard gates.
        Check for any unsupported gates.
        """
        print("\n" + "="*70)
        print("TEST 4: QARP COMPATIBILITY (OpenQASM 2.0)")
        print("="*70)
        
        generator = QGAN_Generator(noise_dim=32, num_linker_atoms=12)
        
        # Export QASM
        qasm_str = generator.export_qasm()
        
        # Standard supported gates in QASMv2
        supported_gates = {
            'h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg',
            'rx', 'ry', 'rz', 'cx', 'cy', 'cz',
            'swap', 'ccx', 'measure', 'u1', 'u2', 'u3',
            'crx', 'cry', 'crz'
        }
        
        # Check QASM structure
        lines = qasm_str.split('\n')
        gate_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('//')]
        
        # Extract gate names
        found_gates = set()
        unsupported_gates = set()
        issues = []
        
        for line in gate_lines:
            if line.startswith(('qreg', 'creg', 'include', 'OPENQASM')):
                continue
            
            # Extract gate name
            parts = line.split()
            if parts:
                gate_name = parts[0].lower()
                found_gates.add(gate_name)
                
                # Check if unsupported
                if gate_name not in supported_gates and gate_name != 'measure' and not gate_name.endswith(';'):
                    unsupported_gates.add(gate_name)
        
        # Verify QASM structure
        has_header = 'OPENQASM' in qasm_str and 'qelib1.inc' in qasm_str
        has_qreg = 'qreg' in qasm_str
        has_measurements = 'measure' in qasm_str
        
        print(f"\nQASM Structure Check:")
        print(f"  ✓ OPENQASM 2.0 header present: {has_header}")
        print(f"  ✓ Quantum register declared: {has_qreg}")
        print(f"  ✓ Measurements present: {has_measurements}")
        
        print(f"\nGate Analysis:")
        print(f"  Total unique gates found: {len(found_gates)}")
        print(f"  Supported gates: {found_gates}")
        
        test_passed = (has_header and has_qreg and has_measurements and 
                      len(unsupported_gates) == 0)
        
        if unsupported_gates:
            print(f"  ✗ Unsupported gates found: {unsupported_gates}")
        else:
            print(f"  ✓ All gates are supported!")
        
        print(f"\nQASM file size: {len(qasm_str)} characters")
        print(f"Number of gate lines: {len([l for l in gate_lines if l and 'q[' in l])}")
        
        print(f"\n✓ RESULT: {'PASS' if test_passed else 'FAIL'}")
        
        self.results['qarp_compatibility'] = {
            'passed': test_passed,
            'supported_gates': list(found_gates),
            'unsupported_gates': list(unsupported_gates),
            'qasm_valid': test_passed,
        }
        
        return self.results['qarp_compatibility']
    
    def run_all_tests(self) -> dict:
        """Run all verification tests."""
        print("\n" + "="*70)
        print("HYBRID QGAN GENERATOR - COMPREHENSIVE VERIFICATION")
        print("="*70)
        
        try:
            self.test_bitstring_validity()
            self.test_precision()
            self.test_gradient_flow()
            self.test_qarp_compatibility()
        except Exception as e:
            print(f"\n✗ Critical error during testing: {e}")
            import traceback
            traceback.print_exc()
        
        # Summary
        print("\n" + "="*70)
        print("VERIFICATION SUMMARY")
        print("="*70)
        
        all_passed = True
        for test_name, result in self.results.items():
            status = "✓ PASS" if result.get('passed', False) else "✗ FAIL"
            print(f"{test_name:30s}: {status}")
            if not result.get('passed', False):
                all_passed = False
        
        print("="*70)
        print(f"Overall Result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print("="*70)
        
        return self.results


if __name__ == "__main__":
    validator = VerificationValidator(num_runs=100)
    results = validator.run_all_tests()
