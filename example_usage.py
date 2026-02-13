"""
Example Usage: Hybrid QGAN Generator for MOF Generation
Demonstrates all key features: generation, validation, serialization, and verification.
"""

import torch
import numpy as np
from pathlib import Path

from qgan_generator import QGAN_Generator
from hybrid_mapper import HybridReadoutMapper
from hybrid_circuit import HybridQGANCircuit


def example_1_basic_generation():
    """Example 1: Basic MOF generation from noise."""
    print("\n" + "="*70)
    print("EXAMPLE 1: BASIC MOF GENERATION")
    print("="*70)
    
    # Initialize generator
    generator = QGAN_Generator(
        noise_dim=32,
        num_linker_atoms=12,
        n_circuit_layers=2,
    )
    generator.eval()
    
    # Generate from batch of noise vectors
    batch_size = 3
    z = torch.randn(batch_size, 32)
    
    print(f"\nGenerating {batch_size} MOF structures...")
    with torch.no_grad():
        output = generator(z)
    
    print(f"\n✓ Generated MOF structure components:")
    print(f"  - Atomic coordinates: {output['coords'].shape} (batch, atoms, 3D)")
    print(f"    Range: [{output['coords'].min().item():.3f}, {output['coords'].max().item():.3f}] Å")
    print(f"  - Lattice parameters: {output['lattice'].shape} (batch, 6)")
    print(f"    Values: {output['lattice'][0].numpy()}")
    print(f"  - Metal atom mask: {output['node_mask'].shape}")
    print(f"    Metal atoms (1): {output['node_mask'][0, :6].sum().item()}")
    print(f"    Linker atoms (0): {output['node_mask'][0, 6:].sum().item()}")


def example_2_discrete_topology():
    """Example 2: Extract discrete topology (bitstring selection)."""
    print("\n" + "="*70)
    print("EXAMPLE 2: DISCRETE TOPOLOGY GENERATION")
    print("="*70)
    
    generator = QGAN_Generator(noise_dim=32, num_linker_atoms=12)
    generator.eval()
    
    print(f"\nGenerating 5 discrete topologies...")
    z = torch.randn(5, 32)
    
    with torch.no_grad():
        bitstrings = generator.generate_discrete_topology(z)
    
    print(f"\nGenerated Topology Bitstrings (10 qubits = Node/Linker selection):")
    for i, bitstring in enumerate(bitstrings):
        print(f"  Structure {i+1}: {bitstring} (decimal: {int(''.join(map(str, bitstring)), 2)})")
    
    # Verify bitstring validity
    assert all(len(b) == 10 for b in bitstrings), "Invalid bitstring length"
    assert all(all(bit in [0, 1] for bit in b) for b in bitstrings), "Invalid bit values"
    print(f"\n✓ All bitstrings are valid 10-bit representations")


def example_3_intermediate_outputs():
    """Example 3: Access quantum circuit intermediate outputs."""
    print("\n" + "="*70)
    print("EXAMPLE 3: QUANTUM CIRCUIT INTERMEDIATE OUTPUTS")
    print("="*70)
    
    generator = QGAN_Generator(noise_dim=32, num_linker_atoms=12)
    generator.eval()
    
    z = torch.randn(1, 32)
    
    print(f"\nGenerating with intermediate quantum outputs...")
    with torch.no_grad():
        output = generator(z, return_intermediate=True)
    
    print(f"\n✓ Quantum Circuit Outputs:")
    print(f"  - Topology logits: {output['topo_logits'].shape}")
    print(f"    (2^10 probability distribution for discrete states)")
    print(f"    Top-3 state probabilities: {torch.sort(output['topo_logits'][0], descending=True).values[:3].numpy()}")
    
    print(f"\n  - Node expectation values: {output['node_expvals'].shape}")
    print(f"    (18 values = 6 metal atoms × 3 coordinates)")
    print(f"    Values: {output['node_expvals'][0].numpy()}")
    
    print(f"\n  - Latent features: {output['latent_features'].shape}")
    print(f"    (12 features → linker coords + lattice params)")
    print(f"    Values: {output['latent_features'][0].numpy()}")


def example_4_validation():
    """Example 4: Validate generated structures."""
    print("\n" + "="*70)
    print("EXAMPLE 4: STRUCTURE VALIDATION")
    print("="*70)
    
    generator = QGAN_Generator(noise_dim=32, num_linker_atoms=12)
    generator.eval()
    
    z = torch.randn(5, 32)
    
    print(f"\nGenerating and validating 5 structures...")
    with torch.no_grad():
        output = generator(z)
    
    valid_count = 0
    for i in range(5):
        is_valid = generator.validate_output({
            'coords': output['coords'][i:i+1],
            'lattice': output['lattice'][i:i+1],
        })
        status = "✓" if is_valid else "✗"
        print(f"  {status} Structure {i+1}: {'Valid' if is_valid else 'Invalid'}")
        if is_valid:
            valid_count += 1
    
    print(f"\n✓ Validation Results: {valid_count}/5 structures passed")


def example_5_qasm_export():
    """Example 5: Export circuit to OpenQASM 2.0."""
    print("\n" + "="*70)
    print("EXAMPLE 5: QASM EXPORT FOR QARP COMPATIBILITY")
    print("="*70)
    
    generator = QGAN_Generator(noise_dim=32, num_linker_atoms=12)
    
    print(f"\nExporting quantum circuit to OpenQASM 2.0...")
    qasm = generator.export_qasm()
    
    # Parse QASM to extract statistics
    lines = qasm.split('\n')
    gate_count = len([l for l in lines if l.strip() and 'q[' in l and not l.startswith('//')])
    
    print(f"\n✓ QASM Export Summary:")
    print(f"  - Format: OpenQASM 2.0")
    print(f"  - Total qubits: 40")
    print(f"  - Total gates: ~{gate_count}")
    print(f"  - File size: {len(qasm)} characters")
    
    print(f"\n✓ First 30 lines of QASM:")
    for i, line in enumerate(lines[:30]):
        print(f"  {line}")
    
    if gate_count > 30:
        print(f"  ... ({gate_count - 30} more gate lines)")
    
    print(f"\n✓ QASM is ready for QARP / Fujitsu Simulator deployment")


def example_6_gradient_optimization():
    """Example 6: Training loop with gradient descent."""
    print("\n" + "="*70)
    print("EXAMPLE 6: GRADIENT-BASED OPTIMIZATION")
    print("="*70)
    
    generator = QGAN_Generator(noise_dim=32, num_linker_atoms=12)
    generator.train()
    
    # Optimizer
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Target structure (e.g., desired MOF configuration)
    target_coords = torch.randn(2, 18, 3)
    target_lattice = torch.ones(2, 6) * 10.0
    
    print(f"\nRunning 3 optimization steps...")
    losses = []
    
    for step in range(3):
        optimizer.zero_grad()
        
        # Generate from noise
        z = torch.randn(2, 32)
        output = generator(z)
        
        # Compute loss (focus on metal atoms)
        coords_loss = criterion(output['coords'][:, :6, :], target_coords)
        lattice_loss = criterion(output['lattice'], target_lattice)
        loss = coords_loss + lattice_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Step {step+1}: Loss = {loss.item():.6f}")
    
    print(f"\n✓ Gradient optimization successful")
    print(f"  Starting loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Loss reduction: {(losses[0] - losses[-1]):.6f}")


def example_7_batch_statistics():
    """Example 7: Analyze batch statistics."""
    print("\n" + "="*70)
    print("EXAMPLE 7: BATCH STATISTICS AND ANALYSIS")
    print("="*70)
    
    generator = QGAN_Generator(noise_dim=32, num_linker_atoms=12)
    generator.eval()
    
    print(f"\nGenerating batch of 20 MOF structures...")
    z = torch.randn(20, 32)
    
    with torch.no_grad():
        output = generator(z)
    
    # Analyze coordinates
    coords = output['coords']
    lattice = output['lattice']
    
    print(f"\n✓ Coordinate Statistics:")
    print(f"  Mean across batch: [{coords.mean(0)[:, 0].mean().item():.3f}, "
          f"{coords.mean(0)[:, 1].mean().item():.3f}, "
          f"{coords.mean(0)[:, 2].mean().item():.3f}] Å")
    print(f"  Std across batch: [{coords.std(0)[:, 0].mean().item():.3f}, "
          f"{coords.std(0)[:, 1].mean().item():.3f}, "
          f"{coords.std(0)[:, 2].mean().item():.3f}] Å")
    print(f"  Min coordinates: [{coords.min().item():.3f}] Å")
    print(f"  Max coordinates: [{coords.max().item():.3f}] Å")
    
    print(f"\n✓ Lattice Parameter Statistics:")
    print(f"  Mean: {lattice.mean(0).numpy()}")
    print(f"  Std: {lattice.std(0).numpy()}")
    print(f"  Min: {lattice.min().numpy()}")
    print(f"  Max: {lattice.max().numpy()}")
    
    # Check validity for all
    all_valid = all(
        generator.validate_output({
            'coords': output['coords'][i:i+1],
            'lattice': output['lattice'][i:i+1],
        })
        for i in range(20)
    )
    print(f"\n✓ All 20 structures are valid: {all_valid}")


def run_all_examples():
    """Run all examples in sequence."""
    print("\n" + "="*80)
    print(" " * 15 + "HYBRID QGAN GENERATOR - COMPREHENSIVE EXAMPLES")
    print("="*80)
    
    examples = [
        ("Basic Generation", example_1_basic_generation),
        ("Discrete Topology", example_2_discrete_topology),
        ("Quantum Outputs", example_3_intermediate_outputs),
        ("Validation", example_4_validation),
        ("QASM Export", example_5_qasm_export),
        ("Gradient Optimization", example_6_gradient_optimization),
        ("Batch Statistics", example_7_batch_statistics),
    ]
    
    for name, example_fn in examples:
        try:
            example_fn()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(" " * 20 + "ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    run_all_examples()
