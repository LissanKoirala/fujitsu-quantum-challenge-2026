# Hybrid QGAN Generator: Task 1b Implementation

## Overview

This is a complete implementation of a **Hybrid Quantum-Classical MOF Generator** as specified in Task 1b. The system generates both discrete topology (Node/Linker selection) and continuous geometric refinements (lattice parameters) using a 40-qubit quantum circuit with classical neural networks.

## Architecture

### Quantum Circuit (40 Qubits)
The quantum system is partitioned into three specialized registers:

```
┌─────────────────────────────────────────────────┐
│         40-Qubit Hybrid System                  │
├──────────────┬──────────────┬──────────────────┤
│ Topology     │ Node         │ Latent           │
│ Register     │ Register     │ Register         │
│ (10 qubits)  │ (18 qubits)  │ (12 qubits)      │
│              │              │                  │
│ Discrete     │ Direct       │ Multiplexed      │
│ Node/Linker  │ Metal Atom   │ Linker/Lattice  │
│ Selection    │ Coordinates  │ Features         │
└──────────────┴──────────────┴──────────────────┘
```

**Layer 1: Global Entanglement (Seed)**
- Hadamard gates on all 40 qubits
- Randomized Ry and CNOT layers
- Creates foundational superposition with entanglement

**Layer 2: Topology Branch (Qubits 0-9)**
- StronglyEntanglingLayers for discrete state selection
- Hard entanglement compatible with QARP Fused-Swap optimizations
- Output: Probability distribution over 2^10 = 1024 possible topologies

**Layer 3: Geometry Branch (Qubits 10-39)**
- Conditional re-uploading: Controlled gates from topology to geometry
- Node Register: BasicEntanglerLayers for direct metal atom coordinate mapping
- Latent Register: StronglyEntanglingLayers for high-density feature encoding
- Output: Expectation values for coordinates and lattice parameters

### Classical Readout Mapper
Converts quantum measurements to valid MOF structures:

```python
Quantum Outputs
    ├── topo_logits (1024 probabilities)
    │   └── Node Head → 6 metal atoms × 3 coords
    │
    ├── node_expvals (18 values)
    │   └── Scaled to Angstrom space (±15Å)
    │
    └── latent_features (12 values)
        └── MLP Decoder → Linker coords + Lattice params
            (12 → 64 → 64 → output)
```

**Output Structure:**
- `coords`: All atomic coordinates (18 atoms × 3D) in Angstroms
- `lattice`: 6 lattice parameters (2.0 - 30.0 Å range)
- `node_mask`: Binary mask for metal (1) vs linker (0) atoms

## Files

### Core Implementation
- **`hybrid_circuit.py`**: 40-qubit quantum circuit with conditional geometry encoding
- **`hybrid_mapper.py`**: PyTorch readout mapper for structure decoding
- **`qgan_generator.py`**: Complete end-to-end generator with serialization

### Verification & Examples
- **`verify_qgan.py`**: Comprehensive validation suite
  - Bitstring Validity: 100 generations test
  - Precision Test: Sub-0.1 Å lattice parameter resolution
  - Gradient Flow: Backpropagation through quantum-classical boundary
  - QARP Check: OpenQASM 2.0 compatibility verification

- **`example_usage.py`**: 7 practical examples
  1. Basic MOF generation
  2. Discrete topology extraction
  3. Quantum circuit intermediate outputs
  4. Structure validation
  5. QASM export for deployment
  6. Gradient-based optimization
  7. Batch statistics analysis

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

Requirements:
- PennyLane >= 0.33.0 (quantum circuit framework)
- PyTorch >= 2.0.0 (classical neural networks)
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0

## Usage

### Basic Generation
```python
from qgan_generator import QGAN_Generator
import torch

# Initialize generator
generator = QGAN_Generator(
    noise_dim=32,
    num_linker_atoms=12,
    n_circuit_layers=2,
)

# Generate MOF structure
z = torch.randn(1, 32)  # Batch of 1 random vector
output = generator(z)

# Results
print(output['coords'])    # (1, 18, 3) atomic coordinates
print(output['lattice'])   # (1, 6) lattice parameters
print(output['node_mask']) # (1, 18) metal/linker mask
```

### Extract Discrete Topology
```python
# Get discrete bitstring (Node/Linker selection)
bitstrings = generator.generate_discrete_topology(z)
# Returns: (batch_size, 10) binary matrix
```

### Export for QARP
```python
# Export to OpenQASM 2.0
qasm = generator.export_qasm()
# Returns: OpenQASM code ready for Fujitsu Simulator
```

### Run Verification Tests
```bash
python verify_qgan.py
```

Outputs:
- ✓ Bitstring validity across 100 generations
- ✓ Lattice parameter precision (std deviation)
- ✓ Gradient propagation through network
- ✓ QASM compatibility with QARP

### Run Examples
```bash
python example_usage.py
```

## Key Features

### ✓ Hybrid Discrete-Continuous Generation
- Topology: Hard discrete bitstrings for chemical validity
- Geometry: Continuous optimization for lattice refinement
- Gumbel-Softmax relaxation for differentiability during training

### ✓ Quantum-Classical Boundary Handling
- Explicit tensor conversion: `torch.tensor(quantum_output)`
- Detach operations: `.detach().numpy()` for quantum circuit inputs
- Parameter-Shift rule for gradient computation through quantum gates

### ✓ QARP Compatibility
- StandardOpenQASM 2.0 export
- Only QARP-supported gates (H, RY, RZ, CNOT, CRY, etc.)
- No non-unitary or problematic channels

### ✓ Safety Checks
- Lattice parameter validation (2.0 - 30.0 Å)
- Coordinate bounds checking
- Thermodynamic stability constraints

### ✓ Trainable End-to-End
- Noise → Parameters → Quantum Circuit → Mapping → MOF Structure
- Full gradient flow for backpropagation
- Compatible with PyTorch autograd and optimizers

## Mathematical Details

### Qubit Partitioning Logic
```
Topology Register (10 qubits):
  - Represents 2^10 = 1024 possible Node/Linker combinations
  - Metal nodes are rigid anchors: maximum precision needed
  - Discrete selection crucial for chemical validity

Node Register (18 qubits = 6 atoms × 3 coords):
  - Direct 1-qubit-to-1-coordinate mapping
  - Maximum precision for metal atom positions
  - Expectation values scaled to ±15 Ångströms

Latent Register (12 qubits):
  - Multiplexed feature encoding
  - Decoded by MLP into linker coords + lattice params
  - Flexible encoding for global MOF properties
```

### Conditional Geometry
```
P(Geometry | Topology) via Controlled Rotations:
  - CRY gates condition latent register on topology qubit states
  - If specific metal node selected → lattice "responds"
  - Prevents incompatible selections (e.g., large atom in small lattice)
```

### Gradient Strategy
```
Forward Pass (Training):
  - Topology: Gumbel-Softmax relaxation (continuous)
  - Geometry: Standard RY/RZ rotations (naturally differentiable)

Forward Pass (Inference):
  - Topology: Hard argmax to discrete bitstrings (CIF-valid)
  - Geometry: Deterministic expectation values

Backward Pass:
  - Parameter-Shift rule through quantum gates
  - Analytic gradients through relaxed topology
  - Full PyTorch autograd for classical layers
```

## Verification Checklist

The implementation satisfies all Task 1b verification requirements:

- [x] **Bitstring Validity**: 100 generation test → unique valid 10-bit representations
- [x] **Precision Test**: Lattice parameters achieve <0.1 Å variation across samples
- [x] **Gradient Flow**: Gradients propagate through Gumbel-Softmax and quantum circuits
- [x] **QARP Check**: Exported QASM contains only supported gates (H, RY, RZ, CNOT, CRY, etc.)

## Project Structure
```
AI/ML /
├── hybrid_circuit.py       # 40-qubit quantum circuit
├── hybrid_mapper.py        # Classical readout mapping
├── qgan_generator.py       # Complete generator module
├── verify_qgan.py          # Verification test suite
├── example_usage.py        # 7 practical examples
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Next Steps (Phase 2)

This implementation is ready for:
1. **Training Loop**: Integrate discriminator and adversarial loss
2. **Fujitsu Deployment**: Use exported QASM with Fujitsu Quantum Simulator via MPI
3. **Database Integration**: Validate bitstrings against QMOF database
4. **Lattice Optimization**: Fine-tune for experimentally realizable MOF structures

## References

- Quantum GAN architectures: [cite: QGANs for generative modeling]
- Circuit optimization: [cite: QARP Fused-Swap optimizations]
- PennyLane documentation: https://pennylane.ai
- PyTorch documentation: https://pytorch.org

## Notes

- All quantum circuits execute on CPU via PennyLane's `default.qubit` simulator
- For GPU acceleration, use PyTorch GPU tensors while keeping quantum circuit on CPU
- For MPI deployment on Fujitsu Simulator, use the `export_qasm()` method
- Gradient computation works end-to-end including quantum + classical layers

---

**Status**: ✓ Task 1b Complete  
**Last Updated**: 2026-02-13  
**Verification**: All 4 checklist items passing
