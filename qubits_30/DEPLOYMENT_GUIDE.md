# 30-Qubit Hybrid QGAN Generator - Complete Deployment Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture & Theory](#architecture--theory)
3. [Local Development & Testing](#local-development--testing)
4. [Quantum Computer Deployment](#quantum-computer-deployment)
5. [HPC Deployment](#hpc-deployment)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

---

## System Overview

### Purpose
The 30-qubit Hybrid QGAN (Quantum Generative Adversarial Network) is an optimized quantum-classical system for generating metal-organic framework (MOF) structures. It balances computational complexity with physical feasibility on near-term quantum processors.

### Key Metrics
| Metric | Value |
|--------|-------|
| Total Qubits | 30 |
| Memory Atoms | 5 metal atoms + 10 linker atoms |
| Topology States | $2^8 = 256$ discrete structures |
| Latent Features | 7 quantum features |
| Classical Parameters | ~5,400 |
| Quantum Parameters | ~270-360 (2-3 layers) |
| Estimated Circuit Depth | 150-200 gates |
| Classical Sim Memory | ~1 GB |
| Model Size | ~22 KB (FP32) |

### Hardware Requirements

**Local Development (Laptop)**
```
- CPU: 2+ cores
- RAM: 4 GB minimum, 8 GB recommended
- Storage: 500 MB
- GPU: Optional (not required)
- Python: 3.8+
```

**HPC Deployment**
```
- Node RAM: 32+ GB
- Network: MPI-capable (for distributed simulations)
- Storage: 1 TB+ (for structure databases)
- Python 3.8+ with scientific stack
```

**Quantum Computer**
```
- Minimum qubits: 30
- T1/T2: >50 μs (for 2-qubit gate fidelity >98%)
- Gate errors: <0.1% (2-qubit), <0.01% (1-qubit)
- Connectivity: Linear or 2D grid recommended
```

---

## Architecture & Theory

### Quantum Circuit Design

#### Qubit Allocation (30 total)
```
Qubits 0-7:   Topology Register (8 qubits → 256 states)
              Purpose: Select discrete MOF topology
              Gates: StronglyEntanglingLayers (high entanglement)
              
Qubits 8-22:  Node Register (15 qubits → 5 atoms × 3 coordinates)
              Purpose: Encode metal atom positions
              Gates: BasicEntangler + CRX conditionals
              Conditional on: Topology register outputs
              
Qubits 23-29: Latent Register (7 qubits → latent features)
              Purpose: Extract continuous features for linker atoms
              Gates: StronglyEntanglingLayers (feature richness)
```

#### Circuit Stages

**Stage 1: Global Entanglement Seed**
```python
# Initialize uniform superposition
for wire in range(30):
    Hadamard(wire)

# Create dense entanglement via CNOT ladder
for depth in [0, 1]:  # 2 passes
    for wire in range(29):
        CNOT(control=wire, target=wire+1)
    # Reverse direction
    for wire in range(28, 0, -1):
        CNOT(control=wire, target=wire-1)

# Purpose: Information mixing between all qubits
# Depth: ~60 gates
```

**Stage 2: Topology Branch (8 qubits)**
```python
StronglyEntanglingLayers(
    weights=params_topo[n_layers, 8, 3],
    wires=[0, 1, 2, 3, 4, 5, 6, 7],
    imprimitive=CNOT
)

# Purpose: Discrete state selection (256 possibilities)
# Depth: ~24-36 gates per layer
```

**Stage 3: Data Re-uploading (Conditional Encoding)**
```python
# Topology → Node conditioning via CRX gates
for i in range(5):  # First 5 topo qubits
    for j in range(5):  # First 5 node qubits
        CRX(angle=0.15π, control=topo_wire[i], target=node_wire[j])

# Purpose: Create quantum correlation between discrete & continuous
# This is the KEY innovation: geometry depends on topology
# Depth: ~25 gates
```

**Stage 4: Node Register (15 qubits)**
```python
for layer in range(n_layers):
    # Per-qubit rotations
    for wire in node_wires:
        RX(params[layer, i, 0])
        RY(params[layer, i, 1])
        RZ(params[layer, i, 2])
    
    # Inter-qubit entanglement (alternating pattern)
    for i in range(0, 14, 2):
        CNOT(control=node_wire[i], target=node_wire[i+1])

# Purpose: Encode metal atom coordinates
# Acts as classical neural network layer in quantum
# Depth: ~45-60 gates per layer
```

**Stage 5: Latent Register (7 qubits)**
```python
StronglyEntanglingLayers(
    weights=params_latent[n_layers, 7, 3],
    wires=[23, 24, 25, 26, 27, 28, 29]
)

# Purpose: Rich feature extraction for organic linker design
# Depth: ~21-28 gates per layer
```

**Stage 6: Measurements**
```python
# Topology: Bitstring (256 classical outcomes)
topo_result ← measure(qubits=[0-7]) in computational basis

# Node & Latent: Expectation values (continuous)
node_expvals[i] ← <Z> for qubit [8+i]
latent_features[i] ← <Z> for qubit [23+i]

# Convert to structure:
metal_coords = node_expvals × angstrom_scale  # [Å]
linker_coords = map(latent_features) via MLP
lattice_params = MLP(latent_features)
```

### Classical Mapper

#### Input Processing
```
topo_logits (256-dim):       P(topology_state=k) for k ∈ {0,...,255}
node_expvals (15-dim):       ⟨Z⟩ for each node qubit → [-1, 1]
latent_features (7-dim):     ⟨Z⟩ for each latent qubit → [-1, 1]
```

#### Output Generation

**Metal Atoms** (Direct Mapping)
```
metal_coords = tanh(node_expvals × scale + bias) × angstrom_scale
              ∈ R^(5×3)  [Angstroms]
              
Physical constraint: 4-20 Å from origin
```

**Linker Atoms** (MLP-Projected)
```
latent_encoded ← BatchNorm(ReLU(Linear(latent_features, hidden=64)))
linker_coords ← tanh(Linear(latent_encoded, 30)) × angstrom_scale
               ∈ R^(10×3)  [Angstroms]
```

**Lattice Parameters** (Physics Constrained)
```
lattice_lengths = Sigmoid(raw[0:3]) × 15 + 5      ∈ [5, 20] Å
lattice_angles  = Sigmoid(raw[3:6]) × 40 + 70     ∈ [70, 110]°

Constraints enforce:
- Valid unit cell (no negative volumes)
- MOF-scale dimensions (not too small/large)
- Crystal system compatibility
```

#### Validity Scoring (0-1)
```
score = 1.0
score -= 0.1 × clamp(n_close_atoms / 5)        # Penalty for clashes
score -= 0.05 × clamp(n_far_atoms / 10)        # Penalty for fragmentation
score *= is_lattice_valid()                     # Hard constraint
score *= is_topology_diverse()                  # Soft constraint
return clamp(score, 0, 1)
```

---

## Local Development & Testing

### Installation

#### Step 1: Create Virtual Environment
```bash
# 30-qubit version
cd /path/to/qubits_30
python3.8 -m venv qgan30_env
source qgan30_env/bin/activate  # On Windows: qgan30_env\Scripts\activate
```

#### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r ../requirements.txt

# Core dependencies
pip install pennylane>=0.33.0
pip install torch>=2.0.0
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0

# Optional: GPU support
pip install torch-cuda  # For NVIDIA GPUs (if applicable)
```

#### Verify Installation
```bash
python -c "import pennylane as qml; import torch; print(f'PennyLane: {qml.__version__}'); print(f'PyTorch: {torch.__version__}')"
```

### Step 3: Quick Verification
```bash
python -c "
from hybrid_circuit_30q import HybridQGANCircuit30Q
import numpy as np

circuit = HybridQGANCircuit30Q(n_layers=2)
p = np.random.randn(2, 8, 3) * 0.1
result = circuit.forward(p, np.random.randn(2, 15, 3) * 0.1, np.random.randn(2, 7, 3) * 0.1)
print('✓ Quantum circuit working')
print(f'  Topology logits: {result[\"topo_logits\"].shape}')
"
```

### Running Local Tests

#### Full Test Suite
```bash
# Run all tests (comprehensive)
python test_qgan_30q.py --verbose

# Output should show:
# ✓ PASS | Circuit Initialization ...
# ✓ PASS | Circuit Forward Pass ...
# ... (9 tests total)
# SUMMARY: 9/9 tests passed

# Approximate runtime: 2-5 minutes (depends on CPU)
```

#### Individual Component Tests
```bash
# Test only quantum circuit
python -c "
from hybrid_circuit_30q import HybridQGANCircuit30Q
circuit = HybridQGANCircuit30Q(n_layers=2)
# [Test runs]
"

# Test only mapper
python -c "
from hybrid_mapper_30q import HybridReadoutMapper30Q
mapper = HybridReadoutMapper30Q(num_linker_atoms=10)
# [Test runs]
"

# Test end-to-end generation
python -c "
from qgan_generator_30q import QGAN_Generator30Q
gen = QGAN_Generator30Q()
import torch
noise = torch.randn(1, 32)
structures = gen(noise)
print(f'Generated structure: {structures[\"validity\"].item():.3f}')
"
```

#### Benchmark
```bash
python -c "
from qgan_generator_30q import QGAN_Generator30Q
import torch
import time

gen = QGAN_Generator30Q()

# Benchmark different batch sizes
for batch_size in [1, 2, 4]:
    noise = torch.randn(batch_size, 32)
    
    start = time.time()
    for _ in range(5):  # Average over 5 runs
        gen(noise, profile_timing=True)
    elapsed = (time.time() - start) / 5
    
    print(f'Batch size {batch_size}: {elapsed*1000:.1f}ms per generation')
"
```

### Debugging Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ImportError: No module 'pennylane'` | PennyLane not installed | `pip install pennylane>=0.33.0` |
| `CUDA out of memory` | GPU memory exhausted | Reduce batch size, use CPU |
| `Shape mismatch error` | Parameter dimension mismatch | Verify `n_layers` parameter |
| `Circuit too deep` | Circuit exceeds hardware limits | Reduce `n_circuit_layers` to 1 |
| Slow performance | Classical simulator saturated | Move to hardware or HPC |

---

## Quantum Computer Deployment

### Overview
The 30-qubit circuit is designed for near-term quantum processors with:
- **Minimum requirements**: 30 contiguous qubits
- **Ideal connectivity**: Linear or 2D grid
- **Gate fidelity**: >99% (1-qubit), >98% (2-qubit)
- **Coherence**: T1, T2 > 50 μs

### Supported Platforms

#### 1. IBM Quantum (Qiskit)

**Setup**
```bash
pip install qiskit>=0.41.0
pip install qiskit-aer  # High-performance simulator
pip install qiskit-ibmq  # IBM cloud access
```

**Authentication**
```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Save API token (one-time)
#QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")

# Load for future sessions
service = QiskitRuntimeService()
```

**Export & Execution**
```python
from qgan_generator_30q import QGAN_Generator30Q
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Session

# Generate parameters
gen = QGAN_Generator30Q(n_circuit_layers=2)
noise = torch.randn(1, 32)
_, params_topo, params_node, params_latent = gen.encode_noise(noise)

# Build Qiskit circuit
qr = QuantumRegister(30, 'q')
cr = ClassicalRegister(30, 'c')
circuit = QuantumCircuit(qr, cr, name='MOF_Generator_30Q')

# Add gates (Stage 1: Global entanglement)
for i in range(30):
    circuit.h(qr[i])
for i in range(29):
    circuit.cx(qr[i], qr[i+1])

# [Add parameterized rotations from params_topo, etc.]
# ... (detailed gate construction)

# Stage 6: Measurements
circuit.measure(qr, cr)

# Execute on IBM hardware
with Session(service=service, backend="ibm_nairobi"):
    # ibm_nairobi has 27 qubits; adjust for 30q systems
    # Alternatives: ibm_brisbane (127q), ibm_heron (133q)
    
    transpiled = transpile(circuit, backend=backend, optimization_level=3)
    job = backend.run(transpiled, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    # Parse topology state from measurement outcomes
    topo_state = max(counts, key=counts.get)  # Most probable outcome
    probability = counts[topo_state] / 1024
```

**Key Configuration**
```python
# For optimal mapping to IBM hardware:
from qiskit.transpiler import passes

# Strategy: Map 30 physical qubits to available backend
layout = Layout({qubits[i]: backend.configuration().qargs[i] for i in range(30)})

# Use native gates of backend
basis_gates = backend.configuration().basis_gates
# Typically: ['id', 'rz', 'sx', 'x', 'cx']
```

#### 2. Fujitsu Quantum (QARP/QSIMULATOR)

**Key Advantage**: Dedicated tensor network simulator (can handle 30+ qubits efficiently)

**Export to OpenQASM 2.0**
```python
from qgan_generator_30q import QGAN_Generator30Q
import torch

gen = QGAN_Generator30Q()
noise = torch.randn(1, 32)

# Export circuit
qasm_code = gen.export_openqasm(noise, filepath="mof_generator_30q.qasm")

# File will contain standard OpenQASM 2.0:
"""
OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];

// Your circuit here with rx, ry, rz, cx gates

measure q -> c;
"""
```

**Upload to Fujitsu Cloud**
```bash
# Via QSIMULATOR online platform at https://www.fujitsu.com/jp/solutions/quantum/
# 1. Navigate to circuit designer
# 2. Import qasm file
# 3. Configure settings:
#    - Backend: Tensor Network Simulator (if available)
#    - Shots: 1024
#    - Optimization: Level 3
# 4. Submit job
# 5. Download results (JSON format)
```

**Programmatic Submission** (if API available)
```python
import requests
import json

# Fujitsu QARP API (example endpoint)
QARP_ENDPOINT = "https://api.quantum.fujitsu.com/v1/execute"

with open("mof_generator_30q.qasm", "r") as f:
    circuit_qasm = f.read()

payload = {
    "circuit": circuit_qasm,
    "backend": "tensor_network_simulator",
    "shots": 1024,
    "optimization_level": 3
}

response = requests.post(
    QARP_ENDPOINT,
    json=payload,
    headers={"Authorization": f"Bearer {FUJITSU_TOKEN}"}
)

job_id = response.json()["job_id"]
print(f"Submitted job: {job_id}")

# Poll for results
import time
while True:
    status = requests.get(
        f"{QARP_ENDPOINT}/jobs/{job_id}",
        headers={"Authorization": f"Bearer {FUJITSU_TOKEN}"}
    ).json()
    
    if status["status"] == "completed":
        results = status["results"]
        break
    
    time.sleep(5)  # Check every 5 seconds
```

#### 3. IonQ

**Network Access**
```bash
pip install qiskit-ionq
```

**Integration**
```python
from qiskit_providers_ionq import IonQ
from qiskit import QuantumCircuit, transpile

# Set up IonQ credentials
ionq = IonQ(apikey="YOUR_IONQ_API_KEY", workspace="YOUR_WORKSPACE")

# Build circuit (same Qiskit circuit as above)
circuit = QuantumCircuit(30, 30)
# ... add gates ...

# Submit
job = ionq.run(circuit, shots=1024)

# Results
results = job.result()
counts = results.get_counts()
```

### Noise Mitigation Techniques

For 30+ qubit systems, errors accumulate. Apply these strategies:

#### 1. Circuit Optimization
```python
from qiskit.transpiler import optimize_for_execution

# Reduce two-qubit gate count
optimized = optimize_for_execution(circuit, backend)
print(f"Original depth: {circuit.depth()}")
print(f"Optimized depth: {optimized.depth()}")
```

#### 2. Error Suppression (Zero Noise Extrapolation)
```python
# Scale circuit depths: 1x, 3x, 5x
# Extrapolate to 0x (ideal)

def scale_circuit(circuit, scale_factor):
    scaled = circuit.copy()
    # Insert identity gates to increase depth
    for _ in range(scale_factor - 1):
        for qubit in scaled.qregs[0]:
            scaled.id(qubit)
    return scaled

# Execute scaled versions
for scale in [1, 3, 5]:
    scaled_circuit = scale_circuit(circuit, scale)
    result = backend.run(scaled_circuit, shots=1024).result()
    # ... extract observable ...
    results_dict[scale] = observable_value

# Extrapolate
import numpy as np
scales = np.array([1, 3, 5])
values = np.array([results_dict[s] for s in scales])
ideal_value = np.polyfit(scales, values, 1)[1]  # Intercept at scale=0
```

#### 3. Dynamical Decoupling
```python
# Add π pulses during idle times
from qiskit.transpiler.passes import DynamicalDecoupling

dd_pass = DynamicalDecoupling(backend.configuration().dt, ["x"])
circuit = dd_pass.run(circuit)
```

### Expected Hardware Performance

| Processor | Qubits | Fidelity | Max Depth | Typical Latency |
|-----------|--------|----------|-----------|-----------------|
| IBM Nairobi | 27 | 99.5% | 50 | 5 min |
| IBM Heron | 133 | 99.8% | 200 | 2 min |
| Fujitsu Quantum | 30+ (sim) | — | 500+ | 30 sec |
| IonQ | 11 (current) | 99.7% | 100 | 10 min |

---

## HPC Deployment

### Architecture

Deploy on HPC for:
1. **Ensemble averaging**: Run 100s of independent circuits, average results
2. **Variational training**: Optimize parameters via batched gradient descent
3. **Uncertainty quantification**: Bootstrap sampling for error bars
4. **Structure validation**: Compute MOF properties (bandgaps, porosity, etc.)

### Multi-Node MPI Setup

#### File: `hpc_deploy_30q.py`

```python
"""
Distributed HPC deployment for 30-qubit QGAN.
Requires: PETSc/mpi4py for distributed setup
"""

from mpi4py import MPI
import torch
import numpy as np
from pathlib import Path

from qgan_generator_30q import QGAN_Generator30Q

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main():
    """
    Master process (rank 0): Coordinate generation
    Worker processes (rank > 0): Execute quantum circuits
    """
    
    # Configuration
    num_structures = 1000
    batch_per_worker = 50
    noise_dim = 32
    
    if rank == 0:
        # Master: Distribute tasks
        print(f"HPC Deployment: {size} processes, generating {num_structures} structures")
        
        # Allocate noise vectors
        all_noise = torch.randn(num_structures, noise_dim)
        chunks = np.array_split(all_noise, size)
        
        # Scatter to workers
        for i in range(1, size):
            comm.send(chunks[i], i)
        
        # Process master's own chunk
        local_noise = chunks[0]
    else:
        # Worker: Receive tasks
        local_noise = comm.recv(source=0)
    
    # Initialize local generator on each process
    # Note: Use CPU or local GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = QGAN_Generator30Q(device=device)
    
    # Process local batch
    structures_local = []
    validities_local = []
    
    for batch_idx in range(0, len(local_noise), 50):
        batch_noise = local_noise[batch_idx:batch_idx+50]
        outputs = gen(batch_noise)
        
        structures_local.append(outputs['metal_atoms'].detach().cpu().numpy())
        validities_local.append(outputs['validity'].detach().cpu().numpy())
        
        if rank == 0:
            print(f"Progress: {batch_idx + 50}/{len(local_noise)}")
    
    # Gather results
    all_structures = comm.gather(np.concatenate(structures_local), root=0)
    all_validities = comm.gather(np.concatenate(validities_local), root=0)
    
    if rank == 0:
        # Master: Aggregate results
        all_structures = np.concatenate(all_structures)
        all_validities = np.concatenate(all_validities)
        
        print(f"\nResults Summary:")
        print(f"  Total structures: {len(all_validities)}")
        print(f"  Avg validity: {all_validities.mean():.3f}")
        print(f"  Validity std: {all_validities.std():.3f}")
        
        # Save to HDF5 for post-processing
        import h5py
        with h5py.File("mof_structures_30q.h5", "w") as f:
            f.create_dataset("metal_atoms", data=all_structures)
            f.create_dataset("validities", data=all_validities)
        
        print("Saved to: mof_structures_30q.h5")

if __name__ == "__main__":
    main()
```

#### Submission Script: `submit_hpc.sh`

```bash
#!/bin/bash
#SBATCH --job-name=QGAN_30q
#SBATCH --nodes=8              # 8 compute nodes
#SBATCH --ntasks-per-node=4    # 4 ranks per node (MPI)
#SBATCH --gpus-per-task=1      # 1 GPU per rank (if available)
#SBATCH --time=02:00:00        # 2 hour limit
#SBATCH --partition=gpu        # GPU partition
#SBATCH --output=qgan_%j.log

module load anaconda3
module load openmpi  # or equivalent MPI module

# Activate environment
source activate qgan30_env

# Run distributed job
mpirun -n 32 python hpc_deploy_30q.py > results_30q.txt 2>&1

# Expected near-linear scaling:
# - 4 ranks: ~2 min
# - 8 ranks: ~1 min
# - 32 ranks: ~15 sec
```

#### Execution
```bash
# Submit to HPC scheduler
sbatch submit_hpc.sh

# Monitor
squeue -u $USER
tail -f qgan_*.log

# Retrieve results after completion
scp -r login_node:/path/to/mof_structures_30q.h5 ./
```

### Post-Processing Pipeline

Once structures are generated, validate and analyze:

#### Script: `analyze_structures_30q.py`

```python
import h5py
import numpy as np
from pathlib import Path

# Load generated structures
with h5py.File("mof_structures_30q.h5", "r") as f:
    metal_atoms = f["metal_atoms"][:]
    validities = f["validities"][:]

print(f"Dataset: {metal_atoms.shape[0]} structures")
print(f"Validity distribution:")
print(f"  Min: {validities.min():.3f}")
print(f"  Mean: {validities.mean():.3f}")
print(f"  Median: {np.median(validities):.3f}")
print(f"  Max: {validities.max():.3f}")

# Filter high-quality structures
high_validity = metal_atoms[validities > 0.8]
print(f"\nHigh-quality structures (validity > 0.8): {len(high_validity)}")

# Export top 10 for manual inspection
# ... (CIF file generation, property calculation, etc.) ...
```

### Performance Expectations

| Configuration | Throughput | Cost (AWS) |
|---------------|-----------|-----------|
| 1 node (CPU) | 10 str/min | $0.30/hr |
| 1 node (GPU) | 60 str/min | $1.50/hr |
| 8 nodes (GPU) | 300 str/min | $12/hr |
| 32 nodes (GPU) | 1000 str/min | $48/hr |

---

## Performance Optimization

### Memory Optimization

```python
# Reduce precision if allowed
import torch
torch.set_float32_matmul_precision('medium')  # Faster matmul

# Use gradient checkpointing
torch.utils.checkpoint.checkpoint(gen, noise)
```

### Computational Optimization

```python
# Set PyTorch threading
torch.set_num_threads(4)

# Use tensor cores (GPU)
torch.backends.cudnn.benchmark = True
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend timeout | Reduce circuit complexity or increase timeout |
| Qubit connectivity error | Check hardware topology; may need circuit transpilation |
| High error rates | Apply error mitigation techniques |
| Memory overflow | Reduce batch size or use classical simulation |

---

## Summary

**This 30-qubit configuration is ideal for:**
- Learning quantum ML on real hardware
- Testing MOF generation workflows
- Resource-constrained environments

**Next Steps:**
- Run `python test_qgan_30q.py` on your laptop
- Export to quantum hardware via OpenQASM
- Validate with HPC ensemble
- Train discriminator in GAN loop (see 40q guide for advanced training)
