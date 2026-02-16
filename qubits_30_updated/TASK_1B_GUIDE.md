# Task 1b: Complete Guide to the QGAN Generator

## Table of Contents
1. [What Is Task 1b?](#what-is-task-1b)
2. [Why Quantum?](#why-quantum)
3. [The Big Picture](#the-big-picture)
4. [Stage-by-Stage Breakdown](#stage-by-stage-breakdown)
5. [The Output Format](#the-output-format)
6. [What Happens After 1b](#what-happens-after-1b)
7. [The Feedback Loop](#the-feedback-loop)
8. [How to Run and Test](#how-to-run-and-test)
9. [File Reference](#file-reference)

---

## What Is Task 1b?

**Task 1b is building the Generator half of a Quantum GAN (QGAN).**

A GAN has two parts:
- **Generator (Task 1b):** Takes random noise → produces fake data (in our case, MOF crystal structures)
- **Discriminator (Task 1a):** Takes data → decides if it's real or fake

Your job is to build a machine that can generate candidate Metal-Organic Framework (MOF) structures from nothing but random numbers. The generator doesn't know if its outputs are good or bad — that's the discriminator's job.

**What you're building:**
```
Random Noise (32 numbers) → [Your Generator] → MOF Structure (atoms + lattice)
```

**What you're NOT building:**
- The discriminator (Task 1a)
- The training loop (integration task)
- Real MOF data loading (discriminator's responsibility)

---

## Why Quantum?

The quantum circuit isn't just for show. It solves a real problem.

**The problem:** MOFs are defined by:
- A discrete choice (which metal? which linker? which topology?) — 256+ options
- A continuous choice (where exactly are the atoms? what's the lattice shape?) — infinite options

These choices are *correlated*. If you pick a Zinc-based node, the lattice angles are constrained. If you pick a large linker, the unit cell must be bigger.

**Why quantum helps:**
1. **Entanglement creates correlations for free.** When qubits are entangled, measuring one affects the others. This means the topology choice automatically influences the geometry — no extra code needed.

2. **Exponential state space.** 30 qubits can represent 2^30 ≈ 1 billion states simultaneously. This lets us explore a vast design space in a single circuit execution.

3. **Native probability distributions.** Quantum measurement gives us probabilities directly. Perfect for sampling diverse MOF candidates.

A classical neural network can do all this too, but it needs explicit wiring to couple discrete and continuous choices. The quantum circuit does it naturally through physics.

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GENERATOR PIPELINE (Task 1b)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────────┐            │
│  │  NOISE   │    │  NOISE ENCODER  │    │  QUANTUM CIRCUIT │            │
│  │ (32 dim) │ ──►│ (Classical MLP) │ ──►│   (30 qubits)    │            │
│  └──────────┘    └─────────────────┘    └────────┬─────────┘            │
│                         │                        │                      │
│              Produces 180 rotation        Produces:                     │
│              angles for the circuit       - 256 topology probs          │
│                                           - 15 node expectation vals    │
│                                           - 7 latent expectation vals   │
│                                                  │                      │
│                                                  ▼                      │
│                                    ┌──────────────────────────┐         │
│                                    │    CLASSICAL MAPPER      │         │
│                                    │    (PyTorch neural net)  │         │
│                                    └────────────┬─────────────┘         │
│                                                 │                       │
│                                                 ▼                       │
│                              ┌──────────────────────────────────┐       │
│                              │          MOF STRUCTURE           │       │
│                              │  • metal_atoms: (5, 3) coords    │       │
│                              │  • linker_atoms: (10, 3) coords  │       │
│                              │  • lattice_params: (6,) params   │       │
│                              │  • topo_state: integer 0-255     │       │
│                              │  • validity: float 0-1           │       │
│                              └──────────────────────────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Stage-by-Stage Breakdown

### Stage 1: Noise Input

**What:** 32 random numbers sampled from a normal distribution.

**Why:** This is standard GAN design. The noise vector is the "seed" that determines which MOF gets generated. Different noise → different MOF. Same noise → same MOF (deterministic once trained).

**Code:** `noise = torch.randn(batch_size, 32)`

---

### Stage 2: Noise Encoding

**What:** A classical neural network (MLP) transforms the 32 noise values into 180 rotation angles.

**Why:** The quantum circuit has gates that need angles (like "rotate this qubit by 0.45 radians"). We learn a mapping from simple noise to these angles, rather than optimizing the angles directly.

**Architecture:**
```
32 → 128 → 256 → 128 → 180
     (ReLU + LayerNorm between each)
```

**The 180 parameters split into:**
| Register | Qubits | Parameters | Calculation |
|----------|--------|------------|-------------|
| Topology | 8 | 48 | 2 layers × 8 qubits × 3 rotations |
| Node | 15 | 90 | 2 layers × 15 qubits × 3 rotations |
| Latent | 7 | 42 | 2 layers × 7 qubits × 3 rotations |

---

### Stage 3: Quantum Circuit Execution

This is the heart of the generator. 30 qubits split into three registers.

#### 3a. Global Entanglement (All Qubits)

First, every qubit gets:
1. A **Hadamard gate** — puts it in superposition (equal chance of 0 or 1)
2. A **CNOT ladder** — entangles qubit 0→1, 1→2, 2→3, ..., 28→29

**Why this matters:** After entanglement, the registers are no longer independent. Topology influences node positions, node positions influence latent features. This is how we get correlated discrete+continuous outputs.

#### 3b. Topology Register (Qubits 0-7)

**Job:** Decide which *type* of MOF to build.

**Circuit:** `StronglyEntanglingLayers` — the most expressive template in PennyLane. Each layer applies RX, RY, RZ rotations on every qubit plus entangling CNOTs.

**Output:** Probability distribution over 2^8 = 256 states. Each state is a bitstring like `01101010` encoding a discrete topology choice (metal type, linker family, connectivity pattern).

**Why discrete:** You can't have "half a Zinc node" or "0.7 of a BDC linker." Chemical identity is categorical.

#### 3c. Conditioning Bridge (Topo → Nodes)

5 **CRX gates** connect topology qubits to node qubits.

A CRX gate says: "If the control qubit is |1⟩, rotate the target qubit by this angle. If |0⟩, do nothing."

This creates the dependency: the geometry changes based on which topology was selected. If the circuit "chose" a Zinc-based MOF, the node positions get rotated differently than if it chose Copper.

#### 3d. Node Register (Qubits 8-22)

**Job:** Directly encode metal atom coordinates.

15 qubits → 15 expectation values ⟨Z⟩ → 5 atoms × 3 coordinates (x, y, z).

**Circuit:** RX/RY/RZ rotations with alternating CNOT entanglement. Simpler than topology because we want direct, interpretable readout.

**Output:** 15 continuous values in [-1, +1]. These become atomic coordinates after scaling.

#### 3e. Latent Register (Qubits 23-29)

**Job:** Encode compressed features for linker atoms and lattice parameters.

7 qubits must encode info for:
- 10 linker atoms × 3 coordinates = 30 values
- 6 lattice parameters

That's 36 outputs from 7 qubits — a 5× compression. The classical mapper will decompress it.

**Output:** 7 expectation values in [-1, +1].

---

### Stage 4: Classical Mapping

The quantum circuit outputs raw numbers. The mapper turns them into physical MOF parameters.

#### Metal Atoms (Direct Path)
```
15 node expvals → learnable scale+bias → tanh → × 12.0 Å → reshape to (5, 3)
```
Result: 5 metal atom positions, each coordinate in [-12, +12] Ångströms.

**Why direct:** Metal atoms are the rigid skeleton. One qubit = one coordinate gives maximum precision and interpretability.

#### Linker Atoms + Lattice (Multiplexed Path)
```
7 latent features + 8 topo features = 15 inputs
    → MLP (15 → 64 → 64 → 36)
    → Split into:
        - 30 linker coords → tanh → × 12.0 Å → reshape to (10, 3)
        - 6 lattice params → sigmoid → scale to physical bounds
```

**Lattice bounds (hardcoded physics):**
| Parameter | Range | Physical Meaning |
|-----------|-------|------------------|
| a, b, c | 5-20 Å | Unit cell lengths — can't be smaller than an atom |
| α, β, γ | 70-110° | Unit cell angles — extreme angles collapse the crystal |

#### Topology State
```
256 topology probabilities → argmax → integer 0-255
```
This identifies which MOF "family" was chosen.

---

### Stage 5: Validity Scoring

The mapper runs quick physics checks:
- **Atom overlap:** Any atoms closer than 1.0 Å? (impossible — atoms would collide)
- **Fragmentation:** Any atoms farther than 25.0 Å from each other? (not a connected structure)
- **Lattice sanity:** Are lengths in [4, 25] Å?

Output: validity score in [0, 1]. This is a self-assessment, NOT the discriminator's judgment. Just a rough filter to flag obviously broken structures.

---

## The Output Format

When you call `generator(noise)`, you get:

```python
{
    'metal_atoms':    torch.Tensor (batch, 5, 3),   # 5 metal atoms, xyz in Ångströms
    'linker_atoms':   torch.Tensor (batch, 10, 3),  # 10 linker atoms, xyz in Ångströms
    'lattice_params': torch.Tensor (batch, 6),      # [a, b, c, α, β, γ]
    'topo_state':     torch.Tensor (batch,),        # Integer 0-255
    'validity':       torch.Tensor (batch,)         # Float 0-1
}
```

**Units:**
- Coordinates: Ångströms (1 Å = 10⁻¹⁰ m, typical bond length ~1.5 Å)
- Lattice lengths a, b, c: Ångströms
- Lattice angles α, β, γ: Degrees

**Together these define a crystal:** The lattice_params define the unit cell shape. The atom coordinates are positions inside that cell. The topo_state identifies the MOF family. You can write this to a CIF file for downstream simulation.

---

## What Happens After 1b

**Right now, your generator produces garbage.**

It's randomly initialized. It doesn't know what a valid MOF looks like. Every output is nonsense atoms scattered randomly in space.

**To make it useful, you need training.** That's where Task 1a comes in.

### The Training Loop (Not Your Job, But Here's How It Works)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            GAN TRAINING LOOP                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Step 1: Generate fake MOFs                                            │
│   ─────────────────────────                                             │
│   noise = torch.randn(batch, 32)                                        │
│   fake_mof = generator(noise)     ← Your Task 1b code                   │
│                                                                         │
│   Step 2: Get real MOFs                                                 │
│   ────────────────────                                                  │
│   real_mof = dataset.sample(batch)  ← From QMOF database                │
│                                                                         │
│   Step 3: Discriminator scores both                                     │
│   ───────────────────────────────                                       │
│   fake_score = discriminator(fake_mof)  ← Task 1a code                  │
│   real_score = discriminator(real_mof)  ← Task 1a code                  │
│                                                                         │
│   Step 4: Compute losses                                                │
│   ──────────────────────                                                │
│   D_loss = -log(real_score) - log(1 - fake_score)   # D wants to catch │
│   G_loss = -log(fake_score)                          # G wants to fool  │
│                                                                         │
│   Step 5: Update weights                                                │
│   ──────────────────────                                                │
│   D_loss.backward() → update discriminator                              │
│   G_loss.backward() → update generator     ← Gradients flow into 1b    │
│                                                                         │
│   Repeat 10,000+ times                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**After training:**
- Generator has learned: "If I produce structures like X, the discriminator says they're real."
- You can sample novel MOFs by feeding random noise and getting realistic structures.

---

## The Feedback Loop

### Is There a Feedback Loop Inside 1b?

**No.** Task 1b is a pure forward pass.

```
noise in → processing → MOF out
```

There's no loop inside the generator. It doesn't look at its output. It doesn't try again. It just transforms noise to structure, once.

### Where Does Feedback Come From?

**From the discriminator, via training.**

The discriminator (Task 1a) looks at the generator's output and says "this is 20% realistic" or "this is 80% realistic." That score becomes a loss, and the loss flows backward through the generator via backpropagation.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FEEDBACK FLOW                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   FORWARD (what 1b builds):                                             │
│   ─────────────────────────                                             │
│   noise → [Encoder] → [Quantum Circuit] → [Mapper] → fake_mof          │
│                                                          │              │
│   EVALUATION (what 1a builds):                           ▼              │
│   ─────────────────────────────                   [Discriminator]       │
│                                                          │              │
│   BACKWARD (the training loop):                          ▼              │
│   ─────────────────────────────                   loss = -log(score)    │
│                                                          │              │
│                                                    loss.backward()      │
│                                                          │              │
│                              ┌───────────────────────────┘              │
│                              ▼                                          │
│   Gradients update:                                                     │
│   • Noise encoder MLP weights                                           │
│   • Mapper MLP weights                                                  │
│   • Quantum circuit params (via parameter-shift rule)                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key insight:** Tasks 1a and 1b don't contain feedback themselves. They're both forward-pass machines. The training loop (a separate task) connects them and creates the feedback.

### Why This Separation?

Modularity. You can:
- Test the generator independently (does it produce outputs in the right shape?)
- Test the discriminator independently (can it tell apart random vs. structured data?)
- Swap in different generators or discriminators without rewriting everything

---

## How to Run and Test

### Install Dependencies
```bash
cd "AI/ML /qubits_30"
pip install -r requirements.txt
```

### Quick Smoke Test
```bash
python quick_test_30q.py
```
Runs fast, checks that the generator initializes and produces outputs.

### Full Test Suite
```bash
python test_qgan_30q.py --verbose
```
Runs 9 tests covering:
1. Circuit initialization
2. Circuit forward pass
3. Mapper initialization
4. Mapper forward pass (with physics checks)
5. Generator initialization
6. End-to-end generation
7. Batch processing
8. OpenQASM export
9. Checkpoint save/load

### Generate a Sample MOF
```python
from qgan_generator_30q import QGAN_Generator30Q
import torch

gen = QGAN_Generator30Q(split_registers=True)  # True for laptop, False for HPC
noise = torch.randn(1, 32)
output = gen(noise)

print(f"Metal atoms: {output['metal_atoms'].shape}")      # (1, 5, 3)
print(f"Linker atoms: {output['linker_atoms'].shape}")    # (1, 10, 3)
print(f"Lattice: {output['lattice_params']}")             # [a, b, c, α, β, γ]
print(f"Topology: {output['topo_state'].item()}")         # 0-255
print(f"Validity: {output['validity'].item():.3f}")       # 0-1
```

---

## File Reference

| File | Purpose |
|------|---------|
| `qgan_generator_30q.py` | Main generator class — ties quantum + classical together |
| `hybrid_circuit_30q.py` | The 30-qubit quantum circuit definition |
| `hybrid_mapper_30q.py` | Classical neural net: quantum outputs → MOF structure |
| `test_qgan_30q.py` | Comprehensive test suite (9 tests) |
| `quick_test_30q.py` | Fast smoke test |
| `requirements.txt` | Python dependencies |
| `TASK_1B_GUIDE.md` | This document |

---

## Summary

**Task 1b = Build the Generator**

- **Input:** 32 random floats
- **Output:** MOF structure (atoms + lattice + topology + validity score)
- **Architecture:** Noise Encoder (MLP) → Quantum Circuit (30q) → Classical Mapper (MLP)
- **No feedback inside 1b:** It's a pure forward pass
- **Feedback comes from training:** Discriminator (1a) scores outputs, gradients flow back

Your generator is ready. It produces garbage right now, but it produces garbage in exactly the right format. Once connected to training, it will learn to produce valid MOFs.
