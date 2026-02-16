# 30-Qubit QGAN — HPC Run Guide & Failsafe

## Quick Start

```bash
# 1. SSH into HPC
ssh lissan@test

# 2. Activate env
conda activate fujitsu-quantum-challenge-2026

# 3. Navigate
cd ~/GITHUB-STUFF/fujitsu-quantum-challenge-2026/qubits_30

# 4. Install lightning backend (CRITICAL — without this, sim hangs)
pip install pennylane-lightning

# 5. Verify lightning installed
python -c "import pennylane as qml; dev = qml.device('lightning.qubit', wires=2); print('lightning OK')"

# 6. Run the test
python quick_test_30q.py --timeout 180
```

---

## What Each Test Does

| Test | What | Quantum? | Expected Time |
|------|------|----------|---------------|
| 1. Config | Validates 30 = 8+15+7 qubit split | No | <1s |
| 2. Circuit Init | Creates PennyLane device + QNode | No | <5s |
| 3. Circuit Forward | **THE HANG TEST** — runs 30q sim once | **YES** | 10-120s with lightning |
| 4. Mapper | Classical PyTorch forward pass | No | <1s |
| 5. Generator Init | Creates full QGAN_Generator30Q | No | <5s |
| 6. E2E Generation | noise → circuit → mapper → MOF (batch=1) | **YES** | 10-120s |
| 7. Batch | Same but batch=2 (2 circuit calls) | **YES** | 20-240s |
| 8. QASM Export | Generates OpenQASM string | No | <1s |
| 9. Checkpoint | Save + load + re-generate | **YES** | 30-360s |

**Total expected time**: ~2-10 min with `lightning.qubit`, **infinite hang** without it.

---

## Reading the Output

Every component prints timestamped checkpoints. Here's what healthy output looks like:

```
[generator pid=12345] encode_noise (batch=1)...
[generator pid=12345] encode_noise done in 0.01s
[generator pid=12345] quantum circuit sample 1/1...
  [circuit.forward pid=12345] QNode execution starting (30q, device=lightning.qubit)...
  [circuit.forward pid=12345] QNode done in 45.23s, unpacking results...
  [circuit.forward pid=12345] total=45.25s (qnode=45.23s, convert=0.02s)
[generator pid=12345] sample 1/1 done (45.26s this sample, 45.28s total)
[generator pid=12345] classical mapper...
    [mapper] forward start (batch=1)
    [mapper] metal_atoms done (0.001s)
    [mapper] topo_head done (0.001s)
    [mapper] linker+lattice done (0.002s)
    [mapper] validity done (0.003s)
[generator pid=12345] mapper done in 0.01s
[generator pid=12345] forward total: 45.29s
```

**If it hangs**, the last line you see tells you exactly where:

| Last line you see | What's stuck | Fix |
|---|---|---|
| `QNode execution starting...` | PennyLane statevector sim | Install `pennylane-lightning` |
| `quantum circuit sample 1/1...` | Same — inside circuit.forward | Same |
| `encode_noise (batch=1)...` | PyTorch MLP forward | Check CUDA/memory |
| `classical mapper...` | Mapper forward | Check tensor shapes |

---

## Failsafe Action Guide

### Scenario 1: Test 3 times out (circuit forward hangs)

**Symptom**: Output stops at `QNode execution starting...` for 180+ seconds

**Actions in order**:
```bash
# Step A: Check if lightning is actually being used
# Look for this line in output:
#   [circuit] Using lightning.qubit (C++ optimized) for 30 qubits
# If you see "Falling back to default.qubit" instead, lightning isn't installed

# Step B: Install lightning
pip install pennylane-lightning

# Step C: Verify
python -c "import pennylane as qml; dev = qml.device('lightning.qubit', wires=30); print('30q device OK')"

# Step D: Re-run with longer timeout
python quick_test_30q.py --timeout 300

# Step E: If STILL hanging — reduce to 1 layer (halves circuit depth)
# Edit qgan_generator_30q.py line 328:
#   gen = QGAN_Generator30Q(noise_dim=32, num_linker_atoms=10, n_circuit_layers=1)
# Or pass it in the test
```

### Scenario 2: Import errors

**Symptom**: `ModuleNotFoundError: No module named 'hybrid_circuit_30q'`

```bash
# Make sure you're in the right directory
cd ~/GITHUB-STUFF/fujitsu-quantum-challenge-2026/qubits_30
ls *.py  # should show hybrid_circuit_30q.py, hybrid_mapper_30q.py, etc.

# If files are missing, sync from local
scp /Users/advayjain/Desktop/Fujitsu\ QC/Coding/AI/ML\ /qubits_30/*.py lissan@test:~/GITHUB-STUFF/fujitsu-quantum-challenge-2026/qubits_30/
```

### Scenario 3: Out of memory

**Symptom**: `Killed` or `MemoryError` during circuit forward

```bash
# 30q statevector needs ~16 GB RAM
# Check available memory
free -h

# If not enough, request more in SLURM
#SBATCH --mem=32G

# Or reduce qubits temporarily for testing (not recommended for final)
```

### Scenario 4: Process won't die

**Symptom**: Ctrl+C doesn't work, process stuck

```bash
# From another terminal
ps aux | grep quick_test
kill -9 <PID>

# Or kill all python
pkill -9 -f quick_test_30q
```

### Scenario 5: Tests 1-4 pass but 5+ fail

**Symptom**: Circuit forward works alone but generator fails

```bash
# This means quantum sim works but something in the integration is off
# Run with verbose to see exactly where
python quick_test_30q.py --timeout 300 -v

# Check the checkpoint prints — they'll show exactly which step broke
```

### Scenario 6: Everything passes locally but fails on HPC

```bash
# Check Python version matches
python --version  # need 3.10+

# Check PennyLane version
python -c "import pennylane; print(pennylane.__version__)"  # need 0.35+

# Check PyTorch
python -c "import torch; print(torch.__version__)"
```

---

## Files Modified (sync all of these to HPC)

```
qubits_30/
├── hybrid_circuit_30q.py    # lightning.qubit fallback + CRX reduction + hang detectors
├── hybrid_mapper_30q.py     # checkpoint prints in forward()
├── qgan_generator_30q.py    # per-sample timing in batch loop
├── quick_test_30q.py        # NEW — smoke test with timeouts
└── test_qgan_30q.py         # OLD — don't run this one (no timeouts, will hang)
```

**Sync command**:
```bash
scp /Users/advayjain/Desktop/Fujitsu\ QC/Coding/AI/ML\ /qubits_30/{hybrid_circuit_30q,hybrid_mapper_30q,qgan_generator_30q,quick_test_30q}.py \
    lissan@test:~/GITHUB-STUFF/fujitsu-quantum-challenge-2026/qubits_30/
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All 9 tests passed — safe to run full training |
| 1 | One or more tests failed — check output |
| 2 | Tests skipped due to circuit hang — need lightning.qubit |

---

## After All Tests Pass

Once `quick_test_30q.py` reports all 9 passed, you can safely run:

```bash
# The original full test suite (now with hang detectors baked in)
python test_qgan_30q.py --verbose --timeout 300

# Or go straight to training
python qgan_generator_30q.py
```
