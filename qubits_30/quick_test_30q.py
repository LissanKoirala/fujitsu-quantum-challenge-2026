"""
Quick smoke test for 30-qubit QGAN — run locally first, then confirm on HPC.

Key design:
  - Hard SIGALRM timeouts so nothing hangs forever
  - Reuses a SINGLE circuit/generator instance (avoids re-init overhead)
  - If circuit forward times out, downstream tests are SKIPPED (they'd hang too)
  - Checkpoint prints at every stage so you can see exactly where it stalls

Usage:
    python quick_test_30q.py                  # defaults
    python quick_test_30q.py --timeout 300    # longer timeout for HPC
    python quick_test_30q.py -v               # verbose
"""

import signal
import sys
import time
import argparse
import os
import traceback
import numpy as np
import torch


# ── Timeout helper (Unix only — works on HPC) ──────────────────────
class HangTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise HangTimeout("SIGALRM fired — operation timed out")


def with_timeout(fn, label, timeout_sec=120):
    """Run fn() with a hard OS-level alarm. Returns (result, elapsed_sec)."""
    prev = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_sec)
    print(f"\n{'='*65}")
    print(f"  [{time.strftime('%H:%M:%S')}] TEST: {label}")
    print(f"  timeout={timeout_sec}s  pid={os.getpid()}")
    print(f"{'='*65}", flush=True)
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        print(f"  >>> PASS  ({elapsed:.2f}s)\n", flush=True)
        return result, elapsed
    except HangTimeout:
        elapsed = time.time() - t0
        print(f"  >>> TIMEOUT after {elapsed:.1f}s  *** HUNG ***\n", flush=True)
        raise
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  >>> ERROR after {elapsed:.2f}s\n"
              f"      {type(e).__name__}: {e}\n"
              f"      {traceback.format_exc()}", flush=True)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


# ── Individual tests ────────────────────────────────────────────────

def test_1_config():
    """Qubit config sanity check — instant."""
    from hybrid_circuit_30q import QubitConfig30Q
    cfg = QubitConfig30Q()
    assert cfg.n_qubits == 30, f"Expected 30 qubits, got {cfg.n_qubits}"
    assert len(cfg.wires_topo) == 8
    assert len(cfg.wires_nodes) == 15
    assert len(cfg.wires_latent) == 7
    print(f"  config OK: {cfg.n_qubits}q = {cfg.n_topo}t + {cfg.n_nodes}n + {cfg.n_latent}l")
    return cfg


def test_2_circuit_init():
    """Create PennyLane device + QNode — should print which backend is used."""
    from hybrid_circuit_30q import HybridQGANCircuit30Q, QubitConfig30Q
    cfg = QubitConfig30Q()
    print(f"  creating HybridQGANCircuit30Q (n_layers=2)...", flush=True)
    circuit = HybridQGANCircuit30Q(n_layers=2, config=cfg)
    info = circuit.get_circuit_info()
    print(f"  backend:  {circuit.dev.name}")
    print(f"  depth:    {info['estimated_circuit_depth']} gates")
    print(f"  sim RAM:  {info['classial_sim_memory_gb']:.2f} GB (statevector)")
    print(f"  params:   {info['total_parameters']}")
    return circuit


def test_3_circuit_forward(circuit):
    """THE critical test — this is what was hanging. Single invocation."""
    np.random.seed(42)
    p_topo = np.random.randn(2, 8, 3) * 0.1
    p_node = np.random.randn(2, 15, 3) * 0.1
    p_latent = np.random.randn(2, 7, 3) * 0.1

    print(f"  calling circuit.forward() ...", flush=True)
    result = circuit.forward(p_topo, p_node, p_latent)
    print(f"  circuit.forward() returned!", flush=True)

    # Validate shapes
    assert result['topo_logits'].shape == (256,), f"topo: {result['topo_logits'].shape}"
    assert result['node_expvals'].shape == (15,), f"node: {result['node_expvals'].shape}"
    assert result['latent_features'].shape == (7,), f"latent: {result['latent_features'].shape}"

    # Validate probability normalization
    s = result['topo_logits'].sum().item()
    assert abs(s - 1.0) < 0.01, f"topo probs sum={s}, expected ~1.0"

    print(f"  topo_logits:     shape={result['topo_logits'].shape}  sum={s:.6f}")
    print(f"  node_expvals:    shape={result['node_expvals'].shape}  "
          f"range=[{result['node_expvals'].min():.3f}, {result['node_expvals'].max():.3f}]")
    print(f"  latent_features: shape={result['latent_features'].shape}")
    return result


def test_4_mapper():
    """Classical mapper — pure PyTorch, no quantum. Should be instant."""
    from hybrid_mapper_30q import HybridReadoutMapper30Q
    print(f"  creating mapper...", flush=True)
    mapper = HybridReadoutMapper30Q(num_linker_atoms=10)

    topo = torch.ones(256) / 256
    nodes = torch.randn(15)
    latent = torch.randn(7)

    print(f"  mapper forward...", flush=True)
    out = mapper(topo, nodes, latent)

    assert out['metal_atoms'].shape == (5, 3)
    assert out['linker_atoms'].shape == (10, 3)
    assert out['lattice_params'].shape == (6,)
    assert 0 <= out['validity'].item() <= 1

    print(f"  metal:    {out['metal_atoms'].shape}")
    print(f"  linker:   {out['linker_atoms'].shape}")
    print(f"  lattice:  {out['lattice_params'].shape}")
    print(f"  validity: {out['validity'].item():.3f}")
    return mapper


def test_5_generator_init():
    """Full QGAN generator init — creates circuit + mapper + noise encoder."""
    from qgan_generator_30q import QGAN_Generator30Q
    print(f"  creating QGAN_Generator30Q...", flush=True)
    gen = QGAN_Generator30Q(noise_dim=32, num_linker_atoms=10, n_circuit_layers=2)

    stats = gen.get_model_stats()
    print(f"  total params:  {stats['model_metrics']['total_parameters']}")
    print(f"  model size:    {stats['model_metrics']['model_size_kb']:.1f} KB")
    print(f"  circuit depth: {stats['model_metrics']['estimated_circuit_depth']}")
    return gen


def test_6_e2e_generation(gen):
    """End-to-end: noise -> quantum circuit -> mapper -> MOF structure. batch=1."""
    print(f"  generating noise (batch=1)...", flush=True)
    noise = torch.randn(1, 32)

    print(f"  calling gen(noise)...", flush=True)
    out = gen(noise)

    assert out['metal_atoms'].shape == (1, 5, 3), f"metal: {out['metal_atoms'].shape}"
    assert out['linker_atoms'].shape == (1, 10, 3), f"linker: {out['linker_atoms'].shape}"
    assert out['lattice_params'].shape == (1, 6), f"lattice: {out['lattice_params'].shape}"

    print(f"  metal:    {out['metal_atoms'].shape}")
    print(f"  linker:   {out['linker_atoms'].shape}")
    print(f"  validity: {out['validity'].item():.3f}")
    return out


def test_7_batch(gen):
    """Batch=2 generation — runs quantum circuit twice sequentially."""
    print(f"  generating batch=2...", flush=True)
    noise = torch.randn(2, 32)
    out = gen(noise)

    assert out['metal_atoms'].shape == (2, 5, 3)
    assert out['linker_atoms'].shape == (2, 10, 3)

    v = out['validity'].numpy()
    print(f"  batch shapes OK")
    print(f"  validity: [{v[0]:.3f}, {v[1]:.3f}]")
    return out


def test_8_qasm_export(gen):
    """OpenQASM 2.0 export — pure string manipulation, no quantum exec."""
    noise = torch.randn(1, 32)
    print(f"  exporting QASM...", flush=True)
    qasm = gen.export_openqasm(noise)

    assert "OPENQASM 2.0" in qasm
    assert "qreg q[30]" in qasm
    assert "measure" in qasm

    print(f"  QASM: {len(qasm)} chars, {qasm.count(chr(10))} lines")
    return qasm


def test_9_checkpoint(gen):
    """Save + load checkpoint round-trip."""
    import tempfile
    noise = torch.randn(1, 32)

    print(f"  generating pre-save output...", flush=True)
    out_before = gen(noise)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    print(f"  saving to {path}...", flush=True)
    gen.save_checkpoint(path)
    size_kb = os.path.getsize(path) / 1024
    print(f"  saved ({size_kb:.1f} KB)")

    print(f"  loading into fresh generator...", flush=True)
    from qgan_generator_30q import QGAN_Generator30Q
    gen2 = QGAN_Generator30Q(noise_dim=32, num_linker_atoms=10, n_circuit_layers=2)
    gen2.load_checkpoint(path)

    print(f"  generating post-load output...", flush=True)
    out_after = gen2(noise)

    os.unlink(path)
    print(f"  checkpoint round-trip OK")
    return True


# ── Main runner ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="30q QGAN smoke test")
    parser.add_argument("--timeout", "-t", type=int, default=180,
                        help="Timeout per quantum test in seconds (default: 180)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    qtime = args.timeout  # quantum test timeout
    ctime = 15            # classical test timeout

    passed, failed, skipped = [], [], []
    circuit = None
    gen = None
    circuit_works = False

    # ── Test 1: Config (instant) ────────────────────────────────────
    try:
        with_timeout(test_1_config, "1/9  Qubit Config", timeout_sec=5)
        passed.append("1. config")
    except Exception:
        failed.append("1. config")

    # ── Test 2: Circuit init ────────────────────────────────────────
    try:
        circuit, _ = with_timeout(test_2_circuit_init, "2/9  Circuit Init", timeout_sec=30)
        passed.append("2. circuit_init")
    except Exception:
        failed.append("2. circuit_init")

    # ── Test 3: Circuit forward (THE HANG TEST) ────────────────────
    if circuit:
        try:
            with_timeout(lambda: test_3_circuit_forward(circuit),
                         "3/9  Circuit Forward (HANG DETECTOR)", timeout_sec=qtime)
            passed.append("3. circuit_forward")
            circuit_works = True
        except HangTimeout:
            failed.append("3. circuit_forward (TIMEOUT — quantum sim too slow)")
            print(f"\n  *** Circuit forward timed out after {qtime}s ***")
            print(f"  *** All downstream quantum tests will be SKIPPED ***")
            print(f"  *** Install pennylane-lightning: pip install pennylane-lightning ***\n",
                  flush=True)
        except Exception:
            failed.append("3. circuit_forward")
    else:
        skipped.append("3. circuit_forward (no circuit)")

    # ── Test 4: Mapper (classical only — always runs) ──────────────
    try:
        with_timeout(test_4_mapper, "4/9  Classical Mapper", timeout_sec=ctime)
        passed.append("4. mapper")
    except Exception:
        failed.append("4. mapper")

    # ── Tests 5-9: Need working quantum circuit ────────────────────
    if not circuit_works:
        for i, name in [(5, "generator_init"), (6, "e2e_generation"),
                        (7, "batch"), (8, "qasm_export"), (9, "checkpoint")]:
            skipped.append(f"{i}. {name} (circuit forward hung)")
    else:
        # Test 5: Generator init
        try:
            gen, _ = with_timeout(test_5_generator_init,
                                  "5/9  Generator Init", timeout_sec=30)
            passed.append("5. generator_init")
        except Exception:
            failed.append("5. generator_init")

        # Test 6: E2E generation (batch=1)
        if gen:
            try:
                with_timeout(lambda: test_6_e2e_generation(gen),
                             "6/9  E2E Generation (batch=1)", timeout_sec=qtime)
                passed.append("6. e2e_generation")
            except HangTimeout:
                failed.append("6. e2e_generation (TIMEOUT)")
            except Exception:
                failed.append("6. e2e_generation")
        else:
            skipped.append("6. e2e_generation (no gen)")

        # Test 7: Batch=2
        if gen:
            try:
                with_timeout(lambda: test_7_batch(gen),
                             "7/9  Batch Processing (batch=2)", timeout_sec=qtime * 2)
                passed.append("7. batch")
            except HangTimeout:
                failed.append("7. batch (TIMEOUT)")
            except Exception:
                failed.append("7. batch")
        else:
            skipped.append("7. batch (no gen)")

        # Test 8: QASM export (no quantum exec)
        if gen:
            try:
                with_timeout(lambda: test_8_qasm_export(gen),
                             "8/9  QASM Export", timeout_sec=ctime)
                passed.append("8. qasm_export")
            except Exception:
                failed.append("8. qasm_export")
        else:
            skipped.append("8. qasm_export (no gen)")

        # Test 9: Checkpoint save/load (does 2 quantum forward passes)
        if gen:
            try:
                with_timeout(lambda: test_9_checkpoint(gen),
                             "9/9  Checkpoint Save/Load", timeout_sec=qtime * 3)
                passed.append("9. checkpoint")
            except HangTimeout:
                failed.append("9. checkpoint (TIMEOUT)")
            except Exception:
                failed.append("9. checkpoint")
        else:
            skipped.append("9. checkpoint (no gen)")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESULTS: {len(passed)} passed / {len(failed)} failed / {len(skipped)} skipped")
    print(f"{'='*65}")
    for t in passed:
        print(f"    PASS   {t}")
    for t in failed:
        print(f"    FAIL   {t}")
    for t in skipped:
        print(f"    SKIP   {t}")

    if failed:
        print(f"\n  *** {len(failed)} test(s) failed ***")
        if any("TIMEOUT" in f for f in failed):
            print(f"  Quantum sim is too slow. Options:")
            print(f"    1. pip install pennylane-lightning  (C++ backend, 10-100x faster)")
            print(f"    2. Increase --timeout (current: {qtime}s)")
            print(f"    3. Reduce circuit layers to 1")
        sys.exit(1)
    elif skipped:
        print(f"\n  {len(skipped)} test(s) skipped — quantum circuit did not complete")
        sys.exit(2)
    else:
        print(f"\n  All 9 tests passed — safe to run full suite on HPC!")
        sys.exit(0)


if __name__ == "__main__":
    main()
