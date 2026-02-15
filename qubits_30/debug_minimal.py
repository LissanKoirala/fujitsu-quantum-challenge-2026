"""
Minimal debug script to isolate hang point
"""
import sys
import time
from datetime import datetime

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}", flush=True)

try:
    log("=== IMPORT STAGE ===")
    log("Importing torch...")
    import torch
    log("✓ torch imported")
    
    log("Importing numpy...")
    import numpy as np
    log("✓ numpy imported")
    
    log("Importing pennylane...")
    import pennylane as qml
    log("✓ pennylane imported")
    
    log("\n=== LOCAL MODULES ===")
    log("Importing QubitConfig30Q...")
    from hybrid_circuit_30q import QubitConfig30Q
    log("✓ QubitConfig30Q imported")
    
    log("Importing HybridQGANCircuit30Q...")
    from hybrid_circuit_30q import HybridQGANCircuit30Q
    log("✓ HybridQGANCircuit30Q imported")
    
    log("\n=== TEST 1: CONFIG ===")
    log("Creating QubitConfig30Q()...")
    start = time.time()
    config = QubitConfig30Q()
    log(f"✓ Config created ({time.time()-start:.3f}s)")
    
    log("\n=== TEST 2: CIRCUIT INIT ===")
    log("Creating HybridQGANCircuit30Q(n_layers=1, config=config)...")
    start = time.time()
    circuit = HybridQGANCircuit30Q(n_layers=1, config=config)
    elapsed = time.time() - start
    log(f"✓ Circuit created ({elapsed:.3f}s)")
    
    log("\n=== TEST 3: CIRCUIT INFO ===")
    log("Calling circuit.get_circuit_info()...")
    start = time.time()
    info = circuit.get_circuit_info()
    elapsed = time.time() - start
    log(f"✓ Circuit info retrieved ({elapsed:.3f}s)")
    log(f"  - n_qubits: {info['n_qubits']}")
    log(f"  - depth: {info['estimated_circuit_depth']}")
    
    log("\n=== TEST 4: PARAMETERS ===")
    log("Generating parameters...")
    p_topo = np.random.randn(1, 8, 3) * 0.1
    p_node = np.random.randn(1, 15, 3) * 0.1
    p_latent = np.random.randn(1, 7, 3) * 0.1
    log(f"✓ Parameters generated")
    log(f"  - p_topo shape: {p_topo.shape}")
    log(f"  - p_node shape: {p_node.shape}")
    log(f"  - p_latent shape: {p_latent.shape}")
    
    log("\n=== TEST 5: FORWARD PASS (THIS CAN HANG!) ===")
    log("Calling circuit.forward(...)")
    log("  HANG DETECTION: Will print checkpoint every 5 seconds")
    
    start = time.time()
    checkpoint_interval = 5.0
    last_checkpoint = start
    
    # Add timeout wrapper
    import signal
    
    class TimeoutException(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutException("Forward pass timed out!")
    
    # Set 30 second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    try:
        output = circuit.forward(p_topo, p_node, p_latent)
        signal.alarm(0)  # Cancel alarm
        elapsed = time.time() - start
        log(f"✓ Forward pass completed ({elapsed:.3f}s)")
        log(f"  - topo_logits shape: {output['topo_logits'].shape}")
        log(f"  - node_expvals shape: {output['node_expvals'].shape}")
        log(f"  - latent_features shape: {output['latent_features'].shape}")
    except TimeoutException as e:
        log(f"✗ TIMEOUT: {e}")
        sys.exit(1)
    
    log("\n=== ALL TESTS PASSED ===")
    
except Exception as e:
    import traceback
    log(f"\n✗ ERROR: {type(e).__name__}: {e}")
    log(traceback.format_exc())
    sys.exit(1)
