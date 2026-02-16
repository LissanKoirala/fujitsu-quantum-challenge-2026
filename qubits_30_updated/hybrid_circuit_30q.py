"""
Hybrid QGAN Quantum Circuit: 30-Qubit Configuration
Optimized for balanced topology, node, and latent register allocation.
Topology (8) + Node (15) + Latent (7) = 30 qubits total

[cite: Quantum GAN papers, Circuit optimization for MOF generation, Fujitsu Quantum Computing]
"""

from typing import Dict, Tuple, Optional, Union
import pennylane as qml
import numpy as np
import torch
from dataclasses import dataclass
import time
import sys
import os


@dataclass
class QubitConfig30Q:
    """Configuration for 30-qubit partitioning"""
    n_qubits: int = 30
    n_topo: int = 8      # Topology: 256 states
    n_nodes: int = 15    # Node coordinates: 5 atoms × 3 coords
    n_latent: int = 7    # Latent features
    
    def __post_init__(self):
        assert self.n_topo + self.n_nodes + self.n_latent == self.n_qubits
        self.wires_topo = list(range(self.n_topo))
        self.wires_nodes = list(range(self.n_topo, self.n_topo + self.n_nodes))
        self.wires_latent = list(range(self.n_topo + self.n_nodes, self.n_qubits))


class HybridQGANCircuit30Q:
    """
    Optimized 30-qubit Hybrid Quantum Circuit for MOF Generation.
    
    Qubit Allocation:
    - Qubits 0-7:   Topology Register (8 qubits => 256 discrete topology states)
    - Qubits 8-22:  Node Register (15 qubits => 5 metal atoms + coordinates)
    - Qubits 23-29: Latent Register (7 qubits => latent feature extraction)
    
    Circuit Depth Strategy:
    - Global entanglement seed: O(n) CNOT depth
    - Topology branch: StronglyEntangling (tunable depth)
    - Geometry branch: Controlled rotations + BasicEntangler
    - Expected total depth: ~150-200 gates (hardware-feasible)
    
    Memory Usage (on classical simulator):
    - Statevector: 2^30 = ~1GB (feasible on modern laptops)
    - Parameter count: ~O(n_layers × 3n) rotations = ~270-360 params
    """
    
    def __init__(
        self,
        n_layers: int = 2,
        device_name: str = "default.qubit",
        config: Optional[QubitConfig30Q] = None,
        seed: int = 42,
        split_registers: bool = False
    ):
        """
        Initialize the 30-Qubit Hybrid QGAN Circuit.

        Args:
            n_layers: Number of entangling layers (1-3 recommended for 30q)
            device_name: PennyLane device ("default.qubit", "qasm_simulator", etc.)
            config: QubitConfig30Q instance (uses default if None)
            seed: Random seed for reproducibility
            split_registers: If True, run 3 independent small circuits (8q+15q+7q)
                instead of one 30q circuit. Uses ~500MB instead of ~30GB.
                Set True for laptop testing, False for HPC/hardware.
        """
        np.random.seed(seed)
        self.config = config or QubitConfig30Q()
        self.n_layers = n_layers
        self.device_name = device_name
        self.split_registers = split_registers

        def _make_device(n_wires, label=""):
            """Create a quantum device, preferring lightning.qubit."""
            if device_name == "default.qubit":
                try:
                    dev = qml.device("lightning.qubit", wires=n_wires)
                    print(f"[circuit] Using lightning.qubit (C++ optimized) for {n_wires} qubits {label}")
                    return dev
                except Exception:
                    dev = qml.device(device_name, wires=n_wires)
                    print(f"[circuit] Falling back to default.qubit for {n_wires} qubits {label}")
                    return dev
            else:
                return qml.device(device_name, wires=n_wires)

        if split_registers:
            # === SPLIT MODE: 3 independent circuits (~500MB total) ===
            print(f"[circuit] SPLIT MODE: running 3 independent registers "
                  f"({self.config.n_topo}q + {self.config.n_nodes}q + {self.config.n_latent}q)")
            self.dev_topo = _make_device(self.config.n_topo, "(topology)")
            self.dev_node = _make_device(self.config.n_nodes, "(node)")
            self.dev_latent = _make_device(self.config.n_latent, "(latent)")

            self.qnode_topo = qml.QNode(self._circuit_topo, self.dev_topo)
            self.qnode_node = qml.QNode(self._circuit_node, self.dev_node)
            self.qnode_latent = qml.QNode(self._circuit_latent, self.dev_latent)

            # Aliases for compatibility
            self.dev = self.dev_topo
            self.qnode = None  # Not used in split mode
            self.qnode_grad = None
        else:
            # === FULL MODE: single 30-qubit circuit (needs ~16GB+ RAM) ===
            self.dev = _make_device(self.config.n_qubits)
            self.qnode = qml.QNode(self._circuit, self.dev)
            self.qnode_grad = qml.QNode(self._circuit_with_grads, self.dev, diff_method="parameter-shift")
    
    # =====================================================================
    # SPLIT-REGISTER CIRCUITS (laptop-safe, ~500MB total)
    # =====================================================================

    def _circuit_topo(self, params_topo: np.ndarray):
        """Topology register only (8 qubits → 256 probabilities)."""
        wires = list(range(self.config.n_topo))
        for w in wires:
            qml.Hadamard(wires=w)
        # CNOT ladder within register
        for w in range(len(wires) - 1):
            qml.CNOT(wires=[wires[w], wires[w + 1]])
        qml.StronglyEntanglingLayers(
            weights=params_topo, wires=wires, ranges=None, imprimitive=qml.CNOT
        )
        return qml.probs(wires=wires)

    def _circuit_node(self, params_node: np.ndarray):
        """Node register only (15 qubits → 15 expvals)."""
        wires = list(range(self.config.n_nodes))
        for w in wires:
            qml.Hadamard(wires=w)
        for layer in range(self.n_layers):
            for i, w in enumerate(wires):
                qml.RX(params_node[layer, i, 0], wires=w)
                qml.RY(params_node[layer, i, 1], wires=w)
                qml.RZ(params_node[layer, i, 2], wires=w)
            for i in range(0, len(wires) - 1, 2):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    def _circuit_latent(self, params_latent: np.ndarray):
        """Latent register only (7 qubits → 7 expvals)."""
        wires = list(range(self.config.n_latent))
        for w in wires:
            qml.Hadamard(wires=w)
        qml.StronglyEntanglingLayers(
            weights=params_latent, wires=wires, ranges=None, imprimitive=qml.CNOT
        )
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    # =====================================================================
    # FULL 30-QUBIT CIRCUIT (HPC only)
    # =====================================================================

    def _circuit(
        self,
        params_topo: np.ndarray,
        params_node: np.ndarray,
        params_latent: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Complete 30-qubit hybrid circuit.
        
        Args:
            params_topo: Shape (n_layers, 8, 3) - topology entangling parameters
            params_node: Shape (n_layers, 15, 3) - node rotation parameters
            params_latent: Shape (n_layers, 7, 3) - latent rotation parameters
        
        Returns:
            {
                'topo_logits': np.array of shape (256,) - topology state probabilities
                'node_expvals': np.array of shape (15,) - node coordinate expectation values
                'latent_features': np.array of shape (7,) - latent feature values
            }
        """
        
        # ===== STAGE 1: GLOBAL SEED ENTANGLEMENT =====
        # Initialize all qubits with Hadamard (equal superposition)
        for wire in range(self.config.n_qubits):
            qml.Hadamard(wires=wire)
        
        # Create dense entanglement via CNOT ladder
        # This ensures information from all registers is mixed
        for depth in range(min(2, self.n_layers)):
            for wire in range(self.config.n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
            # Reverse direction for 2nd pass
            for wire in range(self.config.n_qubits - 2, 0, -1):
                qml.CNOT(wires=[wire, wire - 1])
        
        # ===== STAGE 2: TOPOLOGY BRANCH (8 QUBITS) =====
        # Discrete state selection via StronglyEntanglingLayers
        # Shape: (n_layers, 8, 3) where 3 are RX, RY, RZ parameters
        qml.StronglyEntanglingLayers(
            weights=params_topo,
            wires=self.config.wires_topo,
            ranges=None,
            imprimitive=qml.CNOT
        )
        
        # ===== STAGE 3: DATA RE-UPLOADING (Topology → Node Conditioning) =====
        # Controlled rotations based on topology state to encode geometry
        # Uses linear pairing (5 CRX gates) instead of quadratic (25 CRX gates)
        # to keep simulation tractable on classical hardware
        n_cross = min(5, len(self.config.wires_topo), len(self.config.wires_nodes))
        for i in range(n_cross):
            qml.CRX(0.15 * np.pi, wires=[
                self.config.wires_topo[i],
                self.config.wires_nodes[i]
            ])
        
        # ===== STAGE 4: NODE REGISTER (15 QUBITS) =====
        # Direct coordinate encoding via BasicEntangler (lower depth than StronglyEntangling)
        for layer in range(self.n_layers):
            for i, wire in enumerate(self.config.wires_nodes):
                qml.RX(params_node[layer, i, 0], wires=wire)
                qml.RY(params_node[layer, i, 1], wires=wire)
                qml.RZ(params_node[layer, i, 2], wires=wire)
            
            # Light entanglement: alternating CNOT pattern
            for i in range(0, len(self.config.wires_nodes) - 1, 2):
                qml.CNOT(wires=[
                    self.config.wires_nodes[i],
                    self.config.wires_nodes[i + 1]
                ])
        
        # ===== STAGE 5: LATENT REGISTER (7 QUBITS) =====
        # StronglyEntangling for rich feature extraction
        qml.StronglyEntanglingLayers(
            weights=params_latent,
            wires=self.config.wires_latent,
            ranges=None,
            imprimitive=qml.CNOT
        )
        
        # ===== STAGE 6: MEASUREMENTS =====
        # Topology: Computational basis sampling (bitstring)
        topo_probs = qml.probs(wires=self.config.wires_topo)
        
        # Node & Latent: Expectation values (< Z >)
        node_expvals = [
            qml.expval(qml.PauliZ(wire)) for wire in self.config.wires_nodes
        ]
        latent_features = [
            qml.expval(qml.PauliZ(wire)) for wire in self.config.wires_latent
        ]
        
        # Return measurements as a sequence (compatible with PennyLane QNode)
        # Order: topo_probs, then all node expvals, then all latent features
        return (topo_probs, *node_expvals, *latent_features)
    
    def _circuit_with_grads(
        self,
        params_topo: np.ndarray,
        params_node: np.ndarray,
        params_latent: np.ndarray
    ) -> float:
        """
        Circuit variant for gradient computation via parameter shift rule.
        Returns a scalar for optimization.
        """
        result = self._circuit(params_topo, params_node, params_latent)
        # Unpack tuple: (topo_probs, *node_expvals, *latent_features)
        topo_logits = result[0]
        node_expvals = np.array(result[1:1+self.config.n_nodes])
        
        # Loss: maximize entropy of topology + magnitude of node expvals
        topo_entropy = -np.sum(topo_logits * np.log(topo_logits + 1e-10))
        node_magnitude = np.sum(np.abs(node_expvals))
        return -(topo_entropy + node_magnitude)  # Negate for minimization
    
    def forward(
        self,
        params_topo: np.ndarray,
        params_node: np.ndarray,
        params_latent: np.ndarray,
        return_raw: bool = False
    ) -> Union[Dict, Tuple]:
        """
        Execute the quantum circuit.

        Args:
            params_topo: Topology parameters
            params_node: Node parameters
            params_latent: Latent parameters
            return_raw: If True, return raw numpy dict; if False, return torch tensors

        Returns:
            Dictionary or tuple of torch tensors (based on return_raw flag)
        """
        pid = os.getpid()
        t0 = time.time()

        if self.split_registers:
            # === SPLIT MODE: run 3 small circuits independently ===
            print(f"  [circuit.forward pid={pid}] SPLIT mode: "
                  f"{self.config.n_topo}q + {self.config.n_nodes}q + {self.config.n_latent}q ...",
                  flush=True)

            topo_logits = np.array(self.qnode_topo(params_topo))
            t_topo = time.time()
            print(f"  [circuit.forward pid={pid}] topo done ({t_topo - t0:.2f}s)", flush=True)

            node_result = self.qnode_node(params_node)
            node_expvals = np.array(node_result)
            t_node = time.time()
            print(f"  [circuit.forward pid={pid}] node done ({t_node - t0:.2f}s)", flush=True)

            latent_result = self.qnode_latent(params_latent)
            latent_features = np.array(latent_result)
            t_latent = time.time()
            print(f"  [circuit.forward pid={pid}] latent done ({t_latent - t0:.2f}s)", flush=True)

        else:
            # === FULL MODE: single 30-qubit QNode ===
            print(f"  [circuit.forward pid={pid}] FULL mode: "
                  f"{self.config.n_qubits}q (device={self.dev.name})...",
                  flush=True)

            result = self.qnode(params_topo, params_node, params_latent)
            t1 = time.time()
            print(f"  [circuit.forward pid={pid}] QNode done in {t1 - t0:.2f}s", flush=True)

            topo_logits = result[0]
            node_expvals = np.array(result[1:1+self.config.n_nodes])
            latent_features = np.array(result[1+self.config.n_nodes:])

        output_dict = {
            'topo_logits': topo_logits,
            'node_expvals': node_expvals,
            'latent_features': latent_features
        }

        if return_raw:
            return output_dict

        # Convert to torch tensors
        t2 = time.time()
        out = {
            'topo_logits': torch.tensor(np.asarray(output_dict['topo_logits']).tolist(), dtype=torch.float32),
            'node_expvals': torch.tensor(np.asarray(output_dict['node_expvals']).tolist(), dtype=torch.float32),
            'latent_features': torch.tensor(np.asarray(output_dict['latent_features']).tolist(), dtype=torch.float32)
        }
        print(f"  [circuit.forward pid={pid}] total={time.time() - t0:.2f}s",
              flush=True)
        return out
    
    def get_circuit_info(self) -> Dict:
        """Return detailed circuit analysis and statistics"""
        n_params_topo = self.n_layers * self.config.n_topo * 3
        n_params_node = self.n_layers * self.config.n_nodes * 3
        n_params_latent = self.n_layers * self.config.n_latent * 3
        total_params = n_params_topo + n_params_node + n_params_latent

        # Estimate circuit depth
        depth_seed = 4 * (self.n_layers - 1)  # Global entanglement
        depth_topo = self.n_layers * (self.config.n_topo + 2)  # StronglyEntangling
        depth_node = self.n_layers * (self.config.n_nodes + 1)  # BasicEntangler
        depth_latent = self.n_layers * (self.config.n_latent + 2)  # StronglyEntangling
        total_depth = depth_seed + depth_topo + depth_node + depth_latent

        if self.split_registers:
            # In split mode, memory is max of the 3 registers (15q is largest)
            sim_memory_gb = 2**(self.config.n_nodes) * 16 / (1024**3)
        else:
            sim_memory_gb = 2**(self.config.n_qubits) * 16 / (1024**3)

        return {
            'n_qubits': self.config.n_qubits,
            'n_layers': self.n_layers,
            'split_registers': self.split_registers,
            'total_parameters': total_params,
            'parameters_breakdown': {
                'topology': n_params_topo,
                'node': n_params_node,
                'latent': n_params_latent
            },
            'estimated_circuit_depth': total_depth,
            'estimated_2qubit_gates': int(total_depth * 0.3),
            'classial_sim_memory_gb': sim_memory_gb,
        }


if __name__ == "__main__":
    # Quick test — uses split mode by default (laptop-safe)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Use full 30q circuit (needs ~16GB+ RAM, for HPC only)")
    args = parser.parse_args()

    split = not args.full
    config = QubitConfig30Q()
    circuit = HybridQGANCircuit30Q(n_layers=2, config=config, split_registers=split)

    # Random parameters
    p_topo = np.random.randn(2, 8, 3)
    p_node = np.random.randn(2, 15, 3)
    p_latent = np.random.randn(2, 7, 3)

    result = circuit.forward(p_topo, p_node, p_latent)

    print("30-Qubit Circuit Test:")
    print(f"Topology logits shape: {result['topo_logits'].shape}")
    print(f"Node expvals shape: {result['node_expvals'].shape}")
    print(f"Latent features shape: {result['latent_features'].shape}")

    info = circuit.get_circuit_info()
    print("\nCircuit Info:")
    for key, val in info.items():
        if key != "parameters_breakdown":
            print(f"  {key}: {val}")
