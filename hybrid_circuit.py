"""
Hybrid QGAN Quantum Circuit: Topology + Geometry Generator
Generates both discrete topology (Node/Linker selection) and continuous geometry (lattice parameters).
[cite: Quantum GAN papers, Circuit optimization for MOF generation]
"""

from typing import Dict, List, Tuple
import pennylane as qml
import numpy as np
import torch


class HybridQGANCircuit:
    """
    Hybrid partitioned quantum circuit for MOF generation (9-qubit mini version for testing).
    Architecture follows 40-qubit design but scaled for computational feasibility.
    
    Qubit Partitioning (9-qubit version):
    - Topology Register (qubits 0-2): Discrete node/linker selection (3 qubits = 8 states)
    - Node Register (qubits 3-5): Direct readout of coordinates (3 qubits = 3 coords)
    - Latent Register (qubits 6-8): Feature encoding (3 qubits for latent features)
    
    Full 40-qubit version uses the same architecture with:
    - Topology Register: 10 qubits (1024 states)
    - Node Register: 18 qubits (6 atoms × 3 coords)
    - Latent Register: 12 qubits
    """
    
    def __init__(self, n_layers: int = 2, device_name: str = "default.qubit", mini: bool = True):
        """
        Initialize the Hybrid QGAN Circuit.
        
        Args:
            n_layers: Number of entangling layers in each branch
            device_name: PennyLane device name
            mini: If True, use 9-qubit version; if False, use 40-qubit version
        """
        self.mini = mini
        self.n_layers = n_layers
        
        if mini:
            # 9-qubit mini version (for testing/demo)
            self.n_qubits = 9
            self.wires_topo = list(range(3))        # Topology: qubits 0-2
            self.wires_nodes = list(range(3, 6))    # Node coords: qubits 3-5
            self.wires_latent = list(range(6, 9))   # Latent features: qubits 6-8
        else:
            # 40-qubit full version (for production)
            self.n_qubits = 40
            self.wires_topo = list(range(10))       # Topology: qubits 0-9
            self.wires_nodes = list(range(10, 28))  # Node coords: qubits 10-27
            self.wires_latent = list(range(28, 40)) # Latent features: qubits 28-39
        
        # Create device
        self.dev = qml.device(device_name, wires=self.n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev)
    
    def _circuit(self, params_topo: np.ndarray, params_node: np.ndarray, 
                 params_latent: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Full hybrid circuit with conditional geometry encoding.
        
        Args:
            params_topo: Parameters for topology register (shape: n_layers, 10, 3)
            params_node: Parameters for node register (shape: n_layers, 18, 3)
            params_latent: Parameters for latent register (shape: n_layers, 12, 3)
        
        Returns:
            Dictionary with topo_logits, node_expvals, and latent_features
        """
        
        # ===== LAYER 1: Global Entanglement (Seed) =====
        # Hadamard on all qubits
        for wire in range(self.n_qubits):
            qml.Hadamard(wires=wire)
        
        # Randomized layer: Ry rotations and CNOTs for initial entanglement
        for i in range(self.n_qubits - 1):
            qml.RY(0.5 * np.pi, wires=i)  # Fixed rotation for seed
            qml.CNOT(wires=[i, i + 1])
        
        # ===== LAYER 2: Topology Branch (Qubits 0-9) =====
        # Hard entanglement for discrete state selection
        # StronglyEntanglingLayers on topology qubits (expects shape: n_layers, n_wires, 3)
        qml.StronglyEntanglingLayers(
            weights=params_topo,
            wires=self.wires_topo,
            ranges=None,
            imprimitive=qml.CNOT
        )
        
        # ===== LAYER 3: Geometry Branch (Qubits 10-39) =====
        # Data Re-uploading: Condition geometry on topology via controlled rotations
        for i, topo_wire in enumerate(self.wires_topo[:3]):
            for j, node_wire in enumerate(self.wires_nodes[:3]):
                # CRX: controlled rotation based on topology state
                qml.CRX(0.1, wires=[topo_wire, node_wire])
        
        # Node register: BasicEntanglerLayers (direct mapping for metal atoms)
        for layer in range(self.n_layers):
            for i, wire in enumerate(self.wires_nodes):
                qml.RY(params_node[layer, i, 0], wires=wire)
                qml.RZ(params_node[layer, i, 1], wires=wire)
            # Entangle with CNOTs
            for i in range(len(self.wires_nodes) - 1):
                qml.CNOT(wires=[self.wires_nodes[i], self.wires_nodes[i + 1]])
        
        # Latent register: StronglyEntanglingLayers (high-density feature encoding)
        # Expects shape: n_layers, n_wires, 3
        qml.StronglyEntanglingLayers(
            weights=params_latent,
            wires=self.wires_latent,
            ranges=None,
            imprimitive=qml.CNOT
        )
        
        # ===== OUTPUT: Measurements =====
        # Topology: probability distribution (discrete)
        topo_probs = qml.probs(wires=self.wires_topo)
        
        # Node coordinates: expectation values of PauliZ (18 values -> 6 atoms * 3 coords)
        node_expvals = [qml.expval(qml.PauliZ(wire)) for wire in self.wires_nodes]
        
        # Latent features: expectation values of PauliZ (12 values for linker/lattice encoding)
        latent_expvals = [qml.expval(qml.PauliZ(wire)) for wire in self.wires_latent]
        
        return {
            'topo_logits': topo_probs,
            'node_expvals': node_expvals,
            'latent_features': latent_expvals
        }
    
    def forward(self, params_topo: np.ndarray, params_node: np.ndarray, 
                params_latent: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute the quantum circuit forward pass.
        
        Args:
            params_topo: Topology branch parameters
            params_node: Node branch parameters
            params_latent: Latent branch parameters
        
        Returns:
            Dictionary with quantum measurements
        """
        return self.qnode(params_topo, params_node, params_latent)
    
    def export_qasm(self) -> str:
        """
        Export circuit to OpenQASM 2.0 format for QARP compatibility.
        
        Returns:
            OpenQASM 2.0 string representation of a sample circuit
        """
        # Create parameter-free circuit for QASM export
        params_topo = np.random.random((self.n_layers, len(self.wires_topo), 3))
        params_node = np.random.random((self.n_layers, len(self.wires_nodes), 3))
        params_latent = np.random.random((self.n_layers, len(self.wires_latent), 3))
        
        qasm_str = "OPENQASM 2.0;\n"
        qasm_str += 'include "qelib1.inc";\n'
        qasm_str += f"qreg q[{self.n_qubits}];\n"
        qasm_str += f"creg c[{len(self.wires_topo)}];\n\n"
        
        qasm_str += "// Global Entanglement\n"
        for wire in range(self.n_qubits):
            qasm_str += f"h q[{wire}];\n"
        for i in range(self.n_qubits - 1):
            qasm_str += f"ry(pi/4) q[{i}];\n"
            qasm_str += f"cx q[{i}],q[{i + 1}];\n"
        
        qasm_str += "\n// Topology Branch (StronglyEntanglingLayers)\n"
        for wire in self.wires_topo:
            qasm_str += f"ry(0.5) q[{wire}];\n"
            qasm_str += f"rz(0.5) q[{wire}];\n"
        for i in range(len(self.wires_topo) - 1):
            qasm_str += f"cx q[{self.wires_topo[i]}],q[{self.wires_topo[i + 1]}];\n"
        
        qasm_str += "\n// Conditional Geometry (CRY gates)\n"
        for topo_wire in self.wires_topo[:3]:
            for node_wire in self.wires_nodes[:3]:
                qasm_str += f"// Controlled RY: control={topo_wire}, target={node_wire}\n"
                qasm_str += f"cry(0.1) q[{topo_wire}],q[{node_wire}];\n"
        
        qasm_str += "\n// Node Register Branch\n"
        for wire in self.wires_nodes:
            qasm_str += f"ry(0.5) q[{wire}];\n"
            qasm_str += f"rz(0.5) q[{wire}];\n"
        for i in range(len(self.wires_nodes) - 1):
            qasm_str += f"cx q[{self.wires_nodes[i]}],q[{self.wires_nodes[i + 1]}];\n"
        
        qasm_str += "\n// Latent Register Branch\n"
        for wire in self.wires_latent:
            qasm_str += f"ry(0.5) q[{wire}];\n"
            qasm_str += f"rz(0.5) q[{wire}];\n"
        for i in range(len(self.wires_latent) - 1):
            qasm_str += f"cx q[{self.wires_latent[i]}],q[{self.wires_latent[i + 1]}];\n"
        
        qasm_str += "\n// Measurement\n"
        for i, wire in enumerate(self.wires_topo):
            qasm_str += f"measure q[{wire}] -> c[{i}];\n"
        
        return qasm_str
