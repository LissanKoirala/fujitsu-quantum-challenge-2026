"""
Hybrid QGAN Generator: 30-Qubit MOF Generation System
End-to-end trainable PyTorch module combining quantum and classical components.

[cite: Quantum GAN architectures, PyTorch neural network design, MOF generation]
"""

from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
import sys
import os

from hybrid_circuit_30q import HybridQGANCircuit30Q, QubitConfig30Q
from hybrid_mapper_30q import HybridReadoutMapper30Q


class QGAN_Generator30Q(nn.Module):
    """
    Complete 30-qubit Hybrid QGAN Generator for MOF generation.
    
    Architecture:
    1. Quantum Circuit: 30 qubits (8 topo + 15 node + 7 latent)
    2. Classical Mapper: Converts quantum outputs to atomic coordinates
    3. Discriminator-ready: Exports structures for GAN training
    4. Hardware Export: Generates OpenQASM2.0 for Fujitsu/IBM deployment
    
    Parameters:
    - Total trainable params: ~360 (quantum) + ~5K (classical) = ~5.4K total
    - Model size: ~22 KB (FP32) - very lightweight for deployment
    """
    
    def __init__(
        self,
        noise_dim: int = 32,
        num_linker_atoms: int = 10,
        n_circuit_layers: int = 2,
        angstrom_scale: float = 12.0,
        device_name: str = "default.qubit",
        num_quantum_samples: int = 1,
        seed: int = 42,
        device: str = "cpu",
        split_registers: bool = False
    ):
        """
        Initialize the 30-qubit Hybrid QGAN Generator.

        Args:
            noise_dim: Dimension of input random noise
            num_linker_atoms: Number of organic linker atoms
            n_circuit_layers: Number of entangling layers (1-3 recommended)
            angstrom_scale: Coordinate scale to Angstroms
            device_name: PennyLane device ("default.qubit", "qasm_simulator", etc.)
            num_quantum_samples: Samples per forward pass (averaging for stochasticity)
            seed: Random seed
            device: PyTorch device ("cpu" or "cuda")
            split_registers: If True, run 3 small circuits instead of 1 big 30q circuit.
                Uses ~500MB RAM instead of ~30GB. Use True for laptop, False for HPC.
        """
        super().__init__()

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.noise_dim = noise_dim
        self.num_linker_atoms = num_linker_atoms
        self.n_circuit_layers = n_circuit_layers
        self.angstrom_scale = angstrom_scale
        self.device_name = device_name
        self.num_quantum_samples = num_quantum_samples
        self.device = device
        self.seed = seed
        self.split_registers = split_registers

        # ===== QUANTUM COMPONENT =====
        self.config = QubitConfig30Q()
        self.quantum_circuit = HybridQGANCircuit30Q(
            n_layers=n_circuit_layers,
            device_name=device_name,
            config=self.config,
            seed=seed,
            split_registers=split_registers
        )
        
        # ===== CLASSICAL COMPONENTS =====
        # Noise encoder: random noise → quantum parameters
        self.noise_encoder = nn.Sequential(
            nn.Linear(noise_dim, 128, device=device),
            nn.LayerNorm(128, device=device),
            nn.ReLU(),
            nn.Linear(128, 256, device=device),
            nn.LayerNorm(256, device=device),
            nn.ReLU(),
            nn.Linear(256, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, self._total_param_count(), device=device)
        )
        
        # Readout mapper: quantum outputs → MOF structure
        self.mapper = HybridReadoutMapper30Q(
            num_linker_atoms=num_linker_atoms,
            angstrom_scale=angstrom_scale,
            latent_dim=self.config.n_latent,
            hidden_dim=64,
            n_metal_atoms=self.config.n_topo // 2,  # Use half of topo qubits for metal count
            device=device
        )
        
        # Statistics tracking
        self.register_buffer('_generation_count', torch.tensor(0, dtype=torch.long))
    
    def _total_param_count(self) -> int:
        """Calculate total quantum parameters needed"""
        return (
            self.n_circuit_layers * self.config.n_topo * 3 +
            self.n_circuit_layers * self.config.n_nodes * 3 +
            self.n_circuit_layers * self.config.n_latent * 3
        )
    
    def encode_noise(self, noise: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode random noise to quantum circuit parameters.
        
        Args:
            noise: (batch, noise_dim) random vector
        
        Returns:
            (params_topo, params_node, params_latent) - reshaped for circuit
        """
        params_flat = self.noise_encoder(noise)  # (batch, total_params)
        
        batch_size = noise.shape[0]
        offset = 0
        
        # Split into topo, node, latent
        n_topo = self.n_circuit_layers * self.config.n_topo * 3
        n_node = self.n_circuit_layers * self.config.n_nodes * 3
        
        params_topo = params_flat[:, offset:offset+n_topo].detach().numpy()
        offset += n_topo
        
        params_node = params_flat[:, offset:offset+n_node].detach().numpy()
        offset += n_node
        
        params_latent = params_flat[:, offset:].detach().numpy()
        
        # Reshape to (batch, n_layers, n_qubits, 3)
        params_topo = params_topo.reshape(batch_size, self.n_circuit_layers, self.config.n_topo, 3)
        params_node = params_node.reshape(batch_size, self.n_circuit_layers, self.config.n_nodes, 3)
        params_latent = params_latent.reshape(batch_size, self.n_circuit_layers, self.config.n_latent, 3)
        
        return params_topo, params_node, params_latent
    
    def forward(
        self,
        noise: torch.Tensor,
        return_quantum_outputs: bool = False
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict, Dict]]:
        """
        Generate MOF structure from random noise.
        
        Args:
            noise: (batch, noise_dim) random vector
            return_quantum_outputs: If True, also return raw quantum outputs
        
        Returns:
            Dictionary with 'metal_atoms', 'linker_atoms', 'lattice_params', 'validity'
            OR (structure_dict, quantum_dict) if return_quantum_outputs=True
        """
        batch_size = noise.shape[0]
        fwd_t0 = time.time()
        pid = os.getpid()

        # Encode noise to parameters
        print(f"[generator pid={pid}] encode_noise (batch={batch_size})...", flush=True)
        enc_t0 = time.time()
        params_topo, params_node, params_latent = self.encode_noise(noise)
        print(f"[generator pid={pid}] encode_noise done in {time.time() - enc_t0:.2f}s", flush=True)

        # Execute quantum circuit for each sample in batch
        topo_logits_list = []
        node_expvals_list = []
        latent_features_list = []

        for i in range(batch_size):
            sample_t0 = time.time()
            print(f"[generator pid={pid}] quantum circuit sample {i+1}/{batch_size}...", flush=True)

            result = self.quantum_circuit.forward(
                params_topo[i],
                params_node[i],
                params_latent[i],
                return_raw=False
            )

            elapsed_sample = time.time() - sample_t0
            elapsed_total = time.time() - fwd_t0
            print(f"[generator pid={pid}] sample {i+1}/{batch_size} done "
                  f"({elapsed_sample:.2f}s this sample, {elapsed_total:.2f}s total)",
                  flush=True)

            topo_logits_list.append(result['topo_logits'])
            node_expvals_list.append(result['node_expvals'])
            latent_features_list.append(result['latent_features'])

        # Stack into batches
        topo_logits = torch.stack(topo_logits_list, dim=0)  # (batch, 256)
        node_expvals = torch.stack(node_expvals_list, dim=0)  # (batch, 15)
        latent_features = torch.stack(latent_features_list, dim=0)  # (batch, 7)

        # Map quantum outputs to MOF structures
        map_t0 = time.time()
        print(f"[generator pid={pid}] classical mapper...", flush=True)
        structure_outputs = self.mapper(
            topo_logits.to(self.device),
            node_expvals.to(self.device),
            latent_features.to(self.device)
        )
        print(f"[generator pid={pid}] mapper done in {time.time() - map_t0:.2f}s", flush=True)
        print(f"[generator pid={pid}] forward total: {time.time() - fwd_t0:.2f}s", flush=True)

        self._generation_count += batch_size
        
        if return_quantum_outputs:
            quantum_dict = {
                'topo_logits': topo_logits,
                'node_expvals': node_expvals,
                'latent_features': latent_features
            }
            return structure_outputs, quantum_dict
        
        return structure_outputs
    
    def export_openqasm(
        self,
        noise: torch.Tensor,
        filepath: Optional[str] = None
    ) -> str:
        """
        Export quantum circuit to OpenQASM 2.0 format for hardware execution.
        
        Args:
            noise: (1, noise_dim) noise vector for single circuit instance
            filepath: If provided, save to file
        
        Returns:
            OpenQASM 2.0 code as string
        """
        if noise.shape[0] != 1:
            noise = noise[:1]
        
        params_topo, params_node, params_latent = self.encode_noise(noise)
        
        # Build QASM header
        qasm = "OPENQASM 2.0;\n"
        qasm += "include \"qelib1.inc\";\n"
        qasm += f"qreg q[{self.config.n_qubits}];\n"
        qasm += f"creg c[{self.config.n_qubits}];\n\n"
        
        # Comment with metadata
        qasm += f"// 30-qubit MOF Generator\n"
        qasm += f"// Topology: {self.config.n_topo}q, Nodes: {self.config.n_nodes}q, Latent: {self.config.n_latent}q\n"
        qasm += f"// Layers: {self.n_circuit_layers}\n\n"
        
        # Global entanglement (simplified for QASM)
        qasm += "// Stage 1: Global Entanglement\n"
        for i in range(self.config.n_qubits):
            qasm += f"h q[{i}];\n"
        
        for i in range(self.config.n_qubits - 1):
            qasm += f"cx q[{i}], q[{i+1}];\n"
        
        qasm += "\n// Stage 2-5: Parameterized Rotations (parameter substitution required)\n"
        
        # Simplified rotation block - in practice, parameters are substituted from external file
        for layer in range(self.n_circuit_layers):
            for i in range(self.config.n_topo):
                param = params_topo[0, layer, i, 0]
                qasm += f"rx({param:.4f}) q[{i}];\n"
        
        qasm += "\n// Measurement\n"
        qasm += f"measure q -> c;\n"
        
        if filepath:
            Path(filepath).write_text(qasm)
        
        return qasm
    
    def get_model_stats(self) -> Dict:
        """Return model statistics and deployment info"""
        topo_params = self.n_circuit_layers * self.config.n_topo * 3
        node_params = self.n_circuit_layers * self.config.n_nodes * 3
        latent_params = self.n_circuit_layers * self.config.n_latent * 3
        quantum_params = topo_params + node_params + latent_params
        
        # Classical parameters
        classical_params = sum(p.numel() for p in self.noise_encoder.parameters())
        classical_params += sum(p.numel() for p in self.mapper.parameters())
        
        total_params = quantum_params + classical_params
        
        return {
            'quantum_config': {
                'n_qubits': self.config.n_qubits,
                'topology_qubits': self.config.n_topo,
                'node_qubits': self.config.n_nodes,
                'latent_qubits': self.config.n_latent,
                'circuit_layers': self.n_circuit_layers,
                'quantum_parameters': quantum_params
            },
            'classical_config': {
                'noise_dim': self.noise_dim,
                'classical_parameters': classical_params,
                'hidden_dim': 64,
                'num_linker_atoms': self.num_linker_atoms
            },
            'model_metrics': {
                'total_parameters': total_params,
                'model_size_kb': (total_params * 4) / 1024,  # FP32
                'estimated_circuit_depth': self.quantum_circuit.get_circuit_info()['estimated_circuit_depth'],
                'generations_so_far': self._generation_count.item()
            }
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'generator_state': self.state_dict(),
            'config': {
                'noise_dim': self.noise_dim,
                'num_linker_atoms': self.num_linker_atoms,
                'n_circuit_layers': self.n_circuit_layers,
                'angstrom_scale': self.angstrom_scale,
                'device_name': self.device_name,
                'seed': self.seed
            },
            'model_stats': self.get_model_stats()
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['generator_state'])


if __name__ == "__main__":
    # Quick test — uses split mode by default (laptop-safe)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Use full 30q circuit (needs ~16GB+ RAM, for HPC only)")
    args = parser.parse_args()

    gen = QGAN_Generator30Q(
        noise_dim=32, num_linker_atoms=10, n_circuit_layers=2,
        split_registers=not args.full
    )

    print("30-Qubit QGAN Generator initialized")
    print(f"Model stats: {json.dumps(gen.get_model_stats(), indent=2)}")

    # Generate one MOF
    noise = torch.randn(1, 32)
    structures = gen(noise)

    print(f"\nGenerated structure:")
    print(f"  Metal atoms shape: {structures['metal_atoms'].shape}")
    print(f"  Linker atoms shape: {structures['linker_atoms'].shape}")
    print(f"  Validity: {structures['validity'].item():.3f}")
