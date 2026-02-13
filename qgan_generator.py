"""
Hybrid QGAN Generator: Complete Quantum-Classical MOF Generator
Wraps HybridQGANCircuit and HybridReadoutMapper into an end-to-end PyTorch module.
[cite: QGAN architectures, High-performance quantum computing integration]
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Import the quantum circuit and mapper
from hybrid_circuit import HybridQGANCircuit
from hybrid_mapper import HybridReadoutMapper


class QGAN_Generator(nn.Module):
    """
    Complete Hybrid QGAN Generator for MOF generation.
    
    Architecture:
    1. Quantum Circuit: 40-qubit partitioned system (10 topo + 18 node + 12 latent)
    2. Readout Mapper: Converts quantum outputs to atomic coordinates and lattice
    3. Safety Checks: Validates physical feasibility
    4. Serialization: Exports to OpenQASM 2.0 for QARP compatibility
    """
    
    def __init__(
        self,
        noise_dim: int = 32,
        num_linker_atoms: int = 12,
        n_circuit_layers: int = 2,
        angstrom_scale: float = 15.0,
        device_name: str = "default.qubit",
        num_quantum_samples: int = 1,
        use_mini_circuit: bool = True,
    ):
        """
        Initialize the Hybrid QGAN Generator.
        
        Args:
            noise_dim: Dimension of input noise vector
            num_linker_atoms: Number of linker atoms in MOF
            n_circuit_layers: Number of entangling layers in quantum circuit
            angstrom_scale: Scale for coordinate normalization (Angstroms)
            device_name: PennyLane device name
            num_quantum_samples: Number of samples per forward pass (for averaging)
            use_mini_circuit: If True, use 9-qubit mini circuit; if False, use 40-qubit
        """
        super().__init__()
        
        self.noise_dim = noise_dim
        self.num_linker_atoms = num_linker_atoms
        self.n_circuit_layers = n_circuit_layers
        self.angstrom_scale = angstrom_scale
        self.device_name = device_name
        self.num_quantum_samples = num_quantum_samples
        self.use_mini_circuit = use_mini_circuit
        
        # Quantum circuit initialization
        self.quantum_circuit = HybridQGANCircuit(
            n_layers=n_circuit_layers,
            device_name=device_name,
            mini=use_mini_circuit
        )
        
        # Classical readout mapper
        self.mapper = HybridReadoutMapper(
            num_linker_atoms=num_linker_atoms,
            angstrom_scale=angstrom_scale,
            latent_dim=12,
            hidden_dim=64,
        )
        
        # Noise-to-parameters encoder (classical)
        self.noise_encoder = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
        )
        
        # Parameter generators from encoded noise
        # Output: 3 sets of parameters for (topo, node, latent) branches
        n_topo_params = self.n_circuit_layers * len(self.quantum_circuit.wires_topo) * 3
        n_node_params = self.n_circuit_layers * len(self.quantum_circuit.wires_nodes) * 3
        n_latent_params = self.n_circuit_layers * len(self.quantum_circuit.wires_latent) * 3
        
        self.topo_param_gen = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_topo_params),
        )
        
        self.node_param_gen = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_node_params),
        )
        
        self.latent_param_gen = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_latent_params),
        )
        
        self.n_topo_params = n_topo_params
        self.n_node_params = n_node_params
        self.n_latent_params = n_latent_params
    
    def forward(self, z: torch.Tensor, return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generate MOF structure from noise vector.
        
        Args:
            z: Input noise tensor (batch_size, noise_dim)
            return_intermediate: If True, return quantum outputs before mapping
        
        Returns:
            Dictionary containing:
            - 'coords': Atomic coordinates (batch_size, total_atoms, 3)
            - 'lattice': Lattice parameters (batch_size, 6)
            - 'node_mask': Metal atom mask (batch_size, total_atoms)
            - 'topo_logits': Topology probabilities (batch_size, 1024)
            - (optional) intermediate quantum outputs
        """
        batch_size = z.shape[0]
        device = z.device
        
        # Encode noise to parameters
        encoded = self.noise_encoder(z)
        
        # Generate quantum circuit parameters
        params_topo_flat = self.topo_param_gen(encoded)
        params_node_flat = self.node_param_gen(encoded)
        params_latent_flat = self.latent_param_gen(encoded)
        
        # Reshape parameters
        params_topo = params_topo_flat.view(
            batch_size, self.n_circuit_layers, 
            len(self.quantum_circuit.wires_topo), 3
        )
        params_node = params_node_flat.view(
            batch_size, self.n_circuit_layers,
            len(self.quantum_circuit.wires_nodes), 3
        )
        params_latent = params_latent_flat.view(
            batch_size, self.n_circuit_layers,
            len(self.quantum_circuit.wires_latent), 3
        )
        
        # Execute quantum circuit (batch processing)
        topo_logits_list = []
        node_expvals_list = []
        latent_features_list = []
        
        for i in range(batch_size):
            # Convert to numpy for quantum circuit
            params_topo_np = params_topo[i].detach().cpu().numpy()
            params_node_np = params_node[i].detach().cpu().numpy()
            params_latent_np = params_latent[i].detach().cpu().numpy()
            
            # Execute quantum circuit
            quantum_output = self.quantum_circuit.forward(
                params_topo_np, params_node_np, params_latent_np
            )
            
            topo_logits_list.append(quantum_output['topo_logits'])
            node_expvals_list.append(quantum_output['node_expvals'])
            latent_features_list.append(quantum_output['latent_features'])
        
        # Convert to tensors
        topo_logits = torch.tensor(
            np.array(topo_logits_list), dtype=torch.float32, device=device
        )
        node_expvals = torch.tensor(
            np.array(node_expvals_list), dtype=torch.float32, device=device
        )
        latent_features = torch.tensor(
            np.array(latent_features_list), dtype=torch.float32, device=device
        )
        
        # Apply readout mapper
        mapped_output = self.mapper(topo_logits, node_expvals, latent_features)
        
        # Apply safety checks on lattice
        mapped_output['lattice'] = self.mapper.safety_check(mapped_output['lattice'], threshold=2.0)
        
        result = {
            'coords': mapped_output['coords'],
            'lattice': mapped_output['lattice'],
            'node_mask': mapped_output['node_mask'].to(device),
            'topo_logits': mapped_output['topo_logits'],
        }
        
        if return_intermediate:
            result['node_expvals'] = node_expvals
            result['latent_features'] = latent_features
        
        return result
    
    def generate_discrete_topology(self, z: torch.Tensor) -> np.ndarray:
        """
        Generate hard discrete topology (bitstring) from noise.
        
        Uses argmax on topology probabilities to get discrete selections.
        
        Args:
            z: Input noise tensor (batch_size, noise_dim)
        
        Returns:
            Discrete topology bitstrings (batch_size, 10)
        """
        with torch.no_grad():
            output = self.forward(z)
            # Get topology bitstring from probabilities
            # Shape: (batch_size, 1024) -> (batch_size, 10)
            topo_logits = output['topo_logits']
            batch_size = topo_logits.shape[0]
            
            # Decode 1024-dim probability vector back to 10-bit representation
            bitstrings = []
            for i in range(batch_size):
                # Hardest: just take the argmax index as the selected state
                max_state = torch.argmax(topo_logits[i]).item()
                # Convert to binary representation
                binary_str = format(max_state, '010b')
                bitstring = np.array([int(b) for b in binary_str])
                bitstrings.append(bitstring)
            
            return np.array(bitstrings)
    
    def export_qasm(self, filename: Optional[str] = None) -> str:
        """
        Export quantum circuit to OpenQASM 2.0 format.
        
        Args:
            filename: Optional file to save QASM to
        
        Returns:
            QASM string
        """
        qasm_str = self.quantum_circuit.export_qasm()
        
        if filename:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(qasm_str)
        
        return qasm_str
    
    def validate_output(self, output: Dict[str, torch.Tensor]) -> bool:
        """
        Validate generated MOF structure for physical feasibility.
        
        Args:
            output: Generator output dictionary
        
        Returns:
            True if structure is valid, False otherwise
        """
        coords = output['coords']
        lattice = output['lattice']
        
        # Check 1: All coordinates are finite
        if not torch.isfinite(coords).all():
            return False
        
        # Check 2: Lattice parameters are in reasonable range
        if (lattice < 1.5).any() or (lattice > 40.0).any():
            return False
        
        # Check 3: Coordinates are within lattice bounds
        for dim in range(3):
            if (torch.abs(coords[:, :, dim]) > lattice[:, dim % 6].unsqueeze(1)).any():
                return False
        
        return True
    
    def get_config(self) -> Dict:
        """Return generator configuration for serialization."""
        return {
            'noise_dim': self.noise_dim,
            'num_linker_atoms': self.num_linker_atoms,
            'n_circuit_layers': self.n_circuit_layers,
            'angstrom_scale': self.angstrom_scale,
            'device_name': self.device_name,
            'num_quantum_samples': self.num_quantum_samples,
            'n_qubits': self.quantum_circuit.n_qubits,
            'mapper_config': self.mapper.get_config(),
        }


# ===== Testing and Example Usage =====
if __name__ == "__main__":
    # Initialize generator
    generator = QGAN_Generator(
        noise_dim=32,
        num_linker_atoms=12,
        n_circuit_layers=2,
    )
    
    # Generate from noise
    z = torch.randn(2, 32)
    print("Generating MOF structures...")
    output = generator(z, return_intermediate=True)
    
    print(f"Generated coordinates shape: {output['coords'].shape}")
    print(f"Generated lattice shape: {output['lattice'].shape}")
    print(f"Node mask shape: {output['node_mask'].shape}")
    print(f"Topology logits shape: {output['topo_logits'].shape}")
    
    # Test discrete topology generation
    discrete_topo = generator.generate_discrete_topology(z)
    print(f"Discrete topology bitstrings shape: {discrete_topo.shape}")
    print(f"Example bitstring: {discrete_topo[0]}")
    
    # Test validation
    is_valid = generator.validate_output(output)
    print(f"Structure validity check passed: {is_valid}")
    
    # Export QASM
    qasm = generator.export_qasm()
    print(f"\nGenerated QASM (first 500 chars):\n{qasm[:500]}...")
