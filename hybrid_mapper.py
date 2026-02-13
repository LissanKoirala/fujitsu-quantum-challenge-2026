"""
Hybrid Readout Mapper: Convert quantum outputs to physical MOF structures
Decodes topo_logits, node_expvals, and latent_features into atomic coordinates and lattice parameters.
[cite: Quantum-classical interface papers, MOF generation architectures]
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np


class HybridReadoutMapper(nn.Module):
    """
    Converts quantum circuit outputs into valid MOF structures.
    
    Partitions output into:
    1. Node Head (Direct): 18 expvals -> 6 metal atoms (X, Y, Z) in Angstroms
    2. Linker/Lattice Head (Multiplexed): 12 latent features -> linker coords + lattice parameters
    """
    
    def __init__(
        self,
        num_linker_atoms: int = 12,
        angstrom_scale: float = 15.0,
        latent_dim: int = 12,
        hidden_dim: int = 64,
    ):
        """
        Initialize the Hybrid Readout Mapper.
        
        Args:
            num_linker_atoms: Number of linker atoms beyond the 6 metal nodes
            angstrom_scale: Scale factor for coordinate normalization (Angstroms)
            latent_dim: Dimension of latent features from quantum circuit (fixed: 12)
            hidden_dim: Hidden dimension for MLP head
        """
        super().__init__()
        
        self.num_metal_atoms = 6
        self.num_linker_atoms = num_linker_atoms
        self.total_atoms = self.num_metal_atoms + num_linker_atoms
        self.angstrom_scale = angstrom_scale
        self.latent_dim = latent_dim
        
        # Node Head (Direct Mapping): Adapt to actual # of node qubits
        # Will be resized in forward() based on actual input dimensions
        self.node_scale = nn.Parameter(torch.full((18,), 1.0))  # Default for 18, resized as needed
        
        # Latent Head (Multiplexed): adapt to actual latent dimension (12 or 3)
        # Output dimension: linker_atoms * 3 (coords) + 6 (lattice parameters)
        self.linker_output_dim = num_linker_atoms * 3 + 6
        
        self.latent_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.linker_output_dim)
        )
        
        # Also support adaptive latent dimension - create a projector if needed
        self.latent_projector = None  # Created in forward if needed
        self.actual_latent_dim = latent_dim
        
        # Initialize weights
        for layer in self.latent_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        topo_logits: torch.Tensor,
        node_expvals: torch.Tensor,
        latent_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Map quantum outputs to MOF structure.
        
        Args:
            topo_logits: Topology probabilities, shape (batch_size, 2^n_topo_qubits)
            node_expvals: Node expectation values, shape (batch_size, n_node_qubits)
            latent_features: Latent features, shape (batch_size, 12 or 3 for mini)
        
        Returns:
            Dictionary containing:
            - 'coords': Full atomic coordinates (batch_size, total_atoms, 3)
            - 'lattice': Lattice parameters (batch_size, 6)
            - 'node_mask': Mask for metal atoms (batch_size, total_atoms)
            - 'topo_logits': Topology probabilities (for analysis)
        """
        
        batch_size = node_expvals.shape[0]
        n_node_dims = node_expvals.shape[1]  # Adapt to actual # of node qubits (3 or 18)
        
        # Adjust node scaling to match actual dimensions
        if n_node_dims != self.node_scale.shape[0]:
            # Reinitialize node_scale to match
            self.node_scale = nn.Parameter(torch.full((n_node_dims,), 1.0))
        
        # ===== Node Head (Direct Mapping) =====
        # Scale expvals from [-1, 1] to physical space [-angstrom_scale, angstrom_scale]
        node_coords_normalized = node_expvals * self.node_scale  # (batch_size, n_node_dims)
        node_coords = torch.tanh(node_coords_normalized) * self.angstrom_scale  # (batch_size, n_node_dims)
        
        # Reshape to (batch_size, n_atoms, 3) - pad if needed for mini circuit
        if n_node_dims % 3 == 0:
            n_atoms = n_node_dims // 3
        else:
            n_atoms = 1
        
        node_coords = node_coords.view(batch_size, n_atoms, 3)  # (batch_size, n_atoms, 3)
        
        # ===== Latent Head (Multiplexed Decoding) =====
        # Handle variable latent dimensions (3 for mini, 12 for full)
        latent_input = latent_features
        if latent_features.shape[1] != self.actual_latent_dim:
            # Dimension mismatch - create projector if needed
            first_layer = self.latent_head[0]  # Get first Linear layer
            current_in_features = first_layer.in_features if hasattr(first_layer, 'in_features') else None
            
            if self.latent_projector is None or current_in_features != latent_features.shape[1]:
                hidden_dim = 64
                self.latent_projector = nn.Sequential(
                    nn.Linear(latent_features.shape[1], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.actual_latent_dim)
                ).to(latent_features.device)
            latent_input = self.latent_projector(latent_features)
        
        latent_decoded = self.latent_head(latent_input)  # (batch_size, linker_dim + 6)
        
        # Split into linker coordinates and lattice parameters
        linker_coords_flat = latent_decoded[:, :self.num_linker_atoms * 3]
        lattice_raw = latent_decoded[:, self.num_linker_atoms * 3:]
        
        # Reshape linker coordinates
        linker_coords = linker_coords_flat.view(batch_size, self.num_linker_atoms, 3)
        linker_coords = torch.tanh(linker_coords) * self.angstrom_scale
        
        # Lattice parameters: scale to physical range (2.0 - 30.0 Angstroms)
        # Using sigmoid for bounded output, then scale
        lattice = 2.0 + 28.0 * torch.sigmoid(lattice_raw)  # (batch_size, 6)
        
        # ===== Combine Coordinates =====
        full_coords = torch.cat([node_coords, linker_coords], dim=1)  # (batch_size, total_atoms, 3)
        
        # ===== Create Node Mask =====
        # 1 for metal atoms (first 6), 0 for linker atoms
        node_mask = torch.zeros(batch_size, self.total_atoms, dtype=torch.float32)
        node_mask[:, :self.num_metal_atoms] = 1.0
        
        return {
            'coords': full_coords,
            'lattice': lattice,
            'node_mask': node_mask,
            'topo_logits': topo_logits,
            'node_coords': node_coords,
            'linker_coords': linker_coords,
        }
    
    def safety_check(self, lattice: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
        """
        Safety check: prevent physical instabilities by clamping lattice parameters.
        
        Args:
            lattice: Lattice parameters (batch_size, 6)
            threshold: Minimum lattice parameter in Angstroms
        
        Returns:
            Clamped lattice tensor
        """
        return torch.clamp(lattice, min=threshold)
    
    def get_config(self) -> Dict:
        """Return mapper configuration for serialization."""
        return {
            'num_metal_atoms': self.num_metal_atoms,
            'num_linker_atoms': self.num_linker_atoms,
            'total_atoms': self.total_atoms,
            'angstrom_scale': self.angstrom_scale,
            'latent_dim': self.latent_dim,
        }
