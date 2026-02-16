"""
Hybrid Readout Mapper: 30-Qubit Configuration
Converts quantum outputs to physical MOF structures with optimized 30-qubit register sizes.

Output node count: 5 metal atoms (vs 6 in 40q) - balanced for 30-qubit register
Latent dimension: 7 (vs 12 in 40q) - refined via projection layer

[cite: Quantum-classical interface, MOF generation, PennyLane-PyTorch integration]
"""

from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import numpy as np
import time
import sys


class HybridReadoutMapper30Q(nn.Module):
    """
    Converts 30-qubit quantum circuit outputs to MOF structures.
    
    Input Mapping:
    - topo_logits: 256-dim probability distribution from 8 topology qubits
    - node_expvals: 15-dim expvals from 15 node qubits → 5 metal atoms (X,Y,Z)
    - latent_features: 7-dim expvals from 7 latent qubits
    
    Output:
    - metal_atoms: (5, 3) torch tensor in Angstroms
    - linker_atoms: (num_linker, 3) torch tensor
    - lattice_params: (6,) torch tensor (a, b, c, α, β, γ)
    """
    
    def __init__(
        self,
        num_linker_atoms: int = 10,
        angstrom_scale: float = 12.0,
        latent_dim: int = 7,
        hidden_dim: int = 64,
        n_metal_atoms: int = 5,
        device: str = "cpu"
    ):
        """
        Initialize 30-qubit Mapper.
        
        Args:
            num_linker_atoms: Number of linker atoms (organic ligands)
            angstrom_scale: Scale factor for normalized coordinates to Angstroms
            latent_dim: Latent feature dimension from quantum circuit (=7 for 30q)
            hidden_dim: Hidden MLP dimension
            n_metal_atoms: Number of metal atoms (=5 for 30q)
            device: "cpu" or "cuda"
        """
        super().__init__()
        
        self.n_metal_atoms = n_metal_atoms
        self.n_nodes = n_metal_atoms * 3  # 15 for 30q
        self.num_linker_atoms = num_linker_atoms
        self.total_atoms = n_metal_atoms + num_linker_atoms
        self.angstrom_scale = angstrom_scale
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # ===== NODE HEAD: Direct Expectation Value Mapping =====
        # Maps 15 node expvals → 5 metal atoms × 3 coordinates
        self.node_scale = nn.Parameter(
            torch.ones(self.n_nodes, device=device),
            requires_grad=True
        )
        self.node_bias = nn.Parameter(
            torch.zeros(self.n_nodes, device=device),
            requires_grad=True
        )
        
        # ===== LATENT HEAD: Multiplexed Projection =====
        # Projects (7 latent + 8 topo features) → linker coords + lattice params
        linker_coord_dim = num_linker_atoms * 3
        lattice_param_dim = 6
        self.latent_output_dim = linker_coord_dim + lattice_param_dim
        self.topo_feature_dim = 8

        self.latent_head = nn.Sequential(
            nn.Linear(latent_dim + self.topo_feature_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_output_dim, device=device)
        )
        
        # ===== TOPOLOGY HEAD: Discrete State Selection =====
        # Selects topology from 256 possible states and maps to linker type
        self.topo_head = nn.Sequential(
            nn.Linear(256, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, 64, device=device),
            nn.ReLU(),
            nn.Linear(64, 8, device=device)  # 8 linker type features
        )
        
        # Initialization
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable training"""
        for module in [self.latent_head, self.topo_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        topo_logits: torch.Tensor,
        node_expvals: torch.Tensor,
        latent_features: torch.Tensor,
        validate_physical: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Map quantum outputs to MOF geometry.
        
        Args:
            topo_logits: Shape (batch, 256) or (256,) - topology probabilities
            node_expvals: Shape (batch, 15) or (15,) - node expectation values
            latent_features: Shape (batch, 7) or (7,) - latent features
            validate_physical: If True, apply physical constraints
        
        Returns:
            {
                'metal_atoms': (batch, 5, 3) - metal atomic coordinates [Å]
                'linker_atoms': (batch, num_linker, 3) - linker coordinates [Å]
                'lattice_params': (batch, 6) - lattice parameters [Å, deg]
                'topo_state': (batch,) - selected topology state index
                'validity': (batch,) - physical feasibility score [0,1]
            }
        """
        
        t0 = time.time()

        # Handle unbatched inputs
        is_unbatched = topo_logits.dim() == 1
        if is_unbatched:
            topo_logits = topo_logits.unsqueeze(0)
            node_expvals = node_expvals.unsqueeze(0)
            latent_features = latent_features.unsqueeze(0)

        batch_size = topo_logits.shape[0]
        print(f"    [mapper] forward start (batch={batch_size})", flush=True)

        # ===== METAL ATOMS (Direct Mapping from Node Expvals) =====
        node_scaled = (node_expvals * self.node_scale + self.node_bias).tanh()
        metal_atoms = (node_scaled * self.angstrom_scale).view(batch_size, self.n_metal_atoms, 3)
        print(f"    [mapper] metal_atoms done ({time.time() - t0:.3f}s)", flush=True)

        # ===== TOPOLOGY SELECTION =====
        topo_features = self.topo_head(topo_logits)  # (batch, 8)
        topo_state = torch.argmax(topo_logits, dim=1)  # (batch,)
        print(f"    [mapper] topo_head done ({time.time() - t0:.3f}s)", flush=True)

        # ===== LINKER + LATTICE PARAMETERS (Topology-Conditioned) =====
        latent_enhanced = torch.cat([latent_features, topo_features], dim=1)  # (batch, 15)

        # Project to linker + lattice space (topology-conditioned via latent_enhanced)
        linker_lattice_raw = self.latent_head(latent_enhanced)  # (batch, 3*num_linker + 6)

        # Split into linker coords and lattice params
        linker_flat = linker_lattice_raw[:, :self.num_linker_atoms * 3]
        lattice_raw = linker_lattice_raw[:, self.num_linker_atoms * 3:]

        linker_atoms = (torch.tanh(linker_flat) * self.angstrom_scale).view(
            batch_size, self.num_linker_atoms, 3
        )

        # ===== LATTICE PARAMETERS WITH PHYSICAL CONSTRAINTS =====
        lattice_lengths = torch.sigmoid(lattice_raw[:, :3]) * 15.0 + 5.0  # [5, 20] Å
        lattice_angles = torch.sigmoid(lattice_raw[:, 3:]) * 40.0 + 70.0  # [70, 110] °
        lattice_params = torch.cat([lattice_lengths, lattice_angles], dim=1)
        print(f"    [mapper] linker+lattice done ({time.time() - t0:.3f}s)", flush=True)

        # ===== NODE MASK: 1 for metals, 0 for linkers =====
        node_mask = torch.zeros(batch_size, self.total_atoms, device=metal_atoms.device)
        node_mask[:, :self.n_metal_atoms] = 1.0

        # ===== VALIDITY SCORING (Physical Feasibility) =====
        validity = self._compute_validity(
            metal_atoms, linker_atoms, lattice_params, topo_logits
        )
        print(f"    [mapper] validity done ({time.time() - t0:.3f}s)", flush=True)

        # Unbatch if input was unbatched
        if is_unbatched:
            metal_atoms = metal_atoms.squeeze(0)
            linker_atoms = linker_atoms.squeeze(0)
            lattice_params = lattice_params.squeeze(0)
            topo_state = topo_state.squeeze(0)
            node_mask = node_mask.squeeze(0)
            validity = validity.squeeze(0)

        return {
            'metal_atoms': metal_atoms,
            'linker_atoms': linker_atoms,
            'lattice_params': lattice_params,
            'topo_state': topo_state.detach(),
            'node_mask': node_mask,
            'validity': validity,
        }
    
    def _compute_validity(
        self,
        metal_atoms: torch.Tensor,
        linker_atoms: torch.Tensor,
        lattice_params: torch.Tensor,
        topo_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute physical feasibility score.
        
        Checks:
        - Inter-atomic distances (no clashes, reasonable bonding)
        - Lattice parameters (positive, reasonable MOF sizes)
        - Topology entropy (avoid degenerate states)
        """
        batch_size = metal_atoms.shape[0]
        validity = torch.ones(batch_size, device=metal_atoms.device)
        
        # Check 1: Metal-linker distances (should be ~1.5-3 Å for coordination)
        all_atoms = torch.cat([metal_atoms, linker_atoms], dim=1)  # (batch, n_total, 3)
        distances = torch.cdist(all_atoms, all_atoms)  # (batch, n_total, n_total)
        
        # Penalize if too close (clashing)
        too_close = (distances < 1.0).float().sum(dim=(1, 2))
        validity -= 0.1 * torch.clamp(too_close, 0, 5) / 5
        
        # Penalize if too far (fragmented)
        too_far = (distances > 25.0).float().sum(dim=(1, 2))
        validity -= 0.05 * torch.clamp(too_far, 0, 10) / 10
        
        # Check 2: Lattice parameters (reasonable MOF sizes)
        lattice_lengths = lattice_params[:, :3]
        feasible_lengths = torch.all(
            (lattice_lengths > 4.0) & (lattice_lengths < 25.0),
            dim=1
        ).float()
        validity *= feasible_lengths
        
        # Check 3: Topology entropy (avoid peak of single state)
        topo_entropy = -torch.sum(topo_logits * torch.log(topo_logits + 1e-10), dim=1)
        entropy_valid = (topo_entropy > np.log(256) * 0.1).float()  # >10% theoretical max
        validity *= entropy_valid
        
        return torch.clamp(validity, 0, 1)
    
    def to_structure_dict(
        self,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict:
        """
        Convert mapper outputs to standard MOF structure dictionary.
        
        Returns:
            {
                'atoms': [...],  # List of (symbol, coords) tuples
                'lattice': [...],  # Lattice vectors
                'composition': str  # Chemical formula
            }
        """
        metal_atoms = outputs['metal_atoms'].detach().cpu().numpy()
        linker_atoms = outputs['linker_atoms'].detach().cpu().numpy()
        lattice_params = outputs['lattice_params'].detach().cpu().numpy()
        
        # Build structure
        structure = {
            'atoms': [],
            'lattice': [],
            'validity': outputs['validity'].item() if outputs['validity'].dim() == 0 else outputs['validity'][0].item()
        }
        
        # Add metal atoms (assume Cu or Zn)
        for coords in metal_atoms if metal_atoms.ndim == 2 else [metal_atoms]:
            structure['atoms'].append(('Cu', coords.tolist()))
        
        # Add linker atoms (assume C from organic linkers)
        linker_data = linker_atoms if linker_atoms.ndim == 2 else linker_atoms.reshape(-1, 3)
        for coords in linker_data:
            structure['atoms'].append(('C', coords.tolist()))
        
        # Build lattice vectors from parameters
        a, b, c = lattice_params[0, :3] if lattice_params.ndim == 2 else lattice_params[:3]
        alpha, beta, gamma = lattice_params[0, 3:] if lattice_params.ndim == 2 else lattice_params[3:]
        
        # Convert degrees to radians
        alpha_rad = np.deg2rad(alpha)
        beta_rad = np.deg2rad(beta)
        gamma_rad = np.deg2rad(gamma)
        
        # Build lattice vectors (standard crystallography)
        v1 = np.array([a, 0, 0])
        v2 = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0])
        v3 = np.array([
            c * np.cos(beta_rad),
            c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad),
            c * np.sqrt(1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 + 2*np.cos(alpha_rad)*np.cos(beta_rad)*np.cos(gamma_rad)) / np.sin(gamma_rad)
        ])
        
        structure['lattice'] = [v1.tolist(), v2.tolist(), v3.tolist()]
        
        return structure


if __name__ == "__main__":
    # Quick test
    mapper = HybridReadoutMapper30Q(num_linker_atoms=10)
    
    # Dummy quantum outputs
    topo_logits = torch.ones(256) / 256
    node_expvals = torch.randn(15)
    latent_features = torch.randn(7)
    
    outputs = mapper(topo_logits, node_expvals, latent_features)
    
    print("30-Qubit Mapper Test:")
    print(f"Metal atoms shape: {outputs['metal_atoms'].shape}")
    print(f"Linker atoms shape: {outputs['linker_atoms'].shape}")
    print(f"Lattice params shape: {outputs['lattice_params'].shape}")
    print(f"Node mask shape: {outputs['node_mask'].shape}")
    print(f"Node mask: {outputs['node_mask']}")
    print(f"Validity: {outputs['validity'].item():.3f}")
    print(f"Gradient flows: {outputs['metal_atoms'].requires_grad}")
