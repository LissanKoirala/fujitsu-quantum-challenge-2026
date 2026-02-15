"""
Minimal fast test to verify the param_shift and LayerNorm fixes
"""
import sys
import time
import torch
import numpy as np

from hybrid_circuit_30q import HybridQGANCircuit30Q, QubitConfig30Q
from hybrid_mapper_30q import HybridReadoutMapper30Q
from qgan_generator_30q import QGAN_Generator30Q


def test_circuit_init():
    """Test circuit initialization with corrected diff_method"""
    start = time.time()
    try:
        config = QubitConfig30Q()
        # This should not fail on diff_method
        circuit = HybridQGANCircuit30Q(n_layers=1, config=config)
        assert hasattr(circuit, 'qnode_grad'), "qnode_grad not created"
        duration = time.time() - start
        print(f"✓ Circuit Init: {duration:.3f}s")
        return True
    except Exception as e:
        print(f"✗ Circuit Init: {str(e)}")
        return False


def test_mapper_init():
    """Test mapper initialization with LayerNorm"""
    start = time.time()
    try:
        mapper = HybridReadoutMapper30Q(num_linker_atoms=10, latent_dim=7)
        assert hasattr(mapper, 'latent_head'), "latent_head not created"
        # Check that latent_head uses LayerNorm not BatchNorm
        modules = list(mapper.latent_head.modules())
        has_layernorm = any('LayerNorm' in str(type(m)) for m in modules)
        assert has_layernorm, "LayerNorm not found in mapper"
        duration = time.time() - start
        print(f"✓ Mapper Init: {duration:.3f}s")
        return True
    except Exception as e:
        print(f"✗ Mapper Init: {str(e)}")
        return False


def test_mapper_forward_single_sample():
    """Test mapper with single sample (where BatchNorm would fail)"""
    start = time.time()
    try:
        mapper = HybridReadoutMapper30Q(num_linker_atoms=10, latent_dim=7)
        
        # Single sample (batch_size=1) - this would fail with BatchNorm
        topo_logits = torch.randn(256)  # Unbatched
        node_expvals = torch.randn(15)
        latent_features = torch.randn(7)
        
        output = mapper(topo_logits, node_expvals, latent_features)
        
        assert output['metal_atoms'].shape == (5, 3), f"Metal shape: {output['metal_atoms'].shape}"
        assert output['linker_atoms'].shape == (10, 3), f"Linker shape: {output['linker_atoms'].shape}"
        assert output['lattice_params'].shape == (6,), f"Lattice shape: {output['lattice_params'].shape}"
        
        duration = time.time() - start
        print(f"✓ Mapper Forward (single): {duration:.3f}s")
        return True
    except Exception as e:
        print(f"✗ Mapper Forward (single): {str(e)}")
        return False


def test_qgan_init():
    """Test QGAN generator initialization"""
    start = time.time()
    try:
        gen = QGAN_Generator30Q(noise_dim=32, num_linker_atoms=10, n_circuit_layers=1)
        assert hasattr(gen, 'quantum_circuit'), "quantum_circuit not created"
        assert hasattr(gen, 'mapper'), "mapper not created"
        duration = time.time() - start
        print(f"✓ QGAN Init: {duration:.3f}s")
        return True
    except Exception as e:
        print(f"✗ QGAN Init: {str(e)}")
        return False


def test_noise_encoder():
    """Test noise encoder with LayerNorm"""
    start = time.time()
    try:
        gen = QGAN_Generator30Q(noise_dim=32, num_linker_atoms=10, n_circuit_layers=1)
        
        # Check noise_encoder uses LayerNorm not BatchNorm
        modules = list(gen.noise_encoder.modules())
        has_layernorm = any('LayerNorm' in str(type(m)) for m in modules)
        assert has_layernorm, "LayerNorm not found in noise_encoder"
        
        # Test with batch_size=1 (would fail with BatchNorm)
        noise = torch.randn(1, 32)
        params = gen.noise_encoder(noise)
        assert params.shape[0] == 1, f"Param batch shape: {params.shape}"
        
        duration = time.time() - start
        print(f"✓ Noise Encoder (LayerNorm): {duration:.3f}s")
        return True
    except Exception as e:
        print(f"✗ Noise Encoder: {str(e)}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("MINIMAL TEST SUITE - Verification of key fixes")
    print("=" * 60)
    
    results = [
        test_circuit_init(),
        test_mapper_init(),
        test_mapper_forward_single_sample(),
        test_qgan_init(),
        test_noise_encoder(),
    ]
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)
    
    sys.exit(0 if passed == total else 1)
