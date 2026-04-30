"""
Installation Test for ABSTLight Model
======================================

This script verifies that all dependencies are installed correctly and that
the ABSTLight model components can be initialized and run basic inference.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_success(msg):
    """Print success message."""
    print(f"✓ {msg}")


def print_error(msg):
    """Print error message."""
    print(f"✗ {msg}")


def print_info(msg):
    """Print info message."""
    print(f"ℹ {msg}")


def test_dependencies():
    """Test if all required packages are installed."""
    print_header("Testing Dependencies")
    
    required_packages = {
        "torch": torch,
        "numpy": np,
    }
    
    optional_packages = {
        "traci": "traci",
        "sumolib": "sumolib",
    }
    
    # Test required packages
    for name, module in required_packages.items():
        try:
            version = getattr(module, "__version__", "unknown")
            print_success(f"{name} {version}")
        except Exception as e:
            print_error(f"{name}: {e}")
            return False
    
    # Test optional packages
    for name, import_name in optional_packages.items():
        try:
            __import__(import_name)
            print_info(f"{name} is available (optional)")
        except ImportError:
            print_info(f"{name} is not installed (optional for basic tests)")
    
    return True


def test_model_components():
    """Test if model components can be imported and instantiated."""
    print_header("Testing Model Components Import")
    
    try:
        from model_components.observation_embedding import ObservationEmbedding
        print_success("ObservationEmbedding imported")
    except Exception as e:
        print_error(f"Failed to import ObservationEmbedding: {e}")
        return False
    
    try:
        from model_components.gcn_mha import GCNMHAStack
        print_success("GCNMHAStack imported")
    except Exception as e:
        print_error(f"Failed to import GCNMHAStack: {e}")
        return False
    
    try:
        from model_components.q_value_head import QValueHead
        print_success("QValueHead imported")
    except Exception as e:
        print_error(f"Failed to import QValueHead: {e}")
        return False
    
    return True


def test_component_initialization():
    """Test if model components can be initialized."""
    print_header("Testing Model Component Initialization")
    
    try:
        from model_components.observation_embedding import ObservationEmbedding
        from model_components.gcn_mha import GCNMHAStack
        from model_components.q_value_head import QValueHead
        
        # Test parameters
        embed_dim = 64
        num_agents = 4
        num_actions = 4
        observation_size = 20  # Example observation size
        
        # Initialize components
        embedding = ObservationEmbedding(
            obs_dim=observation_size,
            embed_dim=embed_dim
        )
        print_success(f"ObservationEmbedding initialized (obs_size={observation_size}, embed_dim={embed_dim})")
        
        # Create adjacency matrix (fully connected for this test)
        adj = torch.ones(num_agents, num_agents)
        
        gcn_mha = GCNMHAStack(
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=2
        )
        print_success(f"GCNMHAStack initialized (embed_dim={embed_dim}, heads=4, layers=2)")
        
        q_head = QValueHead(embed_dim=embed_dim, num_actions=num_actions)
        print_success(f"QValueHead initialized (embed_dim={embed_dim}, num_actions={num_actions})")
        
        return embedding, gcn_mha, q_head, adj, num_actions
        
    except Exception as e:
        print_error(f"Failed to initialize components: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(embedding, gcn_mha, q_head, adj, num_actions):
    """Test forward pass through the model."""
    print_header("Testing Forward Pass")
    
    try:
        torch.manual_seed(42)
        
        # Create sample input
        batch_size = 2
        num_agents = 4
        observation_size = 20
        
        # Random observation for each agent
        observations = torch.randn(batch_size, num_agents, observation_size)
        print_info(f"Input shape: {observations.shape}")
        
        # Embedding layer
        embedded = embedding(observations)
        print_success(f"Embedding output shape: {embedded.shape}")
        
        # Reshape for GCN-MHA (batch-wise processing)
        # Expected: (batch_size, num_agents, embed_dim)
        
        # GCN-MHA layer processing
        gcn_output = gcn_mha(embedded, adj)
        print_success(f"GCN-MHA output shape: {gcn_output.shape}")
        
        # Q-head layer for first agent
        q_values = q_head(gcn_output[:, 0, :])  # Take first agent
        print_success(f"Q-values output shape: {q_values.shape}")
        print_info(f"Q-values range: [{q_values.min():.4f}, {q_values.max():.4f}]")
        
        # Verify dimensions
        assert embedded.shape == (batch_size, num_agents, embedding.embed_dim)
        assert gcn_output.shape == (batch_size, num_agents, gcn_mha.embed_dim)
        assert q_values.shape == (batch_size, num_actions)
        
        print_success("All forward pass dimensions verified")
        return True
        
    except Exception as e:
        print_error(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_import():
    """Test if agent can be imported."""
    print_header("Testing Agent Import")
    
    try:
        from agent.abst_light_agent import ABSTLightAgent
        print_success("ABSTLightAgent imported")
        return True
    except Exception as e:
        print_error(f"Failed to import ABSTLightAgent: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test if config can be imported."""
    print_header("Testing Configuration")
    
    try:
        from config import Config, ABSTConfig
        print_success("Config imported")
        print_info(f"Learning rate: {Config.LEARNING_RATE}")
        print_info(f"Batch size: {Config.BATCH_SIZE}")
        print_info(f"Ensemble networks: {Config.N_NETWORKS}")
        print_info(f"Memory size: {Config.MEMORY_SIZE}")

        # Debug check requested: ensure ABST input dimension is 20.
        print_info(f"ABST OBS_DIM: {ABSTConfig.OBS_DIM}")
        if int(ABSTConfig.OBS_DIM) != 20:
            print_error(f"ABST OBS_DIM mismatch: expected 20, got {ABSTConfig.OBS_DIM}")
            return False
        print_success("ABST OBS_DIM check passed (20)")
        return True
    except Exception as e:
        print_error(f"Failed to import Config: {e}")
        return False


def test_device():
    """Test PyTorch device detection."""
    print_header("Testing PyTorch Device")
    
    if torch.cuda.is_available():
        print_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print_info(f"CUDA version: {torch.version.cuda}")
    else:
        print_info("CUDA not available - will use CPU")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_success(f"Device set to: {device}")
    
    return device


def test_environment_import():
    """Test if environment can be imported."""
    print_header("Testing Environment Import")
    
    try:
        from environment.traffic_env import TrafficEnv
        print_success("TrafficEnv imported")
        return True
    except Exception as e:
        print_info(f"TrafficEnv import skipped (SUMO may not be configured): {e}")
        return True  # Don't fail on this - SUMO might need manual setup


def main():
    """Run all tests."""
    print("\n" + "#" * 70)
    print("#  ABSTLight Installation Test Suite")
    print("#" * 70)
    
    all_passed = True
    
    # Test dependencies
    if not test_dependencies():
        print_error("Dependency test failed!")
        all_passed = False
    
    # Test device
    device = test_device()
    
    # Test configuration
    if not test_config():
        all_passed = False
    
    # Test model components
    if not test_model_components():
        all_passed = False
    
    # Test initialization
    result = test_component_initialization()
    if result is None:
        all_passed = False
    else:
        embedding, gcn_mha, q_head, adj, num_actions = result
        
        # Test forward pass
        if not test_forward_pass(embedding, gcn_mha, q_head, adj, num_actions):
            all_passed = False
    
    # Test agent
    if not test_agent_import():
        all_passed = False
    
    # Test environment
    test_environment_import()
    
    # Final summary
    print_header("Test Summary")
    if all_passed:
        print_success("All core tests passed! ✓")
        print_info("ABSTLight model is ready to use.")
    else:
        print_error("Some tests failed. Please check errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
