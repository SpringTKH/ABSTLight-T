"""
RELight Agent - Adaptive Random Ensemble Light

This is the main agent class that manages the ensemble of Q-networks and
coordinates the training process. The agent:

1. Maintains N independent Q-networks
2. Uses ensemble for action selection (voting or averaging)
3. Delegates updates to the EnsembleUpdater
4. Manages exploration-exploitation tradeoff

The key innovation is the ensemble-based approach which reduces overestimation
bias and improves robustness compared to single-network DQN.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Tuple, List, Dict
from pathlib import Path

from .base_dqn import BaseDQN
from legacy.core.ensemble_updater import EnsembleUpdater


class RELightAgent:
    """
    RELight Agent - Main controller for the ensemble-based DQN approach.
    
    This agent manages N independent Q-networks and uses them collectively
    for both action selection and value estimation. Unlike standard DQN:
    
    - Action Selection: Uses ensemble voting or averaging across all networks
    - Value Estimation: Uses random subset sampling with error thresholds
    - Updates: Each network is updated independently via EnsembleUpdater
    
    Args:
        state_shape (tuple): Shape of input state (C, H, W)
        num_actions (int): Number of possible actions
        N (int): Number of networks in ensemble
        M (int): Size of random subset for target calculation
        learning_rate (float): Learning rate for optimizers
        gamma (float): Discount factor
        error_threshold (float): Error threshold for resampling
        max_resampling_attempts (int): Max resampling attempts
        epsilon_start (float): Initial exploration rate
        epsilon_end (float): Final exploration rate
        epsilon_decay (float): Exploration decay rate
        device (str): Computation device ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, int, int],
        num_actions: int,
        N: int = 10,
        M: int = 4,
        learning_rate: float = 1e-4,
        gamma: float = 0.95,
        error_threshold: float = 150.0,
        reward_scale_divisor: float = 1000.0,
        use_dynamic_error_threshold: bool = True,
        relative_error_threshold_ratio: float = 0.05,
        min_target_scale_for_threshold: float = 1.0,
        max_resampling_attempts: int = 5,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        device: str = 'cpu'
    ):
        """
        Initialize the RELight agent with N Q-networks.
        
        Args:
            state_shape: Input state dimensions (channels, height, width)
            num_actions: Number of discrete actions
            N: Total number of Q-networks in ensemble
            M: Number of networks to sample for target calculation
            learning_rate: Learning rate for network updates
            gamma: Discount factor for rewards
            error_threshold: Threshold for error-triggered resampling
            max_resampling_attempts: Maximum resampling attempts per update
            epsilon_start: Initial exploration probability
            epsilon_end: Minimum exploration probability
            epsilon_decay: Exploration decay factor
            device: Device for computation
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.N = N  # Total networks
        self.M = M  # Subset size
        self.device = device
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize N independent Q-networks
        print(f"Initializing RELight agent with {N} networks...")
        self.networks = []
        for i in range(N):
            network = BaseDQN(state_shape, num_actions).to(device)
            self.networks.append(network)
            print(f"  Network {i+1}/{N} initialized")
        
        # Initialize the Ensemble Updater
        self.updater = EnsembleUpdater(
            networks=self.networks,
            learning_rate=learning_rate,
            gamma=gamma,
            error_threshold=error_threshold,
            reward_scale_divisor=reward_scale_divisor,
            use_dynamic_error_threshold=use_dynamic_error_threshold,
            relative_error_threshold_ratio=relative_error_threshold_ratio,
            min_target_scale_for_threshold=min_target_scale_for_threshold,
            max_resampling_attempts=max_resampling_attempts,
            device=device
        )
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        
    def select_action(self, state: torch.Tensor, mode: str = 'voting') -> int:
        """
        Select action using ensemble strategy with epsilon-greedy exploration.
        
        Action Selection Strategies:
        - 'voting': Each network votes for best action, majority wins
        - 'averaging': Average Q-values across all networks, select max
        - 'random_network': Randomly select one network and use its action
        
        Args:
            state: Current state tensor (C, H, W) or (1, C, H, W)
            mode: Ensemble aggregation mode ('voting', 'averaging', 'random_network')
            
        Returns:
            Selected action (int)
        """
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Random action (exploration)
            return np.random.randint(0, self.num_actions)
        
        # Greedy action using ensemble (exploitation)
        with torch.no_grad():
            # Ensure state has batch dimension
            if state.dim() == 3:
                state = state.unsqueeze(0)
            
            state = state.to(self.device)
            
            if mode == 'voting':
                # Each network votes for its best action
                votes = np.zeros(self.num_actions)
                for network in self.networks:
                    q_values = network(state)
                    best_action = q_values.argmax(dim=1).item()
                    votes[best_action] += 1
                
                # Return action with most votes (ties broken randomly)
                max_votes = votes.max()
                best_actions = np.where(votes == max_votes)[0]
                action = np.random.choice(best_actions)
                
            elif mode == 'averaging':
                # Average Q-values across all networks
                q_values_sum = torch.zeros(1, self.num_actions).to(self.device)
                for network in self.networks:
                    q_values_sum += network(state)
                
                avg_q_values = q_values_sum / self.N
                action = avg_q_values.argmax(dim=1).item()
                
            elif mode == 'random_network':
                # Randomly select one network
                random_network = np.random.choice(self.networks)
                q_values = random_network(state)
                action = q_values.argmax(dim=1).item()
                
            else:
                raise ValueError(f"Unknown action selection mode: {mode}")
        
        return action
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> List[Dict[str, float]]:
        """
        Update all networks in the ensemble using the RELight algorithm.
        
        This method delegates to the EnsembleUpdater which handles:
        - Random subset sampling
        - Minimum target Q-value calculation
        - Error-triggered resampling
        - Gradient updates
        
        Args:
            states: Batch of current states (B, C, H, W)
            actions: Batch of actions (B,)
            rewards: Batch of rewards (B,)
            next_states: Batch of next states (B, C, H, W)
            dones: Batch of terminal flags (B,)
            
        Returns:
            List of update statistics for each network
        """
        # Move data to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update all networks via EnsembleUpdater
        update_results = self.updater.update_all_networks(
            states, actions, rewards, next_states, dones, self.M
        )
        
        # Increment training step
        self.training_step += 1
        
        # Decay epsilon
        self.decay_epsilon()
        
        return update_results
    
    def decay_epsilon(self):
        """
        Decay exploration rate epsilon.
        """
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
    
    def set_epsilon(self, epsilon: float):
        """
        Manually set exploration rate.
        
        Args:
            epsilon: New exploration rate
        """
        self.epsilon = max(self.epsilon_end, min(self.epsilon_start, epsilon))
    
    def get_epsilon(self) -> float:
        """
        Get current exploration rate.
        
        Returns:
            Current epsilon value
        """
        return self.epsilon
    
    def train_mode(self):
        """Set all networks to training mode."""
        for network in self.networks:
            network.train()
    
    def eval_mode(self):
        """Set all networks to evaluation mode."""
        for network in self.networks:
            network.eval()
    
    def save(self, directory: str, episode: int = None):
        """
        Save all networks and agent state.
        
        Args:
            directory: Directory to save models
            episode: Optional episode number for filename
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save each network
        for i, network in enumerate(self.networks):
            if episode is not None:
                filename = f"network_{i}_episode_{episode}.pth"
            else:
                filename = f"network_{i}.pth"
            
            filepath = os.path.join(directory, filename)
            network.save(filepath)
        
        # Save agent state
        agent_state = {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'updater_stats': self.updater.get_statistics()
        }
        
        if episode is not None:
            state_filename = f"agent_state_episode_{episode}.pth"
        else:
            state_filename = "agent_state.pth"
        
        state_filepath = os.path.join(directory, state_filename)
        torch.save(agent_state, state_filepath)
        
        print(f"Agent saved to {directory}")
    
    def load(self, directory: str, episode: int = None):
        """
        Load all networks and agent state.
        
        Args:
            directory: Directory to load models from
            episode: Optional episode number for filename
        """
        # Load each network
        for i, network in enumerate(self.networks):
            if episode is not None:
                filename = f"network_{i}_episode_{episode}.pth"
            else:
                filename = f"network_{i}.pth"
            
            filepath = os.path.join(directory, filename)
            network.load(filepath)
        
        # Load agent state
        if episode is not None:
            state_filename = f"agent_state_episode_{episode}.pth"
        else:
            state_filename = "agent_state.pth"
        
        state_filepath = os.path.join(directory, state_filename)
        agent_state = torch.load(state_filepath)
        
        self.training_step = agent_state['training_step']
        self.episode_count = agent_state['episode_count']
        self.epsilon = agent_state['epsilon']
        
        print(f"Agent loaded from {directory}")
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'updater_stats': self.updater.get_statistics()
        }
        return stats
    
    def increment_episode(self):
        """Increment episode counter."""
        self.episode_count += 1
