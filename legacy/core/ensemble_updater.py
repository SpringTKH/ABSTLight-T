"""
Ensemble Updater - Data Repeated Sampling and Updating Module

This is the core algorithmic component of RELight that differentiates it from
standard DQN. It implements:

1. Random Subset Sampling: Randomly selects M networks from N total networks
2. Minimum Target Q-value: Computes target Q-value as min over the subset
3. Error-Triggered Resampling: Rejects subsets with high error and resamples

The key innovation is the error threshold loop which reduces overestimation
bias by ensuring only "confident" network subsets are used for updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict


class EnsembleUpdater:
    """
    Data Repeated Sampling and Updating Module for RELight.
    
    This module manages the update process for the ensemble of Q-networks.
    Unlike standard DQN which uses a fixed target network, RELight:
    
    1. Maintains N independent Q-networks
    2. For each update, randomly samples M networks as "target ensemble"
    3. Computes target Q-value as the MINIMUM among the M sampled networks
    4. Calculates update error between predicted and target Q-values
    5. If error > threshold, rejects the subset and samples a new one
    6. Repeats until error <= threshold or max attempts reached
    
    This adaptive resampling mechanism helps reduce overestimation bias
    while maintaining diversity in the ensemble.
    
    Args:
        networks (List[nn.Module]): List of N Q-networks in the ensemble
        learning_rate (float): Learning rate for optimizer
        gamma (float): Discount factor for future rewards
        error_threshold (float): Maximum acceptable update error
        max_resampling_attempts (int): Maximum number of subset resampling attempts
        device (str): Device to run computations on ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        networks: List[nn.Module],
        learning_rate: float = 1e-4,
        gamma: float = 0.95,
        error_threshold: float = 150.0,
        reward_scale_divisor: float = 1000.0,
        use_dynamic_error_threshold: bool = True,
        relative_error_threshold_ratio: float = 0.05,
        min_target_scale_for_threshold: float = 1.0,
        max_resampling_attempts: int = 5,
        device: str = 'cpu'
    ):
        """
        Initialize the Ensemble Updater.
        
        Args:
            networks: List of Q-networks (length N)
            learning_rate: Learning rate for gradient descent
            gamma: Discount factor
            error_threshold: Threshold for error-triggered resampling
            max_resampling_attempts: Max resampling attempts before giving up
            device: Computation device
        """
        self.networks = networks
        self.N = len(networks)  # Total number of networks
        self.gamma = gamma
        self.error_threshold = error_threshold
        self.reward_scale_divisor = float(reward_scale_divisor)
        self.use_dynamic_error_threshold = bool(use_dynamic_error_threshold)
        self.relative_error_threshold_ratio = float(relative_error_threshold_ratio)
        self.min_target_scale_for_threshold = float(min_target_scale_for_threshold)
        self.max_resampling_attempts = max_resampling_attempts
        self.device = device

        if self.reward_scale_divisor <= 0.0:
            raise ValueError('reward_scale_divisor must be > 0')
        if self.relative_error_threshold_ratio < 0.0:
            raise ValueError('relative_error_threshold_ratio must be >= 0')
        if self.min_target_scale_for_threshold <= 0.0:
            raise ValueError('min_target_scale_for_threshold must be > 0')
        
        # Create separate optimizers for each network
        self.optimizers = [
            optim.Adam(net.parameters(), lr=learning_rate)
            for net in networks
        ]
        
        # Huber loss is more robust to outlier TD errors than MSE.
        self.criterion = nn.SmoothL1Loss()
        
        # Statistics tracking
        self.update_stats = {
            'total_updates': 0,
            'resampling_count': 0,
            'avg_resampling_per_update': 0.0,
            'rejected_subsets': 0,
            'skipped_updates': 0,
            'threshold_evaluations': 0,
            'avg_effective_threshold': 0.0,
            'avg_target_abs_mean': 0.0
        }
    
    def sample_subset(self, M: int, exclude_indices: List[int] = None) -> List[int]:
        """
        Randomly sample M network indices from the ensemble.
        
        This implements the "Random Subset Sampling" mechanism of RELight.
        
        Args:
            M: Number of networks to sample
            exclude_indices: Optional list of indices to exclude from sampling
            
        Returns:
            List of M randomly sampled network indices
            
        Note:
            Sampling is done WITHOUT replacement to ensure diversity.
        """
        available_indices = list(range(self.N))
        
        # Remove excluded indices if specified
        if exclude_indices is not None:
            available_indices = [i for i in available_indices if i not in exclude_indices]
        
        # Ensure we don't sample more than available
        M = min(M, len(available_indices))
        
        # Random sampling without replacement
        sampled_indices = np.random.choice(
            available_indices,
            size=M,
            replace=False
        ).tolist()
        
        return sampled_indices
    
    def compute_target_q_value(
        self,
        subset_indices: List[int],
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute target Q-value using the minimum over the sampled subset.
        
        This is the "Random Subset Target Calculation" step of RELight.
        Target Q-value = reward + gamma * min(Q_i(s', a*)) for i in subset
        
        Args:
            subset_indices: Indices of networks in the target subset
            next_states: Next states from transitions (batch_size, C, H, W)
            rewards: Rewards from transitions (batch_size,)
            dones: Terminal flags (batch_size,)
            
        Returns:
            Target Q-values (batch_size,)
            
        Algorithm:
            1. For each network in subset, compute Q-values for next states
            2. For each state, select the max Q-value (best action)
            3. Take the MINIMUM across all networks in subset
            4. Compute: target = reward + gamma * min_q_value * (1 - done)
        """
        batch_size = next_states.size(0)
        
        # Collect Q-values from all networks in the subset
        next_q_values_list = []
        
        with torch.no_grad():
            for idx in subset_indices:
                # Get Q-values from network idx for next states
                q_values = self.networks[idx](next_states)  # (batch_size, num_actions)
                
                # Select max Q-value for each state (best action)
                max_q_values = q_values.max(dim=1)[0]  # (batch_size,)
                
                next_q_values_list.append(max_q_values)
        
        # Stack into tensor: (len(subset), batch_size)
        next_q_values_tensor = torch.stack(next_q_values_list, dim=0)
        
        # Take MINIMUM across the subset for each state: (batch_size,)
        min_next_q_values = next_q_values_tensor.min(dim=0)[0]
        
        # Compute target: scaled_r + gamma * min(Q) * (1 - done)
        scaled_rewards = rewards / self.reward_scale_divisor
        target_q_values = scaled_rewards + self.gamma * min_next_q_values * (1 - dones)
        
        return target_q_values

    def _effective_error_threshold(self, target_q_values: torch.Tensor) -> Tuple[float, float]:
        """Compute absolute+relative threshold based on current target scale."""
        target_abs_mean = float(target_q_values.detach().abs().mean().item())
        if self.use_dynamic_error_threshold:
            relative_component = self.relative_error_threshold_ratio * max(
                target_abs_mean,
                self.min_target_scale_for_threshold
            )
            threshold = max(self.error_threshold, relative_component)
        else:
            threshold = self.error_threshold
        return float(threshold), target_abs_mean
    
    def compute_update_error(
        self,
        network_idx: int,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_q_values: torch.Tensor
    ) -> float:
        """
        Compute the update error between predicted and target Q-values.
        
        This error is used in the "Error-Triggered Resampling" mechanism.
        
        Args:
            network_idx: Index of network to compute error for
            states: Current states (batch_size, C, H, W)
            actions: Actions taken (batch_size,)
            target_q_values: Target Q-values from subset (batch_size,)
            
        Returns:
            Update error (scalar)
        """
        with torch.no_grad():
            # Get predicted Q-values for the taken actions
            predicted_q_values = self.networks[network_idx](states)
            predicted_q_values = predicted_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute MSE error
            error = self.criterion(predicted_q_values, target_q_values).item()
        
        return error
    
    def update_network(
        self,
        network_idx: int,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        M: int
    ) -> Dict[str, float]:
        """
        Update a single network using the RELight algorithm.
        
        This implements the complete "Error-Triggered Resampling Loop":
        
        1. Sample M networks as target subset
        2. Compute target Q-value as min over subset
        3. Calculate update error
        4. If error > threshold, resample subset and repeat
        5. If error <= threshold, perform gradient update
        6. Repeat until success or max attempts reached
        
        Args:
            network_idx: Index of network to update
            states: Batch of current states
            actions: Batch of actions taken
            rewards: Batch of rewards received
            next_states: Batch of next states
            dones: Batch of terminal flags
            M: Number of networks to sample for target subset
            
        Returns:
            Dictionary containing update statistics
        """
        attempts = 0
        successful = False
        final_error = float('inf')
        effective_threshold = self.error_threshold
        target_abs_mean = 0.0
        
        # Error-triggered resampling loop
        while attempts < self.max_resampling_attempts and not successful:
            attempts += 1
            
            # 1. Random Subset Sampling (exclude the network being updated)
            subset_indices = self.sample_subset(M, exclude_indices=[network_idx])
            
            # 2. Compute target Q-value as minimum over subset
            target_q_values = self.compute_target_q_value(
                subset_indices, next_states, rewards, dones
            )
            
            # 3. Calculate update error
            error = self.compute_update_error(
                network_idx, states, actions, target_q_values
            )
            effective_threshold, target_abs_mean = self._effective_error_threshold(target_q_values)
            self.update_stats['threshold_evaluations'] += 1
            n_evals = self.update_stats['threshold_evaluations']
            self.update_stats['avg_effective_threshold'] += (
                effective_threshold - self.update_stats['avg_effective_threshold']
            ) / n_evals
            self.update_stats['avg_target_abs_mean'] += (
                target_abs_mean - self.update_stats['avg_target_abs_mean']
            ) / n_evals
            
            final_error = error
            
            # 4. Check error threshold
            if error <= effective_threshold:
                successful = True
                # Perform the actual gradient update
                self._gradient_update(network_idx, states, actions, target_q_values)
            else:
                # Reject this subset and resample
                self.update_stats['rejected_subsets'] += 1
        
        # If max attempts reached without success, skip this update safely.
        if not successful:
            self.update_stats['total_updates'] += 1
            self.update_stats['resampling_count'] += attempts
            self.update_stats['skipped_updates'] += 1
            self.update_stats['avg_resampling_per_update'] = (
                self.update_stats['resampling_count'] / self.update_stats['total_updates']
            )
            return {
                'error': final_error,
                'attempts': attempts,
                'successful': False,
                'subset_size': M,
                'effective_threshold': effective_threshold,
                'target_abs_mean': target_abs_mean,
                'skipped': True
            }
        
        # Update statistics
        self.update_stats['total_updates'] += 1
        self.update_stats['resampling_count'] += attempts
        self.update_stats['avg_resampling_per_update'] = (
            self.update_stats['resampling_count'] / self.update_stats['total_updates']
        )
        
        return {
            'error': final_error,
            'attempts': attempts,
            'successful': successful,
            'subset_size': M,
            'effective_threshold': effective_threshold,
            'target_abs_mean': target_abs_mean,
            'skipped': False
        }
    
    def _gradient_update(
        self,
        network_idx: int,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_q_values: torch.Tensor
    ):
        """
        Perform gradient descent update on a single network.
        
        Args:
            network_idx: Index of network to update
            states: Batch of states
            actions: Batch of actions
            target_q_values: Target Q-values
        """
        # Set network to training mode
        self.networks[network_idx].train()
        
        # Zero gradients
        self.optimizers[network_idx].zero_grad()
        
        # Forward pass
        predicted_q_values = self.networks[network_idx](states)
        
        # Select Q-values for taken actions
        predicted_q_values = predicted_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute loss
        loss = self.criterion(predicted_q_values, target_q_values)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (optional, helps with stability)
        torch.nn.utils.clip_grad_norm_(self.networks[network_idx].parameters(), max_norm=10.0)
        
        # Update weights
        self.optimizers[network_idx].step()
    
    def update_all_networks(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        M: int
    ) -> List[Dict[str, float]]:
        """
        Update all N networks in the ensemble.
        
        Args:
            states: Batch of current states
            actions: Batch of actions taken
            rewards: Batch of rewards received
            next_states: Batch of next states
            dones: Batch of terminal flags
            M: Number of networks in target subset
            
        Returns:
            List of update statistics for each network
        """
        update_results = []
        
        for i in range(self.N):
            result = self.update_network(
                i, states, actions, rewards, next_states, dones, M
            )
            update_results.append(result)
        
        return update_results
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get update statistics.
        
        Returns:
            Dictionary containing update statistics
        """
        return self.update_stats.copy()
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.update_stats = {
            'total_updates': 0,
            'resampling_count': 0,
            'avg_resampling_per_update': 0.0,
            'rejected_subsets': 0,
            'skipped_updates': 0,
            'threshold_evaluations': 0,
            'avg_effective_threshold': 0.0,
            'avg_target_abs_mean': 0.0
        }
