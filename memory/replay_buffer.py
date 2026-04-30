"""
Replay Buffer - Experience Replay Memory

This module implements the experience replay buffer used in DQN-based algorithms.
The replay buffer stores transitions (state, action, reward, next_state, done) and
provides random sampling for training, which helps break temporal correlations
in the data and improves learning stability.

The buffer uses a circular/ring buffer approach where old experiences are
overwritten when the buffer reaches capacity.
"""

import numpy as np
import torch
from collections import deque
from typing import Tuple, List


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions.
    
    The replay buffer stores transitions in the form:
        (state, action, reward, next_state, done)
    
    Features:
    - Fixed size circular buffer (oldest experiences overwritten)
    - Random batch sampling for training
    - Efficient numpy storage for memory optimization
    - Automatic type conversion for PyTorch compatibility
    
    Args:
        capacity (int): Maximum number of transitions to store
        state_shape (tuple): Shape of state observations.
            For ABSTLight: (N_agents, obs_dim), where obs_dim may include
            temporal stacking (T * raw_obs_dim).
        device (str): Device to move sampled batches to ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        capacity: int,
        state_shape: Tuple[int, ...],
        device: str = 'cpu'
    ):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum buffer size
            state_shape: Dimensions of state.
                For ABSTLight: (N_agents, obs_dim), where obs_dim may include
                temporal stacking (T * raw_obs_dim).
            device: Device for tensor conversion
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.device = device
        self.num_agents = int(state_shape[0]) if len(state_shape) > 0 else 1
        
        # Initialize storage buffers
        # Using numpy arrays for memory efficiency
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, self.num_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, self.num_agents), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # Buffer tracking
        self.position = 0  # Current write position
        self.size = 0      # Current number of stored transitions
    
    def push(
        self,
        state: np.ndarray,
        action,
        reward,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add a new transition to the buffer.
        
        If buffer is full, the oldest transition is overwritten (circular buffer).
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode terminated
        """
        # Normalize per-agent action/reward vectors for storage.
        if np.isscalar(action):
            action_vec = np.full((self.num_agents,), int(action), dtype=np.int64)
        else:
            action_vec = np.asarray(action, dtype=np.int64).reshape(-1)
            if action_vec.shape[0] != self.num_agents:
                raise ValueError(
                    f"Action shape mismatch: got {action_vec.shape[0]}, expected {self.num_agents}."
                )

        if np.isscalar(reward):
            reward_vec = np.full((self.num_agents,), float(reward), dtype=np.float32)
        else:
            reward_vec = np.asarray(reward, dtype=np.float32).reshape(-1)
            if reward_vec.shape[0] != self.num_agents:
                raise ValueError(
                    f"Reward shape mismatch: got {reward_vec.shape[0]}, expected {self.num_agents}."
                )

        # Store transition at current position
        self.states[self.position] = state
        self.actions[self.position] = action_vec
        self.rewards[self.position] = reward_vec
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        
        # Update position (circular)
        self.position = (self.position + 1) % self.capacity
        
        # Update size (capped at capacity)
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as PyTorch tensors
            
        Raises:
            ValueError: If batch_size > buffer size
        """
        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer with only {self.size} items"
            )
        
        # Random sampling without replacement
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # Extract batch
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]
        
        # Convert to PyTorch tensors and move to device
        states_tensor = torch.from_numpy(batch_states).to(self.device)
        actions_tensor = torch.from_numpy(batch_actions).to(self.device)
        rewards_tensor = torch.from_numpy(batch_rewards).to(self.device)
        next_states_tensor = torch.from_numpy(batch_next_states).to(self.device)
        dones_tensor = torch.from_numpy(batch_dones).to(self.device)
        
        return (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor
        )
    
    def __len__(self) -> int:
        """
        Get current number of transitions in buffer.
        
        Returns:
            Number of stored transitions
        """
        return self.size
    
    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough samples for training.
        
        Args:
            min_size: Minimum required number of samples
            
        Returns:
            True if buffer size >= min_size
        """
        return self.size >= min_size
    
    def clear(self):
        """
        Clear all transitions from the buffer.
        """
        self.position = 0
        self.size = 0
        
        # Reset arrays (optional, for memory cleanup)
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_states.fill(0)
        self.dones.fill(0)
    
    def get_last_n(self, n: int) -> Tuple[torch.Tensor, ...]:
        """
        Get the last n transitions from the buffer (most recent).
        
        Useful for on-policy updates or debugging.
        
        Args:
            n: Number of recent transitions to retrieve
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if n > self.size:
            n = self.size
        
        # Calculate indices for last n items (accounting for circular buffer)
        if self.position >= n:
            indices = np.arange(self.position - n, self.position)
        else:
            # Wrap around the buffer
            indices = np.concatenate([
                np.arange(self.capacity - (n - self.position), self.capacity),
                np.arange(0, self.position)
            ])
        
        # Extract transitions
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]
        
        # Convert to tensors
        states_tensor = torch.from_numpy(batch_states).to(self.device)
        actions_tensor = torch.from_numpy(batch_actions).to(self.device)
        rewards_tensor = torch.from_numpy(batch_rewards).to(self.device)
        next_states_tensor = torch.from_numpy(batch_next_states).to(self.device)
        dones_tensor = torch.from_numpy(batch_dones).to(self.device)
        
        return (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor
        )
    
    def get_statistics(self) -> dict:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary with buffer statistics
        """
        stats = {
            'capacity': self.capacity,
            'size': self.size,
            'position': self.position,
            'utilization': self.size / self.capacity
        }
        
        if self.size > 0:
            stats.update({
                'avg_reward': float(np.mean(self.rewards[:self.size].sum(axis=1))),
                'min_reward': float(np.min(self.rewards[:self.size].sum(axis=1))),
                'max_reward': float(np.max(self.rewards[:self.size].sum(axis=1))),
                'done_ratio': float(np.mean(self.dones[:self.size]))
            })
        
        return stats


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer (optional enhancement).
    
    This is an advanced version that samples transitions based on their
    TD-error, giving priority to more "surprising" experiences.
    
    Note: This is a stub for future implementation. For RELight baseline,
    use the standard ReplayBuffer above.
    """
    
    def __init__(
        self,
        capacity: int,
        state_shape: Tuple[int, ...],
        alpha: float = 0.6,
        beta: float = 0.4,
        device: str = 'cpu'
    ):
        """
        Initialize Prioritized Replay Buffer.
        
        Args:
            capacity: Maximum buffer size
            state_shape: State dimensions.
                For ABSTLight: (N_agents, obs_dim), where obs_dim may include
                temporal stacking (T * raw_obs_dim).
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
            device: Device for tensors
        """
        super().__init__(capacity, state_shape, device)
        self.alpha = alpha
        self.beta = beta
        
        # Priority storage (to be implemented)
        self.priorities = np.ones(capacity, dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done, priority=None):
        """Add transition with priority (to be implemented)."""
        # TODO: Implement prioritized storage
        super().push(state, action, reward, next_state, done)
    
    def sample(self, batch_size: int):
        """Sample based on priorities (to be implemented)."""
        # TODO: Implement prioritized sampling
        return super().sample(batch_size)
