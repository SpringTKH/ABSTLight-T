"""
Base DQN Network for RELight Architecture

This module implements the standard 3-layer CNN architecture used as the
fundamental Q-network in the RELight ensemble. Each network in the ensemble
is an instance of this architecture.

Architecture:
- Conv2D: 32 filters, 8x8 kernel, stride 4
- Conv2D: 64 filters, 4x4 kernel, stride 2
- Conv2D: 64 filters, 3x3 kernel, stride 1
- Fully Connected layers for Q-value output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseDQN(nn.Module):
    """
    Base Deep Q-Network with 3-layer CNN architecture.
    
    This network serves as a single Q-value estimator within the RELight
    ensemble. The architecture follows the standard DQN design for
    processing visual state representations.
    
    Args:
        state_shape (tuple): Shape of input state (channels, height, width)
        num_actions (int): Number of possible actions
        
    Architecture Details:
        - Layer 1: Conv2D(in_channels, 32, kernel=8, stride=4) + ReLU
        - Layer 2: Conv2D(32, 64, kernel=4, stride=2) + ReLU
        - Layer 3: Conv2D(64, 64, kernel=3, stride=1) + ReLU
        - Flatten
        - FC Layer 1: Linear(flattened_size, 512) + ReLU
        - FC Layer 2: Linear(512, num_actions)
    """
    
    def __init__(self, state_shape, num_actions):
        """
        Initialize the Base DQN network.
        
        Args:
            state_shape (tuple): Shape of the input state (C, H, W)
            num_actions (int): Number of discrete actions
        """
        super(BaseDQN, self).__init__()
        
        self.state_shape = state_shape
        self.num_actions = num_actions
        
        # Extract dimensions
        channels, height, width = state_shape
        
        # Convolutional layers following the standard DQN architecture
        # Layer 1: Conv(32 filters, 8x8 kernel, stride 4)
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=32,
            kernel_size=8,
            stride=4,
            padding=0
        )
        
        # Layer 2: Conv(64 filters, 4x4 kernel, stride 2)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=0
        )
        
        # Layer 3: Conv(64 filters, 3x3 kernel, stride 1)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        
        # Calculate the flattened size after convolutions
        self.flatten_size = self._get_conv_output_size(state_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_conv_output_size(self, shape):
        """
        Calculate the output size after all convolutional layers.
        
        Args:
            shape (tuple): Input shape (C, H, W)
            
        Returns:
            int: Flattened size after convolutions
        """
        # Create a dummy input tensor
        dummy_input = torch.zeros(1, *shape)
        
        # Pass through convolutional layers
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Return flattened size
        return int(np.prod(x.size()))
    
    def _initialize_weights(self):
        """
        Initialize network weights using He initialization for ReLU activation.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): Input state tensor of shape (batch, C, H, W)
            
        Returns:
            torch.Tensor: Q-values for each action, shape (batch, num_actions)
        """
        # Normalize input to [0, 1] if needed
        if state.dtype == torch.uint8:
            state = state.float() / 255.0
        
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values
    
    def get_action(self, state, epsilon=0.0):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state (torch.Tensor): Current state
            epsilon (float): Exploration rate
            
        Returns:
            int: Selected action index
        """
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Random action (exploration)
            return np.random.randint(0, self.num_actions)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                # Add batch dimension if needed
                if state.dim() == 3:
                    state = state.unsqueeze(0)
                
                q_values = self.forward(state)
                action = q_values.argmax(dim=1).item()
                
            return action
    
    def save(self, filepath):
        """
        Save model weights to file.
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'state_dict': self.state_dict(),
            'state_shape': self.state_shape,
            'num_actions': self.num_actions
        }, filepath)
    
    def load(self, filepath):
        """
        Load model weights from file.
        
        Args:
            filepath (str): Path to load the model from
        """
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['state_dict'])
        
    def copy_weights_from(self, source_network):
        """
        Copy weights from another network.
        
        Args:
            source_network (BaseDQN): Source network to copy from
        """
        self.load_state_dict(source_network.state_dict())
