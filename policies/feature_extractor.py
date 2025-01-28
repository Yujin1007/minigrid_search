import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        # Initialize the BaseFeaturesExtractor
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # Ensure observation_space is compatible with a CNN (e.g., 2D grid)
        n_input_channels = observation_space.shape[0]  # For example, 1 channel for grayscale
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # Calculate the output size after the CNN layers
        with th.no_grad():
            sample_input = th.zeros((1,) + observation_space.shape)  # Dummy input for size calculation
            n_flatten = self.cnn(sample_input).shape[1]

        # Fully connected layer to output `features_dim`
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # s = observations.shape
        # print(s,"$$$$$$$$$$$$$$$$$$")
        # print(observations)
        # if len(s) == 3:
        #     obs = observations.reshape(s[0], 1, s[1], s[2])
        # else:
        #     obs = observations.reshape(1, s[0], s[1])
        cnn_out = self.cnn(observations)
        return self.fc(cnn_out)