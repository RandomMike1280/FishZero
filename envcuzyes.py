import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import numpy as np
import cv2
from gymnasium import spaces
import ale_py

class BreakoutEnv(gym.Env):
    """Custom Breakout environment that follows gym interface"""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Default render_mode to 'rgb_array' if None is provided
        if render_mode is None:
            render_mode = 'rgb_array'
        
        # Initialize the Atari environment with frameskip=1 to disable default frame-skipping
        self.env = gym.make("ALE/Breakout-v5", render_mode=render_mode, frameskip=1)
        
        # Apply standard Atari preprocessing, including custom frame_skip
        self.env = AtariPreprocessing(
            self.env,
            noop_max=30,
            frame_skip=6,  # Set frame_skip here
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=False
        )
        
        # Define action and observation spaces
        self.action_space = self.env.action_space
        
        # Observation space: 64x64 RGB image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(64, 64, 3),
            dtype=np.uint8
        )

    def _process_observation(self, obs):
        """Resize observation to 64x64."""
        return cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        processed_obs = self._process_observation(obs)
        return processed_obs, info

    def step(self, action):
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self._process_observation(obs)
        return processed_obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()


# Example usage
if __name__ == "__main__":
    # Create environment
    env = BreakoutEnv(render_mode="human")
    
    # Test environment
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}") # (64, 64, 3)
    print(f"Action space: {env.action_space}") # discrete(4)
    
    for _ in range(1000):
        # Random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
