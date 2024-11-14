import torch
import torchinfo
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import os
import logging

# Import custom modules
from losses import *
from envcuzyes import BreakoutEnv
from agent import Agent
from replaybuffer import ReplayBuffer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
torch.set_default_device(device)

# class ModelConfig:
#     """Configuration class for model parameters."""
#     ht_size: int = 256
#     image_size: Tuple[int, int, int] = (64, 64)
#     img_channel: int = 3
#     vector_dim: int = None
#     num_codes: int = 32
#     num_latents: int = 32
#     embed_dim: int = 128
#     bin_num: int = 10
#     bin_start: float = -20.0
#     bin_end: float = 20.0
#     imagine_horizon: int = 4
#     action_size: int = 4  # Breakout has 4 actions
#     action_type: str = 'discrete'
#     action_embed_size: int = 16
#     device: str = device

@dataclass
class ModelConfig:
    """Configuration class for model parameters."""
    ht_size: int = 256
    image_size: Tuple[int, int, int] = (64, 64)
    img_channel: int = 3
    vector_dim: int = None
    num_codes: int = 10
    num_latents: int = 16
    embed_dim: int = 128
    bin_num: int = 6
    bin_start: float = -20.0
    bin_end: float = 20.0
    imagine_horizon: int = 32
    action_size: int = 4  # Breakout has 4 actions
    action_type: str = 'discrete'
    action_embed_size: int = 16
    device: str = device

class TrainingEnvironment:
    """Class to handle the training environment setup and configuration."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.env = BreakoutEnv(render_mode='human')
        self.replay_buffer1 = self._setup_replay_buffer()
        self.agent1 = self._setup_agents()
        self.agent1_opt= self._setup_optimizers()

        checkpoint_path = r'runs\run_20241031_004835\checkpoints\checkpoint_ep_170.pt'
        self.agent1.load_state_dict(torch.load(checkpoint_path)['agent1_state_dict'])

        self.setup_logging_and_checkpoints()
        
    def setup_logging_and_checkpoints(self):
        """Initialize logging configuration and checkpoint directories"""
        # Create run directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = Path(f'runs/run_{timestamp}')
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.log_file = self.run_dir / 'training_log.log'
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Log initial configuration
        self.logger.info("=== Training Configuration ===")
        for key, value in self.config.__dict__.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("==========================")

    def save_checkpoint(self, episode: int, metrics: dict):
        """Save model checkpoint and metrics"""
        checkpoint = {
            'episode': episode,
            'agent1_state_dict': self.agent1.state_dict(),
            'agent1_optimizer': self.agent1_opt.state_dict(),
            'metrics': metrics
        }
        checkpoint_path = self.checkpoint_dir / f'checkpoint_ep_{episode}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint at episode {episode}")

    def _setup_replay_buffer(self) -> ReplayBuffer:
        """Initialize the replay buffer with configuration parameters."""
        return ReplayBuffer(
            main_capacity=10000,
            online_capacity=1000,
            batch_size=128,
            chunk_length=self.config.imagine_horizon,
            latent_dim=self.config.num_latents * self.config.num_codes,
            recurrent_dim=self.config.ht_size,
            bin_number=self.config.bin_num,
            obs_dim=self.config.image_size,
            action_dim=self.config.action_size,
            reward_dim=1,
            rt_probs_dim=self.config.bin_num,
            include_obs=True,
            include_time=True
        )

    def _setup_agents(self) -> Tuple[Agent, Agent]:
        """Initialize both agents with the model configuration."""
        agent1 = Agent(config=self.config.__dict__).eval()
        return agent1

    def _setup_optimizers(self) -> Tuple[optim.Adam, optim.Adam]:
        """Initialize optimizers for both agents."""
        return optim.Adam(self.agent1.parameters(), lr=1e-4)

    def _get_state(self, state: np.ndarray) -> torch.Tensor:
        """
        Convert the state (which is a numpy array) into a PyTorch tensor.
        Args:
            state: Observation from the environment, typically a numpy array.
        Returns:
            state_tensor: The state converted into a PyTorch tensor.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize image
        state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
        return state_tensor

    def train(self, num_episodes: int = 0) -> None:
        """
        Train the agents for a specified number of episodes.
        
        Args:
            num_episodes: Number of training episodes to run
        """
        total_time_step = 0
        episode_metrics = {
            'episode_rewards': [],
            'consistency_losses': [],
            'world_model_losses': [],
            'reconstruction_losses': [],
            'dynamic_losses': [],
            'bce_losses': []
        }
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        summary = torchinfo.summary(self.agent1)
        print(summary)

        # Access specific values:
        total_params = summary.total_params
        trainable_params = summary.trainable_params
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        episode = 0
        while True:
            episode += 1
            if episode - 1 == num_episodes:
                break

            episode_start_time = datetime.now()
            episode_reward = 0
            episode_consistency_losses = []

            # Reset environment and get initial state
            obs, _ = self.env.reset()
            state = self._get_state(obs).to(device)
            
            ht = torch.zeros(1, self.config.ht_size).to(device)
            reward_hidden = (
                torch.zeros(1, 1, self.config.ht_size).to(device),
                torch.zeros(1, 1, self.config.ht_size).to(device)
            )
            
            done = False
            first_state = (state.clone(), ht.clone())
            prev_states = []
            prev_actions = []
            terminal = False
            while not terminal:
                # Select current agent and optimizer based on player turn
                current_agent, current_opt, current_buffer = (self.agent1, self.agent1_opt, self.replay_buffer1)

                # Agent action and state processing
                zt = current_agent.encoder(ht, state)
                vt, vt_probs, next_reward_hidden = current_agent.critic(zt.unsqueeze(1), ht.unsqueeze(1), reward_hidden)
                action, c_rt = current_agent.act(zt, ht, reward_hidden)
                zt1, ht1, ct, rt, rt_probs, recon_img, recon_vect = current_agent.rssm(
                    ht,
                    zt,
                    torch.tensor([action[0]])
                )
                # print(ct)
                # Take a step in the environment
                next_obs, reward, terminated, truncated, _ = self.env.step(action[0])
                done = terminated or truncated
                # print(int(not done))
                next_state = self._get_state(next_obs).to(device)
                prev_states.append((state.clone(), ht1.clone()))
                prev_actions.append(torch.tensor([action[0]]).clone())
                if len(prev_states) > 32:
                    first_state = (prev_states[0][0].clone().detach(), prev_states[0][1].clone().detach())
                    prev_states.pop(0)
                    prev_actions.pop(0)
                # Store experience in the replay buffer
                current_buffer.add_experience(
                    zt.squeeze(), torch.tensor(action[0]), torch.tensor(reward), ct.squeeze(-1), torch.tensor(not done).int(), zt1.squeeze(), ht.squeeze(), vt_probs.squeeze(), reward_hidden, rt_probs.squeeze(), state.squeeze(), time_step=total_time_step
                )

                # Unroll for the imagine horizon to calculate consistency loss
                consistency_loss = 0
                
                ht_next = first_state[1]
                zt_next = current_agent.encoder(ht_next, first_state[0])
                for xt, at in zip(prev_states, prev_actions):  # Unroll for n steps
                    real_zt_pred = current_agent.encoder(xt[1], xt[0].to(self.config.device))
                    # Predict next latent state
                    zt_next, ht_next_, _, _, _, _, _ = current_agent.rssm(
                        ht_next,
                        zt_next,
                        at
                    )

                    consistency_loss += current_agent.consistency(real_zt_pred.detach(), zt_next)

                consistency_loss /= 32
                if total_time_step % 50 == 0:
                    print(f"Timestep: {total_time_step}, Consistency Loss: {consistency_loss.item()+1}")

                episode_reward += reward
                episode_consistency_losses.append(consistency_loss.item())
                
                # Optimize agent
                current_agent.train()
                current_opt.zero_grad()
                consistency_loss.backward()
                current_opt.step()
                current_agent.eval()

                # Update state and reward_hidden
                state = next_state
                reward_hidden = next_reward_hidden

                total_time_step += 1
                # if total_time_step > 10 + 10 * episode:
                #     done = True

                # Render the environment
                self.env.render()
                terminal = done

            if episode % 10 == 0:
                self.save_checkpoint(episode, episode_metrics)

            batch, batch_idx, in_online_queue = self.replay_buffer1.sample_minibatch()
            xt = batch['xt']
            ht = batch['ht']
            zt = self.agent1.encoder(batch['ht'], batch['xt'])
            rt = batch['rt']
            terminality = batch['terminality'].float()
            # print(batch_idx)
            # print(in_online_queue)
            self.agent1.train()
            self.agent1_opt.zero_grad()
            # Compute the reconstruction loss using symlog squared error
            decoder_output = self.agent1.decoder(zt, ht)[0]
            reconstruction_loss = -torch.log(symlog_squared_error(xt, decoder_output))

            # Compute the dynamic prediction loss using symexp twohot loss
            dyna_pred_output = self.agent1.dyna_pred(zt, ht, ht, probs=True)
            dynamic_loss = -torch.log(symexp_twohot_loss(
                dyna_pred_output[3], 
                rt, 
                num_bins=self.config.bin_num, 
                bin_range=(self.config.bin_start, self.config.bin_end)
            ))

            # Compute the binary cross-entropy loss for terminality
            # print(dyna_pred_output[4])
            # print(terminality)
            bce = F.binary_cross_entropy(
                dyna_pred_output[4].squeeze(-1), 
                terminality)

            bce_loss = bce+1e-10

            # Total loss for prediction
            # print(reconstruction_loss.item(), dynamic_loss.item(), bce_loss.item())
            L_pred = reconstruction_loss + dynamic_loss + bce_loss

            # Free bits loss for dynamics prediction
            L_dyn = free_bits_loss(self.agent1.encoder(ht, xt, probs=True).detach(), dyna_pred_output[0])
            L_rep = (free_bits_loss(self.agent1.encoder(ht, xt, probs=True), dyna_pred_output[0].detach()))

            Total_L = L_pred + L_dyn + L_rep * 0.1
            
            episode_duration = datetime.now() - episode_start_time
            episode_metrics['episode_rewards'].append(episode_reward)
            episode_metrics['consistency_losses'].append(np.mean(episode_consistency_losses))
            episode_metrics['world_model_losses'].append(Total_L.item())
            episode_metrics['reconstruction_losses'].append(reconstruction_loss.item())
            episode_metrics['dynamic_losses'].append(dynamic_loss.item())
            episode_metrics['bce_losses'].append(bce_loss.item())

            self.logger.info(
                f"\nEpisode {episode} Summary:\n"
                f"Duration: {episode_duration}\n"
                f"Total Reward: {episode_reward}\n"
                f"Avg Consistency Loss: {np.mean(episode_consistency_losses):.4f}\n"
                f"World Model Loss: {Total_L.item():.4f}\n"
                f"Reconstruction Loss: {reconstruction_loss.item():.4f}\n"
                f"Dynamic Loss: {L_dyn.item():.4f}\n"
                f"BCE Loss: {bce_loss.item():.4f}\n"
                f"Rep Loss: {L_rep.item():.4f}"
            )

            print(f"\033[94mEpisode {episode}, World Model loss: {Total_L.item()}\033[0m")
            Total_L.backward()
            self.agent1_opt.step()
            self.agent1.eval()
            self.replay_buffer1.copy_online_to_main()

        training_duration = datetime.now() - training_start_time
        self.logger.info(
            f"\nTraining Complete!\n"
            f"Total Episodes: {episode}\n"
            f"Total Duration: {training_duration}\n"
            f"Average Reward: {np.mean(episode_metrics['episode_rewards']):.2f}\n"
            f"Best Reward: {max(episode_metrics['episode_rewards']):.2f}\n"
            f"Final World Model Loss: {episode_metrics['world_model_losses'][-1]:.4f}"
        )

def main():
    """Main function to set up and run the training."""
    config = ModelConfig()
    training_env = TrainingEnvironment(config)
    training_env.train(num_episodes=-1)

if __name__ == "__main__":
    main()
