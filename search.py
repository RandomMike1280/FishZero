import torch
from typing import List, Dict, Tuple

class TrajectorySearch:
    def __init__(self, RSSM, Encoder, Actor, Critic, replay_buffer=None):
        """
        Initialize the TrajectorySearch.

        Args:
            RSSM: Recurrent State Space Model.
            Encoder: Encoder network to process observations.
            Actor: Policy network to generate action probabilities.
            Critic: Value network to estimate state values.
            replay_buffer: Buffer to store experiences.
        """
        self.rssm = RSSM
        self.encoder = Encoder
        self.policy = Actor
        self.value = Critic
        self.replay_buffer = replay_buffer

    def gumbel_top_k_trick(self, probs: torch.Tensor, k: int) -> List[List[int]]:
        """
        Apply the Gumbel-Top-K trick to sample top-k actions.

        Args:
            probs (Tensor): Action probabilities. Shape: [batch_size, num_actions]
            k (int): Number of top actions to sample.

        Returns:
            List[List[int]]: Batch-wise list of top-k action indices.
        """
        epsilon = 1e-20
        log_probs = torch.log(probs + epsilon)  # Log probabilities
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(log_probs) + epsilon) + epsilon)
        perturbed_log_probs = log_probs + gumbel_noise

        top_k_indices = torch.topk(perturbed_log_probs, k, dim=-1).indices  # Shape: [batch_size, k]
        return top_k_indices.tolist()

    def search(
        self,
        zt: torch.Tensor,
        ht: torch.Tensor,
        reward_hidden: tuple,
        depth: int,
        num_trajectories: int
    ) -> Tuple[List[Dict[str, List]], List[float]]:
        """
        Perform a trajectory search up to a certain depth and return all trajectories with their cumulative rewards.

        Args:
            zt (Tensor): Current latent state. Shape: [batch_size, latent_dim]
            ht (Tensor): Current hidden state. Shape: [batch_size, hidden_dim]
            depth (int): Depth of the search.
            num_trajectories (int): Number of trajectories to sample at each depth.

        Returns:
            Tuple[List[Dict[str, List]], List[float]]: 
                - List of trajectory dictionaries containing 'zt', 'ht', 'at', 'ct', 'rt', 'vt'.
                - List of cumulative rewards corresponding to each trajectory.
        """
        all_trajectories = []
        all_rewards = []

        def recursive_search(
            current_zt: torch.Tensor,
            current_ht: torch.Tensor,
            reward_hidden: Tuple,
            current_depth: int,
            trajectory_path: Dict[str, List],
            cumulative_reward: float,
            num_trajectories: int
        ):
            nonlocal all_trajectories, all_rewards, zt, ht

            zt = current_zt
            ht = current_ht

            if current_depth == 0:
                # Append the completed trajectory
                trajectory_copy = {
                    'zt': [zt.cpu().numpy() for zt in trajectory_path['zt']],
                    'ht': [ht.cpu().numpy() for ht in trajectory_path['ht']],
                    'at': trajectory_path['at'].copy(),
                    'ct': trajectory_path['ct'].copy(),
                    'rt': trajectory_path['rt'].copy(),
                    'vt': trajectory_path['vt'].copy(),
                    'reward_hidden': trajectory_path['reward_hidden'].copy()
                }
                all_trajectories.append(trajectory_copy)
                all_rewards.append(cumulative_reward)
                return

            # Get action probabilities from the policy network
            if len(trajectory_path['zt']) == 0:
                zts = current_zt.unsqueeze(1).to(current_zt.device)
                hts = current_ht.unsqueeze(1).to(current_zt.device)
            else:
                zts = torch.stack([z.to(current_zt.device) for z in trajectory_path['zt']], dim=1)
                hts = torch.stack([h.to(current_zt.device) for h in trajectory_path['ht']], dim=1)
            vt, vt_probs, reward_hidden = self.value(zts, hts, reward_hidden)
            a_probs = self.policy(current_zt, current_ht)  # Shape: [batch_size, num_actions]

            # Sample top-k actions using Gumbel-Top-K trick
            top_a = self.gumbel_top_k_trick(a_probs, num_trajectories)  # List[List[int]]
            top_a = torch.tensor(top_a, device=current_zt.device)

            # Assuming batch_size=1 for simplicity
            for action in top_a[0]:
                at = torch.tensor([action], dtype=torch.long, device=current_ht.device)  # Shape: [1]

                # Predict the next state using the RSSM
                rssm_output = self.rssm(current_ht, current_zt, at)
                if len(rssm_output) == 7:
                    zt_next, ht_next, ct, rt, rt_probs, _, _ = rssm_output
                else:
                    raise ValueError("RSSM output does not match the expected number of outputs.")

                # Detach tensors to prevent gradient backpropagation
                zt_next_detached = zt_next.detach()
                ht_next_detached = ht_next.detach()
                rt_detached = rt.detach()
                ct_detached = ct.detach()

                # Add the experience to the replay buffer
                if self.replay_buffer is not None:
                    self.replay_buffer.add_experience(
                        zt=current_zt.detach(),
                        at=at.detach(),
                        rt=rt_detached.detach(),
                        ct=ct_detached.detach(),
                        zt1=zt_next_detached.detach(),
                        ht=ht_next_detached.detach(),
                        vt=vt_probs.detach(),  # Replace with 'vt' if available or remove if unused
                        reward_hidden=reward_hidden,
                        rt_probs=rt_probs.detach(),
                        xt=None  # Assuming observation is not used here
                    )

                new_cumulative_reward = cumulative_reward + rt_detached.item()

                # Update the trajectory path
                new_trajectory_path = {
                    'zt': trajectory_path['zt'] + [current_zt.detach().to(current_zt.device)],
                    'ht': trajectory_path['ht'] + [current_ht.detach().to(current_zt.device)],
                    'at': trajectory_path['at'] + [at.item()],
                    'ct': trajectory_path['ct'] + [ct_detached.item()],
                    'rt': trajectory_path['rt'] + [rt_detached.item()],
                    'vt': trajectory_path['vt'] + [vt.to(current_zt.device)],  # Placeholder, adjust as needed
                    'reward_hidden': trajectory_path['reward_hidden'] + [reward_hidden]
                }

                # Check the continuation flag 'ct' to decide whether to continue searching
                # Assuming 'ct' is a binary flag indicating whether to continue (1) or terminate (0)
                if ct_detached.item() == 1:
                    # Continue searching deeper
                    recursive_search(
                        current_zt=zt_next_detached,
                        current_ht=ht_next_detached,
                        reward_hidden = reward_hidden,
                        current_depth=current_depth - 1,
                        trajectory_path=new_trajectory_path,
                        cumulative_reward=new_cumulative_reward,
                        num_trajectories = 1
                    )
                else:
                    # Terminate this trajectory as 'ct' indicates
                    trajectory_copy = {
                        'zt': [zt.cpu().numpy() for zt in new_trajectory_path['zt']] + [zt_next_detached.cpu().numpy()],
                        'ht': [ht.cpu().numpy() for ht in new_trajectory_path['ht']] + [ht_next_detached.cpu().numpy()],
                        'at': new_trajectory_path['at'] + [at.item()],
                        'ct': new_trajectory_path['ct'] + [ct_detached.item()],
                        'rt': new_trajectory_path['rt'] + [rt_detached.item()],
                        'vt': new_trajectory_path['vt'] + [vt],  # Placeholder, adjust as needed
                        'reward_hidden': trajectory_path['reward_hidden'] + [reward_hidden]
                    }
                    all_trajectories.append(trajectory_copy)
                    all_rewards.append(new_cumulative_reward)

        # Start the recursive search with an empty trajectory and zero cumulative reward
        initial_trajectory = {
            'zt': [],
            'ht': [],
            'at': [],
            'ct': [],
            'rt': [],
            'vt': [],
            'reward_hidden': []
        }
        recursive_search(
            current_zt=zt,
            current_ht=ht,
            reward_hidden=reward_hidden,
            current_depth=depth,
            trajectory_path=initial_trajectory,
            cumulative_reward=0.0,
            num_trajectories = num_trajectories
        )

        return all_trajectories, all_rewards
    
class SequentialHalving:
    def __init__(self, RSSM, Encoder, Actor, Critic, action_size: int) -> None:
        self.action_size = action_size
        self.rssm = RSSM
        self.encoder = Encoder
        self.policy = Actor
        self.value = Critic
        self.traj_search = TrajectorySearch(RSSM, Encoder, Actor, Critic)
    
    @staticmethod
    def gumbel_top_k_trick(probs: torch.Tensor, k: int) -> List[int]:
        """Efficient Gumbel-Top-k sampling without replacement."""
        log_probs = probs - torch.logsumexp(probs, dim=0)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(log_probs)))
        perturbed_log_probs = log_probs + gumbel_noise
        _, top_k_indices = torch.topk(perturbed_log_probs, k)
        return top_k_indices.tolist()
    
    def search(self, zt: torch.Tensor, ht: torch.Tensor, reward_hidden: tuple, init_depth: int, k: int):
        """Sequential halving search using beam search trajectories.
        
        Args:
            zt: Current latent observation
            ht: Current hidden state
            reward_hidden: Hidden state for reward prediction
            init_depth: Initial search depth
            k: Initial number of trajectories to consider
            
        Returns:
            Tuple containing:
                - best_trajectory: The trajectory with highest expected reward
                - best_reward: The accumulated reward for the best trajectory
        """
        # Initial number of trajectories and current depth
        num_trajectories = k
        current_depth = init_depth
        
        # Initial beam search
        trajectories, rewards = self.traj_search.search(
            zt, ht, reward_hidden, current_depth, num_trajectories)
        
        # num_nodes = 0
        # Continue until only one trajectory remains
        while num_trajectories > 1:
            # Convert rewards to probabilities for selection
            rewards_tensor = torch.tensor(rewards)
            probs = torch.softmax(rewards_tensor, dim=0)
            
            # Select top half of trajectories using Gumbel-Top-k
            num_trajectories = max(1, num_trajectories // 2)
            selected_indices = self.gumbel_top_k_trick(probs, num_trajectories)

            # Keep only selected trajectories
            trajectories = [trajectories[i] for i in selected_indices]
            rewards = [rewards[i] for i in selected_indices]
            
            # Double the search depth for remaining trajectories
            current_depth *= 2
            
            # Continue search from the end states of remaining trajectories
            new_trajectories = []
            new_rewards = []
            
            for traj, prev_reward in zip(trajectories, rewards):
                # Get the final states from the previous trajectory
                final_zt = torch.tensor(traj['zt'][-1]).to(zt.device)
                final_ht = torch.tensor(traj['ht'][-1]).to(zt.device)
                final_reward_hidden = traj['reward_hidden'][-1]
                
                # Search deeper from this point
                extended_trajectories, extended_rewards = self.traj_search.search(
                    final_zt, final_ht, final_reward_hidden, 
                    current_depth, num_trajectories
                )
                # num_nodes += 1
                # Combine previous trajectory with new extensions
                for ext_traj, ext_reward in zip(extended_trajectories, extended_rewards):
                    combined_traj = {
                        'zt': traj['zt'] + ext_traj['zt'][1:],  # Skip first state to avoid duplication
                        'ht': traj['ht'] + ext_traj['ht'][1:],
                        'reward_hidden': traj['reward_hidden'] + ext_traj['reward_hidden'][1:],
                        'at': traj['at'] + ext_traj['at']
                    }
                    combined_reward = prev_reward + ext_reward
                    
                    new_trajectories.append(combined_traj)
                    new_rewards.append(combined_reward)
            
            trajectories = new_trajectories
            rewards = new_rewards
        
        # Return the best trajectory and its reward
        # print("wtf", num_nodes)
        return trajectories[0], rewards[0]
    
def bt_Lambda_A_return(vt, rt, ct, gamma, lambda_):
    """
    Computes Bootstrapped Lambda A-Return

    Parameters:
    vt (list of float): List of vt values.
    rt (list of float): List of rt values.
    ct (list of float): List of ct values.
    gamma (float): Constant gamma.
    lambda_ (float): Constant lambda.

    Returns:
    float: The computed Rt value.
    """
    n = len(vt)
    if not (len(rt) == n and len(ct) == n):
        raise ValueError("All input lists must have the same length.")

    # Initialize R list
    R = [0.0] * n
    R[-1] = vt[-1]  # R_last = vt_last

    # Iterate from second last to first
    for t in range(n - 2, -1, -1):
        R_next = R[t + 1]
        R[t] = rt[t] + gamma * ct[t] * ((1 - lambda_) * vt[t] + lambda_ * R_next)

    return R[0]
