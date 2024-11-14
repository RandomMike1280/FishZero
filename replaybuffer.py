import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, main_capacity, online_capacity, batch_size, chunk_length,
                 latent_dim, recurrent_dim, bin_number, obs_dim, action_dim, reward_dim, rt_probs_dim,
                 include_obs=False, include_time=False):
        """
        Initialize the ReplayBuffer with specified capacities and dimensions.

        Args:
            main_capacity (int): Capacity of the main replay buffer.
            online_capacity (int): Capacity of the online queue.
            batch_size (int): Size of the minibatch for training.
            chunk_length (int): Length of the sequential chunk from the online queue.
            latent_dim (int): Dimension of the latent state z_t.
            recurrent_dim (int): Dimension of the recurrent state h_t.
            obs_dim (int): Dimension of the observation x_t.
            action_dim (int): Dimension of the action a_t.
            reward_dim (int): Dimension of the reward r_t.
            rt_probs_dim (int): Dimension of the reward probabilities r_t_probs.
            include_obs (bool): Whether to include observation data (xt)
            include_time (bool): Whether to include timestep data
        """
        self.main_capacity = main_capacity
        self.online_capacity = online_capacity
        self.batch_size = batch_size
        self.chunk_length = chunk_length
        self.include_obs = include_obs
        self.include_time = include_time

        # Initialize the main buffer with required fields
        self.main_buffer = {
            'zt': torch.zeros((main_capacity, latent_dim)),
            'zt1': torch.zeros((main_capacity, latent_dim)),
            'ht': torch.zeros((main_capacity, recurrent_dim)),
            'vt': torch.zeros((main_capacity, bin_number)),
            'reward_hidden': (torch.zeros(main_capacity, recurrent_dim), 
                            torch.zeros(main_capacity, recurrent_dim)),
            'rt_probs': torch.zeros((main_capacity, rt_probs_dim)),
            'rt': torch.zeros((main_capacity)),
            'ct': torch.zeros((main_capacity, 1)),
            'terminality': torch.zeros((main_capacity)),
            'at': torch.zeros((main_capacity)),
        }

        # Add optional fields if specified
        if include_obs:
            self.main_buffer['xt'] = torch.zeros((main_capacity, 3, 64, 64))
        if include_time:
            self.main_buffer['time_step'] = torch.zeros((main_capacity, 1))

        # Initialize the online queue
        self.online_queue = {key: [] for key in self.main_buffer.keys()}

        # Pointers
        self.main_size = 0
        self.main_pos = 0
        self.online_size = 0

    def add_experience(self, zt, at, rt, ct, terminality, zt1, ht, vt_probs, reward_hidden, rt_probs, 
                      xt=None, time_step=None):
        """
        Add a new experience to the online queue.

        Args:
            zt (torch.Tensor): Latent state at time t.
            at (torch.Tensor): Action at time t.
            rt (torch.Tensor): Reward at time t.
            ct (torch.Tensor): Continuation flag at time t.
            zt1 (torch.Tensor): Latent state at time t+1.
            ht (torch.Tensor): Recurrent state at time t.
            vt (torch.Tensor): Predicted value at time t.
            reward_hidden (tuple): Reward hidden states.
            rt_probs (torch.Tensor): Reward probabilities at time t.
            xt (torch.Tensor, optional): Observation at time t.
            time_step (int, optional): Current training timestep.
        """
        # Validate optional parameters match initialization
        if (xt is not None) != self.include_obs:
            raise ValueError("Observation (xt) was not expected but provided" if xt is not None
                           else "Observation (xt) was expected but not provided")
        
        if (time_step is not None) != self.include_time:
            raise ValueError("Time step was not expected but provided" if time_step is not None
                           else "Time step was expected but not provided")

        # Create experience dictionary with required fields
        experience = {
            'zt': zt,
            'at': at,
            'rt': rt,
            'ct': ct,
            'terminality': terminality,
            'zt1': zt1,
            'ht': ht,
            'vt': vt_probs,
            'reward_hidden': reward_hidden,
            'rt_probs': rt_probs,
        }

        # Add optional fields if they were initialized
        if self.include_obs:
            experience['xt'] = xt
        if self.include_time:
            experience['time_step'] = time_step

        # Add to online queue
        for key, value in experience.items():
            self.online_queue[key].append(value)

        self.online_size += 1

        # Remove oldest if capacity exceeded
        if self.online_size > self.online_capacity:
            for key in self.online_queue:
                self.online_queue[key].pop(0)
            self.online_size -= 1

    def copy_online_to_main(self):
        """
        Copy experiences from the online queue to the main replay buffer.
        """
        num_experiences = len(self.online_queue['zt'])
        if num_experiences == 0:
            return

        for key in self.main_buffer:
            online_data = torch.stack(self.online_queue[key]) if key not in ['reward_hidden', 'time_step'] else self.online_queue[key]

            # Handle reward_hidden separately as it's a tuple
            if key == 'reward_hidden':
                tensor_1_list = [t[0] for t in online_data]
                tensor_2_list = [t[1] for t in online_data]
                online_data = (torch.stack(tensor_1_list), torch.stack(tensor_2_list))
                
                # Squeeze extra dimensions for both tensors
                online_data = (online_data[0].squeeze(1).squeeze(1), 
                            online_data[1].squeeze(1).squeeze(1))
                
                # Update first tensor
                end_pos = self.main_pos + num_experiences
                if end_pos <= self.main_capacity:
                    self.main_buffer[key][0][self.main_pos:end_pos] = online_data[0]
                else:
                    first_len = self.main_capacity - self.main_pos
                    second_len = end_pos % self.main_capacity
                    self.main_buffer[key][0][self.main_pos:] = online_data[0][:first_len]
                    self.main_buffer[key][0][:second_len] = online_data[0][first_len:]
                
                # Update second tensor
                if end_pos <= self.main_capacity:
                    self.main_buffer[key][1][self.main_pos:end_pos] = online_data[1]
                else:
                    self.main_buffer[key][1][self.main_pos:] = online_data[1][:first_len]
                    self.main_buffer[key][1][:second_len] = online_data[1][first_len:]
            else:
                if key != 'time_step':
                    end_pos = self.main_pos + num_experiences
                    if end_pos <= self.main_capacity:
                        # print(self.main_buffer[key][self.main_pos:end_pos].shape)
                        # print(online_data.shape)
                        # print(key)
                        self.main_buffer[key][self.main_pos:end_pos] = online_data
                    else:
                        first_len = self.main_capacity - self.main_pos
                        second_len = end_pos % self.main_capacity
                        self.main_buffer[key][self.main_pos:] = online_data[:first_len]
                        self.main_buffer[key][:second_len] = online_data[first_len:]

        # Update pointers
        self.main_size = min(self.main_size + num_experiences, self.main_capacity)
        self.main_pos = (self.main_pos + num_experiences) % self.main_capacity

        # Clear online queue
        self.online_queue = {key: [] for key in self.main_buffer.keys()}
        self.online_size = 0

    def sample_minibatch(self):
        """
        Sample a minibatch of experiences for training.

        Returns:
            batch (dict): A dictionary containing the sampled experiences.
            batch_indices (list): Indices of the experiences in the buffers.
            in_online_queue (list): Flags indicating if the experience is from the online queue.
        """
        batch = {key: [] for key in self.main_buffer.keys()}
        batch_indices = []

        # Sample from online queue
        online_chunk_size = min(self.chunk_length, len(self.online_queue['zt']))
        if online_chunk_size > 0:
            for key in batch:
                if key == 'reward_hidden':
                    tensor_1_list = [t[0] for t in self.online_queue[key][:online_chunk_size]]
                    tensor_2_list = [t[1] for t in self.online_queue[key][:online_chunk_size]]
                    batch[key].append((torch.stack(tensor_1_list), torch.stack(tensor_2_list)))
                else:
                    if key not in ['time_step']:
                        try:
                            online_data = torch.stack(self.online_queue[key][:online_chunk_size])
                        except TypeError:
                            # print(key)
                            import sys
                            sys.exit()
                        batch[key].append(online_data)
            batch_indices.extend(range(online_chunk_size))

        # Sample from main buffer
        remaining_size = self.batch_size - online_chunk_size
        in_online_queue = [True] * online_chunk_size

        if self.main_size > 0 and remaining_size > 0:
            # print(remaining_size)
            # print(self.main_size)
            main_indices = np.random.choice(self.main_size, min(self.main_size,remaining_size), replace=False)
            adjusted_indices = (main_indices + self.main_pos - self.main_size) % self.main_capacity
            batch_indices.extend(adjusted_indices.tolist())
            in_online_queue.extend([False] * remaining_size)

            for key in batch:
                if key == 'reward_hidden':
                    hidden_1 = self.main_buffer[key][0][adjusted_indices]
                    hidden_2 = self.main_buffer[key][1][adjusted_indices]
                    batch[key].append((hidden_1, hidden_2))
                else:
                    batch[key].append(self.main_buffer[key][adjusted_indices])

        # Concatenate batch data
        for key in batch:
            if len(batch[key]) > 0:
                if key == 'reward_hidden':
                    if len(batch[key]) == 1:
                        batch[key] = batch[key][0]
                    else:
                        h1 = torch.cat([t[0].squeeze() for t in batch[key]], dim=0)
                        h2 = torch.cat([t[1].squeeze() for t in batch[key]], dim=0)
                        batch[key] = (h1, h2)
                else:
                    # print(batch[key][0].size(), batch[key][-1].size())
                    # print(key)
                    batch[key] = torch.cat(batch[key], dim=0)
            else:
                batch[key] = torch.Tensor()

        return batch, batch_indices, in_online_queue

    def update_latents_in_buffer(self, batch_indices, new_zts, new_zt1s, in_online_queue):
        """
        Update latent states in the replay buffer after training the world model.

        Args:
            batch_indices (list): Indices of the experiences in the buffers.
            new_zts (torch.Tensor): Updated latent states z_t.
            new_zt1s (torch.Tensor): Updated latent states z_{t+1}.
            in_online_queue (list): Flags indicating if the experience is from the online queue.
        """
        for i, idx in enumerate(batch_indices):
            if in_online_queue[i]:
                self.online_queue['zt'][idx] = new_zts[i]
                self.online_queue['zt1'][idx] = new_zt1s[i]
            else:
                self.main_buffer['zt'][idx] = new_zts[i]
                self.main_buffer['zt1'][idx] = new_zt1s[i]

    def time_step_sampling(self):
        """
        Sample a single random latent state from the main buffer along with its timestep.

        Returns:
            latent_state (torch.Tensor): The sampled latent state z_t.
            time_step (torch.Tensor): The timestep when the latent state was added.
        """
        if not self.include_time:
            raise ValueError("Time step sampling is not available when time_step data is not included")
        
        if self.main_size == 0:
            raise ValueError("Main buffer is empty. Cannot perform timestep sampling.")

        idx = np.random.choice(self.main_size)
        adjusted_idx = (idx + self.main_pos - self.main_size) % self.main_capacity

        latent_state = self.main_buffer['zt'][adjusted_idx]
        time_step = self.main_buffer['time_step'][adjusted_idx]

        return latent_state, time_step

# Initialize the replay buffer with appropriate dimensions
# buffer = ReplayBuffer(
#     main_capacity=10000,
#     online_capacity=1000,
#     batch_size=64,
#     chunk_length=16,
#     latent_dim=128,
#     recurrent_dim=256,
#     obs_dim=84*84*3,
#     action_dim=4,
#     reward_dim=1,
#     rt_probs_dim=10
# )

# ... During training ...

# Add experiences to the buffer
# buffer.add_experience(zt, at, rt, ct, zt1, ht, vt, rt_probs, xt)

# Periodically copy from online queue to main buffer
# if should_copy_to_main_buffer:
#     buffer.copy_online_to_main()

# # Sample minibatch for training
# batch, batch_indices, in_online_queue = buffer.sample_minibatch()

# # Update latents in buffer after training the world model
# buffer.update_latents_in_buffer(batch_indices, new_zts, new_zt1s, in_online_queue)

# # Perform age sampling
# latent_state, age = buffer.time_step_sampling()
