import torch
import torch.nn as nn

class BlockDiagonalGRU(nn.Module):
    def __init__(self, z_dim, a_dim, hidden_size, action_type='continuous', num_actions=None, action_embed_dim=16, 
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Args:
            z_dim (int): Dimension of the latent variable z_t
            a_dim (int): Dimension of the action a_t (for continuous) or number of discrete actions
            hidden_size (int): Total hidden size H (must be divisible by num_blocks)
            action_type (str): 'continuous' or 'discrete'
            num_actions (int, optional): Number of discrete actions (required if action_type is 'discrete')
            action_embed_dim (int, optional): Embedding dimension for discrete actions
            device (torch.device): Device to run the model on ('cuda' or 'cpu').
        """
        super(BlockDiagonalGRU, self).__init__()
        self.device = device

        self.z_dim = z_dim
        self.a_dim = a_dim
        self.hidden_size = hidden_size
        self.num_blocks = 8  # Fixed to 8 blocks as per requirement
        assert hidden_size % self.num_blocks == 0, "hidden_size must be divisible by num_blocks"
        self.block_size = hidden_size // self.num_blocks
        self.action_type = action_type

        # Handle action embedding based on action type
        if self.action_type == 'discrete':
            assert num_actions is not None, "num_actions must be specified for discrete actions"
            self.action_embedding = nn.Embedding(num_actions, action_embed_dim)
            embedded_a_dim = action_embed_dim
        elif self.action_type == 'continuous':
            embedded_a_dim = a_dim
        else:
            raise ValueError("action_type must be either 'continuous' or 'discrete'")

        # Linear embedding layer to mix z_t, a_t, and h_t
        self.embedding = nn.Linear(z_dim + embedded_a_dim + hidden_size, hidden_size)

        # Input-to-hidden weights for gates and candidate activation
        self.W_z = nn.Linear(hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)

        # Block-diagonal recurrent weights (hidden-to-hidden) without biases
        self.U_z = nn.ModuleList([
            nn.Linear(self.block_size, self.block_size, bias=False) for _ in range(self.num_blocks)
        ])
        self.U_r = nn.ModuleList([
            nn.Linear(self.block_size, self.block_size, bias=False) for _ in range(self.num_blocks)
        ])
        self.U_h = nn.ModuleList([
            nn.Linear(self.block_size, self.block_size, bias=False) for _ in range(self.num_blocks)
        ])
        self.to(device)

    def forward(self, z_t, a_t, h_t):
        z_t, a_t, h_t = z_t.to(self.device), a_t.to(self.device), h_t.to(self.device)
        """
        Forward pass for one time step.

        Args:
            z_t (Tensor): Sampled latent variable at time t, shape (batch_size, z_dim)
            a_t (Tensor): Action at time t
                - If continuous: shape (batch_size, a_dim)
                - If discrete: shape (batch_size,) containing action indices
            h_t (Tensor): Hidden state at time t, shape (batch_size, hidden_size)

        Returns:
            Tensor: Next hidden state h_{t+1}, shape (batch_size, hidden_size)
        """
        if self.action_type == 'discrete':
            a_t = self.action_embedding(a_t)  # Shape: (batch_size, action_embed_dim)

        # Concatenate z_t, a_t, and h_t to form the input
        # print(z_t.size(), a_t.size(), h_t.size())
        # print(a_t)
        x_t_input = torch.cat([z_t, a_t, h_t], dim=-1)  # Shape: (batch_size, z_dim + embedded_a_dim + hidden_size)

        # Compute the embedding
        x_t = self.embedding(x_t_input)  # Shape: (batch_size, hidden_size)

        # Compute input-to-hidden transformations
        Wz_x = self.W_z(x_t)
        Wr_x = self.W_r(x_t)
        Wh_x = self.W_h(x_t)

        # Initialize lists to store outputs for each block
        z_t_list = []
        r_t_list = []
        h_tilde_list = []

        # Process each block separately
        for b in range(self.num_blocks):
            start = b * self.block_size
            end = (b + 1) * self.block_size

            # Slice the hidden state for the current block
            h_t_block = h_t[:, start:end]

            # Slice the pre-activations for the current block
            Wz_x_block = Wz_x[:, start:end]
            Wr_x_block = Wr_x[:, start:end]
            Wh_x_block = Wh_x[:, start:end]

            # Compute hidden-to-hidden transformations
            Uz_h = self.U_z[b](h_t_block)
            Ur_h = self.U_r[b](h_t_block)

            # Update gate z_t
            z_t_block = torch.sigmoid(Wz_x_block + Uz_h)
            z_t_list.append(z_t_block)

            # Reset gate r_t
            r_t_block = torch.sigmoid(Wr_x_block + Ur_h)
            r_t_list.append(r_t_block)

            # Candidate activation h̃_t
            h_t_reset = r_t_block * h_t_block
            Uh_h_reset = self.U_h[b](h_t_reset)
            h_candidate = torch.tanh(Wh_x_block + Uh_h_reset)
            h_tilde_list.append(h_candidate)

        # Concatenate all blocks back together
        z_t_gate = torch.cat(z_t_list, dim=-1)
        r_t_gate = torch.cat(r_t_list, dim=-1)
        h_tilde = torch.cat(h_tilde_list, dim=-1)

        # Compute the next hidden state
        h_next = (1 - z_t_gate) * h_t + z_t_gate * h_tilde

        return h_next

# ----------------------------------------
# Usage Example with 8 Times the Recurrent Units
# ----------------------------------------

def usage_example_with_8x_recurrent_units():
    # Define base hidden size H
    H = 16  # Example base hidden size per block

    # Total hidden size is 8 times H
    hidden_size = 8 * H  # 128 in this example

    # Define other dimensions
    batch_size = 2
    z_dim = 16          # Dimension of latent variable z_t
    a_dim_continuous = 8  # Dimension of continuous action a_t
    num_actions = 10    # Number of discrete actions (for discrete case)
    action_embed_dim = 32  # Embedding dimension for discrete actions

    # Example for Continuous Actions
    print("=== Continuous Actions ===")
    # Create random tensors for z_t, a_t, and h_t
    z_t_cont = torch.randn(batch_size, z_dim)
    a_t_cont = torch.randn(batch_size, a_dim_continuous)  # Continuous actions
    h_t_cont = torch.randn(batch_size, hidden_size)

    # Initialize the BlockDiagonalGRU for continuous actions
    block_gru_cont = BlockDiagonalGRU(
        z_dim=z_dim,
        a_dim=a_dim_continuous,
        hidden_size=hidden_size,
        action_type='continuous'
    )

    # Forward pass to get h_{t+1}
    h_next_cont = block_gru_cont(z_t_cont, a_t_cont, h_t_cont)
    print(f"h_next_cont shape: {h_next_cont.shape}")  # Should output: torch.Size([32, 128])

    # Example for Discrete Actions
    print("\n=== Discrete Actions ===")
    # Create random tensors for z_t and h_t
    z_t_disc = torch.randn(batch_size, z_dim)
    h_t_disc = torch.randn(batch_size, hidden_size)

    # Create random action indices for discrete actions
    a_t_indices = torch.randint(0, num_actions, (batch_size,))  # Shape: (batch_size,)
    print(a_t_indices)

    # Initialize the BlockDiagonalGRU for discrete actions
    block_gru_disc = BlockDiagonalGRU(
        z_dim=z_dim,
        a_dim=num_actions,  # Here, a_dim represents the number of discrete actions
        hidden_size=hidden_size,
        action_type='discrete',
        num_actions=num_actions,
        action_embed_dim=action_embed_dim
    )

    # Forward pass to get h_{t+1}
    h_next_disc = block_gru_disc(z_t_disc, a_t_indices, h_t_disc)
    print(f"h_next_disc shape: {h_next_disc.shape}")  # Should output: torch.Size([32, 128])

# if __name__ == "__main__":
#     usage_example_with_8x_recurrent_units()
