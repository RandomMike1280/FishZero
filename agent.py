from world_model import *
from nets import *
from replaybuffer import *
from search import *
from losses import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

class Agent(nn.Module):
    def __init__(self, config:dict):
        super(Agent, self).__init__()
        # Device configuration
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Common parameters
        self.recurrent_size = config['ht_size']
        self.image_size = config['image_size']
        self.img_channels = config['img_channel']
        self.vector_dim = config['vector_dim']
        self.num_codes = config['num_codes']
        self.num_latents = config['num_latents']
        self.zt_size = self.num_codes * self.num_latents
        self.embedding_dim = config['embed_dim']
        self.bin_numbers = config['bin_num']
        self.bin_start = config['bin_start']
        self.bin_end = config['bin_end']
        self.imagination_horizon = config['imagine_horizon']

        # Action parameters
        self.action_size = config['action_size']
        self.action_type = config['action_type']
        if self.action_type == 'discrete':
           self.action_embedding_size = config['action_embed_size']

        self.encoder = Encoder(
            recurrent_size=self.recurrent_size,
            image_size=self.image_size,
            img_channels=self.img_channels,
            vector_dim=self.vector_dim,
            num_codes=self.num_codes,
            num_latents=self.num_latents,
            device=self.device
        ).to(self.device)

        # Initialize Decoder
        self.decoder = Decoder(
            image_size=self.image_size,
            img_channels=self.img_channels,
            vector_dim=self.vector_dim,
            num_codes=self.num_codes,
            num_latents=self.num_latents,
            recurrent_size=self.recurrent_size,
            embedding_dim=self.embedding_dim
        ).to(self.device)

        # Initialize DynaPred
        self.dyna_pred = DynaPred(
            zt_size=self.zt_size,
            recurrent_size=self.recurrent_size,
            num_codes=self.num_codes,
            num_latents=self.num_latents,
            bin_numbers=self.bin_numbers,
            bin_start=self.bin_start,
            bin_end=self.bin_end,
            Decoder=self.decoder,
            device=self.device
        ).to(self.device)

        # Initialize RSSM
        self.rssm = RSSM(
            zt_size=self.zt_size,
            ht_size=self.recurrent_size,
            action_size=self.action_size,
            action_embedding_size=self.action_embedding_size if self.action_type == 'discrete' else 0,
            num_latent=self.num_latents,
            num_codes=self.num_codes,
            device=self.device,
            DynaPred=self.dyna_pred,
            action_type=self.action_type
        ).to(self.device)

        # Initialize Actor
        self.actor = Actor(
            num_codes=self.num_codes,
            num_latent=self.num_latents,
            recurrent_size=self.recurrent_size,
            latent_dim=self.zt_size,
            action_size=self.action_size,
            unimix=0.01,
            device=self.device
        ).to(self.device)

        # Initialize Critic
        self.critic = Critic(
            num_codes=self.num_codes,
            num_latent=self.num_latents,
            recurrent_size=self.recurrent_size,
            latent_dim=self.zt_size,
            bin_min=self.bin_start,
            bin_max=self.bin_end,
            bin_numbers=self.bin_numbers,
            unimix=0.01,
            device=self.device
        ).to(self.device)

        self.projection = Projection(self.num_codes, self.num_latents, 64, self.device)
        self.prediction = Prediction(64, device=self.device)

        # Initialize ReplayBuffer
        self.replay_buffer = ReplayBuffer(
            main_capacity=10000,
            online_capacity=1000,
            batch_size=32,
            chunk_length=self.imagination_horizon,
            latent_dim=self.num_latents * self.num_codes,
            recurrent_dim=self.recurrent_size,
            bin_number=self.bin_numbers,
            obs_dim=self.img_channels * self.image_size[0] * self.image_size[1],
            action_dim=self.action_size,
            reward_dim=1,  # Assuming scalar rewards
            rt_probs_dim=self.bin_numbers
        )

        # Initialize TrajectorySearch
        self.trajectory_search = TrajectorySearch(
            RSSM=self.rssm,
            Encoder=self.encoder,
            Actor=self.actor,
            Critic=self.critic,
            replay_buffer=self.replay_buffer
        )

        self.sequential_halving = SequentialHalving(self.rssm, self.encoder, self.actor, self.critic, action_size=self.action_size)

    def imd_act(self, zt, ht) -> torch.tensor:
        return [torch.argmax(self.actor(zt, ht))]
    
    def evaluate(self, xt, ht, reward_hidden) -> tuple:
        zt = self.encoder(ht, xt)
        return self.critic(zt, ht, reward_hidden)
    
    def act(self, zt, ht, reward_hidden) -> tuple:
        with torch.no_grad():
            best_traj, best_reward = self.sequential_halving.search(
                zt=zt,
                ht=ht,
                reward_hidden=reward_hidden,
                init_depth=32,
                k=self.action_size
            )
        return best_traj['at'], best_reward
    
    def consistency(self, real_zt1, zt1):
        real_gt1 = self.projection(real_zt1.to(self.device))
        gt1 = self.projection(zt1.to(self.device))
        gt1 = self.prediction(gt1.to(self.device))
        loss = negative_cosine(gt1, real_gt1.detach(), dim=-1)
        return loss

