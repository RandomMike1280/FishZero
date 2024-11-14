import torch
import torch.nn as nn
import torch.nn.functional as F
from diagonal_gru import BlockDiagonalGRU
from sub_nets import *

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) +1)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

class RSSM(nn.Module):
    def __init__(self,
                zt_size:int,
                ht_size:int,
                action_size:int,
                action_embedding_size:int,
                num_latent:int,
                num_codes:int,
                DynaPred:nn.Module,
                action_type:str='discrete',
                device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ):
        super(RSSM, self).__init__()
        self.device = device
        
        # Initialize GRU cell based on the action type (discrete or continuous)
        if action_type == 'discrete':
            self.gru_cell = BlockDiagonalGRU(
                z_dim=zt_size,
                a_dim=action_size,  # Number of discrete actions
                hidden_size=ht_size,  # Size of the hidden state
                device=self.device,
                action_type='discrete',
                num_actions=action_size,  # Total number of discrete actions
                action_embed_dim=action_embedding_size  # Size of the action embedding
            )
        elif action_type == 'continuous':
            self.gru_cell = BlockDiagonalGRU(
                z_dim=zt_size,  # Size of the latent state
                a_dim=action_size,  # Dimension of the continuous action space
                hidden_size=ht_size,  # Size of the hidden state,
                device=self.device,
                action_type='continuous'
            )
        else:
            raise ValueError(f"Invalid action type: {action_type}. Expected 'discrete' or 'continuous'.")
        
        # Dynamics prediction module, used to predict future latent states and auxiliary variables
        self.dyna_pred = DynaPred
        
        # Encoder module, used to encode the input observation into a latent representation
        self.encoder = Encoder

        # Linear projection layer to transform latent variables to the desired size
        self.projection = nn.Linear(num_codes * num_latent, zt_size)

    def forward(self, ht, zt, at):
        # xt: observation image of size [batch_size, channels, x, y]
        # vector: additional environment information [batch, dim] (e.g., health, exp)
        # ht: recurrent state [batch, dim]
        # zt: latent state [batch, num_codes * num_latents]

        # Project the encoded latent state to match the required latent size
        zt = self.projection(zt.to(self.device))
        
        # Update the recurrent state using the GRU cell
        ht_next = self.gru_cell(zt, at, ht)
        # Predict the next latent state, along with context, reward, reconstructed image, and vector
        zt_next, ct, rt, rt_probs, ct_probs, recon_img, recon_vect = self.dyna_pred(zt, ht, ht_next)

        return zt_next, ht_next, ct, rt, rt_probs, recon_img, recon_vect

class Encoder(nn.Module):
    def __init__(
        self,
        recurrent_size: int,
        image_size: tuple = None,
        img_channels: int = None,
        vector_dim: int = None,
        num_codes: int = 256,
        num_latents: int = 10,
        unimix:float = 0.01,
        temperature:float = 1,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__()
        self.device = device
        self.is_image = image_size is not None and img_channels is not None
        self.is_vect = vector_dim is not None
        self.num_codes = num_codes
        self.num_latents = num_latents
        self.unimix = unimix
        self.temperature = temperature

        if self.is_image:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=img_channels, out_channels=16, kernel_size=3, stride=2, padding=1),  # e.g., 64x64 -> 32x32
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
                nn.ReLU()
            )

        if self.is_vect:
            self.mlps = MLP(input_dim=vector_dim, output_dim=256, depth=3, hidden_dim=64)

        # Projection layer outputs logits for the categorical distribution
        projection_input_dim = (
            recurrent_size
            + (128 * (image_size[0] // 16) * (image_size[1] // 16) if self.is_image else 0)
            + (256 if self.is_vect else 0)
        )
        self.projection = nn.Linear(projection_input_dim, num_latents * num_codes)
        self.symlog = symlog


    def forward(self, recurrent, img=None, vec=None, probs:bool=False):
        batch_size = recurrent.size(0)

        if self.is_image and img is not None:
            img_lat = self.conv_layers(img)
            image_lat = img_lat.view(batch_size, -1)
        else:
            image_lat = None

        if self.is_vect and vec is not None:
            vec = self.symlog(vec)
            vect_lat = self.mlps(vec)
        else:
            vect_lat = None

        if image_lat is not None and vect_lat is not None:
            combined_latent = torch.cat((image_lat, vect_lat), dim=1)
        elif vect_lat is not None:
            combined_latent = vect_lat
        elif image_lat is not None:
            combined_latent = image_lat
        else:
            raise ValueError("Either 'img' or 'vec' must be provided.")

        # Concatenate with recurrent input
        combined_latent = torch.cat((combined_latent, recurrent), dim=-1)

        # Get logits for the categorical distribution
        logits = self.projection(combined_latent)

        # Apply softmax to get the learned probability distribution
        learned_probs = F.softmax(logits.view(-1, self.num_codes), dim=-1)  # Shape: (batch_size * num_latents, num_codes)

        if probs:
            return learned_probs

        uniform_probs = torch.full_like(learned_probs, 1.0 / self.num_codes)
        mixture_probs = (1.0 - self.unimix) * learned_probs + self.unimix * uniform_probs # Assuming self.unimix = 0.01

        # Straight-through Gumbel-Softmax
        gumbels = -torch.empty_like(mixture_probs).exponential_().log()  # ~Gumbel(0,1)
        logits = (mixture_probs.log() + gumbels) / self.temperature  # temperature is a hyperparameter
        soft_one_hot = F.softmax(logits, dim=-1) # Differentiable (used for backprop)

        # Get hard samples for forward pass
        indices = soft_one_hot.max(dim=-1, keepdim=True)[1] # Non-differentiable
        hard_one_hot = torch.zeros_like(soft_one_hot).scatter_(-1, indices, 1.0) 
        zt = hard_one_hot - soft_one_hot.detach() + soft_one_hot # Straight-through
        zt = zt.view(batch_size, self.num_latents, -1) # Shape zt: (batch_size, num_latents, num_codes)
        zt = zt.view(batch_size, -1)

        return zt

class Decoder(nn.Module):
    def __init__(
        self, 
        image_size: tuple = None, 
        img_channels: int = None, 
        vector_dim: int = None,
        num_codes: int = 256,
        num_latents: int = 10,
        recurrent_size: int = 256,
        embedding_dim: int = 128
    ):
        super(Decoder, self).__init__()
        self.is_image = image_size is not None and img_channels is not None
        self.is_vect = vector_dim is not None

        self.num_codes = num_codes
        self.num_latents = num_latents
        self.embedding_dim = embedding_dim

        if self.is_image:
            # After aggregation, you need to map to (128, 4, 4)
            self.image_linear = nn.Linear(num_codes * num_latents + recurrent_size, 128 * 4 * 4)

            self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),   # 8x8 -> 16x16
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),   # 16x16 -> 32x32
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=16, out_channels=img_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32 -> 64x64
                nn.Sigmoid()  # Changed from ReLU to Sigmoid for image output
            )

        if self.is_vect:
            # After aggregation, map to the vector dimension
            self.vector_linear = nn.Linear(num_latents * num_codes + recurrent_size, 256)

            self.mlps_reverse = ReverseMLP(input_dim=256, output_dim=vector_dim, depth=3, hidden_dim=128)
        self.symexp = symexp  # Ensure symlog is defined elsewhere

    def forward(self, zt, ht):
        """
        Args:
            zt: Tensor of shape (batch_size, num_latents), containing integer codes.
        Returns:
            img_out: Reconstructed image tensor (if is_image).
            vec_out: Reconstructed vector tensor (if is_vect).
        """
        img_out = None
        vec_out = None

        if self.is_image:
            # Flatten and pass through linear layer
            img_embedded = zt.view(zt.size(0), -1)
            img_embedded = torch.cat((img_embedded, ht), dim=-1)
            latent_img = self.image_linear(img_embedded)
            latent_img = latent_img.view(-1, 128, 4, 4)
            # Pass through deconvolution layers
            img_out = self.deconv_layers(latent_img)

        if self.is_vect:
            # Flatten and pass through linear layer
            vec_embedded = zt.view(zt.size(0), -1)
            vec_embedded = torch.cat((vec_embedded, ht), dim=-1)
            latent_vec = self.vector_linear(vec_embedded)
            # Pass through MLP to reconstruct vector
            vec_out = self.mlps_reverse(latent_vec)
            vec_out = self.symexp(vec_out)

        return img_out, vec_out
    
class DynaPred(nn.Module):
    def __init__(self, 
        zt_size:int, 
        recurrent_size:int, 
        num_codes:int, 
        num_latents:int, 
        bin_numbers:int, 
        bin_start:float, 
        bin_end:float, 
        unimix:float=0.01, 
        temperature:float=1,
        Decoder:nn.Module=Decoder(),
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super(DynaPred, self).__init__()
        self.device = device
        self.num_codes = num_codes
        self.bin_numbers = bin_numbers
        self.num_latents = num_latents
        self.unimix = unimix
        self.temperature = temperature

        self.DynamicsPredictor = MLP(input_dim=recurrent_size, output_dim=self.num_codes*self.num_latents, depth=3, hidden_dim=32)

        self.ContinuePredictor = MLP(input_dim=zt_size+recurrent_size, output_dim=1, depth=2, hidden_dim=32)
        
        self.RewardPredictor = MLP(input_dim=zt_size+recurrent_size, output_dim=bin_numbers, depth=2, hidden_dim=64)
        for m in self.RewardPredictor.network:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.decoder = Decoder
        self.symexp = symexp
        self.symlog = symlog

        self.bin_start = bin_start
        self.bin_end = bin_end
        self.bin_start = bin_start
        self.bin_end = bin_end
        bin_edges = self.symexp(torch.linspace(bin_start, bin_end, bin_numbers))
        self.register_buffer('bin_centers', bin_edges)

    def forward(self, z, h, h1, probs:bool=False):
        batch_size = h.size(0)
        z = z.to(self.device)  # Ensure z is on the same device as h
        h = h.to(self.device)  # Ensure h is on the correct device
        s = torch.cat((z, h), dim=-1)
        zt_probs = self.DynamicsPredictor(h1)
        ct_probs = self.ContinuePredictor(s)
        rt_probs = self.RewardPredictor(s)
        
        # The same as encoder
        learned_probs = F.softmax(zt_probs.view(-1, self.num_codes), dim=-1)  # Shape: (batch_size * num_latents, num_codes)

        uniform_probs = torch.full_like(learned_probs, 1.0 / self.num_codes)
        mixture_probs = (1.0 - self.unimix) * learned_probs + self.unimix * uniform_probs
        gumbels = -torch.empty_like(mixture_probs).exponential_().log()  # ~Gumbel(0,1)
        logits = (mixture_probs.log() + gumbels) / self.temperature  # temperature is a hyperparameter
        soft_one_hot = F.softmax(logits, dim=-1) # Differentiable (used for backprop)
        indices = soft_one_hot.max(dim=-1, keepdim=True)[1] # Non-differentiable
        hard_one_hot = torch.zeros_like(soft_one_hot).scatter_(-1, indices, 1.0) 
        zt = hard_one_hot - soft_one_hot.detach() + soft_one_hot # Straight-through
        zt = zt.view(batch_size, self.num_latents, -1) # Shape zt: (batch_size, num_latents, num_codes)
        zt = zt.view(batch_size, -1)

        # ct_probs = F.softmax(ct_probs, dim=1)
        u = torch.rand_like(ct_probs)
        continue_rollout = (u <= ct_probs)
        continue_rollout = continue_rollout.int()

        rt_probs = F.softmax(rt_probs, dim=1)

        bin_centers = self.bin_centers.to(self.device)
        r_hat = torch.sum(rt_probs.double() * bin_centers.double(), dim=1) 
        recon_img, recon_vect = self.decoder(z, h)
        if probs:
            zt = learned_probs
        return zt, continue_rollout, r_hat, rt_probs, ct_probs, recon_img, recon_vect
    
# Testing the model's outputs
import unittest

class TestModels(unittest.TestCase):
    def setUp(self):
        # Common parameters
        self.batch_size = 4
        self.recurrent_size = 256
        self.image_size = (64, 64)
        self.img_channels = 3
        self.vector_dim = 10
        self.num_codes = 256
        self.num_latents = 10
        self.zt_size = 128  # Assuming zt_size matches num_latents
        self.embedding_dim = 128
        self.bin_numbers = 10
        self.bin_start = -20.
        self.bin_end = 20.
        
        # Initialize Encoder
        self.encoder = Encoder(
            recurrent_size=self.recurrent_size,
            image_size=self.image_size,
            img_channels=self.img_channels,
            vector_dim=self.vector_dim,
            num_codes=self.num_codes,
            num_latents=self.num_latents
        ).eval()

        # Initialize Decoder
        self.decoder = Decoder(
            image_size=self.image_size,
            img_channels=self.img_channels,
            vector_dim=self.vector_dim,
            num_codes=self.num_codes,
            num_latents=self.num_latents,
            embedding_dim=self.embedding_dim
        ).eval()

        # Initialize DynaPred
        self.dyna_pred = DynaPred(
            zt_size=self.zt_size,
            recurrent_size=self.recurrent_size,
            num_codes=self.num_codes,
            num_latents=self.num_latents,
            bin_numbers=self.bin_numbers,
            bin_start=self.bin_start,
            bin_end=self.bin_end,
            Decoder=self.decoder
        ).eval()

    def test_encoder(self):
        # Create dummy inputs
        recurrent = torch.randn(self.batch_size, self.recurrent_size)
        img = torch.randn(self.batch_size, self.img_channels, *self.image_size)
        vec = torch.randn(self.batch_size, self.vector_dim)

        with torch.no_grad():
            zt = self.encoder(recurrent, img=img, vec=vec)

        print("Output zt", zt.size())

        # Assertions
        self.assertEqual(zt.shape, (self.batch_size, self.num_latents * self.num_codes),
                         f"zt shape mismatch. Expected {(self.batch_size, self.num_latents * self.num_codes)}, got {zt.shape}")

        # Check that zt contains valid code indices
        self.assertTrue(torch.all(zt >= 0) and torch.all(zt < self.num_codes),
                        "zt contains invalid code indices.")

        print("Encoder Test Passed.")

    def test_decoder(self):
        # Create dummy zt
        # zt = torch.randint(0, self.num_codes, (self.batch_size, self.num_latents * self.num_codes))
        recurrent = torch.randn(self.batch_size, self.recurrent_size)
        img = torch.randn(self.batch_size, self.img_channels, *self.image_size)
        vec = torch.randn(self.batch_size, self.vector_dim)

        with torch.no_grad():
            zt = self.encoder(recurrent, img=img, vec=vec)

        with torch.no_grad():
            img_out, vec_out = self.decoder(zt)

        # Assertions
        if self.decoder.is_image:
            expected_img_shape = (self.batch_size, self.img_channels, *self.image_size)
            self.assertEqual(img_out.shape, expected_img_shape,
                             f"img_out shape mismatch. Expected {expected_img_shape}, got {img_out.shape}")
        else:
            self.assertIsNone(img_out, "img_out should be None when image decoding is disabled.")

        if self.decoder.is_vect:
            expected_vec_shape = (self.batch_size, self.vector_dim)
            self.assertEqual(vec_out.shape, expected_vec_shape,
                             f"vec_out shape mismatch. Expected {expected_vec_shape}, got {vec_out.shape}")
        else:
            self.assertIsNone(vec_out, "vec_out should be None when vector decoding is disabled.")

        print("Decoder Test Passed.")

    def test_dyna_pred(self):
        # Create dummy inputs
        z = torch.randint(0, self.num_codes, (self.batch_size, self.zt_size))
        h = torch.randn(self.batch_size, self.recurrent_size)

        with torch.no_grad():
            zt, continue_rollout, r_hat, r_probs, reconstruction_img, reconstruction_vect = self.dyna_pred(z, h)

        # Assertions
        print("Output r_hat", r_hat)
        # print("Output r_probs", r_probs)
        self.assertEqual(zt.shape, (self.batch_size, self.num_latents * self.num_codes),
                         f"zt shape mismatch. Expected {(self.batch_size, self.num_latents * self.num_codes)}, got {zt.shape}")
        self.assertEqual(continue_rollout.shape, (self.batch_size,1),
                         f"continue_rollout shape mismatch. Expected {(self.batch_size,)}, got {continue_rollout.shape}")
        self.assertEqual(r_hat.shape, (self.batch_size,),
                         f"r_hat shape mismatch. Expected {(self.batch_size,)}, got {r_hat.shape}")
        self.assertEqual(reconstruction_img.shape, (self.batch_size, self.img_channels, *self.image_size),
                         f"r_hat shape mismatch. Expected {(self.batch_size, self.img_channels, *self.image_size)}, got {reconstruction_img.shape}")
        self.assertEqual(reconstruction_vect.shape, (self.batch_size, self.vector_dim),
                         f"r_hat shape mismatch. Expected {(self.batch_size, self.vector_dim)}, got {reconstruction_vect.shape}")

        # Check that zt contains valid code indices
        self.assertTrue(torch.all(zt >= 0) and torch.all(zt < self.num_codes),
                        "zt contains invalid code indices.")

        # Check that continue_rollout is binary
        unique_values = torch.unique(continue_rollout)
        self.assertTrue(torch.all((unique_values == 0) | (unique_values == 1)),
                        "continue_rollout should be binary (0 or 1).")

        print("DynaPred Test Passed.")

    def test_rssm(self):
        """
        Test the RSSM model by initializing it with both discrete and continuous action types,
        passing dummy inputs, and verifying the outputs.
        """
        action_types = ['discrete', 'continuous']
        for action_type in action_types:
            with self.subTest(action_type=action_type):
                # Define action-related parameters
                if action_type == 'discrete':
                    action_size = 5  # Example number of discrete actions
                    action_embedding_size = 16
                else:
                    action_size = 3  # Example dimension for continuous actions
                    action_embedding_size = None  # Not used for continuous actions

                # Initialize RSSM
                rssm = RSSM(
                    zt_size=self.zt_size,
                    ht_size=self.recurrent_size,
                    action_size=action_size,
                    action_embedding_size=action_embedding_size if action_type == 'discrete' else 0,
                    num_latent=self.num_latents,
                    num_codes=self.num_codes,
                    DynaPred=self.dyna_pred,
                    action_type=action_type
                ).eval()

                # Create dummy inputs
                ht = torch.randn(self.batch_size, self.recurrent_size)
                xt = torch.randn(self.batch_size, self.img_channels, *self.image_size)
                vector = torch.randn(self.batch_size, self.vector_dim)
                with torch.no_grad():
                    zt = self.encoder(ht, img=xt, vec=vector)

                if action_type == 'discrete':
                    # For discrete actions, actions are represented as integers
                    at = torch.randint(0, action_size, (self.batch_size,))
                else:
                    # For continuous actions, actions are represented as floats
                    at = torch.randn(self.batch_size, action_size)

                with torch.no_grad():
                    outputs = rssm(ht, zt, at)

                # Unpack outputs
                zt_next, ht_next, ct, rt, rt_probs, recon_img, recon_vect = outputs

                # Define expected shapes
                expected_zt_next_shape = (self.batch_size, self.num_latents * self.num_codes)
                expected_ht_next_shape = (self.batch_size, self.recurrent_size)
                expected_ct_shape = (self.batch_size, 1)
                expected_rt_shape = (self.batch_size,)
                expected_recon_img_shape = (self.batch_size, self.img_channels, *self.image_size) if self.decoder.is_image else None
                expected_recon_vect_shape = (self.batch_size, self.vector_dim) if self.decoder.is_vect else None

                # Assertions for zt_next
                self.assertEqual(zt_next.shape, expected_zt_next_shape,
                                 f"zt_next shape mismatch for action_type '{action_type}'. "
                                 f"Expected {expected_zt_next_shape}, got {zt_next.shape}")
                self.assertTrue(torch.all(zt_next >= 0) and torch.all(zt_next < self.num_codes),
                                "zt_next contains invalid code indices.")

                # Assertions for ht_next
                self.assertEqual(ht_next.shape, expected_ht_next_shape,
                                 f"ht_next shape mismatch for action_type '{action_type}'. "
                                 f"Expected {expected_ht_next_shape}, got {ht_next.shape}")

                # Assertions for ct
                self.assertEqual(ct.shape, expected_ct_shape,
                                 f"ct shape mismatch for action_type '{action_type}'. "
                                 f"Expected {expected_ct_shape}, got {ct.shape}")
                self.assertTrue(torch.all((ct >= 0) & (ct <= 1)),
                                "ct contains values outside [0, 1].")

                # Assertions for rt
                self.assertEqual(rt.shape, expected_rt_shape,
                                 f"rt shape mismatch for action_type '{action_type}'. "
                                 f"Expected {expected_rt_shape}, got {rt.shape}")
                # Assuming rewards are in a specific range, adjust as necessary
                self.assertTrue(torch.all(rt >= self.bin_start) and torch.all(rt <= self.bin_end),
                                "rt contains values outside the expected range.")

                # Assertions for recon_img
                if self.decoder.is_image:
                    self.assertEqual(recon_img.shape, expected_recon_img_shape,
                                     f"recon_img shape mismatch for action_type '{action_type}'. "
                                     f"Expected {expected_recon_img_shape}, got {recon_img.shape}")
                    self.assertTrue(torch.all((recon_img >= 0) & (recon_img <= 1)),
                                    "recon_img contains values outside [0, 1].")
                else:
                    self.assertIsNone(recon_img, "recon_img should be None when image decoding is disabled.")

                # Assertions for recon_vect
                if self.decoder.is_vect:
                    self.assertEqual(recon_vect.shape, expected_recon_vect_shape,
                                     f"recon_vect shape mismatch for action_type '{action_type}'. "
                                     f"Expected {expected_recon_vect_shape}, got {recon_vect.shape}")
                else:
                    self.assertIsNone(recon_vect, "recon_vect should be None when vector decoding is disabled.")

                print(f"RSSM Test Passed for action_type '{action_type}'.")

# if __name__ == '__main__':
#     unittest.main(argv=[''], exit=False)
