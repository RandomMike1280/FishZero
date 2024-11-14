import torch
import torch.nn as nn
import torch.nn.functional as F
from sub_nets import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) +1)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

class Actor(nn.Module):
    def __init__(self, num_codes:int, num_latent:int, recurrent_size:int, latent_dim:int, action_size:int, unimix:float=0.01,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Actor, self).__init__()
        self.projection = nn.Linear(num_codes * num_latent + recurrent_size, latent_dim)
        self.num_codes = num_codes
        self.num_latent = num_latent
        self.unimix = unimix
        self.action_size = action_size
        self.device = device

        self.mlp = MLP(input_dim=latent_dim, output_dim=action_size, depth=3, hidden_dim=32)
        self.to(device)

    def forward(self, zt, ht):
        st = torch.cat((zt, ht), dim=-1).to(self.device)
        st = self.projection(st)
        logits = self.mlp(st)
        learned_probs = F.softmax(logits.view(-1, self.action_size), dim=-1)

        uniform_probs = torch.full_like(learned_probs, 1.0 / self.num_codes)
        mixture_probs = (1.0 - self.unimix) * learned_probs + self.unimix * uniform_probs

        return mixture_probs
    
class Critic(nn.Module):
    def __init__(self, num_codes:int, num_latent:int, recurrent_size:int, latent_dim:int, bin_min:float, bin_max:float, bin_numbers:int, unimix:float=0.01,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Critic, self).__init__()
        self.projection = nn.Linear(num_codes * num_latent + recurrent_size, latent_dim)
        self.lstm = nn.LSTM(latent_dim, recurrent_size, batch_first=True)
        self.bn_value_prefix = nn.BatchNorm1d(recurrent_size, momentum=0.1)
        self.output_layer = MLP(input_dim=recurrent_size, depth=1, output_dim=bin_numbers)
        self.out_relu = nn.ReLU()
        self.out_bn = nn.BatchNorm1d(bin_numbers, momentum=0.1)
        self.symexp = symexp
        bin_edges = self.symexp(torch.linspace(bin_min, bin_max, bin_numbers))
        self.register_buffer('bin_centers', bin_edges)
        self.device = device
        self._initialize_weights_zero()
        self.to(device)

    def _initialize_weights_zero(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name or 'bias' in param_name:
                        nn.init.constant_(param, 0)
    
    def forward(self, z, h, reward_hidden):
        combined = torch.cat([z, h], dim=-1).to(self.device)
        g = self.projection(combined)
        # g = g.view(-1, 128).unsqueeze(0)
        lstm_out, reward_hidden = self.lstm(g,reward_hidden)
        lstm_out = lstm_out.squeeze(0)
        lstm_out = self.bn_value_prefix(lstm_out)
        lstm_out = nn.functional.relu(lstm_out)
        probs = self.output_layer(lstm_out)
        vt_probs = F.softmax(probs, dim=0)
        bin_centers = self.bin_centers.to(vt_probs.device)
        vt = torch.sum(vt_probs.double() * bin_centers.double(), dim=0)
        return vt, vt_probs, reward_hidden
    
class Projection(nn.Module):
    def __init__(self, num_codes, num_latent, projection_size, device='cpu'):
        self.device = device
        super(Projection, self).__init__()
        self.proj = MLP(input_dim=num_codes * num_latent, output_dim=projection_size, depth=3, hidden_dim=32, device=device)
        self.to(torch.device(device))
    
    def forward(self, x):
        return self.proj(x)
    
class Prediction(nn.Module):
    def __init__(self, projection_size, device='cpu'):
        super(Prediction, self).__init__()
        self.pred = MLP(input_dim=projection_size, output_dim=projection_size, depth=2)
        self.to(torch.device(device))

    def forward(self, x):
        return self.pred(x)
