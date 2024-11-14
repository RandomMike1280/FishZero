import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

class MLP(nn.Module):
    def __init__(self, input_dim: int, depth: int, output_dim: int = None, hidden_dim: int = None, device = 'cpu'):
        super(MLP, self).__init__()
        
        layers = []
        if depth > 1:
            # Initialize hidden_dim if not provided
            hidden_dim = input_dim * 2 if hidden_dim is None else hidden_dim
            layers += [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            
            # Create hidden layers with progressively doubling dimensions
            for _ in range(1, depth - 1):
                next_hidden_dim = hidden_dim * 2
                layers += [nn.Linear(hidden_dim, next_hidden_dim), nn.ReLU()]
                hidden_dim = next_hidden_dim
            
            # Define the output layer
            output_dim = hidden_dim * 2 if output_dim is None else output_dim
            layers.append(nn.Linear(hidden_dim, output_dim))
            layers.append(nn.ReLU())
        elif depth == 1:
            if output_dim is None:
                raise ValueError("output_dim must be specified when depth is 1.")
            layers += [nn.Linear(input_dim, output_dim)]
            layers.append(nn.ReLU())
        else:
            raise ValueError(f"Depth must be a positive integer greater than or equal to 1. Got {depth}")
        
        # Assign the layers to self.network
        self.network = nn.Sequential(*layers)
        self.network.to(torch.device(device))
    
    def forward(self, x):
        return self.network(x)

class ReverseMLP(nn.Module):
    def __init__(self, input_dim:int, depth:int, output_dim:int=None, hidden_dim:int=None):
        super(ReverseMLP, self).__init__()
        
        # First layer: from input to the first hidden layer
        layers = []
        if depth > 1:
            # Initialize hidden_dim if not provided
            hidden_dim = input_dim // 2 if hidden_dim is None else hidden_dim
            layers += [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            
            # Create hidden layers with progressively halving dimensions
            for _ in range(1, depth - 1):
                next_hidden_dim = hidden_dim // 2
                layers += [nn.Linear(hidden_dim, next_hidden_dim), nn.ReLU()]
                hidden_dim = next_hidden_dim
            
            # Define the output layer
            output_dim = hidden_dim // 2 if output_dim is None else output_dim
            layers.append(nn.Linear(hidden_dim, output_dim))
        elif depth == 1:
            if output_dim is None:
                raise ValueError("output_dim must be specified when depth is 1.")
            layers += [nn.Linear(input_dim, output_dim)]
            layers.append(nn.ReLU())
        else:
            raise ValueError(f"Depth must be a positive integer greater than or equal to 1. Got {depth}")
        
        # Store layers as a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self,x):
        return self.network(x)