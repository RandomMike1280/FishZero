import torch
import torch.nn.functional as F

# Symmetric logarithm function
def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

# Inverse of symlog function
def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

# Encodes a target value using two nearest bins
# Ensures smooth representation within the bin range
def twohot(y, num_bins: int, bin_range: tuple = (-20.0, 20.0)):
    # Ensure y is a float tensor
    y = y.to(torch.float32)
    bin_min, bin_max = bin_range
    device = y.device  # Ensure operations are on the same device as y
    # Create bin edges
    B = symexp(torch.linspace(bin_min, bin_max, steps=num_bins, device=device))
    B = (B - B.min()) / (B.max() - B.min()) * (bin_max - bin_min) + bin_min
    # Initialize the probabilities tensor
    batch_size = y.size(0)
    probs = torch.zeros((batch_size, num_bins), dtype=torch.float32, device=device)

    # Boundary conditions
    below_min = y <= B[0]
    above_max = y >= B[-1]
    within_range = (~below_min) & (~above_max)
    # Assign 1.0 to the first bin where y <= min
    probs[below_min, 0] = 1.0

    # Assign 1.0 to the last bin where y >= max
    probs[above_max, -1] = 1.0
    if within_range.any():
        y_in = y[within_range]
        # Find indices of the right bins for each y_in
        # torch.searchsorted returns indices where elements should be inserted to maintain order
        # Subtract 1 to get the left bin index
        k = torch.searchsorted(B, y_in, right=False) - 1
        # Clamp indices to ensure they are within valid range
        k = torch.clamp(k, 0, num_bins - 2)
        # Gather the bin edges for the left and right bins
        B_left = B[k]
        B_right = B[k + 1]
        # Calculate the width of each bin
        bin_width = B_right - B_left
        # Avoid division by zero by setting bin_width to 1 where it's zero
        bin_width = torch.where(bin_width > 0, bin_width, torch.ones_like(bin_width))
        # Calculate weights for the left and right bins
        w1 = (B_right - y_in).abs() / bin_width
        w2 = 1.0 - w1

        # Scatter the weights into the probs tensor
        probs[within_range, k] = w1
        probs[within_range, k + 1] = w2

    return probs

# Mean squared error using symlog transform
def symlog_squared_error(x, y):
    return torch.mean((x - symlog(y)) ** 2)

# Loss function combining symexp with two-hot encoding
def symexp_twohot_loss(x, y, num_bins=10, bin_range=(-20., 20.)):
    target_twohot = twohot(y, num_bins, bin_range)
    
    # Apply log softmax to the predicted logits
    log_probs = F.log_softmax(x, dim=-1)
    
    # Compute the cross-entropy loss
    loss = -(target_twohot * log_probs).sum(dim=-1)
    return loss.mean()

# Free bits loss to prevent over-regularization
def free_bits_loss(x, y):
    return torch.clamp(F.kl_div(x.log(), y, reduction='batchmean'), min=1.0)

def negative_cosine(x, y, dim:int=1):
    return torch.mean(-F.cosine_similarity(x, y, dim=dim))

# test two hot

# batch_y = torch.tensor([-25.0, -10.0, 0.0, 15.0, 25.0])

# # # Parameters
# num_bins = 5
# bin_range = (-20.0, 20.0)

# # Get two-hot encodings
# two_hot_encodings = twohot(batch_y, num_bins, bin_range)

# print(two_hot_encodings)

# enc = torch.tensor([0.1000, 0.1000, 0.1000, 0.1000, 0.1000])

# print(symexp_twohot_loss())

# y = enc.double()
# bin_min, bin_max = bin_range
# bins = symexp(torch.linspace(bin_min, bin_max, steps=num_bins))

# print(torch.sum(y*bins))