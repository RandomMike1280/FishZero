import torch

def log_normallize(distribution:torch.Tensor):
    exp_dist = torch.exp(distribution)
    sum_dist = torch.sum(exp_dist)
    log_distribution = exp_dist / sum_dist
    return log_distribution

def gumbel_max_trick(probs):
    log_probs = log_normallize(probs)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(log_probs)))
    pertubed_log_probs = log_probs + gumbel_noise
    return torch.argmax(pertubed_log_probs)

def gumbel_top_k_trick(probs, k):
    top_k = [None in range(k)]
    trunc_probs = probs
    for iteration in range(k):
        i = gumbel_max_trick(trunc_probs)
        top_k[iteration] = i
        trunc_probs = torch.cat([trunc_probs[0:i], trunc_probs[i+1:-1]])
        
if __name__ == '__main__':
    t = torch.randn((1, 5))
    print(t)
    print(gumbel_top_k_trick(t, 3))