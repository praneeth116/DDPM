import torch

if torch.cuda.is_available():
    print("GPU")
else:
    print("CPU")
    
class LinearNoiseScheduler:
    def __init__(self, beta_start, beta_end, num_timesteps):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps

        self.betas = torch.linespace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.aplha_cum_prod = torch.cumprod(self.alphas, dim=0)
        
    def forward(self):
        pass