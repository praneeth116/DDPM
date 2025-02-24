import torch

X = torch.linspace(0,1,4)
Y = torch.cumprod(X, dim = 0)

print(Y[2].reshape(4))

class LinearNoiseScheduler:
    def __init__(self, beta_start, beta_end, num_timesteps):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.aplha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.aplha_cum_prod)
        self.one_minus_sqrt_alpha_cum_prod = torch.sqrt(1-self.aplha_cum_prod)
        
    def add_noise(self, original, noise, t): ## This is the forward process
        original_shape = original.shape # B,C,H,W
        batch_size = original_shape[0]

        sqrt_alpha_cum_prod = self.aplha_cum_prod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.one_minus_sqrt_alpha_cum_prod[t].reshape(batch_size)

        for _ in range(len(original_shape)-1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise