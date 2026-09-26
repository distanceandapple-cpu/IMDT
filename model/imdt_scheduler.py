import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np
import math


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_schedule_jump(t_T, n_sample, jump_length, jump_n_sample,
                      jump2_length=1, jump2_n_sample=1,
                      jump3_length=1, jump3_n_sample=1,
                      start_resampling=100000000):
    """DDIM-style time-jump schedule for fast sampling with IOR harmonization.

    Generates a sequence of timestep pairs (t_last, t_cur) where:
    - t_cur < t_last: reverse step (denoise, go forward in diffusion time)
    - t_cur > t_last: forward jump  (re-noising, go backward in diffusion time)

    The key insight is that during forward jumps, the GT region is re-noised
    and re-injected, enabling iterative harmonization without freezing known values.
    """
    jumps = {}
    for j in range(0, t_T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    jumps2 = {}
    for j in range(0, t_T - jump2_length, jump2_length):
        jumps2[j] = jump2_n_sample - 1

    jumps3 = {}
    for j in range(0, t_T - jump3_length, jump3_length):
        jumps3[j] = jump3_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:
        t = t - 1
        ts.append(t)

        if (
            t + 1 < t_T - 1 and
            t <= start_resampling
        ):
            for _ in range(n_sample - 1):
                t = t + 1
                ts.append(t)

                if t >= 0:
                    t = t - 1
                    ts.append(t)

        if (
            jumps3.get(t, 0) > 0 and
            t <= start_resampling - jump3_length
        ):
            jumps3[t] = jumps3[t] - 1
            for _ in range(jump3_length):
                t = t + 1
                ts.append(t)

        if (
            jumps2.get(t, 0) > 0 and
            t <= start_resampling - jump2_length
        ):
            jumps2[t] = jumps2[t] - 1
            for _ in range(jump2_length):
                t = t + 1
                ts.append(t)
            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

        if (
            jumps.get(t, 0) > 0 and
            t <= start_resampling - jump_length
        ):
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)
            jumps2 = {}
            for j in range(0, t_T - jump2_length, jump2_length):
                jumps2[j] = jump2_n_sample - 1

            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

    ts.append(-1)

    def _check_times(times, t_0, t_T):
        assert times[0] > times[1], (times[0], times[1])
        assert times[-1] == -1, times[-1]
        for t_last, t_cur in zip(times[:-1], times[1:]):
            assert abs(t_last - t_cur) == 1, (t_last, t_cur)
        for t in times:
            assert t >= t_0, (t, t_0)
            assert t <= t_T, (t, t_T)

    _check_times(ts, -1, t_T)
    return ts


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """ beta schedule
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):
        # forward diffusion step
        self.num_timesteps = num_timesteps

        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == 'cosine':
            self.betas = torch.from_numpy(betas_for_alpha_bar(num_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,).astype(np.float32))

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # type: ignore
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterIOR
        self.posterIOR_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterIOR_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # IOR: cache for cumulative noise of ground truth
        # Stores (num_timesteps, *x_shape) tensor of the forward-diffused GT at each t
        self.gt_noises = None

    def q_sample(self, x_start, timesteps):
        """
        Forward diffusion: x_t = sqrt(bar_alpha[t]) * x_0 + sqrt(1 - bar_alpha[t]) * noise.

        This is used to re-noise ground truth during IOR harmonization.
        Each call draws FRESH random noise, so the result is stochastic.
        Use get_gt_noised() for the cached version that reuses training noise.
        """
        noise = torch.randn_like(x_start)
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]
        s1 = s1.reshape(-1, 1).to(x_start.device)
        s2 = s2.reshape(-1, 1).to(x_start.device)
        return s1 * x_start + s2 * noise, noise

    def store_gt_noises(self, x_start, noises):
        """
        Cache the forward-diffused noise trajectory of the ground truth.

        This stores the noise tensor ε₀ (shape = x_start.shape) used during training.
        Later, get_gt_noised() uses this ε₀ to reconstruct x_t at any timestep t
        via: x_t = sqrt(bar_alpha[t]) * x_0 + sqrt(1 - bar_alpha[t]) * ε₀

        This ensures sampling reuses the same noise path as training,
        providing deterministic harmonization.

        Args:
            x_start: ground truth tensor, shape (B, G)
            noises: the random noise tensor ε₀ used during forward diffusion, same shape as x_start
        """
        self.gt_noises = noises

    def get_gt_noised(self, gt, t):
        """
        Reconstruct the forward-diffused ground truth at timestep t.

        Uses the cached ε₀ from store_gt_noises() (or draws fresh noise if uncached).
        Returns: x_t = sqrt(bar_alpha[t]) * gt + sqrt(1 - bar_alpha[t]) * ε₀

        This is called during IOR harmonization to get the GT at noise level t,
        ensuring consistency with the training noise trajectory.

        Args:
            gt: ground truth tensor, shape (B, G)
            t: integer timestep (0 to num_timesteps-1)
        Returns:
            x_t: gt forward-diffused to timestep t, same shape as gt
        """
        device = gt.device
        t_tensor = torch.tensor([t], device=device)

        if self.gt_noises is not None:
            noise = self.gt_noises.to(device)
        else:
            noise = torch.randn_like(gt)

        sqrt_alphas_cumprod_t = _extract_into_tensor(
            self.alphas_cumprod.cpu().numpy(), t_tensor.cpu(), gt.shape).to(device)
        sqrt_one_minus_t = torch.sqrt(1 - sqrt_alphas_cumprod_t ** 2)

        return sqrt_alphas_cumprod_t * gt + sqrt_one_minus_t * noise

    def reset_gt_noises(self):
        """Reset the cached GT noise. Call between samples to avoid cross-contamination."""
        self.gt_noises = None

    def reconstruct_x0(self, x_t, t, noise):
        """ Get x0 from xt, noise.
        """        
        
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1).to(x_t.device)
        s2 = s2.reshape(-1, 1).to(x_t.device)
        
        x0 = s1 * x_t - s2 * noise
        return torch.clamp(x0,min=-1,max=1)

    def q_posterIOR(self, x_0, x_t, t):
        """x_t-1  mean  as part of Reparameteriation
        """        
        
        s1 = self.posterIOR_mean_coef1[t]
        s2 = self.posterIOR_mean_coef2[t]
        
        s1 = s1.reshape(-1, 1).to(x_t.device)
        s2 = s2.reshape(-1, 1).to(x_t.device)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        
        variance = variance.clip(1e-20)
        return variance.to(t.device)
    
    # reverse
    def step(self,
             model_output,
             timestep,
             sample,
             model_pred_type: str = 'noise',
             sample_pred_xstart: bool = False):
        """Reverse diffusion step: x_{t-1} from x_t via model prediction.

        Args:
            model_output: either predicted noise (model_pred_type='noise')
                         or predicted x0 (model_pred_type='x_start')
            timestep: scalar tensor of current timestep t
            sample: current noisy sample x_t
            model_pred_type: 'noise' or 'x_start'
            sample_pred_xstart: if True, sample x_{t-1} from N(mean, var).
                               If False (default), use mean only (deterministic).

        Returns:
            pred_prev_sample: x_{t-1} (or x_0 if t==0)
            pred_original_sample: predicted x0
        """
        t = timestep

        if model_pred_type == 'noise':
            pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        elif model_pred_type == 'x_start':
            pred_original_sample = model_output
        else:
            raise NotImplementedError()

        pred_prev_sample = self.q_posterIOR(pred_original_sample, sample, t)

        # Reparameterization trick: add stochastic noise at t > 0
        if sample_pred_xstart and t > 0:
            variance = self.get_variance(t)
            noise = torch.randn_like(model_output)
            pred_prev_sample = pred_prev_sample + (variance ** 0.5) * noise

        return pred_prev_sample, pred_original_sample

    def add_noise(self, x_start, x_noise, timesteps):  # forward
        # input x_0,noise,t , output x_t
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1).to(x_start.device)
        s2 = s2.reshape(-1, 1).to(x_start.device)
        return s1 * x_start + s2 * x_noise

    def undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
        """
        Forward diffusion step: re-noise img_after_model from timestep t to t+1.

        In IOR's time-jump schedule, when t_cur > t_last (forward jump),
        we need to go back one diffusion step: x_{t+1} = q(x_{t+1} | x_t, x_0=est_x_0).

        Args:
            image_before_step: x_t (the output of p_sample at t)
            img_after_model: x_{t-1} (result of denoising step, same as image_before_step here)
            est_x_0: estimated x_0 from the model
            t: scalar timestep (x_t is at timestep t)
        """
        return self._undo(img_after_model, t)

    def _undo(self, img_out, t):
        """
        Add one step of forward diffusion noise: x_{t+1} = q(x_{t+1} | x_t, x_0).

        Uses the model estimate est_x_0 embedded in img_out and the posterIOR mean.
        Equivalent to: img_{t+1} = posterIOR_mean(x_0=img_out, x_t=img_out, t=t+1)

        This is an approximation since we don't have the true x_0 during sampling.
        Matches IOR's convention: undo at t means go from x_t to x_{t+1}.
        """
        # Get posterIOR at t+1 (one step back in diffusion time)
        t_plus_1 = t + 1 if t + 1 < self.num_timesteps else t
        t_tensor = torch.tensor([t_plus_1], device=img_out.device, dtype=torch.long)

        # PosterIOR mean: mu = coef1 * x_0 + coef2 * x_t
        coef1 = self.posterIOR_mean_coef1[t_tensor].reshape(-1, 1).to(img_out.device)
        coef2 = self.posterIOR_mean_coef2[t_tensor].reshape(-1, 1).to(img_out.device)

        img_in_est = coef1 * img_out + coef2 * img_out

        # Add noise scaled by posterIOR variance
        variance = self.get_variance(t_tensor.to(img_out.device))
        img_in_est = img_in_est + (variance ** 0.5) * torch.randn_like(img_out)

        return img_in_est

    def __len__(self):
        return self.num_timesteps
