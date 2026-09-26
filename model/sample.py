import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from .imdt_scheduler import get_schedule_jump


# ─────────────────────────────────────────────────────────────────────────────
# Helper: model inference over full sample tensor (chunked evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def model_sample_IMDt(model, device, dataloader, total_sample, time,
                        is_condi, condi_flag):
    """
    Evaluate the model in chunks over the full sample tensor.

    The model needs cell-type conditioning from dataloader, but we need to
    evaluate on the current x_t (which is not cell-type information).
    This function splits total_sample into batches matching the dataloader
    and runs model(x_t_chunk, t, conditioning_chunk).

    Args:
        model: DiT_IMDt model
        device: torch device
        dataloader: yields (data, celltype) tuples; celltype is used as conditioning
        total_sample: full x_t tensor, shape (num_cells, num_genes)
        time: scalar timestep
        is_condi: whether to use conditioning
        condi_flag: True for conditional forward, False for unconditional (CFG)

    Returns:
        model output tensor, same shape as total_sample
    """
    noise = []
    i = 0
    for _, x_cond in dataloader:
        x_cond = x_cond.float().to(device)
        t = torch.from_numpy(np.repeat(time, x_cond.shape[0])).long().to(device)
        if not is_condi:
            n = model(total_sample[i:i + len(x_cond)], t, None)
        else:
            n = model(total_sample[i:i + len(x_cond)], t, x_cond, condi_flag=condi_flag)
        noise.append(n)
        i = i + len(x_cond)
    noise = torch.cat(noise, dim=0)
    return noise


# ─────────────────────────────────────────────────────────────────────────────
# IOR injection: the core of harmonization
# ─────────────────────────────────────────────────────────────────────────────

def IOR_inject(x_t, gt, gt_keep_mask, noise_scheduler, t,
                   use_cumnoise=True):
    """
    IOR GT re-noising and injection.

    At each reverse diffusion step (before calling the model), the known
    region (mask=1) is replaced with the GT forward-diffused to the current
    noise level t. This allows known and unknown regions to co-evolve.

    Two strategies:
      - use_cumnoise=True:  use the cached noise path from training (deterministic,
                             matches the exact noise trajectory used during training)
      - use_cumnoise=False: draw fresh noise each time (stochastic)

    Args:
        x_t: current sample at timestep t, shape (B, G)
        gt: ground truth tensor, same shape as x_t
        gt_keep_mask: binary mask, 1=known region to keep, 0=unknown/impute
        noise_scheduler: NoiseScheduler instance
        t: scalar timestep
        use_cumnoise: use cached GT noise path (True) or fresh noise (False)

    Returns:
        x_t with known regions replaced by re-noised GT
    """
    gt_keep_mask = gt_keep_mask.to(x_t.device)
    gt = gt.to(x_t.device)

    if use_cumnoise:
        # Deterministic: reuse the exact noise path from training
        weighed_gt = noise_scheduler.get_gt_noised(gt, t)
    else:
        # Stochastic: draw fresh noise each step
        t_tensor = torch.tensor([t], device=x_t.device)
        alphas_cumprod_t = _extract_into_tensor(
            noise_scheduler.alphas_cumprod.numpy(),
            t_tensor.cpu(),
            x_t.shape
        ).to(x_t.device)

        # x_t = sqrt(bar_alpha[t]) * gt + sqrt(1 - bar_alpha[t]) * fresh_noise
        noise = torch.randn_like(x_t)
        weighed_gt = (alphas_cumprod_t ** 0.5) * gt + \
                     (1 - alphas_cumprod_t ** 0.5) ** 0.5 * (1 - alphas_cumprod_t) ** 0.5 * noise

    # Inject: x = mask * weighed_gt + (1 - mask) * x
    x_t = gt_keep_mask * weighed_gt + (1 - gt_keep_mask) * x_t
    return x_t


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """Extract values from a 1-D numpy array for a batch of indices."""
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


# ─────────────────────────────────────────────────────────────────────────────
# Main sampling function: IOR-style with time-jump scheduling
# ─────────────────────────────────────────────────────────────────────────────

def sample_IMDt(model,
                dataloader,
                noise_scheduler,
                mask=None,
                gt=None,
                device=torch.device('cuda:1'),
                num_step=1000,
                sample_shape=(7060, 2000),
                is_condi=False,
                model_pred_type: str = 'noise',
                is_classifier_guidance=False,
                omega=0.1,
                # ── IOR parameters ──────────────────────────────────────
                use_IOR=False,
                IOR_jump=1,
                IOR_start=200,
                use_cumnoise=True,
                # ── Time-jump schedule (DDIM-style fast sampling) ──────────
                schedule_jump_params=None,
                # ── Classifier guidance (from test.py) ────────────────────
                classifier=None,
                classifier_scale=1.0,
                # ── Optional GT noise cache ───────────────────────────────
                gt_noise=None,
                ):
    """
    IOR-style sampling for IMDt.

    Supports three modes:
      1. Plain inpainting (use_IOR=False, schedule_jump_params=None):
         Standard DDPM reverse diffusion, mask enforced after each step.
      2. IOR harmonization (use_IOR=True):
         GT region re-noised at each step before model evaluation,
         enabling iterative co-evolution of known and unknown regions.
      3. Time-jump fast sampling (schedule_jump_params set):
         DDIM-style schedule with forward+reverse jumps,
         further accelerated by combining with IOR.

    Args:
        model: DiT_IMDt denoising model
        dataloader: yields (data, celltype) tuples for conditioning
        noise_scheduler: NoiseScheduler instance
        mask: binary mask, 1=known/keep, 0=unknown/impute. Shape (B, G)
        gt: ground truth tensor, same shape as mask. Shape (B, G)
        device: torch device
        num_step: number of diffusion timesteps
        sample_shape: (num_cells, num_genes)
        is_condi: whether to use conditioning
        model_pred_type: 'noise' (predict noise ε) or 'x_start' (predict x0)
        is_classifier_guidance: whether to use CFG
        omega: CFG hyperparameter, higher = stronger conditioning
        use_IOR: enable IOR harmonization (GT re-noising per step)
        IOR_jump: number of harmonization iterations per timestep
        IOR_start: only apply IOR for timesteps < IOR_start
        use_cumnoise: use cached GT noise (True=deterministic, False=stochastic)
        schedule_jump_params: dict with keys:
            t_T (int): starting timestep
            n_sample (int): num reverse steps per forward jump
            jump_length (int): length of forward jump
            jump_n_sample (int): num forward jumps
            jump2_length, jump2_n_sample, jump3_length, jump3_n_sample (optional)
            start_resampling (int): timestep after which jumps start
        classifier: optional classifier model for guidance
        classifier_scale: classifier guidance strength
        gt_noise: optional pre-generated noise tensor (B, G) for caching

    Returns:
        recon_x: imputed gene expression, numpy array shape (B, G)
    """
    model.eval()

    # ── Initialize x_T ────────────────────────────────────────────────────
    x_t = torch.randn(sample_shape[0], sample_shape[1]).to(device)

    # Mask and GT to device
    if mask is not None:
        mask = torch.tensor(mask, dtype=torch.float32).to(device)
    if gt is not None:
        gt = torch.tensor(gt, dtype=torch.float32).to(device)

    # IOR init: fill known region with GT at t=T (high noise level)
    if use_IOR and gt is not None and mask is not None:
        # Forward-diffuse GT to timestep T, then inject
        if gt_noise is not None:
            noise_scheduler.store_gt_noises(gt, gt_noise)
        else:
            # No cached noise, use stochastic init for known region
            t_T = num_step - 1
            t_init = torch.tensor([t_T], device=device)
            alpha_cumprod_T = noise_scheduler.alphas_cumprod[t_init].reshape(-1, 1).to(device)
            noise = torch.randn_like(x_t)
            gt_noised = (alpha_cumprod_T ** 0.5) * gt + \
                        ((1 - alpha_cumprod_T) ** 0.5) * noise
            x_t = mask * gt_noised + (1 - mask) * x_t
    elif mask is not None and gt is not None:
        # Plain inpainting init: fill known region with clean GT
        x_t = mask * gt + (1 - mask) * x_t

    # Reset cached GT noises to avoid cross-contamination
    noise_scheduler.reset_gt_noises()

    # ── Classifier guidance function (from test.py) ──────────────────────
    cond_fn = None
    if classifier is not None and classifier_scale > 0:
        def cond_fn(x_in, t_in, y=None, **kwargs):
            """Classifier guidance: gradients toward target class."""
            with torch.enable_grad():
                x_in_detach = x_in.detach().requires_grad_(True)
                logits = classifier(x_in_detach, t_in)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return torch.autograd.grad(selected.sum(), x_in_detach)[0] * classifier_scale

    # ── Time-jump schedule vs simple linear schedule ───────────────────────
    if schedule_jump_params is not None:
        times = get_schedule_jump(**schedule_jump_params)
    else:
        times = list(range(num_step))[::-1]  # [T-1, T-2, ..., 0]

    # Create (t_last, t_cur) pairs for iteration
    time_pairs = list(zip(times[:-1], times[1:]))

    # ── Main sampling loop ────────────────────────────────────────────────
    pbar = tqdm(time_pairs, disable=False)
    pred_xstart = None

    for t_last, t_cur in pbar:
        t_last_tensor = torch.tensor([t_last], device=device).long()
        t_cur_is_forward_jump = t_cur > t_last  # True = going forward in time (re-noising)

        with torch.no_grad():
            if t_cur_is_forward_jump and use_IOR and t_last < IOR_start:
                # ── IOR harmonization (forward jump): ──────────────────
                # 1. Run IOR_jump iterations of denoise → re-noise → inject
                for IOR_iter in range(IOR_jump):
                    # Step A: GT re-noising injection (before model call)
                    if gt is not None and mask is not None:
                        x_t = IOR_inject(
                            x_t, gt, mask, noise_scheduler,
                            t=t_last, use_cumnoise=use_cumnoise)

                    # Step B: Model prediction
                    model_output = model_sample_IMDt(
                        model=model, device=device, dataloader=dataloader,
                        total_sample=x_t, time=t_last,
                        is_condi=is_condi, condi_flag=True)

                    # Step C: CFG if enabled
                    if is_classifier_guidance:
                        model_output_uncond = model_sample_IMDt(
                            model=model, device=device, dataloader=dataloader,
                            total_sample=x_t, time=t_last,
                            is_condi=is_condi, condi_flag=False)
                        model_output = (1 + omega) * model_output - omega * model_output_uncond

                    # Step D: One reverse diffusion step: x_{t-1} from x_t
                    x_t, pred_xstart = noise_scheduler.step(
                        model_output=model_output,
                        timestep=t_last_tensor,
                        sample=x_t,
                        model_pred_type=model_pred_type,
                    )

                # Step E: Forward jump (undo) — move from x_{t-1} to x_t again
                # The "forward jump" in IOR means: take the estimated x_0,
                # re-noise it to the SAME timestep t_last (or t_last+shift),
                # then re-inject. Here we use a simple re-noising step.
                img_before = x_t.clone()
                img_after = x_t  # est_x_0 ≈ x_t at this point after last step
                x_t = noise_scheduler.undo(
                    image_before_step=img_before,
                    img_after_model=img_after,
                    est_x_0=pred_xstart,
                    t=t_last_tensor,
                )

            elif t_cur_is_forward_jump and not use_IOR:
                # Forward jump without IOR: just re-noise slightly
                img_before = x_t.clone()
                x_t = noise_scheduler.undo(img_before, x_t, None, t=t_last_tensor)

            else:
                # ── Reverse jump (standard denoising step): ───────────────
                # 1. GT re-noising injection (for IOR mode)
                if use_IOR and gt is not None and mask is not None and t_last < IOR_start:
                    x_t = IOR_inject(
                        x_t, gt, mask, noise_scheduler,
                        t=t_last, use_cumnoise=use_cumnoise)

                # 2. Model prediction
                model_output = model_sample_IMDt(
                    model=model, device=device, dataloader=dataloader,
                    total_sample=x_t, time=t_last,
                    is_condi=is_condi, condi_flag=True)

                # 3. CFG
                if is_classifier_guidance:
                    model_output_uncond = model_sample_IMDt(
                        model=model, device=device, dataloader=dataloader,
                        total_sample=x_t, time=t_last,
                        is_condi=is_condi, condi_flag=False)
                    model_output = (1 + omega) * model_output - omega * model_output_uncond

                # 4. One reverse diffusion step
                x_t, pred_xstart = noise_scheduler.step(
                    model_output=model_output,
                    timestep=t_last_tensor,
                    sample=x_t,
                    model_pred_type=model_pred_type,
                )

                # 5. Plain mask enforcement (for non-IOR mode or after IOR_start)
                if not use_IOR and mask is not None and gt is not None:
                    x_t = x_t * (1. - mask) + mask * gt

        pbar.set_description_str(desc=f'IOR t={t_last}→{t_cur}  '
                                      f'{"[harmonize]" if use_IOR and t_last < IOR_start else ""}')

    recon_x = x_t.detach().cpu().numpy()
    return recon_x


# ─────────────────────────────────────────────────────────────────────────────
# Legacy wrapper: preserve backward compatibility with existing code
# ─────────────────────────────────────────────────────────────────────────────

def sample_IMDt_legacy(model,
                         dataloader,
                         noise_scheduler,
                         mask=None,
                         gt=None,
                         device=torch.device('cuda:1'),
                         num_step=1000,
                         sample_shape=(7060, 2000),
                         is_condi=False,
                         sample_intermediate=200,
                         model_pred_type: str = 'noise',
                         is_classifier_guidance=False,
                         omega=0.1):
    """
    Legacy sampling function — equivalent to the original sample.py behavior.
    Preserved for backward compatibility with existing code that calls sample_IMDt directly.
    """
    model.eval()
    x_t = torch.randn(sample_shape[0], sample_shape[1]).to(device)
    timesteps = list(range(num_step))[::-1]

    mask_t = torch.tensor(mask).to(device) if mask is not None else None
    gt_t = torch.tensor(gt).to(device) if gt is not None else None

    if mask_t is not None and gt_t is not None:
        x_t = x_t * (1 - mask_t) + gt_t * mask_t

    if sample_intermediate:
        timesteps = timesteps[:sample_intermediate]

    ts = tqdm(timesteps)
    for t_idx, time in enumerate(ts):
        ts.set_description_str(desc=f'time: {time}')
        with torch.no_grad():
            model_output = model_sample_IMDt(
                model, device, dataloader, x_t, time, is_condi, condi_flag=True)

            if is_classifier_guidance:
                model_output_uncond = model_sample_IMDt(
                    model, device, dataloader, x_t, time, is_condi, condi_flag=False)
                model_output = (1 + omega) * model_output - omega * model_output_uncond

        x_t, _ = noise_scheduler.step(
            model_output,
            torch.tensor([time], device=device).long(),
            x_t,
            model_pred_type=model_pred_type)

        if mask_t is not None and gt_t is not None:
            x_t = x_t * (1. - mask_t) + mask_t * gt_t

    return x_t.detach().cpu().numpy()
