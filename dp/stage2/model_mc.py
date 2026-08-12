"""Stage-2 head for MoveCube: fresh ConditionalUnet1D over frozen features,
with an optional TTT memory channel.

The UNet is RoboMME's own (eval_envs.model.nn_modules.ConditionalUnet1D) with
the stage-1 hyperparameters — down_dims [256,512,1024], kernel 5, groups 8,
step-embed 128, DDPM 100/100 epsilon — so both arms share the architecture the
24% baseline validated, minus the trainable encoder.

Arms (RMBench naming):
  use_ttt=False                 -> arm 2 (frozen-encoder control, no memory)
  use_ttt=True, inject="concat" -> arm 4a (memory gets its own cond dims)

Actions/state arrive PRE-NORMALIZED (stage-1 minmax stats, see dataset_mc) —
this module trains and predicts in normalized space; the eval script
unnormalizes with the same stats.
"""
import os
import pathlib
import sys

import torch
import torch.nn as nn

_DP_REPO = os.environ.get("DP_REPO")
assert _DP_REPO, "set DP_REPO to your clone of github.com/RoboMME/DP"
sys.path.insert(0, _DP_REPO)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # noqa: E402
from eval_envs.model.nn_modules import ConditionalUnet1D  # noqa: E402
from ttt_cell import TTTMemory  # noqa: E402


class DPMemoryPolicyMC(nn.Module):

    def __init__(
        self,
        d_feat=136,
        action_dim=8,
        horizon=16,           # action_pred_horizon (stage-1 recipe)
        n_obs_steps=2,        # obs_horizon
        n_action_steps=8,     # action_exec_horizon
        use_ttt=True,
        ttt_kwargs=None,
        inject="concat",
        num_train_timesteps=100,
        num_inference_steps=100,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        diffusion_step_embed_dim=128,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.num_inference_steps = num_inference_steps

        obs_cond_dim = d_feat * n_obs_steps               # 136 * 2 = 272
        self.inject = inject
        global_cond_dim = obs_cond_dim
        if use_ttt and inject == "concat":
            global_cond_dim = obs_cond_dim + (ttt_kwargs or {}).get("d_out", 256)

        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self.use_ttt = use_ttt
        if use_ttt:
            self.ttt = TTTMemory(d_in=d_feat, **(ttt_kwargs or {}))
            if inject == "add":
                self.adapter = nn.Linear(self.ttt.d_out, obs_cond_dim)
                nn.init.zeros_(self.adapter.weight)
                nn.init.zeros_(self.adapter.bias)
            else:
                self.adapter = None
        else:
            self.ttt = None
            self.adapter = None

    # ---- memory ------------------------------------------------------------
    def rollout_memory(self, feat, mask):
        """Full-episode TTT rollout INCLUDING the video prefix."""
        if not self.use_ttt:
            return None, {}
        return self.ttt(feat, mask)

    def condition(self, global_cond, mem):
        if not (self.use_ttt and mem is not None):
            return global_cond
        if self.inject == "add":
            return global_cond + self.adapter(mem)
        return torch.cat([global_cond, mem], dim=-1)

    # ---- training ----------------------------------------------------------
    def forward(self, feat, action, mask, lengths, starts, horizon, n_obs_steps):
        """TTT rollout -> exec-window gather -> diffusion loss.

        Must stay a single forward() so DDP's reducer sees the whole graph.
        """
        from dataset_mc import gather_windows
        m, diag = self.rollout_memory(feat, mask)
        gc, mem, act = gather_windows(feat, action, m, starts, horizon,
                                      n_obs_steps, lengths)
        return self.compute_loss(gc, mem, act), diag

    def compute_loss(self, global_cond, mem, action_chunk):
        cond = self.condition(global_cond, mem)
        traj = action_chunk                                # pre-normalized
        noise = torch.randn_like(traj)
        bsz = traj.shape[0]
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                          (bsz,), device=traj.device).long()
        noisy = self.noise_scheduler.add_noise(traj, noise, t)
        pred = self.unet(noisy, t, global_cond=cond)
        return nn.functional.mse_loss(pred, noise)

    # ---- inference (normalized action space) -------------------------------
    @torch.no_grad()
    def predict_action(self, global_cond, mem):
        cond = self.condition(global_cond, mem)
        B = cond.shape[0]
        traj = torch.randn(B, self.horizon, self.action_dim, device=cond.device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_out = self.unet(traj, t, global_cond=cond)
            traj = self.noise_scheduler.step(model_out, t, traj).prev_sample
        # stage-1 convention: the chunk starts AT the current frame
        return traj, traj[:, : self.n_action_steps]
