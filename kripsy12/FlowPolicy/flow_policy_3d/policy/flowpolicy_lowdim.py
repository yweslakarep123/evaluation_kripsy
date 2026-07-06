"""FlowPolicy for low-dimensional observation (e.g. Kitchen 60-dim)."""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from termcolor import cprint

from flow_policy_3d.sde_lib import ConsistencyFM
from flow_policy_3d.model.common.normalizer import LinearNormalizer
from flow_policy_3d.policy.base_policy import BasePolicy
from flow_policy_3d.model.flow.conditional_unet1d import ConditionalUnet1D
from flow_policy_3d.model.flow.mask_generator import LowdimMaskGenerator
from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.common.model_util import print_params
from flow_policy_3d.model.vision.pointnet_extractor import create_mlp


class FlowPolicyLowdim(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        horizon,
        n_action_steps,
        n_obs_steps,
        obs_dim: int = 60,
        action_dim: int = 9,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        encoder_output_dim=256,
        obs_mlp_hidden=(256, 256),
        Conditional_ConsistencyFM=None,
        eta=0.01,
        **kwargs,
    ):
        super().__init__()

        self.condition_type = condition_type
        if shape_meta is not None and "action" in shape_meta:
            action_shape = shape_meta["action"]["shape"]
            if len(action_shape) == 1:
                action_dim = action_shape[0]
            elif len(action_shape) == 2:
                action_dim = action_shape[0] * action_shape[1]

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        obs_feature_dim = encoder_output_dim

        mlp_layers = list(
            create_mlp(
                obs_dim,
                encoder_output_dim,
                list(obs_mlp_hidden),
                activation_fn=nn.ReLU,
            )
        )
        self.obs_encoder = nn.Sequential(*mlp_layers)

        input_dim = action_dim
        global_cond_dim = None
        if obs_as_global_cond:
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )
        self.model = model

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond

        if Conditional_ConsistencyFM is None:
            Conditional_ConsistencyFM = {
                "eps": 1e-2,
                "num_segments": 2,
                "boundary": 1,
                "delta": 1e-2,
                "alpha": 1e-5,
                "num_inference_step": 1,
            }
        self.eta = eta
        self.eps = Conditional_ConsistencyFM["eps"]
        self.num_segments = Conditional_ConsistencyFM["num_segments"]
        self.boundary = Conditional_ConsistencyFM["boundary"]
        self.delta = Conditional_ConsistencyFM["delta"]
        self.alpha = Conditional_ConsistencyFM["alpha"]
        self.num_inference_step = Conditional_ConsistencyFM["num_inference_step"]

        cprint(
            f"[FlowPolicyLowdim] obs_dim={obs_dim} action_dim={action_dim} "
            f"encoder_output_dim={encoder_output_dim}",
            "yellow",
        )
        print_params(self)

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (N, obs_dim) -> (N, obs_feature_dim)"""
        return self.obs_encoder(obs)

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        *,
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(obs_dict, torch.Tensor):
            obs_dict = {"obs": obs_dict}
        nobs = self.normalizer.normalize(obs_dict)
        obs = nobs["obs"]
        B = obs.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps
        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_obs = obs[:, :To, :].reshape(-1, self.obs_dim)
            nobs_features = self._encode_obs(this_obs)
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(B, -1)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_obs = obs[:, :To, :].reshape(-1, self.obs_dim)
            nobs_features = self._encode_obs(this_obs).reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + self.obs_feature_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        if deterministic:
            noise = torch.zeros(
                size=cond_data.shape, dtype=cond_data.dtype, device=cond_data.device
            )
        else:
            noise = torch.randn(
                size=cond_data.shape,
                dtype=cond_data.dtype,
                device=cond_data.device,
                generator=generator,
            )
        z = noise.detach().clone()

        sde = ConsistencyFM(
            "gaussian",
            noise_scale=1.0,
            use_ode_sampler="rk45",
            sigma_var=0.0,
            ode_tol=1e-5,
            sample_N=self.num_inference_step,
        )
        eps = min(max(float(self.eps), 1e-8), 1.0 - 1e-6)
        dt = 1.0 / self.num_inference_step

        for i in range(sde.sample_N):
            num_t = i / sde.sample_N * (1 - eps) + eps
            t = torch.ones(z.shape[0], device=noise.device) * num_t
            pred = self.model(z, t * 99, local_cond=local_cond, global_cond=global_cond)
            sigma_t = sde.sigma_t(num_t)
            pred_sigma = pred + (sigma_t**2) / (
                2 * (sde.noise_scale**2) * ((1.0 - num_t) ** 2)
            ) * (
                0.5 * num_t * (1.0 - num_t) * pred - 0.5 * (2.0 - num_t) * z.detach().clone()
            )
            if deterministic:
                z = z.detach().clone() + pred_sigma * dt
            else:
                inc = sigma_t * float(np.sqrt(dt)) * torch.randn(
                    pred_sigma.shape,
                    dtype=pred_sigma.dtype,
                    device=pred_sigma.device,
                    generator=generator,
                )
                z = z.detach().clone() + pred_sigma * dt + inc
        z[cond_mask] = cond_data[cond_mask].to(dtype=z.dtype)
        naction_pred = z[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {"action": action, "action_pred": action_pred}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        eps = self.eps
        num_segments = self.num_segments
        boundary = self.boundary
        delta = self.delta
        alpha = self.alpha
        reduce_op = torch.mean

        nobs = self.normalizer.normalize({"obs": batch["obs"]})["obs"]
        nactions = self.normalizer["action"].normalize(batch["action"])
        target = nactions
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            this_obs = nobs[:, : self.n_obs_steps, :].reshape(-1, self.obs_dim)
            nobs_features = self._encode_obs(this_obs)
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_obs = nobs.reshape(-1, self.obs_dim)
            nobs_features = self._encode_obs(this_obs).reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)
        # #region agent log
        try:
            import json as _json, time as _time
            from pathlib import Path as _Path
            _log_path = _Path(__file__).resolve().parents[3] / ".cursor" / "debug-c84b5d.log"
            _log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(_log_path, "a") as _lf:
                _lf.write(_json.dumps({"sessionId":"c84b5d","hypothesisId":"A,B,E","location":"flowpolicy_lowdim.py:cond_data","message":"dtype_before_model","data":{"cond_data":str(cond_data.dtype),"target":str(target.dtype),"nactions":str(nactions.dtype),"global_cond":str(global_cond.dtype) if global_cond is not None else None},"timestamp":int(_time.time()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
        a0 = torch.randn(trajectory.shape, device=trajectory.device)

        t = torch.rand(target.shape[0], device=target.device) * (1 - eps) + eps
        r = torch.clamp(t + delta, max=1.0)
        t_expand = t.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        r_expand = r.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        xt = t_expand * target + (1.0 - t_expand) * a0
        xr = r_expand * target + (1.0 - r_expand) * a0
        xt[condition_mask] = cond_data[condition_mask]
        xr[condition_mask] = cond_data[condition_mask]
        # #region agent log
        try:
            import json as _json, time as _time
            from pathlib import Path as _Path
            _log_path = _Path(__file__).resolve().parents[3] / ".cursor" / "debug-c84b5d.log"
            with open(_log_path, "a") as _lf:
                _lf.write(_json.dumps({"sessionId":"c84b5d","hypothesisId":"C","location":"flowpolicy_lowdim.py:xt_mask","message":"dtype_after_xt_mask","data":{"xt":str(xt.dtype),"xr":str(xr.dtype),"cond_data":str(cond_data.dtype)},"timestamp":int(_time.time()*1000)})+"\n")
        except Exception:
            pass
        # #endregion

        segments = torch.linspace(0, 1, num_segments + 1, device=target.device)
        seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
        segment_ends = segments[seg_indices]
        segment_ends_expand = segment_ends.view(-1, 1, 1).repeat(
            1, target.shape[1], target.shape[2]
        )
        x_at_segment_ends = segment_ends_expand * target + (1.0 - segment_ends_expand) * a0

        def f_euler(t_expand, segment_ends_expand, xt, vt):
            return xt + (segment_ends_expand - t_expand) * vt

        def threshold_based_f_euler(
            t_expand, segment_ends_expand, xt, vt, threshold, x_at_segment_ends
        ):
            if (threshold, int) and threshold == 0:
                return x_at_segment_ends
            less_than_threshold = t_expand < threshold
            return less_than_threshold * f_euler(
                t_expand, segment_ends_expand, xt, vt
            ) + (~less_than_threshold) * x_at_segment_ends

        vt = self.model(xt, t * 99, local_cond=local_cond, global_cond=global_cond)
        vr = self.model(xr, r * 99, local_cond=local_cond, global_cond=global_cond)
        # #region agent log
        try:
            import json as _json, time as _time
            from pathlib import Path as _Path
            _log_path = _Path(__file__).resolve().parents[3] / ".cursor" / "debug-c84b5d.log"
            with open(_log_path, "a") as _lf:
                _lf.write(_json.dumps({"sessionId":"c84b5d","runId":"post-fix","hypothesisId":"A","location":"flowpolicy_lowdim.py:vt_pre_assign","message":"dtype_before_vt_mask","data":{"vt":str(vt.dtype),"vr":str(vr.dtype),"cond_data":str(cond_data.dtype),"dtype_match":str(vt.dtype)==str(cond_data.dtype)},"timestamp":int(_time.time()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
        masked_cond = cond_data[condition_mask].to(dtype=vt.dtype)
        vt[condition_mask] = masked_cond
        vr[condition_mask] = masked_cond.to(dtype=vr.dtype)
        vr = torch.nan_to_num(vr)

        ft = f_euler(t_expand, segment_ends_expand, xt, vt)
        fr = threshold_based_f_euler(
            r_expand, segment_ends_expand, xr, vr, boundary, x_at_segment_ends
        )

        losses_f = torch.square(ft - fr)
        losses_f = reduce_op(losses_f.reshape(losses_f.shape[0], -1), dim=-1)

        def masked_losses_v(vt, vr, threshold, segment_ends, t):
            if (threshold, int) and threshold == 0:
                return 0
            less_than_threshold = t_expand < threshold
            far_from_segment_ends = (segment_ends - t) > 1.01 * delta
            far_from_segment_ends = far_from_segment_ends.view(-1, 1, 1).repeat(
                1, trajectory.shape[1], trajectory.shape[2]
            )
            losses_v = torch.square(vt - vr)
            losses_v = less_than_threshold * far_from_segment_ends * losses_v
            return reduce_op(losses_v.reshape(losses_v.shape[0], -1), dim=-1)

        losses_v = masked_losses_v(vt, vr, boundary, segment_ends, t)
        loss = torch.mean(losses_f + alpha * losses_v)
        return loss, {"bc_loss": loss.item()}
