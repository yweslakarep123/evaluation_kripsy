import sys
sys.path.append('FlowPolicy/flow_policy_3d')
from typing import Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
import copy
import time
import numpy as np
from flow_policy_3d.sde_lib import ConsistencyFM
from flow_policy_3d.model.common.normalizer import LinearNormalizer
from flow_policy_3d.policy.base_policy import BasePolicy
from flow_policy_3d.model.flow.conditional_unet1d import ConditionalUnet1D
from flow_policy_3d.model.flow.mask_generator import LowdimMaskGenerator
from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.common.model_util import print_params
from flow_policy_3d.model.vision.pointnet_extractor import build_obs_encoder
import warnings
warnings.filterwarnings("ignore")

class FlowPolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict, 
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="mlp",
            pointcloud_encoder_cfg=None,
            obs_encoder_type="pointnet",
            state_encoder_cfg=None,
            Conditional_ConsistencyFM=None,           
            eta=0.01,
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: 
            # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])
        
        self.obs_encoder_type = str(obs_encoder_type).lower()
        # #region agent log
        try:
            import json as _json
            import time as _time
            from pathlib import Path as _Path
            _lp = _Path(__file__).resolve().parents[3] / ".cursor" / "debug-f725e3.log"
            _lp.parent.mkdir(parents=True, exist_ok=True)
            with open(_lp, "a", encoding="utf-8") as _lf:
                _lf.write(
                    _json.dumps(
                        {
                            "sessionId": "f725e3",
                            "hypothesisId": "A",
                            "location": "flowpolicy.py:__init__",
                            "message": "obs_encoder before build_obs_encoder",
                            "data": {
                                "obs_encoder_type": self.obs_encoder_type,
                                "obs_keys": list(obs_dict.keys()),
                            },
                            "timestamp": int(_time.time() * 1000),
                            "runId": "init",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        obs_encoder = build_obs_encoder(
            observation_space=obs_dict,
            encoder_type=self.obs_encoder_type,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            state_encoder_cfg=state_encoder_cfg,
        )

        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        #obs_as_global_cond=true
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(
            f"[FlowPolicy] obs_encoder_type: {self.obs_encoder_type}", "yellow"
        )
        if self.obs_encoder_type == "pointnet":
            cprint(f"[FlowPolicy] use_pc_color: {self.use_pc_color}", "yellow")
            cprint(f"[FlowPolicy] pointnet_type: {self.pointnet_type}", "yellow")


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

        self.obs_encoder = obs_encoder
        self.model = model
        
        
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        
        if Conditional_ConsistencyFM is None:
                    Conditional_ConsistencyFM = {
                        'eps': 1e-2,
                        'num_segments': 2,
                        'boundary': 1,
                        'delta': 1e-2,
                        'alpha': 1e-5,
                        'num_inference_step': 1
                    }
        self.eta = eta
        self.eps = Conditional_ConsistencyFM['eps']
        self.num_segments = Conditional_ConsistencyFM['num_segments']
        self.boundary = Conditional_ConsistencyFM['boundary']
        self.delta = Conditional_ConsistencyFM['delta']
        self.alpha = Conditional_ConsistencyFM['alpha']
        self.num_inference_step = Conditional_ConsistencyFM['num_inference_step']

        print_params(self)
        
    # ========= inference  ============
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        *,
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # #region agent log
        if not getattr(self, "_dbg_norm_logged", False):
            self._dbg_norm_logged = True
            try:
                import json as _json
                _norm_scales = {}
                for _k, _v in getattr(self.normalizer, "params_dict", {}).items():
                    if hasattr(_v, "get") and "scale" in _v:
                        _norm_scales[_k] = int(_v["scale"].shape[0])
                _obs_shapes = {
                    _k: list(_v.shape) for _k, _v in obs_dict.items()
                }
                with open(
                    "/home/daffa/Documents/kripsy12/.cursor/debug-8a2c7a.log", "a"
                ) as _lf:
                    _lf.write(
                        _json.dumps(
                            {
                                "sessionId": "8a2c7a",
                                "location": "flowpolicy.py:predict_action",
                                "message": "obs vs normalizer dims before normalize",
                                "data": {
                                    "obs_shapes": _obs_shapes,
                                    "normalizer_scale_dims": _norm_scales,
                                },
                                "hypothesisId": "H1,H2,H4",
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
        # #endregion
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if self.obs_encoder_type == "pointnet" and not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            #print(f'1 : {nobs_features.shape}')#2,128
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True
        
        # run sampling (deterministic: SDE tanpa noise acak — mean policy untuk RL / eval)
        if deterministic:
            noise = torch.zeros(
                size=cond_data.shape,
                dtype=cond_data.dtype,
                device=cond_data.device,
            )
        else:
            noise = torch.randn(
                size=cond_data.shape,
                dtype=cond_data.dtype,
                device=cond_data.device,
                generator=generator,
            )
        z = noise.detach().clone()  # a0

        sde = ConsistencyFM('gaussian', 
                            noise_scale=1.0,  
                            use_ode_sampler='rk45', # unused
                            sigma_var=0.0, 
                            ode_tol=1e-5, 
                            sample_N= self.num_inference_step)

        # Uniform
        dt = 1./self.num_inference_step
        eps = float(self.eps)
        # num_t = i/N*(1-eps)+eps must stay < 1 so (1-num_t)^2 in the SDE update stays > 0
        eps = min(max(eps, 1e-8), 1.0 - 1e-6)

        for i in range(sde.sample_N):
            num_t = i /sde.sample_N * (1 - eps) + eps
            t = torch.ones(z.shape[0], device=noise.device) * num_t
            pred = self.model(z, t*99, local_cond=local_cond, global_cond=global_cond) ### Copy from models/utils.py 
            # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability 
            sigma_t = sde.sigma_t(num_t)
            # #region agent log
            try:
                import json as _json
                import time as _time
                from pathlib import Path as _Path

                _log_path = (
                    _Path(__file__).resolve().parents[3] / ".cursor" / "debug-db6e34.log"
                )
                _log_path.parent.mkdir(parents=True, exist_ok=True)

                _den = float(2 * (float(sde.noise_scale) ** 2) * ((1.0 - float(num_t)) ** 2))
                if i < 5 or i >= int(sde.sample_N) - 1 or _den < 1e-18:
                    with open(_log_path, "a", encoding="utf-8") as _lf:
                        _lf.write(
                            _json.dumps(
                                {
                                    "sessionId": "db6e34",
                                    "hypothesisId": "A",
                                    "location": "flowpolicy.py:predict_action",
                                    "message": "inference_step",
                                    "data": {
                                        "i": i,
                                        "sample_N": int(sde.sample_N),
                                        "eps": float(eps),
                                        "num_t": float(num_t),
                                        "denom": _den,
                                        "noise_scale": float(sde.noise_scale),
                                    },
                                    "timestamp": int(_time.time() * 1000),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            except Exception:
                pass
            # #endregion
            pred_sigma = pred + (sigma_t**2)/(2*(sde.noise_scale**2)*((1.-num_t)**2)) * (0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*z.detach().clone())
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
        z[cond_mask] = cond_data[cond_mask].to(dtype=z.dtype)  # a1
        # unnormalize prediction
        naction_pred = z[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result
    
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def compute_loss(self, batch):
        eps = self.eps
        num_segments = self.num_segments
        boundary = self.boundary
        delta  = self.delta
        alpha =  self.alpha
        reduce_op = torch.mean
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        target = nactions

        if self.obs_encoder_type == "pointnet" and not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        # gt & noise
        target = target
        a0 = torch.randn(trajectory.shape, device=trajectory.device)
       
        t = torch.rand(target.shape[0], device=target.device) * (1 - eps) + eps # 1=sde.T
        r = torch.clamp(t + delta, max=1.0)
        t_expand = t.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        r_expand = r.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        xt = t_expand * target + (1.-t_expand) * a0
        xr = r_expand * target + (1.-r_expand) * a0
        #apply mask
        xt[condition_mask] = cond_data[condition_mask]
        xr[condition_mask] = cond_data[condition_mask]

        segments = torch.linspace(0, 1, num_segments + 1, device=target.device)
        seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1) # .clamp(min=1) prevents the inclusion of 0 in indices.
        segment_ends = segments[seg_indices]
        segment_ends_expand = segment_ends.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        x_at_segment_ends = segment_ends_expand * target + (1.-segment_ends_expand) * a0
    
        def f_euler(t_expand, segment_ends_expand, xt, vt):
            return xt + (segment_ends_expand - t_expand) * vt
        def threshold_based_f_euler(t_expand, segment_ends_expand, xt, vt, threshold, x_at_segment_ends):
            if (threshold, int) and threshold == 0:
                return x_at_segment_ends
      
            less_than_threshold = t_expand < threshold
      
            res = (
        less_than_threshold * f_euler(t_expand, segment_ends_expand, xt, vt)
        + (~less_than_threshold) * x_at_segment_ends
        )
            return res
        vt = self.model(xt, t*99, cond=local_cond, global_cond=global_cond)
        vr = self.model(xr, r*99, local_cond=local_cond, global_cond=global_cond)
        # mask
        masked_cond = cond_data[condition_mask].to(dtype=vt.dtype)
        vt[condition_mask] = masked_cond
        vr[condition_mask] = masked_cond.to(dtype=vr.dtype)

        vr = torch.nan_to_num(vr)
      
        ft = f_euler(t_expand, segment_ends_expand, xt, vt)
        fr = threshold_based_f_euler(r_expand, segment_ends_expand, xr, vr, boundary, x_at_segment_ends)

        ##### loss #####
        losses_f = torch.square(ft - fr)
        losses_f = reduce_op(losses_f.reshape(losses_f.shape[0], -1), dim=-1)
    
        def masked_losses_v(vt, vr, threshold, segment_ends, t):
            if (threshold, int) and threshold == 0:
                return 0
    
            less_than_threshold = t_expand < threshold
      
            far_from_segment_ends = (segment_ends - t) > 1.01 * delta
            far_from_segment_ends = far_from_segment_ends.view(-1, 1, 1).repeat(1, trajectory.shape[1], trajectory.shape[2])
      
            losses_v = torch.square(vt - vr)
            losses_v = less_than_threshold * far_from_segment_ends * losses_v
            losses_v = reduce_op(losses_v.reshape(losses_v.shape[0], -1), dim=-1)
      
            return losses_v
    
        losses_v = masked_losses_v(vt, vr, boundary, segment_ends, t)

        loss = torch.mean(losses_f + alpha * losses_v)
        loss_dict = { 'bc_loss': 
                     loss.item(),}
        
        return loss, loss_dict
