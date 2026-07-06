import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

    


class StateFlowPolicyEncoder(nn.Module):
    """Encoder observasi state-only (tanpa point cloud) untuk lingkungan state-based."""

    def __init__(
        self,
        observation_space: Dict,
        out_channel: int = 256,
        state_mlp_size=(256, 256),
        state_mlp_activation_fn=nn.ReLU,
        use_layernorm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.state_key = "agent_pos"
        if self.state_key not in observation_space:
            raise KeyError(
                f"StateFlowPolicyEncoder membutuhkan key {self.state_key!r} di observation_space"
            )
        self.state_shape = observation_space[self.state_key]
        state_dim = int(self.state_shape[0])

        if len(state_mlp_size) == 0:
            raise RuntimeError("state_mlp_size tidak boleh kosong")
        elif len(state_mlp_size) == 1:
            net_arch = []
            output_dim = int(state_mlp_size[0])
        else:
            net_arch = list(state_mlp_size[:-1])
            output_dim = int(state_mlp_size[-1])

        if int(out_channel) != output_dim:
            output_dim = int(out_channel)

        layers = create_mlp(
            state_dim, output_dim, net_arch, state_mlp_activation_fn
        )
        if use_layernorm and len(layers) > 0:
            # Sisipkan LayerNorm setelah setiap Linear (sebelum ReLU berikutnya).
            wrapped = []
            for mod in layers:
                wrapped.append(mod)
                if isinstance(mod, nn.Linear):
                    wrapped.append(nn.LayerNorm(mod.out_features))
            layers = wrapped

        self.state_mlp = nn.Sequential(*layers)
        self.n_output_channels = output_dim
        cprint(
            f"[StateFlowPolicyEncoder] state dim={state_dim} -> out={self.n_output_channels}",
            "yellow",
        )

    def forward(self, observations: Dict) -> torch.Tensor:
        state = observations[self.state_key]
        assert state.ndim == 2, cprint(
            f"state shape harus (B, D), dapat {state.shape}", "red"
        )
        return self.state_mlp(state)

    def output_shape(self):
        return self.n_output_channels


def build_obs_encoder(
    observation_space: Dict,
    *,
    encoder_type: str = "pointnet",
    img_crop_shape=None,
    out_channel=256,
    state_mlp_size=(64, 64),
    state_mlp_activation_fn=nn.ReLU,
    pointcloud_encoder_cfg=None,
    use_pc_color=False,
    pointnet_type="mlp",
    state_encoder_cfg=None,
):
    """Factory encoder observasi: ``pointnet`` (default) atau ``state``."""
    encoder_type = str(encoder_type).lower()
    # #region agent log
    try:
        import json as _json
        import time as _time
        from pathlib import Path as _Path
        _lp = _Path(__file__).resolve().parents[4] / ".cursor" / "debug-f725e3.log"
        _lp.parent.mkdir(parents=True, exist_ok=True)
        with open(_lp, "a", encoding="utf-8") as _lf:
            _lf.write(
                _json.dumps(
                    {
                        "sessionId": "f725e3",
                        "hypothesisId": "A",
                        "location": "pointnet_extractor.py:build_obs_encoder",
                        "message": "encoder branch",
                        "data": {
                            "encoder_type": encoder_type,
                            "obs_keys": list(observation_space.keys()),
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
    if encoder_type == "state":
        cfg = dict(state_encoder_cfg or {})
        mlp_hidden = cfg.get("mlp_hidden_dims", None)
        if mlp_hidden is not None:
            hidden = list(mlp_hidden)
            state_mlp_size = tuple(hidden + [int(out_channel)])
        return StateFlowPolicyEncoder(
            observation_space=observation_space,
            out_channel=int(out_channel),
            state_mlp_size=state_mlp_size,
            state_mlp_activation_fn=state_mlp_activation_fn,
            use_layernorm=bool(cfg.get("use_layernorm", False)),
        )
    if encoder_type == "pointnet":
        return FlowPolicyEncoder(
            observation_space=observation_space,
            img_crop_shape=img_crop_shape,
            out_channel=out_channel,
            state_mlp_size=state_mlp_size,
            state_mlp_activation_fn=state_mlp_activation_fn,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )
    raise NotImplementedError(f"encoder_type: {encoder_type}")


class FlowPolicyEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
                   
        cprint(f"[FlowPolicyEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[FlowPolicyEncoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[FlowPolicyEncoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "mlp":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[FlowPolicyEncoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        pn_feat = self.extractor(points)    # B * out_channel
        pn_feat_dim  = pn_feat.shape
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64

        state_feat_dim = state_feat.shape

        # cprint(f"[FlowPolicyEncoder] pn_feat_dim: {pn_feat_dim}", "red")
        # cprint(f"[FlowPolicyEncoder] state_feat dim: {state_feat_dim}", "red")

        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        final_feat_dim = final_feat.shape
        # cprint(f"[FlowPolicyEncoder] final_feat_dim: {final_feat_dim}", "red")

        return final_feat


    def output_shape(self):
        return self.n_output_channels