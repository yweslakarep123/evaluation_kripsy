import gym
import numpy as np
import torch
import os

from termcolor import cprint
from flow_policy_3d.gym_util.mujoco_point_cloud import PointCloudGenerator
from typing import NamedTuple, Any
from dm_env import StepType

ADROIT_PC_TRANSFORM = np.array([
                    [1, 0, 0],
                    [0, np.cos(np.radians(45)), np.sin(np.radians(45))],
                    [0, -np.sin(np.radians(45)), np.cos(np.radians(45))]])

def _fps_sample_farthest_points(points_b_n3: torch.Tensor, K: int) -> torch.Tensor:
    """Greedy farthest-point sampling; mirrors PyTorch3D-style iterative FPS when pytorch3d is absent.

    Args:
        points_b_n3: (B, N, 3) tensor on any torch device.
        K: number of points to sample.

    Returns:
        Long tensor of shape (B, K) with indices into N.
    """
    B, N, _ = points_b_n3.shape
    device = points_b_n3.device
    centroids = torch.zeros(B, K, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(K):
        centroids[:, i] = farthest
        centroid = points_b_n3[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((points_b_n3 - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance = torch.where(mask, dist, distance)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


ENV_POINT_CLOUD_CONFIG = {
    # adroit
    'adroit_hammer': {
        'min_bound': [-10, -10, -0.099],
        'max_bound': [10, 10, 10],
        'num_points': 512,
        'point_sampling_method': 'fps',
        'cam_names':['top'],
        'transform': ADROIT_PC_TRANSFORM,
        'scale': np.array([1, 1, 1]),
        'offset': np.array([0, 0, 1.]),
    },
    
    'adroit_door': {
        'min_bound': [-10, -10, -0.499],
        'max_bound': [10, 10, 10],
        'num_points': 512,
        'point_sampling_method': 'fps',
        'cam_names':['top'],
        'transform': ADROIT_PC_TRANSFORM,
        'scale': np.array([1, 1, 1]),
        'offset': np.array([0, 0, 1.]),
    },
    
    'adroit_pen': {
        'min_bound': [-10, -10, -0.79],
        'max_bound': [10, 10, 10],
        'num_points': 512,
        'point_sampling_method': 'fps',
        'cam_names':['vil_camera'],
        'transform': None,
        'scale': np.array([1, 1, 1]),
        'offset': np.array([0, 0, 0.]),
    },
    
    
}

def point_cloud_sampling(
    point_cloud: np.ndarray,
    num_points: int,
    method: str = "fps",
    device: str = "cuda",
):
    """
    support different point cloud sampling methods
    point_cloud: (N, 6), xyz+rgb or (N, 3), xyz
    device: torch device for FPS (e.g. 'cuda', 'cuda:0', 'cpu'); falls back to cpu if CUDA unavailable.
    """
    if num_points == "all":  # use all points
        return point_cloud

    if point_cloud.shape[0] <= num_points:
        # cprint(f"warning: point cloud has {point_cloud.shape[0]} points, but we want to sample {num_points} points", 'yellow')
        # pad with zeros
        point_cloud_dim = point_cloud.shape[-1]
        point_cloud = np.concatenate(
            [
                point_cloud,
                np.zeros((num_points - point_cloud.shape[0], point_cloud_dim)),
            ],
            axis=0,
        )
        return point_cloud

    if method == "uniform":
        # uniform sampling
        sampled_indices = np.random.choice(
            point_cloud.shape[0], num_points, replace=False
        )
        point_cloud = point_cloud[sampled_indices]
    elif method == "fps":
        dev_str = device
        if dev_str.startswith("cuda") and not torch.cuda.is_available():
            dev_str = "cpu"
        dev = torch.device(dev_str)
        point_cloud_t = torch.from_numpy(point_cloud).unsqueeze(0).to(dev)
        pts3 = point_cloud_t[..., :3]
        try:
            import pytorch3d.ops as torch3d_ops

            _, sampled_indices = torch3d_ops.sample_farthest_points(
                points=pts3, K=int(num_points)
            )
        except ImportError:
            sampled_indices = _fps_sample_farthest_points(pts3, int(num_points))
        point_cloud = point_cloud_t.squeeze(0).cpu().numpy()
        point_cloud = point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
    else:
        raise NotImplementedError(
            f"point cloud sampling method {method} not implemented"
        )

    return point_cloud
    

class ExtendedTimeStepAdroit(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    observation_sensor: Any
    observation_pointcloud: Any
    observation_depth: Any
    action: Any
    n_goal_achieved: Any
    time_limit_reached: Any
    

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)
    

class MujocoPointcloudWrapperAdroit(gym.Wrapper):
    """
    fetch point cloud from mujoco and add it to obs
    """
    def __init__(self, env, env_name:str, use_point_crop=True):
        super().__init__(env)
        self.env_name = env_name
        # point cloud cropping
        self.min_bound = ENV_POINT_CLOUD_CONFIG[env_name].get('min_bound', None)
        self.max_bound = ENV_POINT_CLOUD_CONFIG[env_name].get('max_bound', None)

        self.use_point_crop = use_point_crop
        cprint(f"[MujocoPointcloudWrapper] use_point_crop: {self.use_point_crop}", 'green')
        
        # point cloud sampling
        self.num_points = ENV_POINT_CLOUD_CONFIG[env_name].get('num_points', 512)
        self.point_sampling_method = ENV_POINT_CLOUD_CONFIG[env_name].get('point_sampling_method', 'uniform')
        cprint(f"[MujocoPointcloudWrapper] sampling {self.num_points} points from point cloud using {self.point_sampling_method}", 'green')
        assert self.point_sampling_method in ['uniform', 'fps'], \
            f"point_sampling_method should be one of ['uniform', 'fps'], but got {self.point_sampling_method}"
        
        # point cloud generator
        self.pc_generator = PointCloudGenerator(sim=env.get_mujoco_sim(),
                                                cam_names=ENV_POINT_CLOUD_CONFIG[env_name]['cam_names'])
        self.pc_transform = ENV_POINT_CLOUD_CONFIG[env_name].get('transform', None)
        self.pc_scale = ENV_POINT_CLOUD_CONFIG[env_name].get('scale', None)
        self.pc_offset = ENV_POINT_CLOUD_CONFIG[env_name].get('offset', None)

    

    def get_point_cloud(self, use_RGB=True):

        # set save_img_dir to save images for debugging
        # save_img_dir = "/home/yanjieze/projects/diffusion-for-dex/imgs"
        save_img_dir = None
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(save_img_dir=save_img_dir) # (N, 6), xyz+rgb
        
        
        
        # do transform, scale, offset, and crop
        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        
        
        if self.pc_offset is not None:    
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset

        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]
            

        
        # sampling to fixed number of points
        point_cloud = point_cloud_sampling(point_cloud=point_cloud, 
                                           num_points=self.num_points, 
                                           method=self.point_sampling_method)
        
        if not use_RGB:
            point_cloud = point_cloud[:, :3]
        return point_cloud, depth


    def step(self, action):
        timestep = self.env.step(action)
        point_cloud, depth = self.get_point_cloud()
        
        # wrap point cloud into obs
        if 'adroit' in self.env_name: # adroit uses a namedtuple for obs
            # so we need to create a new namedtuple
            timestep = ExtendedTimeStepAdroit(step_type=timestep.step_type,
                                         reward=timestep.reward,
                                         discount=timestep.discount,
                                         observation=timestep.observation,
                                         observation_sensor=timestep.observation_sensor,
                                         observation_pointcloud=point_cloud,
                                         observation_depth=depth,
                                         action=timestep.action,
                                         n_goal_achieved=timestep.n_goal_achieved,
                                         time_limit_reached=timestep.time_limit_reached)                        
        else:
            raise NotImplementedError
        return timestep

    def reset(self):
        timestep = self.env.reset()
        point_cloud, depth = self.get_point_cloud()
        
        # wrap point cloud into obs
        if 'adroit' in self.env_name: # adroit uses a namedtuple for obs
            # so we need to create a new namedtuple
            timestep = ExtendedTimeStepAdroit(step_type=timestep.step_type,
                                         reward=timestep.reward,
                                         discount=timestep.discount,
                                         observation=timestep.observation,
                                         observation_sensor=timestep.observation_sensor,
                                         observation_pointcloud=point_cloud,
                                         observation_depth=depth,
                                         action=timestep.action,
                                         n_goal_achieved=timestep.n_goal_achieved,
                                         time_limit_reached=timestep.time_limit_reached)                        
        else:
            raise NotImplementedError
        return timestep


    



