from typing import List, Dict, Optional, Optional
import numpy as np
import gym
from gym.spaces import Box
from flow_policy_3d.env.kitchen.base import KitchenBase
#region agent log
import json as _json, time as _time; open('/home/daffa/Documents/experiment/.cursor/debug-374ef1.log','a').write(_json.dumps({"sessionId":"374ef1","runId":"post-fix","hypothesisId":"A","location":"kitchen_lowdim_wrapper.py:import","message":"KitchenBase import ok","data":{"module":KitchenBase.__module__},"timestamp":int(_time.time()*1000)})+'\n')
#endregion

class KitchenLowdimWrapper(gym.Env):
    def __init__(self,
            env: KitchenBase,
            init_qpos: Optional[np.ndarray]=None,
            init_qvel: Optional[np.ndarray]=None,
            render_hw = (240,360)
        ):
        self.env = env
        self.init_qpos = init_qpos
        self.init_qvel = init_qvel
        self.render_hw = render_hw

    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        if self.init_qpos is not None:
            # reset anyway to be safe, not very expensive
            _ = self.env.reset()
            # start from known state
            self.env.set_state(self.init_qpos, self.init_qvel)
            obs = self.env._get_obs()
            return obs
            # obs, _, _, _ = self.env.step(np.zeros_like(
            #     self.action_space.sample()))
            # return obs
        else:
            return self.env.reset()

    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, width=w, height=h)
    
    def step(self, a):
        return self.env.step(a)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
