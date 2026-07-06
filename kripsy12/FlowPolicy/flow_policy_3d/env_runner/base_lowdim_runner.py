from typing import Dict

from flow_policy_3d.policy.base_lowdim_policy import BaseLowdimPolicy
from flow_policy_3d.env_runner.base_runner import BaseRunner


class BaseLowdimRunner(BaseRunner):
    def run(self, policy: BaseLowdimPolicy) -> Dict:
        raise NotImplementedError()
