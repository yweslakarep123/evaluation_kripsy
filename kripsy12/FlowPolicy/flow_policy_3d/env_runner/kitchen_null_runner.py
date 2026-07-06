from flow_policy_3d.env_runner.base_runner import BaseRunner


class KitchenNullRunner(BaseRunner):
    """No-op runner for training when rollout is disabled."""

    def run(self, policy):
        return {}

    def close(self):
        pass
