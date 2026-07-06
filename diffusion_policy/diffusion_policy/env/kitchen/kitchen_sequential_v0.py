from typing import List

from diffusion_policy.env.kitchen.base import KitchenBase


def make_kitchen_sequential4_env(
    task_sequence: List[str],
    use_abs_action: bool = False,
) -> KitchenBase:
    """Create a Kitchen env with exactly 4 subtasks in prescribed sequential order."""
    assert len(task_sequence) == 4, "Exactly 4 subtasks required"
    for task in task_sequence:
        assert task in KitchenBase.ALL_TASKS, f"Unknown task: {task}"

    class _KitchenSequential4V0(KitchenBase):
        TASK_ELEMENTS = list(task_sequence)
        COMPLETE_IN_ANY_ORDER = False

    return _KitchenSequential4V0(use_abs_action=use_abs_action)


KitchenSequential4V0 = make_kitchen_sequential4_env
