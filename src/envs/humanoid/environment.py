"""Minimal humanoid embodiment adapter for Open-DiffLoco."""

from src.envs.go2.variants.blind_linvel_nokinref.environment import Go2Env


class HumanoidEnv(Go2Env):
    """Use the upstream blind velocity-locomotion task with a humanoid model."""

    def __init__(
        self,
        xml_path: str = "src/envs/humanoid/models/humanoid_mjx.xml",
        **kwargs,
    ):
        super().__init__(
            xml_path=xml_path,
            foot_names=("FL", "FR", "RL", "RR"),
            torso_body_names=("pelvis", "torso"),
            target_height=0.88,
            termination_height=0.528,
            **kwargs,
        )
