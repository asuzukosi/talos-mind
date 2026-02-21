import multiprocessing as mp
from queue import Empty
import torch
from pathlib import Path
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
from lerobot.rl.gym_manipulator import make_robot_env
from lerobot.teleoperators.utils import TeleopEvents

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20

def make_policy_obs(obs, device: torch.device = "cpu"):
    return {
        "observation.state": torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device),
        ** {
            f"observation.image.{k}": torch.from_numpy(obs["pixels"][k]).float().unsqueeze(0).to(device)
            for k in obs["pixels"]
        }
    }

def run_actor(
        transitions_queue: mp.Queue,
        parameters_queue: mp.Queue,
        shutdown_event,
        policy_actor: SACPolicy,
        reward_classifier: Classifier,
        env_cfg: HILSerlRobotEnvConfig,
        device: torch.device = "cpu",
        output_directory: Path | None = None
):
    policy_actor.eval()
    policy_actor.to(device)
    
    reward_classifier.eval()
    reward_classifier.to(device)
    # create robot environment inside the actor process
    env, teleop_device = make_robot_env(env_cfg)

    try:
        for episode in range(MAX_EPISODES):
            if shutdown_event.is_set():
                pass


    except KeyboardInterrupt:
        print("[actor] interopted by user")
    finally:
        # clean up
        if hasattr(env, "robot") and env.robot.is_connected():
            env.robot.disconnect()
        if teleop_device and hasattr(teleop_device, "disconnect"):
            teleop_device.disconnect()
        if output_directory is not None:
            policy_actor.save_pretrained(output_directory)
            print(f"[actor] policy saved to {output_directory}")

    print("[actor] process finished")