import multiprocessing as mp
import signal
from pathlib import Path
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.envs.configs import HILSerlProcessorConfig, HILSerlRobotEnvConfig
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.gym_manipulator import make_robot_env
from lerobot.robots.so_follower import SOFollower, SOFollowerConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from src.learning.rl.actor import run_actor
from src.learning.rl.learner import run_learner

device = "cuda" if torch.cuda.is_available() else "cpu"
output_directory = Path("output/robot_learning_tutorial/hil_serl")
output_directory.mkdir(parents=True, exist_ok=True)

follower_port = 5000
leader_port = 5001

# the robot ids are use to load the right calibration files
follower_id = "so101_pickplace_follower"
leader_id = "so101_pickplace_leader"


# a pretrained model (to be used in-distribution)
reward_classifier_id = "lerobot/reward_classifier_hil_serl_example"
reward_classifier = Classifier.from_pretrained(reward_classifier_id)
reward_classifier.to(device)
reward_classifier.eval()

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20

# robot and environment configurations
robot_cfg = SOFollowerConfig(port=follower_port, id=follower_id)
teleop_cfg = SOLeaderConfig(port=leader_port, id=leader_id)
processor_cfg = HILSerlProcessorConfig(control_mode="leader")

env_cfg = HILSerlRobotEnvConfig(robot=robot_cfg, teleop=teleop_cfg, processor=processor_cfg)
# create robot environment
env, teleop_device = make_robot_env(env_cfg)

obs_features = hw_to_dataset_features(env.robot.observation_features, "observation")
action_features = hw_to_dataset_features(env.robot.action_features, "action")

# create sac policy for action selection
policy_cfg = SACConfig(
    device=device, 
    input_features=obs_features,
    output_features=action_features
    )
policy_actor = SACPolicy(config=policy_cfg)
policy_learner = SACPolicy(config=policy_cfg)

demonstration_repo_id = "lerobot/example_hil_serl_dataset"
offline_dataset = LeRobotDataset(repo_id=demonstration_repo_id)

# online buffer:initiated from scratch
online_replay_buffer = ReplayBuffer(device=device, state_keys=list(obs_features.keys()))
# offline buffer: created from dataset (pre-populated it with demonstrations)
offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
    lerobot_dataset=offline_dataset,
    device=device,
    state_keys=list(obs_features.keys()),
)

# create communication channels between learners and actor processes
transitions_queue = mp.Queue(maxsize=10)
parameters_queue = mp.Queue(maxsize=2)
shutdown_event = mp.Event()

# signal hanlder for graceful shutdown
def signal_handler(sig):
    print(f"signal {sig} received, shutting down")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# create processes
learner_process = mp.Process(
    target=run_learner,
    args=(
        transitions_queue,
        parameters_queue,
        shutdown_event,
        policy_learner,
        online_replay_buffer,
        offline_replay_buffer,
    )
    kwargs={"device": device}
)

actor_process = mp.Process(
    target=run_actor,
    args=(
        transitions_queue,
        parameters_queue,
        shutdown_event,
        policy_actor,
        reward_classifier,
        env_cfg,
    ),
    kwargs={
        "device": device,
        "output_directory": output_directory,
    }
)

learner_process.start()
actor_process.start()

try:
    # wait for actor to finish 
    actor_process.join()
    shutdown_event.set()
    learner_process.join(timeout=10)
except KeyboardInterrupt:
    print("[main] interrupted by user")
    shutdown_event.set()
    actor_process.join(timeout=10)
    learner_process.join(timeout=10)
finally:
    if actor_process.is_alive():
        actor_process.terminate()
    if learner_process.is_alive():
        learner_process.terminate()

print("[main] process finished")