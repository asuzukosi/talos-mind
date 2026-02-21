import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SOFollowerConfig, SOFollower
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "talos-thinking/robot_learning_diffusion_example_model"
model = DiffusionPolicy.from_pretrained(model_id)

# creater the lerobot dataset and preprocess and postprocess functions
dataset_id = "lerobot/svla_so101_pickplace"
dataset_metadata = LeRobotDatasetMetadata(repo_id=dataset_id)
preprocess, postprocess = make_pre_post_processors(policy_cfg=model.config,
                                                    dataset_meta=dataset_metadata.stats)

MAX_EPISODES = 5
MAX_SPEED_PER_EPISODE = 20

# configure the follower robot
follower_port = 5000
follower_id = "so101_pickplace_follower"
camera_config = defaultdict()
camera_config["side"] = OpenCVCameraConfig(index_to_path=0, width=640, height=480, fps=30)
camera_config["top"] = OpenCVCameraConfig(index_to_path=2, width=640, height=480, fps=30)
robot_cfg = SOFollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
robot = SOFollower(config=robot_cfg)
robot.connect()

for _ in range(MAX_EPISODES):
    for _ in range(MAX_SPEED_PER_EPISODE):
        obs = robot.get_observation()
        obs_frame = build_inference_frame(obs, dataset_metadata.features, device)
        obs = preprocess(obs_frame)
        action = model.select_action(obs)
        action = postprocess(action)
        action = make_robot_action(action, dataset_metadata.features)
        robot.send_action(action)
    print(f"Episode {_ + 1} completed, starting new episode...")

robot.disconnect()
print("Simulation completed")