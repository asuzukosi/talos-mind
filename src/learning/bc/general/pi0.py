import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from collections import defaultdict

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20

device = torch.device("mps")  # or "cuda" or "cpu"
model_id = "lerobot/pi0_base"

model = PI0Policy.from_pretrained(model_id)

preprocess, postprocess = make_pre_post_processors(
    model.config,
    model_id,
    # This overrides allows to run on MPS, otherwise defaults to CUDA (if available)
    preprocessor_overrides={"device_processor": {"device": "mps"}},
)

# find ports using lerobot-find-port
follower_port = 5000
follower_id = "so101_pickplace_follower"

camera_config = defaultdict()
camera_config["side"] = OpenCVCameraConfig(index_to_path=0, width=640, height=480, fps=30)
camera_config["top"] = OpenCVCameraConfig(index_to_path=2, width=640, height=480, fps=30)

robot_cfg = SOFollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
robot = SOFollower(robot_cfg)
robot.connect()

task = "pick the red block"
robot_type = "so_follower"

# This is used to match the raw observation keys to the keys expected by the policy
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

for _ in range(MAX_EPISODES):
    for _ in range(MAX_STEPS_PER_EPISODE):
        obs = robot.get_observation()
        obs_frame = build_inference_frame(
            obs, dataset_features, device, task=task, robot_type=robot_type
        )

        obs = preprocess(obs_frame)

        action = model.select_action(obs)
        action = postprocess(action)
        action = make_robot_action(action, dataset_features)
        robot.send_action(action)

    print("Episode finished! Starting new episode...")