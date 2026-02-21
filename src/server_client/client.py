import threading
from lerobot.robots.so_follower import SOFollowerConfig, SOFollower
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.async_inference.helpers import visualize_action_queue_size
from collections import defaultdict
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
camera_config = defaultdict()
camera_config["side"] = OpenCVCameraConfig(index_to_path=0, width=640, height=480, fps=30)
camera_config["top"] = OpenCVCameraConfig(index_to_path=2, width=640, height=480, fps=30)

follower_port = 5000
follower_id = "so101_pickplace_follower"

robot_cfg = SOFollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
server_host = "localhost"
server_port = 5000
server_address = f"http://{server_host}:{server_port}"

client_cfg = RobotClientConfig(
    robot=robot_cfg,
    server_address=server_address,
    policy_device = device,
    policy_type = "smolvla",
    pretrained_name_or_path="fracapauano/smolvla_async"
    chunk_size_thresold=0.5
    actions_per_chunk=50,
)
client = RobotClient(client_cfg)

task = "pick and place the block into the bin"

if client.start():
    # start action receiver thread
    action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver_thread.start()

    try:
        client.control_loop(task)
    except KeyboardInterrupt:
        client.stop()
        action_receiver_thread.join()
        print("Client stopped")
        visualize_action_queue_size(client.action_queue)
    finally:
        client.stop()
        action_receiver_thread.join()
        print("Client stopped")
        visualize_action_queue_size(client.action_queue)