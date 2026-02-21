from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so_follower import SOFollower, SOFollowerConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from collections import defaultdict

NUM_EPISODES = 5
FPS=30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Pick and place a block into a bin"

HF_USER ="talos-mind"

follower_port = 5000
leader_port = 5001
follower_id="so101_pickplace_follower"
leader_id="so101_pickplace_leader"

# create the robot and the teleoperator configurations
camera_config = defaultdict(OpenCVCameraConfig)
# set the front camera to the camera configuration
camera_config["front"] = OpenCVCameraConfig(
    index_to_path=0, width=640, height=480, fps=FPS)

# configure the follower robot
robot_config = SOFollowerConfig(port=follower_port,
                                id=follower_id,
                                camera_config=camera_config)

# configure the leader robot
teleop_config = SOLeaderConfig(port=leader_port,
                               id=leader_id,)

# initialize the robot and teleoperator
robot = SOFollower(config=robot_config)
teleop = SOLeader(config=teleop_config)

# configure the daaset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# create the dataset where to store the data
dataset = LeRobotDataset(
    repo_id="talos-mind/svla_so101_pickplace",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# initiate the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# connect the robot and teleoperator
robot.connect()
teleop.connect()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"recording episode {episode_idx + 1} of {NUM_EPISODES}")
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # reset the environment if not stoppiong or re-recording
    if (not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecored_episode"])):
        log_say(f"resetting environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            dataset=dataset,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )
    if events["rerecord_episode"]:
        log_say(f"rerecording episode {episode_idx + 1}")
        events["rerecord_episode"] = False
        events["exit_early"] = True
        dataset.clear_episode_buffer()
        continue
    dataset.save_episode()
    episode_idx += 1

# clean up
robot.disconnect()
teleop.disconnect()
dataset.push_to_hub()
log_say(f"dataset saved to hub")
log_say(f"dataset closed")
log_say(f"dataset saved to hub")