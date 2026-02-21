import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from torch.utils.data import DataLoader

delta_timestamps = {
    "observation.images.wrist_camera": [-0.2, -0.1, 0.0]
    # 0.2 and 0.1 seconds before each frame
}  

# static loaded dataset, very memory intensive
dataset = LeRobotDataset(
    path="lerobot/svla_so101_pickplace",
    delta_timestamps=delta_timestamps,
)

# streaming dataset, very memory efficient
dataset = StreamingLeRobotDataset(
    path="lerobot/svla_so101_pickplace",
    delta_timestamps=delta_timestamps,
)

batch_size = 16

dataloader: DataLoader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=batch_size)

num_epochs = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

for epoch in range(num_epochs):
    for batch in dataloader:
        observation = batch["observation.state"].to(device)
        action = batch["action"].to(device)
        images = batch["observation.images.wrist_camera"].to(device)

        # feed the data to the model for training