from pathlib import Path
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from typing import List

def make_delta_timestamps(delta_indices: List[int] | None, fps: int) -> List[float]:
    if delta_indices is None:
        return [0]
    return [i/fps for i in delta_indices]

output_directory = Path("output/robot_learning_tutorial/diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

# select your device
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_id = "lerobot/svla_so101_pickplace"

dataset_metadata = LeRobotDatasetMetadata(dataset_id)
features = dataset_to_policy_features(dataset_metadata.features)

output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}

cfg = DiffusionConfig(
    input_features=input_features,
    output_features=output_features,
)
policy = DiffusionPolicy(config=cfg)
preprocessor, postprocessor = make_pre_post_processors(policy_cfg=cfg,
                                                       dataset_stats=dataset_metadata.stats)

policy.train()
policy.to(device)

# to perform action chunking, act expects a given numer of actions as targets
delta_timestamps = {
    "observation.state": make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
    "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps)
}

# add image features if they are present
delta_timestamps |= {
    k: make_delta_timestamps(cfg.image_delta_indices, dataset_metadata.fps) for k in cfg.image_features
}

# instantiate the dataset
dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)

# create the optimizer and dataloader for offline training
optimizer = cfg.get_optimizer_preset().build(policy.parameters())
batch_size = 32
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, \
                                         pin_memory=device.type != "cpu", drop_last=True)
# number of trianing steps and log requency
training_steps = 1
log_freq = 1

step = 0
done = False
while not done:
    for batch in dataloader:
        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            print(f"[training] step {step}, loss: {loss.item():.4f}")
        step += 1
        if step >= training_steps:
            done = True
            break
policy.save_pretrained(output_directory)
preprocessor.save_pretrained(output_directory)
postprocessor.save_pretrained(output_directory)

# save all assets to the hub
model_id = "talos-thinking/robot_learning_diffusion_training_example_model"
policy.push_to_hub(model_id)
preprocessor.push_to_hub(model_id)
postprocessor.push_to_hub(model_id)
print(f"[training] policy pushed to the hub as {model_id}")
print(f"[training] preprocessor pushed to the hub as {model_id}")
print(f"[training] postprocessor pushed to the hub as {model_id}")
print(f"[training] training completed after {step} steps")
print(f"[training] training completed after {step} steps")