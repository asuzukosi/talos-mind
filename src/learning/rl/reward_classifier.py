import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig

# device to use for training
device = "cuda" if torch.cuda.is_available() else "cpu"
# load the dataset used for training
repo_id = "lerobot/example_hil_serl_dataset"
dataset = LeRobotDataset(repo_id=repo_id)

# configure the policy to extract features from the image frames
camera_keys = dataset.meta.camera_keys

rc_config = RewardClassifierConfig(
    num_cameras=len(camera_keys),
    device=device,
    model_name="microsoft/resnet-18"
    )

# make policy, preprocessor, and optimer
policy = make_policy(rc_config, ds_meta=dataset.meta)
optimizer = rc_config.get_optimizer_preset().build(policy.parameters())
preprocessor, _ = make_pre_post_processors(policy_cfg=rc_config, dataset_meta=dataset.meta.stats)


# load reward classifer
classifier_id = "lerobot/reward_classifier_hil_serl_example"
# initiate a dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    total_accuracy = 0

    for batch in dataloader:
        # preprocess the batch and move it to the correct device
        batch = preprocessor(batch)
        # forward pass
        loss, output_dict = policy.forward(batch)
        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += output_dict["accuracy"]
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    print(f"epoch {epoch + 1} - loss: {avg_loss:.4f} - accuracy: {avg_accuracy:.4f}")

print(f"training complete")

policy.push_to_hub(classifier_id)