import multiprocessing as mp
from queue import Empty, Full
import torch
import torch.optim as optim

from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import ReplayBuffer

LOG_EVERY = 10
SEND_EVERY = 10

def run_learner(
        transitions_queue: mp.Queue,
        parameters_queue: mp.Queue,
        shutdown_event,
        policy_learner: SACPolicy,
        online_buffer: ReplayBuffer, 
        offline_buffer: ReplayBuffer,
        lr: float = 3e-4,
        batch_size: int = 32,
        device: torch.device = "mps"
):
    policy_learner.train()
    policy_learner.to(device)

    # create adam optimizer from scratch  - simple and clean
    optimizer = optim.Adam(policy_learner.parameters(), lr=lr)

    print(f"[learner] online buffer capacity: {online_buffer.capacity}")
    print(f"[learner] offline buffer capacity: {offline_buffer.capacity}")

    training_step = 0
    while not shutdown_event.is_set():
        try:
            transitions = transitions_queue.get(timeout=1)
            for transition in transitions:
                online_buffer.add(**transition)
                is_intervention = transition.get("complementary_info", {}).get("is_intervention", False)
                if is_intervention:
                    offline_buffer.add(**transition)
                    print(f"[learner] added intervention transition to offline buffer offline buffer size: {len(offline_buffer)}")
        except Empty:
            pass

        if len(online_buffer) >= policy_learner.config.online_step_before_learning:
            # sample from the online buffer (autonomous + human data)
            online_batch = online_buffer.sample(batch_size // 2)
            # sample from offline buffer (human demostrations only)
            offline_batch = offline_buffer.sample(batch_size // 2)
            # combine batches - this is the key hil-serl mechanism
            batch = {}
            for key in online_batch.keys():
                if key in offline_batch:
                    batch[key] = torch.cat([online_batch[key], offline_batch[key]], dim=0)
                else:
                    batch[key] = online_batch[key]
            loss, _ = policy_learner.forward(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_step += 1

            if training_step % LOG_EVERY == 0:
                print(f"[learner] training step {training_step}, loss: {loss.item():.4f}, online buffer size: {len(online_buffer)}, offline buffer size: {len(offline_buffer)}")
            
            # send updated parameters to the actor after 10 trinaing steps
            if training_step % SEND_EVERY == 0:
                try:
                    state_dict = {k: v.cpu() for k, v in policy_learner.state_dict().items()}
                    parameters_queue.put(state_dict)
                    print(f"[learner] sent policy parameters to actor")
                except Full:
                    pass
    print("[learner] process finished")

