from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import serve

host = "0.0.0.0"
port = 5000
policy_id = "fracapauano/robot_learning_act_example_model"
policy_server_cfg = PolicyServerConfig(host=host, port=port, policy_id=policy_id)
serve(policy_server_cfg)