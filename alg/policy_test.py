import torch
import numpy as np
import os
import sys
from core import MLPActorCritic

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
sys.path.append(PARENT_DIR + '/utils/')

from task_envs import PnPNewRobotEnv
from env_wrappers import reconstruct_state, ActionNormalizer, ResetWrapper, TimeLimitWrapper


def load_final_policy(path_to_saved_policy):
    """
    Load your final trained policy

    Args:
        path_to_saved_policy (str): the path to your saved policy model

    Returns:
        The loaded policy model
    """
    obs_dim = 22
    act_dim = 4 

    dummy_obs_space = type('DummySpace', (), {'shape': (obs_dim,)})()
    dummy_act_space = type('DummySpace', (), {'shape': (act_dim,), 'high': np.ones(act_dim)})()

    policy_model = MLPActorCritic(dummy_obs_space, dummy_act_space, special_policy='awac', hidden_sizes=[256, 256])
    policy_model.load_state_dict(torch.load(path_to_saved_policy, map_location=torch.device('cpu')))
    policy_model.eval()

    return policy_model


def get_policy_action(state, saved_policy_model):
    """
    Get the action that the policy decides to take for the given environment state

    Args:
        state (dict): the environment state with keys: 'observation', 'achieved_goal', 'desired_goal'
        saved_policy_model: the policy model loaded using load_final_policy()

    Returns:
        action (np.array): the action chosen by the policy
    """
    obs_flat = reconstruct_state(state)
    obs_tensor = torch.as_tensor(obs_flat, dtype=torch.float32)
    with torch.no_grad():
        action = saved_policy_model.act(obs_tensor, deterministic=True)
    return action

if __name__ == "__main__":
    # Just testing if it works
    model_path= os.path.join("saved_policies", "model_1000.pt")

    print(f"Loading model from: {model_path}")
    policy = load_final_policy(model_path)

    env = PnPNewRobotEnv(render=True)
    env = ActionNormalizer(env)
    env = ResetWrapper(env)
    env = TimeLimitWrapper(env, max_steps=150)

    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0

    print("Running policy for one episode...")
    while not done:
        action = get_policy_action(obs, policy)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"Episode finished in {steps} steps with total reward: {total_reward:.2f}")