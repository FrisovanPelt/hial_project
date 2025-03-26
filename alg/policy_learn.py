import numpy as np
from gym.envs.registration import register
from gym.spaces import Box
import os
import sys
import torch
from awac import AWAC
from core import MLPActorCritic
from pref_learn import feature_function, prepare_demo_pool
import aprel

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
sys.path.append(PARENT_DIR + '/utils/')

from task_envs import PnPNewRobotEnv
from env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper, reconstruct_state

register(
    id='PnPNewRobotEnv-v0',  
    entry_point='task_envs:PnPNewRobotEnv',  # Ensure correct module path
    max_episode_steps=150,
)

# Load learned reward weights
weights_path = 'final_feature_weights.csv'
weights = np.loadtxt(weights_path, delimiter=',', skiprows=1).flatten()

# Learned reward function
def learned_reward_fn(traj):
    """
    Computes the reward for an entire trajectory of (state, action) pairs.
    """
    feat = feature_function(traj)
    return np.dot(weights, feat)

# Setup env
def make_env():
    env = PnPNewRobotEnv(render=False)
    env = ActionNormalizer(env)
    env = ResetWrapper(env)
    env = TimeLimitWrapper(env, max_steps=150)
    
    # Manually override observation_space with the flattened version
    obs_dim = 19 + 3  # len(observation) + len(achieved_goal)
    env.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    return env

# Load expert demos
demo_path = PARENT_DIR + '/demo_data/PickAndPlace/'
demos_raw = prepare_demo_pool(demo_path)

# # Convert states to flattened format
# for demo in demos_raw:
#     state = demo['state_trajectory']
#     next_state = demo['next_state_trajectory']

#     print(f"state: {len(state)}")
#     print(f"action: {type(state)}")
#     state = reconstruct_state(state)
#     next_state = reconstruct_state(next_state)

# Create and train agent
agent = AWAC(
    env_fn=make_env,
    actor_critic=MLPActorCritic,
    ac_kwargs=dict(hidden_sizes=[256, 256]),
    steps_per_epoch=1000,
    epochs=500,
    replay_size=int(1e6),
    batch_size=256,
    update_after=1000,
    update_every=50,
    num_test_episodes=10,
    max_ep_len=150,
)

print("Populating replay buffer with expert demonstrations...")
agent.populate_replay_buffer(demos_raw)

print("Starting AWAC training with learned reward function...")
agent.run(learned_reward_fn)
