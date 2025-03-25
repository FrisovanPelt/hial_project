import numpy as np
import gym
import panda_gym
import cv2
import os
import sys
from aprel.learning import PreferenceBasedRewardLearning
from aprel import Trajectory, Environment, Feature
from time import sleep

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
sys.path.append(PARENT_DIR + '/utils/')

from task_envs import PnPNewRobotEnv
from env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper

class RobotEnv(Environment):
    def __init__(self, feature_func):
        super().__init__()
        self.feature_func = feature_func

    def features(self, trajectory):
        return self.feature_func(trajectory)

def feature_function(trajectory):
     # Filtering out invalid rows (inf or NaN) 
    valid_mask = np.all(np.isfinite(trajectory[:, [0,1,2, 7,8,9, 19,20,21]]), axis=1)
    trajectory = trajectory[valid_mask]
    if len(trajectory) == 0:
        return np.zeros(6)  # fallback if all rows were invalid

    # Extracting the positions from state 
    ee_positions = trajectory[:, 0:3]    # End-effector (robot hand)
    obj_positions = trajectory[:, 7:10]  # Banana
    goal_positions = trajectory[:, 19:22]  # Plate /goa
    obj_z = obj_positions[:, 2]  # z-position of banana

    #  ---features

    # Final distance from banana to goal
    final_obj = obj_positions[-1]
    final_goal = goal_positions[-1]
    final_dist_to_goal = np.linalg.norm(final_obj - final_goal)

    # the closest the banana got to the plate 
    min_dist_to_goal = np.min(np.linalg.norm(obj_positions - goal_positions, axis=1))

    #  Average distance from gripper to banana - small: hovers near . large :probably not anywhere clsoe most the time
    avg_grip_obj_dist = np.mean(np.linalg.norm(ee_positions - obj_positions, axis=1))

    #  Max height of the banana - kinda tells if it was ever picked up
    max_obj_z = np.max(obj_z)

    # Ratio of time banana was lifted above a threshold - kinda tells if it was ever actually held or maybe thrown /dropped 
    lift_threshold = 0.025
    held_ratio = np.sum(obj_z > lift_threshold) / len(obj_z)

    #  Binary success (if final distance is within 5cm)
    success = 1 if final_dist_to_goal < 0.05 else 0

    return np.array([
        final_dist_to_goal,
        min_dist_to_goal,
        avg_grip_obj_dist,
        max_obj_z,
        held_ratio,
        success
    ])

def prepare_demo_pool(demo_path):
    """Load the expert demonstration data into a structured format."""
    state_traj = np.genfromtxt(demo_path + 'state_traj.csv', delimiter=' ')
    action_traj = np.genfromtxt(demo_path + 'action_traj.csv', delimiter=' ')
    next_state_traj = np.genfromtxt(demo_path + 'next_state_traj.csv', delimiter=' ')
    reward_traj = np.genfromtxt(demo_path + 'reward_traj.csv', delimiter=' ')
    done_traj = np.genfromtxt(demo_path + 'done_traj.csv', delimiter=' ')

    reward_traj = np.reshape(reward_traj, (-1, 1))
    done_traj = np.reshape(done_traj, (-1, 1))

    starting_ids = [i for i in range(state_traj.shape[0]) if state_traj[i][0] == np.inf]
    total_demo_num = len(starting_ids)

    demos = []
    for i in range(total_demo_num):
        start_step_id = starting_ids[i]
        end_step_id = starting_ids[i + 1] if i < total_demo_num - 1 else state_traj.shape[0]

        demo = {
            'state_trajectory': state_traj[(start_step_id + 1):end_step_id, :],
            'action_trajectory': action_traj[(start_step_id + 1):end_step_id, :],
            'next_state_trajectory': next_state_traj[(start_step_id + 1):end_step_id, :],
            'reward_trajectory': reward_traj[(start_step_id + 1):end_step_id, :],
            'done_trajectory': done_traj[(start_step_id + 1):end_step_id, :]
        }
        demos.append(demo)

    return demos

def record_video(env, action_sequence, filename, init=None):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = f'videos/{filename}.mp4'

    height = 480
    width = 720

    os.makedirs('videos', exist_ok=True)
    out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

    step = 0
    done = False
    
    trajectory = {
        'state_trajectory': [],
        'action_trajectory': [],
        'reward_trajectory': [],
        'next_state_trajectory': [],
        'done_trajectory': []
    }

    if init is not None:
        obs = env.reset(whether_random=False, object_pos=init)
    else:
        obs = env.reset(whether_random=True)

    while not done and step < len(action_sequence):
        action = action_sequence[step]
        next_obs, reward, done, info = env.step(action)

        # Save trajectory data
        trajectory['state_trajectory'].append(obs)
        trajectory['action_trajectory'].append(action)
        trajectory['reward_trajectory'].append(reward)
        trajectory['next_state_trajectory'].append(next_obs)
        trajectory['done_trajectory'].append(done)

        img_tuple = env.render(mode='rgb_array')
        img = np.array(img_tuple, dtype=np.uint8)
        img = img.reshape((height, width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)
        
        obs = next_obs
        step += 1

    out.release()
    print(f"✅ Video saved: {video_path}")
    
    # Convert lists to numpy arrays for consistency
    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])
    
    return trajectory

def generate_expert_videos():
    """Generates 20 expert demonstration video clips."""
    demo_path = PARENT_DIR + '/demo_data/PickAndPlace/'
    demos = prepare_demo_pool(demo_path)

    env = PnPNewRobotEnv(render=True)
    env = ActionNormalizer(env)
    env = ResetWrapper(env=env)
    env = TimeLimitWrapper(env=env, max_steps=150)

    trajectories = []
    for i, demo in enumerate(demos[:10]):  # First 20 demos
        init = demo['state_trajectory'][0][7:10]  # grab the initial position
        traj = record_video(env, demo['action_trajectory'], f'expert_demo_{i + 1}', init)
        aprel_traj = convert_to_aprel_trajectory(traj)
        trajectories.append(aprel_traj)

    return trajectories

def generate_random_videos():
    """Generates 10 videos of random agent behavior."""
    env = PnPNewRobotEnv(render=True)
    env = ActionNormalizer(env)
    env = ResetWrapper(env=env)
    env = TimeLimitWrapper(env=env, max_steps=150)

    trajectories = []
    for i in range(10):
        action_sequence = [env.action_space.sample() for _ in range(150)]
        traj = record_video(env, action_sequence, f'random_demo_{i + 1}')
        aprel_traj = convert_to_aprel_trajectory(traj)
        trajectories.append(aprel_traj)
    
    return trajectories

def convert_to_aprel_trajectory(traj_data):
    traj = [
        (
            {
                "observation": s[:18],
                "achieved_goal": s[7:10],
                "desired_goal": np.array([0.0, -0.15, 0.02]),
            },
            a,
            r,
            {
                "observation": ns[:18],
                "achieved_goal": ns[7:10],
                "desired_goal": np.array([0.0, -0.15, 0.02]),
            },
            d,
        )
        for s, a, r, ns, d in zip(
            traj_data["state_trajectory"],
            traj_data["action_trajectory"],
            traj_data["reward_trajectory"],
            traj_data["next_state_trajectory"],
            traj_data["done_trajectory"],
        )
    ]

    return traj


if __name__ == '__main__':
    env = RobotEnv(feature_function)

    # trajectories = generate_expert_videos() + generate_random_videos()
    trajectories = generate_expert_videos()

    reward_learning = PreferenceBasedRewardLearning(env)

    # Generate queries for human feedback
    queries, _ = reward_learning.generate_queries(
        trajectories, 
        query_type='preference', 
        num_queries=10
    )

    # Collect human preferences interactively
    responses = reward_learning.collect_responses(queries)

    # Recover reward function based on human preferences
    reward_learning.train(responses)

    # Save the learned reward feature weights
    learned_weights = reward_learning.get_reward_weights()
    np.savetxt('final_feature_weights.csv', learned_weights, delimiter=',', fmt='%f')

    print("Reward weights saved successfully.")


    # generate_expert_videos()
    # generate_random_videos()