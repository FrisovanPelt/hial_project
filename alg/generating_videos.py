import numpy as np
import gym
import panda_gym
import cv2
import os
import sys
from time import sleep

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
sys.path.append(PARENT_DIR + '/utils/')

from task_envs import PnPNewRobotEnv
from env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper


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
   
    if init is not None:
        obs = env.reset(whether_random=False, object_pos=init)
    else:
        obs = env.reset(whether_random=True)

    while not done and step < len(action_sequence):
        action = action_sequence[step]
        obs, reward, done, info = env.step(action)

        img_tuple = env.render(mode='rgb_array')
        img = np.array(img_tuple, dtype=np.uint8)
        img = img.reshape((height, width, 4))

        # Convert to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        out.write(img)
        print(f"✅ Writing frame {step} to video...")

        step += 1

    out.release()
    print(f"✅ Video saved: {video_path}")

def generate_expert_videos():
    """Generates 20 expert demonstration video clips."""
    demo_path = PARENT_DIR + '/demo_data/PickAndPlace/'
    demos = prepare_demo_pool(demo_path)

    env = PnPNewRobotEnv(render=True)
    env = ActionNormalizer(env)
    env = ResetWrapper(env=env)
    env = TimeLimitWrapper(env=env, max_steps=150)

    for i, demo in enumerate(demos[:20]):  # First 20 demos
        initial_object_pos = demo['state_trajectory'][0][7:10]  # grab the initial position
        record_video(env, demo['action_trajectory'], f'expert_demo_{i + 1}', initial_object_pos)


def generate_random_videos():
    """Generates 10 videos of random agent behavior."""
    env = PnPNewRobotEnv(render=True)
    env = ActionNormalizer(env)
    env = ResetWrapper(env=env)
    env = TimeLimitWrapper(env=env, max_steps=150)

    for i in range(10):
        action_sequence = [env.action_space.sample() for _ in range(150)]
        record_video(env, action_sequence, f'random_demo_{i + 1}')


if __name__ == '__main__':
    generate_expert_videos()
    generate_random_videos()