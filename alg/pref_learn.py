import numpy as np
import gym
import panda_gym
from gym.envs.registration import register
import cv2
import csv 
import os
import sys
import aprel
from time import sleep

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
sys.path.append(PARENT_DIR + '/utils/')

from task_envs import PnPNewRobotEnv
from env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper

register(
    id='PnPNewRobotEnv-v0',  
    entry_point='task_envs:PnPNewRobotEnv',  
    max_episode_steps=150,
)


def feature_function(traj):
    """Returns the features of the given trajectory.

    Args:
        traj: List of state-action tuples, e.g. [(state0, action0), (state1, action1), ...]

    Returns:
        features: a numpy vector corresponding the features of the trajectory
    
    Observation: 
    # 0-2: End-effector position
    
    # 3-5: End-effector velocity

    # 6: Gripper width

    # 7-9: Object (banana) position

    # 10-12: Object orientation (probably quaternion or Euler)

    # 13-15: Object linear velocity

    # 16-18: Object angular velocity
    
    """
    
    states = np.array([pair[0] for pair in traj])
    
    #  Extract components from the state 
    ee_pos = states[:, 0:3]     # End-effector position
    ee_vel = states[:, 3:6]     # End-effector velocity
    obj_pos = states[:, 7:10]   # Banana position
    goal_pos = states[:, -3:]   # Goal position (assumed at end)

    #  Feature 1: Mean end-effector speed 
    # reflects how fast the robot was moving on average
    #distinguish smooth vs jittery
    ee_speed = np.linalg.norm(ee_vel, axis=1)
    mean_ee_speed = np.mean(ee_speed)

    #  Feature 2: Max banana height (Z) 
    # determines if banana was picked up (thrown in air accidentally?)
    max_obj_z = np.max(obj_pos[:, 2])

    #  Feature 3: Lifted ratio (above 2.5 cm) 
    # time banan lifeted: picked and carreid or dragged/ dropped / thrown
    lifted_ratio = np.mean(obj_pos[:, 2] > 0.025)

    # -- Feature 4: Final banana-to-goal distance 
    # how clsoe banana ended up
    final_obj_pos = obj_pos[-1]
    final_goal_pos = goal_pos[-1]
    final_dist_obj_goal = np.linalg.norm(final_obj_pos - final_goal_pos)

    #  Feature 5: Minimum banana-to-goal distance over time 
    # closets banana got
    dists_obj_goal = np.linalg.norm(obj_pos - goal_pos, axis=1)
    min_dist_obj_goal = np.min(dists_obj_goal)

    #  Feature 6: Average banana-to-gripper distance 
    # how close gripper was to banana overall
    avg_dist_obj_gripper = np.mean(np.linalg.norm(obj_pos - ee_pos, axis=1))

    # Combininign into final np arry feature for output
    features = np.array([
        mean_ee_speed,
        max_obj_z,
        lifted_ratio,
        final_dist_obj_goal,
        min_dist_obj_goal,
        avg_dist_obj_gripper
    ])

    #  Normalising 
    # done to provide equal influencing to account for differences in scale 
    
    mean_vec = np.array([0.27620503, 0.11339422, 0.25267853, 0.23587602, 0.17840652, 0.29698574])
    std_vec =  np.array([0.28688117, 0.06822429, 0.18223242, 0.08579391, 0.06190555, 0.12427733])

    normed_features = (features - mean_vec) / std_vec

    return normed_features

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
    for i, demo in enumerate(demos[:20]):  # First 20 demos
        init = demo['state_trajectory'][0][7:10]  # grab the initial position
        traj = record_video(env, demo['action_trajectory'], f'demo_{i + 1}', init)
        trajectories.append(traj)

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
        traj = record_video(env, action_sequence, f'demo_{i + 21}')
        trajectories.append(traj)
    
    return trajectories

def convert_to_aprel_trajectory(traj_data_list, env):
    """Converts a list of recorded trajectory data to a list of APReL Trajectory objects."""

    aprel_trajectories = []
    
    for i, traj_data in enumerate(traj_data_list):
        if not isinstance(traj_data, dict):
            raise ValueError(f"Expected a dictionary for trajectory {i}, but got {type(traj_data)}")
        
        if "state_trajectory" not in traj_data or "action_trajectory" not in traj_data:
            raise ValueError(f"Trajectory {i} is missing 'state_trajectory' or 'action_trajectory' keys.")

        states = []
        for state_dict in traj_data["state_trajectory"]:
            states.append(state_dict["observation"])
        states = np.array(states)
        actions = np.array(traj_data["action_trajectory"])
        
        if len(states) != len(actions):
            raise ValueError(f"Mismatch in length of states ({len(states)}) and actions ({len(actions)}) in trajectory {i}.")
        
        # Construct video file path
        video_filename = f"demo_{i + 1}.mp4"  # Assuming expert demos are named as such
        video_path = os.path.join("videos", video_filename)

        # Create APReL Trajectory object
        trajectory = aprel.Trajectory(env, list(zip(states, actions)), video_path)
        
        aprel_trajectories.append(trajectory)

    return aprel_trajectories



if __name__ == '__main__':

    env_name = 'PnPNewRobotEnv'
    gym_env = gym.make(env_name)

    env = aprel.Environment(gym_env, feature_function)

    trajectories = generate_expert_videos() + generate_random_videos()

    aprel_trajectories = aprel.TrajectorySet(convert_to_aprel_trajectory(trajectories, env))

    features_dim = len(aprel_trajectories[0].features)

    query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(aprel_trajectories)

    true_user = aprel.HumanUser(delay=0.5)

    params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}
    user_model = aprel.SoftmaxUser(params)

    belief = aprel.SamplingBasedBelief(user_model, [], params)
    print('Estimated user parameters: ' + str(belief.mean))

    query = aprel.PreferenceQuery(aprel_trajectories[:2])

    for query_no in range(10):
        queries, objective_values = query_optimizer.optimize('mutual_information', belief, query)
        # queries and objective_values are lists even when we do not request a batch of queries.
        print('Objective Value: ' + str(objective_values[0]))

        responses = true_user.respond(queries[0])
        belief.update(aprel.Preference(queries[0], responses[0]))
        print('Estimated user parameters: ' + str(belief.mean))

    # Save the weights to a CSV file
    weights = belief.mean['weights']
    filename = 'final_feature_weights.csv'

    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Weight'])  # Write header
            for weight in weights:
                writer.writerow([weight])
        print(f"Weights saved to {filename}")
    except Exception as e:
        print(f"Error saving weights: {e}")

    print("Reward weights saved successfully.")