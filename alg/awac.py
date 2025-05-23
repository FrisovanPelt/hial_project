# CITATION:
# @software{Sikchi_pytorch-AWAC,
# author = {Sikchi, Harshit and Wilcox, Albert},
# doi = {10.5281/zenodo.5121023},
# title = {{pytorch-AWAC}},
# url = {https://github.com/hari-sikchi/AWAC}
# }
from copy import deepcopy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
import gym
import time
import core as core
import torch.nn.functional as F
import os
import warnings
import sys
from termcolor import colored

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
sys.path.append(PARENT_DIR + '/utils/')

from env_wrappers import reconstruct_state


device = torch.device("cpu")

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, idxs=None):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class AWAC:

    def __init__(self, env_fn, test_env_fn, actor_critic=core.MLPActorCritic,
                 ac_kwargs=dict(),
                 seed=0,
                 steps_per_epoch=100,
                 epochs=10000,
                 replay_size=int(2000000),
                 gamma=0.99,
                 polyak=0.995,
                 lr=3e-4,
                 p_lr=3e-4,
                 alpha=0.0,
                 batch_size=1024,
                 start_steps=10000,
                 update_after=0,
                 update_every=50,
                 num_test_episodes=10,
                 max_ep_len=1000,
                 save_freq=1,
                 algo='SAC'):
        """
        Soft Actor-Critic (SAC)


        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with an ``act`` 
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of 
                observations as inputs, and ``q1`` and ``q2`` should accept a batch 
                of observations and a batch of actions as inputs. When called, 
                ``act``, ``q1``, and ``q2`` should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                            | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                            | of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current 
                                            | estimate of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ===========  ================  ======================================

                Calling ``pi`` should return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                            | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                            | actions in ``a``. Importantly: gradients
                                            | should be able to flow back into ``a``.
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
                you provided to SAC.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target 
                networks. Target networks are updated towards main networks 
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow 
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
                close to 1.)

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to 
                inverse of reward scale in the original SAC paper.)

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.

            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long 
                you wait between updates, the ratio of env steps to gradient steps 
                is locked to 1.

            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

            """
        
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env = env_fn()
        self.test_env = test_env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space, self.env.action_space,
                               special_policy='awac', **ac_kwargs)
        self.ac_targ = actor_critic(self.env.observation_space, self.env.action_space,
                                    special_policy='awac', **ac_kwargs)
        self.ac_targ.load_state_dict(self.ac.state_dict())
        self.gamma = gamma

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                          size=replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.algo = algo

        self.p_lr = p_lr
        self.lr = lr
        self.alpha = 0
        # # Algorithm specific hyperparams

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.p_lr, weight_decay=1e-4)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.update_after = update_after
        self.update_every = update_every
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.polyak = polyak
        # Set up model saving
        print("Running Offline RL algorithm: {}".format(self.algo))

# Changed to use the expert demos
    def populate_replay_buffer(self, demos):
        """
        Load expert demonstrations into the replay buffer.
        Args:
            demos: a list of expert trajectories, each trajectory is a dict with keys
                'state_trajectory', 'action_trajectory', 'reward_trajectory',
                'next_state_trajectory', 'done_trajectory'.
        """
        for traj in demos:
            states = traj['state_trajectory']
            actions = traj['action_trajectory']
            rewards = traj['reward_trajectory']
            next_states = traj['next_state_trajectory']
            dones = traj['done_trajectory']

            for s, a, r, s2, d in zip(states, actions, rewards, next_states, dones):
                self.replay_buffer.store(s, a, r, s2, d)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']

        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        v_pi = torch.min(q1_pi, q2_pi)

        beta = 2
        q1_old_actions = self.ac.q1(o, data['act'])
        q2_old_actions = self.ac.q2(o, data['act'])
        q_old_actions = torch.min(q1_old_actions, q2_old_actions)

        adv_pi = q_old_actions - v_pi
        weights = F.softmax(adv_pi / beta, dim=0)
        policy_logpp = self.ac.pi.get_logprob(o, data['act'])
        loss_pi = (-policy_logpp * len(weights) * weights.detach()).mean()

        # Useful info for logging
        pi_info = dict(LogPi=policy_logpp.detach().numpy())

        return loss_pi, pi_info
    
    # Added to track the amount of successful banana placements
    def evaluate_policy(self):
        """Evaluate current policy for 10 runs, return success rate."""
        success_count = 0
        for _ in range(10):
            obs = self.test_env.reset()
            obs_flat = reconstruct_state(obs)
            done = False
            for _ in range(self.max_ep_len):
                act = self.get_action(obs_flat, deterministic=True)
                next_obs, _, done, info = self.test_env.step(act)
                obs_flat = reconstruct_state(next_obs)
                if done:
                    if 'is_success' in info and info['is_success']:
                        success_count += 1
                    break
        return success_count / 10

    def update(self, data, update_timestep):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        # o_flat = reconstruct_state(o)
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = self.test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1

    # Our implementation of the run function, using 50k instead of 500k steps because of time constraints and not getting a successful policy
    def run(self, learned_reward_fn, max_env_steps=50_000, save_path="saved_policies"):
        total_steps = 0
        step_checkpoints = []
        success_rates = []

        os.makedirs(save_path, exist_ok=True)
        print("Populating replay buffer with expert demonstrations...")

    # Training loop
        while total_steps < max_env_steps:
            obs = self.env.reset()
            done = False
            episode = []

            while not done and len(episode) < self.max_ep_len:
                obs_flat = reconstruct_state(obs)
                act = self.get_action(obs_flat, deterministic=False)
                next_obs, _, done, info = self.env.step(act)

                episode.append((obs, act, next_obs, done))
                obs = next_obs
                total_steps += 1

                # Save model every 1000 steps
                if total_steps % 1000 == 0:
                    model_path = os.path.join(save_path, f"model_{total_steps}.pt")
                    torch.save(self.ac.state_dict(), model_path)
                    print(f"Saved model to {model_path}")

            # Calculates the total reward 
            traj = [(reconstruct_state(s), a) for (s, a, _, _) in episode]
            total_reward = learned_reward_fn(traj)
            avg_reward = total_reward / len(episode)
            print(f"Reward: {total_reward}")

            # Stores episode in replay buffer
            for (s, a, s2, d) in episode:
                s_flat = reconstruct_state(s)
                s2_flat = reconstruct_state(s2)
                self.replay_buffer.store(s_flat, a, avg_reward, s2_flat, d)

            for _ in range(self.update_every):
                batch = self.replay_buffer.sample_batch(self.batch_size)
                self.update(data=batch, update_timestep=total_steps)

            success_rate = self.evaluate_policy()
            print(f"Evaluation after {total_steps} steps: Success Rate = {success_rate:.2f}")
            step_checkpoints.append(total_steps)
            success_rates.append(success_rate)

        print("Training complete.")

        # Shows the plot of the success rate over time, saved with the saved_policies
        plt.figure(figsize=(8, 5))
        plt.plot(step_checkpoints, success_rates, marker='o')
        plt.xlabel("Environment Steps")
        plt.ylabel("Average Success Rate")
        plt.title("Policy Learning Curve")
        plt.grid(True)
        plt.savefig(os.path.join(save_path, "learning_curve.png"))
        plt.show()
