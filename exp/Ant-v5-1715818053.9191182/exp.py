# -*- coding: utf-8 -*-

import sdac
import gymnasium as gym
import argparse
import os
import numpy as np
import torch
from torch.nn import functional as F
import time

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="CartPole-v1")
parser.add_argument("--mujoco", action="store_true", default=False)
parser.add_argument("--send", action="store_true", default=False)
parser.add_argument("--local", action="store_true", default=False)
args = parser.parse_args()

g_exp = dict()

def Exp(env_id):
    def _f1(f):
        global g_exp
        def _f2():
            f(env_id)
        g_exp[env_id] = _f2
        return _f2
    return _f1


def GymDiscrete(env_id):
    algo = sdac.SDAC()
    algo.buffer_size = int(1e5)
    algo.total_timesteps = int(5e4)
    algo.hidden = 512
    algo.n_collect_data = 100
    algo.n_optimizer_step = 100
    algo.reward_norm = 0.001
    algo.n_atoms = 61
    algo.v_min = -30
    algo.v_max = 30
    algo.gamma = 1 - 1/30
    algo.learning_rate = 1e-3
    algo.learning_starts = 1e3
    algo.batch_size = 256

    class Env(sdac.Wrapper):
        def __init__(self):
            self.name = env_id
            self.path = "."
            self.env = gym.make(env_id)
            self.state_dim = self.env.observation_space.shape[0]
            self.action_dim = 1
            self.action_atoms = self.env.action_space.n
        
        def reset(self):
            obs, info = self.env.reset()
            return obs
        
        def step(self, act):
            obs, r, d, t, info = self.env.step(act[0])
            return obs, r, d, t
    
    class EvalEnv(Env):
        def __init__(self):
            super().__init__()
            self.env = gym.make(env_id, render_mode="rgb_array")
            self.env = gym.wrappers.RecordVideo(self.env, f"{self.path}/videos/", self.episode_trigger)
        
        def episode_trigger(self, episode_id):
            return episode_id % 100 == 0
    
    algo.env = Env()
    algo.eval_env = EvalEnv()

    return algo


@Exp("CartPole-v1")
def _f(env_id):
    algo = GymDiscrete(env_id)
    algo.batch_size = 64
    algo.beta = 0.1
    algo.train()


def GymContinuous(env_id, env1, env2):
    algo = sdac.SDAC()
    algo.buffer_size = int(1e5)
    algo.total_timesteps = int(5e4)
    algo.hidden = 2048
    algo.n_collect_data = 100
    algo.n_optimizer_step = 100
    algo.reward_norm = 0.001
    algo.n_atoms = 401
    algo.v_min = -200
    algo.v_max = 200
    algo.gamma = 1 - 1/200
    algo.learning_rate = 1e-4
    algo.learning_starts = 1e3
    algo.batch_size = 512

    class Env(sdac.Wrapper):
        def __init__(self):
            self.name = env_id
            self.path = "."
            self.env = env1
            self.state_dim = self.env.observation_space.shape[0]
            self.action_dim = self.env.action_space.shape[0]
            self.action_atoms = 51
        
        def reset(self):
            obs, info = self.env.reset()
            return obs
        
        def step(self, act):
            obs, r, d, t, info = self.env.step(self.to_float(act))
            return obs, r, d, t
    
    class EvalEnv(Env):
        def __init__(self):
            super().__init__()
            self.env = env2
            self.env = gym.wrappers.RecordVideo(self.env, f"{self.path}/videos/", self.episode_trigger)
        
        def episode_trigger(self, episode_id):
            return episode_id % 100 == 0
    
    algo.env = Env()
    algo.eval_env = EvalEnv()

    return algo


@Exp("BipedalWalkerHardcore-v3")
def BipedalWalkerHardcore(env_id):
    algo = sdac.SDAC()
    algo.buffer_size = int(1e6)
    algo.total_timesteps = int(2e7)
    algo.hidden = 2048
    algo.n_collect_data = 1000
    algo.n_optimizer_step = 120
    algo.auto_reward_norm = False
    algo.reward_norm = 330 / 1000
    algo.n_atoms = 51
    # algo.atoms_lst = [5] * 10 + [1] * 51
    algo.v_min = -300
    algo.v_max = 50
    algo.gamma = 1 - 1/50
    algo.learning_rate = 2.5 * 1e-4
    algo.learning_starts = 2e4
    algo.batch_size = 512
    algo.tau = 0.005
    algo.h1 = 0.5
    algo.policy_frequency = 2
    algo.random_seed()

    class Env(sdac.Wrapper):
        def __init__(self):
            self.name = env_id
            self.path = "."
            self.env = gym.make(env_id)
            self.state_dim = self.env.observation_space.shape[0]
            self.action_dim = self.env.action_space.shape[0]
            self.action_atoms = 51
        
        def reset(self):
            obs, info = self.env.reset()
            return obs
        
        def step(self, act):
            obs, r, d, t, info = self.env.step(self.to_float(act))
            t = d or t
            if r <= -100:
                d = True
            return obs, r, d, t
    
    class EvalEnv(Env):
        def __init__(self):
            super().__init__()
            self.env = gym.make(env_id, render_mode="rgb_array")
            self.env = gym.wrappers.RecordVideo(self.env, f"{self.path}/videos/", self.episode_trigger)
        
        def episode_trigger(self, episode_id):
            return episode_id % 100 == 0
    
    algo.env = Env()
    algo.eval_env = EvalEnv()

    algo.train()


def Panda(env_id, gamma=0.98):
    import panda_gym

    algo = sdac.SDAC()
    algo.buffer_size = int(1e6)
    algo.total_timesteps = int(2e7)
    algo.hidden = 2048
    algo.n_collect_data = 1
    algo.n_optimizer_step = 1
    algo.reward_norm = 1.0
    algo.auto_reward_norm = False
    algo.v_max = 0
    algo.v_min = - 1 / (1 - gamma)
    algo.n_atoms = 51
    algo.gamma = gamma
    algo.learning_rate = 1e-3
    algo.learning_starts = 1e4
    algo.batch_size = 1024
    algo.h1 = 0.5
    algo.tau = 0.01
    algo.her = True
    algo.her_batch_size = 1024
    algo.her_buffer_size = int(2e6)

    algo.noise_setting = ([1/20, 0, -1], [0.5, 0.3, 0.2])
    algo.n_optimizer_step = 10
    algo.n_collect_data = 50
    algo.batch_size = 1024
    algo.her = True
    algo.her_batch_size = 1024
    algo.learning_starts = 30000

    algo.random_seed()

    class Env(sdac.Wrapper):
        def __init__(self):
            self.name = env_id
            self.path = "."
            self.env = gym.make(env_id)
            dim = 0
            for k, b in self.env.observation_space.items():
                dim += b.shape[0]
            self.state_dim = dim
            self.action_dim = self.env.action_space.shape[0]
            self.action_atoms = 51
            self.tj = []
            self.cnt = 0
        
        def flatten_obs(self, obs):
            _obs = []
            for k in sorted(list(obs.keys())):
                _obs.append(obs[k])
            _obs = np.concatenate(_obs, axis=-1)
            return _obs
        
        def her(self):
            algo = self.sdac()
            device = algo.device
            while len(self.tj) > 0:
                goal = self.tj[-1][1]["achieved_goal"]
                end = -1
                for i, tpl in enumerate(self.tj):
                    _obs, _next_obs, _act, _info = tpl
                    _obs["desired_goal"] = goal
                    _next_obs["desired_goal"] = goal
                    _d = self.env.unwrapped.task.is_success(_next_obs["achieved_goal"], _next_obs["desired_goal"])
                    if _d:
                        end = i
                        break
                if end == 0:
                    end = -1
                _return = 0.0
                for i in range(end, -1, -1):
                    tpl = self.tj[i]
                    _obs, _next_obs, _act, _info = tpl
                    _obs["desired_goal"] = goal
                    _next_obs["desired_goal"] = goal
                    _r = self.env.unwrapped.task.compute_reward(_next_obs["achieved_goal"], _next_obs["desired_goal"], _info)
                    _d = self.env.unwrapped.task.is_success(_next_obs["achieved_goal"], _next_obs["desired_goal"])

                    _return = _r + algo.gamma * _return * _d
                
                    cursor = algo.her_mgr.add()
                    algo.her_obs[cursor]      = torch.tensor(self.flatten_obs(_obs)).float().to(device)
                    algo.her_next_obs[cursor] = torch.tensor(self.flatten_obs(_next_obs)).float().to(device)
                    algo.her_act[cursor]      = torch.tensor(_act).to(device)
                    algo.her_reward[cursor]   = torch.tensor(_r).float().to(device)
                    algo.her_done[cursor]     = torch.tensor(float(_d)).to(device)
                    algo.her_return[cursor]   = torch.tensor(float(_return)).to(device)
                    self.cnt += 1

                self.tj = self.tj[: max(end, 0)]
                # self.tj = []
            print("her:", self.cnt)
    
        def reset(self):
            obs, info = self.env.reset()
            self.tj = []
            self.obs = obs
            return self.flatten_obs(obs)
        
        def step(self, act):
            obs, r, d, t, info = self.env.step(self.to_float(act))
            self.obs = obs
            if not d:
                self.tj.append((self.obs, obs, act, info))
            if d or t:
                self.her()
            return self.flatten_obs(obs), r, d, t
    
    class EvalEnv(Env):
        def __init__(self):
            super().__init__()
            self.env = gym.make(env_id, render_mode="rgb_array")
            self.env = gym.wrappers.RecordVideo(self.env, f"{self.path}/videos/", self.episode_trigger)
        
        def episode_trigger(self, episode_id):
            return episode_id % 100 == 0
        
        def her(self):
            pass
    
    algo.env = Env()
    algo.eval_env = EvalEnv()

    return algo


@Exp("PandaPickAndPlace-v3")
def _f(env_id):
    algo = Panda(env_id)
    algo.train()


@Exp("PandaStack-v3")
def _f(env_id):
    algo = Panda(env_id)
    algo.train()


@Exp("PandaSlide-v3")
def _f(env_id):
    algo = Panda(env_id)
    algo.train()


@Exp("PandaPush-v3")
def _f(env_id):
    algo = Panda(env_id)
    algo.train()


@Exp("PandaReach-v3")
def _f(env_id):
    algo = Panda(env_id)
    algo.train()


def Fetch(env_id, el=100):
    import gymnasium_robotics

    algo = sdac.SDAC()
    algo.buffer_size = int(1e6)
    algo.total_timesteps = int(1e7)
    algo.hidden = 2048
    algo.n_collect_data = 1000
    algo.n_optimizer_step = 100
    algo.reward_norm = 1.0
    algo.auto_reward_norm = False
    algo.n_atoms = 1 + 2 * el
    algo.v_min = - el
    algo.v_max = el
    algo.gamma = 1 - 1/el
    algo.learning_rate = 1e-4
    algo.learning_starts = 1e4
    algo.batch_size = 512
    algo.h2 = -1.0
    algo.success = el
    algo.beta = 0.3

    class Env(sdac.Wrapper):
        def __init__(self):
            self.name = env_id
            self.path = "."
            self.env = gym.make(env_id)
            dim = 0
            for k, b in self.env.observation_space.items():
                dim += b.shape[0]
            self.state_dim = dim
            self.action_dim = self.env.action_space.shape[0]
            self.action_atoms = 51
            self.tj = []
        
        def flatten_obs(self, obs):
            _obs = []
            for k in sorted(list(obs.keys())):
                _obs.append(obs[k])
            _obs = np.concatenate(_obs, axis=-1)
            return _obs
        
        def her(self):
            algo = self.sdac()
            device = algo.device
            goal = self.tj[-1][1]["achieved_goal"]
            for _obs, _next_obs, _act, _info in self.tj:
                _obs["desired_goal"] = goal
                _next_obs["desired_goal"] = goal
                _r = self.env.unwrapped.compute_reward(_next_obs["achieved_goal"], _next_obs["desired_goal"], _info)
                _d = self.env.unwrapped.compute_terminated(_next_obs["achieved_goal"], _next_obs["desired_goal"], _info)
                cursor = algo.rb_mgr.add()
                algo.rb_obs[cursor]      = torch.tensor(self.flatten_obs(_obs)).float().to(device)
                algo.rb_next_obs[cursor] = torch.tensor(self.flatten_obs(_next_obs)).float().to(device)
                algo.rb_prob[cursor]     = F.one_hot(torch.tensor(_act), self.action_atoms).flatten().float().to(device)
                algo.rb_act[cursor]      = torch.tensor(_act).to(device)
                algo.rb_reward[cursor]   = torch.tensor(_r).float().to(device)
                algo.rb_done[cursor]     = torch.tensor(float(_d)).to(device)
                if _d:
                    break
    
        def reset(self):
            if len(self.tj) > 0:
                self.her()
            obs, info = self.env.reset()
            self.tj = []
            self.obs = obs
            return self.flatten_obs(obs)
        
        def step(self, act):
            obs, r, d, t, info = self.env.step(self.to_float(act))
            self.tj.append([self.obs, obs, act, info])
            self.obs = obs
            if d:
                self.tj = []
            return self.flatten_obs(obs), r, d, t
    
    class EvalEnv(Env):
        def __init__(self):
            super().__init__()
            self.env = gym.make(env_id, render_mode="rgb_array")
            self.env = gym.wrappers.RecordVideo(self.env, f"{self.path}/videos/", self.episode_trigger)
        
        def episode_trigger(self, episode_id):
            return episode_id % 100 == 0
        
        def her(self):
            pass
    
    algo.env = Env()
    algo.eval_env = EvalEnv()

    return algo


def MiniGrid(env_id):
    import minigrid

    algo = sdac.SDAC()
    algo.buffer_size = int(1e6)
    algo.total_timesteps = int(1e7)
    algo.hidden = 2048
    algo.n_collect_data = 1000
    algo.n_optimizer_step = 100
    algo.reward_norm = 1.0
    algo.auto_reward_norm = False
    algo.n_atoms = 201
    algo.v_min = 0
    algo.v_max = 1
    algo.gamma = 1
    algo.learning_rate = 1e-4
    algo.learning_starts = 1e4
    algo.batch_size = 512
    algo.beta = 0.3

    class Env(sdac.Wrapper):
        def __init__(self):
            self.name = env_id
            self.path = "."
            self.env = gym.make(env_id)
            obs = self.env.observation_space
            dim = obs["direction"].n
            dim += int(np.prod(obs["image"].shape) * 20)
            dim += 3000
            self.state_dim = dim
            self.action_dim = 1
            self.action_atoms = self.env.action_space.n
        
        def flatten_obs(self, obs):
            _obs = np.zeros((self.state_dim, ))
            ix = 0
            _obs[ix + obs["direction"]] = 1.0
            ix += self.env.observation_space["direction"].n
            h, w, c = self.env.observation_space["image"].shape
            image = obs["image"]
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        _obs[ix + image[i, j, k]] = 1
                        ix += 20
            mission = obs["mission"]
            for ch in mission:
                _obs[ix + ord(ch) % 50] = 1
                ix += 50
            return _obs
    
        def reset(self):
            obs, info = self.env.reset()
            self.obs = obs
            return self.flatten_obs(obs)
        
        def step(self, act):
            obs, r, d, t, info = self.env.step(act[0])
            self.obs = obs
            return self.flatten_obs(obs), r, d, t
    
    class EvalEnv(Env):
        def __init__(self):
            super().__init__()
            self.env = gym.make(env_id, render_mode="rgb_array")
            self.env = gym.wrappers.RecordVideo(self.env, f"{self.path}/videos/", self.episode_trigger)
        
        def episode_trigger(self, episode_id):
            return episode_id % 100 == 0
    
    algo.env = Env()
    algo.eval_env = EvalEnv()

    return algo
        

def MiniAtar(env_id):
    import minatar

    algo = sdac.SDAC()
    algo.buffer_size = int(1e6)
    algo.total_timesteps = int(1e7)
    algo.hidden = 2048
    algo.n_collect_data = 1000
    algo.n_optimizer_step = 100
    algo.reward_norm = 1.0
    algo.auto_reward_norm = False
    algo.n_atoms = 401
    algo.v_min = -200
    algo.v_max = 200
    algo.gamma = 1 - 1 / 200
    algo.learning_rate = 1e-4
    algo.learning_starts = 1e4
    algo.batch_size = 512
    algo.beta = 0.3

    class Env(sdac.Wrapper):
        def __init__(self):
            self.name = env_id
            self.path = "."
            self.env = gym.make(env_id)
            self.state_dim = int(np.prod(self.env.observation_space.shape))
            self.action_dim = 1
            self.action_atoms = self.env.action_space.n
        
        def flatten_obs(self, obs: np.ndarray):
            return obs.flatten().astype(np.float32)
    
        def reset(self):
            obs, info = self.env.reset()
            self.obs = obs
            self.result = 0.0
            self.cnt = 0
            return self.flatten_obs(obs)
        
        def step(self, act):
            obs, r, d, t, info = self.env.step(act[0])
            self.obs = obs
            self.result += r
            self.cnt += 1
            if d or t:
                algo = self.sdac()
                algo.reward_norm = max(algo.reward_norm, self.result / self.cnt)
            return self.flatten_obs(obs), r, d, t
    
    class EvalEnv(Env):
        def __init__(self):
            super().__init__()
            self.env = gym.make(env_id, render_mode="rgb_array")
            self.env = gym.wrappers.RecordVideo(self.env, f"{self.path}/videos/", self.episode_trigger)
        
        def episode_trigger(self, episode_id):
            return episode_id % 100 == 0
    
    algo.env = Env()
    algo.eval_env = EvalEnv()

    return algo


def MuJoCo(env_id):

    algo = sdac.SDAC()
    algo.buffer_size = int(1e6)
    algo.total_timesteps = int(2e7)
    algo.hidden = 2048
    algo.n_collect_data = 500
    algo.n_optimizer_step = 100
    algo.reward_norm = 10
    algo.auto_reward_norm = False
    algo.n_atoms = 51
    algo.v_min = -0
    algo.v_max = 50
    algo.gamma = 1 - 1 / 50
    algo.learning_rate = 7.3 * 1e-4
    algo.learning_starts = 2e4
    algo.batch_size = 512
    algo.observation_norm = False
    algo.tau = 0.005
    algo.h1 = 0.5

    class Env(sdac.Wrapper):
        def __init__(self):
            self.name = env_id
            self.path = "."
            self.env = gym.make(env_id)
            self.state_dim = self.env.observation_space.shape[0]
            self.action_dim = self.env.action_space.shape[0]
            self.action_atoms = 51
            self.max_score = - np.inf
            self.score = 0.0
    
        def reset(self):
            obs, info = self.env.reset()
            self.score = 0.0
            return obs
        
        def step(self, act):
            obs, r, d, t, info = self.env.step(self.to_float(act))
            self.score += r
            # if d or t:
            #     self.max_score = max(self.max_score, self.score)
            #     algo.reward_norm = max(algo.reward_norm, self.max_score/1000)
            return obs, r, d, t
    
    class EvalEnv(Env):
        def __init__(self):
            super().__init__()
            self.env = gym.make(env_id, render_mode="rgb_array")
            self.env = gym.wrappers.RecordVideo(self.env, f"{self.path}/videos/", self.episode_trigger)
        
        def episode_trigger(self, episode_id):
            return episode_id % 100 == 0
    
    algo.env = Env()
    algo.eval_env = EvalEnv()

    return algo


@Exp("Ant-v5")
def _f(env_id):
    algo = MuJoCo(env_id)
    algo.reward_norm = 10
    algo.train()

@Exp("HalfCheetah-v5")
def _f(env_id):
    algo = MuJoCo(env_id)
    algo.reward_norm = 20
    algo.train()

@Exp("Hopper-v5")
def _f(env_id):
    algo = MuJoCo(env_id)
    algo.reward_norm = 4
    algo.v_max = 200
    algo.gamma = 1 - 1 / 200
    algo.train()

@Exp("Swimmer-v5")
def _f(env_id):
    algo = MuJoCo(env_id)
    algo.reward_norm = 0.4
    algo.v_max = 200
    algo.gamma = 1 - 1 / 200
    algo.train()

@Exp("Walker2d-v5")
def _f(env_id):
    algo = MuJoCo(env_id)
    algo.reward_norm = 8
    algo.train()

@Exp("Humanoid-v5")
def _f(env_id):
    algo = MuJoCo(env_id)
    algo.reward_norm = 10
    algo.train()


def Bullet(env_id):
    import pybullet_envs_gymnasium
    algo = MuJoCo(env_id)

    return algo



sh = """#! /bin/bash
#SBATCH -p gpu_ai
#SBATCH -n 1
#SBATCH -o %J.out
#SBATCH -G 1

python -u ./exp.py --exp {exp}
"""
if args.send:
    exp_dir = args.exp + "-" + str(time.time())
    os.system("mkdir exp")
    os.system(f"rm -rf exp/{exp_dir}")
    os.system(f"mkdir exp/{exp_dir}")
    os.system(f"cp sdac.py exp/{exp_dir}/sdac.py")
    os.system(f"cp exp.py exp/{exp_dir}/exp.py")
    with open(f"exp/{exp_dir}/run.sh", "wb") as f:
        f.write(sh.replace("{exp}", args.exp).encode("utf-8"))
    os.system(f"python upload.py exp/{exp_dir}")
    os.system(f"python launch.py exp/{exp_dir}")
    os.system(f"python tail.py {exp_dir}")
elif args.mujoco:
    for exp in ["Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Swimmer-v5", "Walker2d-v5", "Humanoid-v5"]:
        exp_dir = exp + "-" + str(time.time())
        os.system("mkdir exp")
        os.system(f"rm -rf exp/{exp_dir}")
        os.system(f"mkdir exp/{exp_dir}")
        os.system(f"cp sdac.py exp/{exp_dir}/sdac.py")
        os.system(f"cp exp.py exp/{exp_dir}/exp.py")
        with open(f"exp/{exp_dir}/run.sh", "wb") as f:
            f.write(sh.replace("{exp}", exp).encode("utf-8"))
        os.system(f"python upload.py exp/{exp_dir}")
        os.system(f"python launch.py exp/{exp_dir}")
elif args.local:
    exp_dir = args.exp + "-" + str(time.time())
    os.system("mkdir exp")
    os.system(f"rm -rf exp/{exp_dir}")
    os.system(f"mkdir exp/{exp_dir}")
    os.system(f"cp sdac.py exp/{exp_dir}/sdac.py")
    os.system(f"cp exp.py exp/{exp_dir}/exp.py")
    with open(f"exp/{exp_dir}/run.sh", "wb") as f:
        f.write(sh.replace("{exp}", args.exp).encode("utf-8"))
    os.system(f"cd exp/{exp_dir} && python exp.py --exp {args.exp}")
else:
    g_exp[args.exp]()
