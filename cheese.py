import numpy as np
import gymnasium as gym

class TrapCheese(gym.Env):
    """
    陷阱奶酪问题

    小鼠现在有三个选择，
    往左走，有50%几率得到奶酪
    往中间走，掉进陷阱，死亡
    往右走，有50%几率得到奶酪
    """

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1, ))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1, ))

    def reset(self, seed=0):
        return np.array([0.0]), {}
    
    def step(self, action):
        action = np.clip(action, -1, 1)
        reward = 0.0
        if np.abs(action) >= 0.9:
            if np.random.randint(2) == 0:
                reward = 1.0
            else:
                reward = 0.0
        else:
            reward = -1.0
        
        return np.array([0.0]), reward, True, True, {}


def run_sdac():
    import sdac

    algo = sdac.SDAC()
    algo.buffer_size = int(1e4)
    algo.total_timesteps = int(3e4)
    algo.hidden = 512
    algo.n_collect_data = 1
    algo.n_optimizer_step = 1
    algo.n_atoms = 3
    algo.v_min = -1
    algo.v_max = 1
    algo.gamma = 1
    algo.learning_rate = 1e-3
    algo.learning_starts = 1e3
    algo.batch_size = 32

    class Env(sdac.Wrapper):
        def __init__(self):
            self.name = "cheese"
            self.path = "."
            self.env = TrapCheese()
            self.state_dim = self.env.observation_space.shape[0]
            self.action_dim = 1
            self.action_atoms = 51
        
        def reset(self):
            obs, info = self.env.reset()
            return obs
        
        def step(self, act):
            obs, r, d, t, info = self.env.step(self.to_float(act))
            return obs, r, d, t
        
    algo.env = Env()
    algo.eval_env = Env()

    algo.train()


def run_sac():
    from stable_baselines3 import SAC
    from stable_baselines3.common.evaluation import evaluate_policy
    env = TrapCheese()
    model = SAC("MlpPolicy", env).learn(int(3e4))
    print(evaluate_policy(model, env, n_eval_episodes=100, return_episode_rewards=True))


run_sdac()