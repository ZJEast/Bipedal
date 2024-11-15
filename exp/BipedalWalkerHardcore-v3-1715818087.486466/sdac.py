# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch import Tensor
from torch.distributions import Categorical, Normal
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.nn import functional as F
import time
import weakref


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_atoms, n_atoms=101, hidden=256, mode=1, atoms_lst=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_atoms = action_atoms
        self.hidden = hidden

        assert mode in [0, 1]
        if mode == 0:
            embedding = torch.eye(action_atoms)
            embedding = embedding - 1 / action_atoms
        elif mode == 1:
            mid = action_atoms // 2 + 1
            embedding = torch.empty((action_atoms, 2 * mid))
            ix = torch.arange(action_atoms)
            d = 1
            for i in range(mid):
                embedding[:, 2*i] = (ix - mid) % action_atoms // d % 2
                embedding[:, 2*i + 1] = 1 - (mid - ix) % action_atoms // d % 2
                d += 1
            embedding = 2 * embedding - 1
            embedding = embedding / action_atoms
        d_embedding = embedding.shape[1]
        self.embedding = nn.Parameter(embedding, requires_grad=False)

        self.state = nn.Linear(state_dim, hidden)
        self.act = nn.Linear(action_dim * d_embedding, hidden)
        self.network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_atoms)
        )

    def _net(self, x: Tensor, act: Tensor):
        B = act.shape[0]
        n, m = self.action_dim, self.action_atoms

        act = torch.einsum("bnm, md -> bnd", act.view(B, n, m), self.embedding)
        _x: Tensor = self.state(x) + self.act(act.view(B, -1))
        _x = self.network(_x)
        if hasattr(self, "atoms_w"):
            _x = torch.einsum("bq, aq -> ba", _x, self.atoms_w) + self.atoms_b
        return _x

    def forward(self, x: Tensor, act: Tensor):
        _x = self._net(x, act)
        _log = F.log_softmax(_x, dim=-1)
        return _log
    
    def logcumsumexp(self, x: Tensor, act: Tensor):
        _x = self._net(x, act)

        _log1 = torch.logcumsumexp(_x, dim=-1)

        _log2 = torch.logcumsumexp(_x.flip([-1]), dim=-1)
        _log2 = _log2.flip([-1])

        _log1, _log2 = _log1[:, :-1], _log2[:, 1:]

        return torch.logaddexp(np.log(1e-4) + _log1, _log2) - torch.logaddexp(_log1, _log2)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_atoms, hidden=256):
        super().__init__()
        self.action_dim = action_dim
        self.action_atoms = action_atoms
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim * action_atoms),
        )

    def forward(self, x: Tensor):
        B = x.shape[0]
        n, m = self.action_dim, self.action_atoms

        logits: Tensor = self.network.forward(x)
        logits = logits.view(B, -1)
        return logits
    
    def log_softmax(self, logits: Tensor):
        B = logits.shape[0]
        n, m = self.action_dim, self.action_atoms

        logits = logits.view(-1, m)
        logits = logits.log_softmax(dim=-1)
        logits = logits.view(B, -1)
        return logits
    
    def sample(self, logits: Tensor):
        B = logits.shape[0]
        n, m = self.action_dim, self.action_atoms

        logits = logits.view(-1, m)
        logits = Categorical(logits=logits)
        act = logits.sample().view(-1, n)
        return act
    
    def eval(self, logits: Tensor):
        B = logits.shape[0]
        n, m = self.action_dim, self.action_atoms

        logits = logits.view(-1, m)
        act = logits.argmax(dim=-1).view(-1, n)
        return act


class ReplayBufferManager:
    def __init__(self, device, buffer_size=int(1e6)):
        self.buffer_size = buffer_size
        self.cursor = 0
        self.n_sample = 0
        self.device = device
        
    def add(self):
        cursor = self.cursor
        n_sample = self.n_sample
        buffer_size = self.buffer_size

        self.cursor = (cursor + 1) % buffer_size
        self.n_sample = min(n_sample + 1, buffer_size)

        return cursor
    
    def sample(self, batch_size):
        n_sample = self.n_sample
        sample = torch.randint(0, n_sample, (batch_size, ), device=self.device)
        return sample


class Wrapper:
    def __init__(self):
        self.name = "test"
        self.state_dim = 10
        self.action_dim = 4
        self.action_atoms = 51
        self.env = None
        self.algo = None
        self.path = "."
    
    def reset(self):
        return None
    
    def step(self, act):
        return None, 0.0, False, False
    
    def to_float(self, act):
        return act / (self.action_atoms - 1) * 2 - 1
    
    def sdac(self):
        algo: SDAC = self.algo()
        return algo


class SDAC:
    def __init__(self):
        self.seed                     = 1
        self.torch_deterministic      = True
        self.cuda                     = True
        self.total_timesteps          = int(2e6)
        self.learning_rate            = 1e-4
        self.n_atoms                  = 401
        self.v_min                    = -200.0
        self.v_max                    = 200.0
        self.buffer_size              = int(3e5)
        self.gamma                    = 1 - 1 / 200
        self.target_network_frequency = 2
        self.batch_size               = 512
        self.learning_starts          = 10000
        self.train_frequency          = 1
        self.policy_frequency         = 2
        self.tau                      = 0.005
        self.tensorboard_frequency    = 100
        self.env: Wrapper = None
        self.eval_env: Wrapper = None
        self.hidden = 2048
        self.reward_norm = 1.0
        self.auto_reward_norm = True
        self.n_collect_data = 100
        self.n_optimizer_step = 10
        self.n_eval = 100
        self.eval_frequency = 25000
        self.h1 = 1.0
        self.h2 = 0.0
        self.beta = 0.5
        self.success = 0.0
        self.observation_norm = False
        self.her = False
        self.her_batch_size = 1024
        self.her_buffer_size = int(1e6)
        # noise = [1/20, 1/40, 1/80, 1/160, 1/320, 0.0, -1]
        # self.noise_setting = (noise, [1/len(noise)] * len(noise))
        self.noise_setting = ([1/20, 0, -1], [0.5, 0.3, 0.2])
        self.atoms_lst = None
    
    def random_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

    def build_writer(self):
        self.writer = SummaryWriter(f"{self.env.path}/tf-logs")
    
    def build_network(self):
        state_dim = self.env.state_dim
        action_dim = self.env.action_dim
        action_atoms = self.env.action_atoms
        hidden = self.hidden
        device = self.device = torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")

        arg1 = [state_dim, action_dim, action_atoms]
        arg2 = {"n_atoms":self.n_atoms, "hidden":hidden, "atoms_lst":self.atoms_lst}

        qf1 = QNetwork(*arg1, **arg2).to(device)
        qf2 = QNetwork(*arg1, **arg2).to(device)
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=self.learning_rate)
        target_qf1 = QNetwork(*arg1, **arg2).to(device)
        target_qf1.load_state_dict(qf1.state_dict())
        target_qf2 = QNetwork(*arg1, **arg2).to(device)
        target_qf2.load_state_dict(qf2.state_dict())

        arg1 = [state_dim, action_dim, action_atoms]
        arg2 = {"hidden":hidden}

        actor = Actor(*arg1, **arg2).to(device)
        a_optimizer = optim.Adam(actor.parameters(), lr=self.learning_rate)
        target_actor = Actor(*arg1, **arg2).to(device)
        target_actor.load_state_dict(actor.state_dict())
        best_actor = Actor(*arg1, **arg2).to(device)
        best_actor.load_state_dict(actor.state_dict())

        self.qf1 = qf1
        self.qf2 = qf2
        self.q_optimizer = q_optimizer
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.actor = actor
        self.a_optimizer = a_optimizer
        self.target_actor = target_actor
        self.best_actor = best_actor

        self.collect_actor = self.actor

    def build_replay_buffer(self):
        device = self.device
        state_dim = self.env.state_dim
        action_dim = self.env.action_dim

        self.rb_mgr = ReplayBufferManager(device=device, buffer_size=self.buffer_size)
        self.rb_obs = torch.empty((self.buffer_size, state_dim), requires_grad=False, device=device)
        self.rb_next_obs = torch.empty((self.buffer_size, state_dim), requires_grad=False, device=device)
        self.rb_act = torch.empty((self.buffer_size, action_dim), requires_grad=False, device=device, dtype=torch.long)
        self.rb_reward = torch.empty((self.buffer_size, 1), requires_grad=False, device=device)
        self.rb_done = torch.empty((self.buffer_size, 1), requires_grad=False, device=device)
        self.rb_return = torch.full((self.buffer_size, 1), -torch.inf, requires_grad=False, device=device)

        if self.her:
            self.her_mgr = ReplayBufferManager(device=device, buffer_size=self.buffer_size)
            self.her_obs = torch.empty((self.her_buffer_size, state_dim), requires_grad=False, device=device)
            self.her_next_obs = torch.empty((self.her_buffer_size, state_dim), requires_grad=False, device=device)
            self.her_act = torch.empty((self.her_buffer_size, action_dim), requires_grad=False, device=device, dtype=torch.long)
            self.her_reward = torch.empty((self.her_buffer_size, 1), requires_grad=False, device=device)
            self.her_done = torch.empty((self.her_buffer_size, 1), requires_grad=False, device=device)
            self.her_return = torch.full((self.her_buffer_size, 1), -torch.inf, requires_grad=False, device=device)

        self.obervation_max = None
        self.obervation_min = None
        self.obervation_mean = None
        self.obervation_apply = None

    def train(self):
        self.random_seed()
        self.build_writer()
        self.build_network()
        self.build_replay_buffer()
        self.env.algo = weakref.ref(self)

        self.train_step = 0
        self.env_step = 0
        self.eval_step = 0
        self.eval_best = -float("inf")
        self.need_init = True
        self.start_time = time.time()
        self.action_queue = np.zeros((self.env.action_dim, self.env.action_atoms))

        while self.env_step < self.total_timesteps:
            self.collect_data()
            self.optimizer_step()
        
        self.eval()
        self.writer.close()
    
    def update_observation_norm(self, obs):
        if self.obervation_max is None:
            self.obervation_max = obs
            self.obervation_min = obs
            self.obervation_mean = obs
        self.obervation_max = np.maximum(obs, self.obervation_max)
        self.obervation_min = np.minimum(obs, self.obervation_min)
        a = max(1 / self.env_step, 1/(2e6))
        self.obervation_mean = a * obs + (1 - a) * self.obervation_mean

    def apply_observation_norm(self, b_obs: Tensor, freeze=True):
        B, d = b_obs.shape
        if not freeze:
            self.obervation_apply = (self.obervation_mean, self.obervation_max, self.obervation_min)
        if self.obervation_apply is None:
            return b_obs
        _mean, _max, _min = self.obervation_apply
        _mean = torch.tensor(_mean).float().to(self.device).view(1, d).expand(B, d)
        _max = torch.tensor(_max).float().to(self.device).view(1, d).expand(B, d)
        _min = torch.tensor(_min).float().to(self.device).view(1, d).expand(B, d)
        b_obs = b_obs - _mean
        _max = _max - _mean
        _min = _mean - _min
        ix = b_obs > 0
        b_obs[ix] /= _max[ix]
        ix = b_obs < 0
        b_obs[ix] /= _min[ix]
        return b_obs
    
    def collect_data(self):
        self.collect_actor = self.actor
        _n, _p = self.noise_setting
        noise = np.random.choice(_n, p=_p)
        for t in range(self.n_collect_data):
            self._collect_data(noise=noise)
    
    def _collect_data(self, noise=0):
        env = self.env
        device = self.device

        if self.need_init:
            self.obs = env.reset()
            self.result = 0.0
            self.need_init = False

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            _obs = torch.tensor(self.obs).float().view(1, -1).to(device)
            if self.observation_norm:
                _obs = self.apply_observation_norm(_obs)
            logits = self.collect_actor.forward(_obs)
            action = self.collect_actor.sample(logits)
            if noise < 0:
                action = self.collect_actor.eval(logits)

        to_action = action.flatten().cpu().numpy()
        to_env = to_action
        for _choice in range(self.env.action_dim):
            if np.random.random() < noise / self.env.action_dim:
                _value: np.ndarray = self.action_queue[_choice, :]
                _value = _value.argmin()
                to_action[_choice] = _value

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, truncated = env.step(to_env)
        self.env_step += 1
        self.result += reward
        if self.auto_reward_norm:
            self.reward_norm = max(self.reward_norm, reward)
        if self.observation_norm:
            self.update_observation_norm(next_obs)

        for i, _a in enumerate(to_action):
            self.action_queue[i, _a] = self.env_step
        
        cursor = self.rb_mgr.add()
        self.rb_obs[cursor]      = torch.tensor(self.obs).float().to(device)
        self.rb_next_obs[cursor] = torch.tensor(next_obs).float().to(device)
        self.rb_act[cursor]      = torch.tensor(to_action).to(device)
        self.rb_reward[cursor]   = torch.tensor(reward).float().to(device)
        self.rb_done[cursor]     = torch.tensor(float(done)).to(device)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        self.obs = next_obs

        if done or truncated:
            self.need_init = True
            self.writer.add_scalar("losses/reward_norm", self.reward_norm, self.env_step)
            self.writer.add_scalar("losses/result", self.result, self.env_step)
            print(f"train {self.result} {self.train_step} {self.env_step}")

    def optimizer_step(self):
        if self.rb_mgr.n_sample < self.learning_starts:
            return
        device = self.device
        action_atoms = self.env.action_atoms
        action_dim = self.env.action_dim
        for _ in range(self.n_optimizer_step):
            train_step = self.train_step
            B = self.batch_size
            b_i: Tensor = self.rb_mgr.sample(B).view(B, 1)
            rb = self.rb_obs
            b_obs = rb.gather(0, b_i.expand(B, rb.shape[-1]))
            rb = self.rb_next_obs
            b_next_obs = rb.gather(0, b_i.expand(B, rb.shape[-1]))
            rb = self.rb_act
            b_act = rb.gather(0, b_i.expand(B, rb.shape[-1]))
            rb = self.rb_reward
            b_reward = rb.gather(0, b_i.expand(B, rb.shape[-1]))
            b_reward = b_reward / self.reward_norm
            rb = self.rb_done
            b_done = rb.gather(0, b_i.expand(B, rb.shape[-1]))
            rb = self.rb_return
            b_return = rb.gather(0, b_i.expand(B, rb.shape[-1]))
            b_return = b_return / self.reward_norm
            
            if self.her and self.her_mgr.n_sample >= self.her_batch_size:
                B = self.her_batch_size
                h_i: Tensor = self.her_mgr.sample(B).view(B, 1)
                rb = self.her_obs
                h_obs = rb.gather(0, h_i.expand(B, rb.shape[-1]))
                rb = self.her_next_obs
                h_next_obs = rb.gather(0, h_i.expand(B, rb.shape[-1]))
                rb = self.her_act
                h_act = rb.gather(0, h_i.expand(B, rb.shape[-1]))
                rb = self.her_reward
                h_reward = rb.gather(0, h_i.expand(B, rb.shape[-1]))
                h_reward = h_reward / self.reward_norm
                rb = self.her_done
                h_done = rb.gather(0, h_i.expand(B, rb.shape[-1]))
                rb = self.her_return
                h_return = rb.gather(0, h_i.expand(B, rb.shape[-1]))
                h_return = h_return / self.reward_norm

                b_i = torch.cat([b_i, h_i], dim=0)
                b_obs = torch.cat([b_obs, h_obs], dim=0)
                b_next_obs = torch.cat([b_next_obs, h_next_obs], dim=0)
                b_act = torch.cat([b_act, h_act], dim=0)
                b_reward = torch.cat([b_reward, h_reward], dim=0)
                b_done = torch.cat([b_done, h_done], dim=0)
                b_return = torch.cat([b_return, h_return], dim=0)

                B = b_i.shape[0]

            if self.observation_norm:
                b_obs = self.apply_observation_norm(b_obs, freeze=False)
                b_next_obs = self.apply_observation_norm(b_next_obs, freeze=False)
            
            with torch.no_grad():
                next_act_p = self.target_actor.forward(b_next_obs)
                next_act_p = self.target_actor.log_softmax(next_act_p).exp()
                n, m = self.env.action_dim, self.env.action_atoms
                # next_act_p = F.one_hot(next_act_p.view(B, n, m).argmax(dim=-1), m).float().view(next_act_p.shape)
                next_z = torch.linspace(self.v_min, self.v_max, self.n_atoms, device=device)
                next_atoms = b_reward + self.gamma * next_z * (1 - b_done) + b_done * self.success
                next_atoms = torch.maximum(next_atoms, b_return)

                next_cmfs1: Tensor = self.target_qf1.forward(b_next_obs, next_act_p).exp().cumsum(-1)
                next_cmfs2: Tensor = self.target_qf2.forward(b_next_obs, next_act_p).exp().cumsum(-1)

                next_cmfs = torch.maximum(next_cmfs1, next_cmfs2)
                next_pmfs = next_cmfs.clone()
                next_pmfs[:, 1:] = next_cmfs[:, 1:] - next_cmfs[:, :-1]

                # projection
                delta_z = next_z[1] - next_z[0]
                tz = next_atoms.clamp(self.v_min, self.v_max)
                b = (tz - self.v_min) / delta_z
                l = b.floor().clamp(0, self.n_atoms - 1)
                u = b.ceil().clamp(0, self.n_atoms - 1)
                # (l == u).float() handles the case where bj is exactly an integer
                # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                d_m_l = (u + (l == u).float() - b) * next_pmfs
                d_m_u = (b - l) * next_pmfs
                target_pmfs = torch.zeros_like(next_pmfs)
                for i in range(target_pmfs.size(0)):
                    target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                    target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

            b_act_p: Tensor = F.one_hot(b_act, m).float().view(B, -1)

            old_logits1 = self.qf1.forward(b_obs, b_act_p)
            old_logits2 = self.qf2.forward(b_obs, b_act_p)

            q1_loss = ( - target_pmfs * old_logits1).sum(-1).mean()
            q2_loss = ( - target_pmfs * old_logits2).sum(-1).mean()
            q_loss = (q1_loss + q2_loss)

            # optimize the model
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

            if train_step % self.policy_frequency == 0:
                act_logits: Tensor = self.actor.forward(b_obs)
                act_log = self.actor.log_softmax(act_logits)
                act_p = act_log.exp()
                n = self.env.action_dim
                m = self.env.action_atoms
                logcmfs: Tensor = self.qf1.logcumsumexp(b_obs, act_p)
                cmfs = 1 - logcmfs.exp()
                h = torch.linspace(self.h1, self.h2, cmfs.shape[1], device=device).clamp(0.0, 1.0).view(1, -1)
                h = (h * cmfs).detach().max(dim=-1).values
                # entropy: Tensor = Categorical(logits=act_logits.view(-1, m)).entropy()
                entropy = (- act_p * act_log).view(-1, n * m).sum(-1)
                entropy = entropy / (n * np.log(m))
                actor_loss: Tensor = - logcmfs.mean() + self.beta * (h - entropy).relu().mean()

                self.a_optimizer.zero_grad()
                actor_loss.backward()
                self.a_optimizer.step()

            # update the target network
            if train_step % self.target_network_frequency == 0:
                for target_param, param in zip(self.target_qf1.parameters(), self.qf1.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1.0 - self.tau) * target_param.data
                    )
                for target_param, param in zip(self.target_qf2.parameters(), self.qf2.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1.0 - self.tau) * target_param.data
                    )
                for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1.0 - self.tau) * target_param.data
                    )

            if train_step % 100 == 0:
                writer = self.writer
                writer.add_scalar("losses/entropy", entropy.mean().item(), train_step)
                # writer.add_scalar("losses/lambda", lambda_.mean().item(), train_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), train_step)
                writer.add_scalar("losses/q1_loss", q1_loss.item(), train_step)
                writer.add_scalar("losses/q2_loss", q2_loss.item(), train_step)
                writer.add_scalar("losses/q_loss", q_loss.item(), train_step)

                z = torch.linspace(self.v_min, self.v_max, self.n_atoms, device=device)
                old_val1 = (old_logits1.exp() * z).sum(1)
                writer.add_scalar("losses/q1_values", old_val1.mean().item(), train_step)
                old_val2 = (old_logits2.exp() * z).sum(1)
                writer.add_scalar("losses/q2_values", old_val2.mean().item(), train_step)
                print("SPS:", int(train_step / (time.time() - self.start_time)))
                writer.add_scalar("charts/SPS", int(train_step / (time.time() - self.start_time)), train_step)
        
            if train_step % self.eval_frequency == 0:
                self.eval()

            self.train_step += 1

    def eval(self):
        eval_env = self.eval_env
        target_actor = self.target_actor
        device = self.device
        writer = self.writer
        eval_step = self.eval_step
        mean = []
        for _ in range(self.n_eval):
            eval_obs = eval_env.reset()
            eval_result = 0.0
            while True:
                # ALGO LOGIC: put action logic here
                with torch.no_grad():
                    _obs = torch.tensor(eval_obs).float().view(1, -1).to(device)
                    if self.observation_norm:
                        _obs = self.apply_observation_norm(_obs)
                    logits = target_actor.forward(_obs)
                    action = target_actor.eval(logits)

                to_action = action.flatten().cpu().numpy()
                to_env = to_action

                # TRY NOT TO MODIFY: execute the game and log data.
                eval_next_obs, eval_reward, eval_terminated, eval_truncated = eval_env.step(to_env)
                eval_result += eval_reward

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                eval_obs = eval_next_obs

                if eval_terminated or eval_truncated:
                    break

            print("losses/test", eval_result, eval_step, self.env_step)
            writer.add_scalar("losses/test", eval_result, eval_step)

            mean.append(eval_result)
            eval_step += 1
            self.eval_step = eval_step

        mean = sum(mean) / len(mean)
        writer.add_scalar("losses/mean", mean, eval_step)

        if self.eval_best < mean:
            self.eval_best = mean
            writer.add_scalar("losses/eval_best", self.eval_best, eval_step)
            self.best_actor.load_state_dict(target_actor.state_dict())
        torch.save({
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "actor": self.actor.state_dict(),
            "target_qf1": self.target_qf1.state_dict(),
            "target_qf2": self.target_qf2.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "best_actor": self.best_actor.state_dict(),
        }, f"{self.env.path}/model.pth")
        if self.obervation_apply is not None:
            torch.save(torch.tensor(self.obervation_apply), f"{self.env.path}/obs-norm.pth")
