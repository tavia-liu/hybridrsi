"""DrQ-v2 with a low-dimensional proprioceptive vector next to the pixels (the robosuite / SERL convention).

Same encoder, augmentation, losses and schedules as drqv2.DrQV2Agent; the only change is that the actor and the
critics see [encoder features, state] instead of encoder features alone. The state is the robot's own report
(TCP pose, gripper width, joints, wrist wrench); nothing about the part.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import drqv2
import utils


class Actor(nn.Module):
    def __init__(self, repr_dim, state_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.policy = nn.Sequential(nn.Linear(feature_dim + state_dim, hidden_dim), nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))
        self.apply(utils.weight_init)

    def forward(self, obs, state, std):
        h = torch.cat([self.trunk(obs), state], dim=-1)
        mu = torch.tanh(self.policy(h))
        return utils.TruncatedNormal(mu, torch.ones_like(mu) * std)


class Critic(nn.Module):
    def __init__(self, repr_dim, state_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

        def q():
            return nn.Sequential(nn.Linear(feature_dim + state_dim + action_shape[0], hidden_dim), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.Q1, self.Q2 = q(), q()
        self.apply(utils.weight_init)

    def forward(self, obs, state, action):
        h = torch.cat([self.trunk(obs), state, action], dim=-1)
        return self.Q1(h), self.Q2(h)


class DrQV2ProprioAgent(drqv2.DrQV2Agent):
    def __init__(self, obs_shape, state_shape, action_shape, device, lr, feature_dim, hidden_dim, critic_target_tau,
                 num_expl_steps, update_every_steps, stddev_schedule, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        sd = state_shape[0]
        self.encoder = drqv2.Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, sd, action_shape, feature_dim, hidden_dim).to(device)
        self.critic = Critic(self.encoder.repr_dim, sd, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, sd, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.aug = drqv2.RandomShiftsAug(pad=4)
        self.train()
        self.critic_target.train()

    def act(self, obs, state, step, eval_mode):
        obs = self.encoder(torch.as_tensor(obs, device=self.device).unsqueeze(0))
        state = torch.as_tensor(state, device=self.device).unsqueeze(0)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, state, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, state, action, reward, discount, next_obs, next_state, step):
        metrics = dict()
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            next_action = self.actor(next_obs, next_state, stddev).sample(clip=self.stddev_clip)
            tq1, tq2 = self.critic_target(next_obs, next_state, next_action)
            target_Q = reward + discount * torch.min(tq1, tq2)
        Q1, Q2 = self.critic(obs, state, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        if self.use_tb:
            metrics.update(critic_target_q=target_Q.mean().item(), critic_q1=Q1.mean().item(), critic_q2=Q2.mean().item(),
                           critic_loss=critic_loss.item())
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, state, step):
        metrics = dict()
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, state, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, state, action)
        actor_loss = -torch.min(Q1, Q2).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        if self.use_tb:
            metrics.update(actor_loss=actor_loss.item(), actor_logprob=log_prob.mean().item(),
                           actor_ent=dist.entropy().sum(dim=-1).mean().item())
        return metrics

    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics
        obs, state, action, reward, discount, next_obs, next_state = utils.to_torch(next(replay_iter), self.device)
        obs = self.encoder(self.aug(obs.float()))
        with torch.no_grad():
            next_obs = self.encoder(self.aug(next_obs.float()))
        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()
        metrics.update(self.update_critic(obs, state, action, reward, discount, next_obs, next_state, step))
        metrics.update(self.update_actor(obs.detach(), state, step))
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics


# ---------------------------------------------------------------------- replay with the state alongside the pixels
import random  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import replay_buffer as rb  # noqa: E402


class ReplayBufferProprio(rb.ReplayBuffer):
    def _sample(self):
        try:
            self._try_fetch()
        except Exception:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        idx = np.random.randint(0, rb.episode_len(episode) - self._nstep + 1) + 1
        obs, state = episode["observation"][idx - 1], episode["state"][idx - 1]
        action = episode["action"][idx]
        next_obs, next_state = episode["observation"][idx + self._nstep - 1], episode["state"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            reward += discount * episode["reward"][idx + i]
            discount *= episode["discount"][idx + i] * self._discount
        return (obs, state, action, reward, discount, next_obs, next_state)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers, save_snapshot, nstep, discount, worker_init_fn):
    iterable = ReplayBufferProprio(replay_dir, max_size // max(1, num_workers), num_workers, nstep, discount,
                                   fetch_every=1000, save_snapshot=save_snapshot)
    return torch.utils.data.DataLoader(iterable, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                       worker_init_fn=worker_init_fn)
