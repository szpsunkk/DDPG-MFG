#!/usr/bin/env python
# coding=utf-8
'''
Author: kangkang sun
Email: szpsunkk@163.com
Date:
LastEditor:
LastEditTime:
Discription: 
Environment: 
'''
import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    ''' A2C网络模型，包含一个Actor和Critic
    '''

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        x = self.actor(x)
        # dist = Categorical(probs)
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        value = self.critic(x)
        return value


class A2C:
    ''' A2C算法
    '''

    def __init__(self, state_dim, action_dim, cfg, env) -> None:
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.critic = Critic(state_dim, cfg.hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters())

    def compute_returns(self, next_value, rewards):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R
            returns.insert(0, R)
        return returns

    def choose_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        dist, value = self.actor(state)
        return dist, value

    def update(self, action, next_state, total_loss, cfg):
        value = self.critic(action)
        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        log_probs.append(log_prob)
        values.append(value)
        # rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(cfg.device))
        # masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(cfg.device))

        # print("frame_idx:", frame_idx)
        # if frame_idx % 100 == 0:
        #     test_reward = np.mean([test_env(env, model) for _ in range(10)])
        #     print(f"frame_idx:{frame_idx}, test_reward:{test_reward}")   # 每进行100次进行评估模型的精度
        #     test_rewards.append(test_reward)
        #     if test_ma_rewards:
        #         test_ma_rewards.append(0.9 * test_ma_rewards[-1] + 0.1 * test_reward)
        #         # print(test_ma_rewards)
        #     else:
        #         test_ma_rewards.append(test_reward)
        #         # plot(frame_idx, test_rewards)

        next_state = torch.FloatTensor(next_state).to(cfg.device)
        next_action = self.actor(next_state)
        next_value = self.critic(next_state, next_action)
        returns = self.compute_returns(next_value, rewards)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        total_loss.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
