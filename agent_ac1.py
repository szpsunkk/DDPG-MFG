#!/usr/bin/env python
# coding=utf-8
'''
@Author: skk
@Email: szpsunkk@163com
@Date: 2020-06-09 20:25:52
@LastEditor: skk
LastEditTime: 2021-09-16 00:55:30
@Discription:
@Environment: python 3.7.7
'''
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class ReplayBuffer:
    def __init__(self, capacity, s_dim, a_dim):
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # index = self.position % self.capacity
        # print("state {} action {} reward {} next_state".format(state.ndim, action.ndim, type(reward), type(next_state)))
        self.buffer[self.position] = (state, action.flatten(), [reward], next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # 随机采出小批量转移
        state, action, reward, next_state = zip(*batch)  # 解压成状态，动作等
        return state, action, reward, next_state

    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, r_dim, b_dim, o_dim, hidden_dim, init_w=3e-3):
        super(Actor, self).__init__()
        self.r_dim = r_dim
        self.net = nn.Linear(state_dim, hidden_dim)
        # resource
        self.linear1_r = nn.Linear(hidden_dim, hidden_dim)
        self.linear2_r = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_r = nn.Linear(hidden_dim, hidden_dim)
        self.linear4_r = nn.Linear(hidden_dim, hidden_dim)
        self.linear5_r = nn.Linear(hidden_dim, r_dim)

        self.linear5_r.weight.data.uniform_(-init_w, init_w)
        self.linear5_r.bias.data.uniform_(-init_w, init_w)
        # bandwidth
        self.linear1_b = nn.Linear(hidden_dim, hidden_dim)
        self.linear2_b = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_b = nn.Linear(hidden_dim, hidden_dim)
        self.linear4_b = nn.Linear(hidden_dim, hidden_dim)
        self.linear5_b = nn.Linear(hidden_dim, b_dim)

        self.linear5_b.weight.data.uniform_(-init_w, init_w)
        self.linear5_b.bias.data.uniform_(-init_w, init_w)
        # offloading
        self.user_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, int(o_dim / r_dim)))
        # self.user_layer.append(nn.ReLU())

        #     for user_id in range(r_dim):
        #     self.layer[user_id][0] = nn.Linear(state_dim, hidden_dim)
        #     self.layer[user_id][1] = nn.Linear(hidden_dim, hidden_dim)
        #     self.layer[user_id][2] = nn.Linear(hidden_dim, (o_dim / r_dim))
        #
        #     self.layer[user_id][2].weight.data.uniform_(-init_w, init_w)
        #     self.layer[user_id][2].bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = torch.abs(x)
        net = self.net(x)  # linear
        # resource
        r_1 = self.linear1_r(net)
        r_2 = self.linear2_r(r_1)
        r_3 = self.linear3_r(r_2)
        r_4 = self.linear4_r(r_3)
        r_5 = F.relu(self.linear5_r(r_4))
        # r_3 = r_3.t()
        # bandwidth
        b_1 = self.linear1_b(net)
        b_2 = self.linear2_b(b_1)
        b_3 = self.linear3_b(b_2)
        b_4 = self.linear4_b(b_3)
        b_5 = F.relu(self.linear5_b(b_4))
        # b_3 = b_3.t()
        # offloading
        # for user_id in range(self.r_dim):
        #     layer_0[user_id] = F.relu(self.layer[user_id][0](net))
        #     layer_1[user_id] = F.relu(self.layer[user_id][1](layer_0[user_id]))
        #     layer_2[user_id] = F.softmax(self.layer[user_id][2](layer_1[user_id]))
        # for i, l in enumerate(self.user_layer):
        #     u = l(net)

        a = torch.cat([r_5, b_5], 1)

        for i in range(self.r_dim):
            user = self.user_layer(net)
            # user = user.t()
            a = torch.cat([a, user], 1)
        return torch.abs(a)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, 1)
        # 随机初始化为较小的值
        self.linear5.weight.data.uniform_(-init_w, init_w)
        self.linear5.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # 按维数1拼接
        x = torch.cat([state, action], 1)
        # x = torch.abs(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        x = torch.abs(x)
        return x.squeeze(1)


class AC:
    def __init__(self, state_dim, action_dim, cfg, env_info):
        self.device = cfg.device
        self.s_dim = env_info.s_dim
        self.r_dim = env_info.r_dim
        self.b_dim = env_info.b_dim
        self.o_dim = env_info.o_dim
        self.a_dim = self.r_dim + self.b_dim + self.o_dim
        # self.n = self.o_dim / self.r_dim
        self.critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.actor = Actor(state_dim, env_info.r_dim, env_info.b_dim, env_info.o_dim, cfg.hidden_dim).to(cfg.device)
        self.target_critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_actor = Actor(state_dim, env_info.r_dim, env_info.b_dim, env_info.o_dim, cfg.hidden_dim).to(
            cfg.device)

        # 复制参数到目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity, self.s_dim, self.a_dim)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau  # 软更新参数
        self.gamma = cfg.gamma
        self.writer = SummaryWriter()
        self.p = 0
        if cfg.graph:
            s = torch.rand(1, 10)
            actor = Actor(10, 2, 2, 256)
            self.writer.add_graph(actor, s)

        # with self.writer as w:
        #     w.add_graph(self.actor)
        # if cfg.graph:
        # self.writer.add_graph(self.critic, state_dim)
        # self.writer.add_graph(self.actor)
        # self.writer.add_graph(self.target_actor)
        # self.writer.add_graph(self.target_critic)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()

    def update(self, cfg, i_ep, i_step, state, next_state, action, reward):
        # if len(self.memory) < self.batch_size:  # 当 memory 中不满足一个批量时，不更新策略
        #     return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        # state, action, reward, next_state = self.memory.sample(self.batch_size)
        # batch = self.memory.sample(self.batch_size)
        # state = batch[:, : self.s_dim]
        # action = batch[:, self.s_dim: self.s_dim + self.a_dim]
        # reward = batch[:, -self.s_dim - 1: -self.s_dim]
        # next_state = batch[:, -self.s_dim:]
        # 转变为张量
        ###########----------------#############

        ###########----------------#############

        state = torch.FloatTensor(state.reshape(1, state.shape[0])).to(self.device)
        next_state = torch.FloatTensor(next_state.reshape(1, next_state.shape[0])).to(self.device)
        action = torch.FloatTensor(action.transpose()).to(self.device)
        reward = torch.FloatTensor([np.float32(reward)]).to(self.device)
        # done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        policy_loss = self.critic(state, self.actor(state))
        # if self.graph:
        #     self.writer.add_graph(self.actor, state)
        policy_loss = -policy_loss.mean()
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        # 真实状态下的value值
        value = self.critic(state, action)
        # 预测的value值和真实的值之间的差别
        value_loss = nn.MSELoss()(value, expected_value.detach())
        # print("value {} value_loss {}".format(value, value_loss))
        # print("iep {} istep {} value_loss {} policy loss {}".format(i_ep, i_step, value_loss, policy_loss))
        if i_ep == self.p:
            test_value = "value loss of " + cfg.algo + np.str(i_ep)
            test_policy = "policy loss of " + cfg.algo + np.str(i_ep)
            self.writer.add_scalar(test_value, value_loss, i_step)
            self.writer.add_scalar(test_policy, policy_loss, i_step)
        else:
            self.p = self.p + 1

        self.actor_optimizer.zero_grad()  # 梯度归零
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        # 软更新
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        # paras = list(self.actor.parameters())
        # for num, para in enumerate(paras):
        #     print("number:", num)
        #     print(para)
        #     print("-------------------------")

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'checkpoint.pt')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'checkpoint.pt'))


if __name__ == '__main__':
    s = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    a = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])
    actor = Actor(10, 2, 2, 256)
    critic = Critic(10, 8, 256)
    # writer = SummaryWriter()
    # writer.add_graph(actor, s)
    a_out = actor.forward(s)
    print(a_out)
    c_out = critic.forward(s, a_out)
    print(c_out)
