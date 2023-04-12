#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-12-22 10:40:05
LastEditor: JiangJi
LastEditTime: 2021-12-22 10:43:55
Discription: 
'''
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from TD3.memory import ReplayBuffer, ReplayBuffer_MFG
from torch.utils.tensorboard import SummaryWriter


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, state, action, reward, next_state):
        self.state[self.ptr] = state
        self.action[self.ptr] = action.flatten()
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        # self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            # torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


# class Actor(nn.Module):
#
# 	def __init__(self, input_dim, output_dim, max_action):
# 		'''[summary]
#
# 		Args:
# 			input_dim (int): 输入维度，这里等于state_dim
# 			output_dim (int): 输出维度，这里等于action_dim
# 			max_action (int): action的最大值
# 		'''
# 		super(Actor, self).__init__()
#
# 		self.l1 = nn.Linear(input_dim, 256)
# 		self.l2 = nn.Linear(256, 256)
# 		self.l3 = nn.Linear(256, output_dim)
# 		self.max_action = max_action
#
# 	def forward(self, state):
#
# 		a = F.relu(self.l1(state))
# 		a = F.relu(self.l2(a))
# 		return self.max_action * torch.tanh(self.l3(a))
#
#
# class Critic(nn.Module):
# 	def __init__(self, input_dim, output_dim):
# 		super(Critic, self).__init__()
#
# 		# Q1 architecture
# 		self.l1 = nn.Linear(input_dim + output_dim, 256)
# 		self.l2 = nn.Linear(256, 256)
# 		self.l3 = nn.Linear(256, 1)
#
# 		# Q2 architecture
# 		self.l4 = nn.Linear(input_dim + output_dim, 256)
# 		self.l5 = nn.Linear(256, 256)
# 		self.l6 = nn.Linear(256, 1)
#
#
# 	def forward(self, state, action):
# 		sa = torch.cat([state, action], 1)
#
# 		q1 = F.relu(self.l1(sa))
# 		q1 = F.relu(self.l2(q1))
# 		q1 = self.l3(q1)
#
# 		q2 = F.relu(self.l4(sa))
# 		q2 = F.relu(self.l5(q2))
# 		q2 = self.l6(q2)
# 		return q1, q2
#
#
# 	def Q1(self, state, action):
# 		sa = torch.cat([state, action], 1)
#
# 		q1 = F.relu(self.l1(sa))
# 		q1 = F.relu(self.l2(q1))
# 		q1 = self.l3(q1)
# 		return q1
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
        net = F.relu(self.net(x))  # linear
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
        # net 1
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, 1)
        # 随机初始化为较小的值
        self.linear5.weight.data.uniform_(-init_w, init_w)
        self.linear5.bias.data.uniform_(-init_w, init_w)
        # net 2
        self.linear6 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear7 = nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = nn.Linear(hidden_dim, hidden_dim)
        self.linear9 = nn.Linear(hidden_dim, hidden_dim)
        self.linear10 = nn.Linear(hidden_dim, 1)
        # 随机初始化为较小的值
        self.linear10.weight.data.uniform_(-init_w, init_w)
        self.linear10.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # 按维数1拼接
        x = torch.cat([state, action], 1)
        x = torch.abs(x)
        # net 1
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear3(x1))
        x1 = F.relu(self.linear4(x1))
        x1 = self.linear5(x1)
        x1 = torch.abs(x1)

        # 按维数1拼接
        # net 2
        x2 = F.relu(self.linear6(x))
        x2 = F.relu(self.linear7(x2))
        x2 = F.relu(self.linear8(x2))
        x2 = F.relu(self.linear9(x2))
        x2 = self.linear10(x2)
        x2 = torch.abs(x2)
        return x1, x2

    def Q1(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.abs(x)
        # net 1
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear3(x1))
        x1 = F.relu(self.linear4(x1))
        x1 = self.linear5(x1)
        x1 = torch.abs(x1)
        return x1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            cfg,
            env_info
    ):
        # self.max_action = max_action
        self.gamma = cfg.gamma
        self.lr = cfg.lr
        self.policy_noise = cfg.policy_noise
        self.noise_clip = cfg.noise_clip
        self.policy_freq = cfg.policy_freq
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.total_it = 0
        self.writer = SummaryWriter()
        self.s_dim = env_info.s_dim
        self.r_dim = env_info.r_dim
        self.b_dim = env_info.b_dim
        self.o_dim = env_info.o_dim
        self.a_dim = self.r_dim + self.b_dim + self.o_dim

        self.actor = Actor(state_dim, env_info.r_dim, env_info.b_dim, env_info.o_dim, cfg.hidden_dim).to(cfg.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.memory = ReplayBuffer(self.s_dim, self.a_dim)
        self.p = 0

    def choose_action(self, state):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # return self.actor(state).cpu().data.numpy().flatten()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()

    def update(self, cfg, i_ep, i_step):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward = self.memory.sample(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            # noise = (
            # 	torch.randn_like(action) * self.policy_noise
            # ).clamp(-self.noise_clip, self.noise_clip)
            #
            # next_action = (
            # 	self.actor_target(next_state) + noise
            # ).clamp(-self.max_action, self.max_action)
            next_action = self.actor_target(next_state)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        if i_ep == self.p:
            # print("i_ep {} critic_loss {}".format(i_ep, critic_loss))
            test_critic_loss = "critic loss of TD3" + np.str(i_ep)
            test_target_Q = "target_Q" + np.str(i_ep)
            self.writer.add_scalar(test_critic_loss, critic_loss, i_step)
        # self.writer.add_scalar(test_target_Q, target_Q, i_step)
        else:
            self.p += 1
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            if i_ep == self.p:
                # print("i_ep {} actor_loss {}".format(i_ep, actor_loss))
                test_actor_loss = "actor loss of TD3" + np.str(i_ep)
                self.writer.add_scalar(test_actor_loss, actor_loss, i_step)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.lr * param.data + (1 - self.lr) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.lr * param.data + (1 - self.lr) * target_param.data)

    def save(self, path):
        torch.save(self.critic.state_dict(), path + "td3_critic")
        torch.save(self.critic_optimizer.state_dict(), path + "td3_critic_optimizer")

        torch.save(self.actor.state_dict(), path + "td3_actor")
        torch.save(self.actor_optimizer.state_dict(), path + "td3_actor_optimizer")

    def load(self, path):
        self.critic.load_state_dict(torch.load(path + "td3_critic"))
        self.critic_optimizer.load_state_dict(torch.load(path + "td3_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(path + "td3_actor"))
        self.actor_optimizer.load_state_dict(torch.load(path + "td3_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


class Actor_MFG(nn.Module):

    def __init__(self, input_dim, output_dim, max_action):
        '''[summary]

        Args:
            input_dim (int): 输入维度，这里等于state_dim
            output_dim (int): 输出维度，这里等于action_dim
            max_action (int): action的最大值
        '''
        super(Actor_MFG, self).__init__()

        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, output_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic_MFG(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic_MFG, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(input_dim + output_dim + 1, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(input_dim + output_dim + 1, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action, mean_action):
        sa = torch.cat([state, action, mean_action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action, mean_action):
        sa = torch.cat([state, action, mean_action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_MFG(object):
    def __init__(
            self,
            input_dim,
            output_dim,
            max_action,
            cfg,
    ):
        self.max_action = max_action
        self.gamma = cfg.gamma
        self.lr = cfg.lr
        self.policy_noise = cfg.policy_noise
        self.noise_clip = cfg.noise_clip
        self.policy_freq = cfg.policy_freq
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.total_it = 0

        self.actor = Actor_MFG(input_dim, output_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic_MFG(input_dim, output_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.memory = ReplayBuffer_MFG(input_dim, output_dim)

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done, mean_action = self.memory.sample(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            mean_next_action = torch.mean(next_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action, mean_next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, mean_action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.lr * param.data + (1 - self.lr) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.lr * param.data + (1 - self.lr) * target_param.data)

    def save(self, path):
        torch.save(self.critic.state_dict(), path + "td3_critic")
        torch.save(self.critic_optimizer.state_dict(), path + "td3_critic_optimizer")

        torch.save(self.actor.state_dict(), path + "td3_actor")
        torch.save(self.actor_optimizer.state_dict(), path + "td3_actor_optimizer")

    def load(self, path):
        self.critic.load_state_dict(torch.load(path + "td3_critic"))
        self.critic_optimizer.load_state_dict(torch.load(path + "td3_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(path + "td3_actor"))
        self.actor_optimizer.load_state_dict(torch.load(path + "td3_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
