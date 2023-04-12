#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Author: skk
Email: szpsunkk@163.com
Date:
LastEditor: skk
LastEditTime:
Discription: the process of training and testing for algo
'''

import os
import sys

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径
import numpy as np

from torch.utils.tensorboard import SummaryWriter
# from DDPG.env import OUNoise
def exploration(a, env_info, r_var, b_var):
    a = np.transpose(a)
    for i in range(env_info.r_dim + env_info.b_dim):
        # resource
        if i < env_info.r_dim:
            a[i] = np.clip(np.random.normal(a[i], r_var), 0,
                           1) * env_info.r_bound  # 剪切数组（输入数组，最小值，最大值）  normal（均值，方差）从正态分布抽取值
        # bandwidth
        elif i < env_info.r_dim + env_info.b_dim:
            a[i] = np.clip(np.random.normal(a[i], b_var), 0, 1) * env_info.b_bound
    return np.transpose(a)


def train(cfg, env_info, env, agent):
    print('Begin training !')
    print(f'evn：{cfg.env_name}，algo：{cfg.algo}，device：{cfg.device}')
    # ou_noise = OUNoise(env.action_space)  #
    rewards = []  # record all the rewards
    ma_rewards = []  # record all the mean rewards based on smoothness
    var_reward = []
    save_i_step = []
    save_i_ep = []
    save_trans = []
    max_rewards = 0
    r_var = 1  # the var of resources
    b_var = 1  # the var of bandwidth
    CHECK_EPISODE = 4
    CHANGE = False
    var_counter = 0
    writer = SummaryWriter()
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        # ou_noise.reset()
        # done = False
        ep_reward = 0
        i_step = 0
        # save_i_ep.append(save_i_step)
        # Agents freely explore the environment, preserving sequences
        for i_step in range(cfg.train_step):
            # i_step += 1
            action = agent.choose_action(state)
            action = exploration(action, env_info, r_var, b_var)
            # action = ou_noise.get_action(action, i_step)
            next_state, reward, save_user, trans = env.ddpg_step_forward(np.transpose(action), env_info.r_dim, env_info.b_dim)

            save_i_step.append([i_step, save_user])
            save_trans.append([i_step, trans])
            ep_reward += reward
            agent.memory.push(state, np.transpose(action), reward, next_state)
            # learn
            # agent.update(i_ep, i_step)
            # print("position {}".format(agent.memory.position))
            # if agent.memory.position == agent.memory.capacity:
            #     print("start learning")
            # if agent.memory.position > agent.memory.capacity:
            agent.update(cfg, i_ep, i_step)
            if CHANGE:
                r_var *= .999
                b_var *= .999

            state = next_state

        if i_step == cfg.train_step - 1:
            if (i_ep + 1) % 10 == 0:
                print('episode：{}/{}，ep_reward：{:.2f}'.format(i_ep + 1, cfg.train_eps, ep_reward))
                test_name = cfg.algo + 'ep_reward'
                agent.writer.add_scalar(test_name, ep_reward, i_ep)
            rewards.append(ep_reward)  # 记录总的回报
            var_reward.append(ep_reward)  # 调节探索率的
            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
            else:
                ma_rewards.append(ep_reward)
            # var_reward.append(ep_reward[i_ep])
            if var_counter >= CHECK_EPISODE and np.mean(
                    var_reward[-CHECK_EPISODE:]) >= max_rewards:  # CHECK_EPISODE = 4 # mean()函数功能：求取均值
                CHANGE = True  # 记录max_rewards发生改变
                var_counter = 0  # 一旦max_rewards发生改变，从新开始学习
                max_rewards = np.mean(var_reward[-CHECK_EPISODE:])  # 记录最新的max_rewards
                var_reward = []
            else:
                CHANGE = False
                var_counter += 1
    print('Complete the training !')
    return rewards, ma_rewards, save_i_step, save_trans


def train_ac(cfg, env_info, env, agent):
    print('开始训练! ac')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo}, 设备：{cfg.device}')
    # ou_noise = OUNoise(env.action_space)  #
    rewards = []  # record all the rewards
    ma_rewards = []  # record all the mean rewards based on smoothness
    var_reward = []
    max_rewards = 0
    save_i_step = []
    save_trans = []
    r_var = 1  # the var of resources
    b_var = 1  # the var of bandwidth
    CHECK_EPISODE = 4
    CHANGE = False
    var_counter = 0
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        # ou_noise.reset()
        # done = False
        ep_reward = 0
        i_step = 0
        # Agents freely explore the environment, preserving sequences
        for i_step in range(cfg.train_step):
            # i_step += 1
            action = agent.choose_action(state)
            action = exploration(action, env_info, r_var, b_var)
            # action = ou_noise.get_action(action, i_step)
            next_state, reward, save_user, trans = env.ddpg_step_forward(np.transpose(action), env_info.r_dim, env_info.b_dim)
            save_i_step.append([i_step, save_user])
            save_trans.append([i_step, trans])
            ep_reward += reward
            # agent.memory.push(state, np.transpose(action), reward, next_state)
            # learn
            # agent.update(i_ep, i_step)
            # print("position {}".format(agent.memory.position))
            # if agent.memory.position == agent.memory.capacity:
            #     print("start learning")
            # if agent.memory.position > agent.memory.capacity:
            agent.update(cfg, i_ep, i_step, state, next_state, np.transpose(action), reward)
            if CHANGE:
                r_var *= .999
                b_var *= .999

            state = next_state

        if i_step == cfg.train_step - 1:
            if (i_ep + 1) % 10 == 0:
                print('episode：{}/{}，ep_reward：{:.2f}'.format(i_ep + 1, cfg.train_eps, ep_reward))
                test_name = cfg.algo + 'ep_reward'
                agent.writer.add_scalar(test_name, ep_reward, i_ep)
            rewards.append(ep_reward)  # 记录总的回报
            var_reward.append(ep_reward)  # 调节探索率的
            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
            else:
                ma_rewards.append(ep_reward)
            # var_reward.append(ep_reward[i_ep])
            if var_counter >= CHECK_EPISODE and np.mean(
                    var_reward[-CHECK_EPISODE:]) >= max_rewards:  # CHECK_EPISODE = 4 # mean()函数功能：求取均值
                CHANGE = True  # 记录max_rewards发生改变
                var_counter = 0  # 一旦max_rewards发生改变，从新开始学习
                max_rewards = np.mean(var_reward[-CHECK_EPISODE:])  # 记录最新的max_rewards
                var_reward = []
            else:
                CHANGE = False
                var_counter += 1
    print('完成训练！')
    return rewards, ma_rewards, save_i_step, save_trans


def test(cfg, env, agent):
    print('Begin training !')
    print(f'env：{cfg.env_name}, algo：{cfg.algo_name}, device：{cfg.device}')
    rewards = []  #
    ma_rewards = []  #
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
        print('episode：{}/{}, ep_reward：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print(f"episode：{i_ep + 1}/{cfg.test_eps}，reward：{ep_reward:.1f}")
    print('Complete the training !')
    return rewards, ma_rewards
