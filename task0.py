#!/usr/bin/env python
# coding=utf-8
'''
@Author: kangkang sun
@Email: szpsunkk@163.com
@Date: 2022.5.26
@LastEditor: kangkang sun
LastEditTime:
@Discription:  the main file includes the configuration parameters
@Environment: python 3.7.7
'''
import os
import sys

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径sys.path

import datetime
# import gym
import torch
from env import Env

# from env import NormalizedActions
from agent_DDPG import DDPG
from agent_TD3 import TD3
from train import train, train_ac
from common.utils import save_results, make_dir, save_user
from common.utils import plot_rewards_ddpg, plot_rewards_td3
from map import map as map_beijing
from agent_DDPG_mfg import DDPG_MFG
from agent_TD3_mfg import TD3_MFG
from agent_ac1 import AC

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
start = datetime.datetime.now()
ALGO = 'DDPG'  # 采用的算法, DDPG,DDPG-MFG,TD3,TD3-MFG，AC 五种算法， FedDDPG-MFG, FedTD3-MFG
HIDDEN_DIM = 52  # 采用的隐藏层数量
MEMMORY_CAPACITY = 8000  # 存储大小
BATCH_SIZE = 512
GAMMA = 0.9
TRAIN_EPS = 80
TRAIN_STEP = 200

#########  DDPG parameters #########
C_LR = 0.0001
A_LR = 0.0001
SOFT_TAU = 0.01

######## SAC parameters ############
ALPHA = 0.2
POLICY = "Gaussian"  # Gaussian  or Deterministic
target_update_interval = 1
automatic_entropy_tuning = False
LR = 0.0001


class Env_info:
    def __init__(self, s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, task_inf, limit, location, user_number,
                 edge_number, B, P):
        self.s_dim = s_dim  # state 维度
        self.r_dim = r_dim  # resource 维度
        self.b_dim = b_dim  # 带宽维度
        self.o_dim = o_dim  # observation维度
        self.a_dim = self.r_dim + self.b_dim + self.o_dim  # action维度
        self.r_bound = r_bound  # 资源最大
        self.b_bound = b_bound  # 带宽最大
        self.task_info = task_inf  # 任务信息
        self.limit = limit  # 卸载限制
        self.location = location  # 位置
        self.user = user_number
        self.edge = edge_number
        self.B = B
        self.P = P
        self.map_plot = False  # 是否位置图
        if self.map_plot:
            path = 'D:\\MEC\\MEC\\map\\123.png'
            m = map_beijing(path)
            m.plot()


class AC_Config:
    def __init__(self):
        self.algo = "AC"  # 算法名称
        self.env_name = "Vehicle edge computing (VEC) for digital twin"  # 环境名称
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = TRAIN_EPS  # 训练的回合数
        self.train_step = TRAIN_STEP
        self.test_eps = 50  # 测试的回合数
        self.gamma = GAMMA  # 折扣因子
        self.critic_lr = C_LR  # 评论家网络的学习率
        self.actor_lr = A_LR  # 演员网络的学习率
        self.memory_capacity = MEMMORY_CAPACITY  # 经验回放的容量
        self.batch_size = BATCH_SIZE  # mini-batch SGD中的批量大小
        self.target_update = 2  # 目标网络的更新频率
        self.hidden_dim = HIDDEN_DIM  # 网络隐藏层维度
        self.soft_tau = SOFT_TAU  # 软更新参数
        self.graph = False


class SAC_Config:
    def __init__(self):
        self.algo = "SAC"
        self.env_name = "Vehicle edge computing (VEC) for digital twin"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_eps = TRAIN_EPS  # 训练的回合数
        self.train_step = TRAIN_STEP
        self.gamma = GAMMA  # 折扣因子
        self.tau = SOFT_TAU
        self.alpha = ALPHA
        self.policy = POLICY
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.lr = LR
        self.hidden_dim = HIDDEN_DIM  # 网络隐藏层维度


class DDPG_Config:
    def __init__(self):
        self.algo = "DDPG"  # 算法名称
        self.env_name = "Vehicle edge computing (VEC) for digital twin"  # 环境名称
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = TRAIN_EPS  # 训练的回合数
        self.train_step = TRAIN_STEP
        self.test_eps = 50  # 测试的回合数
        self.gamma = GAMMA  # 折扣因子
        self.critic_lr = C_LR  # 评论家网络的学习率
        self.actor_lr = A_LR  # 演员网络的学习率
        self.memory_capacity = MEMMORY_CAPACITY  # 经验回放的容量
        self.batch_size = BATCH_SIZE  # mini-batch SGD中的批量大小
        self.target_update = 2  # 目标网络的更新频率
        self.hidden_dim = HIDDEN_DIM  # 网络隐藏层维度
        self.soft_tau = SOFT_TAU  # 软更新参数
        self.graph = False


class DDPG_MFG_Config:
    def __init__(self):
        self.algo = "DDPG-MFG"  # 算法名称
        self.env_name = "Vehicle edge computing (VEC) for digital twin"  # 环境名称
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = TRAIN_EPS  # 训练的回合数
        self.train_step = TRAIN_STEP
        self.test_eps = 50  # 测试的回合数
        self.gamma = GAMMA  # 折扣因子
        self.critic_lr = C_LR  # 评论家网络的学习率
        self.actor_lr = A_LR  # 演员网络的学习率
        self.memory_capacity = MEMMORY_CAPACITY  # 经验回放的容量
        self.batch_size = BATCH_SIZE  # mini-batch SGD中的批量大小
        self.target_update = 2  # 目标网络的更新频率
        self.hidden_dim = HIDDEN_DIM  # 网络隐藏层维度
        self.soft_tau = SOFT_TAU  # 软更新参数
        self.graph = False


class TD3_Config:
    def __init__(self) -> None:
        self.algo = 'TD3'  # 算法名称
        self.env_name = 'Vehicle edge computing (VEC) for digital twin'  # 环境名称
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = TRAIN_EPS  # 训练的回合数
        self.train_step = TRAIN_STEP  # 训练步数
        self.start_timestep = 25e3  # Time steps initial random policy is used
        self.epsilon_start = 50  # Episodes initial random policy is used
        self.eval_freq = 10  # How often (episodes) we evaluate
        self.max_timestep = 100000  # Max time steps to run environment
        self.expl_noise = 0.1  # Std of Gaussian exploration noise
        self.batch_size = BATCH_SIZE  # Batch size for both actor and critic
        self.gamma = GAMMA  # gamma factor
        self.lr = 0.0005  # 学习率
        self.policy_noise = 0.2  # Noise added to target policy during critic update
        self.noise_clip = 0.3  # Range to clip target policy noise
        self.policy_freq = 2  # Frequency of delayed policy updates
        self.hidden_dim = HIDDEN_DIM  # 网络隐藏层维度
        self.memory_capacity = MEMMORY_CAPACITY  # 经验回放的容量


class TD3_MFG_Config:
    def __init__(self) -> None:
        self.algo = 'TD3-MFG'  # 算法名称
        self.env_name = 'Vehicle edge computing (VEC) for digital twin'  # 环境名称
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = TRAIN_EPS  # 训练的回合数
        self.train_step = TRAIN_STEP  # 训练步数
        self.start_timestep = 25e3  # Time steps initial random policy is used
        self.epsilon_start = 50  # Episodes initial random policy is used
        self.eval_freq = 10  # How often (episodes) we evaluate
        self.max_timestep = 100000  # Max time steps to run environment
        self.expl_noise = 0.1  # Std of Gaussian exploration noise
        self.batch_size = BATCH_SIZE  # Batch size for both actor and critic
        self.gamma = GAMMA  # gamma factor
        self.lr = 0.0005  # 学习率
        self.policy_noise = 0.2  # Noise added to target policy during critic update
        self.noise_clip = 0.3  # Range to clip target policy noise
        self.policy_freq = 2  # Frequency of delayed policy updates
        self.hidden_dim = HIDDEN_DIM  # 网络隐藏层维度
        self.memory_capacity = MEMMORY_CAPACITY  # 经验回放的容量


class PlotConfig:
    def __init__(self, algo_name, env_name):
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.gamma = 0.9
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU


def env_agent_config(cfg, env):
    # env = NormalizedActions(gym.make(cfg.env_name))  #
    # env.seed(seed)  # 随机种子
    s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, task_inf, limit, location, user_number, edge_number, B, P = env.get_inf()
    state_dim = s_dim
    action_dim = r_dim + b_dim + o_dim
    action_dim_mean = 1 + 1 + o_dim
    e = Env_info(s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, task_inf, limit, location, user_number, edge_number, B,
                 P)
    if ALGO == 'DDPG':
        agent = DDPG(state_dim, action_dim, cfg, e)
    elif ALGO == 'TD3':
        agent = TD3(state_dim, action_dim, cfg, e)
    elif ALGO == 'DDPG-MFG':
        agent = DDPG_MFG(state_dim, action_dim, action_dim_mean, cfg, e)
    elif ALGO == 'TD3-MFG':
        agent = TD3_MFG(state_dim, action_dim, action_dim_mean, cfg, e)
    elif ALGO == 'AC':
        agent = AC(state_dim, action_dim, cfg, e)
    return agent, e


if ALGO == 'DDPG':
    cfg = DDPG_Config()
elif ALGO == 'TD3':
    cfg = TD3_Config()
elif ALGO == 'DDPG-MFG':
    cfg = DDPG_MFG_Config()
elif ALGO == 'TD3-MFG':
    cfg = TD3_MFG_Config()
elif ALGO == 'AC':
    cfg = AC_Config()

plot_cfg = PlotConfig(cfg.algo, cfg.env_name)
env = Env()
# 训练
agent, env_info = env_agent_config(cfg, env)
if ALGO == 'DDPG' or ALGO == 'DDPG-MFG' or ALGO == 'TD3' or ALGO == 'TD3-MFG':
    rewards, ma_rewards, user, trans = train(cfg, env_info, env, agent)
elif ALGO == 'AC':
    rewards, ma_rewards, user, trans = train_ac(cfg, env_info, env, agent)
make_dir(plot_cfg.result_path, plot_cfg.model_path)
agent.save(path=plot_cfg.model_path)
save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)
save_user(user, trans, ALGO, tag='train', path=plot_cfg.result_path)
end = datetime.datetime.now()
print('totally time is {}'.format(end - start))
if cfg.algo == 'DDPG' or cfg.algo == 'DDPG-MFG' or cfg.algo == 'AC':  # DDPG and DDPG-MFG algo  plot
    plot_rewards_ddpg(rewards, ma_rewards, cfg, env_info, plot_cfg, tag="train")  # 画出结果
elif cfg.algo == 'TD3' or cfg.algo == 'TD3-MFG':
    plot_rewards_td3(rewards, ma_rewards, cfg, env_info, plot_cfg, tag="train")
    
# # 测试
# env, agent = env_agent_config(cfg, seed=10)
# agent.load(path=plot_cfg.model_path)
# rewards, ma_rewards = test(cfg, env, agent)
# save_results(rewards, ma_rewards, tag='test', path=plot_cfg.result_path)
# plot_rewards(rewards, ma_rewards, plot_cfg, tag="test")  # 画出结果
