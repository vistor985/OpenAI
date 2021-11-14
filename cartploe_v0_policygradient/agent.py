import torch
from torch.distributions import Bernoulli
from torch.autograd import Variable
import numpy as np
from model import *


class PolicyGradient(object):
    def __init__(self, state_dim, cfg):
        self.gamma = cfg.gamma  # 回报的折扣因子
        self.policy_net = MLP(state_dim, hidden_dim=cfg.hidden_dim)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=cfg.lr)
        self.batch_size = cfg.batch_size

    def choose_action(self, state):
        state = torch.from_numpy(state).float()
        state = Variable(state)
        probs = self.policy_net(state)  # 为0到1之间的数(sigmoid)
        m = Bernoulli(probs)  # 伯努利分布，使用伯努利的原因是在该环境下的取值只有0或1，m为取1的概率
        action = m.sample()  # 根据概率做出下一步决策
        print(action)
        action = action.data.numpy().astype(int)[0]  # 转为标量
        return action

    def update(self, reward_pool, state_pool, action_pool):
        # 将回报折现
        running_add = 0
        for i in reversed(range(len(reward_pool))):  # reversed反转
            if reward_pool[i] == 0:  # 分隔batchsize里不同的episode
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add

        # 标准化回报，这一步的处理就是加上基准的意思，使得整个回报处在有正有负的环境下
        reward_mean = np.mean(reward_pool)  # 均值
        reward_std = np.std(reward_pool)  # 标准差
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

        # 梯度上升
        print("reward_pool:", reward_pool)
        self.optimizer.zero_grad()
        for i in range(len(reward_pool)):
            state = state_pool[i]
            action = Variable(torch.FloatTensor([action_pool[i]]))
            reward = reward_pool[i]
            state = Variable(torch.from_numpy(state).float())
            probs = self.policy_net(state)
            m = Bernoulli(probs)
            loss = -m.log_prob(action) * reward  # Negative score function x reward
            loss.backward()  # 反向传播
        self.optimizer.step()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + "pg_checkpoint.pt")

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + "pg_checkpoint.pt"))
