import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    '''
    多层感知机
    输入：state维度
    输出：概率
    '''

    def __init__(self, state_dim, hidden_dim=36):
        super(MLP, self).__init__()
        # 24和36为隐藏层的层数，可根据state_dim, action_dim的情况来改变
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
    '''
    以上是MLP也就是多层感知机的实现，其实是最简单的神经网络，只是用了几层全连接层进行
一个训练，其中的激活函数是relu,最后通过sigmoid函数返回一个类概率，也就是概率。接下来
就是根据概率去做动作选择，再进行一个采样的过程
    '''