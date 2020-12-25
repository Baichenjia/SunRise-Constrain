# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim

from model import DQN


class Agent:
    def __init__(self, args, env):
        self.action_space = env.action_space()
        self.atoms = args.atoms              # 51
        self.Vmin = args.V_min               # -10
        self.Vmax = args.V_max               # 10
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size    # 32
        self.n = args.multi_step             # 3
        self.discount = args.discount        # 0.99

        self.online_net = DQN(args, self.action_space).to(device=args.device)
        if args.model:                       # Load pre-trained model if provided
            if os.path.isfile(args.model):
                state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
                if 'conv1.weight' in state_dict.keys():
                    for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
                        state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pre-trained models
                        del state_dict[old_key]  # Delete old keys for strict load_state_dict
                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + args.model)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(args.model)

        self.online_net.train()   # torch 固有方法，将给模块设置为 train 模式

        self.target_net = DQN(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):    # state.shape=(4, 84, 84)
        with torch.no_grad():
            # 先经过 online_net 得到 Q值输出为[1, n_action, 51], 最后一维是 softmax
            # support.shape=(51,)是从 -10 到 10 的数. 乘以 support 之后为 (1, n_action, 51).
            # 相当于用 support 来加权 Q. 随后取 sum-argmax 得到动作
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

    # Get Q-function
    def ensemble_q(self, state):   # state.shape=(4, 84, 4)
        with torch.no_grad():      # 输出为 support 加权后的 Q.value. shape=(1, n_action)
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2)
        
    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)
    
    # Compute target Q-value
    def get_target_q(self, next_states):
        with torch.no_grad():
            pns = self.online_net(next_states)         # (None, n_action, 51)  Q值的分布  softmax
            dns = self.support.expand_as(pns) * pns    # (None, n_action, 51)  扩展 support 的维度, 对Q加权
            argmax_indices_ns = dns.sum(2).argmax(1)   # (None,)               得到 greedy 选择的 action
            pns = self.target_net(next_states)         # (None, n_action, 51)  target-Q值
            pns_a = pns[range(self.batch_size), argmax_indices_ns]   # (None, 51)  取出greedy选择位置的 Q-value (Double-Q)
            pns_a = pns_a * self.support.expand_as(pns_a)            # (None, 51)  用 support 加权
        return pns_a.sum(1)                                          # (None,)
    
    # Compute Q-value
    def get_online_q(self, states):
        with torch.no_grad():
            pns = self.online_net(states.unsqueeze(0))  # (None, n_action, 51)  Q值-softmax
            dns = self.support.expand_as(pns) * pns     # (None, n_action, 51)  扩展 support 的维度, 对Q加权
        return dns.sum(2)                               # (None, n_action)

    def learn(self, mem):
        # Sample transitions. shapes.shape=next_states.shape=(32,4,84,84). actions.shape=(32,) nonterminals.shape=(32,)其实0-1值. weights为全1向量
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)          # Log probabilities log p(s_t, ·; θonline)  (None, n_action, 51)  Q值-log-softmax
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)  shape=(None, 51)  被选择的action

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)  shape=(None, n_action, 51)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))   shape=(None, n_action, 51)
            argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]  # shape=(None)
            self.target_net.reset_noise()  # Sample new target net noise
            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)   shape=(None, n_action, 51)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)  shape=(None, 51)

            # Compute Tz (Bellman operator T applied to z)
            # reward + γ^n * support 是根据γ和reward对 support 进行平移和伸缩. Tz.shape=(None,51)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values   (None,51)
            # Compute L2 projection of Tz onto fixed support z ,  判断当前的 Tz 处于哪个atom
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)   # l.shape=u.shape=(None,51)
            # Fix disappearing probability mass when l = b = u (b is int). 具体原因不明，但是这两个操作一般不会改变l,u的值
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz. 投影操作, b计算和l,u之间的距离,距离越大权重越小. 将 pns_a投影到l,u所在的atom.
            m = states.new_zeros(self.batch_size, self.atoms)  # m.shape=(None,51). 全0. 与 states 无关
            # offset.shape=(32,51). 每行的数值相同. 第一行的数值为0, 最后一行的数值为 31*51=1581.
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t))) 交叉熵损失，忽略KL中的第一项，保留交叉熵
        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        self.optimiser.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions
        
    def ensemble_learn(self, idxs, states, actions, returns, next_states, nonterminals, weights, masks, weight_Q=None):
        # 多了一个 mask 作为输入
        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            self.target_net.reset_noise()  # Sample new target net noise
            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        self.online_net.zero_grad()
        if weight_Q is None:
            (weights* masks * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        else:
            (weight_Q * weights * masks * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        self.optimiser.step()
        
        return loss.detach().cpu().numpy()

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()


if __name__ == '__main__':
    class ARGS:
        atoms = 51
        architecture = 'data-efficient'
        history_length = 4
        hidden_size = 512
        noisy_std = 0.1
    args = ARGS()
    x = torch.randn(size=(10, 4, 84, 84))
    dqn = DQN(args, action_space=4)
    y = dqn(x)
    print(y.shape)

    # act
    support = torch.linspace(-10, 10, 51)
    state = torch.randn((4, 84, 84))
    action = (dqn(state.unsqueeze(0)) * support).sum(2).argmax(1).item()
    # print(state.unsqueeze(0).shape, dqn(state.unsqueeze(0)).shape, "\n", (dqn(state.unsqueeze(0))*support).shape)
    print("action:", action)

    # Target Q
    next_states = torch.randn((32, 4, 84, 84))
    pns = dqn(next_states)  # (None, n_action, 51)
    dns = support.expand_as(pns) * pns
    action = dns.sum(2).argmax(1)

    print(action)
    y = pns[np.arange(0, 32), action]
    print(y.shape)
