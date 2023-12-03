import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections 
import random
import argparse
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from env.PriusV0 import PriusEnv

PATH1 = "....../Models/DQN/NEDC_"

PATH2 = "....../Result/DQN/NEDC_"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_args():
    """ 
        超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name', default='DQN', type=str, help="name of algorithm")
    parser.add_argument('--train_eps', default=500, type=int, help="episodes of training")
    parser.add_argument('--gamma', default=0.98, type=float, help="discounted factor")
    parser.add_argument('--learning_rate', default=2e-4, type=float, help="learning rate of critic")
    parser.add_argument('--memory_capacity', default=100000, type=int, help="memory capacity")
    parser.add_argument('--minimal_size', default=1000, type=int, help="memory capacity")
    parser.add_argument('--epsilon', default=1, type=int)
    parser.add_argument('--eps_min', default=0.005, type=float)
    parser.add_argument('--eps_dec', default=5e-5, type=float)
    parser.add_argument('--target_update', default=50, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--hidden_dim1', default=100, type=int)
    parser.add_argument('--seed', default=250, type=int, help="random seed")
    args = parser.parse_args([])    
    args = {**vars(args)}          
    return args


class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, hidden_dim1, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim1)   # 共享网络部分
        self.fc3 = torch.nn.Linear(hidden_dim1, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self): 
        return len(self.buffer)


class DQN:
    def __init__(self, state_dim, hidden_dim, hidden_dim1, action_dim, cfg):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, hidden_dim1, self.action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, hidden_dim1, self.action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr = cfg['learning_rate'])

        self.gamma = cfg['gamma']
        self.epsilon = cfg['epsilon']
        self.eps_min = cfg['eps_min']
        self.eps_dec = cfg['eps_dec']
        self.target_update = cfg['target_update']
        self.count = 0
        self.writer = SummaryWriter("Logs_NEDC/DQN_HEV1")

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec 
        else: 
            self.epsilon = self.eps_min

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(device)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        ddqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        ddqn_loss.backward()
        self.optimizer.step()

        self.decrement_epsilon()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1

    def save(self):
        torch.save(self.q_net.state_dict(), PATH1 + 'actor_parameters.path')
        print("====================================")
        print("Model has been saved!!!")
        print("====================================")

    def deal(self, list0, list1, list2, list3):
        df0 = pd.DataFrame(list0, columns=['Reward'])
        df1 = pd.DataFrame(list1, columns=['Cost'])
        df2 = pd.DataFrame(list2, columns=['SOC'])
        df3 = pd.DataFrame(list3, columns=['SOC_Last'])
        df0.to_excel(PATH2 + "Reward.xlsx", index=False)
        df1.to_excel(PATH2 + "Cost.xlsx", index=False)
        df2.to_excel(PATH2 + "SOC.xlsx", index=False)
        df3.to_excel(PATH2 + "SOC_Last.xlsx", index=False)

def dis_to_con(discrete_action, env, action_dim):
    action_lowbound = env.action_space.low[0]
    action_upbound = env.action_space.high[0]
    return action_lowbound + (discrete_action /(action_dim - 1)) * (action_upbound -action_lowbound)


def main():
    env = PriusEnv()
    cfg = get_args()

    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])

    state_dim = env.observation_space.shape[0]
    action_dim = 21
    hidden_dim = cfg['hidden_dim']
    hidden_dim1 = cfg['hidden_dim1']
    replay_buffer = ReplayBuffer(cfg['memory_capacity'])
    agent = DQN(state_dim, hidden_dim, hidden_dim1, action_dim, cfg)

    Cost_list = []
    Reward_list = []
    SOC_list = []
    SOC_last_list = []
    num_SOC_last = 0
    for total_steps in range(cfg['train_eps']):  

        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            action_continuous = dis_to_con(action, env,agent.action_dim)
            next_state, reward, done, info = env.step([action_continuous])

            if total_steps == cfg['train_eps'] - 2:
                SOC_last_list.append(float(info['SOC']))
                num_SOC_last += 1
                agent.writer.add_scalar('SOC_last', float(info['SOC']), global_step = num_SOC_last)
           
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            if replay_buffer.size() > cfg['minimal_size']:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(cfg['batch_size'])
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                agent.update(transition_dict)

        agent.writer.add_scalar('Reward', episode_return, global_step = total_steps)
        agent.writer.add_scalar('Cost', info['Total_cost'], global_step = total_steps)
        agent.writer.add_scalar('SOC', info['SOC'], global_step = total_steps)
        Cost_list.append(info['Total_cost'])
        SOC_list.append(info['SOC'])
        Reward_list.append(episode_return) 

        if total_steps == cfg['train_eps'] - 1:
            agent.deal(Reward_list, Cost_list, SOC_list, SOC_last_list)
            agent.save()

        print("Cost:{} \nSOC :{} \nEpisode:{} \nTotal Reward: {:0.2f} \n".format(info['Total_cost'], info['SOC'], total_steps, episode_return))

if __name__ == '__main__':
    main()