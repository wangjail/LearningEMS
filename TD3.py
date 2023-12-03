import torch
import torch.nn as nn
import torch.nn.functional as F
import collections 
import random
import argparse
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from env.PriusV0 import PriusEnv

PATH1 = "....../Models/TD3/WLTC_"

PATH2 = "....../Result/TD3/WLTC_"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_args():
    """ 
        超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name', default='TD3', type=str, help="name of algorithm")
    parser.add_argument('--train_eps', default=500, type=int, help="episodes of training")
    parser.add_argument('--explore_steps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=2e-4, type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=2e-4, type=float, help="learning rate of actor")
    parser.add_argument('--memory_capacity', default=100000, type=int, help="memory capacity")
    parser.add_argument('--minimal_size', default=1000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--policy_freq', default=2, type=int)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--policy_noise', default=0.15, type=float)
    parser.add_argument('--expl_noise', default=0.1, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--hidden_dim1', default=100, type=int)
    parser.add_argument('--seed', default=1, type=int, help="random seed")
    args = parser.parse_args([])    
    args = {**vars(args)}         
    return args

class Actor(nn.Module):
    def __init__(self, states_dim, actions_dim, hidden_dim, hidden_dim1, init_w=3e-3):
        super(Actor, self).__init__()  
        self.linear1 = nn.Linear(states_dim, hidden_dim) 
        self.linear2 = nn.Linear(hidden_dim, hidden_dim1) 
        self.linear3 = nn.Linear(hidden_dim1, actions_dim)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x
        
class Critic(nn.Module):
    def __init__(self, states_dim, actions_dim, hidden_dim, hidden_dim1, init_w=3e-3):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(states_dim + actions_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim1)
        self.linear3 = nn.Linear(hidden_dim1, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1).float()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


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


class TD3:
    def __init__(self, state_dim, action_dim, hidden_dim, hidden_dim1, env, cfg):
        self.actor = Actor(state_dim, action_dim, hidden_dim, hidden_dim1).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, hidden_dim1).to(device)
        self.critic_1 = Critic(state_dim, action_dim, hidden_dim, hidden_dim1).to(device)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim, hidden_dim1).to(device)
        self.critic_1_target = Critic(state_dim, action_dim, hidden_dim, hidden_dim1).to(device)
        self.critic_2_target = Critic(state_dim, action_dim, hidden_dim, hidden_dim1).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict()) 
        self.critic_2_target.load_state_dict(self.critic_2.state_dict()) 
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = cfg['actor_lr'])
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr = cfg['critic_lr'])
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr = cfg['critic_lr'])
        self.writer = SummaryWriter("Logs_WLTC/TD3_HEV0")

        self.gamma = cfg['gamma'] 
        self.policy_noise = cfg['policy_noise']
        self.noise_clip = cfg['noise_clip']
        self.expl_noise = cfg['expl_noise']
        self.tau = cfg['tau']
        self.sample_count = 0
        self.policy_freq = cfg['policy_freq']
        self.explore_steps = cfg['explore_steps']
        self.action_dim = action_dim
        self.action_space = env.action_space
        self.action_scale = torch.tensor((self.action_space.high - self.action_space.low)/2, dtype=torch.float32).unsqueeze(dim=0).to(device)
        self.action_bias = torch.tensor((self.action_space.high + self.action_space.low)/2, dtype=torch.float32).unsqueeze(dim=0).to(device)

    def sample_action(self, state):
        self.sample_count += 1
        if self.sample_count < self.explore_steps:
            return self.action_space.sample()
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0).to(device)
            action = self.actor(state).cpu().data.numpy().flatten()
            return action

    def update(self, transition_dict):
        state = torch.tensor(transition_dict['states'], dtype=torch.float).to(device)
        action = torch.tensor(transition_dict['actions']).view(-1, 1).to(device)
        reward = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(device)
        next_state = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(device)
        done = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(device)

        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.action_scale + self.action_bias, self.action_scale+self.action_bias)
        target_q1, target_q2 = self.critic_1_target(next_state, next_action).detach(), self.critic_2_target(next_state, next_action).detach()
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + self.gamma * target_q * (1 - done)
        current_q1, current_q2 = self.critic_1(state, action), self.critic_2(state, action)
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        if self.sample_count % self.policy_freq == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self):
        torch.save(self.actor.state_dict(), PATH1 + 'actor_parameters.path')
        torch.save(self.critic_1.state_dict(), PATH1 + 'critic1_parameters.path')
        torch.save(self.critic_2.state_dict(), PATH1 + 'critic2_parameters.path')
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


def main():
    env = PriusEnv()
    cfg = get_args()

    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = cfg['hidden_dim']
    hidden_dim1 = cfg['hidden_dim1']
    replay_buffer = ReplayBuffer(cfg['memory_capacity'])
    agent = TD3(state_dim, action_dim, hidden_dim, hidden_dim1, env, cfg)

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
            action = agent.sample_action(state)
            next_state, reward, done, info = env.step(action)

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
        Reward_list.append(episode_return)
        Cost_list.append(info['Total_cost'])
        SOC_list.append(float(info['SOC']))

        if total_steps == cfg['train_eps'] - 1:
            agent.deal(Reward_list, Cost_list, SOC_list, SOC_last_list)
            agent.save()

        print("Cost:{} \nSOC :{} \nEpisode:{} \nTotal Reward: {:0.2f} \n".format(info['Total_cost'], float(info['SOC']), total_steps, episode_return))
        

if __name__ == '__main__':
    main()

