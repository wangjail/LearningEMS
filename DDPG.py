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

PATH1 = "....../Models/DDPG/WLTC_"

PATH2 = "....../Result/DDPG/WLTC_"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_args():
    """ 
        超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name', default='DDPG', type=str, help="name of algorithm")
    parser.add_argument('--train_eps', default=500, type=int, help="episodes of training")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=2e-4, type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=1e-4, type=float, help="learning rate of actor")
    parser.add_argument('--memory_capacity', default=100000, type=int, help="memory capacity")
    parser.add_argument('--minimal_size', default=1000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--exploration_noise', default=0.1, type=float)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--hidden_dim1', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    args = parser.parse_args([])    
    args = {**vars(args)}          
    return args


class Actor(nn.Module):
    def __init__(self, states_dim, actions_dim, hidden_dim, hidden_dim1):
        super(Actor, self).__init__()  
        self.linear1 = nn.Linear(states_dim, hidden_dim) 
        self.linear2 = nn.Linear(hidden_dim, hidden_dim1) 
        self.linear3 = nn.Linear(hidden_dim1, actions_dim)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class Critic(nn.Module):
    def __init__(self, states_dim, actions_dim, hidden_dim, hidden_dim1):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(states_dim + actions_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim1)
        self.linear3 = nn.Linear(hidden_dim1, 1)

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


class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim, hidden_dim1, cfg):
        self.actor = Actor(state_dim, action_dim, hidden_dim, hidden_dim1).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, hidden_dim1).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim, hidden_dim1).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim, hidden_dim1).to(device)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = cfg['critic_lr'])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = cfg['critic_lr'])

        self.gamma = cfg['gamma']
        self.tau = cfg['tau']  
        self.writer = SummaryWriter("Logs_WLTC/DDPG_HEV0")

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, transition_dict):
        state = torch.tensor(transition_dict['states'], dtype=torch.float).to(device)
        action = torch.tensor(transition_dict['actions']).view(-1, 1).to(device)
        reward = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(device)
        next_state = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(device)
        done = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(device)

        actor_loss = torch.mean(-self.critic(state, self.actor(state)))

        next_action = self.actor_target(next_state)
        target_value = self.critic_target(next_state, next_action.detach())  
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)
        actual_value = self.critic(state, action)
        critic_loss = nn.MSELoss()(actual_value, expected_value.detach())
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self):
        torch.save(self.actor.state_dict(), PATH1 + 'actor_parameters.path')
        torch.save(self.critic.state_dict(), PATH1 + 'critic1_parameters.path')
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
    agent = DDPG(state_dim, action_dim, hidden_dim, hidden_dim1, cfg)

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
            action = (action + np.random.normal(0, cfg['exploration_noise'], size=env.action_space.shape[0])).clip(
                      env.action_space.low, env.action_space.high)

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
        Cost_list.append(info['Total_cost'])
        SOC_list.append(float(info['SOC']))
        Reward_list.append(episode_return) 

        if total_steps == cfg['train_eps'] - 1:
            agent.deal(Reward_list, Cost_list, SOC_list, SOC_last_list)
            agent.save()

        print("Cost:{} \nSOC :{} \nEpisode:{} \nTotal Reward: {:0.2f} \n".format(info['Total_cost'], float(info['SOC']), total_steps, episode_return))
        

if __name__ == '__main__':
    main()
