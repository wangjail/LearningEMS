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

PATH1 = "....../Models/PPO/WLTC_"

PATH2 = "....../Result/PPO/WLTC_"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_args():
    """ 
        超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name', default='PPO', type=str, help="name of algorithm")
    parser.add_argument('--train_eps', default=500, type=int, help="episodes of training")
    parser.add_argument('--gamma', default=0.95, type=float, help="discounted factor")
    parser.add_argument('--lmbda', default=0.98, type=float)
    parser.add_argument('--eps', default=0.2, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--actor_lr', default=2e-4, type=float)
    parser.add_argument('--critic_lr', default=2e-3, type=float)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--hidden_dim1', default=100, type=int)
    parser.add_argument('--seed', default=1, type=int, help="random seed")
    args = parser.parse_args([])    
    args = {**vars(args)}          
    return args


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, hidden_dim1, action_dim, max_action):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim1)
        self.fc_mu = torch.nn.Linear(hidden_dim1, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim1, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.max_action * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, hidden_dim1):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim1)
        self.fc3 = torch.nn.Linear(hidden_dim1, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPOContinuous:
    def __init__(self, state_dim, hidden_dim, hidden_dim1, action_dim, max_action, cfg):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, hidden_dim1, action_dim, max_action).to(device)
        self.critic = ValueNet(state_dim, hidden_dim, hidden_dim1).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = cfg['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = cfg['critic_lr'])

        self.gamma = cfg['gamma']
        self.lmbda = cfg['lmbda']
        self.epochs = cfg['epochs']
        self.eps = cfg['eps']
        self.writer = SummaryWriter("Logs_WLTC/PPO_HEV0")

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return [action.item()]

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),dtype=torch.float).to(device)
        actions = torch.tensor(np.array(transition_dict['actions']),dtype=torch.float).view(-1, 1).to(device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),dtype=torch.float).to(device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save(self):
        torch.save(self.actor.state_dict(), PATH1 + 'actor_parameters.path')
        torch.save(self.critic.state_dict(), PATH1 + 'critic_parameters.path')
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

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


def main():
    env = PriusEnv()
    cfg = get_args()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    hidden_dim = cfg['hidden_dim']
    hidden_dim1 = cfg['hidden_dim1']

    agent = PPOContinuous(state_dim, hidden_dim, hidden_dim1, action_dim, max_action, cfg)

    Cost_list = []
    Reward_list = []
    SOC_list = []
    SOC_last_list = []
    num_SOC_last = 0
    for total_steps in range(cfg['train_eps']):

        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)

            if total_steps == cfg['train_eps'] - 2:
                SOC_last_list.append(float(info['SOC']))
                num_SOC_last += 1
                agent.writer.add_scalar('SOC_last', float(info['SOC']), global_step = num_SOC_last)

            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
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