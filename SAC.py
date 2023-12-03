import torch
import collections 
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from env.PriusV0 import PriusEnv

PATH1 = "....../Models/SAC/WLTC_"

PATH2 = "....../Result/SAC/WLTC_"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_args():
    """ 
        超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name', default='SAC', type=str, help="name of algorithm")
    parser.add_argument('--train_eps', default=500, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--actor_lr', default=2e-4, type=float)
    parser.add_argument('--critic_lr', default=3e-3, type=float)
    parser.add_argument('--alpha_lr', default=3e-4, type=float)
    parser.add_argument('--memory_capacity', default=100000, type=int, help="memory capacity")
    parser.add_argument('--minimal_size', default=1000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--soft_tau', default=0.005, type=float)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--hidden_dim1', default=100, type=int)
    parser.add_argument('--seed', default=1, type=int, help="random seed")
    args = parser.parse_args([])    
    args = {**vars(args)}         
    return args


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)

        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, hidden_dim1, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim1)
        self.fc_out = torch.nn.Linear(hidden_dim1, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(100000)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size 
        self.size = min(self.size + 1, self.max_size) 

    def sample(self,batch_size):
        index = np.random.choice(self.size, size=batch_size)
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class SACContinuous:
    def __init__(self, state_dim, action_dim, hidden_dim, hidden_dim1, action_bound, target_entropy, cfg):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)  
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, hidden_dim1, action_dim).to(device)  
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, hidden_dim1, action_dim).to(device)  
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, hidden_dim1, action_dim).to(device)  
        self.target_critic_2 = QValueNetContinuous(state_dim,hidden_dim, hidden_dim1, action_dim).to(device)  
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = cfg['actor_lr'])
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),lr = cfg['critic_lr'])
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),lr = cfg['critic_lr'])

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True 
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr = cfg['alpha_lr'])
        self.writer = SummaryWriter("Logs_WLTC/SAC_HEV0")

        self.target_entropy = target_entropy
        self.gamma = cfg['gamma']
        self.tau = cfg['soft_tau']
        self.batch_size = cfg['batch_size']
        self.memory = ReplayBuffer(state_dim, action_dim)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)
        action = self.actor(state)[0].detach().cpu()
        action = action.clamp(-1, 1)
        return action.item()

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self):
        if self.memory.size < self.batch_size: 
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        states = torch.tensor(state).to(device)
        actions = torch.tensor(action).view(-1, 1).to(device)
        rewards = torch.tensor(reward).view(-1, 1).to(device)
        next_states = torch.tensor(next_state,dtype=torch.float).to(device)
        dones = torch.tensor(done).view(-1, 1).to(device)

        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

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
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    target_entropy = -env.action_space.shape[0]
    hidden_dim = cfg['hidden_dim']
    hidden_dim1 = cfg['hidden_dim1']
    agent = SACContinuous(state_dim, action_dim, hidden_dim, hidden_dim1, action_bound, target_entropy, cfg)

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

            if total_steps < cfg['test_eps']: 
                action = env.action_space.sample()
            else:
                action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            
            if total_steps == cfg['train_eps'] - 2:
                SOC_last_list.append(float(info['SOC']))
                num_SOC_last += 1
                agent.writer.add_scalar('SOC_last', float(info['SOC']), global_step = num_SOC_last)

            agent.memory.store(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            
            if total_steps >= cfg['test_eps']:
                agent.update()
       
        agent.writer.add_scalar('Cost', info['Total_cost'], global_step = total_steps)
        agent.writer.add_scalar('Reward', episode_return, global_step = total_steps)
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