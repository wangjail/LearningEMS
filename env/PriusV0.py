import numpy as np
import scipy.io as scio
import gym
from gym import spaces
import os
import sys

sys.path.append('....../env/units')
from env.units.Prius_model_base import Prius_model

# 单个工况测试
path = "....../test_data"

class PriusEnv():
    # metadata = {'render.modes': ['human']}

    def __init__(self):

        # super(PriusEnv, self).__init__()
        self.dt = 1
        self.steps = 0
        self.action_space = spaces.Box(low= 0, high=1.0, shape=(1, ), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=np.array([0,0,-3]),
                                            high=np.array([1,40,3]),
                                            dtype=np.float32)
        self.viewer = None

    def reset(self):

        self.SOC = 0.6
        self.ep_reward = 0
        self.steps = 0
        self.total_cost = 0 

        self.done = False
        
        ## 多工况训练
        # path = "E:/Document/Experients/HEV_EMS/training_data"
        path_list = os.listdir(path)
        # random_data = np.random.randint(0,len(path_list))
        # self.base_data = path_list[random_data]
        # print(self.base_data)

        ## 单工况训练
        self.base_data = path_list[2]
        #print(self.base_data)

        self.state = np.array([0.6, 0, 0])
        return self.state


    def step(self, action):

        # path = "E:/Document/Experients/HEV_EMS/training_data"
        
        data = scio.loadmat(path + '/' + self.base_data)
        car_spd_one = data['speed_vector']            
        SOC_origin = 0.6
        total_milage = np.sum(car_spd_one) / 1000 
        action = np.clip(action, 0, 1)

        Eng_pwr_opt = action * 56000
 
        self.velocity = car_spd_one[:, self.steps+1]                     

        self.acceleration = car_spd_one[:, self.steps+1] - car_spd_one[:,self.steps]  
        
        soc  = self.SOC
        v = self.velocity
        a =  self.acceleration
       
        ###  模型数据
        out, cost, I = Prius_model().run(v, a, Eng_pwr_opt, soc)
        self.P_req = float(out['P_req'])
        self.Eng_spd = float(out['Eng_spd'])
        self.Eng_trq= float(out['Eng_trq']) 
        self.Eng_pwr = float(out['Eng_pwr'])
        self.Eng_pwr_opt = float(out['Eng_pwr_opt'])
        self.Mot_spd = float(out['Mot_spd'])
        self.Mot_trq = float(out['Mot_trq'])       
        self.Mot_pwr= float(out['Mot_pwr'])  
        self.Gen_spd = float(out['Gen_spd'])
        self.Gen_trq = float(out['Gen_trq'])       
        self.Gen_pwr = float(out['Gen_pwr'])
        self.Batt_pwr = float(out['Batt_pwr'])   

        self.Mot_eta = float(out['Mot_eta'])
        self.Gen_eta = float(out['Gen_eta'])
        self.T_list = float(out['T']) 
        
        
        self.SOC = float(out['SOC'])     
        next_state = np.hstack([self.SOC, self.velocity, self.acceleration])
        cost = float(cost)
        reward = - cost            
        if self.SOC < 0.5 or self.SOC > 0.85:
            reward = - ((350 * ((0.5 - self.SOC) ** 2)) + cost)

        self.ep_reward = cost + self.ep_reward
        self.total_cost = self.total_cost + cost

        cost_Engine =  (self.ep_reward / 0.72 / 1000) + (self.SOC < SOC_origin) * (SOC_origin - self.SOC) * (201.6 * 6.5) * 3600 /(42600000) / 0.72 
        self.cost_Engine_100Km = cost_Engine * (100 / total_milage)
        self.steps += 1

        if self.steps >= int(car_spd_one.shape[1])-1:
            done = True
            # info = {}
            info = {
                'Eng_spd': self.Eng_spd,
                'Eng_trq': self.Eng_trq,
                'Eng_pwr_opt': self.Eng_pwr_opt,
                'SOC': self.SOC,
                'Total_cost':self.total_cost / 720, 
                'cost_Engine_100Km': self.cost_Engine_100Km

            }
            #print(self.cost_Engine_100Km)
            return next_state, reward, done, info

        else:
            done = False
            info = {
                'Eng_spd': self.Eng_spd,
                'Eng_trq': self.Eng_trq,
                'Eng_pwr_opt': self.Eng_pwr_opt,
                'SOC': self.SOC,
                'Total_cost':self.total_cost / 720, 
                'cost_Engine_100Km': self.cost_Engine_100Km
            }
            return next_state, reward, done, info

    def render(self):
        pass
    


### 离散控制版本
class PriusEnv_():
    # metadata = {'render.modes': ['human']}

    def __init__(self):

        # super(PriusEnv, self).__init__()
        self.dt = 1
        self.SOC = 0.6
        self.steps = 0
        # self.action_space = spaces.Box(low= 0, high=1.0, shape=(1, ), dtype=np.float32)
        self.action_space = spaces.Discrete(14) 
        self.observation_space = spaces.Box(low=np.array([0,0,-3]),
                                            high=np.array([1,40,3]),
                                            dtype=np.float32)
        self.viewer = None

    def reset(self):

        SOC_origin = self.SOC
        self.ep_reward = 0
        self.steps = 0
        self.total_cost = 0 

        self.done = False
        
        ## 多工况训练
        # path = "E:/Document/Experients/HEV_EMS/training_data"
        path_list = os.listdir(path)
        # random_data = np.random.randint(0,len(path_list))
        # self.base_data = path_list[random_data]
        # print(self.base_data)

        ## 单工况训练
        self.base_data = path_list[0]   # driving cycles 0
        #print(self.base_data)

        self.state = np.array([SOC_origin, 0, 0])
        return self.state


    def step(self, action):

        # path = "E:/Document/Experients/HEV_EMS/training_data"
        
        data = scio.loadmat(path + '/' + self.base_data)
        car_spd_one = data['speed_vector']            
        SOC_origin = 0.6
        total_milage = np.sum(car_spd_one) / 1000 
        a = 0            
        
        if action == 0:
            a += 0
        if action == 1:
            a += (1 / 56)
        if action == 2:
            a += (-1 / 56) 
        if action == 3:
            a += (2 / 56)
        if action == 4:
            a += (-2 / 56)
        if action == 5:
            a += (4 / 56)
        if action == 6:
            a += (-4 / 56)
        if action == 7:
            a += (6 / 56)
        if action == 8:
            a += (-6 / 56)
        if action == 9:
            a += (8 / 56)
        if action == 10:
            a += (-8 / 56)
        if action == 11:
            a += (10 / 56)
        if action == 12:
            a += (-10 / 56)
        if action == 13:
            a = 0

        Eng_pwr_opt = action * 56000
 
        self.velocity = car_spd_one[:, self.steps+1]                     

        self.acceleration = car_spd_one[:, self.steps+1] - car_spd_one[:,self.steps]  
        
        soc  = self.SOC
        v = self.velocity
        a =  self.acceleration
       
        ###  模型数据
        out, cost, I = Prius_model().run(v, a, Eng_pwr_opt, soc)
        self.P_req = float(out['P_req'])
        self.Eng_spd = float(out['Eng_spd'])
        self.Eng_trq= float(out['Eng_trq']) 
        self.Eng_pwr = float(out['Eng_pwr'])
        self.Eng_pwr_opt = float(out['Eng_pwr_opt'])
        self.Mot_spd = float(out['Mot_spd'])
        self.Mot_trq = float(out['Mot_trq'])       
        self.Mot_pwr= float(out['Mot_pwr'])  
        self.Gen_spd = float(out['Gen_spd'])
        self.Gen_trq = float(out['Gen_trq'])       
        self.Gen_pwr = float(out['Gen_pwr'])
        self.Batt_pwr = float(out['Batt_pwr'])   

        self.Mot_eta = float(out['Mot_eta'])
        self.Gen_eta = float(out['Gen_eta'])
        self.T_list = float(out['T']) 
        
        
        self.SOC = float(out['SOC'])     
        next_state = np.hstack([self.SOC, self.velocity, self.acceleration])
        cost = float(cost)
        reward = - cost            
        if self.SOC < 0.5 or self.SOC > 0.85:
            reward = - ((300 * ((0.5 - self.SOC) ** 2)) + cost)
    
    
        self.ep_reward = cost + self.ep_reward
        self.total_cost = self.total_cost + cost

        cost_Engine =  (self.ep_reward / 0.72 / 1000) + (self.SOC < SOC_origin) * (SOC_origin - self.SOC) * (201.6 * 6.5) * 3600 /(42600000) / 0.72 
        self.cost_Engine_100Km = cost_Engine * (100 / total_milage)
        self.steps += 1

        if self.steps >= int(car_spd_one.shape[1])-1:
            done = True
            # info = {}
            info = {
                'Eng_spd': self.Eng_spd,
                'Eng_trq': self.Eng_trq,
                'Eng_pwr_opt': self.Eng_pwr_opt,
                'SOC': self.SOC,
                'Total_cost':self.total_cost / 720,
                'cost_Engine_100Km': self.cost_Engine_100Km

            }
            #print(self.cost_Engine_100Km)
            return next_state, reward, done, info

        else:
            done = False
            info = {
                'Eng_spd': self.Eng_spd,
                'Eng_trq': self.Eng_trq,
                'Eng_pwr_opt': self.Eng_pwr_opt,
                'SOC': self.SOC,
                'Total_cost':self.total_cost / 720,
                'cost_Engine_100Km': self.cost_Engine_100Km
            }
            return next_state, reward, done, info

    def render(self):
        pass

if __name__ == "__main__":
          
      env = PriusEnv()
      observation = env.reset()
      print(observation)
      # 循环执行10个动作
      for i in range(10):
          # 随机选择一个动作
          action = np.random.rand()
          print(action)
          # 执行动作，获取下一个观察，奖励，结束标志和信息
          observation, reward, done, info = env.step(action)
          print(observation, reward, done, info)
          # 如果结束标志为真，则退出循环
          if done == True:
              break

