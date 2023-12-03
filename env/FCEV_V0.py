import numpy as np
import scipy.io as scio
import gym
from gym import spaces
import os
import sys
sys.path.append('G:/experiment/HEV_open/HEV_EMS/env/units')
from FCEV_model_base import FCEV_model
path = "G:/experiment/HEV_open/HEV_EMS/training_data"
class FCEVEnv():
    # metadata = {'render.modes': ['human']}

    def __init__(self):

        # super(PriusEnv, self).__init__()
        self.dt = 1
        self.SOC = 0.6
        self.steps = 0
        self.action_space = spaces.Box(low= 0, high=1.0, shape=(1, ), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=np.array([0,0,-3]),
                                            high=np.array([1,40,3]),
                                            dtype=np.float32)
        self.viewer = None

    def reset(self):

        SOC_origin = 0.6

        self.steps = 0

        self.done = False

        path_list = os.listdir(path)
        random_data = np.random.randint(0,len(path_list))
        self.base_data = path_list[random_data]
        print(self.base_data)

        self.state = np.array([SOC_origin, 0, 0])
        return self.state


    def step(self, action):

        data = scio.loadmat(path + '/' + self.base_data)
        car_spd_one = data['speed_vector']            

        action = np.clip(action, 0, 1)

        FC_pwr_opt = action
 
        self.velocity = car_spd_one[:, self.steps+1]                     

        self.acceleration = car_spd_one[:, self.steps+1] - car_spd_one[:,self.steps]  
        
        soc  = self.SOC
        v = self.velocity
        a =  self.acceleration

        out, cost, I = FCEV_model().run(v, a, FC_pwr_opt, soc)
        
        self.SOC = float(out['SOC'])     
        next_state = np.hstack([self.SOC, self.velocity, self.acceleration])
        cost = float(cost)
        reward = - cost            
        if self.SOC < 0.5 or self.SOC > 0.85:
            reward = - ((350 * ((0.3 - self.SOC) ** 2)) + cost)
    

        self.steps += 1

        if self.steps >= int(car_spd_one.shape[1])-1:
            done = True
            info = {}
            return next_state, reward, done, info

        else:
            done = False
            info = {}
            return next_state, reward, done, info

    def render(self):
        pass
    

    

if __name__ == "__main__":
          
      env = FCEVEnv()
      observation = env.reset()
      print(observation)
      # 循环执行10个动作
      for i in range(100):
          # 随机选择一个动作
          action = np.random.rand()
          print(action)
          # 执行动作，获取下一个观察，奖励，结束标志和信息
          observation, reward, done, info = env.step(action)
          print(observation, reward, done, info)
          # 如果结束标志为真，则退出循环
          if done == True:
              break

