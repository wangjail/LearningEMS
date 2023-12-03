# -*- coding: utf-8 -*-
"""
the Model of Prius
"""

import numpy as np
from scipy.interpolate import interp1d, interp2d
import math
import scipy.io as scio


class FCEV_model():
    def __init__(self):
        # paramsmeters of car
        self.Wheel_R = 0.282
        self.mass = 1380
        self.C_roll  = 0.009
        self.density_air = 1.2
        self.area_frontal = 2.0
        self.G = 9.81
        self.C_d = 0.335
        self.Pgs_K = 6.67

        # the factor of F_roll
        self.T_factor = 1

        # Fuel cell system
        self.fc_pwr_map = np.array([0, 2, 5, 7.5, 10, 20, 30, 40, 50]) * 1000
        fc_eff_map = np.array([10, 33, 49.2, 53.3, 55.9, 59.6, 59.1, 56.2, 50.8]) / 100
        fc_fuel_lhv = 120.0 * 1000  # (J/g), lower heating value of the fuel
        self.fc_fuel_map = self.fc_pwr_map * (1 / fc_eff_map) / fc_fuel_lhv  # fuel consumption map (g/s)
        self.FC_fuel_func = interp1d(self.fc_pwr_map,self.fc_fuel_map)
        Fc_pwr_min = 0 # 燃料电池最小功率
        Fc_pwr_max = 50 * 1000 #燃料电池最大功率


        # Motor
        mc_map_spd = np.arange(0, 11000, 1000) * (2 * np.pi) / 60  # motor speed list (rad/s)   0~1000rpm
        mc_map_trq = np.arange(-200, 220, 20) * 4.448 / 3.281  # motor torque list (Nm)        -200~200Nm
        # motor efficiency map
        data_path1 = 'G:/experiment/HEV_open/HEV_EMS/env/units/MotofFCEV_eta_quarter.mat'
        data1 = scio.loadmat(data_path1)
        Mot_eta_map = data1['mc_eff_map']
        self.Mot_eta_map_func = interp2d(mc_map_trq, mc_map_spd, Mot_eta_map)

        #  motor maximum torque
        Mot_trq_max_list = np.array([200,200,200,134.700000000000,110.900000000000,93.2000000000000,80,52.9000000000000,48.1000000000000,39.8000000000000,38])
        # motor minimum torque
        Mot_trq_min_list = - Mot_trq_max_list
        self.Mot_trq_min_func = interp1d(mc_map_spd, Mot_trq_min_list, kind = 'linear', fill_value = 'extrapolate')
        self.Mot_trq_max_func = interp1d(mc_map_spd, Mot_trq_max_list, kind = 'linear', fill_value = 'extrapolate')

        # Battery
        # published capacity of one battery cell
        Batt_Q_cell = 6.5     
        # coulombs, battery package capacity
        self.Batt_Q = Batt_Q_cell * 3600     
        # resistance and OCV list
        Batt_rint_dis_list = [0.7,0.619244814,0.443380117,0.396994948,0.370210379,0.359869599,0.364414573,0.357095093,0.363394618,0.386654377,0.4] # ohm
        Batt_rint_chg_list = [0.7,0.623009741,0.477267027,0.404193372,0.37640518,0.391748667,0.365290105,0.375071555,0.382795632,0.371566564,0.36] # ohm
        Batt_vol_list  = [202,209.3825073,213.471405,216.2673035,218.9015961,220.4855042,221.616806,222.360199,224.2510986,227.8065948,237.293396] # V
        # resistance and OCV
        SOC_list = np.arange(0, 1.01, 0.1) 
        self.Batt_vol_func = interp1d(SOC_list, Batt_vol_list, kind = 'linear', fill_value = 'extrapolate')
        self.Batt_rint_dis_list_func = interp1d(SOC_list, Batt_rint_dis_list, kind = 'linear', fill_value = 'extrapolate')
        self.Batt_rint_chg_list_func = interp1d(SOC_list, Batt_rint_chg_list, kind = 'linear', fill_value = 'extrapolate')  

        #Battery current limitations
        self.Batt_I_max_dis = 196
        self.Batt_I_max_chg = 120
        
        
    def run(self, car_spd, car_a, FC_pwr_opt, SOC):

        # Wheel speed (rad/s)
        Wheel_spd = car_spd / self.Wheel_R  ## 车轮转速
        # Wheel torque (Nm) 地面作用于驱动轮的切向反作用力
        F_roll = self.mass * self.G * self.C_roll * (self.T_factor if car_spd > 0 else 0)  ## 滚动阻力
        ## 如果 car_spd>0， 输出self.T_factor， 否则输出0
        F_drag = 0.5 * self.density_air * self.area_frontal * self.C_d *(car_spd ** 2)  ##空气阻力
        F_a = self.mass * car_a  ##加速阻力

        T = self.Wheel_R * (F_a + F_roll + F_drag )  ##需求转矩
        P_req = T * Wheel_spd  ##需求功率


        # FC system
        if FC_pwr_opt > 50000:
            FC_pwr_opt = 50000
        
        if (FC_pwr_opt < 500) or (T < 0):
            FC_pwr_opt = 0

        FC_fuel_mdot = self.FC_fuel_func(FC_pwr_opt)

        # maximum engine torque boundary (Nm)
        Fc_pwr_max = 50 * 1000  # 燃料电池最大功率


        inf_FC = (FC_pwr_opt > Fc_pwr_max)

        
        # motor rotating speed and torque
        Mot_spd = self.Pgs_K * Wheel_spd
        Mot_trq = T / self.Pgs_K
        Mot_trq = (Mot_trq < 0) * (Mot_trq < self.Mot_trq_min_func(Mot_spd)) * self.Mot_trq_min_func(Mot_spd) +\
                  (Mot_trq < 0) * (Mot_trq > self.Mot_trq_min_func(Mot_spd)) * Mot_trq +\
                  (Mot_trq >= 0) * (Mot_trq > self.Mot_trq_max_func(Mot_spd)) * self.Mot_trq_max_func(Mot_spd) +\
                  (Mot_trq >= 0) * (Mot_trq < self.Mot_trq_max_func(Mot_spd)) * Mot_trq 
        
        Mot_trq = np.array(Mot_trq).flatten()    ## 一维数组
        Mot_eta = (Mot_spd == 0) + (Mot_spd != 0) * self.Mot_eta_map_func(Mot_trq, Mot_spd * np.ones(1)) #need to edit        
        inf_mot = (np.isnan(Mot_eta)) + (Mot_trq < 0) * (Mot_trq < self.Mot_trq_min_func(Mot_spd)) + (Mot_trq >= 0) * (Mot_trq > self.Mot_trq_max_func(Mot_spd))
        Mot_eta[np.isnan(Mot_eta)] = 1        
        # Calculate electric power consumption
        Mot_pwr = (Mot_trq * Mot_spd <= 0) * Mot_spd * Mot_trq * Mot_eta + (Mot_trq * Mot_spd > 0) * Mot_spd * Mot_trq / Mot_eta
        

        Batt_vol = self.Batt_vol_func(SOC)
        Batt_pwr = Mot_pwr
        Batt_rint = (Batt_pwr > 0) * self.Batt_rint_dis_list_func(SOC) + (Batt_pwr <= 0) * self.Batt_rint_chg_list_func(SOC)
        #columbic efficiency (0.9 when charging)
        Batt_eta = (Batt_pwr > 0) + (Batt_pwr <= 0) * 0.9
        Batt_I_max = (Batt_pwr > 0) * self.Batt_I_max_dis + (Batt_pwr <= 0) * self.Batt_I_max_chg
        
        # the limitation of Batt_pwr
        inf_batt_one = (Batt_vol ** 2 < 4 * Batt_rint * Batt_pwr)    
        if Batt_vol ** 2 < 4 * Batt_rint * Batt_pwr:
    #        Eng_pwr = Eng_pwr + Batt_pwr - Batt_vol ** 2 / (4 * Batt_rint)    
    #        Eng_trq = Eng_pwr / Eng_spd       
            Batt_pwr = Mot_pwr - Batt_vol ** 2 / (4 * Batt_rint)              # 放电功率过大以及充电功率过大？
            Batt_I = Batt_eta * Batt_vol / (2 * Batt_rint)
    #        print('battery power is out of bound')
        else:          
            Batt_I = Batt_eta * (Batt_vol - np.sqrt(Batt_vol ** 2 - 4 * Batt_rint * Batt_pwr)) / 0.8
               
        inf_batt = inf_batt_one + (np.abs(Batt_I) > Batt_I_max)
           
        # New battery state of charge
        SOC_new = - Batt_I / self.Batt_Q + SOC   
        # Set new state of charge to real values
        SOC_new = (np.conjugate(SOC_new) + SOC_new) / 2
        
        if SOC_new > 1:
            SOC_new = 1.0
        if SOC_new < 1:
            SOC_new = 0
        
        P_out = FC_pwr_opt + Batt_pwr
        # Cost
        I = (inf_batt + inf_FC + inf_mot != 0)
        # Calculate cost matrix (fuel mass flow)
        cost = FC_fuel_mdot
        
        out = {}
        out['P_req'] = P_req
        out['P_out'] = P_out
        out['FC_pwr'] = FC_pwr_opt
        out['FC_pwr_opt'] = FC_pwr_opt
        out['Mot_spd'] = Mot_spd
        out['Mot_trq'] = Mot_trq
        out['Mot_pwr'] = Mot_pwr
        out['SOC'] = SOC_new     
        out['Batt_vol'] = Batt_vol       
        out['Batt_pwr'] = Batt_pwr
        out['inf_batt'] = inf_batt
        out['inf_batt_one'] = inf_batt_one
        out['T'] = T
        out['Mot_eta'] = Mot_eta
        out['FC_fuel_mdot'] = FC_fuel_mdot
        
        return  out, cost, I

#Prius = Prius_model()
#out, cost, I = Prius.run(20, 1, 30000, 0.8)
