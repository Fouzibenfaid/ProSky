import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import time
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
from IPython.display import clear_output
import os
import pickle

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

torch.manual_seed(0)
np.random.seed(0)

class DQN(nn.Module):
    def __init__(self, NUMBER_OF_ARGUMENTS_PER_STATE):
        super().__init__(),

        self.fc1 = nn.Linear(in_features=NUMBER_OF_ARGUMENTS_PER_STATE, out_features=128) 
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=32)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        q = self.out(t)
        return q


Experience = namedtuple(
            'Experience',
            ('state', 'action', 'next_state', 'reward')
                        )

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class EpsilonGreedyStrategy():

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
                            math.exp(-1. * current_step / self.decay)

class Agent():
    def __init__(self, strategy, num_actions, device):

        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device) # explore    
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device) # exploit

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
    @staticmethod        
    def get_next(target_net, next_states):                 
        return target_net(next_states).max(dim=1)[0].detach()

def plot(values,r1,r2,r3,r4,r5,r6,h1,h2,h3,h4,a1,a2,a3,a4,SUM1,SUM2,SUM3,SUM4,Fairness,H,AVG2, Fairness2, moving_avg_period):
    
    moving_avg_rewards = get_moving_average(moving_avg_period, values)

    Fairness = [element * 100 for element in Fairness]
    moving_avg_fairness = get_moving_average(moving_avg_period, Fairness)
    Fairness2 = [element * 100 for element in Fairness2]
    moving_avg_fairness2 = get_moving_average(moving_avg_period, Fairness2)

    moving_avg_h1 = get_moving_average(moving_avg_period, h1)
    moving_avg_h2 = get_moving_average(moving_avg_period, h2)

    moving_avg_h3 = get_moving_average(moving_avg_period, h3)
    moving_avg_h4 = get_moving_average(moving_avg_period, h4)

    moving_avg_a1 = get_moving_average(moving_avg_period, a1)
    moving_avg_a2 = get_moving_average(moving_avg_period, a2)
    moving_avg_a1 = [element * 100 for element in moving_avg_a1]
    moving_avg_a2 = [element * 100 for element in moving_avg_a2]

    moving_avg_a3 = get_moving_average(moving_avg_period, a3)
    moving_avg_a4 = get_moving_average(moving_avg_period, a4)
    moving_avg_a3 = [element * 100 for element in moving_avg_a3]
    moving_avg_a4 = [element * 100 for element in moving_avg_a4]

    
    moving_avg_SUM1 = get_moving_average(moving_avg_period, SUM1)
    moving_avg_SUM2 = get_moving_average(moving_avg_period, SUM2)
    moving_avg_SUM1 = [element * 2000 for element in moving_avg_SUM1]
    moving_avg_SUM2 = [element * 2000 for element in moving_avg_SUM2]

    moving_avg_SUM3 = get_moving_average(moving_avg_period, SUM3)
    moving_avg_SUM4 = get_moving_average(moving_avg_period, SUM4)
    moving_avg_SUM3 = [element * 2000 for element in moving_avg_SUM3]
    moving_avg_SUM4 = [element * 2000 for element in moving_avg_SUM4]

    SUM = np.add(moving_avg_SUM1,moving_avg_SUM2)
    SUM = np.add(SUM,moving_avg_SUM3)
    SUM = np.add(SUM,moving_avg_SUM4)
    avg2 = get_moving_average(moving_avg_period, AVG2)
    avg2 = [element * 2000 for element in avg2]
    
  
    moving_avg_Height = get_moving_average(moving_avg_period, H)
    

    with open(f"DRL-r={0},different-channel-conditions-SUM-RATE.pickle", "wb") as f:
                   pickle.dump(SUM, f)
    with open(f"DRL-r={0},different-channel-conditions-Height.pickle", "wb") as f:
                   pickle.dump(moving_avg_Height, f)

    print("\nr = ", 0)
    print(moving_avg_period, "Episode moving avg:", moving_avg_rewards[-1], "Sum Rate Moving Average:",round(SUM[-1],2)/1000, "Gbps")
    print("SE1 = ", round(moving_avg_SUM1[-1]/2000, 2) , "SE2 = ", round(moving_avg_SUM2[-1]/2000, 2), "SE3 = ",round(moving_avg_SUM3[-1]/2000, 2), "SE4 = ", round(moving_avg_SUM4[-1]/2000, 2), "\n")

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()
    
def mmLineOfSight_Check(D,H):
    C = 9.6117 # Urban LOS probability parameter 
    Y = 0.1581 # Urban LOS probability parameter
    RAND = random.uniform(0,1)
    teta = math.asin(H/D) * 180/math.pi
    p1 = 1 / ( 1 + (C * math.exp( -Y * (teta - C ) ) ) )
    p2 = 1 - p1
    if p1 >= p2:
        if RAND >= p2:
            L = 1
        else:
            L = 2
    else:
        if RAND >= p1:
            L = 2
        else:
            L = 1
    return L
    
def Average(lst): 
    return sum(lst) / len(lst) 

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)

class Blob():
    def __init__(self, size, USER1=False, USER2=False, USER3=False, USER4=False):
        self.size = size
        if USER1:
            self.x = 35
            self.y = 54
        elif USER2:
            self.x = 94
            self.y = 1
        elif USER3:
            self.x = 29
            self.y = 45
        elif USER4:
            self.x = 1
            self.y = 97
        else:
            self.x = 50
            self.y = 50

    def __str__(self):
        return f"Blob({self.x}, {self.y})"

    def __sub__(self, other):
        return [(self.x-other.x)/10, (self.y-other.y)/10]

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        

        if choice == 0:
            self.move(x=1, y=1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H += 1

        elif choice == 1:
            self.move(x=-1, y=-1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H += 1

        elif choice == 2:
            self.move(x=-1, y=1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H += 1

        elif choice == 3:
            self.move(x=1, y=-1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H += 1
            
        elif choice == 4:
            self.move(x=1, y=1)
            self.a1 += 0.01
            self.a3 -=0.01
            self.H += 1

        elif choice == 5:
            self.move(x=-1, y=-1)
            self.a1 += 0.01
            self.a3 -= 0.01
            self.H += 1

        elif choice == 6:
            self.move(x=-1, y=1)
            self.a1 += 0.01
            self.a3 -= 0.01
            self.H += 1

        elif choice == 7:
            self.move(x=1, y=-1)
            self.a1 += 0.01
            self.a3 -= 0.01
            self.H += 1
            
        elif choice == 8:
            self.move(x=1, y=1)
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H += 1
            
        elif choice == 9:
            self.move(x=-1, y=-1)
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H += 1

        elif choice == 10:
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H += 1

        elif choice == 11:
            self.move(x=1, y=-1)
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H += 1
            
        elif choice == 12:
            self.move(x=1, y=1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H += 1

        elif choice == 13:
            self.move(x=-1, y=-1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H += 1

        elif choice == 14:
            self.move(x=-1, y=1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H += 1

        elif choice == 15:
            self.move(x=1, y=-1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H += 1
            
        if choice == 16:
            self.move(x=1, y=1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H -= 1

        elif choice == 17:
            self.move(x=-1, y=-1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H -= 1

        elif choice == 18:
            self.move(x=-1, y=1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H -= 1

        elif choice == 19:
            self.move(x=1, y=-1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H -= 1
            
        elif choice == 20:
            self.move(x=1, y=1)
            self.a1 += 0.01
            self.a3 -=0.01
            self.H -= 1

        elif choice == 21:
            self.move(x=-1, y=-1)
            self.a1 += 0.01
            self.a3 -= 0.01
            self.H -= 1

        elif choice == 22:
            self.move(x=-1, y=1)
            self.a1 += 0.01
            self.a3 -= 0.01
            self.H -= 1

        elif choice == 23:
            self.move(x=1, y=-1)
            self.a1 += 0.01
            self.a3 -= 0.01
            self.H -= 1
            
        elif choice == 24:
            self.move(x=1, y=1)
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H -= 1
            
        elif choice == 25:
            self.move(x=-1, y=-1)
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H -= 1

        elif choice == 26:
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H -= 1

        elif choice == 27:
            self.move(x=1, y=-1)
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H -= 1
            
        elif choice == 28:
            self.move(x=1, y=1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H -= 1

        elif choice == 29:
            self.move(x=-1, y=-1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H -= 1

        elif choice == 30:
            self.move(x=-1, y=1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H -= 1

        elif choice == 31:
            self.move(x=1, y=-1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H -= 1
            
        if self.a1 > 1:
            self.a1 = 1
        elif self.a1 < 0:
            self.a1 = 0
        if self.a3 > 1:
            self.a3 = 1
        elif self.a3 < 0:
            self.a3 = 0
        if self.H <= 10:
            self.H =10
        

    def move(self, x=False, y=False):

        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1

class BlobEnv():
    SIZE = 100
    MOVE_PENALTY = 1
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    UAV_N = 1  # UAV key in dict
    USER_N = 2  # USER key in dict
    UAV2_N = 4  # UAV2 key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255),
         4: (175, 0, 255)}

    def reset(self):
        P = 0.1 # Transmitted power 20dbm (i.e. .1w)
        N_uav = 8
        N_ue = 8
        G = N_uav * N_ue
        P *= G
        W = 2e9 # Bandwidth 2GHz
        fc = 28e9 # Carrier frequency = 28GHz
        NF = 10**(5/10) # 5dB Noise Figure 
        TN = 10**(-114/10) # -84dBm Thermal Noise
        N = NF * TN
        C_LOS = 10**(-6.4)
        a_LOS = 2
        C_NLOS = 10**(-7.2) 
        a_NLOS = 2.92

        self.UAV = Blob(self.SIZE)
        self.UAV2 = Blob(self.SIZE)
        self.h1 = []
        self.h2 = []
        self.h3 = []
        self.h4 = []
        self.a1 = []
        self.a2 = []
        self.a3 = []
        self.a4 = []
        self.SUM1 = []
        self.SUM2 = []
        self.SUM3 = []
        self.SUM4 = []
        self.Fairness = []
        self.Hl = []
        self.NLOS = []
        self.NOMA = []
        self.reward1 = []
        self.reward2 = []
        self.reward3 = []
        self.reward4 = []
        self.reward5 = []
        self.reward6 = []
        
        self.UAV.a1 = 0.5
        self.UAV.a2 = 0.5
        self.UAV.a3 = 0.5
        self.UAV.a4 = 0.5
        self.UAV.H = 50
        H2 = 50
        
        self.USER1 = Blob(self.SIZE, True, False, False, False)
        self.USER2 = Blob(self.SIZE, False, True, False, False)
        self.USER3 = Blob(self.SIZE, False, False, True, False)
        self.USER4 = Blob(self.SIZE, False, False, False, True)

        #self.UAV2.x = int((self.USER1.x +self.USER2.x + self.USER3.x + self.USER4.x )/4)
        #self.UAV2.y = int((self.USER1.y +self.USER2.y + self.USER3.y + self.USER4.y )/4)
        self.UAV2.x = 50
        self.UAV2.y = 50

        ob1 = self.UAV-self.USER1
        ob2 = self.UAV-self.USER2
        ob3 = self.UAV-self.USER3
        ob4 = self.UAV-self.USER4
        
        D1 =  np.sum(np.sqrt([(10*ob1[0])**2, (10*ob1[1])**2]))
        D2 = np.sum(np.sqrt([(10*ob2[0])**2, (10*ob2[1])**2]))
        D3 = np.sum(np.sqrt([(10*ob3[0])**2, (10*ob3[1])**2]))
        D4 = np.sum(np.sqrt([(10*ob4[0])**2, (10*ob4[1])**2]))
                  
        H = self.UAV.H
        Dt1 = np.sum(np.sqrt([ (10*ob1[0])**2, (10*ob1[1])**2, H**2  ]))
        Dt2 = np.sum(np.sqrt([ (10*ob2[0])**2, (10*ob2[1])**2, H**2  ]))
        Dt3 = np.sum(np.sqrt([ (10*ob3[0])**2, (10*ob3[1])**2, H**2  ]))
        Dt4 = np.sum(np.sqrt([ (10*ob4[0])**2, (10*ob4[1])**2, H**2  ]))
        
        self.L1 = mmLineOfSight_Check(Dt1,H)
        self.L2 = mmLineOfSight_Check(Dt2,H)
        self.L3 = mmLineOfSight_Check(Dt3,H)
        self.L4 = mmLineOfSight_Check(Dt4,H)
        
        if self.L1 == 1:
            h1 = C_LOS * Dt1**(-a_LOS)
        else:
            h1 = C_NLOS * Dt1**(-a_NLOS)

        if self.L2 == 1:
            h2 = C_LOS * Dt2**(-a_LOS)
        else:
            h2 = C_NLOS * Dt2**(-a_NLOS)
        if self.L3 == 1:
            h3 = C_LOS * Dt3**(-a_LOS)
        else:
            h3 = C_NLOS * Dt3**(-a_NLOS)
        if self.L4 == 1:
            h4 = C_LOS * Dt4**(-a_LOS)
        else:
            h4 = C_NLOS * Dt4**(-a_NLOS)
        
        a1 =  self.UAV.a1
        a2 =  1 - a1
        a3 =  self.UAV.a3
        a4 =  1 - a3
        observation = [ob1[0]] + [ob1[1]] + [ob2[0]] + [ob2[1]]+ [ob3[0]] + [ob3[1]]+ [ob4[0]] + [ob4[1]] + [a1] + [a2] + [a3] + [a4] + [h1] + [h2] + [h3] + [h4] + [H]
        
        self.episode_step = 0

        return observation

    def step(self, action):
        
        done= False
        
        P = 0.1 # Transmitted power 20dbm (i.e. .1w)
        N_uav = 8
        N_ue = 8
        G = N_uav * N_ue
        P *= G
        W = 2e9 # Bandwidth 2GHz
        fc = 28e9 # Carrier frequency = 28GHz
        NF = 10**(5/10) # 5dB Noise Figure 
        TN = 10**(-114/10) # -84dBm Thermal Noise
        N = NF * TN
        C_LOS = 10**(-6.4)
        a_LOS = 2
        C_NLOS = 10**(-7.2) 
        a_NLOS = 2.92        
        H = self.UAV.H # antenna Height
        
        self.episode_step += 1
        
        ob1 = self.UAV-self.USER1
        ob2 = self.UAV-self.USER2
        ob3 = self.UAV-self.USER3
        ob4 = self.UAV-self.USER4
        
        D1 =  np.sum(np.sqrt([(10*ob1[0])**2, (10*ob1[1])**2]))
        D2 = np.sum(np.sqrt([(10*ob2[0])**2, (10*ob2[1])**2]))
        D3 = np.sum(np.sqrt([(10*ob3[0])**2, (10*ob3[1])**2]))
        D4 = np.sum(np.sqrt([(10*ob4[0])**2, (10*ob4[1])**2]))
                  
        H = self.UAV.H
        Dt1 = np.sum(np.sqrt([ (10*ob1[0])**2, (10*ob1[1])**2, H**2  ]))
        Dt2 = np.sum(np.sqrt([ (10*ob2[0])**2, (10*ob2[1])**2, H**2  ]))
        Dt3 = np.sum(np.sqrt([ (10*ob3[0])**2, (10*ob3[1])**2, H**2  ]))
        Dt4 = np.sum(np.sqrt([ (10*ob4[0])**2, (10*ob4[1])**2, H**2  ]))
        
        self.L1 = mmLineOfSight_Check(Dt1,H)
        self.L2 = mmLineOfSight_Check(Dt2,H)
        self.L3 = mmLineOfSight_Check(Dt3,H)
        self.L4 = mmLineOfSight_Check(Dt4,H)
        
        if self.L1 == 1:
            h1 = C_LOS * Dt1**(-a_LOS)
            self.NLOS.append(0)
        else:
            h1 = C_NLOS * Dt1**(-a_NLOS)
            self.NLOS.append(1)
        if self.L2 == 1:
            h2 = C_LOS * Dt2**(-a_LOS)
            self.NLOS.append(0)
        else:
            h2 = C_NLOS * Dt2**(-a_NLOS)
            self.NLOS.append(1)
        if self.L3 == 1:
            h3 = C_LOS * Dt3**(-a_LOS)
            self.NLOS.append(0)
        else:
            h3 = C_NLOS * Dt3**(-a_NLOS)
            self.NLOS.append(1)
        if self.L4 == 1:
            h4 = C_LOS * Dt4**(-a_LOS)
            self.NLOS.append(0)
        else:
            h4 = C_NLOS * Dt4**(-a_NLOS)
            self.NLOS.append(1)
        
        self.UAV.action(action)
        
        a1 =  self.UAV.a1
        a2 =  1 - a1
        a3 =  self.UAV.a3
        a4 =  1 - a3
        
        self.h1.append(h1)
        self.h2.append(h2)
        self.h3.append(h3)
        self.h4.append(h4)
        self.a1.append(a1)
        self.a2.append(a2)
        self.a3.append(a3)
        self.a4.append(a4)
        self.Hl.append(H)

        reward = 0
        reward_6 = 0
        
        if h1 >= h2:
            
            SUM1 = math.log2(1 + h1 * a1 * P/N)
            SUM2 = math.log2(1 + a2 * h2 * P / (a1 * h2 * P + N) )

        else: 
        
            SUM1 = math.log2(1 + a1 * h1 * P / (a2 * h1 * P + N) )
            SUM2 =  math.log2(1 + h2 * a2 * P/N)
                
        if h3 >= h4:
            SUM3 = math.log2(1 + h3 * a3 * P/N)
            SUM4 = math.log2(1 + a4 * h4 * P / (a3 * h4 * P + N) ) 
            
        else: 
            
            SUM3 = math.log2(1 + a3 * h3 * P / (a4 * h3 * P + N) )
            SUM4 = math.log2(1 + h4 * a4 * P/N) 
        
        reward_3 = (SUM1 + SUM2 + SUM3 + SUM4)**2 / (4 * (SUM1**2 + SUM2**2 + SUM3**2 + SUM4**2))
        self.Fairness.append(reward_3)

        r = 0.5

        if SUM1 >= r:
            reward += 100
        if SUM2 >= r:
            reward += 100
        if SUM3 >= r:
            reward += 100
        if SUM4 >= r:
            reward += 100

        reward_3 *= 0
        reward_6 += 2e10 * (h1+h2+h3+h4) * 0
        reward +=  (SUM1 + SUM2 + SUM3 + SUM4)  + reward_3  + reward_6
        
        self.SUM1.append(SUM1)
        self.SUM2.append(SUM2)
        self.SUM3.append(SUM3)
        self.SUM4.append(SUM4)

        new_observation_m =  ([ob1[0]] + [ob1[1]] + [ob2[0]] + [ob2[1]]+ [ob3[0]] + [ob3[1]] + [ob4[0]] + [ob4[1]] + [a1] + [a2] + [a3] + [a4] + [h1] + [h2] + [h3] + [h4] + [H] )
        new_observation =  new_observation_m  
        if self.episode_step >= 300:


            ob21 = self.UAV2-self.USER1
            ob22 = self.UAV2-self.USER2
            ob23 = self.UAV2-self.USER3
            ob24 = self.UAV2-self.USER4
            H2 = 50
            
            D21 =  np.sum(np.sqrt([(10*ob21[0])**2, (10*ob21[1])**2]))
            D22 = np.sum(np.sqrt([(10*ob22[0])**2, (10*ob22[1])**2]))
            D23 = np.sum(np.sqrt([(10*ob23[0])**2, (10*ob23[1])**2]))
            D24 = np.sum(np.sqrt([(10*ob24[0])**2, (10*ob24[1])**2]))

            Dt21 = np.sum(np.sqrt([ (10*ob21[0])**2, (10*ob21[1])**2, H2**2  ]))
            Dt22 = np.sum(np.sqrt([ (10*ob22[0])**2, (10*ob22[1])**2, H2**2  ]))
            Dt23 = np.sum(np.sqrt([ (10*ob23[0])**2, (10*ob23[1])**2, H2**2  ]))
            Dt24 = np.sum(np.sqrt([ (10*ob24[0])**2, (10*ob24[1])**2, H2**2  ]))
        
            h221 = C_LOS * Dt21**(-a_LOS)
            h222 = C_LOS * Dt22**(-a_LOS)
            h223 = C_LOS * Dt23**(-a_LOS)
            h224 = C_LOS * Dt24**(-a_LOS)

            if h221 >= h222:
                a222 = ((2**r - 1)/2**r) * (1 + N/(P*h222))
                a221 = 1 - a222
                SUM221 = math.log2(1 + h221 * a221 * P/N)
                SUM222 = math.log2(1 + a222 * h222 * P / (a221 * h222 * P + N) )
            else: 
                a221 = ((2**r - 1)/2**r) * (1 + N/(P*h221))
                a222 = 1-a221
                SUM221 = math.log2(1 + a221 * h221 * P / (a222 * h221 * P + N) )
                SUM222 =  math.log2(1 + h222 * a222 * P/N)
            if h223 >= h224:
                a224 = ((2**r - 1)/2**r) * (1 + N/(P*h224))
                a223 = 1 - a224
                SUM223 = math.log2(1 + h223 * a223 * P/N)
                SUM224 = math.log2(1 + a224 * h224 * P / (a223 * h224 * P + N) ) 
            else: 
                a223 = ((2**r - 1)/2**r) * (1 + N/(P*h223))
                a224 = 1 - a223
                SUM223 = math.log2(1 + a223 * h223 * P / (a224 * h223 * P + N) )
                SUM224 = math.log2(1 + h224 * a224 * P/N)
                
            average_sum_rate2 =  SUM221 + SUM222 + SUM223 + SUM224  
            Fairness222 = (SUM221 + SUM222 + SUM223 + SUM224)**2 / (4 * (SUM221**2 + SUM222**2 + SUM223**2 + SUM224**2))

            
            h11.append(Average(self.h1))
            h22.append(Average(self.h2)) 
            h33.append(Average(self.h3)) 
            h44.append(Average(self.h4)) 
            a11.append(Average(self.a1)) 
            a22.append(Average(self.a2)) 
            a33.append(Average(self.a3)) 
            a44.append(Average(self.a4)) 
            SUM11.append(Average(self.SUM1)) 
            SUM22.append(Average(self.SUM2)) 
            SUM33.append(Average(self.SUM3)) 
            SUM44.append(Average(self.SUM4))
            average_episode_reward = episode_reward/self.episode_step 
            Fairnessl.append(Average(self.Fairness))
            episode_rewards.append(average_episode_reward)
            episode_durations.append(timestep)
            Height.append(Average(self.Hl))
            AVG2.append(average_sum_rate2)
            Fairnessl_2.append(Fairness222)

            print(f"Episode = {episode}")
            if episode == 999:
              plot(episode_rewards,reward1,reward2,reward3,reward4,reward5,reward6,h11,h22,h33,h44,a11,a22,a33,a44,SUM11,SUM22,SUM33,SUM44,Fairnessl,Height,AVG2,Fairnessl_2, 100)
            
              average_h1 = 10 * math.log10(h11[-1])
              average_h2 = 10 * math.log10(h22[-1])
              average_h3 = 10 * math.log10(h33[-1])
              average_h4 = 10 * math.log10(h44[-1])

              average_h21 = 10* math.log10(h221)
              average_h22 = 10* math.log10(h222)
              average_h23 = 10* math.log10(h223)
              average_h24 = 10* math.log10(h224)
            
              average_sum_rate = SUM11[-1] + SUM22[-1] + SUM33[-1] + SUM44[-1]
            
              print("NOMA probability = ", round(np.sum(self.NOMA)/len(self.NOMA )*100,2 ), "%, lenght",len(self.NOMA))
              print("NLOS probability = ", round(np.sum(self.NLOS)/len(self.NLOS )*100,2 ), "%")
              print("h1: ",round(average_h1, 2),"dB, h2: ",round(average_h2, 2),"dB, h3: ",round(average_h3, 2),"dB, h4: ",round(average_h4, 2),"dB")
              print("a1: ",round(a11[-1]*100, 2),"%, a2: ",round(a22[-1]*100, 2),"%, a3: ",round(a33[-1]*100, 2),"%, a4: ",round(a44[-1], 2)*100,"%")
              print("SE1: ",round(SUM11[-1], 2),"Bits/s/Hz, SE2: ",round(SUM22[-1], 2),"Bits/s/Hz, SE3: ",round(SUM33[-1], 2),"Bits/s/Hz, SE4: ",round(SUM44[-1], 2),"Bits/s/Hz")
              print("Total SE = ", round(average_sum_rate, 2), "Bits/s/Hz")
              print("Total Sum Rate = ", round(2*average_sum_rate, 2), "Gbps")
              print("Fairness = ", round(100*Fairnessl[-1],2), "%")
              print("Height = ", round(Height[-1],2),"m")

              print("\n                          UAV2                            ")
              print("Fairness = ", round(Fairness222 *100,2), "%")
              print("h1: ",round(average_h21, 2),"dB, h2: ",round(average_h22, 2),"dB, h3: ",round(average_h23, 2),"dB, h4: ",round(average_h24, 2),"dB")
              print("a1: ",round(a221*100, 2),"%, a2: ",round(a222*100, 2),"%, a3: ",round(a223*100, 2),"%, a4: ",round(a224*100, 2),"%")
              print("SE1: ",round(SUM221, 2),"Bits/s/Hz, SE2: ",round(SUM222, 2),"Bits/s/Hz, SE3: ",round(SUM223, 2),"Bits/s/Hz, SE4: ",round(SUM224, 2),"Bits/s/Hz")
              print("Total SE = ", round(average_sum_rate2, 2), "Bits/s/Hz")
              print("Total Sum Rate = ", round(2*average_sum_rate2, 2), "Gbps")
              print("Height = ", round(H2,2),"m")
            
            done = True
                          
        return new_observation,new_observation_m, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((500, 500)) # resizing
        cv2.imshow("UAV Beta 0.95", np.array(img)) 
        cv2.waitKey(1)

    def get_image(self):
        env = np.full((self.SIZE, self.SIZE, 3), 255, dtype=np.uint8)  # starts an rbg img
        env[self.USER1.x][self.USER1.y] = self.d[(1)]  
        env[self.USER2.x][self.USER2.y] = self.d[(1)]
        env[self.USER3.x][self.USER3.y] = self.d[(2)] 
        env[self.USER4.x][self.USER4.y] = self.d[(2)]
        env[self.UAV.x][self.UAV.y] = self.d[self.UAV_N]
        env[self.UAV2.x][self.UAV2.y] = self.d[3]
        img = Image.fromarray(env, 'RGB')
        return img 

batch_size = 128
gamma = 0.999
eps_start = 0.9
eps_end = 0.05
eps_decay = 200
target_update = 10
memory_size = 15000
lr = 0.001
num_episodes = 1000
num_of_actions = 32
num_of_arg_per_state = 17
SHOW_PREVIEW = False
AGGREGATE_STATS_EVERY = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = BlobEnv()
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, num_of_actions, device)
memory = ReplayMemory(memory_size)
policy_net = DQN(num_of_arg_per_state).to(device)
target_net = DQN(num_of_arg_per_state).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_durations = []
episode_rewards = []
episode_wins = []
h11 = []
h22 = []
h33 = []
h44 = []
a11 = []
a22 = []
a33 = []
a44 = []
SUM11 = []
SUM22 = []
SUM33 = []
SUM44 = []
TOTAL_SUM = []
Fairnessl = []
Height = []
reward1 = []
reward2 = []
reward3 = []
reward4 = []
reward5 = []
reward6 = []
AVG2 = []
Fairnessl_2 = []
BEST_REWARD = 0

for episode in range(num_episodes):
    state = torch.tensor([em.reset()], dtype=torch.float32).to(device)
    episode_reward = 0
    episode_win = 0

    for timestep in count():   
        action = agent.select_action(state, policy_net)
        next_state, next_state_m, reward, done = em.step(action.item())
        episode_reward += reward
        reward = torch.tensor([reward], dtype=torch.int64).to(device)
        next_state = torch.tensor([next_state], dtype=torch.float32).to(device)
        next_state_m = torch.tensor([next_state_m], dtype=torch.float32).to(device)        
        memory.push(Experience(state, action, next_state_m, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)
            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if 1:
            em.render()
            
        if done:         
            break

    if episode_reward > BEST_REWARD:
      torch.save(policy_net, 'SIV_MODEL.pt')
      BEST_REWARD = episode_reward
      print("\nNew Best Reward = ", BEST_REWARD, '\n')

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

