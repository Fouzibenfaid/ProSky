import os
import pickle
import neat
import numpy as np
import math
import random
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle

r = 3



"""	try:
		with open(f'Pickle/Fitness_d.pickle', 'rb') as f:
			fitness_d = pickle.load(f)
			print("Fitness_d has been loaded successfully")
			print(fitness_d)

	except:
		fitness_d = {}
		print("Fitness_d hasn't been loaded successfully")
		print(fitness_d)"""

runs_per_net = 1



rewards = []
rewards_per_generation = []



def set_key(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = value
    elif type(dictionary[key]) == list:
        dictionary[key].append(value)
    else:
        dictionary[key] = [dictionary[key], value]


def mmLineOfSight_Check(D,H):
    return 1
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
        return [(self.x-other.x), (self.y-other.y)]

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
            self.move(x=-1, y=1)
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
            
        elif choice == 16:
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
            self.move(x=-1, y=1)
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
        P = 0.1 # Transmitted power
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
        
        self.UAV.a1 = 0.5
        self.UAV.a2 = 0.5
        self.UAV.a3 = 0.5
        self.UAV.a4 = 0.5
        self.UAV.H = 50
        
        self.USER1 = Blob(self.SIZE, True, False, False, False)
        self.USER2 = Blob(self.SIZE, False, True, False, False)
        self.USER3 = Blob(self.SIZE, False, False, True, False)
        self.USER4 = Blob(self.SIZE, False, False, False, True)
        
        
        ob1 = self.UAV-self.USER1
        ob2 = self.UAV-self.USER2
        ob3 = self.UAV-self.USER3
        ob4 = self.UAV-self.USER4
        
        D1 =  np.sum(np.sqrt([(ob1[0])**2, (ob1[1])**2]))
        D2 = np.sum(np.sqrt([(ob2[0])**2, (ob2[1])**2]))
        D3 = np.sum(np.sqrt([(ob3[0])**2, (ob3[1])**2]))
        D4 = np.sum(np.sqrt([(ob4[0])**2, (ob4[1])**2]))
                  
        H = self.UAV.H
        Dt1 = np.sum(np.sqrt([ (ob1[0])**2, (ob1[1])**2, H**2  ]))
        Dt2 = np.sum(np.sqrt([ (ob2[0])**2, (ob2[1])**2, H**2  ]))
        Dt3 = np.sum(np.sqrt([ (ob3[0])**2, (ob3[1])**2, H**2  ]))
        Dt4 = np.sum(np.sqrt([ (ob4[0])**2, (ob4[1])**2, H**2  ]))
        
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

        observation =   [ob1[0]] + [ob1[1]] + [ob2[0]] + [ob2[1]]+ [ob3[0]] + [ob3[1]] + [ob4[0]] + [ob4[1]] + [a1] + [a2] + [a3] + [a4] +[h1] + [h2] + [h3] + [h4] + [H]
            
        self.episode_step = 0

        return observation

    def step(self, action):
        
        done= False
        P = .1 # Transmitted power
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
        
        D1 =  np.sum(np.sqrt([(ob1[0])**2, (ob1[1])**2]))
        D2 = np.sum(np.sqrt([(ob2[0])**2, (ob2[1])**2]))
        D3 = np.sum(np.sqrt([(ob3[0])**2, (ob3[1])**2]))
        D4 = np.sum(np.sqrt([(ob4[0])**2, (ob4[1])**2]))
                  
        H = self.UAV.H
        Dt1 = np.sum(np.sqrt([ (ob1[0])**2, (ob1[1])**2, H**2  ]))
        Dt2 = np.sum(np.sqrt([ (ob2[0])**2, (ob2[1])**2, H**2  ]))
        Dt3 = np.sum(np.sqrt([ (ob3[0])**2, (ob3[1])**2, H**2  ]))
        Dt4 = np.sum(np.sqrt([ (ob4[0])**2, (ob4[1])**2, H**2  ]))
        
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
        
        self.UAV.action(action)
        
        a1 =  self.UAV.a1
        a2 =  1 - a1
        a3 =  self.UAV.a3
        a4 =  1 - a3

        reward = 0
        
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

        if SUM1 >= r:
       	 	reward += 100
       	if SUM2 >= r:
       		reward += 100
       	if SUM3 >= r:
       		reward += 100
       	if SUM4 >= r:
       		reward += 100

       	if reward >= 400:
       		SUM1*=10
       		SUM2*=10
       		SUM3*=10
       		SUM4*=10

       	reward +=   (SUM1 + SUM2 + SUM3 + SUM4)

        new_observation =  [ob1[0]] + [ob1[1]] + [ob2[0]] + [ob2[1]]+ [ob3[0]] + [ob3[1]] + [ob4[0]] + [ob4[1]] + [a1] + [a2] + [a3] + [a4] +[h1] + [h2] + [h3] + [h4] + [H]   
        if self.episode_step >= 300:
       		done = True

        return new_observation, reward, done

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        em = BlobEnv()
        observation = em.reset()
        fitness = []
        done = False
        while not done:

            action = np.argmax(net.activate(observation))
            observation, reward, done = em.step(action)
            fitness.append(reward)

        fitnesses.append(np.mean(fitness))
        rewards_per_generation.append(fitness)


    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(i):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    
    config_path = os.path.join('', 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate, 1000)

    with open(f'Pickle/1000gen/Winner-r={r}-i={i}.pickle', 'wb') as f:
    	pickle.dump(winner, f)

    statistics = stats

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

"""    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    plt.show()"""

    # Save the winner.

if __name__ == '__main__':

	for i in [1, 2, 3, 4]:

		run(i)
	