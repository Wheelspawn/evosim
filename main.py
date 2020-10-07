#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 14:22:20 2020

@author: nsage
"""

import os
import sys
from noise import snoise2
from scipy.ndimage.filters import gaussian_filter
import numpy as np

np.random.seed(10000)
np.random.seed(1000)

from sklearn.neural_network import MLPClassifier

class Simulation(object):
    def __init__(self,size,simAge=0):
        self.m = size[0]
        self.n = size[1]
        self.agents = []
        self.deadAgents = []
        self.hives = []
        self.food = []
        self.agents = []
        self.simAge = simAge
    
    def initRandomAgents(self,agentCount):
        
        self.agents = [Agent(0,0) for i in range(agentCount)]
        
    def generateMap(self,food_count=40,max_int=640):
        seed1 = np.random.randint(-(10**5),(10**5))
        seed2 = np.random.randint(-(10**5),(10**5))
        self.map = np.array([[snoise2(i+seed1,j+seed2) for i in range(self.n)] for j in range(self.m)])
        self.map = gaussian_filter(self.map,sigma=1.5)
        self.map += abs(self.map.min())
        self.map /= self.map.max()
        self.map *= max_int
        self.map = self.map.astype('int32').astype(object)
        
        self.assignTilesToMap()
        # self.dropStones(200)
        self.foodCount = food_count
        self.dropInitialFood(self.foodCount)
        # self.dropInitialHives(10)
        self.circleWithMountains()
        # self.agents = [Agent(0,0) for i in range(self.agentCount)]
        # self.dropInitialAgents(self.agentCount)
        self.stats = []
        
    def dropFlat(self,item,amount):
        for i in range(amount):
            dropN,dropM = np.random.randint(self.n),np.random.randint(self.m)
            self.map[dropM][dropN].addItem(item)
        
    def dropGaussian(self,item,amount,std,m,n):
        for i in range(amount):
            dropM,dropN = int(np.random.normal(m,std)),int(np.random.normal(n,std))
            self.map[dropM][dropN].add(item)
    
    def dropStones(self,dropNum):
        for i in range(dropNum):
            dropN,dropM = np.random.randint(self.n),np.random.randint(self.m)
            self.map[dropM][dropN] = Tile("Wall",graphic=(97,"≡"),impassable=True)
            
    def dropInitialFood(self,dropNum):
        center = (self.m//2,self.n//2)
        std = min(self.m,self.n) // 4
        y,x=center[0],center[1]
        while dropNum > 0:
            dropN,dropM = int(np.random.normal(x,std)),int(np.random.normal(y,std))
            try:
                if self.map[dropM][dropN].impassable == False:
                    self.map[dropM][dropN].food = True
                    dropNum -= 1
            except IndexError:
                pass
    
    def dropInitialAgents(self):
        center = (self.m//2,self.n//2)
        std = min(self.m,self.n) // 8
        y,x=center[0],center[1]
        agents_left = self.agents[:]
        while agents_left != []:
            agent = agents_left[-1]
            dropN,dropM = int(np.random.normal(x,std)),int(np.random.normal(y,std))
            try:
                if (self.map[dropM][dropN].impassable == False) and (dropM,dropN) not in [(agent.m,agent.n) for agent in self.agents]:
                    agent.m = dropM
                    agent.n = dropN
                    if self.map[dropM][dropN].food == False:
                        self.map[dropM][dropN].food = True # give the evolution a head start by incentivizing agents to chew food tiles
                        self.foodCount += 1
                    agents_left.pop()
            except IndexError:
                pass
    
    '''
    def dropInitialHives(self,dropNum):
        center = (self.m//2,self.n//2)
        std = min(self.m,self.n) // 15
        y,x=center[0],center[1]
        for i in range(dropNum):
            dropN,dropM = int(np.random.normal(x,std)),int(np.random.normal(y,std))
            self.map[dropM][dropN] = Tile("Wall",graphic=(97,"≡"),impassable=True)
    '''
    
    def circleWithMountains(self):
        for i in range(0,self.m):
            self.map[i,0] = Tile("Wall",graphic=(97,"≡"),impassable=True)
            self.map[i,-1] = Tile("Wall",graphic=(97,"≡"),impassable=True)
        for j in range(0,self.n):
            self.map[0,j] = Tile("Wall",graphic=(97,"≡"),impassable=True)
            self.map[-1,j] = Tile("Wall",graphic=(97,"≡"),impassable=True)
    
    def tick(self,showMap=False):
        

        for agent in self.agents:
            
            if agent.hunger > 1:
                '''
                print("~~~~~~~~~~~~~~~~~~")
                print(agent.hunger)
                print("~~~~~~~~~~~~~~~~~~\n")
                '''
            
            inputs = np.array([0.0,0.0,0.0,0.0,0.0]) # np.array([0.0,0.0,0.0,0.0,0.0,np.tanh((agent.health-(agent.maxhealth/2))*0.2),np.tanh((agent.hunger-(agent.maxhunger/2))*0.2)])
            closest_food=(0,0,np.inf)
            closest_agent=(0,0,np.inf)
            for i in range(agent.m-5,agent.m+6):
                for j in range(agent.n-5,agent.n+6):
                    
                    '''
                    if i == agent.m and j == agent.n:
                        print("@",end="")
                    else:
                        print(self.map[i][j],end="")
                    '''
                        
                    try:
                        if self.map[i][j].food == True:
                            r = np.linalg.norm(np.array([agent.m,agent.n])-np.array([i,j]))
                            if r < closest_food[2]:
                                closest_food = (i-agent.m,j-agent.n,r)
                        if (agent.m != i and agent.n != j) and isAgentOnTile((i,j),self.agents):
                            r = np.linalg.norm(np.array([agent.m,agent.n])-np.array([i,j]))
                            if r < closest_agent[2]:
                                closest_agent = (i-agent.m,j-agent.n,r)
                    except IndexError:
                        # if near edge of map, agent's receptive field may go out of bounds. Out-of-bounds tiles should be skipped.
                        pass
                # print()
            
            inputs[0] = 0 if closest_food[0] == 0 else np.tanh(1/(closest_food[0]))
            inputs[1] = 0 if closest_food[1] == 0 else np.tanh(1/(closest_food[1]))
            inputs[2] = 25 if self.map[agent.m][agent.n].food == True else -1
            
            inputs[3] = 0 if closest_agent[0] == 0 else np.tanh(1/(closest_agent[0]))
            inputs[4] = 0 if closest_agent[1] == 0 else np.tanh(1/(closest_agent[1]))
            
            outputs = agent.feedForward(inputs)
            
            decision = np.argmax(outputs)
            
            '''
            print(closest_food,closest_agent)
            print((agent.m,agent.n),": ",inputs.round(4)," ",outputs.round(4))
            print(decision)
            print('~~~~~~~~~~~~~~\n')
            '''
            
            if outputs[decision] > 0.5:
                if (decision == 4) and (self.map[agent.m][agent.n].food == True):
                    # print("==================")
                    # print(agent.hunger)
                    agent.hunger = min(agent.maxhunger,agent.hunger+5)
                    # print(agent.hunger)
                    self.map[agent.m][agent.n].food = False
                    self.foodCount -= 1
                    # print("==================\n")
                else:
                    new_coords = [agent.m,agent.n]
                    if (decision == 0):
                        new_coords[0] -= 1 # go up
                    if (decision == 1):
                        new_coords[0] += 1 # go down
                    if (decision == 2):
                        new_coords[1] -= 1 # go left
                    if (decision == 3):
                        new_coords[1] += 1 # go right
                    
                    if (self.map[new_coords[0],new_coords[1]].impassable != True) and (not isAgentOnTile(tuple(new_coords),self.agents)):
                        agent.m,agent.n = new_coords
                
            agent.tickDown()
            if agent.isDead == True:
                self.agents.remove(agent)
                self.deadAgents.insert(0,agent)
        
        st = Stat(self.simAge,self.foodCount,len(self.agents))
        self.stats.append(st)
        
        if showMap == True:
            self.printMap()
            c = input()
            print(st.age,st.foodCount,st.agentsLeft)
        self.simAge += 1
   
    def assignTilesToMap(self):
        d = {}
        for i in range(0,636):
            d[i] = ("Empty",(32," "),0,0)
        for i in range(636,641):
            d[i] = ("Wall",(97,"≡"),0,0,True)
        
        for m in range(self.m):
            for n in range(self.n):
                self.map[m][n] = Tile(*d[self.map[m][n]])
                
    def printMap(self):
        
        for m in range(self.m):
            for n in range(self.n):
                if isAgentOnTile((m,n),self.agents):
                    print(getTileGraphic(33,"@"),end="")
                else:
                    print(self.map[m][n],end="")
            print()

class Stat():
    def __init__(self,age,foodCount,agentsLeft):
        self.age = age
        self.foodCount=foodCount
        self.agentsLeft=agentsLeft

def isAgentOnTile(coords,agents):
    if coords in [(agent.m,agent.n) for agent in agents]:
        return True
    else:
        return False
    
def getTileGraphic(color,char):
    return '\33[{}m{}'.format(color,char)

def getStraightLineDist(x1,y1,x2,y2):
    '''
    returns angle (theta) and radius (r) from (x1,y1) to (x2,y2).
    '''
    r = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    theta = np.cosh(x2-x1)
    return (round(r,1),round(theta,1))

class Tile(object):
    def __init__(self,name,graphic,damage=0,slows=0,impassable=False,food=False):
        self.name=name
        self.damage = damage
        self.slows = slows
        self.impassable = impassable
        self.food = food
        
        self.setGraphic(graphic)
    
    def addItem(self,item):
        if (self.impassable or self.item != None):
            return -1
        else:
            self.item = item
    
    def setGraphic(self,graphic):
        color,char=graphic
        if type(color) == list:
            self.color = color[np.random.random(len(color))]
        else:
            self.color = color
        if type(char) == list:
            self.char = char[np.random.random(len(char))]
        else:
            self.char = char
    
    def __repr__(self):
        if self.food == True:
            return getTileGraphic(94,"º")
        return getTileGraphic(self.color,self.char)
            
terrainTypes = { "Coast":("Coast",(34,"~")),
                 "Desert":("Desert",(32," ")),
                 "Grass":("Grass",(92,"`")),
                 "Forest":("Forest",(32,"¥")),
                 "Mountain":("Mountain",(33,"^"),True),
                 "Ice Cap":("Ice Cap",(97,"^")) }

def logistic(x):
    return 1/(1+np.e**(-x))

class Agent(object):
    def __init__(self,m,n,color=93,char="☺",origin="Initial",health=30,hunger=20,isDead=False,layers=(5,5,5),recurrencies=[]):
        self.m = m
        self.n = n
        self.char = char
        self.origin = origin
        self.health = health
        self.maxhealth = health
        self.hunger = hunger
        self.maxhunger = 30
        self.isDead = isDead
        self.layers = layers
        self.recurrencies = recurrencies
        self.initializeNN()
        
    def initializeNN(self):
        self.weights = []
        self.size = 0
        self.outputs = [np.zeros(self.layers[i]) for i in range(1,len(self.layers))]
        for i in range(1,len(self.layers)):
            curr_layer = self.layers[i]
            prev_layer = self.layers[i-1]
            for recurrency in self.recurrencies:
                if i == recurrency[0]:
                    prev_layer += self.layers[recurrency[1]]
                    
            self.weights.append(np.random.normal(size=(curr_layer,prev_layer)))
            self.size += curr_layer*prev_layer
        
    def feedForward(self,inputs,actFuncs=[np.tanh,logistic,logistic,logistic,logistic]):
        activation = inputs
        for i in range(0,len(self.weights)):
            for recurrency in self.recurrencies:
                if i == recurrency[1]-1:
                    activation = np.append(activation,self.outputs[recurrency[0]-1])
            activation = np.apply_along_axis(actFuncs[i], axis=0, arr=(activation * self.weights[i]).sum(axis=1))
            self.outputs[i] = activation
        return activation
    
    def tickDown(self):
        if self.hunger < 15:
            self.health -= 1
            self.hunger = max(0,self.hunger-1)
        else:
            self.health = min(self.maxhealth,self.health+1)
            self.hunger = max(0,self.hunger-1)
        if self.health <= 0:
            self.isDead = True
    
    def reset(self):
        self.hunger = self.maxhunger
        self.health = self.maxhealth
        self.isDead = False
        
    def __repr__(self):
        return getTileGraphic(31,"☺")

class Evolver(object):
    def __init__(self,sim):
        self.sim = sim
        self.newAgents = []
        
    def runEpochs(self,epochs=20,trials=10):
        '''
        self.sim.agents = []
        self.sim.deadAgents = []
        for agent in self.newAgents:
            self.sim.agents.append(agent)
        self.sim.simAge = 0
        '''
        
    def runTrials(self,trials):
        for i in range(0,trials):
            self.runTrial(i)
    
    def runTrial(self,num,show=False):
        import time
        t0 = time.time()
        self.sim.generateMap()
        self.sim.dropInitialAgents()
        while self.sim.agents != []:
            self.sim.tick(show)
        print("Simulation {} age: {}  Remaining food: {}".format(num,self.sim.simAge,self.sim.foodCount))
        # print("Dead: {}  Living: {}  New: {}".format(len(self.sim.deadAgents),len(self.sim.agents),len(self.newAgents)))
        print("Winners: ",[agent.origin for agent in self.sim.deadAgents[0:5]])
        self.evolveAgentNets(num)
        print(self.sim.deadAgents[0].weights)
        t1=time.time()
        print("Time to complete in seconds: ", round(t1-t0,2))
        print()
        self.sim.agents = []
        self.sim.deadAgents = []
        for agent in self.newAgents:
            self.sim.agents.append(agent)
        self.sim.simAge = 0
            
    def evolveAgentNets(self,num):
        self.newAgents = []
        for agent1 in self.sim.deadAgents[0:5]:
            for agent2 in self.sim.deadAgents[0:5]:
                if agent1 is not agent2:
                    bredAgent = self.breed(agent1,agent2,np.random.random())
                    bredAgent.origin = "Hybrid (T. {})".format(num)
                    self.newAgents.append(bredAgent) # breed top 5 for 20 new networks
        for i in range(10):
            for agent in self.sim.deadAgents[0:5]:
                mutatedAgent = self.mutate(agent)
                mutatedAgent.origin = "Mutant (T. {})".format(num)
                self.newAgents.append(mutatedAgent)            # mutate top 5, 10 times each, for 50 new networks
        for i in range(5):
            randomAgent = Agent(0,0,origin="Random (T. {})".format(num))
            self.newAgents.append(randomAgent)        # create 5 new random network
        for agent in self.sim.deadAgents[0:5]:
            agent.reset()
            self.newAgents.append(agent)                             # keep original top 5 networks
            
    def breed(self,nn1,nn2,ratio=0.5):
        # creates a hybrid network with neural weights randomly chosen from a pair of others
        if nn1.layers != nn2.layers:
            return False
        nn=Agent(0,0,origin="Hybrid",layers=nn1.layers)
        nn.initializeNN()
        for i in range(len(nn1.weights)):
            for j in range(len(nn1.weights[i])):
                for k in range(len(nn1.weights[i][j])):
                    if np.random.random() < ratio:
                        nn.weights[i][j][k] = nn1.weights[i][j][k]
                    else:
                        nn.weights[i][j][k] = nn2.weights[i][j][k]
        return nn
    
    def mutate(self,nn1,chance=0.2,std=1):
        # randomly tweaks neural network weights
        nn=Agent(0,0,origin="Mutant",layers=nn1.layers)
        nn.initializeNN()
        for i in range(len(nn1.weights)):
            for j in range(len(nn1.weights[i])):
                for k in range(len(nn1.weights[i][j])):
                    if np.random.random() < chance:
                        nn.weights[i][j][k] = nn1.weights[i][j][k] + np.random.normal(0.0,std)
                        nn.weights[i][j][k] = min(max(nn.weights[i][j][k],-10.0),10.0) # constrain to avoid extreme weight values
        return nn

s=Simulation((40,80))
s.initRandomAgents(80)
e=Evolver(s)
e.runTrials(200)
e.runTrial(201,True)
e.runTrial(202,True)
e.runTrial(203,True)
e.runTrial(204,True)
e.runTrial(205,True)
e.runTrial(206,True)
e.runTrial(207,True)
e.runTrial(208,True)
e.runTrial(209,True)
e.runTrial(200,True)

'''
for agent in s.agents[-5:]:
    print(agent.origin)
    print(agent.weights)
    print()
c = input()
'''

'''
s=Simulation((70,140))
s.initRandomAgents(50)
e=Evolver(s)
e.runTrials(10000)
'''

'''
if __name__ == "__main__":
    
    s=Simulation((70,140))
    s.generateMap()
    s.tick()
    
    c = input()
    while c.lower() != "q":
        os.system('clear')
        s.tick(True)
        print("World age: ", s.simAge)
        c = input()
    '''
    
# converged agent (np.random.seed(10000))?
a=Agent(0,0,layers=(5,5,5))
a.initializeNN()
a.weights = [np.array([[-0.50839146, -0.83668727,  1.00816155,  0.74734443,  0.01910776],
                       [-0.43937005, -1.36122086,  0.7504787 ,  1.16811406,  0.19985127],
                       [-0.72029398, -1.55944746, -0.48405408, -0.32161156,  0.65008911],
                       [-1.13554804,  4.77722425,  1.19722424,  0.7363022 ,  1.28432408],
                       [-0.72696608,  1.8614612 ,  0.58240229, -0.04779921,  0.37044165]]),
             np.array([[-0.7458011 ,  0.0257888 ,  0.40147646, -0.19737577, -1.363852  ],
                       [-0.1949952 ,  0.49181767,  0.84829215, -0.42646396,  0.10306288],
                       [-2.22175726, -0.08119867, -0.71641157, -0.00428541,  0.20967456],
                       [-0.89604738, -0.24200379, -0.88109637, -0.50141238,  0.96528897],
                       [ 1.5473343 ,  0.08318062, -1.43061149, -0.44211267,  0.95750232]])]

a.weights = [np.array([[ 2.0348441 , -0.19611184,  0.71990512, -0.15982703, -0.00594514],
                       [ 0.17469971, -0.58774993,  1.02988576, -0.76489626, -1.86581139],
                       [-0.70608404, -1.21372447, -0.60036466,  1.01787637,  0.81302527],
                       [ 0.29332985, -0.594777  ,  0.64086378,  0.82307924,  0.77835893],
                       [ 0.82858818,  1.18003659, -0.96568295, -0.36396246, -1.24324351]]),
             np.array([[ 0.87354746, -2.01926   ,  0.20373906,  0.90445495,  0.16110998],
                       [-1.61483229,  0.96030318,  0.78182762, -2.05465033, -0.31954914],
                       [ 1.33490152,  0.1097562 ,  0.01741557, -3.62380813,  1.0658164 ],
                       [-0.60888363,  0.33932152,  2.42134969, -0.25836357, -0.45955063],
                       [-1.12807218,  2.19775183, -1.56751394, -0.95926735, -2.22072822]])]

# movement testing

# should be [1.0, 0.0, 0.0, 0.0, 0.0]
print(a.feedForward([-5.0, 0.0, 0.0, 0.0, 0.0]))
# should be [0.0, 1.0, 0.0, 0.0, 0.0]
print(a.feedForward([5.0, 0.0, 0.0, 0.0, 0.0]))
# should be [0.0, 0.0, 0.0, 1.0, 0.0]
print(a.feedForward([0.0, -5.0, 0.0, 0.0, 0.0]))
# should be [0.0, 0.0, 1.0, 0.0, 0.0]
print(a.feedForward([0.0, 5.0, 0.0, 0.0, 0.0]))
# should be [0.0, 0.0, 0.0, 0.0, 1.0]
print(a.feedForward([0.0, 0.0, 25.0, 0.0, 0.0]))

