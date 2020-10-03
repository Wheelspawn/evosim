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

from sklearn.neural_network import MLPClassifier

class Simulation(object):
    def __init__(self,size,simAge=0):
        self.m = size[0]
        self.n = size[1]
        self.agents = []
        self.hives = []
        self.food = []
        
        self.simAge = simAge
        
    def generateMap(self,max_int=1280):
        seed1 = np.random.randint(-(10**5),(10**5))
        seed2 = np.random.randint(-(10**5),(10**5))
        self.map = np.array([[snoise2(i+seed1,j+seed2) for i in range(self.n)] for j in range(self.m)])
        self.map = gaussian_filter(s.map,sigma=1.5)
        self.map += abs(self.map.min())
        self.map /= self.map.max()
        self.map *= max_int
        self.map = self.map.astype('int32')
        
        self.dropStones(200)
        self.dropInitialFood(25)
        self.dropInitialHives(10)
        
        self.circleWithMountains()
        
    def dropFlat(self,item,amount):
        for i in range(amount):
            dropN,dropM = np.random.randint(self.n),np.random.randint(self.m)
            self.map[dropM][dropN].add(item)
        
    def dropGaussian(self,item,amount,std,m,n):
        for i in range(amount):
            dropM,dropN = int(np.random.normal(m,std)),int(np.random.normal(n,std))
            self.map[dropM][dropN].add(item)
    
    def dropStones(self,dropNum):
        for i in range(dropNum):
            dropN,dropM = np.random.randint(self.n),np.random.randint(self.m)
            self.map[dropM][dropN] = -2

    def dropInitialFood(self,dropNum):
        center = (self.m//2,self.n//2)
        std = min(self.m,self.n) // 10
        y,x=center[0],center[1]
        for i in range(dropNum):
            dropN,dropM = int(np.random.normal(x,std)),int(np.random.normal(y,std))
            self.map[dropM][dropN] = -3
            
    def dropInitialHives(self,dropNum):
        center = (self.m//2,self.n//2)
        std = min(self.m,self.n) // 15
        y,x=center[0],center[1]
        for i in range(dropNum):
            dropN,dropM = int(np.random.normal(x,std)),int(np.random.normal(y,std))
            self.map[dropM][dropN] = -1
            
    def dropInitialFoodItems(self):
        pass
    
    def circleWithMountains(self):
        for i in range(0,self.m):
            self.map[i,0] = 1279
            self.map[i,-1] = 1279
        for j in range(0,self.n):
            self.map[0,j] = 1280
            self.map[-1,j] = 1280
    
    def tick(self):
        self.printMap()
        self.simAge += 1
   
    def assignTilesToMap(self):
        d = {}
        for i in range(0,256):
            d[i] = getTileGraphic(34," ") # "~~~"
        for i in range(256,512):
            d[i] = " " # "   "
        for i in range(512,896):
            d[i] = getTileGraphic(92," ") # ["`  "," ` ","  `","`` ","` `"," ``","```"][np.random.randint(0,7)]
        for i in range(640,672):
            d[i] = getTileGraphic(32," ")
        for i in range(896,1024):
            d[i] = getTileGraphic(97,"≡") # [" ¥ ","  ¥","¥¥ ","¥ ¥"," ¥¥","¥¥¥"][np.random.randint(0,6)]
        for i in range(1024,1281):
            d[i] = getTileGraphic(97,"≡") # [" ^ ","  ^","^^ ","^ ^"," ^^","^^^"][np.random.randint(0,6)]
        
        d[-3] = getTileGraphic(94,"º")
        d[-2] = getTileGraphic(97,"≡")
        d[-1] = getTileGraphic(31,"¤")
        
    def printMap(self):
        
        for m in range(self.m):
            for n in range(self.n):
                print([self.map[m][n]],end="")
                
            print()
        
def getTileGraphic(color,char):
    return '\33[{}m{}'.format(color,char)

def getStraightLineDist(x1,y1,x2,y2):
    '''
    returns angle (theta) and radius (r) from (x1,y1) to (x2,y2).
    '''
    r = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    theta = np.cosh(x2-x1)
    return (round(r,1),round(theta,1))

print(getStraightLineDist(0,0,0,0))
print(getStraightLineDist(0,0,1,0))
print(getStraightLineDist(0,0,1,1))
print(getStraightLineDist(0,0,0,1))
print(getStraightLineDist(0,0,-1,1))
print(getStraightLineDist(0,0,-1,0))
print(getStraightLineDist(0,0,-1,-1))
print(getStraightLineDist(0,0,0,-1))
print(getStraightLineDist(0,0,1,-1))

class Tile(object):
    def __init__(self,name,graphic,damage=0,slows=0,impassable=False,hive=None,food=None):
        self.name=name
        self.damage = damage
        self.slows = slows
        self.impassable = impassable
        self.hive = hive
        self.food = food
        
        self.setGraphics(graphic)
    
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
            
terrainTypes = { "Coast":("Coast",(34,"~")),
                 "Desert":("Desert",(32," ")),
                 "Grass":("Grass",(92,"`")),
                 "Forest":("Forest",(32,"¥")),
                 "Mountain":("Mountain",(33,"^"),True),
                 "Ice Cap":("Ice Cap",(97,"^")) }

def logistic(x):
    return 1/(1+np.e**(-x))

class Agent(object):
    def __init__(self,color=93,char="☺",health=50,hunger=50,isDead=False,layers=(6,6,6),recurrencies=[]):
        self.char = char
        self.health = health
        self.hunger = hunger
        self.isDead = isDead
        self.layers = layers
        self.recurrencies = recurrencies
        self.initializeNN()
        
    def initializeNN(self):
        self.weights = []
        self.outputs = [np.zeros(self.layers[i]) for i in range(len(self.layers)-1)]
        for i in range(1,len(self.layers)):
            curr_layer = self.layers[i]
            prev_layer = self.layers[i-1]
            for recurrency in self.recurrencies:
                if i == recurrency[0]:
                    prev_layer += self.layers[recurrency[1]]
                    
            self.weights.append(np.random.random((curr_layer,prev_layer)).astype('float32')*2-1)
        
    def feedForward(self,inputs,actFunc=logistic):
        activation = inputs
        for i in range(0,len(self.weights)):
            for recurrency in self.recurrencies:
                if i == recurrency[1]-1:
                    activation = np.append(activation,self.outputs[recurrency[0]-1])
            activation = np.apply_along_axis(actFunc, axis=0, arr=(activation * self.weights[i]).sum(axis=1))
            self.outputs[i] = activation
        return activation
    
    def tickDown(self):
        if self.hunger < 25:
            self.health -= 1
            self.hunger -= 1
        else:
            self.health = min(50,self.health+1)
            self.hunger -= 1
        if self.health <= 0:
            self.isDead = True

a=Agent()
# a.feedForward([-5,5])

if __name__ == "__main__":
    
    
    s=Simulation((60,100))
    s.generateMap()
    '''
    s.tick()
    
    c = input()
    while c.lower() != "q":
        os.system('clear')
        s.tick()
        print("World age: ", s.simAge)
        c = input()
        '''