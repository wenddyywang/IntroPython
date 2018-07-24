"""
Wendy Wang
www2105
"""

import random
import math
from matplotlib import pyplot as plt


def normpdf(x, mean, sd):
    """
    Return the value of the normal distribution 
    with the specified mean and standard deviation (sd) at
    position x.
    You do not have to understand how this function works exactly. 
    """
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def pdeath(x, mean, sd):
    start = x-0.5
    end = x+0.5
    step =0.01    
    integral = 0.0
    while start<=end:
        integral += step * (normpdf(start,mean,sd) + normpdf(start+step,mean,sd)) / 2
        start += step            
    return integral    
    
recovery_time = 4 # recovery time in time-steps
virality = 0.6   # probability that a neighbor cell is infected in 
                  # each time step                                                  

class Cell(object):

    def __init__(self,x, y):
        self.x = x
        self.y = y 
        self.state = "S" # can be "S" (susceptible), "R" (resistant = dead), or 
                         # "I" (infected)
        self.steps = 0
    
    def infect(self):
        self.state = "I"
        self.steps += 1
    
    def recover(self):
        self.state = "S"
        self.steps = 0
    
    def die(self):
        self.state = "R"
    
    def process(self, adjacent_cells):
        if self.state == "I":
            if self.steps == recovery_time:
                self.recover()
            elif random.random() <= pdeath(self.steps,4,1):
                self.die()
            else:
                self.steps += 1
                for cell in adjacent_cells:
                    if cell.get_state() == "S" and random.random() <= virality:
                        cell.infect()
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
    
    def get_state(self):
        return self.state
    
    def get_steps(self):
        return self.steps
    
    def __str__(self):
        return str(self.x) + " " + str(self.y) + " " + self.state
        
class Map(object):
    
    def __init__(self):
        self.height = 150
        self.width = 150           
        self.cells = {}

    def add_cell(self, cell):
        self.cells[(cell.get_x(), cell.get_y())] = cell
        #print(cell.get_x() + ", " + cell.get_y() + ", " + cell.get_state())
        
    def display(self):
        #print(self.cells.values())
        image = [[(0.0,0.0,0.0) for i in range(self.width)] for j in range(self.height)]
        for (key, cell) in self.cells.items():
            state = cell.get_state()
            if state == "S":
                image[cell.get_x()][cell.get_y()] = (0.0,1.0,0.0) #susceptible green
            elif state == "R":
                image[cell.get_x()][cell.get_y()] = (0.5, 0.5, 0.5) #resistant/dead gray
            else:
                image[cell.get_x()][cell.get_y()] = (1.0, 0.0, 0.0) #infected red
        plt.imshow(image)
    
    def adjacent_cells(self, x,y):
        adj_cells = []
        if x > 0:
            if (x - 1,y) in self.cells:
                adj_cells.append(self.cells[(x - 1,y)])
        if x < self.width - 1:
            if (x + 1,y) in self.cells:
                adj_cells.append(self.cells[(x + 1,y)])
        if y > 0:
            if (x,y - 1) in self.cells:
                adj_cells.append(self.cells[(x,y - 1)])
        if y < self.height - 1:
            if (x,y + 1) in self.cells:
                adj_cells.append(self.cells[(x,y + 1)])
        return adj_cells
    
    def time_step(self):
        for (key, cell) in self.cells.items():
            cell.process(self.adjacent_cells(cell.get_x(), cell.get_y()))
        
        self.display()

            
def read_map(filename):
    
    m = Map()
    nyc_map = open(filename,'r')
    for line in nyc_map:
        coord = line.strip().split(",")
        cell = Cell(int(coord[0]),int(coord[1]))
        #print(cell)
        m.add_cell(cell)
    return m
