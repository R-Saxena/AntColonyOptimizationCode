import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
#<----------------------------------------  problem preparation  ------------------------------------------>

#Create the graph
def create_graph():

    x = [0.09, 0.16, 0.84, 0.70,100,20,2000,10]
    y = [0.17, 0.52, 0.92, 0.16, 50,40,30,20]
    graph = {}
    n = len(x)
    node = []
    for i in range(n):
        node.append((x[i], y[i]))

    edges = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            x1 = node[i][0]
            x2 = node[j][0]

            y1 = node[i][1]
            y2 = node[j][1]

            #finding the euclidean distance
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)  
            edges[i][j] = distance
    
    graph['n'] = n
    graph['node'] = node
    graph['edges'] = edges
    return graph

#Draw the graph
def draw_graph(graph):
    for i in range(graph['n'] - 1):
        for j in range(i+1, graph['n']):
            x1 = graph['node'][i][0]
            y1 = graph['node'][i][1]
            x2 = graph['node'][j][0]
            y2 = graph['node'][j][1]
            X = [x1, x2]
            Y = [y1, y2]
            plt.plot(X,Y, '-k')
    
    for i in range(graph['n']):
        X = [i[0] for i in graph['node']]
        Y = [i[1] for i in graph['node']]
        plt.plot(X,Y, 'or')

    plt.show()
    
    return

#<-----------------------------------------------ACO algo--------------------------------------------------> 


#initial parameters of ACO


class ACO:
    def __init__(self, graph, max_iterations = 100, antNo = 10, rho = 0.05, alpha = 1, beta = 1):

        self.max_iterations = max_iterations           #how long you are gonna look for the solution

        self.graph = graph

        self.antNo = antNo                             #how many artifical ant will look for the solution

        self.tau0 = 10*1/(graph['n'] * np.mean(graph['edges']))   # intial concentration of the pheromone

        self.tau = np.ones((graph['n'], graph['n']))   # pheromone matrix

        self.eta = graph['edges']                      # desirability of each edge

        self.rho = rho                                 # evaporation rate

        self.alpha = alpha                             # pheromone exponential parameter

        self.beta = beta                               # desirability exponential parameter

        self.colony = self.initialize_colony()

        return
    

    def initialize_colony(self):
        d = {}
        for i in range(self.antNo):
            d[i] = []
        
        return d

    # function to select the next node based on the cummulative probabilities
    def roulette_wheeler(self, prob):
        cumsum = np.cumsum(prob)
        r = random.random()

        for i in range(len(cumsum)):
            if r <= cumsum[i]:
                nextnode = i
                break

        return nextnode


    def create_ants(self):
        nodeNo = self.graph['n']
        
        for i in range(self.antNo):
            starting_node = random.randint(0, nodeNo-1)   #select a random node

            self.colony[i].append(starting_node)

            # selecting rest of the nodes
            for j in range(1,nodeNo):
                current_node = self.colony[i][-1]

                P_allNodes = (self.tau[current_node,:]**self.alpha)*(self.eta[current_node, :]**self.beta)
                P_allNodes[self.colony[i]] = 0
                P = P_allNodes/ np.sum(P_allNodes)

                next_node = self.roulette_wheeler(P)
                self.colony[i].append(next_node)
            
            #complete the route
            self.colony[i].append(starting_node)
        
        return

    
    def fitness_function(self, ant):   #defines how good ants are 
        fitness = 0   # in our case it is distance 
        for i in range(len(ant)-1):
            current_node = ant[i]
            next_node = ant[i+1]
            fitness += self.graph['edges'][current_node][next_node]
        
        return fitness


    #finding the queen
    def find_best_ant(self):
        queen_index = 0
        queen_fitness = float('inf')                 #best ant 
        for i in range(self.antNo):
            fitness = self.fitness_function(self.colony[i])
            if queen_fitness > fitness:
                queen_index = i
                queen_fitness = fitness
        
        queen = {}
        queen['index'] = queen_index
        queen['fitnessScore'] = queen_fitness
        return queen


    #for stigmergic effect, we need to update pheromone
    def update_pheromone(self):
        nodeNo = len(self.colony[0])
        for i in range(self.antNo):
            for j in range(nodeNo-1):

                current_node = self.colony[i][j]
                next_node = self.colony[i][j+1]
                fitness = self.fitness_function(self.colony[i])
                self.tau[current_node][next_node] += 1/fitness
                self.tau[next_node][current_node] += 1/fitness

        return
    

    #evapourate the pheromone
    def evaporate_pheromone(self):
        self.tau = (1-self.rho) * self.tau
        return

    
    #displaying the results after each iterations
    def display_results(self, iter, fitness, path):
        print('iterations : '+ str(iter) + ', Shortest length : '+ str(fitness)+', path :'+ str(path)+' \n')


    def run_aco(self):
        #starting the main loop
        for i in range(self.max_iterations):

            #first creating the ants colony
            self.create_ants()

            #find the ant with best fitness Score, in our case, queen is with min distance
            queen = self.find_best_ant()

            #now update the pheromone
            self.update_pheromone()

            #evapourate the old pheromone
            self.evaporate_pheromone()

            #display the results
            self.display_results(i, queen['fitnessScore'], self.colony[queen['index']])

            #intializa again the colony for next iteration
            self.colony = self.initialize_colony()

        return




