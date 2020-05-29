#K2 Model based on preferential attachment proportional to 2nd degree connections.
#Max Falkenberg, mff113@ic.ac.uk
#MIT License. Please reference below publication if used for research purposes.
#Reference: Falkenberg et al., Identifying Time Dependence in Network Growth, Physical Review Research, 2020.
#
################################################################################
#   Instructions for use.
#
#   Generate a graph instance and assign it to a free variable:
#
#   G = ba(m,n,initial_graph)
#
#   Graph instance must be initialised with variables m, n and seed:
#
#   m: Number of edges added to network by each new node. Can take any positive integer value.
#   n: Degree of connection for preferential attachment, can be set to n=1 for BA model or n=2 for K2 model.
#   initial_graph: Initialise graph as a complete graph of size m+1 if seed = 0. Otherwise, initialise with a ring graph.
#
#   Add N nodes to graph:
#
#   G.add_nodes(N)
#   N can take any positive integer value.
#   (Note, memory issues typically arise for N > 10**6 for m=1, lower N for larger m.)
#
#   1st Degree connection list (observed network) stored in list of lists called as a.adjlist_k1()
#   2nd Degree connection list (influence network) stored in list of lists called as a.adjlist_k2()
#
#   Export x and y values for degree distribution graph:
#
#   G.degree_dist(n,plot)
#
#   n: Export data for degree distribution of 1st degree if n=1, or 2nd degree if n=2.
#   plot: If plot=True, data is plotted.
#
#   Export cumulative sum of average k2 for nodes with a given k1, see definitions in Eq.(5) and Eq.(8) in [Falkenberg et al., PRR, 2020]:
#
#   G.k2_vs_k1(plot)
#
#   plot: If plot=True, data is plotted.
#
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.optimize as opt
from operator import add
import time


class PA_graph:
    """
    Creates a preferential attachment graph object for BA model or k2 model.

    Attributes
    ----------
    n: 1 or 2
        Sphere of influence for preferential attachment. n=1 for BA model. n=2 for k2 model.
    m: int
        Number of edges to add each timestep.
    nodes: list
        List of node indices in the network.
    N: integer
        Number of nodes in the network
    adjlist_k1: list of lists
        Adjacency list for direct first degree connections in the network (corresponding to observed network)
    adjlist_k2: list of lists
        Adjacency list for first and second degree connections in the network (corresponding to influence network)
    k1: list
        First degree of node i at index i.
    k2: list
        Second degree of node i at index i.
    copy_prob: list
        Effective copying probability for preferential copying model at each timestep.

    Methods
    -------

    add_nodes(N)
        Add N nodes to the network.
    degree_dist(n=1,plot=True)
        Export degree distribution for observed network (n=1) or influence network (n=2).
    k2_vs_k1(plot=True)
        Export cumulative relative attachment kernel, \\tilde{\phi}(k,1), see Eq.(8) in [Falkenberg et al., PRR, 2020]
    """

    def __init__(self,m=1,n=1,initial_graph = 0):
        """
        Initialises preferential attachment graph.

        Parameters
        ----------
        m: int
            Number of new edges to add each time step.
        n: 1 or 2
            Sphere of influence for preferential attachment. n=1 gives BA model. n=2 gives k2 model.
        initial_graph: int
            Initial graph choice. Complete graph if initial_graph = 0. Ring graph otherwise.
        """
        self.m = m
        self.t = 0
        self.n = n
        if initial_graph == 0: #Initialise adjacency lists for complete graph
            self.nodes = list(range(m+1)) #List of nodes in network
            self.N = len(self.nodes)
            self.adjlist_k1 = [list(range(m+1)) for i in self.nodes] #Network adjacency list
            self.adjlist_k2 = [list(range(m+1)) for i in self.nodes]
            for i in range(m+1):
                self.adjlist_k1[i].pop(i)
                self.adjlist_k2[i].pop(i)
            self.targets = self.nodes * m
            self.T = len(self.targets)
        else: #Initialise adjacency lists for ring graph
            self.nodes = list(range(10*(m+1)))
            self.N = len(self.nodes)
            self.adjlist_k1 = [[i-1,i+1] for i in range(len(self.nodes))]
            self.adjlist_k1[0][0] = (10*(m+1)) - 1
            self.adjlist_k1[-1][-1] = 0
            self.adjlist_k2 = [[i-2,i-1,i+1,i+2] for i in range(len(self.nodes))]
            self.adjlist_k2[0][0] = (10*(m+1)) - 2
            self.adjlist_k2[0][1] = (10*(m+1)) - 1
            self.adjlist_k2[-1][-1] = 1
            self.adjlist_k2[-1][-2] = 0
            self.adjlist_k2[1][0] = (10*(m+1)) - 1
            self.adjlist_k2[-2][-1] = 0
            if self.n == 1:
                self.targets = self.nodes * 2
            else:
                self.targets = self.nodes * 4
            self.T = len(self.targets)
        self.k1 = [len(i) for i in self.adjlist_k1] #1st degree of node i
        self.k2 = [len(j) for j in self.adjlist_k1] #2nd degree of node i
        self.copy_prob = [] #Stores effective copying probability at each time step analogously to preferential copying model

    def degree_dist(self,n=1,plot=True):
        """
        Export degree distribution for first degree if n==1 or second degree is n==2.

        Parameters
        ----------
        n: 1 or 2
            Degree distribution for 1st degree (observed) if n=1, or 2nd degree (influence) if n=2.
        plot: boolean
            Plot degree distribution if True.

        Returns
        -------
        x: ndarray
            Array of degrees k.
        y: ndarray
            Array of probability of node with degree k.
        """
        if n == 1:
            y,x = np.histogram(self.k1,bins=int(np.max(self.k1)) - int(np.min(self.k1))) #Histogram of first degrees
        elif n == 2:
            y,x = np.histogram(self.k2,bins=int(np.max(self.k2)) - int(np.min(self.k2))) #Histogram of second degrees
        else:
            raise Exception('n must take take integer value n=1 for k1, or n=2 for k2.')
        x = x[:-1] #Remove final bin edge
        x = x[y != 0]
        y = y[y != 0]
        y = y.astype('float')
        y /= np.sum(y) #Convert histogram into probability distribution
        if plot:
            plt.plot(x,y,ls='',marker='.',label = 'k2 model simulations')
            plt.xscale('log')
            plt.yscale('log')
            plt.plot(x,(2 * self.m * (self.m + 1)) / (x*(x+1)*(x+2)),ls='--',color = 'k',label = 'BA model theory (1st degree)') #Theory for BA model
            if n==1:
                plt.xlabel(r'$k^{(1)}$',fontsize = 21)
                plt.ylabel(r'$P(k^{(1)})$',fontsize = 21)
            else:
                plt.xlabel(r'$k^{(2)}$',fontsize = 21)
                plt.ylabel(r'$P(k^{(2)})$',fontsize = 21)
            plt.tight_layout()
            plt.tick_params(labelsize='large',direction='out',right = False,top=False)
            plt.legend(loc='best')
            plt.show()
        return x,y

    def k2_vs_k1(self,plot=True):
        """
        Calculate \\tilde{\phi}(k,1) for k2 model, see Eq.(8) in [Falkenberg et al., PRR, 2020]

        Parameters
        ----------
        plot: boolean
            Plot cumulative relative attachment kernel if True.

        Returns
        -------
        x: ndarray
            Array of degrees k.
        y: ndarray
            Cumulative relative attachment kernel for nodes with degree k.
        """
        k1 = np.array([len(i) for i in self.adjlist_k1])
        k2 = np.array([len(i) for i in self.adjlist_k2])
        k1_bin = np.arange(np.min(k1),np.max(k1)+1) #Range of first degrees
        k2_bin = []
        for i in k1_bin: #Calculate average second degree for nodes with specific first degree
            if i in k1:
                k2_bin.append(np.mean(k2[k1==i]))
            else:
                k2_bin.append(None)
        k2_bin = np.array(k2_bin)
        k1_bin = k1_bin[k2_bin != None]
        k2_bin = k2_bin[k2_bin != None]
        k2_bin /= k2_bin[0]
        k2_bin *= k1_bin[0] #Rescale relative attachment kernel such that k2_bin[0] = 1
        if plot:
            plt.plot(k1_bin,k2_bin,marker='.',ls='',label = r'$N = $' + str(self.t))
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$k^{(1)}$',fontsize=24)
            plt.ylabel(r'$\tilde{\phi}(k^{(1)},1)$',fontsize=24)
            k1 = np.array([len(i) for i in self.adjlist_k1])
            k1_bin = np.arange(np.min(k1),np.max(k1))
            plt.plot(k1_bin,k1_bin,ls='--',marker='',color='k',label = r'$\tilde{\phi}(k^{(1)},1) \propto k$') #Expected scaling for BA model
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()
        return k1_bin,k2_bin

    def add_nodes(self, N):
        """
        Adds N nodes to the network, one at a time, each adding m edges.

        Parameters
        ----------
        N: int
        Number of nodes to adda to network. Require N>=0.
        """
        start = time.time() #Time counter
        for i in range(N):
            counter = 0
            T_counter = 0
            new_targets_k1 = [] #Stores target nodes in existing network
            self.nodes.append(self.N) #New node added to list of all nodes
            self.k1.append(self.m) #Degree of new node added to list of node degrees

            while counter < self.m: #Loops over m new edges
                r = np.random.randint(self.T) #Selects target according to node index
                if self.m == 1:
                    self.copy_prob.append(self.k1[self.targets[r]]/self.k2[self.targets[r]]) #Store effective copying probability
                if self.targets[r] not in new_targets_k1: #Ensures m targets are unique
                    counter += 1
                    new_targets_k1.append(self.targets[r])
            new_targets_k2 = copy.deepcopy(new_targets_k1)

            for j in new_targets_k1:
                self.k1[j] += 1 #Add one to the degree of each target node
                for k in self.adjlist_k1[j]:
                    if k not in new_targets_k2:
                        new_targets_k2 += [k] #Add target node to second degree list of new node

            self.k2.append(len(new_targets_k2))
            for j in new_targets_k2:
                self.k2[j] += 1 #Second degree increases by 1 for each target node

            if self.m != 1: #If m>1, considers nodes which are connected by a path of length two through the new node after attachment
                for j in range(self.m - 1):
                    for k in range(j+1,self.m):
                        if new_targets_k1[j] not in self.adjlist_k2[new_targets_k1[k]]:
                            if self.n != 1:
                                self.T += 2
                                self.targets += [new_targets_k1[j],new_targets_k1[k]]
                            self.adjlist_k2[new_targets_k1[j]] += [new_targets_k1[k]]
                            self.adjlist_k2[new_targets_k1[k]] += [new_targets_k1[j]]
                            self.k2[new_targets_k1[j]] += 1
                            self.k2[new_targets_k1[k]] += 1

            if self.n == 1: #Adjusts list of target nodes for BA model
                self.targets += new_targets_k1
                self.targets += [self.N] * self.m
                self.T += 2 * self.m
            else: #Adjusts list of target nodes for k2 model
                self.targets += new_targets_k2
                self.targets += [self.N] * len(new_targets_k2)
                self.T += 2 * len(new_targets_k2)

            for j in new_targets_k1:
                self.adjlist_k1[j] += [self.N]
            for j in new_targets_k2:
                self.adjlist_k2[j] += [self.N]
            self.adjlist_k1 += [new_targets_k1] #Adds a list to the first degree (observed) adjacency list of first degree nodes connected to the new node
            self.adjlist_k2 += [new_targets_k2] #Adds a list to the second degree (influence) adjacency list of second degree neigbors to new node

            self.N += 1 #Increase node counter by 1
            self.t += 1 #Increase time counter by 1
        print(time.time()-start)
