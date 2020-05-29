K2 Model Implementation
=======================

K2 Model based on preferential attachment proportional to 2nd degree connections.

Max Falkenberg, mff113@ic.ac.uk

MIT License. 

Please reference below publication if used for research purposes.

Reference: Falkenberg et al., Identifying Time Dependence in Network Growth, Physical Review Research, 2020.

Instructions for use.
---------------------

Generate a graph instance and assign it to a free variable:

    G = PA_graph(m,n,initial_graph)
    
Graph instance must be initialised with variables m, n and seed:

 1. **m**: Number of edges added to network by each new node. Can take any positive integer value.
 2. **n**: Degree of connection for preferential attachment, can be set to n=1 for BA model or n=2 for K2 model.
 3. **initial_graph**: Initialise graph as a complete graph of size m+1 if seed = 0. Otherwise, initialise with a ring graph.

Add N nodes to graph:

    G.add_nodes(N)
    #N can take any positive integer value.
    #(Note, memory issues typically arise for N > 10^6 for m=1, lower N for larger m.)

1st Degree connection list (observed network) stored in list of lists called as G.adjlist_k1()

2nd Degree connection list (influence network) stored in list of lists called as G.adjlist_k2()

Export x and y values for degree distribution graph:

    G.degree_dist(n,plot)
    #n: Export data for degree distribution of 1st degree if n=1, or 2nd degree if n=2.
    #plot: If plot=True, data is plotted.
    
Export cumulative sum of average k2 for nodes with a given k1, see definitions in Eq.(5) and Eq.(8) in [Falkenberg et al., PRR, 2020]:

    G.k2_vs_k1(plot)
    #plot: If plot=True, data is plotted.

