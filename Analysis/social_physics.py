# SOCIAL NETWORK ANALYSIS PACKAGE 
# AUTHORS: Monticone Pietro, Moroni Claudio, Orsenigo Davide
# LAST MODIFIED: 08/07/2020 

# REQUIRED MODULES 
import sys, os                          # Utils
import pandas as pd                     # Data wrangling
import numpy as np                      # Data wrangling
import math as math                     # Maths
import powerlaw as pwl                  # Statistical analysis of power law distributions
import networkx as nx                   # Network Analysis
import EoN                              # Network Epidemiology 
from matplotlib import pyplot as plt    # Data visualization 
import seaborn as sns                   # Data visualization
import matplotlib.ticker as ticker      # Data visualization
import seaborn as sns                   # Data visualization
from netwulf import visualize           # Data visualization
from collections import Counter         # Utils
import sys, os, os.path                 # Utils
import itertools                        # Utils
from progressbar import ProgressBar     # Utils
from progressbar import Bar, Percentage # Utils
from operator import itemgetter         # Utils
from collections import Counter         # Utils
from collections import defaultdict     # Utils
import random as rand                   # Utils

# CONTENTS 
# 0. Basic Utilities 
# 1. Network Data Science  
# 2. Network Epidemiology


# 0. BASIC UTILITIES

### Omit Zeroes 
def omit_by(dct, predicate=lambda x: x!=0):
    return {k: v for k, v in dct.items() if predicate(v)}

### Logarithmic Binning 
def log_bin(dict,n_bins):
    
    # first we need to define the interval of dict values
    min_val=sorted(dict.values())[0]
    max_val=sorted(dict.values())[-1]
    delta=(math.log(float(max_val))-math.log(float(min_val)))/n_bins
    
    # then we create the bins, in this case the log of the bins is equally spaced (bins size increases exponentially)
    bins=np.zeros(n_bins+1,float)
    bins[0]=min_val
    for i in range(1,n_bins+1):
        bins[i]=bins[i-1]*math.exp(delta)
        
    
    # then we need to assign the dict of each node to a bin
    values_in_bin=np.zeros(n_bins+1,float)
    nodes_in_bin=np.zeros(n_bins+1,float)  # this vector is crucial to evalute how many nodes are inside each bin
        
    for i in dict:
        for j in range(1,n_bins+1):
            if j<n_bins:
                if dict[i]<bins[j]:
                    values_in_bin[j]+=dict[i]
                    nodes_in_bin[j]+=1.
                    break
            else:
                if dict[i]<=bins[j]:
                    values_in_bin[j]+=dict[i]
                    nodes_in_bin[j]+=1.
                    break
    
    
    # then we need to evalutate the average x value in each bin
    
    for i in range(1,n_bins+1):
        if nodes_in_bin[i]>0:
            values_in_bin[i]=values_in_bin[i]/nodes_in_bin[i]
            
    # finally we get the binned distribution        
            
    binned=[]
    for i in range(1,n_bins+1):
        if nodes_in_bin[i]>0:
                x=values_in_bin[i]
                y=nodes_in_bin[i]/((bins[i]-bins[i-1])*len(dict))
                binned.append([x,y])
    return binned

### Median 
def median(files):

  ite=len(files)
  out=[]
  if len(files)%2 ==0:

		  median=[]
		  median=files

		  median=sorted(median)

		  median.reverse()
		  ee=int(float(ite)/2.)

		  m_cinq=ee-1-int((ee-1)*0.5)
		  max_cinq=ee +int((ee-1)*0.5)
		  m_novc=ee-1-int((ee-1)*0.95)
		  max_novc=ee +int((ee-1)*0.95)

		  out.append([(median[ee]+median[ee-1])/2.,median[m_cinq],median[max_cinq],median[m_novc],median[max_novc]])

  else:

		  median=[]
		  median=files

		  median=sorted(median)

		  median.reverse()
		  ee=int(float(ite)/2.+0.5)
		  m_cinq=ee-1-int((ee-1)*0.5)
		  max_cinq=ee-1+int((ee-1)*0.5)
		  m_novc=ee-1-int((ee-1)*0.95)
		  max_novc=ee-1+int((ee-1)*0.95)
		  
		  out.append([median[ee-1],median[m_cinq],median[max_cinq],median[m_novc],median[max_novc]])

  return out

# 1. NETWORK DATA SCIENCE 

### Data Wrangling 
def rtweet_to_networkx(fo, so, all = False, save = None):
    """
    Pipeline from rtweet edge-lists to networkx graphs.
    """
    # Read csv datasets 
    fo_friends_csv = pd.read_csv(fo)
    so_edges_csv = pd.read_csv(so)
    
    try:
        fo_friends = fo_friends_csv["Target"].tolist()
    except Exception as err:
        print("Error! Expected column names are 'Source' and 'Target' for all csv.")
        raise err
    so_edges = list(zip(so_edges_csv["Source"].tolist(), so_edges_csv["Target"].tolist())) 
    
    if all == True:
        edge_list = [tup for tup in so_edges]
    else:    
        edge_list = [ tup for tup in so_edges if tup[1] in fo_friends ]
        #edge_list = [ (row["Source"],row["Target"]) for _,row in so_edges_csv.iterrows() if row["Target"] in fo_friends ] # line to be removed if the function works on new data  
    
    # Create directed graph
    G = nx.DiGraph()
    G.add_nodes_from(fo_friends) # add nodes
    G.add_edges_from(edge_list) # add edges
    
    if save is not None:
        nx.write_graphml(G, save)
        
    return G


### Degree Distribution

#### Get Distribution
def get_degree_distribution(G, which):
    """
    """
    if which == "degree":
        degree_view = dict(G.degree())
    elif which == "in_degree":
        try:
            degree_view = dict(G.in_degree())
        except:
            print("Error, check the graph! Is it directed?")
    elif which == "out_degree":
        try: 
            degree_view = dict(G.out_degree())
        except:
            print("error, check the graph! Is it directed?")
    else:
        print("Invalid 'which' argument: it must be one of 'degree', 'in_degree' or 'out_degree'")
        return

    mean = np.mean(np.array(list(degree_view.values())))
    var  = np.var(np.array(list(degree_view.values())))
    
    return (degree_view, mean, var)

##### Visualization
def plot_degree_distribution(degree_distribution, hist = True, kde = True, log_binning = None, color = 'darkblue', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 3}, title = "", log = False, dimensions = (15,8), display_stats = None):
    """
    """
    plt.rcParams['figure.figsize'] = dimensions
    if log_binning is not None:
        degree_distribution_nonzero = omit_by(dct = degree_distribution)
        log_distrib = log_bin(degree_distribution_nonzero,log_binning)
        bins = [0]+[lim[0] for lim in log_distrib]
    else:
        bins = None
    ax = sns.distplot(list(degree_distribution.values()), hist = hist, kde = kde, bins = bins , color = color, hist_kws = hist_kws , kde_kws =  kde_kws)
    ax.set_title(title, fontsize = 16)
    ax.set_xlabel("$k$", fontsize = 14)
    ax.set_ylabel("$P(k)$", fontsize = 14)
    ax.tick_params(labelsize  = 11)
    if log:
        ax.set_yscale("log")
        ax.set_xscale("log")
#     if display_stats is not None:
#         mean  = np.var(np.array(list(degree_distribution.values())))
#         var  = np.mean(np.array(list(degree_distribution.values())))
        #plt.gcf().text(0.9, 0.8, f"mean = {mean} \n var = {var}", fontsize=14) #, xy=(0.005, 700), xytext=(0.005, 700)
    plt.show()

### Centrality Metrics 

##### Get Centralities
def get_centrality(G, type_centrality):
    
    if type_centrality=="degree":
        centrality=[]
        for i in G.nodes():
            centrality.append([G.degree(i),i])
        centrality=sorted(centrality,reverse=True)
        return centrality
        
    elif type_centrality=="closeness":
        l=nx.closeness_centrality(G)
        centrality=[]
        for i in G.nodes():
            centrality.append([l[i],i])
        centrality=sorted(centrality,reverse=True)
        return centrality
    
    elif type_centrality=="betweenness":
        l=nx.betweenness_centrality(G)
        centrality=[]
        for i in G.nodes():
            centrality.append([l[i],i])
        centrality=sorted(centrality,reverse=True)
        return centrality
    
    elif type_centrality=="eigenvector":
        l=nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
        centrality=[]
        for i in G.nodes():
            centrality.append([l[i],i])
        centrality=sorted(centrality,reverse=True)
        return centrality
    
    elif type_centrality=="katz":
        l=nx.katz_centrality(G, alpha=0.001, beta=1.0, max_iter=1000, tol=1e-06)
        centrality=[]
        for i in G.nodes():
            centrality.append([l[i],i])
        centrality=sorted(centrality,reverse=True)
        return centrality
    
    elif type_centrality=="pagerank":
        l=nx.pagerank(G,0.85)
        centrality=[]
        for i in G.nodes():
            centrality.append([l[i],i])
        centrality=sorted(centrality,reverse=True)
        return centrality
    
    elif type_centrality=="random":
        
        centrality=[]
        for i in G.nodes():
            centrality.append([i,i])
        rand.shuffle(centrality)
        return centrality
    else:
        return 0

##### Plot Centrality Distributions
def plot_centrality_distribution(G, list_centrality, color, n_bins):
    
    dict_centrality={}
    for i in list_centrality:
        if i[0]>0.:
            dict_centrality[i[1]]=i[0]
       
    centrality_binned=log_bin(dict_centrality,n_bins)

    # we then plot their binned distribution
    x_centrality=[]
    y_centrality=[]
    for i in centrality_binned:
        x_centrality.append(i[0])
        y_centrality.append(i[1])

    plt.plot(x_centrality,y_centrality, color=color,linewidth=1.1, marker="o",alpha=0.55) 
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$x$', fontsize = 15)
    plt.ylabel('$P(x)$', fontsize = 15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

### POWER LAW ANALYSIS 
def power_law_plot(graph, log = True,linear_binning = False, bins = 90, draw= True,x_min = None):
    degree = list(dict(graph.degree()).values())
    
    #powerlaw does not work if a bin is empty
    #sum([1 if x == 0 else 0 for x in list(degree)])
    corrected_degree = [x for x in degree if x != 0 ]
    if x_min is not None:
        corrected_degree = [x  for x in corrected_degree if x>x_min]
    # fit powerlaw exponent and return distribution
    pwl_distri=pwl.pdf(corrected_degree, bins = bins)
    
    if draw:
        degree_distribution = Counter(degree)

        # Degree distribution
        x=[]
        y=[]
        for i in sorted(degree_distribution):   
            x.append(i)
            y.append(degree_distribution[i]/len(graph)) 
        #plot our distributon compared to powerlaw
        
        #plt.figure(figsize=(10,7))
        plt.yscale('log')
        plt.xscale('log')
        plt.plot(x,y,'ro')

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.xlabel('$k$', fontsize=16)
        plt.ylabel('$P(k)$', fontsize=16)

        if linear_binning:
            pwl.plot_pdf(corrected_degree, linear_bins=True, color='black', linewidth=2)
        else:
            pwl.plot_pdf(corrected_degree, color='black', linewidth=2)
    
    return pwl_distri

### COMMUNITY DETECTION 

##### Modularity Evaluation 
def modularity(partition):
    return nx.community.quality.modularity(G, partition)

##### Partition Mapping
def create_partition_map(partition):
    partition_map = {}
    for idx, cluster_nodes in enumerate(partition):
        for node in cluster_nodes:
            partition_map[node] = idx
    return partition_map


# 2. EPIDEMIC DYNAMICS 

## 2.1 EPIDEMIC DYNAMICS ON STATIC NETWORKS 

#### Multi-Run Simulation 

def network_SIR_multirun_simulation(G, nrun, lambd, mu):
    I_dict = defaultdict(list)   # Define the time series dictionary for I 
    Irun = []                    # Define the multi-run list of lists for I 
    
    for run in range(0,nrun):
        # Create a dictionary of nodal infection/disease states s.t. S=0, I=1, R=-1
        G.disease_status = {} 
    
        # Create a list of infected notes 
        I_nodes = []
    
        # Choose a seed
        node_list = []
        deg = dict(G.degree())
        for i in sorted(deg.items(), key = itemgetter(1)):
            node_list.append(i[0])
        seed = node_list[-1]
    
        # Initialize the network
        I_nodes.append(seed)
    
        for n in G.nodes():
            if n in I_nodes:
                # Infected
                G.disease_status[n] = 1 
            else:
                # Susceptible
                G.disease_status[n] = 0 
            
        t = 0                          # Initialize the clock
    
        I_list = []                    # Define the single-run list for I 
        I_list.append(len(I_nodes))    # Initialize the single-run list for I
        I_dict[t].append(I_nodes)      # Initialize the time series dictionary for I
    
        # Implement the dynamical model 
        while len(I_nodes)>0:
    
            # Transmission dynamics (S -> I)
            for i in I_nodes:                           # For any infected node 
                for j in G.neighbors(i):                # For any of its neighbours 
                    if G.disease_status[j] == 0:        # If it's S, 
                        p = np.random.random()          # then infect it with probability lambda
                        if p < lambd:
                            G.disease_status[j] = 1
                
            # Recovery dynamics (I -> R)
            for k in I_nodes:                           # For any infected node 
                p = np.random.random()                  # It recovers with probability mu
                if p < mu:
                    G.disease_status[k] = -1
    
            # Update infected nodes
            I_nodes = []
            for node in G.nodes():
                if G.disease_status[node] == 1:
                    I_nodes.append(node)
        
            t += 1
            # Register the prevalence for each time step
            #I_graph.append(len(infected_nodes))
            I_list.append(len(I_nodes))
            I_dict[t].append(len(I_nodes))
        
        Irun.append(I_list)
    return Irun 

def network_SIR_finalsize_lambda_sensitivity(G, mu, rho, lambda_min, lambda_max, nruns):
    #average_degree = 2 * G.number_of_edges() / G.number_of_nodes()
    #lc = mu / average_degree
 
    final_size = defaultdict(list) # normalized attack rate
    
    for lambd in np.geomspace(lambda_min, lambda_max, nruns):
    
        for run in range(0, nruns):
            t, S, I, R = EoN.fast_SIR(G, tau=lambd, gamma=mu, rho=rho)
        
            final_size[lambd].append(R[-1]/G.number_of_nodes())
    
    return pd.DataFrame.from_dict(final_size)

#### Visualization 

def plot_ensemble(runs):
    # Plot the ensemble of trajectories
    #plt.figure(figsize = (10,7))
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)
    plt.xlabel('Time', fontsize = 16)
    plt.ylabel('Prevalence', fontsize = 16)

    for run in runs: 
        plt.plot(range(0,len(run)),run)

def boxplot_finalsize_lambda_sensitivity(G, mu, data, ymin, ymax, xlim):
    average_degree = 2 * G.number_of_edges() / G.number_of_nodes()
    lc = mu / average_degree
    
    data.boxplot(positions=np.array(data.columns), 
                 widths=np.array(data.columns)/3)
    
    plt.vlines(x=lc, ymin=ymin, ymax=ymax)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xlim) 
    plt.ylim(0.045, 1.1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Final Epidemic Size ($R_f / |V_G|$)', fontsize=18)
    plt.xlabel('Transmission Rate per Contact ($\lambda$)', fontsize=18)
    plt.show()

def random_walk(G,source,stop,t,nt,visited):
    nt[t]=source # at time t the walker visits node "source"
    visited[source]+=1 # the node has been visited another time
    # the process ends after reaching a certain threshold
    if t<stop:
        # explore the neighbors
        neighbors=list(G.neighbors(source))
        # select one randomly
        target=neighbors[random.randint(0,len(neighbors)-1)]
        # move there using the same function
        random_walk(G,target,stop,t+1,nt,visited)
    else:
        return 0

def get_coverage(nt):
    coverage=np.zeros(nt.size,int)
    v=set()
    for i in range(nt.size):
        v.add(nt[i])
        coverage[i]=len(v) # at each time the coverage is the set up to t 
    return coverage

# let us see another way to do it, without recursion
def random_walk2(G,source,stop,nt,visited):
    t=0
    while t<stop:
        visited[source]+=1
        nt[t]=source
        neighbors=list(G.neighbors(source))
        target=neighbors[random.randint(0,len(neighbors)-1)]
        source=target
        t+=1
        

def SIR_hm(beta,mu,N,status):
    p_1=0.
    delta_1=0.
    delta_2=0.
    
    p_1=beta*float(status[1])/N  ## P(S-->I) 
    p_2=mu                      ## P(I--->R)       

    if p_1>0.:
        # binomial extraction to identify the number of infected people going to I given p_1
        delta_1=np.random.binomial(status[0], p_1)
        
    if status[2]!=0:
        delta_2=np.random.binomial(status[1],p_2)

    # update the compartments
    status[0]-= delta_1

    status[1]+= delta_1
    status[1]-= delta_2
    
    status[2]+= delta_2 # R is id=2
    
    return 0

def ini_subpop(G,average_V,s,x):
    # let assign V people to each subpopulation
    N = G.number_of_nodes()
    V=np.zeros(N,int)
    for i in G.nodes():
        V[i]=average_V

    # inside each subpopulation people are divided in compartments S,I,R
    # let's create a dictionary with the compartments
    compartments={}
    compartments[0]='S'
    compartments[1]='I'
    compartments[2]='R'
    # that that this could be read from file
    # then let's create a dictionary for each subpop that tell us how many people in each compartment are there


    status_subpop={}
    for i in G.nodes():
        status_subpop.setdefault(i,np.zeros(3,int))
        for j in compartments:
            if compartments[j]=='S': # initially they are all S
                status_subpop[i][j]=V[i]
            else:
                status_subpop[i][j]=0

    # now we need to select the subpopulation that are initially seeded
    # let's select a random fraction of s as initially seeded


    n_of_infected=int(s*N)
    # we get the list of nodes and shuffle it
    list_subpop=[]
    for i in range(N):
        list_subpop.append(i)
    random.shuffle(list_subpop)

    # now let's add a number of infected people in the selected subpopulation
    for i in range(n_of_infected):
        seed_subpop=list_subpop[i]
        # for each initial seed we need to change the subpop distribution
        for j in compartments:
            if compartments[j]=='S': # we remove 10 people
                status_subpop[seed_subpop][j]-=x
            if compartments[j]=='I': # we make them infected!
                status_subpop[seed_subpop][j]+=x
            
    return status_subpop

# what about using a different d_kk'? 
# remember from the lecture a more realistic one is d_kk' ~ (kk')^(theta)
# let's create the weights first
def get_p_traveling(theta,G):
    dij={} # this a dictionary we use to compute the rate of travels from any pair ij
    for i in G.nodes():
        l=G.neighbors(i) # we compute the traveling rate to each neighbor
        summ=0.
        dij.setdefault(i,{})
        for j in l:
            # this the numerator of the dij
            w= (G.degree(i)*G.degree(j))**theta
            dij[i].setdefault(j,w)
            summ+=w  # this is the normalization factor: \sum_{j}wij

        for j in dij[i]:
            dij[i][j]=dij[i][j]/summ
    return dij

def random_walk4(G,stop,dij,p,W):
    t=0
    N=G.number_of_nodes()
    while t<stop:
        # temporary vector where to store who moves where at eact t
        temp=np.zeros(N,int)
        temp2=np.zeros(N,int)
        for source in G.nodes():
            # for each node we let diffuse the walkers out of it
            neighbors=list(G.neighbors(source))
            # we need to get the probabilities
            # now p is not 1!!
            prob=[]
            for j in neighbors:
                prob.append(p*dij[source][j])  # with prob p they travel to j with prob p*d_ij
            # with prob 1-p they stay
            prob.append(1.-p)
            output=np.random.multinomial(W[source], prob, size=1)
            # after calling the multinomial we know how to divide W(i)
            id=0
            for j in range(len(output[0])-1):
                temp[neighbors[id]]+=output[0][j] # these are the traveling in
                id+=1 
            temp2[source]=output[0][-1] # these are those staying in source
        # after the loop across all nodes
        # we update the values of W
        for i in G.nodes():
            W[i]=temp[i]+temp2[i]  #since p!=0, this is given by those than arrive plus those that stayed 
            
        t+=1
    
# let's convert all of this into a function

def metapop(t_max,N,compartments,status_subpop,G,beta,mu,p,theta,dij):
    
    diseased={} # for each t let's save the number of diseased subpop
    prevalence={} # for each t let's save the number of infected people
    for t in range(t_max):
        # at each iteration the first thing is to make people travel
        # we make each compartment travel separately
        for j in compartments:
            people_traveling=np.zeros(N,int) # this is the vector of people traveling in comp j
            for k in G.nodes():
                people_traveling[k]+=status_subpop[k][j]

            # we then call the random walk function for 1 time step
            random_walk4(G,1,dij,p,people_traveling)
            # we update the populations given the travels
            for k in G.nodes():
                status_subpop[k][j]=people_traveling[k]

        # after the traveling we can call the SIR model in each subpopulation

        for k in G.nodes():
            tot_pop=0 # we need to know how many people are living in each subpop
            inf=0     # also we run the SIR just if there are infected
            for j in compartments:
                tot_pop+=status_subpop[k][j]
                if j==1:
                    inf=status_subpop[k][j]
            if inf>0:
                SIR_hm(beta,mu,tot_pop,status_subpop[k]) # note how we are passing status_subpop[k] to the function
        #let's see how many diseased subpopulation we have
        disease_sub_pop=0
        tot_inf=0.
        for k in G.nodes():
            if status_subpop[k][1]>0:
                    disease_sub_pop+=1
                    tot_inf+=status_subpop[k][1]
        diseased[t]=disease_sub_pop
        prevalence[t]=tot_inf
        
    return diseased, prevalence

# 4.2 EPIDEMIC DYNAMICS ON TEMPORAL NETWORKS 