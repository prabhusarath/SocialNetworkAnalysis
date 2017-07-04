"""
cluster.py
"""

# Imports you'll need.
from collections import Counter, defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import copy
import math
import pickle
import urllib.request
from TwitterAPI import TwitterAPI

consumer_key = 'gblVvoxj7RAhvaOtZTpXcYjvo'
consumer_secret = 'q92liiEEuSSmuVfaqrFTnS9dbdTKHQWjhaH4NN9Cg6y7Gdxh8A'
access_token = '2194214906-5X29UAWTDuY6qyB7yuthgTuk3JC8i22T0cl2k5e'
access_token_secret = 'jC3xSP6REtnBarBZj5lWa0bQLsG1xdiBC82gqPdvj5NBW'

def read_graph():
	fh=open("Graph.txt", 'rb')
	G=nx.read_edgelist(fh)
	fh.close()
	return G
	pass

def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def bfs(graph, root, max_depth):

    distances,paths,parents,dist,visited,next_level = {},{},{},0,[],[]
    distances[root],paths[root] = dist,1
    nodes,depth = defaultdict(),dist + 1
    current_level = list()
    current_level.append(root)
    
    for i in graph:
        children = set()
        for j in graph.edges(i):
            children.add(j[1])
        nodes[i] = children
    
    while depth <= max_depth:
        while len(current_level) > 0:
            temp = current_level.pop(0)
            for k in nodes[temp]:
                if k not in distances and k is not root:
                    distances[k] = depth
                    parents[k] = [temp]
                    paths[k] = 1
                    if k not in visited:
                        visited.append(k)
                        next_level.append(k)
                else:
                    if k in parents and distances[k] == depth:
                        parents[k].append(temp)
                        paths[k] += 1

        depth += 1
        current_level = copy.deepcopy(next_level)
    
    return distances, paths, parents
    pass

def bottom_up(root, node2distances, node2num_paths, node2parents):
    
    init,value = {},{}
    for i in node2distances.keys():
        if i is not root:
            if len(node2parents[i]) == 1:
                init[i] = 1.0
            else:
                init[i] = 1.0/len(node2parents[i])
    maps = sorted(node2distances.items(), key = lambda val: val[1], reverse = True)
    del maps[-1]
  
    for i in maps:
            initvalue = init[i[0]]
            parent = node2parents[i[0]]
            for j in parent:
                if j is not root:
                    init[j] = init[j]+initvalue
                edges=(i[0],j)
                sorted_edge = sorted((edges))
                value[tuple(sorted_edge)] = initvalue

    return value
    pass

def approximate_betweenness(graph, max_depth):
    
    betweenness_node_value = {}
    for graph_nodes in graph:
        node2distances, node2num_paths, node2parents = bfs(graph, graph_nodes, max_depth)
        credit_value = bottom_up(graph_nodes, node2distances, node2num_paths, node2parents)
        for values in credit_value:
            if values not in betweenness_node_value.keys():
                betweenness_node_value[values] = credit_value[values]
            else:
                betweenness_node_value[values] = betweenness_node_value[values] + credit_value[values]
    
    for key, val in betweenness_node_value.items():
        betweenness_node_value[key] = math.floor(val) / 2

    return betweenness_node_value
    pass

def partition_girvan_newman(graph, max_depth):
    
    graph_backup,obj,i = graph.copy(),(),0
    betweenness = approximate_betweenness(graph_backup, max_depth)
    bet_list = betweenness.items()
    Sorted_terminal = sorted(bet_list, key=lambda x:(x[0],x[1]),reverse= False)
    best_edge = sorted(Sorted_terminal, key=lambda x:x[1],reverse = True)

    while nx.number_connected_components(graph_backup) <= 2:
        graph_backup.remove_edge(*best_edge[i][0])
        i += 1

    for graph_obj in nx.connected_component_subgraphs(graph_backup):
        obj += (graph_obj,)

    return list(obj)
    pass

def get_subgraph(graph, min_degree):
    
    nodes_to_remove = ()
    for node in graph:
        if graph.degree(node) < min_degree:
            nodes_to_remove += (node,)

    graph.remove_nodes_from(list(nodes_to_remove))
    return graph
    pass

def volume(nodes, graph):
    
    volume_nodes = graph.edges(nodes)
    return len(volume_nodes)
    pass

def cut(S, T, graph):
    
    cut_set_value,first_sub,second_sub = 0,S,T                             
    
    for n in first_sub:
        for m in second_sub:
            if graph.has_edge(n,m):
                cut_set_value = cut_set_value + 1 
    
    return cut_set_value
    pass

def norm_cut(S, T, graph):
    
    vol_first = volume(S, graph)
    vol_second = volume(T, graph)
    cut_set = cut(S, T, graph)
    norm_cut_value = float(cut_set / vol_first) + float(cut_set / vol_second)
    return norm_cut_value
    pass

def score_max_depths(graph, max_depths):
    
    values = ()

    for depth_values in max_depths:
        cut_graph = partition_girvan_newman(graph, depth_values) 
        first_set = cut_graph[0]
        second_set = cut_graph[1]
        cut_values = norm_cut(first_set, second_set, graph)
        values += ((depth_values, cut_values),)

    return list(values)
    pass

def make_training_graph(graph, test_node, n):
    
    train_graph = graph.copy()
    f = train_graph.neighbors(test_node)
    rm_edges_count = 0

    while (rm_edges_count < n):
        train_graph.remove_edge(test_node, sorted(f)[rm_edges_count])
        rm_edges_count += 1

    return train_graph
    pass

def gettweets(twitter,clus,clus1,clus2):

    
    Final = []
    for i in clus.nodes():
        request = twitter.request('statuses/user_timeline', {'user_id': i, 'count':1})
        user_obj = {}
        for info in request:
            user_obj['text'] = info['text']
            user_obj['id'] = info['user']['id']
            user_obj['user_name'] = info['user']['screen_name']
            Final.append(user_obj)
         

    Final1 = []
    for j in clus1.nodes():
        request1 = twitter.request('statuses/user_timeline', {'user_id': j, 'count':1})
        user_obj1 = {}
        for info1 in request1:
            user_obj1['text'] = info1['text']
            user_obj1['id'] = info1['user']['id']
            user_obj1['user_name'] = info1['user']['screen_name'] 
            Final1.append(user_obj1)

    Final2 = []
    for k in clus2.nodes():
        request2 = twitter.request('statuses/user_timeline', {'user_id': k, 'count':1})
        user_obj2 = {}
        for info2 in request2:
            user_obj2['text'] = info2['text']
            user_obj2['id'] = info2['user']['id']
            user_obj2['user_name'] = info2['user']['screen_name']
            Final2.append(user_obj2)

    pickle.dump(Final, open('tweets.pkl', 'wb'))
    pickle.dump(Final1, open('tweets1.pkl', 'wb'))
    pickle.dump(Final2, open('tweets2.pkl', 'wb'))
    pass

def main():

    sys.stdout = open('summary_file.txt','a')
    twitter = get_twitter()
    graph = read_graph()
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    subgraph = get_subgraph(graph, 2) 
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    print('Clusters partitioned Into :')
    clusters = partition_girvan_newman(subgraph, 3)
    print('cluster 1 has %d nodes, cluster 2 has %d nodes and cluster 3 has %d nodes' %(clusters[0].order(), clusters[1].order(), clusters[2].order()))
    gettweets(twitter,clusters[0],clusters[1],clusters[2])

    Cluster_1 = pickle.load( open( "tweets.pkl", "rb" ) )
    Cluster_2 = pickle.load( open( "tweets1.pkl", "rb" ) )
    Cluster_3 = pickle.load( open( "tweets2.pkl", "rb" ) )

    print('Number of messages collected for Sub Graph :%d ' %(len(Cluster_1)+len(Cluster_2)+len(Cluster_3)))
    print('Number of communities discovered:%d'%(len(clusters)))
    Average = (len(Cluster_1)+len(Cluster_2)+len(Cluster_3))/len(clusters)
    print('Average number of users per community:%d'%(Average))

if __name__ == '__main__':
    main()
