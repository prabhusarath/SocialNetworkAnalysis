"""
collect.py
"""
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
import sys
import os

consumer_key = 'gblVvoxj7RAhvaOtZTpXcYjvo'
consumer_secret = 'q92liiEEuSSmuVfaqrFTnS9dbdTKHQWjhaH4NN9Cg6y7Gdxh8A'
access_token = '2194214906-5X29UAWTDuY6qyB7yuthgTuk3JC8i22T0cl2k5e'
access_token_secret = 'jC3xSP6REtnBarBZj5lWa0bQLsG1xdiBC82gqPdvj5NBW'

# This method is done for you.
def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):
   
    with open(filename, "r") as inputfile:
        get_names = inputfile.read()
        candidates = get_names.splitlines()

    return candidates
    pass

def get_users(twitter, screen_names):
    
    request = twitter.request('users/lookup', {'screen_name': screen_names})
    user_obj = []
    for info in request:
        user_obj.append(info)

    return user_obj
    pass


def get_friends(twitter, screen_name):
    
    request = twitter.request('friends/ids', {'screen_name': screen_name, 'count':5000})
   
    twitter_ids = []
    for info in request:
        twitter_ids.append(info)
    
    twitter_ids.sort()

    return twitter_ids
    pass


def add_all_friends(twitter, users):
    
    for accounts in users:
        accounts['friends'] = get_friends(twitter, accounts['screen_name'])
    pass


def print_num_friends(users):
    
    for info in users:
        print(" %s  %d" % (info['screen_name'], len(info['friends'])))
    pass


def count_friends(users):
    
    friend_followers = Counter()
    for info in users:
        friend_followers.update(info['friends'])

    return friend_followers
    pass

def create_graph(users, friend_counts):
   
    networkx = nx.Graph()
    Graph_Edges= []

    GraphFile = open('Graph.txt', 'w')
    
    for f in friend_counts:
            for u in range(0,len(users)):
               if f in users[u]['friends']:
                    node = users[u]['id']
                    GraphFile.write("%s %s \n" % (node,f))
                    edges = (node,f)
                    Graph_Edges.append(edges)
                    networkx.add_edges_from(Graph_Edges)

    GraphFile.close()
    return networkx
    pass

def draw_network(graph, users, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).
    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.
    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """
    ###TODO
    candidates_dict = {}
    layout=nx.fruchterman_reingold_layout(graph)
    plt.figure(figsize=(40,50))
    nx.draw_networkx(graph,layout, with_labels=False)

    
    for n in graph.nodes():
        for u in range(0,len(users)):
            val = users[u]['screen_name']
            candidates_dict[users[u]['id']] = val
            nx.draw_networkx_labels(graph,layout,labels=candidates_dict,font_size=12,font_color='k',font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None)
    
    plt.savefig(filename)
    pass


def main():

    sys.stdout = open('summary_file.txt','w')
    """ Main method. You should not modify this. """
    twitter = get_twitter()
    screen_names = read_screen_names('users.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)
    friend_counts = count_friends(users)
    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    print('Number of users collected:%s' %(len(graph.nodes())))
    
if __name__ == '__main__':
    main()