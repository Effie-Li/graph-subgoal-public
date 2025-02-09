import yaml
import argparse
import numpy as np
import dill as pkl
import networkx as nx
import itertools as it

from data.oracle import FindPath

def generate_dataset(config):

    graphs = nx.read_graph6(config['dataset']['raw_graphs_file'])
    num_isomorphs = config['dataset']['num_isomorphs']

    dataset = {
        'graph_num': [],
        'isomorph_num': [],
        'node_label': [],
        'edge_list': [],
        'start_goal': [],
        'shortest_path': [],
        'path_length': [],
        'edge_order_on_path': []
    }

    for i_graph, g in enumerate(graphs):

        edge_list = list(g.edges)

        # construct isomorph graph b/c the graphs might be coded in a sysmatic way (0 is more likely to connect with 123)
        # repeat X times to generate trajectories in X isomorphic graphs
        for isorm in range(num_isomorphs):
            
            node_labels = np.unique(edge_list) # unique node labels
            np.random.shuffle(node_labels)
            remapped_edge_list = [(node_labels[n1], node_labels[n2]) for (n1, n2) in edge_list]
            g = nx.from_edgelist(remapped_edge_list)

            # generate shortest paths for all multi-step problems
            for start, goal in it.permutations(node_labels, 2):
                shortest_paths = FindPath().solve(g, start, goal, mode='best')

                if len(shortest_paths[0]) == 2: continue # skip single-step problems

                # populate the dataset
                dataset['graph_num'].append(i_graph)
                dataset['isomorph_num'].append(isorm)
                dataset['node_label'].append(node_labels)
                dataset['edge_list'].append(remapped_edge_list)
                dataset['start_goal'].append((start, goal))
                dataset['shortest_path'].append(shortest_paths)
                dataset['path_length'].append(len(shortest_paths[0]))

    pkl.dump(dataset, open(config['dataset']['fname'], 'wb'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-config', default='')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    generate_dataset(config=config)