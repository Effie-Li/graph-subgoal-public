import argparse
import networkx as nx

def generate_random_graph(
        N, 
        out_dir, 
        n_node, 
        edge_prob,
        sparse_min,
        sparse_max
):

    # part1: generate
    graphs = []
    while len(graphs) < N:
        g = nx.erdos_renyi_graph(n_node, p=edge_prob)
        r = len(g.edges) / n_node
        if nx.is_connected(g) and r >= sparse_min and r <= sparse_max:
            graphs.append(g)

    # part2: save g6 format
    g6strs = b''
    for g in graphs:
        g6strs += nx.to_graph6_bytes(g, header=False)
    fname = f'{out_dir}/random{N}_nodecount={n_node}_edgeprob={edge_prob}.g6'
    with open(fname, 'wb') as f:
        f.write(g6strs)
        f.close()
    
    print(f'successfully generated {N} graphs and saved to {fname}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--n_node', type=int)
    parser.add_argument('--edge_prob', type=float, default=0.25)
    parser.add_argument('--sparse_min', type=float, default=1.1)
    parser.add_argument('--sparse_max', type=float, default=1.6)
    args = parser.parse_args()

    generate_random_graph(
        N=args.N, 
        out_dir=args.out_dir, 
        n_node=args.n_node, 
        edge_prob=args.edge_prob, 
        sparse_min=args.sparse_min,
        sparse_max=args.sparse_max,
    )