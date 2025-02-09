import networkx as nx

class FindPath():

    @staticmethod
    def solve(graph, start, goal, mode='simple'):
        '''
        returns a list of trajectories from start node to goal node

        args
        ----
        start/goal : nodes
        mode : str
            'simple': returns all simple paths between start and goal
            'best': returns all shortest paths between start and goal
        '''

        if not start in graph or not goal in graph or not nx.has_path(graph, start, goal):
            return []
        
        if mode == 'simple':
            return list(nx.all_simple_paths(graph, start, goal))

        if mode == 'best':
            return list(nx.all_shortest_paths(graph, start, goal))