import networkx as nx


class CausalDAG(nx.DiGraph):
    def __init__(self):
        super().__init__()

    def remove_edges_to(self, node: str):
        self.graph.remove_edges_from(list(self.graph.in_edges(node)))

    def get_parents(self, node: str):
        return list(self.graph.predecessors(node))
