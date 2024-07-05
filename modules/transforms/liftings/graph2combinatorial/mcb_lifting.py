import torch_geometric
import networkx as nx
from modules.transforms.liftings.graph2combinatorial.base import Graph2CombinatorialLifting

from torch_geometric.utils import to_networkx
from toponetx.classes.combinatorial_complex import CombinatorialComplex



class MCBLifting(Graph2CombinatorialLifting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "mcb"

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:

        graph: nx.Graph = to_networkx(data)

        ccc = CombinatorialComplex(graph, graph_based=True)

        mcb = nx.minimum_cycle_basis(graph)

        for cycle in mcb:
            ccc.add_cell(frozenset(cycle))

        return super().lift_topology(data)