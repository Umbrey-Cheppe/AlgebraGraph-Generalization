import re
import networkx as nx

def parse_graph_expression(expr: str) -> nx.Graph:
    expr = expr.replace(" ", "")
    G = nx.Graph()

    match = re.match(r"(\d+)\*(.+)", expr)
    if not match:
        raise ValueError("Expression must be in the form like 1*(2+3+4) or 1*(2+3+...+10)")

    center = match.group(1)
    leaf_expr = match.group(2)

    if "..." in leaf_expr:
        parts = leaf_expr.split("+")
        start = int(parts[0])
        end = int(parts[-1])
        leaves = [str(i) for i in range(start, end + 1)]
    else:
        leaves = leaf_expr.split("+")

    G.add_node(center)
    for leaf in leaves:
        G.add_node(leaf)
        G.add_edge(center, leaf)

    return G
  
