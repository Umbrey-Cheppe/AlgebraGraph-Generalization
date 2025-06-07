import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import re
from itertools import combinations

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Algebraic Graph Visualizer", layout="centered")
st.title("🌟 Algebraic Graph Visualizer (General Graphs) 🌟")
st.markdown("""
Enter a symbolic expression to generate a graph.
- `*` for **clique/complete subgraph** (e.g., `a*b`, `1*2*3`).
- `+` for **graph union** (e.g., `(a*b)+(c*d)`).
- **Star Graph** syntax (e.g., `Center*(Leaf1+Leaf2)`).
- **Range Syntax** `Start...End` (e.g., `1...100` or `v1...v100`).

**Examples:**
- `a*b*c` (triangle)
- `1*(2+3+D*E)` (star graph with complex leaf)
- `A*(1...499)` (large star graph)
- `v1...v5 + v3*v4` (line graph with an extra edge)
""")

# --- Input Field ---
expr = st.text_input("Graph Expression:", "A*(1...100 + B*C)")

# --- Customization Options ---
st.sidebar.header("Graph Customization")
node_color = st.sidebar.color_picker("Node Color", "#ADD8E6")
edge_color = st.sidebar.color_picker("Edge Color", "#808080")
font_color = st.sidebar.color_picker("Label Color", "#333333")
node_size = st.sidebar.slider("Node Size", 100, 5000, 1500)
font_size = st.sidebar.slider("Font Size", 8, 24, 12)

# --- Helper Functions ---

def parse_nodes(node_str):
    """Converts a node string to int if numeric, else keeps as string."""
    try:
        return int(node_str)
    except ValueError:
        return node_str

def generate_clique_edges(nodes_list):
    """Generates edges for a complete graph (clique) given a list of nodes."""
    if len(nodes_list) < 2:
        return set()
    clique_edges = set()
    sorted_nodes = sorted(list(nodes_list), key=str)
    for u, v in combinations(sorted_nodes, 2):
        clique_edges.add((u, v))
    return clique_edges

# --- Main Parsing Logic ---

TOKEN_PATTERN = re.compile(r'(\.\.\.|[A-Za-z0-9_]+|\*|\+|\(|\))')

def tokenize(expression):
    return [match.group(0) for match in TOKEN_PATTERN.finditer(expression)]

def parse_expression(tokens, index):
    """Parses terms separated by '+' (lowest precedence)."""
    edges, index, nodes = parse_term(tokens, index)
    while index < len(tokens) and tokens[index] == '+':
        index += 1
        edges2, index, nodes2 = parse_term(tokens, index)
        edges.update(edges2)
        nodes.update(nodes2)
    return edges, index, nodes

def parse_term(tokens, index):
    """Parses factors separated by '*' (higher precedence)."""
    left_edges, index, left_nodes = parse_factor(tokens, index)

    if index < len(tokens) and tokens[index] == '*':
        # Check for Star Graph syntax: A*(...)
        if index + 1 < len(tokens) and tokens[index+1] == '(':
            if len(left_nodes) != 1:
                raise ValueError("Star graph center must be a single node.")
            center_node = list(left_nodes)[0]
            index += 2 # Consume '*' and '('
            
            leaf_edges, index, leaf_nodes = parse_expression(tokens, index)
            
            if index >= len(tokens) or tokens[index] != ')':
                raise ValueError("Expected ')' to close star graph leaves.")
            index += 1 # Consume ')'

            star_edges = {tuple(sorted((center_node, leaf), key=str)) for leaf in leaf_nodes if center_node != leaf}
            
            left_edges.update(leaf_edges)
            left_edges.update(star_edges)
            left_nodes.update(leaf_nodes)
            return left_edges, index, left_nodes
        
        # Otherwise, it's a standard clique
        else:
            clique_nodes = set(left_nodes)
            while index < len(tokens) and tokens[index] == '*':
                index += 1
                sub_edges, index, sub_nodes = parse_factor(tokens, index)
                if sub_edges:
                     raise ValueError("Cannot multiply complex expressions. Use '+' for union.")
                clique_nodes.update(sub_nodes)
            
            clique_edges = generate_clique_edges(clique_nodes)
            return clique_edges, index, clique_nodes
            
    return left_edges, index, left_nodes

def parse_factor(tokens, index):
    """Parses ranges, parentheses, or single nodes."""
    # Check for range syntax: A...B
    # This must be checked before single node parsing.
    is_range = (index + 2 < len(tokens) and
                re.match(r'[A-Za-z0-9_]+', tokens[index]) and
                tokens[index+1] == '...' and
                re.match(r'[A-Za-z0-9_]+', tokens[index+2]))

    if is_range:
        start_node_str = tokens[index]
        end_node_str = tokens[index+2]
        all_nodes = set()
        
        # Numeric range: "1"..."100"
        if start_node_str.isdigit() and end_node_str.isdigit():
            start, end = int(start_node_str), int(end_node_str)
            if start > end: raise ValueError(f"Range start cannot be > end.")
            for i in range(start, end + 1):
                all_nodes.add(i)
        else:
            # Alphanumeric range: "v1"..."v100"
            match_start = re.match(r'^([A-Za-z_]*)(\d+)$', start_node_str)
            match_end = re.match(r'^([A-Za-z_]*)(\d+)$', end_node_str)
            if match_start and match_end:
                p1, n1 = match_start.groups()
                p2, n2 = match_end.groups()
                if p1 != p2: raise ValueError(f"Range prefixes must match.")
                start, end = int(n1), int(n2)
                if start > end: raise ValueError(f"Range start cannot be > end.")
                for i in range(start, end + 1):
                    all_nodes.add(f"{p1}{i}")
            else:
                raise ValueError(f"Unsupported range: {start_node_str}...{end_node_str}")
        
        # A range creates a path graph (edges between consecutive nodes)
        sorted_range_nodes = sorted(list(all_nodes), key=str)
        edges = {tuple(sorted_range_nodes[i:i+2]) for i in range(len(sorted_range_nodes)-1)}
        return edges, index + 3, all_nodes

    # Check for parentheses
    if tokens[index] == '(':
        index += 1
        edges, index, nodes = parse_expression(tokens, index)
        if index >= len(tokens) or tokens[index] != ')':
            raise ValueError("Mismatched parentheses.")
        index += 1
        return edges, index, nodes
        
    # Check for single node
    if re.match(r'[A-Za-z0-9_]+', tokens[index]):
        node = parse_nodes(tokens[index])
        return set(), index + 1, {node}

    raise ValueError(f"Unexpected token: {tokens[index]}")

# --- Main Driver Function ---
def run_parser(expr_str):
    if not expr_str:
        return set(), set()
    tokens = tokenize(expr_str.replace(" ", ""))
    edges, final_index, nodes = parse_expression(tokens, 0)
    if final_index != len(tokens):
        st.warning(f"Warning: Could not parse entire string. Unparsed part: `{''.join(tokens[final_index:])}`")
    return edges, nodes

# --- Draw Button and Logic ---
if st.button("Draw Graph"):
    try:
        with st.spinner("Generating graph..."):
            edges, nodes = run_parser(expr)
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            if not G.nodes():
                st.warning("Expression resulted in an empty graph.")
            else:
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Choose layout based on graph size
                if G.number_of_nodes() > 100:
                    pos = nx.circular_layout(G)
                else:
                    pos = nx.spring_layout(G, seed=42, k=0.8) # k adjusts spacing
                    
                nx.draw(G, pos,
                        ax=ax,
                        with_labels=True,
                        node_color=node_color,
                        edge_color=edge_color,
                        font_color=font_color,
                        node_size=node_size,
                        font_size=font_size)
                st.pyplot(fig)

    except ValueError as e:
        st.error(f"❌ Syntax Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
