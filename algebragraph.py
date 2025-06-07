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
- `*` for **clique/complete subgraph** (e.g., `a*b` for edge, `1*2*3` for triangle on 1,2,3). Order doesn't matter (associative, commutative).
- `+` for **graph union** (e.g., `(a*b)+(c*d)`). Follows absorption law `G_sup + G_sub = G_sup`.
- **Star Graph** syntax (e.g., `Center*(Leaf1+Leaf2)` or `1*(2+...+10)`). `Center` connects to individual nodes of leaves, and complex leaves (`D*E`) form their own cliques.

**Examples:**
- `a*b*c` (triangle on a,b,c)
- `(1*2)+(3*4)` (two separate edges)
- `1*(2+3+4)` (star graph with 1 as center, leaves 2,3,4)
- `A*(1+2+...+499)` (range-based star graph)
- `(a*b*c) + (a*b)` (simplifies to `a*b*c` due to absorption)
- `A*(B+C+D*E+F)` (A connects to B,C,D,E,F; also D connects to E)
- `1*(1+2+...+10)` (Star graph. 1 is center, connects to 1-10. Self-loops are filtered.)
""")

# --- Input Field ---
expr = st.text_input("Graph Expression:", "A*(1+2+...+499)", help="Try different formats!")

# --- Customization Options ---
st.sidebar.header("Graph Customization")
node_color = st.sidebar.color_picker("Node Color", "#ADD8E6")
edge_color = st.sidebar.color_picker("Edge Color", "#808080")
font_color = st.sidebar.color_picker("Label Color", "#333333")
node_size = st.sidebar.slider("Node Size", 100, 5000, 2000)
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
    # Sort for consistent edge representation (u,v), important for sets
    sorted_nodes = sorted(nodes_list, key=str)
    for u, v in combinations(sorted_nodes, 2):
        clique_edges.add((u, v))
    return clique_edges

def find_matching_paren(tokens, start_index):
    """Finds the matching ')' for a '(' at start_index."""
    if tokens[start_index] != '(':
        return -1
    
    balance = 1
    for i in range(start_index + 1, len(tokens)):
        if tokens[i] == '(':
            balance += 1
        elif tokens[i] == ')':
            balance -= 1
        
        if balance == 0:
            return i
    return -1 # Not found

# --- Main Parsing Logic ---

TOKEN_PATTERN = re.compile(r'([A-Za-z0-9_]+|\*|\+|\(|\)|\.\.\.)')

def tokenize(expression):
    return [match.group(0) for match in TOKEN_PATTERN.finditer(expression)]

def parse_leaves(leaf_tokens):
    """
    Parses the tokens within a star graph's parentheses.
    Handles simple ranges (e.g., 1+...+10) or general sub-expressions.
    Returns (set of edges from complex leaves, set of individual leaf nodes).
    """
    if not leaf_tokens:
        return set(), set()

    # --- Case 1: Detect a simple range pattern like `A + ... + B` ---
    # This pattern must have exactly 5 tokens: [start, '+', '...', '+', end]
    if len(leaf_tokens) == 5 and leaf_tokens[1] == '+' and leaf_tokens[2] == '...' and leaf_tokens[3] == '+':
        start_token, _, _, _, end_token = leaf_tokens
        leaf_nodes = set()
        
        # Numeric range: e.g., 1+...+10
        if start_token.isdigit() and end_token.isdigit():
            start, end = int(start_token), int(end_token)
            if start <= end:
                for i in range(start, end + 1):
                    leaf_nodes.add(i)
                return set(), leaf_nodes # No edges, just nodes
        
        # Alphanumeric range: e.g., v1+...+v10
        match_start = re.match(r'^([A-Za-z_]+)(\d+)$', start_token)
        match_end = re.match(r'^([A-Za-z_]+)(\d+)$', end_token)
        if match_start and match_end:
            prefix_start, num_start = match_start.groups()
            prefix_end, num_end = match_end.groups()
            if prefix_start == prefix_end and int(num_start) <= int(num_end):
                for i in range(int(num_start), int(num_end) + 1):
                    leaf_nodes.add(f"{prefix_start}{i}")
                return set(), leaf_nodes # No edges, just nodes

    # --- Case 2: Treat as a general sub-expression ---
    # If not a simple range, parse the content as a normal expression.
    leaf_edges, final_idx, leaf_nodes = parse_expression(leaf_tokens, 0)
    if final_idx != len(leaf_tokens):
        unparsed = "".join(leaf_tokens[final_idx:])
        raise ValueError(f"Could not fully parse leaf expression. Unparsed portion: `{unparsed}`")
    
    return leaf_edges, leaf_nodes

def parse_expression(tokens, index):
    """
    Parses a graph expression based on operator precedence (+ lowest, * highest).
    Returns (set of edges, new_index, set of all nodes in this sub-expression).
    """
    all_edges, index, all_nodes = parse_sum_term(tokens, index)
    
    # After parsing the first term, look for '+' (union operator)
    while index < len(tokens) and tokens[index] == '+':
        index += 1 # Consume '+'
        term_edges, index, term_nodes = parse_sum_term(tokens, index)
        all_edges.update(term_edges)
        all_nodes.update(term_nodes)
        
    return all_edges, index, all_nodes

def parse_sum_term(tokens, index):
    """Parses terms separated by '*' (clique or star graph operator)."""
    edges, index, nodes = parse_factor(tokens, index)

    # Check for Multiplication (Clique) or Star Graph
    if index < len(tokens) and tokens[index] == '*':
        # --- Star Graph Syntax: A*(...) ---
        if index + 1 < len(tokens) and tokens[index+1] == '(':
            center_node_token = tokens[index-1] 
            center_node = parse_nodes(center_node_token)
            
            # Find the matching parenthesis to isolate leaf tokens
            end_paren_idx = find_matching_paren(tokens, index + 1)
            if end_paren_idx == -1:
                raise ValueError("Mismatched parentheses for star graph.")
            
            leaf_tokens = tokens[index + 2 : end_paren_idx]
            leaves_edges, leaves_nodes = parse_leaves(leaf_tokens)
            
            # Union of edges/nodes from the leaves expression (e.g., from D*E inside)
            edges.update(leaves_edges)
            nodes.update(leaves_nodes)
            nodes.add(center_node) # Ensure center is in the node set

            # Add edges from center to each individual leaf node
            for leaf_node in leaves_nodes:
                if center_node != leaf_node:
                    edges.add(tuple(sorted((center_node, leaf_node), key=str)))
            
            index = end_paren_idx + 1 # Move index past the processed ')'
            
        # --- Clique Syntax: a*b*c ---
        else:
            clique_tokens = {tokens[index-1]}
            while index < len(tokens) and tokens[index] == '*':
                index += 1 # Consume '*'
                if index < len(tokens) and TOKEN_PATTERN.match(tokens[index]):
                    clique_tokens.add(tokens[index])
                    index += 1
                else:
                    raise ValueError("Expected node after '*' operator.")
            
            clique_nodes = {parse_nodes(n) for n in clique_tokens}
            edges = generate_clique_edges(list(clique_nodes))
            nodes = clique_nodes
            
    return edges, index, nodes

def parse_factor(tokens, index):
    """Parses a primary factor: a node or a parenthesized expression."""
    current_token = tokens[index]
    
    if current_token == '(':
        index += 1 # Consume '('
        edges, index, nodes = parse_expression(tokens, index)
        if index < len(tokens) and tokens[index] == ')':
            index += 1 # Consume ')'
            return edges, index, nodes
        else:
            raise ValueError("Mismatched parentheses: Expected ')'")
    elif TOKEN_PATTERN.match(current_token): # It's a node
        node = parse_nodes(current_token)
        index += 1
        return set(), index, {node} # A single node has no edges
    else:
        raise ValueError(f"Unexpected token: {current_token} at index {index}.")

# --- Main Driver Function ---

def parse_and_generate_graph(expr):
    """Top-level function to parse expression and return edges and nodes."""
    expr = expr.replace(" ", "").strip()
    if not expr:
        return set(), set()

    tokens = tokenize(expr)
    if not tokens:
        return set(), set()
        
    try:
        edges, final_index, nodes = parse_expression(tokens, 0)
        if final_index != len(tokens):
            st.error(f"⚠️ Unparsed tokens remaining: `{''.join(tokens[final_index:])}`. Check syntax.")
            return set(), set()
        return edges, nodes
    except ValueError as e:
        st.error(f"❌ Parsing Error: {e}")
        return set(), set()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}. Please check your expression.")
        return set(), set()

# --- Graph Drawing ---
st.markdown("---")
if st.button("Draw Graph"):
    edges_to_draw, all_nodes_in_expr = parse_and_generate_graph(expr)
    
    if not edges_to_draw and not all_nodes_in_expr:
        st.info("No graph generated. Check the expression for errors or try a different one.")
    else:
        # Filter out self-loops (u, u) before creating the graph
        final_edges = [(u, v) for u, v in edges_to_draw if u != v]
        
        G = nx.Graph()
        G.add_nodes_from(list(all_nodes_in_expr))
        G.add_edges_from(final_edges)
        
        if not G.nodes():
            st.info("The expression resulted in an empty graph.")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Use a more suitable layout for very large graphs
            if G.order() > 100:
                st.warning(f"Graph has {G.order()} nodes. Visualization may be slow/crowded. Using circular layout.")
                pos = nx.circular_layout(G)
            else:
                pos = nx.spring_layout(G, seed=42)
            
            nx.draw(G, pos,
                    with_labels=True,
                    node_color=node_color,
                    edge_color=edge_color,
                    font_color=font_color,
                    font_weight="bold",
                    node_size=node_size,
                    font_size=font_size,
                    ax=ax)
            
            ax.set_title(f"Graph for: `{expr}`", size=font_size + 4, color=font_color)
            st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.info("💡 This app visualizes simple graphs from algebraic expressions. Self-loops are filtered and absorption (`a*b*c + a*b = a*b*c`) is handled naturally by set unions.")
