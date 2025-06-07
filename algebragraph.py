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
- **Range Syntax** `Start...End` (e.g., `1...100`) or `PrefixN...PrefixM` (e.g., `v1...v100`).

**Examples:**
- `a*b*c` (triangle on a,b,c)
- `(1*2)+(3*4)` (two separate edges)
- `1*(2+3+4)` (star graph with 1 as center, leaves 2,3,4)
- `A*(B+C+...+E)` (range-based star graph)
- `A*(1...499)` (star graph with center A and 499 leaves)
- `(a*b*c) + (a*b)` (simplifies to `a*b*c` due to absorption)
- `A*(B+C+D*E+F)` (A connects to B,C,D,E,F; also D connects to E)
- `1*(1+2+...+10)` (Star graph. 1 is center, connects to 1-10. Self-loops are filtered.)
""")

# --- Input Field ---
expr = st.text_input("Graph Expression:", "A*(1...499)", help="Try different formats!")

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
    # Sort for consistent edge representation (u,v) regardless of input order
    sorted_nodes = sorted(nodes_list, key=str)
    for u, v in combinations(sorted_nodes, 2):
        clique_edges.add((u, v)) # Edges are inherently symmetric in undirected graph
    return clique_edges

# --- Main Parsing Logic ---

# Tokenization - ensure '...' is treated as a single token
TOKEN_PATTERN = re.compile(r'(\.\.\.|[A-Za-z0-9_]+|\*|\+|\(|\))')

def tokenize(expression):
    return [match.group(0) for match in TOKEN_PATTERN.finditer(expression)]

# Recursive Descent Parser with Operator Precedence and Star Graph Handling

def parse_expression(tokens, index):
    """
    Parses a graph expression based on operator precedence (+ lowest, * highest).
    Handles parentheses and the special Star Graph syntax.
    Returns (set of edges, new_index, set of all nodes in this sub-expression)
    """
    edges_from_term1, index, nodes_in_term1 = parse_sum_term(tokens, index)

    # After parsing the first sum term, look for '+'
    while index < len(tokens) and tokens[index] == '+':
        index += 1 # Consume '+'
        edges_from_term2, index, nodes_in_term2 = parse_sum_term(tokens, index)
        edges_from_term1.update(edges_from_term2) # Union of edges
        nodes_in_term1.update(nodes_in_term2) # Union of nodes

    return edges_from_term1, index, nodes_in_term1

def parse_sum_term(tokens, index):
    """Parses terms separated by '*' or the star graph operator."""
    edges_from_factor1, index, nodes_in_factor1 = parse_factor(tokens, index)

    # Check for Star Graph or Multiplication
    if index < len(tokens) and tokens[index] == '*':
        # Check for Star Graph: Center*(...)
        if index + 1 < len(tokens) and tokens[index+1] == '(':
            center_node = list(nodes_in_factor1)[0] if nodes_in_factor1 else None
            if center_node is None:
                raise ValueError("Star graph must have a center node.")

            # Consume '*' and '('
            index += 2

            # Parse the leaves list within the parentheses
            leaves_edges, index, leaves_nodes = parse_leaf_list(tokens, index)

            # Combine current edges (if any) with leaves_edges (from complex leaves like D*E)
            edges_from_factor1.update(leaves_edges)
            nodes_in_factor1.update(leaves_nodes) # Union of all nodes in leaves

            # Add edges from center to each individual node in the leaves_nodes
            for leaf_node in leaves_nodes:
                if center_node != leaf_node:
                    edges_from_factor1.add(tuple(sorted((center_node, leaf_node), key=str)))

            # Ensure center node is included in the set of nodes
            nodes_in_factor1.add(center_node)

            # Expect ')'
            if index < len(tokens) and tokens[index] == ')':
                index += 1
            else:
                raise ValueError("Expected ')' after star graph leaves.")

        else: # Standard multiplication (clique formation)
            # Collect all nodes connected by '*'
            all_clique_nodes = set(nodes_in_factor1)

            while index < len(tokens) and tokens[index] == '*':
                index += 1 # Consume '*'
                # The next factor could be a single node or a parenthesized expression
                sub_edges, index, sub_nodes = parse_factor(tokens, index)

                # For a simple clique like a*b*c, sub_edges will be empty.
                # If we had a*(b+c), this logic path wouldn't be taken (see star graph above).
                # This handles (a*b)*(c*d) by taking the union of nodes.
                if sub_edges:
                    raise ValueError("Cannot multiply complex sub-expressions directly. Use '+' for union.")
                all_clique_nodes.update(sub_nodes)

            # Generate clique edges for all collected nodes
            edges_from_factor1 = generate_clique_edges(list(all_clique_nodes))
            nodes_in_factor1 = all_clique_nodes

    return edges_from_factor1, index, nodes_in_factor1

def parse_factor(tokens, index):
    """Parses a primary factor: a node, or a parenthesized expression."""
    current_token = tokens[index]

    if current_token == '(':
        index += 1 # Consume '('
        edges, index, nodes = parse_expression(tokens, index)
        if index < len(tokens) and tokens[index] == ')':
            index += 1 # Consume ')'
            return edges, index, nodes
        else:
            raise ValueError("Mismatched parentheses: Expected ')'")
    elif re.match(r'[A-Za-z0-9_]+', current_token): # It's a node
        node = parse_nodes(current_token)
        index += 1
        return set(), index, {node} # A single node forms no edges on its own
    else:
        raise ValueError(f"Unexpected token: {current_token} at index {index}. Expected node or '('.")

def parse_leaf_list(tokens, current_index):
    """
    --- REWRITTEN FOR EFFICIENCY AND CORRECTNESS ---
    Parses the plus-separated list of leaves within a star graph's parentheses.
    Handles ranges (e.g., 2...10) and complex leaf terms (e.g., D*E) efficiently.
    Returns (set of edges from complex leaves, new_index, set of individual leaf nodes)
    """
    all_leaves_edges = set()
    all_individual_leaf_nodes = set()

    if current_index >= len(tokens) or tokens[current_index] == ')':
        return set(), current_index, set() # Empty leaf list

    while current_index < len(tokens) and tokens[current_index] != ')':
        # Parse the start term of a potential range or a standalone leaf
        start_term_edges, index_after_start, start_term_nodes = parse_sum_term(tokens, current_index)

        # Check for range operator '...'
        if index_after_start < len(tokens) and tokens[index_after_start] == '...':
            # This is a range expression like `1...100` or `v1...v100`
            current_index = index_after_start + 1 # Consume '...'

            # The end of the range must be a simple term
            end_term_edges, index_after_end, end_term_nodes = parse_sum_term(tokens, current_index)

            # Validate range syntax
            if not (len(start_term_nodes) == 1 and len(end_term_nodes) == 1 and not start_term_edges and not end_term_edges):
                raise ValueError(f"Invalid range expression. Must be between two single nodes (e.g., '1...100').")

            start_node_str = str(list(start_term_nodes)[0])
            end_node_str = str(list(end_term_nodes)[0])

            # Try to parse as numeric range (e.g., "1" to "100")
            match_start_num = re.match(r'^\d+$', start_node_str)
            match_end_num = re.match(r'^\d+$', end_node_str)
            if match_start_num and match_end_num:
                start_num, end_num = int(start_node_str), int(end_node_str)
                if start_num > end_num: raise ValueError(f"Range start '{start_num}' cannot be greater than end '{end_num}'.")
                for i in range(start_num, end_num + 1):
                    all_individual_leaf_nodes.add(i)
            else:
                # Try to parse as alphanumeric range (e.g., "v1" to "v100")
                match_start = re.match(r'^([A-Za-z_]+)(\d+)$', start_node_str)
                match_end = re.match(r'^([A-Za-z_]+)(\d+)$', end_node_str)
                if match_start and match_end:
                    start_prefix, start_num_str = match_start.groups()
                    end_prefix, end_num_str = match_end.groups()
                    if start_prefix != end_prefix:
                        raise ValueError(f"Range prefixes do not match: '{start_prefix}' vs '{end_prefix}'.")
                    start_num, end_num = int(start_num_str), int(end_num_str)
                    if start_num > end_num: raise ValueError(f"Range start '{start_num}' cannot be greater than end '{end_num}'.")
                    for i in range(start_num, end_num + 1):
                        all_individual_leaf_nodes.add(f"{start_prefix}{i}")
                else:
                    raise ValueError(f"Unsupported range format between '{start_node_str}' and '{end_node_str}'.")

            current_index = index_after_end
        else:
            # This is a regular leaf term (e.g., 'A', or 'B*C', or '(B+C)')
            all_leaves_edges.update(start_term_edges)
            all_individual_leaf_nodes.update(start_term_nodes)
            current_index = index_after_start

        # After processing a term, consume the '+' separator if it exists
        if current_index < len(tokens) and tokens[current_index] == '+':
            current_index += 1
        # If there's another token that isn't ')' or '+', it's a syntax error
        elif current_index < len(tokens) and tokens[current_index] != ')':
            raise ValueError(f"Expected '+' or ')' after leaf term, but found '{tokens[current_index]}'.")

    return all_leaves_edges, current_index, all_individual_leaf_nodes

# --- Main Driver Function for Parsing ---

def parse_and_simplify_graph_expression(expr):
    """
    Tokenizes and parses the expression. Absorption is handled implicitly
    by the set-based union of edges.
    """
    expr = expr.replace(" ", "").strip()
    if not expr:
        return set(), set()

    tokens = tokenize(expr)
    if not tokens:
        return set(), set()

    try:
        raw_edges, final_index, all_nodes_in_expr = parse_expression(tokens, 0)

        if final_index != len(tokens):
            st.warning(f"⚠️ Unparsed tokens remaining: `{''.join(tokens[final_index:])}`. The graph might be incomplete.")

        return raw_edges, all_nodes_in_expr

    except ValueError as e:
        st.error(f"❌ Parsing Error: {e}")
        return set(), set()
    except Exception as e:
        st.error(f"💥 An unexpected error occurred: {e}. Please check your expression.")
        return set(), set()

# --- Graph Drawing ---
st.markdown("---")
if st.button("Draw Graph"):
    with st.spinner('Parsing expression and building graph...'):
        edges_to_draw, all_nodes_in_expr = parse_and_simplify_graph_expression(expr)

    if not edges_to_draw and not all_nodes_in_expr:
        st.error("❌ Failed to generate a graph. Please check your expression for errors.")
    else:
        # --- Filter out self-loops (u, u) ---
        final_edges_for_nx = [(u, v) for u, v in edges_to_draw if u != v]

        # --- Create Graph ---
        G = nx.Graph()
        G.add_nodes_from(list(all_nodes_in_expr))
        G.add_edges_from(final_edges_for_nx)

        # --- Visualization ---
        if not G.nodes():
            st.info("The expression resulted in an empty graph.")
        else:
            fig, ax = plt.subplots(figsize=(12, 10))

            # Use a more suitable layout for large or structured graphs
            pos = None
            if G.order() > 75:
                st.warning(f"Graph has {G.order()} nodes. Using a circular layout for clarity.")
                pos = nx.circular_layout(G)
            else:
                try:
                    # Spring layout is good for general-purpose graphs
                    pos = nx.spring_layout(G, seed=42, iterations=50)
                except nx.NetworkXError: # Handle disconnected graphs
                    pos = nx.kamada_kawai_layout(G)

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
st.info("💡 **Idempotence**: `A*A` is filtered. **Absorption**: `a*b*c + a*b = a*b*c` is handled by set unions. **Distributivity**: `A*(B+C)` creates a star graph.")
