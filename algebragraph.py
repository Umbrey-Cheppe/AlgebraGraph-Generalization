import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import re
from itertools import combinations
import math
import random # For random jitter in layout

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Algebraic Graph Visualizer", layout="wide")
st.title("🌟 Algebraic Graph Visualizer (General Graphs) 🌟")
st.markdown("""
Enter a symbolic expression to generate a graph.
- `*` for **clique/complete subgraph** (e.g., `a*b` for edge, `1*2*3` for triangle on 1,2,3). Order doesn't matter (associative, commutative).
- `+` for **graph union** (e.g., `(a*b)+(c*d)`). Follows absorption law `G_sup + G_sub = G_sup`.
- **Star Graph** syntax (e.g., `Center*(Leaf1+Leaf2)` or `Center*(Leaf1+Leaf2+...+LeafN)`).
  - `Center` connects to individual nodes of leaves.
  - Complex leaves (`D*E`) form their own cliques.
  - **Important for Ranges:** For numerical ranges, **you MUST use `Start + ... + End`** (e.g., `A*(1+2+...+100)`). The `+` signs around `...` are **CRUCIAL** for correct parsing. `A*(1...100)` is **NOT** a valid syntax and will cause an error.
  - **Important for Ranges:** For alphanumeric ranges, use `Prefix_StartNum + ... + Prefix_EndNum` (e.g., `X_1+X_2+...+X_10`).

**Examples:**
- `a*b*c` (triangle on a,b,c)
- `(1*2)+(3*4)` (two separate edges)
- `1*(2+3+4)` (star graph with 1 as center, leaves 2,3,4)
- `A*(1+2+...+100)` (star graph with A as center, connects to 1, 2, ..., 100)
- `(a*b*c) + (a*b)` (simplifies to `a*b*c` due to absorption)
- `A*(B+C+D*E+F)` (A connects to B,C,D,E,F; also D connects to E)
- `(1+2+...+5)` (a set of isolated nodes 1,2,3,4,5)
""")

# --- Input Field ---
expr = st.text_input("Graph Expression:", "A*(1+2+...+100)", help="Try different formats!")

# --- Customization Options (Sidebar) ---
st.sidebar.header("Graph Customization")

# Layout Selection
layout_type = st.sidebar.selectbox(
    "Layout Algorithm",
    ("Auto (Spring/Star/Circular)", "Spring", "Circular", "Kamada-Kawai", "Spectral", "Shell"),
    help="Choose a layout algorithm. 'Auto' tries to optimize for star graphs."
)

# Node Customization
st.sidebar.subheader("Nodes")
node_color = st.sidebar.color_picker("Node Color", "#ADD8E6")
node_size = st.sidebar.slider("Node Size", 100, 5000, 1000) # Default node size changed to 1000
node_shape = st.sidebar.selectbox("Node Shape", ["o", "s", "D", "^", "v", "h"], format_func=lambda x: {"o":"Circle", "s":"Square", "D":"Diamond", "^":"Triangle Up", "v":"Triangle Down", "h":"Hexagon"}[x], help="Matplotlib marker style.")
node_border_color = st.sidebar.color_picker("Node Border Color", "#000000")
node_border_width = st.sidebar.slider("Node Border Width", 0.0, 5.0, 1.0)


# Edge Customization
st.sidebar.subheader("Edges")
edge_color = st.sidebar.color_picker("Edge Color", "#808080")
edge_width = st.sidebar.slider("Edge Width", 0.5, 5.0, 1.0)
edge_style = st.sidebar.selectbox("Edge Style", ["solid", "dashed", "dotted"], help="Matplotlib linestyle.")

# Edge Labels
edge_labels_enabled = st.sidebar.checkbox("Show Edge Labels (Experimental)", False)
edge_label_color = st.sidebar.color_picker("Edge Label Color", "#555555", disabled=not edge_labels_enabled)
edge_label_font_size = st.sidebar.slider("Edge Label Font Size", 6, 18, 9, disabled=not edge_labels_enabled)


# Label Customization (for nodes)
st.sidebar.subheader("Node Labels")
font_color = st.sidebar.color_picker("Label Color", "#333333")
font_size = st.sidebar.slider("Font Size", 8, 24, 12)
label_bgcolor_enabled = st.sidebar.checkbox("Label Background", False)
label_bgcolor = st.sidebar.color_picker("Label Background Color", "#FFFFFF", disabled=not label_bgcolor_enabled)

# Global Font Family
font_family = st.sidebar.selectbox("Font Family", ["sans-serif", "serif", "monospace", "fantasy", "cursive"], help="Global font for all labels.")


# Plot Background
st.sidebar.subheader("Plot Background")
plot_bgcolor = st.sidebar.color_picker("Plot Background Color", "#F0F2F6")

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
    # Sort for consistent edge representation (u,v) where u<v, regardless of input order
    sorted_nodes = sorted(nodes_list, key=str) 
    for u, v in combinations(sorted_nodes, 2):
        clique_edges.add(tuple(sorted((u, v), key=str))) # Store as sorted tuple
    return clique_edges

# --- Main Parsing Logic ---

# Tokenization
TOKEN_PATTERN = re.compile(r'([A-Za-z0-9_]+|\*|\+|\(|\)|\.\.\.)')

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
        if index + 1 < len(tokens) and tokens[index+1] == '(':
            # It's a star graph!
            center_node_token = tokens[index-1] # The token before '*' was the center
            center_node = parse_nodes(center_node_token)

            # Consume '*' and '('
            index += 2
            
            # Parse the leaves list within the parentheses
            leaves_edges, index, leaves_nodes = parse_leaf_list(tokens, index)
            
            # Combine current edges (if any from center_node itself, though usually empty) with leaves_edges (from complex leaves like D*E)
            edges_from_factor1.update(leaves_edges)
            nodes_in_factor1.update(leaves_nodes) # Union of all nodes in leaves

            # Add edges from center to each individual node in the leaves_nodes
            for leaf_node in leaves_nodes:
                # Ensure we don't add self-loops explicitly here; filtering happens later.
                if center_node != leaf_node:
                    edges_from_factor1.add(tuple(sorted((center_node, leaf_node), key=str)))
            
            # Ensure center node is included in the set of nodes for this factor
            nodes_in_factor1.add(center_node)

            # Expect ')'
            if index < len(tokens) and tokens[index] == ')':
                index += 1
            else:
                raise ValueError("Expected ')' after star graph leaves. For numerical ranges, you MUST use `Start+... +End` (e.g., `A*(1+2+...+100)`). The `+` signs around `...` are crucial.")
            
        else: # Standard multiplication (clique formation)
            # Collect all nodes connected by '*'
            node_tokens = [tokens[index-1]] # Start with the node before '*'
            
            while index < len(tokens) and tokens[index] == '*':
                index += 1 # Consume '*'
                if index < len(tokens) and re.match(r'[A-Za-z0-9_]+', tokens[index]):
                    node_tokens.append(tokens[index])
                    index += 1
                else:
                    raise ValueError("Expected node after '*' operator.")
            
            # Convert node tokens to parsed nodes
            clique_nodes = [parse_nodes(n) for n in node_tokens]
            
            # Generate clique edges for these nodes
            edges_from_factor1 = generate_clique_edges(clique_nodes)
            nodes_in_factor1 = set(clique_nodes) # All nodes in this clique
            
    return edges_from_factor1, index, nodes_in_factor1

def parse_factor(tokens, index):
    """Parses a primary factor: a node, or a parenthesized expression."""
    current_token = tokens[index]
    
    if current_token == '(':
        index += 1 # Consume '('
        # Try to parse the content of the parentheses as a full expression (sum or clique)
        edges, index, nodes = parse_expression(tokens, index) # This handles (1+2+...+5)
        if index < len(tokens) and tokens[index] == ')':
            index += 1 # Consume ')'
            return edges, index, nodes
        else:
            raise ValueError("Mismatched parentheses: Expected ')'.")
    elif re.match(r'[A-Za-z0-9_]+', current_token): # It's a node
        node = parse_nodes(current_token)
        index += 1
        return set(), index, {node} # A single node forms no edges on its own
    else:
        raise ValueError(f"Unexpected token: `{current_token}` at index {index}. Expected node or '('. For numerical ranges, use `Start+... +End`.")

def parse_leaf_list(tokens, current_index):
    """
    Parses the plus-separated list of leaves within a star graph's parentheses.
    Handles ranges (e.g., 2+...+10) and complex leaf terms (e.g., D*E).
    Returns (set of edges from complex leaves, new_index, set of individual leaf nodes)
    """
    all_leaves_edges = set()
    all_individual_leaf_nodes = set()
    
    while current_index < len(tokens) and tokens[current_index] != ')':
        is_term_parsed = False # Flag to indicate if current term was handled
        
        # --- Attempt to parse as a range first: N + ... + N or X_N + ... + X_N ---
        # We need at least 5 tokens for this pattern: node, +, ..., +, node
        if current_index + 4 < len(tokens) and \
           tokens[current_index + 1] == '+' and \
           tokens[current_index + 2] == '...' and \
           tokens[current_index + 3] == '+':
            
            start_token = tokens[current_index]
            end_token = tokens[current_index + 4]

            # Numeric range (e.g., 1+2+...+10)
            if re.match(r'^\d+$', start_token) and re.match(r'^\d+$', end_token):
                start_num = int(start_token)
                end_num = int(end_token)
                if start_num <= end_num:
                    for i in range(start_num, end_num + 1):
                        all_individual_leaf_nodes.add(parse_nodes(str(i)))
                    current_index += 5 # Consume: N, +, ..., +, N
                    is_term_parsed = True
                else:
                    st.warning(f"💡 Invalid numeric range: `{start_token}+...+{end_token}`. Start must be <= End. Check syntax: `Start+... +End`.")

            # Alphanumeric range (e.g., A_1+A_2+...+A_10)
            elif re.match(r'^([A-Za-z_]+)(\d+)$', start_token) and \
                 re.match(r'^([A-Za-z_]+)(\d+)$', end_token):
                
                match_start = re.match(r'^([A-Za-z_]+)(\d+)$', start_token)
                match_end = re.match(r'^([A-Za-z_]+)(\d+)$', end_token)
                
                if match_start and match_end and match_start.group(1) == match_end.group(1):
                    start_prefix = match_start.group(1)
                    start_num = int(match_start.group(2))
                    end_num = int(match_end.group(2))
                    
                    if start_num <= end_num:
                        for i in range(start_num, end_num + 1):
                            all_individual_leaf_nodes.add(parse_nodes(f"{start_prefix}{i}"))
                        current_index += 5 # Consume: X_N, +, ..., +, X_N
                        is_term_parsed = True
                    else:
                        st.warning(f"💡 Invalid alphanumeric range: `{start_token}+...+{end_token}`. Start Num <= End Num. Check syntax: `Prefix_StartNum+... +Prefix_EndNum`.")
                else:
                    st.warning(f"💡 Mismatched prefixes or invalid format for alphanumeric range: `{start_token}+...+{end_token}`. Check syntax.")

        if not is_term_parsed: # If it's not a range, parse as a single leaf term (could be a node or complex expression)
            start_of_leaf_term_index = current_index
            temp_paren_level = 0
            end_of_current_leaf_term_index = current_index
            
            while end_of_current_leaf_term_index < len(tokens):
                token = tokens[end_of_current_leaf_term_index]
                if token == '(':
                    temp_paren_level += 1
                elif token == ')':
                    temp_paren_level -= 1
                
                # Break if we hit a '+' outside of any sub-parentheses, or the closing ')'
                if (token == '+' and temp_paren_level == 0) or \
                   (token == ')' and temp_paren_level == -1): # paren_level -1 means we just closed the main leaf list paren
                    break
                
                end_of_current_leaf_term_index += 1
            
            leaf_term_tokens = tokens[start_of_leaf_term_index:end_of_current_leaf_term_index]
            
            if not leaf_term_tokens:
                raise ValueError(f"Empty leaf term encountered in star graph list at index {current_index}.")
                
            sub_expr_string = "".join(leaf_term_tokens).strip()
            
            if not sub_expr_string:
                raise ValueError(f"Empty sub-expression string for leaf term starting at index {start_of_leaf_term_index}.")
            
            try:
                sub_expr_tokens = tokenize(sub_expr_string)
                if not sub_expr_tokens:
                    raise ValueError(f"Could not re-tokenize sub-expression: `{sub_expr_string}`")
                    
                # Recursive call to parse_expression for complex leaf terms
                sub_expr_edges, _, sub_expr_nodes = parse_expression(sub_expr_tokens, 0)
                all_leaves_edges.update(sub_expr_edges)
                all_individual_leaf_nodes.update(sub_expr_nodes)
            except ValueError as e:
                # Add more context to the error for better debugging of leaf terms
                raise ValueError(f"Failed to parse leaf term `{sub_expr_string}` within star graph. Ensure valid syntax: {e}")
            except Exception as e:
                raise ValueError(f"An unexpected error occurred parsing leaf term `{sub_expr_string}` within star graph: {e}")
                
            current_index = end_of_current_leaf_term_index

        # After processing any type of leaf term (range or single term), check for '+' separator
        if current_index < len(tokens) and tokens[current_index] == '+':
            current_index += 1 # Consume the '+' separator for the next leaf term
            
    return all_leaves_edges, current_index, all_individual_leaf_nodes


# --- Main Driver Function for Parsing and Simplification ---

def parse_and_simplify_graph_expression(expr):
    """
    Tokenizes the expression, parses it, and then effectively applies absorption
    through set-based union of edges.
    """
    expr = expr.replace(" ", "").strip()
    if not expr:
        return set(), set() # No edges, no nodes if empty expression

    tokens = tokenize(expr)
    
    try:
        raw_edges, final_index, all_nodes_in_expr = parse_expression(tokens, 0)
        
        if final_index != len(tokens):
            # Improved error message for unparsed tokens
            remaining_tokens_str = ''.join(tokens[final_index:])
            first_unparsed_token = tokens[final_index] if final_index < len(tokens) else "end of expression"
            st.error(f"⚠️ Unparsed tokens remaining after expression: `{remaining_tokens_str}` starting at `{first_unparsed_token}`. Check your expression syntax. Hint: For numerical ranges, use `Start+... +End` (e.g., `A*(1+2+...+100)`). The `...` must be surrounded by `+` signs.")
            return set(), set()
            
        return raw_edges, all_nodes_in_expr

    except ValueError as e:
        st.error(f"❌ Parsing Error: {e}")
        return set(), set()
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred: {e}. Please check your expression format.")
        return set(), set()

# --- Graph Drawing ---
st.markdown("---")
if st.button("Draw Graph"):
    edges_to_draw, all_nodes_in_expr = parse_and_simplify_graph_expression(expr)
    
    if not edges_to_draw and not all_nodes_in_expr:
        st.error("❌ Failed to parse expression or no nodes/edges generated. Check expression and hints provided above.")
    else:
        # --- Filter out self-loops (u, u) ---
        final_edges_for_nx = []
        for u, v in edges_to_draw:
            if u != v:
                final_edges_for_nx.append(tuple(sorted((u,v), key=str))) 
        
        # --- Create Graph ---
        G = nx.Graph()
        G.add_nodes_from(list(all_nodes_in_expr))
        G.add_edges_from(final_edges_for_nx)
        
        # --- Visualization ---
        if not G.nodes(): 
            st.info("The expression resulted in no visible graph (possibly empty or only filtered self-loops).")
            st.warning("Consider an expression that creates distinct connections, like `1*(2+3)` or `a*b`.")
        else:
            # --- Star Graph Detection (for custom layout) ---
            is_star_graph = False
            center_node = None
            if G.order() > 1: # G.order() is number of nodes
                degrees = dict(G.degree())
                # A node is a center of a star graph if its degree is N-1 (connected to all other N-1 nodes)
                potential_centers = [node for node, degree in degrees.items() if degree == G.order() - 1]
                
                if len(potential_centers) == 1:
                    center_node = potential_centers[0]
                    # Verify all other nodes are leaves (degree 1)
                    all_others_are_leaves = True
                    for node, degree in degrees.items():
                        if node != center_node and degree != 1:
                            all_others_are_leaves = False
                            break
                    if all_others_are_leaves:
                        is_star_graph = True
            
            pos = {} # Initialize position dictionary

            # Determine layout based on user selection or auto mode
            if layout_type == "Auto (Spring/Star/Circular)":
                if is_star_graph:
                    st.info("Detected star graph. Using custom radial multi-ring layout for clear center display.")
                    pos[center_node] = (0, 0) # Center node at origin
                    leaf_nodes = [node for node in G.nodes() if node != center_node]
                    num_leaves = len(leaf_nodes)
                    sorted_leaf_nodes = sorted(leaf_nodes, key=str) # Consistent order

                    # --- Multi-Ring Layout Logic ---
                    # Determine number of rings based on number of leaves
                    if num_leaves < 10:
                        num_rings = 1
                    elif num_leaves < 30:
                        num_rings = 2
                    elif num_leaves < 70:
                        num_rings = 3
                    elif num_leaves < 150:
                        num_rings = 4
                    else: # For very large graphs
                        num_rings = 5 # Can go higher if needed
                    
                    # Distribute leaves among rings (simple distribution for now)
                    # This ensures leaves are roughly evenly distributed across rings
                    nodes_per_ring_list = [0] * num_rings
                    for i in range(num_leaves):
                        nodes_per_ring_list[i % num_rings] += 1

                    base_radius = 2.0 
                    ring_spacing_factor = 1.2 # Controls how much larger each subsequent ring is

                    current_leaf_index = 0
                    for ring_idx in range(num_rings):
                        leaves_in_this_ring = nodes_per_ring_list[ring_idx]
                        if leaves_in_this_ring == 0:
                            continue

                        # Calculate radius for this ring
                        # Scale radius based on ring index and total number of leaves
                        # The (num_leaves**0.1) * 0.5 provides a slight global scaling based on total leaves
                        ring_radius = base_radius + (ring_idx * ring_spacing_factor) + (num_leaves**0.1) * 0.5 
                        
                        for i in range(leaves_in_this_ring):
                            leaf_node = sorted_leaf_nodes[current_leaf_index]
                            
                            # Add a small random jitter to angles to prevent perfect overlaps
                            jitter_amount = (2 * math.pi / max(1, leaves_in_this_ring)) * 0.02 # Max 2% of angular separation
                            jitter_angle = (random.random() - 0.5) * jitter_amount 
                            
                            angle = 2 * math.pi * i / leaves_in_this_ring + jitter_angle
                            x = ring_radius * math.cos(angle)
                            y = ring_radius * math.sin(angle)
                            pos[leaf_node] = (x, y)
                            current_leaf_index += 1

                elif G.order() > 50:
                    st.warning(f"Graph has {G.order()} nodes. Using circular layout for better spread.")
                    pos = nx.circular_layout(G)
                    # Apply a small angular offset to nodes to try and reduce edge overlap for circular layout
                    if pos:
                        angle_offset = random.uniform(0, 2 * math.pi / len(G.nodes())) # A random offset based on node density
                        for node in G.nodes():
                            x, y = pos[node]
                            current_angle = math.atan2(y, x)
                            new_angle = current_angle + angle_offset
                            radius = math.sqrt(x**2 + y**2)
                            pos[node] = (radius * math.cos(new_angle), radius * math.sin(new_angle))

                else:
                    pos = nx.spring_layout(G, seed=42) # Use a fixed seed for reproducibility
            elif layout_type == "Spring":
                pos = nx.spring_layout(G, seed=42)
            elif layout_type == "Circular":
                pos = nx.circular_layout(G)
                # Apply a small angular offset to nodes to try and reduce edge overlap
                if pos:
                    angle_offset = random.uniform(0, 2 * math.pi / len(G.nodes()))
                    for node in G.nodes():
                        x, y = pos[node]
                        current_angle = math.atan2(y, x)
                        new_angle = current_angle + angle_offset
                        radius = math.sqrt(x**2 + y**2)
                        pos[node] = (radius * math.cos(new_angle), radius * math.sin(new_angle))
            elif layout_type == "Kamada-Kawai":
                pos = nx.kamada_kawai_layout(G)
            elif layout_type == "Spectral":
                pos = nx.spectral_layout(G)
            elif layout_type == "Shell":
                pos = nx.shell_layout(G)


            # --- Plotting ---
            st.subheader("2D Graph View (Matplotlib)")
            fig, ax = plt.subplots(figsize=(10, 8)) 
            fig.patch.set_facecolor(plot_bgcolor) # Set figure background
            ax.set_facecolor(plot_bgcolor) # Set axes background

            # Draw nodes
            nx.draw_networkx_nodes(G, pos,
                                   node_color=node_color,
                                   edgecolors=node_border_color,
                                   linewidths=node_border_width,
                                   node_shape=node_shape,
                                   node_size=node_size,
                                   ax=ax)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos,
                                   edge_color=edge_color,
                                   width=edge_width,
                                   style=edge_style,
                                   ax=ax)
            
            # Draw node labels
            node_labels = {node: str(node) for node in G.nodes()}
            texts = nx.draw_networkx_labels(G, pos, labels=node_labels,
                                            font_size=font_size,
                                            font_color=font_color,
                                            font_weight="bold",
                                            font_family=font_family, # Apply global font family
                                            ax=ax)
            
            # Add node label background if enabled
            if label_bgcolor_enabled:
                for _, text_obj in texts.items():
                    text_obj.set_bbox(dict(facecolor=label_bgcolor, edgecolor='none', boxstyle='round,pad=0.2'))

            # Draw edge labels (if enabled)
            if edge_labels_enabled:
                edge_labels_display = {}
                for u,v in G.edges():
                    edge_labels_display[(u,v)] = f"{u}-{v}" # Example: shows "Node1-Node2" on the edge

                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_display,
                                             font_size=edge_label_font_size,
                                             font_color=edge_label_color,
                                             font_family=font_family, # Apply global font family
                                             ax=ax)


            ax.set_title(f"Graph for: `{expr}` (Self-loops filtered, terms absorbed)", size=font_size + 4, color=font_color, fontfamily=font_family)
            
            # --- Adjust plot limits to ensure center is indeed centered and all nodes fit ---
            # Calculate min/max x and y coordinates from node positions
            if pos: # Ensure pos is not empty
                all_x = [p[0] for p in pos.values()]
                all_y = [p[1] for p in pos.values()]

                min_x = min(all_x)
                max_x = max(all_x)
                min_y = min(all_y)
                max_y = max(all_y)
                
                # Determine a dynamic padding based on the graph's extent
                # Ensure minimum padding to avoid labels being cut off for small graphs
                range_x = max_x - min_x
                range_y = max_y - min_y
                
                # Use a larger factor for star graphs
                # Increased padding again for very large star graphs (like 100 nodes)
                padding_factor = 0.28 if is_star_graph else 0.18 # Adjusted padding for multi-ring layout
                dynamic_padding = max(range_x * padding_factor, range_y * padding_factor, 1.5) # Minimum 1.5 for very small graphs
                
                ax.set_xlim(min_x - dynamic_padding, max_x + dynamic_padding)
                ax.set_ylim(min_y - dynamic_padding, max_y + dynamic_padding)
                ax.set_aspect('equal', adjustable='box') # Maintain aspect ratio
            
            plt.grid(False) # Turn off grid lines
            plt.axis('off') # Turn off axes
            st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.info("💡 This app visualizes general simple graphs based on algebraic expressions. Self-loops (e.g., `A*A`) are **filtered** due to `A*A=A` idempotence. Absorption (`a*b*c + a*b = a*b*c`) is applied by the set-based union of generated clique edges.")


