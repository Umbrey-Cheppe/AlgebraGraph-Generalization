import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import re
from itertools import combinations
import math
import plotly.graph_objects as go # For 3D visualization

# --- Installation Instructions (for user) ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Install Libraries")
st.sidebar.code("pip install matplotlib plotly streamlit networkx")
st.sidebar.markdown("---")

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Algebraic Graph Visualizer", layout="wide") # Changed to wide layout for more space
st.title("🌟 Algebraic Graph Visualizer (General Graphs) 🌟")
st.markdown("""
Enter a symbolic expression to generate a graph.
- `*` for **clique/complete subgraph** (e.g., `a*b` for edge, `1*2*3` for triangle on 1,2,3). Order doesn't matter (associative, commutative).
- `+` for **graph union** (e.g., `(a*b)+(c*d)`). Follows absorption law `G_sup + G_sub = G_sup`.
- **Star Graph** syntax (e.g., `Center*(Leaf1+Leaf2)` or `Center*(Leaf1+Leaf2+...+LeafN)`). `Center` connects to individual nodes of leaves, and complex leaves (`D*E`) form their own cliques.

**Examples:**
- `a*b*c` (triangle on a,b,c)
- `(1*2)+(3*4)` (two separate edges)
- `1*(2+3+4)` (star graph with 1 as center, leaves 2,3,4)
- `A*(1+2+...+100)` (star graph with A as center, connects to 1, 2, ..., 100)
- `(a*b*c) + (a*b)` (simplifies to `a*b*c` due to absorption)
- `A*(B+C+D*E+F)` (A connects to B,C,D,E,F; also D connects to E)
""")

# --- Input Field ---
expr = st.text_input("Graph Expression:", "A*(1+2+...+100)", help="Try different formats!") # Default to a large star graph

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
node_size = st.sidebar.slider("Node Size", 100, 5000, 2000)
node_shape = st.sidebar.selectbox("Node Shape", ["o", "s", "D", "^", "v", "h"], format_func=lambda x: {"o":"Circle", "s":"Square", "D":"Diamond", "^":"Triangle Up", "v":"Triangle Down", "h":"Hexagon"}[x], help="Matplotlib marker style.")
node_border_color = st.sidebar.color_picker("Node Border Color", "#000000")
node_border_width = st.sidebar.slider("Node Border Width", 0.0, 5.0, 1.0)


# Edge Customization
st.sidebar.subheader("Edges")
edge_color = st.sidebar.color_picker("Edge Color", "#808080")
edge_width = st.sidebar.slider("Edge Width", 0.5, 5.0, 1.0)
edge_style = st.sidebar.selectbox("Edge Style", ["solid", "dashed", "dotted"], help="Matplotlib linestyle.")

# Label Customization
st.sidebar.subheader("Labels")
font_color = st.sidebar.color_picker("Label Color", "#333333")
font_size = st.sidebar.slider("Font Size", 8, 24, 12)
label_bgcolor_enabled = st.sidebar.checkbox("Label Background", False)
label_bgcolor = st.sidebar.color_picker("Label Background Color", "#FFFFFF", disabled=not label_bgcolor_enabled)


# Plot Background
st.sidebar.subheader("Plot Background")
plot_bgcolor = st.sidebar.color_picker("Plot Background Color", "#F0F2F6")

# 3D View Option
st.sidebar.subheader("3D View (Experimental)")
enable_3d_view = st.sidebar.checkbox("Enable 3D View (Plotly)", False)

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
        clique_edges.add((u, v)) # Edges are inherently symmetric in undirected graph
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
                raise ValueError("Expected ')' after star graph leaves.")
            
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
    """Parses a primary factor: a node, a parenthesized expression."""
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
                    st.warning(f"💡 Invalid numeric range: {start_token}+...+{end_token}. Start must be <= End.")

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
                        st.warning(f"💡 Invalid alphanumeric range: {start_token}+...+{end_token}. Start Num <= End Num.")
                else:
                    st.warning(f"💡 Mismatched prefixes or invalid format for alphanumeric range: {start_token}+...+{end_token}. ")

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
                raise ValueError(f"Failed to parse leaf term `{sub_expr_string}`: {e}")
            except Exception as e:
                raise ValueError(f"An unexpected error occurred parsing leaf term `{sub_expr_string}`: {e}")
                
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
            st.error(f"⚠️ Unparsed tokens remaining after expression: `{''.join(tokens[final_index:])}`. Check your expression syntax.")
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
        st.error("❌ Failed to parse expression or no nodes/edges generated. Check expression and hints.")
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
            if G.order() > 1:
                degrees = dict(G.degree())
                potential_centers = [node for node, degree in degrees.items() if degree == G.order() - 1]
                
                if len(potential_centers) == 1:
                    center_node = potential_centers[0]
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
                    st.info("Detected star graph. Using custom radial layout for clear center display.")
                    pos[center_node] = (0, 0)
                    leaf_nodes = [node for node in G.nodes() if node != center_node]
                    num_leaves = len(leaf_nodes)
                    
                    # More robust radius calculation:
                    # Scale based on the cube root of the number of leaves for better distribution
                    # and ensure a minimum spread.
                    radius = 2.0 + (num_leaves**0.33) * 0.5 # Further fine-tuned scaling factor
                    
                    sorted_leaf_nodes = sorted(leaf_nodes, key=str) # Consistent order
                    for i, leaf_node in enumerate(sorted_leaf_nodes):
                        # Add a small random jitter to angles to prevent perfect overlaps for very large N
                        jitter_angle = (random.random() - 0.5) * (2 * math.pi / num_leaves) * 0.1 # Max 10% of angular separation
                        angle = 2 * math.pi * i / num_leaves + jitter_angle
                        x = radius * math.cos(angle)
                        y = radius * math.sin(angle)
                        pos[leaf_node] = (x, y)
                elif G.order() > 50:
                    st.warning(f"Graph has {G.order()} nodes. Visualization might be slow or crowded. Using circular layout.")
                    pos = nx.circular_layout(G)
                else:
                    pos = nx.spring_layout(G, seed=42)
            elif layout_type == "Spring":
                pos = nx.spring_layout(G, seed=42)
            elif layout_type == "Circular":
                pos = nx.circular_layout(G)
            elif layout_type == "Kamada-Kawai":
                pos = nx.kamada_kawai_layout(G)
            elif layout_type == "Spectral":
                pos = nx.spectral_layout(G)
            elif layout_type == "Shell":
                pos = nx.shell_layout(G)


            # --- Plotting ---
            if enable_3d_view:
                st.subheader("3D Graph View (Interactive)")
                # Prepare data for Plotly 3D
                edge_x = []
                edge_y = []
                edge_z = []

                # For 3D star graph, place leaves on a sphere
                if is_star_graph:
                    # Center at (0,0,0)
                    pos_3d = {center_node: (0, 0, 0)}
                    leaf_nodes_3d = [node for node in G.nodes() if node != center_node]
                    num_leaves_3d = len(leaf_nodes_3d)
                    
                    # Sphere radius
                    sphere_radius = 2.0 + (num_leaves_3d**0.33) * 0.3 # Adjusted 3D radius
                    
                    # Distribute leaves on a sphere using Golden Spiral or similar
                    # This is a common way to evenly distribute points on a sphere
                    golden_angle = math.pi * (3 - math.sqrt(5))
                    for i, leaf_node in enumerate(sorted(leaf_nodes_3d, key=str)):
                        y = 1 - (i / float(num_leaves_3d - 1)) * 2  # y goes from 1 to -1
                        radius_at_y = math.sqrt(1 - y * y)
                        theta = golden_angle * i
                        x = math.cos(theta) * radius_at_y
                        z = math.sin(theta) * radius_at_y
                        pos_3d[leaf_node] = (sphere_radius * x, sphere_radius * y, sphere_radius * z)
                else:
                    # For non-star graphs, just project 2D layout onto Z=0 plane
                    pos_3d = {node: (coords[0], coords[1], 0) for node, coords in pos.items()}


                for edge in G.edges():
                    x0, y0, z0 = pos_3d[edge[0]]
                    x1, y1, z1 = pos_3d[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_z.extend([z0, z1, None])

                edge_trace = go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    line=dict(width=edge_width, color=edge_color),
                    hoverinfo='none',
                    mode='lines'
                )

                node_x = [pos_3d[node][0] for node in G.nodes()]
                node_y = [pos_3d[node][1] for node in G.nodes()]
                node_z = [pos_3d[node][2] for node in G.nodes()]
                node_text = [str(node) for node in G.nodes()] # For hover labels

                node_trace = go.Scatter3d(
                    x=node_x, y=node_y, z=node_z,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    marker=dict(
                        showscale=False,
                        color=node_color,
                        size=node_size/300, # Adjust size for Plotly's scale
                        line=dict(width=node_border_width/2, color=node_border_color), # Adjust border width
                        opacity=0.9
                    ),
                    textfont=dict(
                        color=font_color,
                        size=font_size
                    ),
                    textposition="middle center"
                )

                fig_3d = go.Figure(data=[edge_trace, node_trace])
                fig_3d.update_layout(
                    title=f"3D Graph for: `{expr}`",
                    scene=dict(
                        xaxis=dict(showbackground=False, showticklabels=False, showgrid=False, zeroline=False),
                        yaxis=dict(showbackground=False, showticklabels=False, showgrid=False, zeroline=False),
                        zaxis=dict(showbackground=False, showticklabels=False, showgrid=False, zeroline=False),
                        aspectmode='cube', # Ensure proportional scaling
                        bgcolor=plot_bgcolor
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                    showlegend=False,
                    hovermode='closest',
                    plot_bgcolor=plot_bgcolor # Ensure background matches
                )
                st.plotly_chart(fig_3d, use_container_width=True)

            else: # 2D Matplotlib View
                st.subheader("2D Graph View (Matplotlib)")
                fig, ax = plt.subplots(figsize=(10, 8)) 
                ax.set_facecolor(plot_bgcolor) # Set background color

                nx.draw(G, pos,
                        with_labels=True,
                        node_color=node_color,
                        edgecolors=node_border_color, # Node border color
                        linewidths=node_border_width, # Node border width
                        node_shape=node_shape, # Node shape
                        edge_color=edge_color,
                        width=edge_width, # Edge width
                        style=edge_style, # Edge style
                        font_color=font_color,
                        font_weight="bold",
                        node_size=node_size,
                        font_size=font_size,
                        ax=ax)
                
                # Add label background if enabled (matplotlib specific)
                if label_bgcolor_enabled:
                    for node, (x, y) in pos.items():
                        # Find the corresponding text object
                        for text_obj in ax.texts:
                            if text_obj.get_text() == str(node):
                                text_obj.set_bbox(dict(facecolor=label_bgcolor, edgecolor='none', boxstyle='round,pad=0.2'))
                                break # Found it, move to next node

                ax.set_title(f"Graph for: `{expr}` (Self-loops filtered, terms absorbed)", size=font_size + 4, color=font_color)
                # Adjust plot limits to ensure center is indeed centered and all nodes fit
                min_x = min(p[0] for p in pos.values())
                max_x = max(p[0] for p in pos.values())
                min_y = min(p[1] for p in pos.values())
                max_y = max(p[1] for p in pos.values())
                
                padding = max((max_x - min_x) * 0.1, (max_y - min_y) * 0.1, 0.5) # Add 10% padding or minimum 0.5
                ax.set_xlim(min_x - padding, max_x + padding)
                ax.set_ylim(min_y - padding, max_y + padding)
                ax.set_aspect('equal', adjustable='box') # Maintain aspect ratio
                st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.info("💡 This app visualizes general simple graphs based on algebraic expressions. Self-loops (e.g., `A*A`) are **filtered** due to `A*A=A` idempotence. Absorption (`a*b*c + a*b = a*b*c`) is applied by the set-based union of generated clique edges.")

