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
- **Star Graph** syntax (e.g., `Center*(Leaf1+Leaf2)` or `1*(2+3+...+10)`). `Center` connects to individual nodes of leaves, and complex leaves (`D*E`) form their own cliques.

**Examples:**
- `a*b*c` (triangle on a,b,c)
- `(1*2)+(3*4)` (two separate edges)
- `1*(2+3+4)` (star graph with 1 as center, leaves 2,3,4)
- `A*(B+C+...+E)` (range-based star graph)
- `(a*b*c) + (a*b)` (simplifies to `a*b*c` due to absorption)
- `A*(B+C+D*E+F)` (A connects to B,C,D,E,F; also D connects to E)
- `1*(1+2+...+10)` (Star graph. 1 is center, connects to 1-10. Self-loops are filtered.)
""")

# --- Input Field ---
expr = st.text_input("Graph Expression:", "1*(1+2+...+10)", help="Try different formats!")

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
        start_of_leaf_term_index = current_index
        
        # Determine the end of the current leaf term.
        # It's either the next '+' at the same parenthesis level, or the closing ')'
        paren_level = 0
        end_of_leaf_term_index = current_index
        while end_of_leaf_term_index < len(tokens):
            token = tokens[end_of_leaf_term_index]
            if token == '(':
                paren_level += 1
            elif token == ')':
                paren_level -= 1
            
            if token == '+' and paren_level == 0:
                break # Found a '+' at the current level, this is the end of the leaf term
            if token == ')' and paren_level == -1: # This handles the closing ')' for the whole leaf list
                 break
            
            end_of_leaf_term_index += 1

        leaf_term_tokens = tokens[start_of_leaf_term_index:end_of_leaf_term_index]
        
        if not leaf_term_tokens:
            raise ValueError("Empty leaf term found in star graph list.")
            
        # Try to parse as a range first
        # A range pattern typically has at least 5 tokens: NODE, +, ..., +, NODE
        is_range = False
        if len(leaf_term_tokens) >= 5 and leaf_term_tokens[1] == '+' and leaf_term_tokens[2] == '...':
            # Numeric range: N+...N (e.g., 2+...+10)
            if re.match(r'^\d+$', leaf_term_tokens[0]) and re.match(r'^\d+$', leaf_term_tokens[len(leaf_term_tokens)-1]):
                start_num = int(leaf_term_tokens[0])
                end_num = int(leaf_term_tokens[len(leaf_term_tokens)-1])
                if start_num <= end_num:
                    for i in range(start_num, end_num + 1):
                        all_individual_leaf_nodes.add(parse_nodes(str(i)))
                    is_range = True
            # Alphanumeric range: X_N+...+X_N (e.g., x_2+...+x_5)
            # This pattern is more strict: 'prefix', 'num', '+', '...', 'prefix', 'num'
            elif len(leaf_term_tokens) == 5 and \
                 re.match(r'^([A-Za-z_]+)(\d+)$', leaf_term_tokens[0]) and \
                 re.match(r'^([A-Za-z_]+)(\d+)$', leaf_term_tokens[4]):
                
                match_start = re.match(r'^([A-Za-z_]+)(\d+)$', leaf_term_tokens[0])
                match_end = re.match(r'^([A-Za-z_]+)(\d+)$', leaf_term_tokens[4])
                
                if match_start and match_end:
                    start_prefix = match_start.group(1)
                    start_num = int(match_start.group(2))
                    end_prefix = match_end.group(1)
                    end_num = int(match_end.group(2))
                    
                    if start_prefix == end_prefix and start_num <= end_num:
                        for i in range(start_num, end_num + 1):
                            all_individual_leaf_nodes.add(parse_nodes(f"{start_prefix}{i}"))
                        is_range = True
            
            if not is_range: # If it looked like a range but didn't match the specific pattern
                st.warning(f"💡 Range format not fully recognized for: `{''.join(leaf_term_tokens)}`. Trying as general term.")

        if not is_range: # Not a range, treat as a regular sub-expression
            sub_expr_string = "".join(leaf_term_tokens).strip()
            if not sub_expr_string:
                raise ValueError("Empty sub-expression string for leaf term.")
            
            try:
                sub_expr_tokens = tokenize(sub_expr_string)
                if not sub_expr_tokens: # Check for empty token list after re-tokenizing
                    raise ValueError(f"Could not tokenize sub-expression: `{sub_expr_string}`")
                    
                sub_expr_edges, _, sub_expr_nodes = parse_expression(sub_expr_tokens, 0)
                all_leaves_edges.update(sub_expr_edges)
                all_individual_leaf_nodes.update(sub_expr_nodes)
            except ValueError as e:
                raise ValueError(f"Failed to parse leaf term `{sub_expr_string}`: {e}")
            except Exception as e:
                raise ValueError(f"An unexpected error occurred parsing leaf term `{sub_expr_string}`: {e}")
                
        current_index = end_of_leaf_term_index
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
                final_edges_for_nx.append((u, v))
        
        # --- Create Graph ---
        G = nx.Graph()
        
        # Add all nodes identified during parsing, ensuring isolated nodes are included
        G.add_nodes_from(list(all_nodes_in_expr))
        
        # Add filtered edges
        G.add_edges_from(final_edges_for_nx)
        
        # --- Visualization ---
        if not G.nodes(): # No nodes at all after processing
            st.info("The expression resulted in no visible graph (possibly empty or only filtered self-loops).")
            st.warning("Consider an expression that creates distinct connections, like `1*(2+3)` or `a*b`.")
        else:
            # If the graph is very large (e.g., 100 nodes), spring_layout might be slow
            # or result in an unreadable tangle. Max nodes for visualization is a practical limit.
            if G.order() > 50: # If more than 50 nodes, use a simpler layout or warn
                st.warning(f"Graph has {G.order()} nodes. Visualization might be slow or crowded. Using circular layout.")
                pos = nx.circular_layout(G)
            else:
                pos = nx.spring_layout(G, seed=42) # Consistent layout
            
            fig, ax = plt.subplots(figsize=(10, 8)) # Increased figsize for larger graphs
            
            nx.draw(G, pos,
                    with_labels=True,
                    node_color=node_color,
                    edge_color=edge_color,
                    font_color=font_color,
                    font_weight="bold",
                    node_size=node_size,
                    font_size=font_size,
                    ax=ax)
            
            ax.set_title(f"Graph for: `{expr}` (Self-loops filtered, terms absorbed)", size=font_size + 4, color=font_color)
            st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.info("💡 This app visualizes general simple graphs based on algebraic expressions. Self-loops (e.g., `A*A`) are **filtered** due to `A*A=A` idempotence. Absorption (`a*b*c + a*b = a*b*c`) is applied by the set-based union of generated clique edges.")
