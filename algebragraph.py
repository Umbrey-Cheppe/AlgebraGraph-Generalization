import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import re

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Algebraic Graph Visualizer", layout="centered")
st.title("🌟 General Algebraic Graph Visualizer 🌟")
st.markdown("""
Enter a symbolic expression to generate a graph.
Examples:
- **Edge/Path:** `a*b`, `1*2*3`, `node_A*node_B*node_C`
- **Union:** `(a*b)+(c*d)`, `(1*2*3)+(4*5)`
- **Star (still supported):** `Center*(Leaf1+Leaf2)`, `1*(2+3+4)`
- **Range (still supported):** `Alpha*(1+2+...+5)`, `x_1*(x_2+...+x_5)`

*(Note: Advanced algebraic simplification like `a*b*c + a*b = a*b*c` will be handled in subsequent enhancements. For now, it will generate all specified edges.)*
""")

# --- Input Field ---
expr = st.text_input("Graph Expression:", "(1*2*3)+(4*5)", help="Try different formats!")

# --- Customization Options ---
st.sidebar.header("Graph Customization")
node_color = st.sidebar.color_picker("Node Color", "#ADD8E6")
edge_color = st.sidebar.color_picker("Edge Color", "#808080")
font_color = st.sidebar.color_picker("Label Color", "#333333")
node_size = st.sidebar.slider("Node Size", 100, 5000, 2000)
font_size = st.sidebar.slider("Font Size", 8, 24, 12)

# --- Algebraic Graph Parser (Major Refactor) ---

def parse_single_path_expression(path_expr):
    """
    Parses a single path expression (e.g., "a*b*c" or "1*2") into a list of edges.
    """
    path_nodes = [node.strip() for node in path_expr.split('*') if node.strip()]
    
    if len(path_nodes) < 2: # A single node, or empty after split
        return []

    edges = []
    # Convert nodes to int if they are purely numeric, otherwise keep as string
    processed_nodes = []
    for node in path_nodes:
        if node.isdigit():
            processed_nodes.append(int(node))
        else:
            processed_nodes.append(node)

    # Form edges from the path
    for i in range(len(processed_nodes) - 1):
        edges.append((processed_nodes[i], processed_nodes[i+1]))
    
    return edges

def parse_star_graph_expression(star_expr):
    """
    Parses star graph expressions (e.g., "Center*(Leaf1+Leaf2)")
    This is adapted from your previous working parser.
    """
    star_pattern = r"^([A-Za-z0-9_]+)\*\((.*)\)$"
    match = re.match(star_pattern, star_expr)

    if not match:
        return []

    center_label = match.group(1).strip()
    leaves_part = match.group(2).strip()

    try:
        # Pattern 1.1: Numeric Range (e.g., 1*(2+3+...+10))
        numeric_range_pattern = r"^(\d+)\s*\+\s*(?:\.\.\.|…)\s*(\d+)$"
        match_numeric_range = re.match(numeric_range_pattern, leaves_part)

        if match_numeric_range:
            start = int(match_numeric_range.group(1))
            end = int(match_numeric_range.group(2))
            if start > end: return []
            parsed_center = int(center_label) if center_label.isdigit() else center_label
            return [(parsed_center, str(i)) for i in range(start, end + 1)]

        # Pattern 1.2: Alphanumeric/Underscore Label Range (e.g., x_2+...+x_5)
        alphanum_range_pattern = r"^([A-Za-z_]+)(\d+)\s*\+\s*(?:\.\.\.|…)\s*([A-Za-z_]+)(\d+)$"
        match_alphanum_range = re.match(alphanum_range_pattern, leaves_part)

        if match_alphanum_range:
            start_prefix = match_alphanum_range.group(1)
            start_num = int(match_alphanum_range.group(2))
            end_prefix = match_alphanum_range.group(3)
            end_num = int(match_alphanum_range.group(4))
            if start_prefix != end_prefix or start_num > end_num: return []
            parsed_center = int(center_label) if center_label.isdigit() else center_label
            return [(parsed_center, f"{start_prefix}{i}") for i in range(start_num, end_num + 1)]

        # Direct list patterns (numeric or alphanumeric/underscore)
        leaf_labels = [tok.strip() for tok in leaves_part.split("+") if tok.strip()]
        if not leaf_labels: return []

        edges = []
        parsed_center = int(center_label) if center_label.isdigit() else center_label
        all_leaves_are_digits = all(leaf.isdigit() for leaf in leaf_labels)

        for leaf in leaf_labels:
            edges.append((parsed_center, int(leaf)) if all_leaves_are_digits else (parsed_center, leaf))
        
        return edges

    except ValueError:
        return [] # Invalid number format
    except Exception:
        return [] # Other parsing errors for star graphs


def parse_general_graph_expression(expr):
    """
    Parses a general graph expression, handling paths and unions.
    This is a basic implementation; more complex parsing would need a proper tokenizer/parser.
    """
    all_edges = set() # Use a set to automatically handle duplicates (union property)
    
    # First, handle expressions that are unions of sub-expressions
    # This regex attempts to split by '+' outside of parentheses (basic approach)
    # This is a simplification; a full parser would handle nested parentheses better.
    union_parts_raw = re.split(r'\)\s*\+\s*\(', expr)
    union_parts = []

    if len(union_parts_raw) > 1: # It looks like a union of terms in parentheses
        # Reconstruct parts with original parentheses or infer missing ones
        for i, part in enumerate(union_parts_raw):
            clean_part = part.strip()
            if not clean_part.startswith('(') and i > 0:
                clean_part = '(' + clean_part
            if not clean_part.endswith(')') and i < len(union_parts_raw) - 1:
                clean_part = clean_part + ')'
            union_parts.append(clean_part)
    else: # Not a union of parenthesized terms, might be a single term or plain list
        union_parts = [expr] # Treat the whole expression as one part


    for part in union_parts:
        part = part.strip().strip('()') # Clean up outer parentheses for sub-expressions

        # Try to parse as a star graph expression first
        star_edges = parse_star_graph_expression(part)
        if star_edges:
            all_edges.update(star_edges)
            continue # Go to next part

        # Try to parse as a simple path expression (a*b*c)
        # This will also catch single edges (a*b)
        path_edges = parse_single_path_expression(part)
        if path_edges:
            all_edges.update(path_edges)
            continue # Go to next part

        # If it's none of the above, it's an unparseable part
        st.warning(f"💡 Could not interpret part: `{part}`. Please check format.")
        return set() # Return empty set if any part fails to parse

    return list(all_edges) # Convert back to list for NetworkX

# --- Graph Drawing ---
st.markdown("---")
if st.button("Draw Graph"):
    raw_edges = parse_general_graph_expression(expr)
    
    if not raw_edges:
        st.error("❌ Failed to parse expression. Please ensure it follows a supported format. Check hints above.")
    else:
        # --- Filter out self-loops (u, u) based on your algebraic idempotence (A*A=A) ---
        filtered_edges = []
        for u, v in raw_edges:
            if u != v: # Only add edge if the two endpoints are different
                filtered_edges.append((u, v))
        
        if not filtered_edges:
            st.info("The expression resulted in only self-loops, which are filtered based on algebraic idempotence (e.g., `A*A=A`). No distinct edges to display.")
            st.warning("Consider an expression that creates distinct connections, like `1*(2+3)` or `a*b`.")
            
        else:
            G = nx.Graph()
            G.add_edges_from(filtered_edges) # Add the filtered edges
            
            # Ensure any explicit isolated nodes from the expression are added (e.g., if input was just "A")
            # This requires a more robust way to get all nodes from the expression, not just center.
            # For now, NetworkX adds nodes when edges are added. If a node is truly isolated,
            # it might not be in filtered_edges.
            # A more robust solution would be to gather all unique nodes from the parsed expression.

            pos = nx.spring_layout(G, seed=42)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            nx.draw(G, pos,
                    with_labels=True,
                    node_color=node_color,
                    edge_color=edge_color,
                    font_color=font_color,
                    font_weight="bold",
                    node_size=node_size,
                    font_size=font_size,
                    ax=ax)
            
            ax.set_title(f"Graph for: `{expr}` (Self-loops filtered)", size=font_size + 4, color=font_color)
            st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.info("💡 This app visualizes general simple graphs based on algebraic expressions. Self-loops (e.g., `A*A` or `1*(1+...)`) are **filtered** based on the algebraic interpretation that `A*A=A` (idempotence) implies no distinct edge is formed. Basic union (`+`) is handled, but advanced simplification (like `a*b*c + a*b = a*b*c`) is a future enhancement.")


