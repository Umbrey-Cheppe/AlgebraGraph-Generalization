import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import re

# --- Streamlit setup ---
st.set_page_config(page_title="Algebraic Graph Visualizer", layout="centered")
st.title("Algebraic Graph Visualizer")
st.markdown("Enter a symbolic expression like `1*(2+3+...+10)` or `x_1*(x_2+x_3)` to generate a star graph.")

# --- User input ---
expr = st.text_input("Graph Expression:", "1*(2+3+...+10)")
import re
def parse_graph_expression(expr):
    """
    Returns a list of (center, leaf) pairs for expressions like:
      1) Numeric-range star:    "1*(2+3+...+10)"
      2) Direct numeric list:   "1*(2+3+4)"
      3) Alphanumeric star:     "A*(B+C+D)"  or  "Node1*(LeafA+LeafB)"
      Otherwise returns [] if the format does not match.
    """
    expr = expr.strip()

    # 1) Numeric-range star pattern: 1*(2+3+...+10)
    # Corrected: Replaced '' with '\(' and '\)' for literal parentheses.
    range_pattern = r"^([A-Za-z0-9]+)\*\(\s*(\d+)\+(\d+)\.\.\.(\d+)\s*\)$"
    m = re.match(range_pattern, expr)
    if m:
        center_label = m.group(1)           # e.g. "1" or "Node42"
        start_int    = int(m.group(2))      # e.g. 2
        end_int      = int(m.group(4))      # e.g. 10
        # Build leaves 2,3,...,10
        return [(center_label, str(i)) for i in range(start_int, end_int + 1)]

    # 2) Direct numeric list: 1*(2+3+4)
    # Corrected: Replaced '' with '\(' and '\)' for literal parentheses.
    direct_numeric_pattern = r"^(\d+)\*\(\s*(\d+(?:\s*\+\s*\d+)*)\s*\)$"
    m2 = re.match(direct_numeric_pattern, expr)
    if m2:
        center_label = m2.group(1)           # e.g. "1"
        leaves_part  = m2.group(2)           # e.g. "2+3+4"
        leaf_labels  = [tok.strip() for tok in leaves_part.split("+")]
        return [(center_label, leaf) for leaf in leaf_labels]

    # 3) Generic-label star: A*(B+C+D)
    # Corrected: Replaced '' with '\(' and '\)' for literal parentheses.
    generic_pattern = r"^([A-Za-z0-9]+)\*\(\s*([A-Za-z0-9]+(?:\s*\+\s*[A-Za-z0-9]+)*)\s*\)$"
    m3 = re.match(generic_pattern, expr)
    if m3:
        center_label = m3.group(1)           # e.g. "A" or "Node1"
        leaves_part  = m3.group(2)           # e.g. "B+C+D" or "Leaf1+Leaf2"
        leaf_labels  = [tok.strip() for tok in leaves_part.split("+")]
        return [(center_label, leaf) for leaf in leaf_labels]

    # If none of the above patterns matched, return empty
    return []

# --- Draw graph ---
if st.button("Draw Graph"):
    edges = parse_graph_expression(expr)
    if not edges:
        st.error("Invalid expression format. Please use formats like `1*(2+3+4)`, `x_1*(2+...+10)`, or `A*(B+C+D)`.")
    else:
        G = nx.Graph()
        G.add_edges_from(edges)
        pos = nx.spring_layout(G, seed=42) # Use a fixed seed for consistent layouts
        fig, ax = plt.subplots() # Create a Matplotlib figure and axes
        nx.draw(G, pos, with_labels=True, node_color="lightblue", font_weight="bold", ax=ax)
        st.pyplot(fig) # Display the Matplotlib figure in Streamlit

