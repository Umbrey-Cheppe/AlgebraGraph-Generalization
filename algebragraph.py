import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="Algebraic Graph Visualizer", layout="centered")
st.title("Algebraic Graph Visualizer")
st.markdown("Enter a symbolic expression like `1*(2+3+...+10)` to generate a star graph.")

expr = st.text_input("Graph Expression:", "1*(2+3+...+10)")

import re

def parse_graph_expression(expr):
    """
    Returns a list of (center, leaf) pairs for expressions like:
      1) Range:        "1*(2+3+...+10)"
      2) Direct list:  "1*(2+3+4)"
      3) Alphanumeric with _: "x_1*(x_2+x_3+...+x_n)"
    """
    expr = expr.strip()

    # Pattern 1: Numeric range (e.g., 1*(2+3+...+10) or x_1*(2+3+...+10))
    # Allows alphanumeric characters and underscores for the center label.
    range_pattern = r"^([A-Za-z0-9_]+)\*\(\s*(\d+)\+\d+\.\.\.(\d+)\s*\)$"
    m = re.match(range_pattern, expr)
    if m:
        center_label = m.group(1)
        start_int    = int(m.group(2))
        end_int      = int(m.group(3))
        # Ensure the range is valid (start <= end)
        if start_int > end_int:
            return []
        return [(center_label, str(i)) for i in range(start_int, end_int + 1)]

    # Pattern 2: Direct numeric list (e.g., 1*(2+3+4))
    # Specifically for numeric center labels and numeric leaves.
    numeric_list_pattern = r"^(\d+)\*\(\s*(\d+(?:\s*\+\s*\d+)*)\s*\)$"
    m2 = re.match(numeric_list_pattern, expr)
    if m2:
        center_label = m2.group(1)
        leaf_labels  = [tok.strip() for tok in m2.group(2).split("+")]
        return [(center_label, leaf) for leaf in leaf_labels]

    # Pattern 3: Alphanumeric/underscore labels (e.g., A*(B+C+D) or x_1*(x_2+x_3))
    # Catches general alphanumeric and underscore labels for both center and leaves.
    label_pattern = r"^([A-Za-z0-9_]+)\*\(\s*([A-Za-z0-9_]+(?:\s*\+\s*[A-Za-z0-9_]+)*)\s*\)$"
    m3 = re.match(label_pattern, expr)
    if m3:
        center_label = m3.group(1)
        leaf_labels  = [tok.strip() for tok in m3.group(2).split("+")]
        return [(center_label, leaf) for leaf in leaf_labels]

    # If no pattern matches, return an empty list
    return []







if st.button("Draw Graph"):
    edges = parse_graph_expression(expr)
    if not edges:
        st.error("Invalid expression format. Use format 1*(2+3+4) or 1*(2+3+...+10)")
    else:
        G = nx.Graph()
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", font_weight="bold")
        st.pyplot(plt)
