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

# --- Parsing logic ---
def parse_graph_expression(expr):
    """
    Returns a list of (center, leaf) pairs for expressions like:
      1) Range:        "1*(2+3+...+10)"
      2) Direct list:  "1*(2+3+4)"
      3) Alphanumeric with underscores: "x_1*(x_2+x_3+...+x_n)"
    """
    expr = expr.strip()

    # Pattern 1: Numeric range (e.g., 1*(2+...+10) or x_1*(2+...+10))
    # Matches center_label*(start_num+...+end_num)
    # Allows alphanumeric and underscores for the center label.
    # Corrected: Replaced '' with escaped parentheses '\(' and '\)'
    range_pattern = r"^([A-Za-z0-9_]+)\*\(\s*(\d+)\s*\+\s*\.\.\.\s*(\d+)\s*\)$"
    match = re.match(range_pattern, expr)
    if match:
        center = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3))
        # Ensure the range is valid (start must be less than or equal to end)
        if start > end:
            return []
        return [(center, str(i)) for i in range(start, end + 1)]

    # Pattern 2: Direct numeric list (e.g., 1*(2+3+4))
    # Matches numeric_center_label*(num1+num2+...)
    # Corrected: Replaced '' with escaped parentheses '\(' and '\)'
    numeric_list_pattern = r"^(\d+)\*\(\s*(\d+(?:\s*\+\s*\d+)*)\s*\)$"
    match = re.match(numeric_list_pattern, expr)
    if match:
        center = match.group(1)
        leaves = [x.strip() for x in match.group(2).split("+")]
        return [(center, leaf) for leaf in leaves]

    # Pattern 3: Alphanumeric/underscore labels (e.g., A*(B+C), x_1*(x_2+x_3))
    # Matches alphanumeric_center_label*(label1+label2+...)
    # Allows alphanumeric and underscores for both center and leaf labels.
    # Corrected: Replaced '' with escaped parentheses '\(' and '\)'
    label_pattern = r"^([A-Za-z0-9_]+)\*\(\s*([A-Za-z0-9_]+(?:\s*\+\s*[A-Za-z0-9_]+)*)\s*\)$"
    match = re.match(label_pattern, expr)
    if match:
        center = match.group(1)
        leaves = [x.strip() for x in match.group(2).split("+")]
        return [(center, leaf) for leaf in leaves]

    # If no pattern matches, return an empty list
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

