import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import re

# --- Streamlit page setup ---
st.set_page_config(page_title="Algebraic Graph Visualizer", layout="centered")
st.title("Algebraic Graph Visualizer")
st.markdown("""
Enter a symbolic expression like `1*(2+3+...+10)`, `x_1*(x_2+x_3+x_4)`, or `A*(B+C+D)` to generate a star graph.
""")

# --- Input field ---
expr = st.text_input("Graph Expression:", "1*(2+3+...+10)")

# --- Expression parser ---
def parse_graph_expression(expr):
    expr = expr.strip()

    # --- Pattern 1: Numeric range like 1*(2+...+10) or x_1*(2+...+10) ---
    # The regex now correctly looks for literal parentheses '(' and ')'
    # using '\(' and '\)'. It also handles both '...' and '…' for ellipsis.
    range_pattern = r"^([A-Za-z0-9_]+)\*\(\s*(\d+)\s*\+\s*(?:\.\.\.|…)\s*(\d+)\s*\)$"
    match = re.match(range_pattern, expr)
    if match:
        center = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3))
        if start > end:
            return []  # Return empty if the range is invalid
        return [(center, str(i)) for i in range(start, end + 1)]

    # --- Pattern 2: Direct numeric list like 1*(2+3+4) ---
    # The regex now correctly looks for literal parentheses '(' and ')'
    numeric_list_pattern = r"^(\d+)\*\(\s*(\d+(?:\s*\+\s*\d+)*)\s*\)$"
    match = re.match(numeric_list_pattern, expr)
    if match:
        center = match.group(1)
        leaves = [x.strip() for x in match.group(2).split("+")]
        return [(center, leaf) for leaf in leaves]

    # --- Pattern 3: Alphanumeric/underscore label list like A*(B+C+D) or x_1*(x_2+x_3) ---
    # The regex now correctly looks for literal parentheses '(' and ')'
    label_pattern = r"^([A-Za-z0-9_]+)\*\(\s*([A-Za-z0-9_]+(?:\s*\+\s*[A-Za-z0-9_]+)*)\s*\)$"
    match = re.match(label_pattern, expr)
    if match:
        center = match.group(1)
        leaves = [x.strip() for x in match.group(2).split("+")]
        return [(center, leaf) for leaf in leaves]

    # If no pattern matches
    return []

# --- Graph visualization ---
if st.button("Draw Graph"):
    edges = parse_graph_expression(expr)
    if not edges:
        st.error("Invalid expression format. Please use formats like `1*(2+3+4)`, `1*(2+3+...+10)`, or `A*(B+C+D)`.")
    else:
        G = nx.Graph()
        G.add_edges_from(edges)
        pos = nx.spring_layout(G, seed=42)  # Use a fixed seed for consistent layouts
        fig, ax = plt.subplots()  # Create a Matplotlib figure and axes
        nx.draw(G, pos, with_labels=True, node_color="skyblue", font_weight="bold", ax=ax)
        st.pyplot(fig)  # Display the Matplotlib figure in Streamlit
