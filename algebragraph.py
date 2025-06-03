import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import re

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Algebraic Graph Visualizer", layout="centered")
st.title("Algebraic Graph Visualizer")
st.markdown("""
Enter a symbolic expression like `1*(2+3+...+10)`, `x_1*(x_2+x_3+x_4)`, or `A*(B+C+D)` to generate a star graph.
""")

# --- Input ---
expr = st.text_input("Graph Expression:", "1*(2+3+...+10)")

# --- Expression Parser ---
def parse_graph_expression(expr):
    expr = expr.strip()

    # 1. Match range pattern: 1*(2+3+...+10) or x_1*(2+3+...+10)
    # Corrected: Replaced '' with escaped parentheses '\(' and '\)'
    # Also adjusted regex for range: 'start+...end' or 'start+any_number+...+end'
    # Simplified to just 'start+...+end' for clarity, as 'any_number' before '...'
    # is often implicit or not strictly required for the *range* interpretation.
    # The (?:\\d+\\+)? was making it stricter than necessary for typical range notation.
    range_pattern = r"^([A-Za-z0-9_]+)\*\(\s*(\d+)\s*\+\s*\.\.\.\s*(\d+)\s*\)$"
    m = re.match(range_pattern, expr)
    if m:
        center = m.group(1)
        start = int(m.group(2))
        end = int(m.group(3))
        if start > end:
            return [] # Invalid range
        return [(center, str(i)) for i in range(start, end + 1)]

    # 2. Match direct numeric list: 1*(2+3+4)
    # Corrected: Replaced '' with escaped parentheses '\(' and '\)'
    direct_numeric = r"^(\d+)\*\(\s*(\d+(?:\s*\+\s*\d+)*)\s*\)$"
    m2 = re.match(direct_numeric, expr)
    if m2:
        center = m2.group(1)
        leaves = [x.strip() for x in m2.group(2).split("+")]
        return [(center, leaf) for leaf in leaves]

    # 3. Match generic alphanumeric list: x_1*(x_2+x_3+x_4), A*(B+C+D)
    # Corrected: Replaced '' with escaped parentheses '\(' and '\)'
    label_pattern = r"^([A-Za-z0-9_]+)\*\(\s*([A-Za-z0-9_]+(?:\s*\+\s*[A-Za-z0-9_]+)*)\s*\)$"
    m3 = re.match(label_pattern, expr)
    if m3:
        center = m3.group(1)
        leaves = [x.strip() for x in m3.group(2).split("+")]
        return [(center, leaf) for leaf in leaves]

    return []

# --- Graph Drawing ---
if st.button("Draw Graph"):
    edges = parse_graph_expression(expr)
    if not edges:
        st.error("❌ Invalid expression format. Try formats like `1*(2+3+4)`, `1*(2+3+...+10)`, or `x_1*(x_2+x_3)`.")
    else:
        G = nx.Graph()
        G.add_edges_from(edges)
        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots() # Create a figure and an axes object
        nx.draw(G, pos, with_labels=True, node_color="skyblue", font_weight="bold", ax=ax)
        st.pyplot(fig) # Pass the figure object to st.pyplot()
