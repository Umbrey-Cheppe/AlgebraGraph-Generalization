import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="Algebraic Graph Visualizer", layout="centered")
st.title("Algebraic Graph Visualizer")
st.markdown("Enter a symbolic expression like `1*(2+3+...+10)` to generate a star graph.")

expr = st.text_input("Graph Expression:", "1*(2+3+...+10)")

def parse_graph_expression(expr):
    try:
        center, rest = expr.split("*")
        center = center.strip()
        rest = rest.strip("()+ ")
        
        # Handle range expansion like 2+3+...+10
        if "..." in rest:
            parts = rest.split("...")
            start_part = parts[0].split("+")[-1].strip()
            end_part = parts[1].strip()
            start = int(start_part)
            end = int(end_part)
            nodes = list(range(start, end + 1))
        else:
            nodes = [int(x.strip()) for x in rest.split("+")]

        edges = [(int(center), node) for node in nodes]
        return edges
    except Exception as e:
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
