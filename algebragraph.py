import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="Algebraic Graph Visualizer", layout="centered")
st.title("Algebraic Graph Visualizer")

expr = st.text_input("Enter a symbolic expression like 1*(2+3+...+10)")

def parse_expression(expr):
    try:
        center, rest = expr.split('*')
        center = center.strip()
        rest = rest.strip('()+')
        nodes = rest.split('+')
        edges = [(center, node) for node in nodes]
        return edges
    except:
        return []

if st.button("Draw Graph"):
    edges = parse_expression(expr)
    if not edges:
        st.error("Invalid expression format. Use format 1*(2+3+4)")
    else:
        G = nx.Graph()
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", font_weight="bold")
        st.pyplot(plt)
