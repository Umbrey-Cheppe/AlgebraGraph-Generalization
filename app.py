import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from algebragraph import parse_graph_expression

st.set_page_config(page_title="Algebraic Graph Visualizer", layout="centered")

st.title("🔗 Algebraic Graph Visualizer")
st.markdown("Enter a symbolic expression like `1*(2+3+...+10)` to generate a star graph.")

# Input expression
expr = st.text_input("Graph Expression", "1*(2+3+...+10)")

if st.button("Draw Graph"):
    try:
        G = parse_graph_expression(expr)
        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=800, font_weight='bold', ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error: {e}")

  
