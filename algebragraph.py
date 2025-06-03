import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import re # Still useful for a cleaner initial split

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Algebraic Graph Visualizer", layout="centered")
st.title("Algebraic Graph Visualizer")
st.markdown("""
Enter a symbolic expression like `1*(2+3+...+10)`, `x_1*(x_2+x_3+x_4)`, or `A*(B+C+D)` to generate a star graph.
""")

# --- Input Field ---
expr = st.text_input("Graph Expression:", "1*(2+3+...+10)")

# --- Expression Parser ---
def parse_graph_expression(expr):
    """
    Returns a list of (center, leaf) pairs for expressions like:
      1) Range:        "1*(2+3+...+10)" or "x_1*(2+3+...+10)"
      2) Direct list:  "1*(2+3+4)"
      3) Alphanumeric: "A*(B+C+D)" or "x_1*(x_2+x_3+x_4)"
    """
    expr = expr.strip()

    # Regex for a more robust initial split, allowing for alphanumeric/underscore centers
    # and correctly handling the parenthesis for the leaf part.
    pattern = r"^([A-Za-z0-9_]+)\*\((.*)\)$"
    match = re.match(pattern, expr)

    if not match:
        return [] # Does not match the basic 'center*(leaves)' structure

    center_label = match.group(1).strip()
    leaves_part = match.group(2).strip()

    try:
        # Pattern 1: Numeric range (e.g., 1*(2+3+...+10) or x_1*(2+3+...+10))
        # Supports both "..." and "…" as ellipsis
        if "..." in leaves_part or "…" in leaves_part:
            # Handle both '...' and '…'
            parts = re.split(r'\.\.\.|…', leaves_part)
            if len(parts) != 2: # Must have exactly two parts split by ellipsis
                return []
            
            start_str = parts[0].strip()
            end_str = parts[1].strip()

            # The start can be like "2" or "2+3". We only care about the first number.
            try:
                start = int(start_str.split('+')[0].strip())
                end = int(end_str)
            except ValueError:
                return [] # Not valid numbers

            if start > end:
                return [] # Invalid range
            
            # Decide if center is numeric or alphanumeric for consistency in output
            if center_label.isdigit():
                return [(int(center_label), str(i)) for i in range(start, end + 1)]
            else:
                return [(center_label, str(i)) for i in range(start, end + 1)]

        # Pattern 2 & 3: Direct list (numeric or alphanumeric/underscore)
        # Split by '+'
        leaf_labels_raw = leaves_part.split("+")
        leaf_labels = [tok.strip() for tok in leaf_labels_raw]

        edges = []
        # Check if all leaves are digits for potential integer conversion
        all_leaves_are_digits = all(leaf.isdigit() for leaf in leaf_labels)

        # Decide on the type conversion for center and leaves based on content
        if center_label.isdigit():
            # If center is numeric, try to convert it to int
            parsed_center = int(center_label)
            if all_leaves_are_digits:
                # If both center and leaves are numeric, convert both to int
                edges = [(parsed_center, int(leaf)) for leaf in leaf_labels]
            else:
                # If center is numeric but leaves are alphanumeric, keep leaves as strings
                edges = [(parsed_center, leaf) for leaf in leaf_labels]
        else:
            # If center is alphanumeric, keep center as string
            # And leaves remain as strings (they can be alphanumeric or numeric strings)
            edges = [(center_label, leaf) for leaf in leaf_labels]
            
        return edges

    except Exception as e:
        # Catch any unexpected errors during parsing and return empty
        # st.error(f"Debugging parsing error: {e}") # Uncomment for debugging
        return []

# --- Graph Drawing ---
if st.button("Draw Graph"):
    edges = parse_graph_expression(expr)
    if not edges:
        st.error("❌ Invalid expression format. Please use formats like `1*(2+3+4)`, `1*(2+3+...+10)`, `x_1*(x_2+x_3)`, or `A*(B+C+D)`.")
    else:
        G = nx.Graph()
        # Add edges, ensuring consistent data types (int or str)
        # for nodes based on parsing. NetworkX handles mixed types well.
        G.add_edges_from(edges)
        
        pos = nx.spring_layout(G, seed=42) # Use a fixed seed for consistent layouts
        
        fig, ax = plt.subplots() # Create a Matplotlib figure and axes for drawing
        nx.draw(G, pos, with_labels=True, node_color="skyblue", font_weight="bold", ax=ax)
        st.pyplot(fig) # Display the figure in Streamlit

