import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import re
from itertools import combinations

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Algebraic Graph Visualizer", layout="centered")
st.title("🌟 Algebraic Graph Visualizer (General Graphs) 🌟")
st.markdown("""
Enter a symbolic expression using:
- `*` for **clique/complete subgraph** (e.g., `a*b` for edge, `1*2*3` for triangle). Order doesn't matter.
- `+` for **graph union** (e.g., `(a*b)+(c*d)`).
- **Star Graph** syntax (e.g., `Center*(Leaf1+Leaf2)` or `1*(2+3+...+10)`). This is internally converted to a union of edges.

**Examples:**
- `a*b*c` (triangle on a,b,c)
- `(1*2)+(3*4)` (two separate edges)
- `1*(2+3+4)` (star graph with 1 as center)
- `A*(B+C+...+E)` (range-based star graph)
- `(a*b*c) + (a*b)` (should simplify to `a*b*c` due to absorption)
- `(1*2) + (1*3)` (two edges (1,2) and (1,3))
""")

# --- Input Field ---
expr = st.text_input("Graph Expression:", "(a*b*c) + (a*b)", help="Try different formats!")

# --- Customization Options ---
st.sidebar.header("Graph Customization")
node_color = st.sidebar.color_picker("Node Color", "#ADD8E6")
edge_color = st.sidebar.color_picker("Edge Color", "#808080")
font_color = st.sidebar.color_picker("Label Color", "#333333")
node_size = st.sidebar.slider("Node Size", 100, 5000, 2000)
font_size = st.sidebar.slider("Font Size", 8, 24, 12)

# --- Algebraic Graph Parser ---

def parse_nodes(node_str):
    """Converts a node string to int if numeric, else keeps as string."""
    return int(node_str) if node_str.isdigit() else node_str

def generate_clique_edges(nodes_list):
    """Generates edges for a complete graph (clique) given a list of nodes."""
    if len(nodes_list) < 2:
        return set() # A single node or empty doesn't form an edge
    
    clique_edges = set()
    # Use combinations to get all unique pairs for a complete graph
    for u, v in combinations(sorted(nodes_list, key=str), 2): # Sort for consistent edge representation (u,v) where u<v
        clique_edges.add(tuple(sorted((u, v), key=str))) # Ensure (u,v) not (v,u)
    return clique_edges

def parse_star_graph_to_edges(star_expr):
    """
    Parses a star graph expression and converts it into a set of (center*leaf) edges.
    e.g., "1*(2+3+4)" -> {(1,2), (1,3), (1,4)}
    e.g., "x_1*(x_2+...+x_5)" -> {(x_1, x_2), ..., (x_1, x_5)}
    """
    star_pattern = r"^([A-Za-z0-9_]+)\*\((.*)\)$"
    match = re.match(star_pattern, star_expr)

    if not match:
        return set()

    center_label = parse_nodes(match.group(1).strip())
    leaves_part = match.group(2).strip()
    
    star_edges = set()

    # Attempt to parse as a range first
    # Numeric Range (e.g., 2+3+...+10)
    numeric_range_pattern = r"^(\d+)\s*\+\s*(?:\.\.\.|…)\s*(\d+)$"
    match_numeric_range = re.match(numeric_range_pattern, leaves_part)

    if match_numeric_range:
        start = int(match_numeric_range.group(1))
        end = int(match_numeric_range.group(2))
        if start <= end:
            for i in range(start, end + 1):
                star_edges.add(tuple(sorted((center_label, str(i)), key=str)))
        return star_edges

    # Alphanumeric Range (e.g., x_2+...+x_5)
    alphanum_range_pattern = r"^([A-Za-z_]+)(\d+)\s*\+\s*(?:\.\.\.|…)\s*([A-Za-z_]+)(\d+)$"
    match_alphanum_range = re.match(alphanum_range_pattern, leaves_part)

    if match_alphanum_range:
        start_prefix = match_alphanum_range.group(1)
        start_num = int(match_alphanum_range.group(2))
        end_prefix = match_alphanum_range.group(3)
        end_num = int(match_alphanum_range.group(4))

        if start_prefix == end_prefix and start_num <= end_num:
            for i in range(start_num, end_num + 1):
                star_edges.add(tuple(sorted((center_label, f"{start_prefix}{i}"), key=str)))
        return star_edges

    # Direct list of leaves (e.g., 2+3+4 or B+C+D)
    leaf_labels = [parse_nodes(tok.strip()) for tok in leaves_part.split("+") if tok.strip()]
    for leaf in leaf_labels:
        star_edges.add(tuple(sorted((center_label, leaf), key=str)))
        
    return star_edges


def parse_algebraic_expression_to_edges(expr):
    """
    Parses the full algebraic expression into a set of raw edges.
    Handles *, +, and parentheses.
    """
    expr = expr.replace(" ", "").strip() # Remove all spaces

    # Convert star graph syntax to internal (center*leaf) + (center*leaf) form
    # This must be done before general '+' splitting
    star_match = re.match(r"^([A-Za-z0-9_]+)\*\((.*)\)$", expr)
    if star_match:
        # If it's a star graph, directly parse its edges
        return parse_star_graph_to_edges(expr)


    # Regular expression to split by '+' operator,
    # being careful not to split inside parentheses.
    # This is a common challenge for basic regex parsers.
    # A more robust solution might use a Shunting-yard algorithm or similar.
    # For now, let's assume terms are wrapped in parentheses if they contain '+'
    # and are part of a larger union.
    
    # Split by '+' where '+' is NOT inside a pair of parentheses
    # This regex is an attempt to handle basic (A+B)+(C*D) unions,
    # but still imperfect for deeply nested or complex cases without a full parser.
    # It tries to split based on '+', respecting parentheses that contain other operations.
    terms = re.split(r'\)\s*\+\s*\(', expr)
    
    # If no complex union (no ')+(' pattern), treat the whole thing as one term or a simple union
    if len(terms) == 1 and '(' not in terms[0] and ')' not in terms[0] and '+' in terms[0]:
        # Handle simple A+B+C (assuming it's a union of single nodes or simple cliques)
        # This is ambiguous in your algebra. Let's assume (A)+(B)+(C) or (A*B)+(C)
        # For simplicity, if it's A+B without parens, we'll try to parse each as a separate item.
        # This needs more formal definition. For now, we'll stick to parsing A*B or (A*B)+(C*D)
        pass # Fall through to handling single terms if it's not a complex union

    all_raw_edges = set()

    # Re-process terms to handle potential partial parentheses if they were split
    processed_terms = []
    if len(terms) > 1: # Indicates a union was detected
        for i, term in enumerate(terms):
            clean_term = term.strip()
            if i > 0 and not clean_term.startswith('('):
                clean_term = '(' + clean_term
            if i < len(terms) - 1 and not clean_term.endswith(')'):
                clean_term = clean_term + ')'
            processed_terms.append(clean_term)
    else:
        processed_terms = [expr] # No complex union, treat as single term

    
    for term in processed_terms:
        term = term.strip().strip('()') # Remove outermost parentheses if present

        # Split by '*' to get nodes for a clique
        clique_nodes_str = [node.strip() for node in term.split('*') if node.strip()]
        
        if not clique_nodes_str:
            st.warning(f"💡 Could not parse term: `{term}`. Check clique or union format.")
            return set() # Indicate parse failure for the term

        clique_nodes = [parse_nodes(n) for n in clique_nodes_str]
        
        # Generate all edges for this clique
        clique_edges = generate_clique_edges(clique_nodes)
        all_raw_edges.update(clique_edges)

    return all_raw_edges

def simplify_edges_with_absorption(edges_set):
    """
    Applies the absorption law (G_sup + G_sub = G_sup) by filtering out
    edges/cliques that are subgraphs of larger cliques present.
    This is a heuristic for path absorption `a*b*c + a*b = a*b*c`.
    
    This is a challenging problem without a full graph representation for each term.
    A simple approach: if a set of nodes forms a larger clique, and another set forms
    a sub-clique (or edge), the smaller one is absorbed.
    
    For example: if {(a,b), (b,c), (a,c)} (from a*b*c) exists, and {(a,b)} (from a*b) exists,
    then (a,b) should be absorbed.
    """
    
    # Convert edges to a list for iteration
    current_edges = list(edges_set)
    simplified_edges = set(edges_set) # Start with all unique edges

    # Sort edges for consistent comparison (e.g., ('a', 'b') is same as ('b', 'a'))
    # Ensure they are consistently ordered tuples
    normalized_edges = {tuple(sorted(edge, key=str)) for edge in edges_set}

    # This is a very simplified absorption logic for *cliques* and *sub-cliques/edges*.
    # A true "path absorption" (e.g., a*b*c absorbing a*b) requires path detection.
    # Given a*b*c is a clique, then a*b is one of its edges, so the (a,b) edge is part of the (a,b,c) clique.
    # We essentially only keep the largest cliques defined.
    
    # Store sets of nodes that form cliques (e.g., {'a', 'b', 'c'} from a*b*c)
    # and their generated edges.
    
    # To implement absorption, we need to detect the original "terms"
    # and then check if the edges generated by one term are a subset of another.
    # This function would need to be integrated higher up, perhaps at the AST level,
    # or by storing the *original node sets* that formed each clique.

    # For now, if we have (a,b), (a,c), (b,c) AND (a,b), the set() conversion already
    # prevents duplicates. The 'absorption' here refers to the *conceptual* removal
    # of the term `a*b` from the expression `(a*b*c) + (a*b) = a*b*c`.
    # This means the parser should *only* return edges from `a*b*c`.

    # Let's revisit this by ensuring the *parser* directly applies the absorption.
    # The parser should create a list of *clique node sets* from each term.
    # Then, we apply absorption on *these sets*.

    # For now, the `parse_algebraic_expression_to_edges` will gather all edges,
    # and the absorption will need to be applied to the *terms* before combining their edges.
    
    # This function will be called with the *final* set of edges.
    # The absorption logic for `a*b*c + a*b = a*b*c` is best handled by finding all cliques
    # and then removing those cliques whose edges are subsets of other cliques' edges.
    
    # A simpler approach for the current implementation:
    # Collect all unique node sets that form cliques.
    # For each clique C, if there's another clique C_prime such that C_prime is a subset of C,
    # then C_prime is 'absorbed'.
    
    # This is getting complex without a full parser. Let's simplify the initial
    # absorption for this iteration:
    # If the input is exactly (a*b*c) + (a*b), we can hardcode for now.
    # A more general absorption requires recognizing terms and their graphs.
    
    # For now, the `parse_algebraic_expression_to_edges` produces the raw edges.
    # True absorption will require understanding the original terms.
    
    return simplified_edges # Placeholder for now, as direct edge-set absorption is hard here.


# --- Main Parsing Function ---
def parse_and_simplify_graph_expression(expr):
    """
    Combines parsing and a simplified absorption logic.
    """
    expr = expr.replace(" ", "").strip() # Remove all spaces

    # Regex to extract terms that are either:
    # 1. Star graph: CENTER*(LEAVES)
    # 2. Cliques: NODE*NODE...
    # 3. Parenthesized terms for union: (...)
    
    # This pattern tries to find individual terms that are either
    # a star graph, or a group in parentheses, or a simple clique/node string.
    # It attempts to split by '+' while respecting parentheses
    # This is still a heuristic for complex union expressions.
    
    # Let's try to tokenize and build from basic units
    
    # 1. Tokenization: Identify nodes, operators, and parentheses
    tokens = re.findall(r'([A-Za-z0-9_]+|\*|\+|\(|\.|\.\.\.)', expr) # Added 'dot' for ellipsis
    
    # Convert '...' to a single token
    i = 0
    cleaned_tokens = []
    while i < len(tokens):
        if tokens[i] == '.' and i + 2 < len(tokens) and tokens[i+1] == '.' and tokens[i+2] == '.':
            cleaned_tokens.append('...')
            i += 3
        else:
            cleaned_tokens.append(tokens[i])
            i += 1
    
    # Now, process tokens using a very basic precedence for * over +
    # This is effectively a simplified Shunting-Yard like approach for very specific patterns.
    
    # For this iteration, let's simplify how we tackle the specific example:
    # (a*b*c) + (a*b)
    # If the input contains '+', we'll split it into components.
    
    # First, handle the special case of 'Center*(...)' as a star graph.
    # The `parse_star_graph_to_edges` function handles this and converts it to a set of edges.
    star_graph_edges = parse_star_graph_to_edges(expr)
    if star_graph_edges:
        return star_graph_edges # If it's a star graph, directly return its edges

    # Now, handle general algebraic expressions (cliques and unions)
    # This needs a more formal parser structure or a very careful regex.
    # Let's simplify to: split by '+' outside of any parentheses.
    
    # Regex to split terms by '+' if it's NOT inside a pair of parentheses
    # (?: \([^()]*\) | [^()+]+ )+  -- matches balanced parentheses or non-operator sequences
    # Simplified regex for splitting by outer '+', still challenging.
    
    # Simple split by '+' and assume terms are either a*b*c or (a*b*c)
    terms = re.split(r'\s*\+\s*', expr)
    
    final_edges = set()
    all_clique_node_sets = [] # To store node sets from each clique term, for absorption check

    for term in terms:
        term = term.strip().strip('()') # Remove outer parentheses
        
        clique_nodes_str = [node.strip() for node in term.split('*') if node.strip()]
        
        if not clique_nodes_str:
            st.warning(f"💡 Could not parse term: `{term}`. Check clique format or syntax.")
            return set() # Indicate failure for this term

        clique_nodes = [parse_nodes(n) for n in clique_nodes_str]
        
        # Store the set of nodes for this clique for later absorption check
        all_clique_node_sets.append(frozenset(clique_nodes)) # Use frozenset for immutability in a set

        # Generate edges for this clique
        clique_edges = generate_clique_edges(clique_nodes)
        final_edges.update(clique_edges) # Add to the overall set of edges (union)

    # --- Apply Absorption Law (a*b*c + a*b = a*b*c) ---
    # This involves checking if any smaller clique's node set is a subset of a larger clique's node set.
    # If so, the edges from the smaller clique would already be covered by the larger one.
    # Since `final_edges` already uses a set (handling `(a,b) + (a,b)`),
    # the absorption needs to filter out edges that belong *only* to an absorbed smaller clique.

    # We need to determine which 'terms' (cliques) are absorbed.
    unabsorbed_clique_node_sets = set(all_clique_node_sets)

    for c1_set in all_clique_node_sets:
        for c2_set in all_clique_node_sets:
            if c1_set != c2_set and c1_set.issubset(c2_set):
                # If c1_set is a proper subset of c2_set, then c1_set is absorbed by c2_set.
                # Remove c1_set from the set of unabsorbed cliques.
                if c1_set in unabsorbed_clique_node_sets: # Make sure it hasn't been removed by another absorption
                    unabsorbed_clique_node_sets.discard(c1_set)

    # Now, regenerate the final edges ONLY from the unabsorbed cliques
    simplified_final_edges = set()
    for clique_nodes_subset in unabsorbed_clique_node_sets:
        simplified_final_edges.update(generate_clique_edges(list(clique_nodes_subset)))
        
    return simplified_final_edges

# --- Graph Drawing ---
st.markdown("---")
if st.button("Draw Graph"):
    # Call the new combined parser and simplifier
    edges_to_draw = parse_and_simplify_graph_expression(expr)
    
    if not edges_to_draw:
        st.error("❌ Failed to parse or generate edges from expression. Please ensure it follows a supported format. Check hints above.")
        st.warning("Possible issues: Malformed ranges, empty terms, unhandled nesting.")
    else:
        # --- Filter out self-loops (u, u) ---
        # This is already handled by `generate_clique_edges` not adding (u,u) if len(nodes) < 2
        # However, a single node "A" for example won't have edges. We need to explicitly add nodes.
        
        final_edges_for_nx = []
        for u, v in edges_to_draw:
            if u != v: # Final check to ensure no self-loops, even if my logic missed one.
                final_edges_for_nx.append((u, v))
        
        if not final_edges_for_nx:
            # If all edges were self-loops and filtered, or nothing was generated
            st.info("The expression resulted in no distinct edges (possibly only self-loops, which are filtered based on `A*A=A` idempotence). No graph with edges to display.")
            st.warning("Consider an expression that creates distinct connections, like `1*(2+3)` or `a*b`.")
            
            # --- Ensure isolated nodes are still displayed ---
            # Extract all nodes from the original expression, even if they don't form edges.
            all_nodes_in_expr = set()
            try:
                # Find all potential nodes in the expression (very broad search)
                node_candidates = re.findall(r'[A-Za-z0-9_]+', expr)
                for nc in node_candidates:
                    all_nodes_in_expr.add(parse_nodes(nc))
            except Exception:
                pass # Best effort to get nodes

            if all_nodes_in_expr:
                G_temp = nx.Graph()
                G_temp.add_nodes_from(list(all_nodes_in_expr))
                
                pos = nx.spring_layout(G_temp, seed=42)
                fig, ax = plt.subplots(figsize=(8, 6))
                nx.draw(G_temp, pos, with_labels=True, node_color=node_color, font_color=font_color,
                        font_weight="bold", node_size=node_size, font_size=font_size, ax=ax)
                ax.set_title(f"Graph for: `{expr}` (Isolated Nodes)", size=font_size + 4, color=font_color)
                st.pyplot(fig)
                
        else:
            G = nx.Graph()
            G.add_edges_from(final_edges_for_nx)
            
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
            
            ax.set_title(f"Graph for: `{expr}` (Self-loops filtered, absorbed terms simplified)", size=font_size + 4, color=font_color)
            st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.info("💡 This app visualizes general simple graphs based on algebraic expressions. Self-loops (e.g., `A*A` or `1*(1+...)`) are **filtered** due to `A*A=A` idempotence. Absorption law (`a*b*c + a*b = a*b*c`) is applied by filtering sub-cliques.")


