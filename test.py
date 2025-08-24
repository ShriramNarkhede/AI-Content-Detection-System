from graphviz import Digraph

# Create a use case diagram for placing an online order in a sales company
dot = Digraph("UseCaseDiagram", format="png")

# Define actors
dot.node("customer", "Customer", shape="ellipse")

# Define use cases
dot.node("browse", "Browse Products", shape="ellipse")
dot.node("select", "Select Product", shape="ellipse")
dot.node("cart", "Add to Cart", shape="ellipse")
dot.node("checkout", "Checkout", shape="ellipse")
dot.node("payment", "Make Payment", shape="ellipse")
dot.node("confirm", "Receive Confirmation", shape="ellipse")

# Relationships between actor and use cases
dot.edge("customer", "browse")
dot.edge("customer", "select")
dot.edge("customer", "cart")
dot.edge("customer", "checkout")
dot.edge("customer", "payment")
dot.edge("customer", "confirm")

# Create system boundary (Sales System)
with dot.subgraph(name="cluster_system") as c:
    c.attr(label="Sales System")
    c.attr(style="dashed")
    c.node("browse")
    c.node("select")
    c.node("cart")
    c.node("checkout")
    c.node("payment")
    c.node("confirm")

# Render the diagram
file_path = "/mnt/data/use_case_online_order"
dot.render(file_path, cleanup=True)

file_path + ".png"
