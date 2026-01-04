## **What Graph RAG Does**

Converts your dimensional data model (fact + dimension tables) into a knowledge graph, then uses it to answer natural language questions, for example instead of writing SQL, users ask: *"Which customers who bought Product A also bought Product B?"*

## **Step-by-Step Process**

---

### **Build the Knowledge Graph** 
*Transform tables into a network of connected entities* The dimensional model (tables with rows/columns) becomes a *graph* (nodes connected by edges). 

**Current Tables**:
```
CUSTOMERS table:
customer_id | name        | segment     | region
C1          | Acme Corp   | Enterprise  | West
C2          | Tech Inc    | SMB         | East

PRODUCTS table:
product_id  | name        | category    | price
P1          | Widget      | Hardware    | 99.99
P2          | Gadget      | Software    | 49.99

SALES_FACTS table:
sale_id | customer_id | product_id | revenue | quantity
S1      | C1          | P1         | 999.90  | 10
S2      | C1          | P2         | 249.95  | 5
S3      | C2          | P1         | 499.95  | 5
```

**Becomes This Graph**:
```
    [Customer: Acme Corp]
           ↓ bought (revenue=999.90, qty=10)
    [Product: Widget]
           ↓ also bought by
    [Customer: Tech Inc]
           
    [Customer: Acme Corp]
           ↓ bought (revenue=249.95, qty=5)
    [Product: Gadget]
```

#### In Code:
```python
# Dataframes
customers_df = spark.table("catalog.schema.customers").toPandas()
products_df = spark.table("catalog.schema.products").toPandas()
sales_df = spark.table("catalog.schema.sales_facts").toPandas()

# Build graph
graph = nx.MultiDiGraph()

# Add customers as nodes
for row in customers_df.iterrows():
    graph.add_node(
        f"Customer_{row['customer_id']}", 
        name=row['name'],
        segment=row['segment']
    )

# Add products as nodes
for row in products_df.iterrows():
    graph.add_node(
        f"Product_{row['product_id']}", 
        name=row['name'],
        category=row['category']
    )

# Add sales as edges (connections)
for row in sales_df.iterrows():
    graph.add_edge(
        f"Customer_{row['customer_id']}",
        f"Product_{row['product_id']}",
        revenue=row['revenue'],
        quantity=row['quantity']
    )
```
Results graph of nodes i.e., Customers/Products and edges Purchase relationships with revenue/quantity

---

### **Generate Embeddings**
*Create "semantic fingerprints" for every entity* Each node (customer, product) gets converted into a vector (list of numbers) that represents its meaning.So, we can find similar entities using math, not exact keyword matching. For example:
```
"Customer: Acme Corp, segment=Enterprise, region=West"
```
**Becomes Vector**:
```python
[0.23, -0.15, 0.87, 0.44, -0.92, ...] 
```
**Product Node Text**:
```
"Product: Widget, category=Hardware, price=99.99"
```
**Becomes Vector**:
```python
[0.19, -0.21, 0.91, 0.38, -0.88, ...] # 384 numbers
```
#### In Code:
```python
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all nodes
node_embeddings = {}
for node_id, node_data in graph.nodes(data=True):
    text = node_data['text']  # "Customer: Acme Corp, segment=Enterprise"
    embedding = model.encode(text)  # [0.23, -0.15, 0.87, ...]
    node_embeddings[node_id] = embedding
```
With that, every customer and product now has a semantic fingerprint.

---

### **Semantic Search**
*Find relevant entities for the user's question* User asks a question -> Convert question to vector -> Find nodes with similar vectors. For example:
```
"Show me enterprise customers"
```
**Process**:
1. Convert question to vector: `[0.21, -0.18, 0.85, ...]`
2. Compare to all node vectors using cosine similarity
3. Find top 5 most similar nodes

**Similarity Calculation**:
```python
Question embedding:     [0.21, -0.18, 0.85, ...]
Customer_C1 embedding:  [0.23, -0.15, 0.87, ...]
Similarity score:       0.92 (very similar)

Question embedding:     [0.21, -0.18, 0.85, ...]
Product_P1 embedding:   [0.19, -0.21, 0.91, ...]
Similarity score:       0.45 (not very similar)
```

**Results**:
```
1. Customer_C1 (Acme Corp) - similarity: 0.92
2. Customer_C5 (Big Corp) - similarity: 0.88
3. Customer_C9 (Mega Inc) - similarity: 0.85
```

#### In Code:
```python
from sklearn.metrics.pairwise import cosine_similarity

# Convert question to embedding
question = "Show me enterprise customers"
question_embedding = model.encode([question])[0]

# Find similar nodes
similarities = {}
for node_id, node_emb in node_embeddings.items():
    sim = cosine_similarity([question_embedding], [node_emb])[0][0]
    similarities[node_id] = sim

# Get top 5
top_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
# Result: [('Customer_C1', 0.92), ('Customer_C5', 0.88), ...]
```

Results the 5 most semantically relevant entities.

---

### **Graph Traversal**
*Expand context by following connections* Starting from the top 5 nodes found in Step 3, walk the graph to find related entities. This is to get the full picture - not just customers, but what they bought, when, etc. For example:

**Starting Nodes**:
```
Customer_C1 (Acme Corp)
Customer_C5 (Big Corp)
```
**1-Hop Traversal** (immediate connections):
```
Customer_C1 → bought → Product_P1 (Widget)
Customer_C1 → bought → Product_P2 (Gadget)
Customer_C5 → bought → Product_P1 (Widget)
Customer_C5 → bought → Product_P3 (Doohickey)
```
**2-Hop Traversal** (follow connections further):
```
Product_P1 → bought by → Customer_C2 (Tech Inc)
Product_P1 → bought by → Customer_C8 (Small Co)
```
**Final Context Subgraph**:
```
[Customer_C1] ←→ [Product_P1] ←→ [Customer_C2]
      ↓                ↑
[Product_P2]    [Customer_C5]
      ↑                ↓
[Customer_C8] ←→ [Product_P3]
```
#### In Code:
```python
# Start with top nodes
context_nodes = set(['Customer_C1', 'Customer_C5'])

# Expand via graph traversal (2 hops)
for hop in range(2):
    new_nodes = set()
    for node in context_nodes:
        # Get neighbors (connected nodes)
        neighbors = graph.neighbors(node)
        new_nodes.update(neighbors)
    context_nodes.update(new_nodes)

# Create subgraph with all context nodes
subgraph = graph.subgraph(context_nodes)
```
---

### **Generate Answer**
*Extract facts and create natural language response* Extract structured facts from the subgraph → Format into natural language. For example:

**Facts Extracted from Subgraph**:
```
Entities:
- Customer_C1: Acme Corp (Enterprise, West)
- Customer_C5: Big Corp (Enterprise, East)
- Product_P1: Widget (Hardware, $99.99)
- Product_P2: Gadget (Software, $49.99)

Relationships:
- Customer_C1 bought Product_P1 (revenue=$999.90, qty=10)
- Customer_C1 bought Product_P2 (revenue=$249.95, qty=5)
- Customer_C5 bought Product_P1 (revenue=$499.95, qty=5)
```
**Generated Answer**
```
"Enterprise customers include:

1. Acme Corp (West region)
   - Purchased Widget for $999.90 (10 units)
   - Purchased Gadget for $249.95 (5 units)
   - Total spend: $1,249.85

2. Big Corp (East region)
   - Purchased Widget for $499.95 (5 units)
   - Total spend: $499.95

Both customers show preference for hardware products, 
with Widget being the most popular item."
```
Results natural language answer to the user's question.
