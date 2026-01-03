## **Using BGE-M3 with Graph RAG**

### **Model options and selection**

```python
dbutils.widgets.dropdown('embedding_model',
    defaultValue="all-MiniLM-L6-v2",
    choices=[
        "all-MiniLM-L6-v2",
        "BAAI/bge-m3",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
        "intfloat/e5-small-v2"
    ],
    label="embedding_model"
)
```
```python
# Embedding model settings
embedding_model=dbutils.widgets.get("embedding_model"),
top_k_nodes=5,                        # Top results to return
max_hops=2,                           # Graph traversal depth
date_column="sale_date"               # Date column in fact table
```

## BGE-M3 key features

### **Better Quality:**
```python
Query: "Show premium chocolate items with rich flavor"

# MiniLM Results:
1. Chocolate Delight (0.701)
2. Berry Surprise (0.543)      Not chocolate
3. Vanilla Dream (0.521)       Not chocolate

# BGE-M3 Results:
1. Chocolate Delight (0.891)   Higher score
2. Truffle Treat (0.867)       Better ranking
3. Fudge Fantasy (0.801)       All chocolate
```

### **Multi-Lingual Support:**
```python
# Works with international product names
system.query("Show me 巧克力 items")      # Chinese
system.query("Find Chocolat products")   # French
system.query("チョコレート製品")            # Japanese
```

### **Longer Context:**
```python
# Can handle full product descriptions
item_description = """
Premium dark chocolate bar made with 70% cacao from Ecuador.
Features smooth ganache filling with hints of vanilla and sea salt.
Hand-crafted in small batches using traditional methods.
Perfect for special occasions or as a gourmet gift.
"""
# MiniLM: Truncates at 256 tokens
# BGE-M3: Captures all 8192 tokens 
```
---

## **BGE-M3 vs all-MiniLM-L6-v2 Comparison**

| Feature | all-MiniLM-L6-v2 | BAAI/bge-m3 | Difference |
|---------|------------------|-------------|------------|
| Embedding Dimension | 384 | 1024 | 2.7x larger |
| Model Parameters | 22M | 568M | 25x more parameters |
| Semantic Quality | Good | Excellent | Better understanding |
| Multi-lingual | Limited | 139 languages | Much broader |
| Context Length | 256 tokens | 8192 tokens | 32x longer |
| Speed (CPU) | Fast (1-2 sec) | Slower (5-10 sec) | 5x slower |
| Speed (GPU) | Fast (0.2 sec) | Fast (0.5 sec) | Similar with GPU |
| Memory | Low (200MB) | High (2.3GB) | 11x more RAM |

### **BGE-M3 Specifications:**

```python
Model: BAAI/bge-m3
Publisher: Beijing Academy of AI (BAAI)
Architecture: BERT-based
Parameters: 568M
Embedding Dimension: 1024
Max Sequence Length: 8192 tokens
Languages: 139 (including English, Chinese, Japanese, French, German, etc.)
Training Data: Massive multi-lingual corpus
License: MIT
HuggingFace: https://huggingface.co/BAAI/bge-m3
```
---

## **Embedding Storage & Retrieval Flow**

### **1. During Build (Storage)**

```python
# Call system.build()
rag_system = GraphRAGSystem(config, spark)
rag_system.build()  # Embeddings created 
```
**It triggers:**

```python
# Create node text representations
for node_id, attrs in graph.nodes(data=True):
    node_text = "item_id: p-20 | item_name: Chocolate Delight | item_description: Rich..."
    node_texts[node_id] = node_text

# Generate embeddings (batch)
texts = ["item_id: p-20 | ...", "item_id: p-8 | ...", ...]
embeddings = model.encode(texts)  # → numpy arrays

# Store in dictionary (in-memory)
node_embeddings = {
    "item_details_p-20": array([0.23, -0.15, 0.87, ...]),  # 1024 numbers
    "item_details_p-8":  array([0.12, 0.45, -0.23, ...]),  # 1024 numbers
    "customer_details_c-1": array([-0.34, 0.67, 0.12, ...]), # 1024 numbers
    ...
}
```

**Storage Location:** Python dictionary in RAM 

```python
self.embedding_gen.node_embeddings = {
    'item_details_p-20': numpy.array([...]),  # Stored here
    'item_details_p-8': numpy.array([...]),
    ...
}
```

---

### **2. Query (Retrieval)**

```python
# Call system.query()
answer = rag_system.query("Show chocolate items")
```
**It triggers:**

```python
# Encode the query
query_text = "Show chocolate items"
query_embedding = model.encode([query_text])[0]  
# array([0.34, -0.23, 0.56, ...])  # 1024 numbers

# Compare with ALL stored embeddings
from sklearn.metrics.pairwise import cosine_similarity

similarities = []
for node_id, node_embedding in node_embeddings.items():
    # Calculate similarity (dot product / magnitude)
    sim = cosine_similarity([query_embedding], [node_embedding])[0][0]
    similarities.append((node_id, sim))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)
# Results:
# [('item_details_p-20', 0.891),    Chocolate Delight
#  ('item_details_p-35', 0.867),    Truffle Treat
#  ('item_details_p-8',  0.834)]    Fudge Fantasy

# Return top K
top_results = similarities[:5]
```
---

### **Build Phase (Storage):**

```
Graph Nodes                Text Representation              Embeddings (Stored)
─────────────              ───────────────────              ───────────────────
┌─────────────┐           "item_id: p-20 |                [0.23, -0.15, 0.87,
│ item_details│           item_name: Chocolate   BGE-M3    0.45, -0.67, 0.12,
│ p-20        │   ────→   Delight | item_desc:  ────────→  ..., 0.34, -0.21]
│ - id: p-20  │           Rich dark chocolate"             (1024 numbers)
│ - name: ... │                                             ↓
└─────────────┘                                        Stored in RAM
                                                       (Python dict)
```

### **Query Phase (Retrieval):**

```
User Query                 Query Embedding              Similarity Search
──────────                ─────────────────            ─────────────────
"Show chocolate           [0.34, -0.23, 0.56,          Compare with ALL
items"        BGE-M3      0.12, 0.45, -0.78,           stored embeddings
          ─────────────→  ..., 0.23, -0.11]            using cosine similarity
                          (1024 numbers)                        ↓
                                                       ┌──────────────────┐
                                                       │ Top 5 Results:   │
                                                       │ 1. p-20 (0.891)  │
                                                       │ 2. p-35 (0.867)  │
                                                       │ 3. p-8  (0.834)  │
                                                       └──────────────────┘
```

---

## **Embeddings Storage**

```python
# In-memory (RAM) storage
self.embedding_gen.node_embeddings = {
    'item_details_p-20': np.array([...]),   # 1024 floats × 4 bytes = 4 KB
    'item_details_p-8':  np.array([...]),   # 1024 floats × 4 bytes = 4 KB
    ...                                     # 243 nodes × 4 KB ≈ 1 MB
}

# Also stored: Original text (for reference)
self.embedding_gen.node_texts = {
    'item_details_p-20': "item_id: p-20 | item_name: Chocolate...",
    ...
}
```

**Memory Usage:**
- 243 nodes × 1024 dimensions × 4 bytes = ~1 MB (embeddings)
- Plus text storage ≈ 500 KB
- **Total: ~1.5 MB in RAM**

---

## **Cosine Similarity**

**Compare query embedding to node embeddings:**

```python
# Query embedding
query_emb = [0.34, -0.23, 0.56, ...]  # 1024 numbers

# Node embedding
node_emb = [0.23, -0.15, 0.87, ...]   # 1024 numbers

# Cosine similarity formula:
similarity = dot(query_emb, node_emb) / (||query_emb|| × ||node_emb||)

# In code:
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([query_emb], [node_emb])[0][0]
# 0.891 
```
**Key Points:**
- Measures angle between vectors (not distance)
- Range: -1 (opposite) to +1 (identical)
- Works well for semantic similarity

---
