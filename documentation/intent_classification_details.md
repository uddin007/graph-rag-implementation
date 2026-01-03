## **Intent Classification Details**

### **Query Flow for "Top 5 items by revenue"**

```python
answer = rag_system.query("What are the top 5 items by revenue?")
```

**Step-by-Step:**

```python
# Step 1: Intent Classification (NO embeddings, just keywords)
intent = classifier.classify("What are the top 5 items by revenue?")

# Detects:
# - "top" -> aggregation keyword 
# - "revenue" -> measure keyword 
# - "items" -> entity type 
# Result: Route to SQL.

# Step 2: SQL Generation (NO embeddings, just string building)
sql = """
    SELECT 
        d.item_id,
        d.item_name,
        SUM(f.total_sales_value) as total_sales_value
    FROM accenture.sales_analysis.items_sales f
    JOIN accenture.sales_analysis.item_details d 
        ON f.item_id = d.item_id
    GROUP BY d.item_id, d.item_name
    ORDER BY total_sales_value DESC
    LIMIT 5
"""

# Step 3: Execute SQL directly on database (NO embeddings).
result_df = spark.sql(sql).toPandas()

# Step 4: Format results as natural language
answer = format_sql_results(result_df)
```

---

## **Visual Comparison**

### **Semantic Query: "Show chocolate items"**

```
User Query
    ↓
┌─────────────────────────────────────┐
│ Intent Classifier                   │
│ "chocolate" → semantic search       │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ Encode Query                        │
│ "chocolate items"                   │
│         ↓                           │
│ [0.34, -0.23, 0.56, ...]            │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ Search Embeddings (IN RAM)          │
│ Compare with all 243 node embeddings│
│         ↓                           │
│ Cosine similarity                   │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ Return Top 5 Similar Nodes          │
│ 1. p-20 (0.891)                     │
│ 2. p-35 (0.867)                     │
└─────────────────────────────────────┘

Uses: Embeddings Graph nodes
Does NOT use: Database SQL
```

### **Aggregation Query: "Top 5 items by revenue"**

```
User Query
    ↓
┌─────────────────────────────────────┐
│ Intent Classifier                   │
│ "top" + "revenue" → SQL aggregation │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ SQL Query Builder                   │
│ SELECT item_id, SUM(revenue)...     │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ Execute on Database (Spark SQL)     │
│ Direct query to Unity Catalog       │
│         ↓                           │
│ Aggregation happens in database     │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ Return Top 5 Items by Revenue       │
│ 1. Maple Munch: $1,918.96           │
│ 2. Fudge Fantasy: $1,757.37         │
└─────────────────────────────────────┘

Does NOT use: Embeddings Graph nodes
Uses: Database SQL only
```

---

### **Embeddings are for SEMANTIC understanding**

```python
Query: "Show chocolate items"

# Embedding understands:
# "chocolate" ≈ "cocoa" ≈ "dark" ≈ "mocha"
# Finds semantically similar items
```

### **SQL is for ANALYTICAL precision**

```python
Query: "Top 5 items by revenue"

# SQL calculates:
# SUM(total_sales_value) for each item
# ORDER BY that sum
# Returns exact mathematical results 
```

---

## Summary of Query Routing

| Query Type | Embeddings | Graph Nodes | SQL | Use Case |
|------------|-------------|--------------|------|------|
| "Show chocolate items" | Yes | Yes | No | Semantic similarity |
| "Top 5 by revenue" | No | No | Yes | Mathematical aggregation |
| "Top 5 in December" | No | No | Yes | Aggregation + time filter |
| "Customers who bought A" | No | Yes | No | Graph traversal |
| "Customers who bought A in Dec" | No | No | Yes | SQL with joins faster |

---

## **Key Insight: Three Separate Data Sources**

### **1. Embeddings (In RAM)**
```python
# Location: Python dictionary in memory
self.embedding_gen.node_embeddings = {
    'item_details_p-20': array([0.23, -0.15, ...]),
    ...
}

# Used for: Semantic search
# Queries: "chocolate items", "premium products"
```

### **2. Graph Nodes (In RAM)**
```python
# Location: NetworkX graph in memory
self.graph.nodes = {
    'item_details_p-20': {
        'item_id': 'p-20',
        'item_name': 'Chocolate Delight',
        ...
    },
    'customer_details_c-1': {...},
    ...
}

# Used for: Relationship queries
# Queries: "Customers who bought Product A"
```

### **3. Database Tables (Unity Catalog)**
```python
# Location: Databricks Unity Catalog
# Tables: items_sales, item_details, store_location, customer_details

# Used for: Aggregation queries
# Queries: "Top 5 by revenue", "Total sales in December"
```
---

## **Trace a Aggregation Query**

```python
query = "What are the top 5 items by revenue?"
```

### **Step 1: Intent Classification (Keyword Matching)**

```python
def classify(question):
    question_lower = question.lower()
    
    # Check for aggregation keywords
    has_agg = 'top' in question_lower  # Found
    
    # Check for measure
    has_measure = 'revenue' in question_lower  # Found
    
    # Check for entity
    has_entity = 'items' in question_lower  # Found
    
    return {
        'is_aggregation': True,  # Route to SQL
        'measure': 'total_sales_value',
        'entity_type': 'item_details',
        'limit': 5
    }
```

**Simple keyword detection used.

### **Step 2: SQL Generation (String Building)**

```python
def build_aggregation_query(measure, entity_type, limit):
    # Build SQL string (no embeddings needed)
    sql = f"""
        SELECT 
            d.item_id,
            d.item_name,
            SUM(f.{measure}) as {measure}
        FROM {fact_table} f
        JOIN {entity_table} d ON f.item_id = d.item_id
        GROUP BY d.item_id, d.item_name
        ORDER BY {measure} DESC
        LIMIT {limit}
    """
    return sql
```

### **Step 3: Execute SQL (Database Query)**

```python
def execute_query(sql):
    # Execute directly on Unity Catalog
    result_df = spark.sql(sql).toPandas()
    return result_df

# Result:
#   item_id  item_name          total_sales_value
#   p-20     Maple Munch        1918.96
#   p-8      Fudge Fantasy      1757.37
#   p-17     Berry Burst        1729.42
```

**Direct database query.

### **Step 4: Format Results (Text Formatting)**

```python
def format_sql_results(df):
    answer_parts = ["Top 5 results:\n"]
    for idx, row in df.iterrows():
        answer_parts.append(
            f"{idx+1}. {row['item_name']}: ${row['total_sales_value']:,.2f}"
        )
    return "\n".join(answer_parts)

# Result:
# "Top 5 results:
#  1. Maple Munch: $1,918.96
#  2. Fudge Fantasy: $1,757.37
#  ..."
```
---

## **Summary**

1. Keyword matching (intent classification)
2. SQL generation (building query string)
3. Database execution (Spark SQL on Unity Catalog)
4. Text formatting (natural language response)
