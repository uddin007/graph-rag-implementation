# Databricks notebook source
# MAGIC %md
# MAGIC # Graph RAG System - Driver Notebook
# MAGIC
# MAGIC This notebook runs the modular Graph RAG system.
# MAGIC
# MAGIC **Modules:**
# MAGIC - `graph_rag_core.py` - Core classes and base functionality
# MAGIC - `graph_rag_intelligence.py` - Query intelligence and SQL generation
# MAGIC - `graph_rag_enhancements.py` - Enhancement functions
# MAGIC
# MAGIC **Capabilities:**
# MAGIC 1. Semantic Search (Vector RAG)
# MAGIC 2. SQL Aggregation (Structured RAG)
# MAGIC 3. Time-Based Filtering
# MAGIC 4. Pattern/Relationship Queries (Graph RAG)
# MAGIC 5. Hybrid Time+Pattern Queries

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Installation & Imports

# COMMAND ----------

# Install required packages
%pip install networkx sentence-transformers scikit-learn mlflow

# COMMAND ----------

# Import modules
from graph_rag_core import GraphRAGConfig, GraphRAGSystem
from graph_rag_enhancements import enhance_with_all, save_graph_rag_model

import mlflow
import pandas as pd

print("Modules imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

# dbutils.widgets.dropdown('embedding_model',
#     defaultValue="all-MiniLM-L6-v2",
#     choices=[
#         "all-MiniLM-L6-v2",
#         "BAAI/bge-m3",
#         "BAAI/bge-base-en-v1.5",
#         "BAAI/bge-large-en-v1.5",
#         "intfloat/e5-small-v2"
#     ],
#     label="embedding_model"
# )

# COMMAND ----------

# ============================================================================
# CONFIGURE DATASET
# ============================================================================

config = GraphRAGConfig(
    # UPDATE THESE FOR THE DATA
    catalog=dbutils.widgets.get("tables_catalog"),
    schema=dbutils.widgets.get("tables_schema"),      
    fact_table="items_sales",         
    
    dimension_tables=[                
        "item_details",
        "store_location",
        "customer_details"
    ],
    
    # FOREIGN KEY MAPPINGS
    fk_mappings={
        "items_sales": {              
            "item_id": "item_details",         # FK → dimension table 
            "location_id": "store_location",   # FK → dimension table 
            "customer_id": "customer_details"  # FK → dimension table
        }
    },
    
    # Embedding model settings
    embedding_model=dbutils.widgets.get("embedding_model"),
    top_k_nodes=5,                        # Top results to return
    max_hops=2,                           # Graph traversal depth
    date_column="sale_date"               # Date column in fact table
)

print("Configuration created for:", config.fact_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build Graph RAG System

# COMMAND ----------

rag_system = GraphRAGSystem(config, spark)
rag_system.build() # ← Embeddings created here

# COMMAND ----------

# Add enhancements (SQL, Pattern queries, etc.)
enhance_with_all(rag_system)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test Queries
# MAGIC
# MAGIC Test all supported query types using denormalized view `agg_table`

# COMMAND ----------

tables_catalog = dbutils.widgets.get("tables_catalog")
tables_schema = dbutils.widgets.get("tables_schema")
agg_df = spark.table(f"{tables_catalog}.{tables_schema}.aggregated_sales")
agg_df.display()
agg_df.createOrReplaceTempView("agg_table")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Semantic Search Queries

# COMMAND ----------

# Semantic search - finds items by meaning
rag_system.query("Show me chocolate items")

# COMMAND ----------

# Test with keyworkd search
chocolate_items_list = spark.sql("""
SELECT DISTINCT item_name, item_description
FROM agg_table
WHERE LOWER(item_name) LIKE '%chocolate%' OR LOWER(item_description) LIKE '%chocolate%'
""").collect()
print(chocolate_items_list)

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** All semantic search items are captured within the keyword search. 

# COMMAND ----------

# Semantic search - to test embedding model
rag_system.query("Show premium chocolate items with rich flavor")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 SQL Aggregation Queries

# COMMAND ----------

# Top items by revenue
rag_system.query("What are the top 5 items by revenue?")

# COMMAND ----------

top_items_df = spark.sql("""
SELECT item_name, SUM(total_sales_value) AS total_revenue
FROM agg_table
GROUP BY item_name
ORDER BY total_revenue DESC
LIMIT 5
""")
print(top_items_df.collect())

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** The RAG output matches with the SQL query

# COMMAND ----------

# Top customers by sales
rag_system.query("Show me the top 3 customers by total sales")

# COMMAND ----------

top_customers_df = spark.sql("""
SELECT customer_name, SUM(total_sales_value) AS total_sales
FROM agg_table
GROUP BY customer_name
ORDER BY total_sales DESC
LIMIT 3
""")
print(top_customers_df.collect())

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** The RAG output matches with the SQL query

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Time-Based Queries

# COMMAND ----------

# Top items in specific month
rag_system.query("What are the top 5 items by revenue in December 2025?")

# COMMAND ----------

top_items_dec2025_df = spark.sql("""
SELECT item_name, SUM(total_sales_value) AS total_revenue
FROM agg_table
WHERE sale_date >= '2025-12-01' AND sale_date <= '2025-12-31'
GROUP BY item_name
ORDER BY total_revenue DESC
LIMIT 5
""")
print(top_items_dec2025_df.collect())

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** The RAG output matches with the SQL query

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 Pattern/Relationship Queries

# COMMAND ----------

# Which customers bought a specific product
rag_system.query('Which customers bought "Cocoa Swirl"?')

# COMMAND ----------

customers_cocoa_swirl_df = spark.sql("""
SELECT DISTINCT customer_name
FROM agg_table
WHERE item_name = 'Cocoa Swirl'
ORDER BY customer_name
""")
display(customers_cocoa_swirl_df)

# COMMAND ----------

# Customers who bought both products
rag_system.query('Which customers who bought "Cocoa Swirl" also bought "Berry Burst"?')

# COMMAND ----------

customers_both_df = spark.sql("""
SELECT DISTINCT a.customer_name
FROM agg_table a
JOIN agg_table b
  ON a.customer_name = b.customer_name
WHERE a.item_name = 'Cocoa Swirl'
  AND b.item_name = 'Berry Burst'
ORDER BY a.customer_name
""")
display(customers_both_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.5 Time-Based Pattern Queries

# COMMAND ----------

# Customers who bought product in specific timeframe
rag_system.query('Find customers who bought "Cocoa Swirl" in December 2025')

# COMMAND ----------

customers_cocoa_swirl_df = spark.sql("""
SELECT DISTINCT customer_name
FROM agg_table
WHERE item_name = 'Cocoa Swirl'
AND sale_date >= '2025-12-01' 
AND sale_date <= '2025-12-31'
ORDER BY customer_name
""")
display(customers_cocoa_swirl_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. System Statistics

# COMMAND ----------

# Get system statistics
stats = rag_system.get_statistics()

print(" Graph RAG System Statistics:")
print(f"   Total Nodes: {stats['total_nodes']:,}")
print(f"   Total Edges: {stats['total_edges']:,}")
print(f"   Embedding Dimension: {stats['embedding_dim']}")
print(f"   Fact Table: {stats['fact_table']}")
print(f"   Dimension Tables: {', '.join(stats['dimension_tables'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC 1. Built a modular Graph RAG system
# MAGIC 2. Tested all 5 query types (Semantic, SQL, Time, Pattern, Hybrid)
# MAGIC
# MAGIC **Supported Query Types:**
# MAGIC - Semantic Search: "Show chocolate items"
# MAGIC - SQL Aggregation: "Top 10 customers by revenue"
# MAGIC - Time-Based: "Top items in December 2025"
# MAGIC - Pattern: "Customers who bought A and B"
# MAGIC - Hybrid: "Customers who bought A and B in December 2025"

# COMMAND ----------


