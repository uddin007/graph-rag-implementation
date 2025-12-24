# Databricks notebook source
# MAGIC %md
# MAGIC # Graph RAG System - Driver Notebook
# MAGIC
# MAGIC This notebook demonstrates how to use the modular Graph RAG system.
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
# MAGIC
# MAGIC **This is the ONLY section you need to modify for different datasets!**

# COMMAND ----------

# ============================================================================
# CONFIGURE FOR OUR DATASET
# ============================================================================

config = GraphRAGConfig(
    # UPDATE THESE FOR YOUR DATA
    catalog="accenture",              # Your Unity Catalog name
    schema="sales_analysis",          # Your schema name
    fact_table="items_sales",         # Your fact table
    
    dimension_tables=[                # Your dimension tables
        "item_details",
        "store_location",
        "customer_details"
    ],
    
    # FOREIGN KEY MAPPINGS
    fk_mappings={
        "items_sales": {              # Fact table name
            "item_id": "item_details",         # FK → dimension
            "location_id": "store_location",   # FK → dimension
            "customer_id": "customer_details"  # FK → dimension
        }
    },
    
    # Embedding model settings
    embedding_model="all-MiniLM-L6-v2",  # Embedding model
    top_k_nodes=5,                        # Top results to return
    max_hops=2,                           # Graph traversal depth
    date_column="sale_date"               # Date column in fact table
)

print("Configuration created for:", config.fact_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build Graph RAG System

# COMMAND ----------

# Build the system
rag_system = GraphRAGSystem(config, spark)
rag_system.build()

# COMMAND ----------

# Add all enhancements (SQL, Pattern queries, etc.)
enhance_with_all(rag_system)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test Queries
# MAGIC
# MAGIC Test all supported query types.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Semantic Search Queries

# COMMAND ----------

# Semantic search - finds items by meaning
rag_system.query("Show me chocolate items")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 SQL Aggregation Queries

# COMMAND ----------

# Top items by revenue
rag_system.query("What are the top 5 items by revenue?")

# COMMAND ----------

# Top customers by sales
rag_system.query("Show me the top 10 customers by total sales")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Time-Based Queries

# COMMAND ----------

# Top items in specific month
rag_system.query("What are the top 5 items by revenue in December 2025?")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 Pattern/Relationship Queries

# COMMAND ----------

# Which customers bought a specific product
rag_system.query('Which customers bought "Cocoa Swirl"?')

# COMMAND ----------

# Customers who bought both products
rag_system.query('Which customers who bought "Cocoa Swirl" also bought "Berry Burst"?')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.5 Time-Based Pattern Queries

# COMMAND ----------

# Customers who bought product in specific timeframe
rag_system.query('Find customers who bought "Cocoa Swirl" in December 2025')

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
# MAGIC ## 8. Batch Testing

# COMMAND ----------

# Test multiple queries
test_queries = [
    "Show me chocolate items",
    "Top 10 customers by revenue",
    "Top 5 items in December 2025",
    'Which customers bought "Maple Munch"?',
    "Bottom 5 items by sales"
]

print("="*80)
print("BATCH TESTING - MULTIPLE QUERIES")
print("="*80)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*80}")
    print(f"Test {i}/{len(test_queries)}")
    print(f"{'='*80}")
    
    answer = rag_system.query(query, verbose=False)
    
    print(f"\n Query: {query}")
    print(f"\n Answer:")
    print(answer[:500])  # Show first 500 chars
    print()

print(f"\n{'='*80}")
print(f" All {len(test_queries)} tests completed!")
print(f"{'='*80}")

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


