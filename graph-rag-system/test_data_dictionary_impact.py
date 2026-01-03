# Databricks notebook source
# MAGIC %md
# MAGIC # Test: Data Dictionary Impact on LLM SQL Generation
# MAGIC
# MAGIC This notebook demonstrates the improvement in SQL quality when using
# MAGIC a data dictionary vs basic schema information.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# MAGIC %pip install networkx sentence-transformers scikit-learn --quiet

# COMMAND ----------

from graph_rag_core import GraphRAGConfig, GraphRAGSystem
from graph_rag_enhancements import enhance_with_all
from llm_sql_generator import LLMSQLGenerator

# COMMAND ----------

# Configure
config = GraphRAGConfig(
    catalog="accenture",
    schema="sales_analysis",
    fact_table="items_sales",
    dimension_tables=["item_details", "store_location", "customer_details"],
    fk_mappings={
        "items_sales": {
            "item_id": "item_details",
            "location_id": "store_location",
            "customer_id": "customer_details"
        }
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Check if Data Dictionary Exists

# COMMAND ----------

# Check for data dictionary
import os
databricks_catalog = os.environ.get("DATABRICKS_CATALOG")

tables = spark.sql(f"SHOW TABLES IN {databricks_catalog}.sales_analysis").toPandas()

has_dictionary = 'data_dictionary' in tables['tableName'].values

if has_dictionary:
    print(" Data dictionary found!")
    print("\nPreview:")
    spark.sql(f"""
        SELECT table_name, column_name, column_description
        FROM {databricks_catalog}.sales_analysis.data_dictionary
        WHERE column_name IS NOT NULL
        LIMIT 5
    """).display()
else:
    print(" Data dictionary not found!")
    print("\n Please run create_data_dictionary.py first")
    print("   Then re-run this notebook")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test Complex Queries

# COMMAND ----------

# Build system with data dictionary support
print("Building Graph RAG system with data dictionary...")
rag_system = GraphRAGSystem(config, spark)
rag_system.build()
enhance_with_all(rag_system)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test 1: "High-value customers"
# MAGIC
# MAGIC **Without dictionary:** LLM might guess based on zip code  
# MAGIC **With dictionary:** LLM knows to use SUM(total_sales_value)

# COMMAND ----------

print("="*80)
print("TEST 1: High-value customers")
print("="*80)

answer = rag_system.query("Show me high-value customers")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test 2: "Items with above-average revenue per unit"
# MAGIC
# MAGIC **Without dictionary:** Might not know which columns to use  
# MAGIC **With dictionary:** Knows total_sales_value / units_sold

# COMMAND ----------

print("="*80)
print("TEST 2: Above-average revenue per unit")
print("="*80)

answer = rag_system.query("Show me items with above-average revenue per unit sold")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test 3: "Popular locations"
# MAGIC
# MAGIC **Without dictionary:** Might count locations  
# MAGIC **With dictionary:** Knows to measure by sales volume/revenue

# COMMAND ----------

print("="*80)
print("TEST 3: Popular locations")
print("="*80)

answer = rag_system.query("Which are the most popular store locations?")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test 4: "Customer lifetime value"
# MAGIC
# MAGIC **Without dictionary:** Unclear what "value" means  
# MAGIC **With dictionary:** Clearly uses SUM(total_sales_value)

# COMMAND ----------

print("="*80)
print("TEST 4: Customer lifetime value")
print("="*80)

answer = rag_system.query("Calculate customer lifetime value")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Direct LLM Generator Test

# COMMAND ----------

# Test LLM generator directly to see schema context
llm_gen = LLMSQLGenerator(config, spark)

# COMMAND ----------

# View the schema context being sent to LLM
print("="*80)
print("SCHEMA CONTEXT SENT TO LLM:")
print("="*80)
print(llm_gen.schema_context[:2000])  # First 2000 chars
print("\n... (truncated)")

# COMMAND ----------

# Check if using data dictionary
if "Business Rules:" in llm_gen.schema_context:
    print("\n Using DATA DICTIONARY (rich context)")
else:
    print("\n Using BASIC SCHEMA (minimal context)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Quality Comparison

# COMMAND ----------

# Test queries that benefit from business context
test_queries = [
    {
        'query': 'Show high-value customers',
        'without_dict': 'Might use customer_zip_code as proxy',
        'with_dict': 'Uses SUM(total_sales_value) correctly'
    },
    {
        'query': 'Items with best profit margin',
        'without_dict': 'Unclear - no profit column exists',
        'with_dict': 'Knows to use revenue per unit as proxy'
    },
    {
        'query': 'Most loyal customers',
        'without_dict': 'Unclear metric',
        'with_dict': 'Uses purchase frequency (COUNT transactions)'
    },
    {
        'query': 'Underperforming locations',
        'without_dict': 'Unclear definition',
        'with_dict': 'Uses below-average SUM(total_sales_value)'
    }
]

print("="*80)
print("QUALITY COMPARISON")
print("="*80)

for test in test_queries:
    print(f"\nQuery: {test['query']}")
    print(f"  Without Dict: {test['without_dict']}")
    print(f"  With Dict:    {test['with_dict']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Inspect Generated SQL

# COMMAND ----------

# Generate SQL for analysis
question = "Show customers who spend more than average"

print(f"Question: {question}")
print("="*80)

sql = llm_gen.generate_sql(question, verbose=True)

print("\n" + "="*80)
print("GENERATED SQL:")
print("="*80)
print(sql)

# COMMAND ----------

# Execute to verify correctness
if sql:
    try:
        result_df = spark.sql(sql).toPandas()
        print(f"\n SQL executed successfully!")
        print(f"   Rows returned: {len(result_df)}")
        display(result_df.head(10))
    except Exception as e:
        print(f"\n SQL execution failed: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Business Rule Validation

# COMMAND ----------

# Test if LLM respects business rules from dictionary
print("="*80)
print("BUSINESS RULE VALIDATION")
print("="*80)

# Should NOT use customer_zip_code for value calculation
answer = rag_system.query("Find valuable customers in high-income zip codes")

print("\nChecking if SQL avoids using zip_code for value calculation...")
# Manually inspect the generated SQL to verify

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **Data Dictionary Benefits:**
# MAGIC
# MAGIC 1.  **Better column understanding**
# MAGIC    - LLM knows total_sales_value is THE revenue measure
# MAGIC    - Understands units_sold is for volume analysis
# MAGIC    - Knows customer_zip_code is NOT for value calculation
# MAGIC
# MAGIC 2.  **Clearer business logic**
# MAGIC    - "High-value" = high SUM(total_sales_value)
# MAGIC    - "Popular" = high sales volume/revenue
# MAGIC    - "Loyal" = high purchase frequency
# MAGIC
# MAGIC 3.  **Correct relationships**
# MAGIC    - Understands FK relationships
# MAGIC    - Generates proper JOINs
# MAGIC    - Uses correct grain for aggregations
# MAGIC
# MAGIC 4.  **Business rules compliance**
# MAGIC    - Respects data constraints
# MAGIC    - Follows business definitions
# MAGIC    - Uses approved calculations
