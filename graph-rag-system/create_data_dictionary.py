# Databricks notebook source
# MAGIC %md
# MAGIC # Create Data Dictionary for Graph RAG
# MAGIC
# MAGIC This notebook creates a comprehensive data dictionary that enhances
# MAGIC LLM SQL generation by providing business context for tables and columns.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Define Data Dictionary Schema

# COMMAND ----------

import os
databricks_catalog= os.getenv("DATABRICKS_CATALOG")

# Create data dictionary as DataFrame
data_dict_data = [
    # ========================================================================
    # FACT TABLE: items_sales
    # ========================================================================
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'items_sales',
        'table_type': 'fact',
        'table_description': 'Transaction-level sales data capturing individual item purchases with revenue, quantity, date, and customer/location information',
        'column_name': None,
        'column_type': None,
        'column_description': None,
        'is_key': False,
        'is_measure': False,
        'related_table': None,
        'business_rules': 'Grain: One row per item sold per transaction. Updated daily via ETL from POS systems.',
        'sample_values': None
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'items_sales',
        'table_type': 'fact',
        'table_description': 'Transaction-level sales data',
        'column_name': 'item_id',
        'column_type': 'string',
        'column_description': 'Foreign key to item_details. Identifies which product was sold.',
        'is_key': True,
        'is_measure': False,
        'related_table': 'item_details',
        'business_rules': 'Required field. Must exist in item_details table.',
        'sample_values': 'p-1, p-20, p-35'
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'items_sales',
        'table_type': 'fact',
        'table_description': 'Transaction-level sales data',
        'column_name': 'location_id',
        'column_type': 'string',
        'column_description': 'Foreign key to store_location. Identifies which store location made the sale.',
        'is_key': True,
        'is_measure': False,
        'related_table': 'store_location',
        'business_rules': 'Required field. Must exist in store_location table.',
        'sample_values': 'l-1, l-27, l-42'
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'items_sales',
        'table_type': 'fact',
        'table_description': 'Transaction-level sales data',
        'column_name': 'customer_id',
        'column_type': 'string',
        'column_description': 'Foreign key to customer_details. Identifies which customer made the purchase.',
        'is_key': True,
        'is_measure': False,
        'related_table': 'customer_details',
        'business_rules': 'Required field. Must exist in customer_details table.',
        'sample_values': 'c-1, c-50, c-120'
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'items_sales',
        'table_type': 'fact',
        'table_description': 'Transaction-level sales data',
        'column_name': 'sale_date',
        'column_type': 'date',
        'column_description': 'Date when the transaction occurred. Used for time-based analysis and trends.',
        'is_key': False,
        'is_measure': False,
        'related_table': None,
        'business_rules': 'Cannot be future date. Typically within last 2 years for reporting.',
        'sample_values': '2025-12-15, 2025-11-20, 2025-10-05'
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'items_sales',
        'table_type': 'fact',
        'table_description': 'Transaction-level sales data',
        'column_name': 'units_sold',
        'column_type': 'int',
        'column_description': 'Quantity of items sold in this transaction. Used for volume analysis and inventory planning.',
        'is_key': False,
        'is_measure': True,
        'related_table': None,
        'business_rules': 'Must be positive integer. Typically between 1-10 for retail items.',
        'sample_values': '1, 2, 5, 10'
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'items_sales',
        'table_type': 'fact',
        'table_description': 'Transaction-level sales data',
        'column_name': 'total_sales_value',
        'column_type': 'double',
        'column_description': 'Total revenue generated from this transaction in USD. PRIMARY MEASURE for revenue analysis, profitability, and sales performance.',
        'is_key': False,
        'is_measure': True,
        'related_table': None,
        'business_rules': 'Must be positive. Calculated as units_sold × unit_price. Excludes taxes.',
        'sample_values': '25.99, 47.50, 123.45'
    },
    
    # ========================================================================
    # DIMENSION TABLE: item_details
    # ========================================================================
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'item_details',
        'table_type': 'dimension',
        'table_description': 'Product master data containing item attributes, descriptions, and metadata. Use this for product-level analysis and filtering.',
        'column_name': None,
        'column_type': None,
        'column_description': None,
        'is_key': False,
        'is_measure': False,
        'related_table': None,
        'business_rules': 'Type-2 SCD (slowly changing dimension). Current products only.',
        'sample_values': None
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'item_details',
        'table_type': 'dimension',
        'table_description': 'Product master data',
        'column_name': 'item_id',
        'column_type': 'string',
        'column_description': 'Primary key. Unique identifier for each product in the catalog.',
        'is_key': True,
        'is_measure': False,
        'related_table': 'items_sales',
        'business_rules': 'Format: p-{number}. Immutable once assigned.',
        'sample_values': 'p-1, p-20, p-35'
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'item_details',
        'table_type': 'dimension',
        'table_description': 'Product master data',
        'column_name': 'item_name',
        'column_type': 'string',
        'column_description': 'Display name of the product. Use for user-facing reports and product searches.',
        'is_key': False,
        'is_measure': False,
        'related_table': None,
        'business_rules': 'Maximum 100 characters. Must be unique.',
        'sample_values': 'Chocolate Delight, Berry Burst, Maple Munch'
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'item_details',
        'table_type': 'dimension',
        'table_description': 'Product master data',
        'column_name': 'item_description',
        'column_type': 'string',
        'column_description': 'Detailed product description including ingredients, flavors, and features. Use for semantic search and product matching.',
        'is_key': False,
        'is_measure': False,
        'related_table': None,
        'business_rules': 'Maximum 500 characters. Marketing-approved text.',
        'sample_values': 'Rich dark chocolate with smooth ganache filling'
    },
    
    # ========================================================================
    # DIMENSION TABLE: store_location
    # ========================================================================
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'store_location',
        'table_type': 'dimension',
        'table_description': 'Store location master data with geographic and operational details. Use for location-based sales analysis and regional comparisons.',
        'column_name': None,
        'column_type': None,
        'column_description': None,
        'is_key': False,
        'is_measure': False,
        'related_table': None,
        'business_rules': 'Contains active and historical locations. Filter by status if needed.',
        'sample_values': None
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'store_location',
        'table_type': 'dimension',
        'table_description': 'Store location master data',
        'column_name': 'location_id',
        'column_type': 'string',
        'column_description': 'Primary key. Unique identifier for each store location.',
        'is_key': True,
        'is_measure': False,
        'related_table': 'items_sales',
        'business_rules': 'Format: l-{number}. Assigned sequentially.',
        'sample_values': 'l-1, l-27, l-42'
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'store_location',
        'table_type': 'dimension',
        'table_description': 'Store location master data',
        'column_name': 'location_name',
        'column_type': 'string',
        'column_description': 'City or geographic name of the store location. Use for regional reporting.',
        'is_key': False,
        'is_measure': False,
        'related_table': None,
        'business_rules': 'City name only (no state/country). Must be unique.',
        'sample_values': 'Seattle, Portland, Memphis, Tucson'
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'store_location',
        'table_type': 'dimension',
        'table_description': 'Store location master data',
        'column_name': 'location_description',
        'column_type': 'string',
        'column_description': 'Additional details about the store including size, format, and special characteristics.',
        'is_key': False,
        'is_measure': False,
        'related_table': None,
        'business_rules': 'Optional field. Maximum 500 characters.',
        'sample_values': 'Downtown flagship store, Suburban mall location'
    },
    
    # ========================================================================
    # DIMENSION TABLE: customer_details
    # ========================================================================
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'customer_details',
        'table_type': 'dimension',
        'table_description': 'Customer master data with demographic and contact information. Use for customer segmentation, loyalty analysis, and personalization.',
        'column_name': None,
        'column_type': None,
        'column_description': None,
        'is_key': False,
        'is_measure': False,
        'related_table': None,
        'business_rules': 'PII data - handle with care. GDPR/privacy compliant.',
        'sample_values': None
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'customer_details',
        'table_type': 'dimension',
        'table_description': 'Customer master data',
        'column_name': 'customer_id',
        'column_type': 'string',
        'column_description': 'Primary key. Unique identifier for each customer in the loyalty program.',
        'is_key': True,
        'is_measure': False,
        'related_table': 'items_sales',
        'business_rules': 'Format: c-{number}. Assigned at registration.',
        'sample_values': 'c-1, c-50, c-120'
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'customer_details',
        'table_type': 'dimension',
        'table_description': 'Customer master data',
        'column_name': 'customer_name',
        'column_type': 'string',
        'column_description': 'Full name of the customer. Use for personalization and customer service.',
        'is_key': False,
        'is_measure': False,
        'related_table': None,
        'business_rules': 'PII - handle per privacy policy. May contain middle initial.',
        'sample_values': 'John Smith, Jane Doe, Bob Johnson'
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'customer_details',
        'table_type': 'dimension',
        'table_description': 'Customer master data',
        'column_name': 'customer_email',
        'column_type': 'string',
        'column_description': 'Email address for marketing communications and receipts. PRIMARY contact method.',
        'is_key': False,
        'is_measure': False,
        'related_table': None,
        'business_rules': 'PII - must be valid email format. Used for email campaigns.',
        'sample_values': 'john.smith@email.com, jane.doe@example.com'
    },
    {
        'catalog': databricks_catalog,
        'schema': 'sales_analysis',
        'table_name': 'customer_details',
        'table_type': 'dimension',
        'table_description': 'Customer master data',
        'column_name': 'customer_zip_code',
        'column_type': 'string',
        'column_description': 'ZIP/postal code for demographic analysis and regional targeting. NOT for calculating customer value.',
        'is_key': False,
        'is_measure': False,
        'related_table': None,
        'business_rules': 'US ZIP codes only (5 digits). May be used for geo-clustering.',
        'sample_values': '98101, 90210, 10001'
    },
]

# Convert to DataFrame
import pandas as pd
data_dict_df = pd.DataFrame(data_dict_data)

# Show preview
display(data_dict_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Table in Unity Catalog

# COMMAND ----------

# Convert to Spark DataFrame
from pyspark.sql.types import StructType, StructField, StringType, BooleanType

schema = StructType([
    StructField("catalog", StringType(), False),
    StructField("schema", StringType(), False),
    StructField("table_name", StringType(), False),
    StructField("table_type", StringType(), True),
    StructField("table_description", StringType(), True),
    StructField("column_name", StringType(), True),
    StructField("column_type", StringType(), True),
    StructField("column_description", StringType(), True),
    StructField("is_key", BooleanType(), True),
    StructField("is_measure", BooleanType(), True),
    StructField("related_table", StringType(), True),
    StructField("business_rules", StringType(), True),
    StructField("sample_values", StringType(), True),
])

data_dict_spark = spark.createDataFrame(data_dict_df, schema=schema)

# COMMAND ----------

# Create table
data_dict_spark.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{databricks_catalog}.sales_analysis.data_dictionary")

print(f" Data dictionary table created: {databricks_catalog}.sales_analysis.data_dictionary")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Query Examples

# COMMAND ----------

# View all table descriptions
spark.sql(f"""
    SELECT DISTINCT
        table_name,
        table_type,
        table_description
    FROM {databricks_catalog}.sales_analysis.data_dictionary
    WHERE column_name IS NULL
""").display()

# COMMAND ----------

# View all columns for items_sales
spark.sql(f"""
    SELECT 
        column_name,
        column_type,
        column_description,
        is_key,
        is_measure,
        related_table
    FROM {databricks_catalog}.sales_analysis.data_dictionary
    WHERE table_name = 'items_sales'
        AND column_name IS NOT NULL
    ORDER BY is_key DESC, is_measure DESC
""").display()

# COMMAND ----------

# View all measures (for analytics)
spark.sql(f"""SELECT
        table_name,
        column_name,
        column_type,
        column_description,
        business_rules
    FROM {databricks_catalog}.sales_analysis.data_dictionary
    WHERE is_measure = true
""").display()

# COMMAND ----------

# View all foreign key relationships
spark.sql(f"""
    SELECT 
        table_name as source_table,
        column_name as foreign_key,
        related_table as target_table,
        column_description
    FROM {databricks_catalog}.sales_analysis.data_dictionary
    WHERE is_key = true
        AND related_table IS NOT NULL
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verify Table Statistics

# COMMAND ----------

# Count total entries
total_count = spark.sql(f"SELECT COUNT(*) as count FROM {databricks_catalog}.sales_analysis.data_dictionary").collect()[0]['count']
print(f"Total entries: {total_count}")

# Count by table
spark.sql(f"""
    SELECT 
        table_name,
        COUNT(*) as entry_count,
        SUM(CASE WHEN column_name IS NULL THEN 1 ELSE 0 END) as table_level,
        SUM(CASE WHEN column_name IS NOT NULL THEN 1 ELSE 0 END) as column_level
    FROM {databricks_catalog}.sales_analysis.data_dictionary
    GROUP BY table_name
    ORDER BY table_name
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC  Created comprehensive data dictionary table  
# MAGIC  Includes table-level and column-level metadata  
# MAGIC  Documents business rules and sample values  
# MAGIC  Identifies keys, measures, and relationships  
# MAGIC  Ready for LLM SQL generation enhancement  
# MAGIC
# MAGIC **IMPORTANT:** Update LLM SQL generator is required to use this dictionary

# COMMAND ----------


