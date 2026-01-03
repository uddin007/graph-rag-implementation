# Data Dictionary for Graph RAG  - Complete Guide

##  Why Data Dictionary Matters

### **The Problem Without It:**

```python
Query: "Show me high-value customers"

LLM sees only:
  customer_details: customer_id, customer_name, customer_email, customer_zip_code

LLM thinks:
  "What is 'high-value'?"
  "Maybe customer_zip_code = 90210 (Beverly Hills)?" 

Generated SQL:
  SELECT * FROM customer_details WHERE customer_zip_code = '90210'
  
Result: Returns customers in zip code 90210, not high-value customers
```

### **The Solution With Data Dictionary:**

```python
Query: "Show me high-value customers"

LLM sees:
  customer_details:
    - customer_id: "Primary key"
    - customer_zip_code: "ZIP code for demographics, NOT for value calculation"
  
  items_sales:
    - total_sales_value: "PRIMARY MEASURE for revenue analysis" 

LLM thinks:
  "High-value = high total_sales_value!"
  "Need to JOIN and SUM from items_sales"

Generated SQL:
  SELECT c.customer_id, c.customer_name, SUM(s.total_sales_value) as total_value
  FROM customer_details c
  JOIN items_sales s ON c.customer_id = s.customer_id
  GROUP BY c.customer_id, c.customer_name
  ORDER BY total_value DESC
  
Result: Returns customers sorted by actual purchase value
```

---

**Benefit:** Business logic understanding.

---

## **What to Include in Data Dictionary**

### **Table-Level Information:**

```python
{
    'table_name': 'items_sales',
    'table_type': 'fact',
    'table_description': 'Transaction-level sales data capturing purchases',
    'business_rules': 'Grain: One row per item sold. Updated daily via ETL.',
}
```

- LLM understands table granularity
- Knows update frequency
- Understands purpose

---

### **Column-Level Information:**

```python
{
    'column_name': 'total_sales_value',
    'column_type': 'double',
    'column_description': 'Total revenue in USD. PRIMARY MEASURE for revenue analysis.',
    'is_key': False,
    'is_measure': True,  # Tells LLM this is for aggregations
    'business_rules': 'Must be positive. Calculated as units × price. Excludes taxes.',
    'sample_values': '25.99, 47.50, 123.45'
}
```
- LLM knows this is THE revenue measure
- Understands calculation method
- Has examples for validation

---

### **Relationship Information:**

```python
{
    'column_name': 'customer_id',
    'is_key': True,
    'related_table': 'customer_details',  # FK relationship
    'column_description': 'Foreign key to customer_details. Identifies customer.',
}
```
- LLM generates correct JOINs automatically
- Understands data lineage
- Avoids orphaned records

---

### **Business Guidance:**

```python
{
    'business_rules': """
    customer_zip_code is for demographics, NOT for calculating customer value.
    Use SUM(items_sales.total_sales_value) for customer lifetime value.
    """
}
```
- Prevents common mistakes
- Enforces business definitions
- Maintains consistency

---

## Implementation**

### **Step 1: Create Data Dictionary**

**Table Schema:**
```sql
CREATE TABLE data_dictionary (
    catalog STRING,
    schema STRING,
    table_name STRING,
    table_type STRING,              -- 'fact' or 'dimension'
    table_description STRING,
    column_name STRING,             -- NULL for table-level rows
    column_type STRING,
    column_description STRING,
    is_key BOOLEAN,                 -- Primary or foreign key
    is_measure BOOLEAN,             -- Aggregation measure
    related_table STRING,           -- For foreign keys
    business_rules STRING,
    sample_values STRING
)
```

---

### **Step 2: System Auto-Detects Dictionary**

```python
# Build system 
rag_system = GraphRAGSystem(config, spark)
rag_system.build()
enhance_with_all(rag_system)

# LLM generator automatically checks for dictionary.

```

---

##  **Data Dictionary Template**

```python
data_dict = [
    # TABLE LEVEL 
    {
        'catalog': 'your_catalog',
        'schema': 'your_schema',
        'table_name': 'your_fact_table',
        'table_type': 'fact',
        'table_description': 'YOUR DESCRIPTION: What this table contains',
        'column_name': None,  # NULL for table-level
        'business_rules': 'YOUR RULES: Grain, update frequency, etc.'
    },
    
    # COLUMN LEVEL 
    {
        'catalog': 'your_catalog',
        'schema': 'your_schema',
        'table_name': 'your_fact_table',
        'column_name': 'your_column',
        'column_type': 'double',
        'column_description': 'YOUR DESCRIPTION: What this column means',
        'is_key': False,
        'is_measure': True,  # If this is for SUM/AVG/etc
        'related_table': None,  # Or FK target
        'business_rules': 'YOUR RULES: How to use this column',
        'sample_values': 'EXAMPLES: 100.50, 250.00'
    },
    
    # Repeat for each column...
]
```

---

##  **Emphasize in Descriptions**

### **For Measures (Aggregations):**

```python
#  GOOD:
column_description = "Total revenue in USD. PRIMARY MEASURE for sales analysis."

#  NOT GOOD:
column_description = "Sales value"
```

- LLM needs to know THIS is the main revenue measure.

---

### **For Keys:**

```python
#  GOOD:
column_description = "Foreign key to customer_details. Identifies purchasing customer."

#  NOT GOOD:
column_description = "Customer ID"
```

- LLM needs to know the relationship.

---

### **For Descriptive Fields:**

```python
#  GOOD:
column_description = "ZIP code for demographic analysis. NOT for customer value calculation."

#  NOT GOOD:
column_description = "ZIP code"
```

- Prevents misuse (e.g., using zip code as wealth proxy).

---

### **1. Explicit About Purpose**

```python
# Instead of:
"Sales amount"

# Write:
"Total revenue in USD. PRIMARY MEASURE for revenue analysis and profitability."
```
---

### **2. Include Business Rules**

```python
# Instead of:
"Customer ID"

# Write:
"Customer ID. Foreign key to customer_details.
Business Rule: Must exist in customer_details.
Use this to JOIN for customer demographics."
```

---

### **3. Clarify Calculations**

```python
# Instead of:
"Revenue per unit"

# Write:
"Revenue per unit. Calculated as total_sales_value / units_sold.
Use this for pricing analysis and margin optimization."
```

---

### **4. Provide Examples**

```python
# Instead of:
"Sale date"

# Write:
"Transaction date. Format: YYYY-MM-DD.
Sample values: 2025-12-15, 2025-11-20
Use YEAR(sale_date), MONTH(sale_date) for time analysis."
```

---

### **5. Warn About Pitfalls**

```python
# Instead of:
"ZIP code"

# Write:
"ZIP code. Used for demographic clustering and regional analysis.
WARNING: Do NOT use as proxy for customer value or income.
Use SUM(total_sales_value) for customer value instead."
```

---

##  **Validation Queries**

### **Check Dictionary Completeness:**

```sql
-- All tables documented?
SELECT DISTINCT table_name 
FROM data_dictionary 
WHERE column_name IS NULL;

-- All columns documented?
SELECT table_name, COUNT(*) as columns
FROM data_dictionary
WHERE column_name IS NOT NULL
GROUP BY table_name;

-- All measures identified?
SELECT table_name, column_name, column_description
FROM data_dictionary
WHERE is_measure = true;

-- All relationships mapped?
SELECT 
    table_name as source,
    column_name as fk,
    related_table as target
FROM data_dictionary
WHERE is_key = true AND related_table IS NOT NULL;
```
