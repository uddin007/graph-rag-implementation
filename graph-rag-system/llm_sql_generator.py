# ============================================================================
# llm_sql_generator.py
# LLM-powered SQL generation for complex queries
# ============================================================================

"""
LLM-based SQL generator using Databricks Foundation Models.

This module provides a fallback SQL generator that uses Claude Sonnet 4.5
to generate SQL queries for complex requests that the rule-based system
cannot handle.

Usage:
    from llm_sql_generator import LLMSQLGenerator
    
    generator = LLMSQLGenerator(config, spark)
    sql = generator.generate_sql(question)
"""

import requests
import json
import re
from typing import Dict, Any, Optional
import os

databricks_catalog = os.getenv('DATABRICKS_CATALOG')
databricks_endpoint_url = os.getenv('DATABRICKS_ENDPOINT_URL')

class LLMSQLGenerator:
    """Generate SQL queries using Databricks Foundation Models (Claude Sonnet 4.5)"""
    
    def __init__(self, config, spark):
        self.config = config
        self.spark = spark
        
        # Databricks endpoint for Claude Sonnet 4.5
        self.endpoint_url = databricks_endpoint_url
        
        # Get Databricks token
        self.token = self._get_databricks_token()
        
        # Build schema context for prompting
        self.schema_context = self._build_schema_context()
    
    def _get_databricks_token(self) -> str:
        """Get Databricks authentication token"""
        try:
            # Try to get from Databricks context
            from databricks.sdk.runtime import dbutils
            return dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        except:
            # Fallback: Try environment variable
            import os
            token = os.getenv('DATABRICKS_TOKEN')
            if token:
                return token
            else:
                raise ValueError(
                    "Could not get Databricks token. Please set DATABRICKS_TOKEN environment variable "
                    "or run in Databricks notebook."
                )
    
    def _build_schema_context(self) -> str:
        """Build schema information for LLM prompt
        
        First tries to load from data dictionary table (if exists).
        Falls back to basic DESCRIBE if dictionary not available.
        """
        
        # Try to load from data dictionary first
        try:
            dict_table = f"{self.config.catalog}.{self.config.schema}.data_dictionary"
            
            # Check if table exists
            tables = self.spark.sql(f"SHOW TABLES IN {self.config.catalog}.{self.config.schema}").toPandas()
            if 'data_dictionary' in tables['tableName'].values:
                return self._build_context_from_dictionary(dict_table)
            else:
                print("    Data dictionary not found, using basic schema")
                return self._build_basic_schema_context()
        
        except Exception as e:
            print(f"    Could not load data dictionary: {e}")
            return self._build_basic_schema_context()
    
    def _build_context_from_dictionary(self, dict_table: str) -> str:
        """Build rich schema context from data dictionary table"""
        
        # Load dictionary
        dict_df = self.spark.sql(f"SELECT * FROM {dict_table}").toPandas()
        
        schema_text = """
DATABASE SCHEMA WITH BUSINESS CONTEXT:
================================================================================

"""
        
        # Get unique tables
        tables = dict_df[dict_df['table_name'].notna()]['table_name'].unique()
        
        for table_name in tables:
            # Get table-level info
            table_info = dict_df[
                (dict_df['table_name'] == table_name) & 
                (dict_df['column_name'].isna())
            ]
            
            table_full = self.config.get_full_table_name(table_name)
            table_type = table_info['table_type'].iloc[0] if not table_info.empty else 'table'
            table_desc = table_info['table_description'].iloc[0] if not table_info.empty else 'No description'
            
            schema_text += f"""
{'='*80}
TABLE: {table_full} ({table_type.upper()})
{'='*80}
Description: {table_desc}

COLUMNS:
"""
            
            # Get column-level info
            columns = dict_df[
                (dict_df['table_name'] == table_name) & 
                (dict_df['column_name'].notna())
            ].sort_values(['is_key', 'is_measure'], ascending=[False, False])
            
            for _, col in columns.iterrows():
                col_name = col['column_name']
                col_type = col['column_type']
                col_desc = col['column_description']
                is_key = col['is_key']
                is_measure = col['is_measure']
                related = col['related_table']
                business_rules = col['business_rules']
                samples = col['sample_values']
                
                # Build column entry
                key_marker = " [PRIMARY KEY]" if is_key and not related else ""
                fk_marker = f" [FOREIGN KEY → {related}]" if related else ""
                measure_marker = " [MEASURE]" if is_measure else ""
                
                schema_text += f"""
  • {col_name} ({col_type}){key_marker}{fk_marker}{measure_marker}
    Description: {col_desc}
"""
                
                if business_rules:
                    schema_text += f"    Business Rules: {business_rules}\n"
                
                if samples:
                    schema_text += f"    Sample Values: {samples}\n"
        
        # Add relationship summary
        schema_text += """

================================================================================
FOREIGN KEY RELATIONSHIPS:
================================================================================
"""
        
        relationships = dict_df[
            (dict_df['is_key'] == True) & 
            (dict_df['related_table'].notna())
        ]
        
        for _, rel in relationships.iterrows():
            source = f"{self.config.get_full_table_name(rel['table_name'])}.{rel['column_name']}"
            target = self.config.get_full_table_name(rel['related_table'])
            schema_text += f"  {source} → {target}\n"
        
        # Add business guidance
        schema_text += """

================================================================================
BUSINESS GUIDANCE FOR SQL GENERATION:
================================================================================

MEASURES (Use for aggregations):
"""
        
        measures = dict_df[dict_df['is_measure'] == True]
        for _, m in measures.iterrows():
            schema_text += f"  • {m['column_name']}: {m['column_description']}\n"
        
        schema_text += """

KEY RULES:
1. For revenue/sales analysis: Use total_sales_value (PRIMARY revenue measure)
2. For volume analysis: Use units_sold
3. For customer value: JOIN to items_sales and SUM(total_sales_value)
4. For time-based analysis: Use sale_date with YEAR(), MONTH(), QUARTER()
5. For location analysis: JOIN via location_id
6. customer_zip_code is for demographics, NOT for calculating customer value

COMMON QUERIES:
- "High-value customers" = Customers with high SUM(total_sales_value)
- "Popular items" = Items with high COUNT(*) or SUM(units_sold)
- "Top locations" = Locations with high SUM(total_sales_value)
- "Revenue per unit" = SUM(total_sales_value) / SUM(units_sold)

"""
        
        return schema_text
    
    def _build_basic_schema_context(self) -> str:
        """Build basic schema context using DESCRIBE (fallback)"""
        
        # Get fact table schema
        fact_table_full = self.config.full_fact_table
        
        # Get fact table schema
        fact_schema = self.spark.sql(f"DESCRIBE {fact_table_full}").toPandas()
        
        # Get dimension table schemas
        dim_schemas = {}
        for dim_table in self.config.dimension_tables:
            dim_table_full = self.config.get_full_table_name(dim_table)
            dim_schemas[dim_table] = self.spark.sql(f"DESCRIBE {dim_table_full}").toPandas()
        
        # Format as text
        schema_text = f"""
DATABASE SCHEMA:

Fact Table: {fact_table_full}
Columns:
{self._format_schema_df(fact_schema)}

Dimension Tables:
"""
        
        for dim_table, schema_df in dim_schemas.items():
            dim_table_full = self.config.get_full_table_name(dim_table)
            schema_text += f"""
{dim_table_full}:
{self._format_schema_df(schema_df)}
"""
        
        # Add FK relationships
        schema_text += "\nForeign Key Relationships:\n"
        for fk_col, dim_table in self.config.fk_mappings.get(self.config.fact_table, {}).items():
            schema_text += f"  {fact_table_full}.{fk_col} → {self.config.get_full_table_name(dim_table)}\n"
        
        return schema_text
    
    def _format_schema_df(self, df) -> str:
        """Format schema DataFrame as text"""
        lines = []
        for _, row in df.iterrows():
            col_name = row['col_name']
            data_type = row['data_type']
            lines.append(f"  - {col_name} ({data_type})")
        return "\n".join(lines)
    
    def generate_sql(self, question: str, verbose: bool = True) -> Optional[str]:
        """Generate SQL query using LLM
        
        Args:
            question: Natural language question
            verbose: Print debug information
            
        Returns:
            SQL query string, or None if generation failed
        """
        
        if verbose:
            print(f" Using LLM to generate SQL...")
            print(f"   Question: {question}")
        
        # Build prompt
        prompt = self._build_prompt(question)
        
        # Call LLM
        try:
            sql = self._call_llm(prompt, verbose)
            
            if sql:
                if verbose:
                    print(f"\n Generated SQL:")
                    print(f"{sql}\n")
                
                return sql
            else:
                if verbose:
                    print(f" LLM failed to generate SQL")
                return None
        
        except Exception as e:
            if verbose:
                print(f" Error calling LLM: {str(e)}")
            return None
    
    def _build_prompt(self, question: str) -> str:
        """Build prompt for LLM"""
        
        prompt = f"""You are a SQL expert. Generate a SQL query to answer the following question.

{self.schema_context}

QUESTION: {question}

INSTRUCTIONS:
1. Generate ONLY the SQL query, no explanation
2. Use proper table aliases (f for fact table, d for dimension tables)
3. Always use fully qualified table names (catalog.schema.table)
4. For aggregations, include GROUP BY and ORDER BY
5. Use appropriate JOINs based on foreign key relationships
6. Return ONLY executable SQL code
7. Do not include markdown code blocks (no ```sql```)
8. Do not include any text before or after the SQL

EXAMPLE OUTPUT FORMAT:
SELECT 
    d.item_id,
    d.item_name,
    SUM(f.total_sales_value) as total_revenue
FROM {self.config.full_fact_table} f
JOIN {self.config.get_full_table_name('item_details')} d ON f.item_id = d.item_id
GROUP BY d.item_id, d.item_name
ORDER BY total_revenue DESC
LIMIT 10

Now generate SQL for the question above:
"""
        
        return prompt
    
    def _call_llm(self, prompt: str, verbose: bool = True) -> Optional[str]:
        """Call Databricks Foundation Model endpoint"""
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1  # Low temperature for deterministic SQL
        }
        
        if verbose:
            print(f"    Calling Claude Sonnet 4.5 endpoint...")
        
        response = requests.post(
            self.endpoint_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract SQL from response
            sql_text = result['choices'][0]['message']['content']
            
            # Clean up SQL
            sql = self._clean_sql(sql_text)
            
            return sql
        else:
            if verbose:
                print(f"    API Error: {response.status_code}")
                print(f"   {response.text}")
            return None
    
    def _clean_sql(self, sql_text: str) -> str:
        """Clean up SQL from LLM response"""
        
        # Remove markdown code blocks if present
        sql = re.sub(r'```sql\s*', '', sql_text)
        sql = re.sub(r'```\s*', '', sql)
        
        # Remove leading/trailing whitespace
        sql = sql.strip()
        
        # Remove any explanatory text before/after SQL
        # Keep only the SELECT ... statement
        lines = sql.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line_upper = line.strip().upper()
            
            # Start capturing when we see SELECT, WITH, or other SQL keywords
            if line_upper.startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
                in_sql = True
            
            if in_sql:
                sql_lines.append(line)
        
        return '\n'.join(sql_lines)
    
    def validate_and_execute(self, sql: str, verbose: bool = True) -> Optional[Any]:
        """Validate and execute generated SQL
        
        Args:
            sql: SQL query string
            verbose: Print debug information
            
        Returns:
            Pandas DataFrame with results, or None if execution failed
        """
        
        try:
            if verbose:
                print(f" Validating SQL...")
            
            # Try to execute
            result_df = self.spark.sql(sql).toPandas()
            
            if verbose:
                print(f" Query executed successfully!")
                print(f"   Rows returned: {len(result_df)}")
            
            return result_df
        
        except Exception as e:
            if verbose:
                print(f" SQL Execution Error: {str(e)}")
            return None


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

def example_usage():
    """Example of using LLM SQL generator"""
    
    from graph_rag_core import GraphRAGConfig
    
    # Configure
    config = GraphRAGConfig(
        catalog=databricks_catalog,
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
    
    # Initialize LLM SQL generator
    llm_sql = LLMSQLGenerator(config, spark)
    
    # Test complex queries
    test_queries = [
        "Show me items with above-average revenue per unit sold",
        "Which customers have made purchases in at least 3 different locations?",
        "What's the average time between purchases for each customer?",
        "Show items that are popular in Seattle but not in Portland",
        "Calculate the revenue contribution percentage for each item"
    ]
    
    for question in test_queries:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}")
        
        # Generate SQL
        sql = llm_sql.generate_sql(question)
        
        if sql:
            # Execute
            result_df = llm_sql.validate_and_execute(sql)
            
            if result_df is not None:
                print(f"\n Results:")
                print(result_df.head(10))
        
        print()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'LLMSQLGenerator'
]
