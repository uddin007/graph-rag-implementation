# ============================================================================
# graph_rag_intelligence.py
# Query intelligence: Intent classification, SQL generation, Pattern queries
# ============================================================================

"""
Query intelligence module for Graph RAG system.

This module contains the "smart" components that understand user intent
and route queries to appropriate retrieval methods:

- QueryIntentClassifier: Detects query type and extracts parameters
- SQLQueryBuilder: Generates optimized SQL for aggregation queries  
- PatternQueryEngine: Handles relationship/pattern queries
- TimeAwarePatternEngine: Pattern queries with date filtering

These components are added to GraphRAGSystem via enhancement functions.
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import pandas as pd
import networkx as nx
import re


# ============================================================================
# QUERY INTENT CLASSIFIER
# ============================================================================

class QueryIntentClassifier:
    """Classifies user query intent and extracts parameters
    
    Detects:
    - Aggregation queries (top, bottom, highest, lowest)
    - Measure keywords (sales, revenue, units)
    - Entity types (customers, items, locations)
    - Date filters (years, months, quarters)
    - Limits (top 5, bottom 10)
    """
    
    # Query type indicators
    AGGREGATION_KEYWORDS = [
        'top', 'bottom', 'highest', 'lowest', 'most', 'least',
        'best', 'worst', 'maximum', 'minimum', 'largest', 'smallest'
    ]
    
    # Measure mappings
    MEASURE_KEYWORDS = {
        'sales': 'total_sales_value',
        'revenue': 'total_sales_value',
        'value': 'total_sales_value',
        'amount': 'total_sales_value',
        'units': 'units_sold',
        'quantity': 'units_sold',
        'items': 'units_sold',
        'purchases': 'COUNT(*)',
        'orders': 'COUNT(*)',
        'transactions': 'COUNT(*)',
        'count': 'COUNT(*)'
    }
    
    # Month mappings
    MONTH_NAMES = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12
    }
    
    def __init__(self, config):
        self.config = config
    
    def classify(self, question: str) -> Dict[str, Any]:
        """Classify query and extract all parameters
        
        Returns:
            Dict with: is_aggregation, measure, entity_type, limit, 
                      is_bottom, date_filters
        """
        question_lower = question.lower()
        
        # Detect aggregation
        is_aggregation = any(
            kw in question_lower for kw in self.AGGREGATION_KEYWORDS
        )
        
        # Extract measure
        measure = self._extract_measure(question_lower)
        
        # Extract entity type
        entity_type = self._extract_entity_type(question_lower)
        
        # Extract limit (top N, bottom N)
        limit, is_bottom = self._extract_limit(question_lower)
        
        # Extract date filters
        date_filters = self._extract_date_filters(question)
        
        return {
            'is_aggregation': is_aggregation,
            'measure': measure,
            'entity_type': entity_type,
            'limit': limit,
            'is_bottom': is_bottom,
            'date_filters': date_filters
        }
    
    def _extract_measure(self, question_lower: str) -> Optional[str]:
        """Extract measure column from query"""
        for keyword, column in self.MEASURE_KEYWORDS.items():
            if keyword in question_lower:
                return column
        return None
    
    def _extract_entity_type(self, question_lower: str) -> Optional[str]:
        """Extract entity type (which dimension table)"""
        for dim_table in self.config.dimension_tables:
            # Check for table name or common aliases
            if 'customer' in dim_table.lower() and any(
                w in question_lower for w in ['customer', 'client', 'buyer']
            ):
                return dim_table
            elif 'item' in dim_table.lower() and any(
                w in question_lower for w in ['item', 'product', 'sku']
            ):
                return dim_table
            elif 'location' in dim_table.lower() and any(
                w in question_lower for w in ['location', 'store', 'place']
            ):
                return dim_table
        
        return None
    
    def _extract_limit(self, question_lower: str) -> Tuple[int, bool]:
        """Extract limit and direction (top/bottom)
        
        Returns:
            (limit, is_bottom) tuple
        """
        is_bottom = any(w in question_lower for w in ['bottom', 'worst', 'least', 'lowest', 'minimum'])
        
        # Try to find number
        import re
        numbers = re.findall(r'\b(\d+)\b', question_lower)
        
        if numbers:
            return int(numbers[0]), is_bottom
        
        return 5, is_bottom  # Default to 5
    
    def _extract_date_filters(self, question: str) -> Dict[str, Any]:
        """Extract date filters from query
        
        Returns:
            Dict with year, month, quarter (None if not specified)
        """
        filters = {'year': None, 'month': None, 'quarter': None}
        
        question_lower = question.lower()
        
        # Extract year (2024, 2025, etc.)
        year_match = re.search(r'\b(20\d{2})\b', question)
        if year_match:
            filters['year'] = int(year_match.group(1))
        
        # Extract month
        for month_name, month_num in self.MONTH_NAMES.items():
            if month_name in question_lower:
                filters['month'] = month_num
                break
        
        # Extract quarter (Q1, Q2, Q3, Q4)
        quarter_match = re.search(r'\bq([1-4])\b', question_lower)
        if quarter_match:
            filters['quarter'] = int(quarter_match.group(1))
        
        return filters


# ============================================================================
# SQL QUERY BUILDER
# ============================================================================

class SQLQueryBuilder:
    """Builds optimized SQL queries for aggregation"""
    
    def __init__(self, config, spark):
        self.config = config
        self.spark = spark
    
    def build_aggregation_query(
        self,
        measure: str,
        entity_type: str,
        limit: int = 5,
        is_bottom: bool = False,
        date_filters: Dict[str, Any] = None
    ) -> str:
        """Build SQL aggregation query
        
        Args:
            measure: Measure column or function
            entity_type: Dimension table
            limit: Number of results
            is_bottom: Bottom-N instead of Top-N
            date_filters: Year/month/quarter filters
            
        Returns:
            SQL query string
        """
        
        # Get entity info
        entity_info = self._get_entity_info(entity_type)
        
        if not entity_info:
            raise ValueError(f"Unknown entity type: {entity_type}")
        
        id_col = entity_info['id_col']
        name_col = entity_info['name_col']
        fk_col = entity_info['fk_col']
        
        # Determine aggregation function
        if measure == 'COUNT(*)':
            agg_expr = 'COUNT(*)'
            agg_alias = 'transaction_count'
        else:
            agg_expr = f'SUM(f.{measure})'
            agg_alias = measure
        
        # Build WHERE clause for dates
        date_conditions = []
        if date_filters:
            if date_filters.get('year'):
                date_conditions.append(f"YEAR(f.{self.config.date_column}) = {date_filters['year']}")
            if date_filters.get('month'):
                date_conditions.append(f"MONTH(f.{self.config.date_column}) = {date_filters['month']}")
            if date_filters.get('quarter'):
                date_conditions.append(f"QUARTER(f.{self.config.date_column}) = {date_filters['quarter']}")
        
        where_clause = " AND ".join(date_conditions) if date_conditions else "1=1"
        
        # Build query
        order_direction = "ASC" if is_bottom else "DESC"
        
        query = f"""
        SELECT 
            d.{id_col},
            d.{name_col},
            {agg_expr} as {agg_alias}
        FROM {self.config.full_fact_table} f
        JOIN {self.config.get_full_table_name(entity_type)} d 
            ON f.{fk_col} = d.{id_col}
        WHERE {where_clause}
        GROUP BY d.{id_col}, d.{name_col}
        ORDER BY {agg_alias} {order_direction}
        LIMIT {limit}
        """
        
        return query
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return pandas DataFrame"""
        return self.spark.sql(sql).toPandas()
    
    def _get_entity_info(self, entity_type: str) -> Optional[Dict[str, str]]:
        """Get ID and name columns for entity type"""
        
        if 'customer' in entity_type.lower():
            return {
                'id_col': 'customer_id',
                'name_col': 'customer_name',
                'fk_col': 'customer_id'
            }
        elif 'item' in entity_type.lower():
            return {
                'id_col': 'item_id',
                'name_col': 'item_name',
                'fk_col': 'item_id'
            }
        elif 'location' in entity_type.lower():
            return {
                'id_col': 'location_id',
                'name_col': 'location_name',
                'fk_col': 'location_id'
            }
        
        return None


# ============================================================================
# PATTERN QUERY ENGINE
# ============================================================================

class PatternQueryEngine:
    """Executes pattern/relationship queries using graph traversal or SQL"""
    
    def __init__(self, graph: nx.MultiDiGraph, config, spark):
        self.graph = graph
        self.config = config
        self.spark = spark
    
    def execute_also_bought_query(
        self,
        product_a: str,
        product_b: str = None,
        date_filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Find customers who bought products (with optional date filter)"""
        
        # If date filters, use SQL (faster)
        if date_filters and any(date_filters.values()):
            return self._execute_via_sql(product_a, product_b, date_filters)
        else:
            return self._execute_via_graph(product_a, product_b)
    
    def _execute_via_graph(self, product_a: str, product_b: str = None):
        """Execute via graph traversal (no date filters)"""
        
        # Find product A nodes
        product_a_nodes = self._find_product_nodes(product_a)
        
        if not product_a_nodes:
            return {'error': f"Product '{product_a}' not found", 'customers': [], 'count': 0}
        
        # Get customers who bought product A
        customers_with_a = set()
        for prod_node in product_a_nodes:
            neighbors = set(self.graph.neighbors(prod_node))
            predecessors = set(self.graph.predecessors(prod_node))
            all_connected = neighbors.union(predecessors)
            
            for node in all_connected:
                if node.startswith('customer_details_'):
                    customers_with_a.add(node)
        
        # If product B specified, find intersection
        if product_b:
            product_b_nodes = self._find_product_nodes(product_b)
            if not product_b_nodes:
                return {'error': f"Product '{product_b}' not found", 'customers': [], 'count': 0}
            
            customers_with_b = set()
            for prod_node in product_b_nodes:
                neighbors = set(self.graph.neighbors(prod_node))
                predecessors = set(self.graph.predecessors(prod_node))
                all_connected = neighbors.union(predecessors)
                
                for node in all_connected:
                    if node.startswith('customer_details_'):
                        customers_with_b.add(node)
            
            customers_with_both = customers_with_a.intersection(customers_with_b)
            
            customer_details = [
                {
                    'customer_id': self.graph.nodes[c].get('customer_id'),
                    'customer_name': self.graph.nodes[c].get('customer_name'),
                    'customer_email': self.graph.nodes[c].get('customer_email')
                }
                for c in customers_with_both
            ]
            
            return {
                'product_a': product_a,
                'product_b': product_b,
                'customers': customer_details,
                'count': len(customer_details),
                'total_customers_with_a': len(customers_with_a),
                'total_customers_with_b': len(customers_with_b)
            }
        else:
            customer_details = [
                {
                    'customer_id': self.graph.nodes[c].get('customer_id'),
                    'customer_name': self.graph.nodes[c].get('customer_name'),
                    'customer_email': self.graph.nodes[c].get('customer_email')
                }
                for c in customers_with_a
            ]
            
            return {'product_a': product_a, 'customers': customer_details, 'count': len(customer_details)}
    
    def _execute_via_sql(self, product_a: str, product_b: str = None, date_filters: Dict = None):
        """Execute via SQL (with date filters)"""
        
        # Build date WHERE clause
        date_conditions = []
        if date_filters:
            if date_filters.get('year'):
                date_conditions.append(f"YEAR(s.{self.config.date_column}) = {date_filters['year']}")
            if date_filters.get('month'):
                date_conditions.append(f"MONTH(s.{self.config.date_column}) = {date_filters['month']}")
            if date_filters.get('quarter'):
                date_conditions.append(f"QUARTER(s.{self.config.date_column}) = {date_filters['quarter']}")
        
        date_where = " AND ".join(date_conditions) if date_conditions else "1=1"
        
        if product_b:
            # Both products
            query = f"""
            SELECT DISTINCT c.customer_id, c.customer_name, c.customer_email
            FROM {self.config.full_fact_table} s1
            JOIN {self.config.full_fact_table} s2 ON s1.customer_id = s2.customer_id
            JOIN {self.config.get_full_table_name('customer_details')} c ON s1.customer_id = c.customer_id
            JOIN {self.config.get_full_table_name('item_details')} i1 ON s1.item_id = i1.item_id
            JOIN {self.config.get_full_table_name('item_details')} i2 ON s2.item_id = i2.item_id
            WHERE (i1.item_id = '{product_a}' OR i1.item_name LIKE '%{product_a}%')
              AND (i2.item_id = '{product_b}' OR i2.item_name LIKE '%{product_b}%')
              AND s1.item_id != s2.item_id
              AND {date_where.replace('s.', 's1.')}
              AND {date_where.replace('s.', 's2.')}
            """
            
            result_df = self.spark.sql(query).toPandas()
            return {
                'product_a': product_a,
                'product_b': product_b,
                'customers': result_df.to_dict('records'),
                'count': len(result_df),
                'date_filters': date_filters
            }
        else:
            # Single product
            query = f"""
            SELECT DISTINCT c.customer_id, c.customer_name, c.customer_email
            FROM {self.config.full_fact_table} s
            JOIN {self.config.get_full_table_name('customer_details')} c ON s.customer_id = c.customer_id
            JOIN {self.config.get_full_table_name('item_details')} i ON s.item_id = i.item_id
            WHERE (i.item_id = '{product_a}' OR i.item_name LIKE '%{product_a}%')
              AND {date_where}
            """
            
            result_df = self.spark.sql(query).toPandas()
            return {
                'product_a': product_a,
                'customers': result_df.to_dict('records'),
                'count': len(result_df),
                'date_filters': date_filters
            }
    
    def _find_product_nodes(self, product_ref: str) -> List[str]:
        """Find product nodes by ID or name"""
        matching = []
        for node, attrs in self.graph.nodes(data=True):
            if not node.startswith('item_details_'):
                continue
            if (attrs.get('item_id', '').lower() == product_ref.lower() or
                attrs.get('item_name', '').lower() == product_ref.lower() or
                product_ref.lower() in attrs.get('item_name', '').lower()):
                matching.append(node)
        return matching


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'QueryIntentClassifier',
    'SQLQueryBuilder',
    'PatternQueryEngine'
]
