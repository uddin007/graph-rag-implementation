# ============================================================================
# graph_rag_enhancements.py
# Enhancement functions to add capabilities to GraphRAGSystem
# ============================================================================

"""
Enhancement functions for Graph RAG system.

These functions add advanced capabilities to the base GraphRAGSystem:

- enhance_with_sql_queries(): Adds SQL aggregation support
- enhance_with_pattern_queries(): Adds relationship/pattern query support
- enhance_with_all(): Applies all enhancements at once

Usage:
    system = GraphRAGSystem(config, spark)
    system.build()
    enhance_with_all(system)  # Add all capabilities
"""

import re
import mlflow
import mlflow.pyfunc
import pickle
import tempfile
import os
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from graph_rag_intelligence import (
    QueryIntentClassifier,
    SQLQueryBuilder,
    PatternQueryEngine
)


# ============================================================================
# SQL QUERY ENHANCEMENT
# ============================================================================

def enhance_with_sql_queries(rag_system):
    """Add SQL aggregation query support to GraphRAG system
    
    After this enhancement, the system can handle:
    - "Top 10 customers by revenue"
    - "Bottom 5 items by sales"
    - "Top items in December 2025"
    
    Args:
        rag_system: GraphRAGSystem instance
    """
    
    # Add components
    rag_system.intent_classifier = QueryIntentClassifier(rag_system.config)
    rag_system.sql_builder = SQLQueryBuilder(rag_system.config, rag_system.spark)
    
    # Store original query method
    original_query = rag_system.query
    
    def enhanced_query(question: str, verbose: bool = True) -> str:
        """Enhanced query with SQL aggregation support"""
        
        # Classify intent
        intent = rag_system.intent_classifier.classify(question)
        
        if intent['is_aggregation'] and intent['measure'] and intent['entity_type']:
            # Route to SQL aggregation
            
            if verbose:
                print(f"\n{'='*80}")
                print(f" Query: {question}")
                print(f"{'='*80}")
                print(f"\n Query Intent: aggregation")
                print(f"   Measure: {intent['measure']}")
                print(f"   Entity: {intent['entity_type']}")
                print(f"   Limit: {intent['limit']}")
                print(f"   Bottom-N: {intent['is_bottom']}")
                if intent['date_filters'] and any(intent['date_filters'].values()):
                    print(f"   Date Filters: {intent['date_filters']}")
            
            # Build and execute SQL
            sql = rag_system.sql_builder.build_aggregation_query(
                measure=intent['measure'],
                entity_type=intent['entity_type'],
                limit=intent['limit'],
                is_bottom=intent['is_bottom'],
                date_filters=intent['date_filters']
            )
            
            if verbose:
                print(f"\n Executing SQL query...")
            
            result_df = rag_system.sql_builder.execute_query(sql)
            
            # Generate answer
            query_desc = f"{'Bottom' if intent['is_bottom'] else 'Top'} {intent['limit']} results"
            answer = rag_system.answer_gen.format_sql_results(result_df, query_desc)
            
            if verbose:
                print(f"\n💡 Answer:")
                print(answer)
                print(f"\n{'='*80}\n")
            
            return answer
        
        else:
            # Route to semantic search (original method)
            return original_query(question, verbose)
    
    # Replace query method
    rag_system.query = enhanced_query
    
    print(" SQL aggregation queries enabled")


# ============================================================================
# PATTERN QUERY ENHANCEMENT
# ============================================================================

def enhance_with_pattern_queries(rag_system):
    """Add pattern/relationship query support to GraphRAG system
    
    After this enhancement, the system can handle:
    - "Which customers bought Product A?"
    - "Which customers bought A and B?"
    - "Customers who bought A and B in December 2025"
    
    Args:
        rag_system: GraphRAGSystem instance
    """
    
    # Add pattern engine
    rag_system.pattern_engine = PatternQueryEngine(
        rag_system.graph,
        rag_system.config,
        rag_system.spark
    )
    
    # Store original query method
    original_query = rag_system.query
    
    def enhanced_query(question: str, verbose: bool = True) -> str:
        """Enhanced query with pattern support"""
        
        question_lower = question.lower()
        
        # Check if pattern query
        is_pattern = any([
            'also bought' in question_lower,
            'also purchased' in question_lower,
            'which customers bought' in question_lower,
            'which customers purchased' in question_lower,
            'customers who bought' in question_lower
        ])
        
        if is_pattern:
            
            if verbose:
                print(f"\n{'='*80}")
                print(f" Query: {question}")
                print(f"{'='*80}")
                print(f"\n Query Intent: pattern_relationship")
            
            # Extract products
            products = []
            
            # Try quoted
            quoted = re.findall(r'["\']([^"\']+)["\']', question)
            if quoted:
                products = quoted
            
            # Try product IDs
            if not products:
                ids = re.findall(r'\b([a-z]-\d+)\b', question, re.IGNORECASE)
                if ids:
                    products = ids
            
            # Try capitalized (filtered)
            if not products:
                caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question)
                products = [p for p in caps if p.lower() not in 
                           ['which', 'customers', 'product', 'item', 'bought', 'purchased', 'also', 'during', 'in']]
            
            # Extract date filters
            date_filters = None
            if hasattr(rag_system, 'intent_classifier'):
                date_filters = rag_system.intent_classifier._extract_date_filters(question)
            
            if len(products) >= 2:
                product_a, product_b = products[0], products[1]
                
                if verbose:
                    print(f"   Product A: {product_a}")
                    print(f"   Product B: {product_b}")
                    if date_filters and any(date_filters.values()):
                        print(f"   Date Filters: {date_filters}")
                    print(f"\n Executing pattern query...")
                
                result = rag_system.pattern_engine.execute_also_bought_query(
                    product_a=product_a,
                    product_b=product_b,
                    date_filters=date_filters
                )
                
            elif len(products) == 1:
                product_a = products[0]
                
                if verbose:
                    print(f"   Product: {product_a}")
                    if date_filters and any(date_filters.values()):
                        print(f"   Date Filters: {date_filters}")
                    print(f"\n🔍 Executing pattern query...")
                
                result = rag_system.pattern_engine.execute_also_bought_query(
                    product_a=product_a,
                    date_filters=date_filters
                )
            else:
                return "\n Could not identify products. Use format: 'Which customers bought \"Product A\"?' or 'Which customers bought p-20?'"
            
            # Format answer
            answer = _format_also_bought_answer(result)
            
            if verbose:
                print(f"\n Answer:")
                print(answer)
                print(f"\n{'='*80}\n")
            
            return answer
        
        else:
            # Not a pattern query, use original
            return original_query(question, verbose)
    
    # Replace query method
    rag_system.query = enhanced_query
    
    print(" Pattern relationship queries enabled")


def _format_also_bought_answer(result: dict) -> str:
    """Format also bought query results"""
    
    if 'error' in result:
        return f"\n {result['error']}"
    
    # Build date description
    date_desc = ""
    if result.get('date_filters'):
        df = result['date_filters']
        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        if df.get('month') and df.get('year'):
            date_desc = f" in {month_names[df['month']]} {df['year']}"
        elif df.get('year'):
            date_desc = f" in {df['year']}"
        elif df.get('quarter') and df.get('year'):
            date_desc = f" in Q{df['quarter']} {df['year']}"
    
    if 'product_b' in result:
        answer_parts = [
            f"\n Customers who bought both '{result['product_a']}' AND '{result['product_b']}'{date_desc}:\n"
        ]
        
        if result['count'] == 0:
            answer_parts.append(f"No customers found who purchased both products{date_desc}.")
            if 'total_customers_with_a' in result:
                answer_parts.append(f"\nNote:")
                answer_parts.append(f"  • {result['total_customers_with_a']} customers bought '{result['product_a']}'{date_desc}")
                answer_parts.append(f"  • {result['total_customers_with_b']} customers bought '{result['product_b']}'{date_desc}")
        else:
            answer_parts.append(f"Found {result['count']} customers:\n")
            for i, cust in enumerate(result['customers'][:10], 1):
                answer_parts.append(
                    f"{i}. {cust['customer_name']} (ID: {cust['customer_id']}) - {cust['customer_email']}"
                )
            
            if result['count'] > 10:
                answer_parts.append(f"\n... and {result['count'] - 10} more customers")
    else:
        answer_parts = [
            f"\n Customers who bought '{result['product_a']}'{date_desc}:\n"
        ]
        
        if result['count'] == 0:
            answer_parts.append(f"No customers found who purchased this product{date_desc}.")
        else:
            answer_parts.append(f"Found {result['count']} customers:\n")
            for i, cust in enumerate(result['customers'][:10], 1):
                answer_parts.append(
                    f"{i}. {cust['customer_name']} (ID: {cust['customer_id']}) - {cust['customer_email']}"
                )
            
            if result['count'] > 10:
                answer_parts.append(f"\n... and {result['count'] - 10} more customers")
    
    return "\n".join(answer_parts)


# ============================================================================
# COMBINED ENHANCEMENT
# ============================================================================

def enhance_with_all(rag_system):
    """Apply all enhancements to GraphRAG system
    
    This is the recommended way to enhance the system.
    Adds all capabilities in the correct order.
    
    Args:
        rag_system: GraphRAGSystem instance
    """
    
    print("\n🔧 Applying all enhancements...")
    
    enhance_with_sql_queries(rag_system)
    enhance_with_pattern_queries(rag_system)
    
    print("\n All enhancements applied!")
    print("\nSupported query types:")
    print("  1. Semantic Search - 'Show me chocolate items'")
    print("  2. SQL Aggregation - 'Top 10 customers by revenue'")
    print("  3. Time-Based - 'Top items in December 2025'")
    print("  4. Pattern Queries - 'Customers who bought A and B'")
    print("  5. Time+Pattern - 'Customers who bought A and B in December 2025'")


# ============================================================================
# MLFLOW MODEL WRAPPER
# ============================================================================

class GraphRAGModel(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for Graph RAG system"""
    
    def load_context(self, context):
        """Load model artifacts"""
        import networkx as nx
        from sentence_transformers import SentenceTransformer
        
        # Load graph
        with open(context.artifacts["graph"], "rb") as f:
            self.graph = pickle.load(f)
        
        # Load embeddings
        with open(context.artifacts["embeddings"], "rb") as f:
            embeddings_data = pickle.load(f)
            self.node_embeddings = embeddings_data['embeddings']
            self.node_texts = embeddings_data['texts']
        
        # Load config
        with open(context.artifacts["config"], "rb") as f:
            self.config = pickle.load(f)
        
        # Initialize embedding model
        self.model = SentenceTransformer(self.config.embedding_model)
    
    def predict(self, context, model_input):
        """Make predictions"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        questions = model_input['question'].tolist()
        answers = []
        
        for question in questions:
            # Encode query
            query_embedding = self.model.encode([question])[0]
            
            # Compute similarities
            similarities = []
            for node_id, node_embedding in self.node_embeddings.items():
                sim = cosine_similarity([query_embedding], [node_embedding])[0][0]
                similarities.append((node_id, sim))
            
            # Sort and get top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:self.config.top_k_nodes]
            
            # Format answer
            answer_parts = ["Most relevant items:"]
            for node_id, score in top_results:
                if node_id in self.graph.nodes:
                    attrs = self.graph.nodes[node_id]
                    info_parts = [f"{k}: {v}" for k, v in attrs.items() 
                                 if not k.startswith('_') and v is not None]
                    answer_parts.append(f"- {', '.join(info_parts)} (relevance: {score:.3f})")
            
            answers.append("\n".join(answer_parts))
        
        return answers


def save_graph_rag_model(system, model_name: str = "graph_rag_model"):
    """Save Graph RAG system to MLflow with Unity Catalog support
    
    Args:
        system: GraphRAGSystem instance
        model_name: Model name for Unity Catalog
        
    Returns:
        Model URI
    """
    
    print(f"\n Saving model to MLflow...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save artifacts
        graph_path = os.path.join(tmpdir, "graph.pkl")
        with open(graph_path, "wb") as f:
            pickle.dump(system.graph, f)
        
        embeddings_path = os.path.join(tmpdir, "embeddings.pkl")
        with open(embeddings_path, "wb") as f:
            pickle.dump({
                'embeddings': system.embedding_gen.node_embeddings,
                'texts': system.embedding_gen.node_texts
            }, f)
        
        config_path = os.path.join(tmpdir, "config.pkl")
        with open(config_path, "wb") as f:
            pickle.dump(system.config, f)
        
        artifacts = {
            "graph": graph_path,
            "embeddings": embeddings_path,
            "config": config_path
        }
        
        # Create signature (required for Unity Catalog)
        input_schema = Schema([ColSpec("string", "question")])
        output_schema = Schema([ColSpec("string", "answer")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Create input example
        input_example = pd.DataFrame({"question": ["What are the top 5 items by revenue?"]})
        
        # Log model
        with mlflow.start_run() as run:
            mlflow.log_params({
                "catalog": system.config.catalog,
                "schema": system.config.schema,
                "fact_table": system.config.fact_table,
                "embedding_model": system.config.embedding_model,
                "top_k_nodes": system.config.top_k_nodes,
                "max_hops": system.config.max_hops
            })
            
            stats = system.get_statistics()
            mlflow.log_metrics({
                "num_nodes": stats['total_nodes'],
                "num_edges": stats['total_edges'],
                "embedding_dim": stats['embedding_dim']
            })
            
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=GraphRAGModel(),
                artifacts=artifacts,
                signature=signature,
                input_example=input_example,
                pip_requirements=[
                    "networkx>=3.0",
                    "pandas>=1.5.0",
                    "numpy>=1.24.0",
                    "sentence-transformers>=2.2.0",
                    "scikit-learn>=1.3.0"
                ]
            )
            
            print(f" Model saved: {model_info.model_uri}")
            
            # Register to Unity Catalog
            try:
                registered_model = mlflow.register_model(model_info.model_uri, model_name)
                print(f" Model registered as: {model_name}")
                print(f"   Version: {registered_model.version}")
            except Exception as e:
                print(f" Model registration note: {e}")
                print(f"   Model URI: {model_info.model_uri}")
            
            return model_info.model_uri


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'enhance_with_sql_queries',
    'enhance_with_pattern_queries',
    'enhance_with_all',
    'save_graph_rag_model',
    'GraphRAGModel'
]
