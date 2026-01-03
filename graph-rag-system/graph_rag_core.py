# ============================================================================
# graph_rag_core.py
# Core GraphRAG classes and functionality
# ============================================================================

"""
Core module for Graph RAG system with hybrid retrieval methods.

This module contains the foundational classes:
- GraphRAGConfig: Configuration management
- KnowledgeGraphBuilder: Graph construction from dimensional models
- EmbeddingGenerator: Semantic vector generation
- SemanticSearchEngine: Vector-based retrieval
- GraphTraversalEngine: Multi-hop graph traversal
- AnswerGenerator: Natural language response formatting
- GraphRAGSystem: Main orchestration class
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GraphRAGConfig:
    """Configuration for Graph RAG system
    
    This is the ONLY class users need to modify for different datasets.
    All other components auto-configure based on these settings.
    
    Attributes:
        catalog: Unity Catalog name
        schema: Schema name
        fact_table: Fact table name (sales, transactions, etc.)
        dimension_tables: List of dimension table names
        fk_mappings: Foreign key relationships
        embedding_model: SentenceTransformer model name
        top_k_nodes: Number of nodes to return in semantic search
        max_hops: Maximum graph traversal depth
        date_column: Date column in fact table for time filtering
    """
    
    # Required: Data location
    catalog: str
    schema: str
    fact_table: str
    dimension_tables: List[str]
    fk_mappings: Dict[str, Dict[str, str]]
    
    # Optional: Model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_nodes: int = 5
    max_hops: int = 2
    date_column: str = "sale_date"
    
    # Embedding model presets
    EMBEDDING_MODELS = {
        # Fast, lightweight (default)
        "mini": {
            "name": "all-MiniLM-L6-v2",
            "dimensions": 384,
            "max_tokens": 256,
            "description": "Fast, lightweight. Good for small-medium graphs (<100K nodes)",
            "speed": "⚡⚡⚡",
            "quality": "⭐⭐⭐"
        },
        # High quality, recommended for Graph RAG
        "bge-m3": {
            "name": "BAAI/bge-m3",
            "dimensions": 1024,
            "max_tokens": 8192,
            "description": "High quality, multi-lingual. Best for Graph RAG applications",
            "speed": "⚡",
            "quality": "⭐⭐⭐⭐⭐"
        },
        # Balanced option
        "bge-base": {
            "name": "BAAI/bge-base-en-v1.5",
            "dimensions": 768,
            "max_tokens": 512,
            "description": "Balanced speed and quality",
            "speed": "⚡⚡",
            "quality": "⭐⭐⭐⭐"
        },
        # Large, highest quality
        "bge-large": {
            "name": "BAAI/bge-large-en-v1.5",
            "dimensions": 1024,
            "max_tokens": 512,
            "description": "Highest quality for English text",
            "speed": "⚡",
            "quality": "⭐⭐⭐⭐⭐"
        },
        # Alternative fast option
        "e5-small": {
            "name": "intfloat/e5-small-v2",
            "dimensions": 384,
            "max_tokens": 512,
            "description": "Fast, good quality alternative to MiniLM",
            "speed": "⚡⚡⚡",
            "quality": "⭐⭐⭐"
        }
    }
    
    @classmethod
    def list_models(cls):
        """Print available embedding models"""
        print("\n" + "="*80)
        print("AVAILABLE EMBEDDING MODELS")
        print("="*80)
        
        for key, info in cls.EMBEDDING_MODELS.items():
            print(f"\n {key.upper()}: {info['name']}")
            print(f"   Dimensions: {info['dimensions']}")
            print(f"   Max Tokens: {info['max_tokens']}")
            print(f"   Speed: {info['speed']} | Quality: {info['quality']}")
            print(f"   {info['description']}")
        
        print("\n" + "="*80)
        print("USAGE:")
        print("="*80)
        print('  config = GraphRAGConfig(..., embedding_model="bge-m3")')
        print('  config = GraphRAGConfig(..., embedding_model="BAAI/bge-m3")')
        print("\n")
    
    def __post_init__(self):
        """Auto-resolve model name from preset or use custom"""
        if self.embedding_model in self.EMBEDDING_MODELS:
            # Use preset
            preset = self.EMBEDDING_MODELS[self.embedding_model]
            self.embedding_model = preset['name']
            print(f" Using preset '{self.embedding_model}' model")
            print(f"   Dimensions: {preset['dimensions']} | Max tokens: {preset['max_tokens']}")
            print(f"   {preset['description']}")
        else:
            # Custom model name
            print(f" Using custom model: {self.embedding_model}")
    
    # Computed properties
    @property
    def full_fact_table(self) -> str:
        """Full qualified fact table name"""
        return f"{self.catalog}.{self.schema}.{self.fact_table}"
    
    def get_full_table_name(self, table: str) -> str:
        """Get fully qualified table name"""
        return f"{self.catalog}.{self.schema}.{table}"


# ============================================================================
# KNOWLEDGE GRAPH BUILDER
# ============================================================================

class KnowledgeGraphBuilder:
    """Builds knowledge graph from dimensional data model
    
    Automatically constructs a graph where:
    - Nodes = Dimension table records (customers, products, locations)
    - Edges = Fact table relationships (purchases, transactions)
    - Attributes = All columns from source tables
    """
    
    def __init__(self, config: GraphRAGConfig, spark):
        self.config = config
        self.spark = spark
        self.graph = nx.MultiDiGraph()
    
    def build(self) -> nx.MultiDiGraph:
        """Build complete knowledge graph"""
        print(f"\n Building Knowledge Graph...")
        print(f"   Catalog: {self.config.catalog}")
        print(f"   Schema: {self.config.schema}")
        
        # Add dimension nodes
        for dim_table in self.config.dimension_tables:
            self._add_dimension_nodes(dim_table)
        
        # Add fact edges
        self._add_fact_edges()
        
        print(f"\n Graph built successfully!")
        print(f"   Total Nodes: {self.graph.number_of_nodes()}")
        print(f"   Total Edges: {self.graph.number_of_edges()}")
        
        return self.graph
    
    def _add_dimension_nodes(self, table_name: str):
        """Add nodes from dimension table"""
        full_table = self.config.get_full_table_name(table_name)
        df = self.spark.table(full_table).toPandas()
        
        # Detect ID column (first column is typically the ID)
        id_column = df.columns[0]
        
        for _, row in df.iterrows():
            node_id = f"{table_name}_{row[id_column]}"
            attrs = row.to_dict()
            attrs['_table'] = table_name
            attrs['_id_column'] = id_column
            
            self.graph.add_node(node_id, **attrs)
        
        print(f"   ✓ Added {len(df)} nodes from {table_name}")
    
    def _add_fact_edges(self):
        """Add edges from fact table"""
        full_table = self.config.full_fact_table
        df = self.spark.table(full_table).toPandas()
        
        # Get FK mappings for fact table
        fk_mappings = self.config.fk_mappings.get(self.config.fact_table, {})
        
        edge_count = 0
        for _, row in df.iterrows():
            # Create edges between all dimension nodes connected via this fact
            fact_attrs = row.to_dict()
            connected_nodes = []
            
            for fk_col, dim_table in fk_mappings.items():
                if fk_col in row:
                    node_id = f"{dim_table}_{row[fk_col]}"
                    if self.graph.has_node(node_id):
                        connected_nodes.append(node_id)
            
            # Create edges between all pairs of connected nodes
            for i, node1 in enumerate(connected_nodes):
                for node2 in connected_nodes[i+1:]:
                    self.graph.add_edge(node1, node2, **fact_attrs)
                    edge_count += 1
        
        print(f"   ✓ Added {edge_count} edges from {self.config.fact_table}")


# ============================================================================
# EMBEDDING GENERATOR
# ============================================================================

class EmbeddingGenerator:
    """Generate semantic embeddings for graph nodes"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.model = None
        self.node_embeddings = {}
        self.node_texts = {}
    
    def generate_embeddings(self, graph: nx.MultiDiGraph):
        """Generate embeddings for all nodes"""
        print(f"\n Loading embedding model: {self.config.embedding_model}")
        
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.config.embedding_model)
        
        print(f" Generating embeddings...")
        
        # Prepare text for each node
        texts = []
        node_ids = []
        
        for node_id, attrs in graph.nodes(data=True):
            # Combine all text attributes
            text_parts = []
            for key, value in attrs.items():
                if not key.startswith('_') and value is not None:
                    text_parts.append(f"{key}: {value}")
            
            node_text = " | ".join(text_parts)
            texts.append(node_text)
            node_ids.append(node_id)
            self.node_texts[node_id] = node_text
        
        # Generate embeddings in batch
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Store embeddings
        for node_id, embedding in zip(node_ids, embeddings):
            self.node_embeddings[node_id] = embedding
        
        print(f" Generated embeddings for {len(self.node_embeddings)} nodes")


# ============================================================================
# SEMANTIC SEARCH ENGINE
# ============================================================================

class SemanticSearchEngine:
    """Vector-based semantic search over graph nodes"""
    
    def __init__(self, config: GraphRAGConfig, embedding_gen: EmbeddingGenerator):
        self.config = config
        self.embedding_gen = embedding_gen
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """Search for semantically similar nodes
        
        Args:
            query: Natural language query
            top_k: Number of results (defaults to config.top_k_nodes)
            
        Returns:
            List of (node_id, similarity_score) tuples
        """
        if top_k is None:
            top_k = self.config.top_k_nodes
        
        # Encode query
        query_embedding = self.embedding_gen.model.encode([query])[0]
        
        # Compute similarities
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = []
        for node_id, node_embedding in self.embedding_gen.node_embeddings.items():
            sim = cosine_similarity(
                [query_embedding], 
                [node_embedding]
            )[0][0]
            similarities.append((node_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


# ============================================================================
# GRAPH TRAVERSAL ENGINE
# ============================================================================

class GraphTraversalEngine:
    """Multi-hop graph traversal for context expansion"""
    
    def __init__(self, config: GraphRAGConfig, graph: nx.MultiDiGraph):
        self.config = config
        self.graph = graph
    
    def expand_context(self, node_ids: List[str], max_hops: int = None) -> List[str]:
        """Expand context by traversing graph from seed nodes
        
        Args:
            node_ids: Starting nodes
            max_hops: Maximum traversal depth (defaults to config.max_hops)
            
        Returns:
            List of all reachable node IDs
        """
        if max_hops is None:
            max_hops = self.config.max_hops
        
        visited = set(node_ids)
        current_layer = set(node_ids)
        
        for hop in range(max_hops):
            next_layer = set()
            
            for node in current_layer:
                if node in self.graph:
                    # Get neighbors (both directions)
                    neighbors = set(self.graph.neighbors(node))
                    predecessors = set(self.graph.predecessors(node))
                    all_connected = neighbors.union(predecessors)
                    
                    for neighbor in all_connected:
                        if neighbor not in visited:
                            next_layer.add(neighbor)
                            visited.add(neighbor)
            
            current_layer = next_layer
            
            if not current_layer:
                break
        
        return list(visited)


# ============================================================================
# ANSWER GENERATOR
# ============================================================================

class AnswerGenerator:
    """Generate natural language answers from retrieved data"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
    
    def format_semantic_results(self, results: List[Tuple[str, float]]) -> str:
        """Format semantic search results as natural language
        
        Args:
            results: List of (node_id, similarity_score)
            
        Returns:
            Natural language answer
        """
        if not results:
            return "\nNo relevant results found."
        
        answer_parts = ["\nMost relevant items:\n"]
        
        for node_id, score in results:
            if node_id in self.graph.nodes:
                attrs = self.graph.nodes[node_id]
                
                # Format node info
                info_parts = []
                for key, value in attrs.items():
                    if not key.startswith('_') and value is not None:
                        info_parts.append(f"{key}: {value}")
                
                node_info = ", ".join(info_parts)
                answer_parts.append(f"- {node_info} (relevance: {score:.3f})")
        
        return "\n".join(answer_parts)
    
    def format_sql_results(self, df: pd.DataFrame, query_desc: str = "") -> str:
        """Format SQL results as natural language
        
        Args:
            df: Results DataFrame
            query_desc: Query description
            
        Returns:
            Natural language answer
        """
        if df.empty:
            return f"\nNo results found for: {query_desc}"
        
        answer_parts = []
        if query_desc:
            answer_parts.append(f"\n{query_desc}:\n")
        else:
            answer_parts.append("\nQuery results:\n")
        
        for idx, row in df.iterrows():
            row_parts = []
            for col, val in row.items():
                if pd.notna(val):
                    # Format numbers nicely
                    if isinstance(val, (int, float)):
                        if col.lower() in ['revenue', 'sales', 'total', 'amount', 'value']:
                            row_parts.append(f"{col}: ${val:,.2f}")
                        else:
                            row_parts.append(f"{col}: {val:,}")
                    else:
                        row_parts.append(f"{col}: {val}")
            
            answer_parts.append(f"{idx + 1}. {', '.join(row_parts)}")
        
        return "\n".join(answer_parts)


# ============================================================================
# MAIN GRAPH RAG SYSTEM
# ============================================================================

class GraphRAGSystem:
    """Main Graph RAG orchestration system
    
    Coordinates all components and provides unified query interface.
    Automatically routes queries to appropriate retrieval method.
    """
    
    def __init__(self, config: GraphRAGConfig, spark):
        self.config = config
        self.spark = spark
        
        # Components (initialized during build)
        self.graph_builder = None
        self.graph = None
        self.embedding_gen = None
        self.semantic_search = None
        self.graph_traversal = None
        self.answer_gen = None
        
        # Will be added by enhancement modules
        self.intent_classifier = None
        self.sql_builder = None
        self.pattern_engine = None
    
    def build(self):
        """Build the complete Graph RAG system"""
        print(f"\n{'='*80}")
        print(f"BUILDING GRAPH RAG SYSTEM")
        print(f"{'='*80}")
        
        # Step 1: Build knowledge graph
        self.graph_builder = KnowledgeGraphBuilder(self.config, self.spark)
        self.graph = self.graph_builder.build()
        
        # Step 2: Generate embeddings
        self.embedding_gen = EmbeddingGenerator(self.config)
        self.embedding_gen.generate_embeddings(self.graph)
        
        # Step 3: Initialize search engines
        self.semantic_search = SemanticSearchEngine(self.config, self.embedding_gen)
        self.graph_traversal = GraphTraversalEngine(self.config, self.graph)
        self.answer_gen = AnswerGenerator(self.graph)
        
        print(f"\n{'='*80}")
        print(f" GRAPH RAG SYSTEM READY")
        print(f"{'='*80}\n")
    
    def query(self, question: str, verbose: bool = True) -> str:
        """Execute query using semantic search (base implementation)
        
        This is the base query method. Enhanced versions with SQL,
        pattern matching, etc. are added by enhancement modules.
        
        Args:
            question: Natural language question
            verbose: Print detailed execution info
            
        Returns:
            Natural language answer
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f" Query: {question}")
            print(f"{'='*80}")
            print(f"\n Query Intent: semantic_search")
        
        # Execute semantic search
        results = self.semantic_search.search(question)
        
        if verbose:
            print(f"\n Executing semantic search...")
            print(f"   Top {len(results)} similar nodes:")
            for node_id, score in results:
                print(f"   - {node_id}: {score:.3f}")
        
        # Generate answer
        answer = self.answer_gen.format_semantic_results(results)
        
        if verbose:
            print(f"\n💡 Answer:")
            print(answer)
            print(f"\n{'='*80}\n")
        
        return answer
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'embedding_dim': len(list(self.embedding_gen.node_embeddings.values())[0]),
            'dimension_tables': self.config.dimension_tables,
            'fact_table': self.config.fact_table
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'GraphRAGConfig',
    'KnowledgeGraphBuilder',
    'EmbeddingGenerator',
    'SemanticSearchEngine',
    'GraphTraversalEngine',
    'AnswerGenerator',
    'GraphRAGSystem'
]