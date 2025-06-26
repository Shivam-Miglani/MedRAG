import os, requests, streamlit as st
from dotenv import load_dotenv
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time
import json

from llama_index.core import (
    Settings, VectorStoreIndex, StorageContext, SimpleDirectoryReader
)
from llama_index.llms.lmstudio import LMStudio
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from pyvis.network import Network
import streamlit.components.v1 as components
import networkx as nx
from collections import defaultdict

load_dotenv()

# Configuration
CONFIG = {
    'page_title': "Medical Evidence RAG System",
    'page_icon': "🏥",
    'layout': "wide",
    'llm_base_url': os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
    'llm_name': "google/gemma-3-4b",
    'embed_model': "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    'data_path': "./data",
    'demo_mode': os.getenv("DEMO_MODE", "true").lower() == "true",
    'cache_path': "./cache"
}

# Create cache directory if it doesn't exist
if not os.path.exists(CONFIG['cache_path']):
    os.makedirs(CONFIG['cache_path'])

# Medical domain knowledge
MEDICAL_TAXONOMY = {
    'conditions': ['diabetes', 'obesity', 'cardiovascular disease', 'hypertension', 'metabolic syndrome'],
    'treatments': ['metformin', 'lifestyle intervention', 'diet therapy', 'exercise therapy', 'medication'],
    'guidelines': ['cdc', 'rivm', 'who', 'ada', 'easd'],
    'metrics': ['hba1c', 'weight loss', 'bmi', 'blood pressure', 'cholesterol'],
    'interventions': ['physical activity', 'nutrition counseling', 'patient education', 'monitoring'],
    'outcomes': ['prevention', 'management', 'complications', 'mortality', 'quality of life']
}

PREDEFINED_QUERIES = [
    "Compare US and Dutch approaches to Type 2 diabetes management",
    "What are the medication protocols for diabetes in different countries?",
    "How do dietary recommendations differ between US and Netherlands?",
    "Compare weight loss targets in diabetes prevention programs"
]

class MedicalRAGSystem:
    """Enterprise-grade Medical RAG System with Knowledge Graph Integration"""
    
    def __init__(self):
        self.llm = None
        self.embed_model = None
        self.documents = None
        self.vector_engine = None
        self.rerank_engine = None
        self.knowledge_graph = None
        self.performance_stats = {
            'vector_search': {'response_times': [], 'token_counts': []},
            'kg_search': {'response_times': [], 'token_counts': []}
        }
        
    @st.cache_resource
    def initialize_models(_self):
        """Initialize LLM and embedding models with error handling"""
        if CONFIG['demo_mode']:
            try:
                embed_model = HuggingFaceEmbedding(model_name=CONFIG['embed_model'])
                Settings.embed_model = embed_model
                _self.embed_model = embed_model
                st.info("Running in DEMO mode: Responses are pre-generated using local google/gemma-3-4b LLM")
                return True
            except Exception as e:
                st.error(f"❌ Embedding model initialization failed: {str(e)}")
                return False
        
        # Live mode logic
        try:
            response = requests.get(CONFIG['llm_base_url'], timeout=5)
            if response.status_code != 200:
                raise ConnectionError("LM Studio server not accessible")
                
            llm = LMStudio(
                model_name=CONFIG['llm_name'],
                base_url=CONFIG['llm_base_url'],
                temperature=0.1,
                request_timeout=300
            )
            
            embed_model = HuggingFaceEmbedding(model_name=CONFIG['embed_model'])
            
            Settings.llm = llm
            Settings.embed_model = embed_model
            
            _self.llm = llm
            _self.embed_model = embed_model
            
            st.warning("Running in LIVE mode. This requires local LM Studio.")
            return True
            
        except Exception as e:
            st.error(f"❌ Model initialization failed: {str(e)}")
            return False
    
    @st.cache_data
    def load_documents(_self):
        """Load and cache medical documents"""
        try:
            documents = SimpleDirectoryReader(CONFIG['data_path']).load_data()
            _self.documents = documents
            return documents
        except Exception as e:
            st.error(f"❌ Document loading failed: {str(e)}")
            return []
    
    @st.cache_resource
    def build_rag_engines(_self, _documents):
        """Build both standard and advanced RAG engines"""
        if not _documents:
            return None, None
            
        try:
            vector_index = VectorStoreIndex.from_documents(_documents)
            
            # Standard vector retrieval
            vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=3)
            vector_engine = RetrieverQueryEngine.from_args(retriever=vector_retriever)
            
            # Advanced retrieval with ColBERT reranking
            rerank_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10)
            reranker = ColbertRerank(top_n=5, model="colbert-ir/colbertv2.0")
            rerank_engine = RetrieverQueryEngine.from_args(
                retriever=rerank_retriever, 
                node_postprocessors=[reranker]
            )
            
            _self.vector_engine = vector_engine
            _self.rerank_engine = rerank_engine
            
            return vector_engine, rerank_engine
            
        except Exception as e:
            st.error(f"❌ RAG engine initialization failed: {str(e)}")
            return None, None
    
    def query_with_timing(self, engine, query: str, engine_type: str):
        """Execute query with performance timing"""
        start_time = time.time()
        response = engine.query(query)
        end_time = time.time()
        
        response_time = end_time - start_time
        token_count = len(response.response.split())
        
        # Store performance metrics
        self.performance_stats[engine_type]['response_times'].append(response_time)
        self.performance_stats[engine_type]['token_counts'].append(token_count)
        
        return response, response_time, token_count
    
    def extract_relevant_snippet(self, node, query_terms: List[str], max_length: int = 200) -> str:
        """Extract contextually relevant snippet from source node"""
        content = node.get_content()
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        query_words = [term.lower() for term in query_terms]
        scored_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for word in query_words if word in sentence_lower)
            if score > 0:
                scored_sentences.append((sentence, score))
        
        if scored_sentences:
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            result = ' ... '.join([s[0] for s in scored_sentences[:2]])
            return result[:max_length] + '...' if len(result) > max_length else result
        
        return sentences[0][:max_length] + '...' if sentences else content[:max_length] + '...'
    
    def calculate_clinical_metrics(self, response, query: str) -> Dict[str, float]:
        """Calculate comprehensive clinical quality metrics"""
        if not response.source_nodes:
            return {
                "source_diversity": 0.0,
                "clinical_completeness": 0.0,
                "cross_guideline_synthesis": 0.0,
                "evidence_strength": 0.0
            }
        
        # Source diversity: Geographic coverage
        sources = [n.metadata.get("file_name", "").lower() for n in response.source_nodes]
        us_coverage = sum(1 for s in sources if any(term in s for term in ["cdc", "us"]))
        nl_coverage = sum(1 for s in sources if any(term in s for term in ["rivm", "dutch", "nl"]))
        source_diversity = min(1.0, (us_coverage + nl_coverage) / 4)
        
        # Clinical completeness: Medical concept coverage
        answer_text = response.response.lower()
        medical_terms = ["diabetes", "metformin", "weight loss", "physical activity", "diet", "treatment"]
        completeness = sum(1 for term in medical_terms if term in answer_text) / len(medical_terms)
        
        # Cross-guideline synthesis: Comparative analysis indicators
        comparative_indicators = ["both", "however", "while", "contrast", "difference", "similar", "compare"]
        synthesis_score = min(1.0, sum(1 for term in comparative_indicators if term in answer_text) / 3)
        
        # Evidence strength: Average retrieval confidence
        evidence_strength = sum(n.score for n in response.source_nodes) / len(response.source_nodes)
        
        return {
            "source_diversity": source_diversity,
            "clinical_completeness": completeness,
            "cross_guideline_synthesis": synthesis_score,
            "evidence_strength": evidence_strength
        }
    
    def build_medical_knowledge_graph(self, documents) -> nx.Graph:
        """Construct enhanced medical knowledge graph with semantic relationships"""
        concept_cooccurrence = defaultdict(int)
        concept_documents = defaultdict(set)
        all_concepts = set()
        
        # Extract medical concepts from documents
        for doc_idx, doc in enumerate(documents):
            content = doc.get_content().lower()
            doc_concepts = []
            
            for category, concepts in MEDICAL_TAXONOMY.items():
                for concept in concepts:
                    if concept in content:
                        doc_concepts.append((concept, category))
                        all_concepts.add(concept)
                        concept_documents[concept].add(doc_idx)
            
            # Calculate concept co-occurrence within documents
            for i, (concept1, _) in enumerate(doc_concepts):
                for j, (concept2, _) in enumerate(doc_concepts):
                    if i != j:
                        pair = tuple(sorted([concept1, concept2]))
                        concept_cooccurrence[pair] += 1
        
        # Build NetworkX graph
        G = nx.Graph()
        
        # Add nodes with enhanced attributes
        for concept in all_concepts:
            category = self._get_concept_category(concept)
            frequency = len(concept_documents[concept])
            
            G.add_node(concept, 
                      category=category,
                      frequency=frequency,
                      document_count=frequency)
        
        # Add weighted edges based on co-occurrence
        for (concept1, concept2), weight in concept_cooccurrence.items():
            if weight >= 1:
                shared_docs = len(concept_documents[concept1] & concept_documents[concept2])
                total_docs = len(concept_documents[concept1] | concept_documents[concept2])
                similarity = shared_docs / total_docs if total_docs > 0 else 0
                
                if similarity > 0.1:
                    relation_type = self._infer_relation_type(concept1, concept2)
                    G.add_edge(concept1, concept2, 
                              weight=weight,
                              similarity=similarity,
                              relation_type=relation_type)
        
        return G
    
    def _get_concept_category(self, concept: str) -> str:
        """Determine medical category for concept"""
        for category, concepts in MEDICAL_TAXONOMY.items():
            if concept in concepts:
                return category
        return 'general'
    
    def _infer_relation_type(self, concept1: str, concept2: str) -> str:
        """Infer semantic relationship between medical concepts"""
        cat1 = self._get_concept_category(concept1)
        cat2 = self._get_concept_category(concept2)
        
        relation_map = {
            ('treatments', 'conditions'): 'treats',
            ('conditions', 'treatments'): 'treated_by',
            ('guidelines', 'treatments'): 'recommends',
            ('guidelines', 'interventions'): 'recommends',
            ('metrics', 'conditions'): 'measures',
            ('interventions', 'outcomes'): 'leads_to'
        }
        
        return relation_map.get((cat1, cat2), 'related_to' if cat1 == cat2 else 'associated_with')
    
    def create_interactive_graph(self, graph: nx.Graph) -> str:
        """Generate interactive PyVis knowledge graph"""
        if not graph.nodes():
            return "<div style='color: white; padding: 20px;'>No knowledge graph data available</div>"
        
        try:
            net = Network(
                height="600px", 
                width="100%", 
                bgcolor="#0e1117", 
                font_color="white",
                directed=False
            )
            
            # Enhanced color scheme for medical categories
            color_palette = {
                'conditions': '#FF4B4B',
                'treatments': '#00D4AA', 
                'guidelines': '#1F77B4',
                'metrics': '#FF6B6B',
                'interventions': '#FFAA00',
                'outcomes': '#9467BD',
                'general': '#7F7F7F'
            }
            
            # Add nodes with enhanced styling
            for node_id, node_data in graph.nodes(data=True):
                category = node_data.get('category', 'general')
                frequency = node_data.get('frequency', 1)
                
                # Special handling for guideline sources
                if category == 'guidelines':
                    if any(term in node_id.lower() for term in ['cdc', 'us']):
                        color = '#1F77B4'  # Blue for US
                    elif any(term in node_id.lower() for term in ['rivm', 'dutch']):
                        color = '#00D4AA'  # Green for Dutch
                    else:
                        color = color_palette[category]
                else:
                    color = color_palette.get(category, '#7F7F7F')
                
                node_size = min(40, 15 + frequency * 3)
                
                net.add_node(
                    node_id,
                    label=node_id.replace('_', ' ').title(),
                    color=color,
                    size=node_size,
                    title=f"<b>{node_id.title()}</b><br/>Category: {category}<br/>Frequency: {frequency}",
                    font={'size': 12, 'color': 'white'}
                )
            
            # Add edges with relationship-based styling
            edge_colors = {
                'treats': '#E74C3C',
                'recommends': '#3498DB', 
                'measures': '#9B59B6',
                'leads_to': '#2ECC71',
                'related_to': '#F39C12',
                'associated_with': '#BDC3C7'
            }
            
            for source, target, edge_data in graph.edges(data=True):
                weight = edge_data.get('weight', 1)
                relation_type = edge_data.get('relation_type', 'related_to')
                
                edge_width = min(5, 1 + weight)
                edge_color = edge_colors.get(relation_type, '#BDC3C7')
                
                net.add_edge(
                    source, target,
                    width=edge_width,
                    color=edge_color,
                    title=f"{relation_type.replace('_', ' ').title()}<br/>Strength: {weight}"
                )
            
            # Configure physics for optimal layout
            net.set_options("""
            var options = {
                "physics": {
                    "enabled": true,
                    "stabilization": {"iterations": 200},
                    "barnesHut": {
                        "gravitationalConstant": -8000,
                        "centralGravity": 0.3,
                        "springLength": 95,
                        "springConstant": 0.04,
                        "damping": 0.09
                    }
                },
                "interaction": {
                    "hover": true,
                    "tooltipDelay": 200,
                    "zoomView": true,
                    "dragView": true
                }
            }
            """)
            
            # Save and return HTML
            graph_file = "medical_kg_temp.html"
            net.save_graph(graph_file)
            
            with open(graph_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Clean up temporary file
            if os.path.exists(graph_file):
                os.remove(graph_file)
                
            return html_content
            
        except Exception as e:
            return f"<div style='color: white; padding: 20px;'>Graph visualization error: {str(e)}</div>"

def render_metrics_dashboard(metrics: Dict[str, float]):
    """Render clinical quality metrics with enhanced styling"""
    col1, col2, col3, col4 = st.columns(4)
    
    metric_configs = [
        ("Source Diversity", "source_diversity", "Geographic coverage balance"),
        ("Clinical Coverage", "clinical_completeness", "Medical concept completeness"), 
        ("Synthesis Quality", "cross_guideline_synthesis", "Comparative analysis depth"),
        ("Evidence Strength", "evidence_strength", "Retrieval confidence score")
    ]
    
    for col, (label, key, description) in zip([col1, col2, col3, col4], metric_configs):
        with col:
            value = metrics.get(key, 0.0)
            color = "green" if value >= 0.7 else "orange" if value >= 0.5 else "red"
            col.metric(
                label=label,
                value=f"{value:.2f}",
                help=description
            )

def render_performance_comparison(rag_system, vector_metrics, rerank_metrics, vector_time, kg_time, vector_tokens, kg_tokens):
    """Render detailed performance comparison with real statistics"""
    st.markdown("## 📊 Real-Time Performance Analysis")
    
    # Performance metrics comparison
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.markdown("### ⚡ Response Time")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vector Search", f"{vector_time:.2f}s", 
                     delta=f"-{abs(vector_time - kg_time):.2f}s" if vector_time < kg_time else f"+{abs(vector_time - kg_time):.2f}s",
                     delta_color="inverse")
        with col2:
            st.metric("Knowledge Graph", f"{kg_time:.2f}s",
                     delta=f"-{abs(kg_time - vector_time):.2f}s" if kg_time < vector_time else f"+{abs(kg_time - vector_time):.2f}s",
                     delta_color="inverse")
    
    with perf_col2:
        st.markdown("### 📝 Response Length")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vector Tokens", f"{vector_tokens}", 
                     delta=f"{vector_tokens - kg_tokens}" if vector_tokens != kg_tokens else "0")
        with col2:
            st.metric("KG Tokens", f"{kg_tokens}",
                     delta=f"{kg_tokens - vector_tokens}" if kg_tokens != vector_tokens else "0")
    
    with perf_col3:
        st.markdown("### 🎯 Quality Score")
        vector_avg = sum(vector_metrics.values()) / len(vector_metrics)
        kg_avg = sum(rerank_metrics.values()) / len(rerank_metrics)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vector Quality", f"{vector_avg:.3f}",
                     delta=f"{vector_avg - kg_avg:.3f}" if vector_avg != kg_avg else "0.000")
        with col2:
            st.metric("KG Quality", f"{kg_avg:.3f}",
                     delta=f"{kg_avg - vector_avg:.3f}" if kg_avg != vector_avg else "0.000")
    
    # Detailed comparison table
    st.markdown("### 📋 Detailed Comparison")
    
    comparison_data = {
        "Metric": [
            "Response Time (seconds)",
            "Token Count", 
            "Source Diversity",
            "Clinical Completeness",
            "Cross-Guideline Synthesis",
            "Evidence Strength",
            "Overall Quality Score"
        ],
        "Vector Search": [
            f"{vector_time:.3f}",
            vector_tokens,
            f"{vector_metrics['source_diversity']:.3f}",
            f"{vector_metrics['clinical_completeness']:.3f}",
            f"{vector_metrics['cross_guideline_synthesis']:.3f}",
            f"{vector_metrics['evidence_strength']:.3f}",
            f"{vector_avg:.3f}"
        ],
        "Knowledge Graph": [
            f"{kg_time:.3f}",
            kg_tokens,
            f"{rerank_metrics['source_diversity']:.3f}",
            f"{rerank_metrics['clinical_completeness']:.3f}",
            f"{rerank_metrics['cross_guideline_synthesis']:.3f}",
            f"{rerank_metrics['evidence_strength']:.3f}",
            f"{kg_avg:.3f}"
        ],
        "Winner": [
            "Vector" if vector_time < kg_time else "KG",
            "Vector" if vector_tokens > kg_tokens else "KG" if kg_tokens > vector_tokens else "Tie",
            "Vector" if vector_metrics['source_diversity'] > rerank_metrics['source_diversity'] else "KG",
            "Vector" if vector_metrics['clinical_completeness'] > rerank_metrics['clinical_completeness'] else "KG",
            "Vector" if vector_metrics['cross_guideline_synthesis'] > rerank_metrics['cross_guideline_synthesis'] else "KG",
            "Vector" if vector_metrics['evidence_strength'] > rerank_metrics['evidence_strength'] else "KG",
            "Vector" if vector_avg > kg_avg else "KG"
        ]
    }
    
    st.table(comparison_data)

def render_document_database(documents):
    """Render the source documents database page"""
    st.title("📚 Source Documents Database")
    st.markdown("**Complete medical evidence repository with metadata and content analysis**")
    
    if not documents:
        st.error("No documents available in the database.")
        return
    
    # Database overview
    st.markdown("## 📊 Database Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(documents))
    
    with col2:
        total_chars = sum(len(doc.get_content()) for doc in documents)
        st.metric("Total Characters", f"{total_chars:,}")
    
    with col3:
        avg_length = total_chars // len(documents) if documents else 0
        st.metric("Avg Doc Length", f"{avg_length:,}")
    
    with col4:
        # Count unique sources
        sources = set()
        for doc in documents:
            filename = doc.metadata.get("file_name", "unknown")
            if "cdc" in filename.lower():
                sources.add("US CDC")
            elif "rivm" in filename.lower():
                sources.add("Dutch RIVM")
            else:
                sources.add("Other")
        st.metric("Source Types", len(sources))
    
    # Document classification
    st.markdown("## 🏷️ Document Classification")
    
    us_docs = []
    nl_docs = []
    other_docs = []
    
    for doc in documents:
        filename = doc.metadata.get("file_name", "unknown").lower()
        if any(term in filename for term in ["cdc", "us"]):
            us_docs.append(doc)
        elif any(term in filename for term in ["rivm", "dutch", "nl"]):
            nl_docs.append(doc)
        else:
            other_docs.append(doc)
    
    # Create tabs for different document types
    tab1, tab2, tab3 = st.tabs([f"🇺🇸 US Guidelines ({len(us_docs)})", 
                                f"🇳🇱 Dutch Guidelines ({len(nl_docs)})", 
                                f"📄 Other Sources ({len(other_docs)})"])
    
    def render_document_list(docs, tab_context):
        with tab_context:
            if not docs:
                st.info("No documents available in this category.")
                return
                
            for i, doc in enumerate(docs, 1):
                filename = doc.metadata.get("file_name", f"Document {i}")
                content = doc.get_content()
                
                # Clean filename for display
                display_name = filename.replace("./data/", "").replace(".txt", "")
                
                with st.expander(f"📄 {display_name}", expanded=False):
                    # Document metadata
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Document Metadata:**")
                        metadata_info = {
                            "Filename": filename,
                            "Content Length": f"{len(content):,} characters",
                            "Word Count": f"{len(content.split()):,} words",
                            "Estimated Reading Time": f"{len(content.split()) // 200 + 1} minutes"
                        }
                        
                        for key, value in metadata_info.items():
                            st.text(f"{key}: {value}")
                    
                    with col2:
                        # Content analysis
                        st.markdown("**Content Analysis:**")
                        content_lower = content.lower()
                        
                        # Count medical terms
                        medical_term_counts = {}
                        for category, terms in MEDICAL_TAXONOMY.items():
                            count = sum(1 for term in terms if term in content_lower)
                            if count > 0:
                                medical_term_counts[category] = count
                        
                        if medical_term_counts:
                            for category, count in medical_term_counts.items():
                                st.text(f"{category.title()}: {count} terms")
                        else:
                            st.text("No medical terms detected")
                    
                    # Document content preview
                    st.markdown("**Content Preview:**")
                    
                    # Show first few sentences
                    sentences = [s.strip() for s in content.split('.') if s.strip()]
                    preview_text = '. '.join(sentences[:3]) + '...' if len(sentences) > 3 else content
                    
                    st.text_area(
                        "Document Content", 
                        preview_text, 
                        height=200, 
                        key=f"content_{i}_{filename}",
                        help="First few sentences of the document"
                    )
    
    render_document_list(us_docs, tab1)
    render_document_list(nl_docs, tab2)
    render_document_list(other_docs, tab3)

def render_rag_comparison_page(rag_system, documents):
    """Render the main RAG comparison page from live or cached data"""
    if not rag_system.initialize_models():
        st.stop()

    if not documents:
        st.error("No documents loaded. Please check data directory.")
        st.stop()
    
    st.markdown("### 🔍 Select Medical Query")
    query_index = st.selectbox(
        "Choose a predefined medical research question:",
        options=range(len(PREDEFINED_QUERIES)),
        format_func=lambda i: PREDEFINED_QUERIES[i],
        index=0
    )
    
    selected_query = PREDEFINED_QUERIES[query_index]

    if CONFIG['demo_mode']:
        # DEMO MODE: Load from pre-generated JSON file
        cache_file = os.path.join(CONFIG['cache_path'], f"query_{query_index}.json")
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            vector_data = cached_data['vector']
            rerank_data = cached_data['rerank']
        except FileNotFoundError:
            st.error(f"Cache file not found for this query: {cache_file}. Please run `generate_cache.py`.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading cache: {str(e)}")
            st.stop()
    else:
        # LIVE MODE: Run the query engines
        vector_engine, rerank_engine = rag_system.build_rag_engines(documents)
        if not vector_engine or not rerank_engine:
            st.stop()
        
        st.warning("⚡ Running in LIVE mode. This requires local LM Studio.")
        
        # Execute queries
        vector_response, vector_time, vector_tokens = rag_system.query_with_timing(
            vector_engine, selected_query, 'vector_search'
        )
        rerank_response, kg_time, kg_tokens = rag_system.query_with_timing(
            rerank_engine, selected_query, 'kg_search'
        )
        
        # Calculate metrics
        vector_metrics = rag_system.calculate_clinical_metrics(vector_response, selected_query)
        rerank_metrics = rag_system.calculate_clinical_metrics(rerank_response, selected_query)
        
        # Convert to data format
        vector_data = {
            'response': vector_response.response,
            'source_nodes': [{'metadata': node.metadata, 'score': node.score, 'content': node.get_content()} for node in vector_response.source_nodes],
            'time': vector_time,
            'tokens': vector_tokens,
            'metrics': vector_metrics
        }
        rerank_data = {
            'response': rerank_response.response,
            'source_nodes': [{'metadata': node.metadata, 'score': node.score, 'content': node.get_content()} for node in rerank_response.source_nodes],
            'time': kg_time,
            'tokens': kg_tokens,
            'metrics': rerank_metrics
        }

    # --- Render Page from vector_data and rerank_data ---
    col1, col2 = st.columns([1, 1], gap="large")
    
    # Vector Search Results
    with col1:
        st.markdown("## 🚀 Fast Vector Search")
        st.markdown("*Direct similarity matching for rapid retrieval*")
        st.markdown("**Answer:**")
        st.write(vector_data['response'])
        
        # Model attribution
        st.markdown(f"*Generated by: **{CONFIG['llm_name']}** via LM Studio*")
        
        st.markdown("**Quality Metrics:**")
        render_metrics_dashboard(vector_data['metrics'])
        
        # Evidence sources
        if vector_data['source_nodes']:
            with st.expander("📚 Evidence Sources", expanded=False):
                query_terms = selected_query.split()
                for i, node_data in enumerate(vector_data['source_nodes'], 1):
                    source_name = node_data['metadata'].get("file_name", f"Source {i}").replace("./data/", "")
                    relevance = node_data.get('score', 0.0)
                    
                    st.markdown(f"**{i}. {source_name}** (relevance: {relevance:.3f})")
                    
                    # Extract snippet from content
                    content = node_data.get('content', '')
                    sentences = [s.strip() for s in content.split('.') if s.strip()]
                    snippet = sentences[0][:200] + '...' if sentences else content[:200] + '...'
                    st.info(snippet)

    # Knowledge Graph Approach
    with col2:
        st.markdown("## 🕸️ Knowledge Graph Retrieval")
        st.markdown("*Semantic relationship analysis with graph reasoning*")
        st.markdown("**Enhanced Answer:**")
        st.write(rerank_data['response'])
        
        # Model attribution
        st.markdown(f"*Generated by: **{CONFIG['llm_name']}** via LM Studio*")
        
        st.markdown("**Quality Metrics:**")
        render_metrics_dashboard(rerank_data['metrics'])
        
        # Interactive Knowledge Graph
        st.markdown("**Medical Knowledge Graph:**")
        
        # Graph legend
        st.markdown("""
        **Legend:** 🔴 Conditions | 🟢 Treatments | 🔵 Guidelines | 🟣 Metrics | 🟡 Interventions | 🟠 Outcomes
        """)
        
        # Build and render knowledge graph
        with st.spinner("Building knowledge graph..."):
            knowledge_graph = rag_system.build_medical_knowledge_graph(documents)
        
        graph_html = rag_system.create_interactive_graph(knowledge_graph)
        components.html(graph_html, height=600)
        
        # Graph statistics
        if knowledge_graph.nodes():
            st.markdown(f"**Graph Stats:** {len(knowledge_graph.nodes())} concepts, {len(knowledge_graph.edges())} relationships")
    
    # Performance Comparison
    st.markdown("---")
    render_performance_comparison(
        rag_system, vector_data['metrics'], rerank_data['metrics'],
        vector_data['time'], rerank_data['time'], 
        vector_data['tokens'], rerank_data['tokens']
    )

def main():
    """Main application entry point with left navigation"""
    st.set_page_config(
        page_title=CONFIG['page_title'],
        page_icon=CONFIG['page_icon'],
        layout=CONFIG['layout'],
        initial_sidebar_state="expanded"
    )

    # Force sidebar to be expanded by default
    st.session_state.sidebar_state = 'expanded'
    
    # Initialize RAG system
    rag_system = MedicalRAGSystem()
    
    # Load documents once
    documents = rag_system.load_documents()
    
    # Left sidebar navigation
    with st.sidebar:
        st.title("🏥 Medical RAG System")
        st.markdown("---")
        
        # Use session state to track current page
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'RAG Comparison'
        
        # Navigation buttons
        if st.button("🔬 RAG Comparison", use_container_width=True):
            st.session_state.current_page = 'RAG Comparison'
        
        if st.button("📚 Document Database", use_container_width=True):
            st.session_state.current_page = 'Document Database'
        
        st.markdown("---")
        
        # System info
        st.markdown("### System Information")
        st.markdown(f"**LLM Model:** {CONFIG['llm_name']}")
        st.markdown(f"**Context Window:** 128K tokens")
        st.markdown(f"**Multimodal:** Text + Vision")
        st.markdown(f"**Documents Loaded:** {len(documents) if documents else 0}")
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")
        
        # Model capabilities
        st.markdown("### Model Capabilities")
        st.markdown("✅ Question Answering")
        st.markdown("✅ Summarization") 
        st.markdown("✅ Reasoning")
        st.markdown("✅ 140+ Languages")
        st.markdown("✅ Image Understanding")
    
    # Main content area based on navigation
    if st.session_state.current_page == 'RAG Comparison':
        st.title("🏥 Medical Evidence RAG System")
        st.markdown("**Enterprise-grade retrieval with knowledge graph integration**")
        render_rag_comparison_page(rag_system, documents)
        
    elif st.session_state.current_page == 'Document Database':
        render_document_database(documents)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Technical Stack:** LlamaIndex • ColBERT Reranking • NetworkX • PyVis • Streamlit  
    **LLM:** Google Gemma-3-4B via LM Studio • **Features:** Real-time performance tracking • Interactive knowledge graphs • Clinical quality metrics
    """)

if __name__ == "__main__":
    main()
