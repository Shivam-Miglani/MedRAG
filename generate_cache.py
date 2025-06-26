import os
import json
from app import MedicalRAGSystem, PREDEFINED_QUERIES, CONFIG

def generate_and_cache_results():
    """Generate cached responses for all predefined queries"""
    print("🚀 Initializing RAG system in LIVE mode...")
    
    # Temporarily disable demo mode
    original_demo_mode = CONFIG['demo_mode']
    CONFIG['demo_mode'] = False
    
    rag_system = MedicalRAGSystem()
    
    # Initialize models
    if not rag_system.initialize_models():
        print("❌ Could not connect to LM Studio. Aborting.")
        print("📝 Make sure LM Studio is running with Gemma-3-4B model loaded.")
        return

    print("📚 Loading documents and building RAG engines...")
    documents = rag_system.load_documents()
    
    if not documents:
        print("❌ No documents found. Check your data directory.")
        return
    
    vector_engine, rerank_engine = rag_system.build_rag_engines(documents)

    if not vector_engine or not rerank_engine:
        print("❌ Failed to build RAG engines.")
        return

    print(f"🔄 Generating responses for {len(PREDEFINED_QUERIES)} predefined queries...")
    
    for i, query in enumerate(PREDEFINED_QUERIES):
        print(f"  📝 Processing query {i+1}/{len(PREDEFINED_QUERIES)}: '{query[:50]}...'")
        
        try:
            # --- Vector Search ---
            print(f"    🚀 Running vector search...")
            vector_response, vector_time, vector_tokens = rag_system.query_with_timing(
                vector_engine, query, 'vector_search'
            )
            vector_metrics = rag_system.calculate_clinical_metrics(vector_response, query)
            
            # --- Rerank/KG Search ---
            print(f"    🕸️ Running knowledge graph search...")
            rerank_response, kg_time, kg_tokens = rag_system.query_with_timing(
                rerank_engine, query, 'kg_search'
            )
            rerank_metrics = rag_system.calculate_clinical_metrics(rerank_response, query)
            
            # --- Save results to JSON ---
            result_data = {
                'vector': {
                    'response': vector_response.response,
                    'source_nodes': [
                        {
                            'metadata': node.metadata,
                            'score': node.score,
                            'content': node.get_content()
                        } for node in vector_response.source_nodes
                    ],
                    'time': vector_time,
                    'tokens': vector_tokens,
                    'metrics': vector_metrics
                },
                'rerank': {
                    'response': rerank_response.response,
                    'source_nodes': [
                        {
                            'metadata': node.metadata,
                            'score': node.score,
                            'content': node.get_content()
                        } for node in rerank_response.source_nodes
                    ],
                    'time': kg_time,
                    'tokens': kg_tokens,
                    'metrics': rerank_metrics
                }
            }
            
            filename = os.path.join(CONFIG['cache_path'], f"query_{i}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
                
            print(f"    ✅ Cached results to {filename}")
            
        except Exception as e:
            print(f"    ❌ Error processing query {i}: {str(e)}")
            continue
    
    # Restore original demo mode setting
    CONFIG['demo_mode'] = original_demo_mode
    
    print("\n🎉 All query responses have been cached successfully!")
    print(f"📁 Cache files saved to: {CONFIG['cache_path']}")
    print("🚀 You can now deploy your app in DEMO mode.")

if __name__ == "__main__":
    print("🏥 Medical RAG Cache Generator")
    print("=" * 50)
    generate_and_cache_results()
