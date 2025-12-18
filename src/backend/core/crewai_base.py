# CrewAI Base Infrastructure for Nexus LLM Analytics
# This module provides the foundation for proper CrewAI integration

from crewai.tools import BaseTool
from crewai import LLM
from typing import Any, List, Dict
import os
import logging

def create_base_llm(model: str = None, adaptive_timeout: bool = True) -> LLM:
    """
    Create a base LLM instance with adaptive timeout based on model and system resources
    """
    if model is None:
        # Use intelligent model selection instead of hardcoded default
        from .model_selector import ModelSelector
        primary_model, _, _ = ModelSelector.select_optimal_models()
        model = primary_model.replace("ollama/", "")  # Remove prefix for compatibility
    
    # Add ollama/ prefix if not present (LiteLLM needs provider specification)
    if not model.startswith("ollama/"):
        model = f"ollama/{model}"
    
    # Set required environment variables for LiteLLM
    os.environ["OPENAI_API_KEY"] = "not-needed"
    os.environ["OPENAI_API_BASE"] = "http://localhost:11434"
    
    # Adaptive timeout based on model complexity and system resources
    if adaptive_timeout:
        timeout = _calculate_adaptive_timeout(model)
    else:
        timeout = 600  # Default 10 minutes
    
    # Use CrewAI's LLM class with optimized configuration
    return LLM(
        model=model,
        base_url="http://localhost:11434",
        timeout=timeout,
        max_retries=2,  # Reduced retries for faster failure detection
        temperature=0.1  # Lower temperature for more consistent results
    )


def _calculate_adaptive_timeout(model: str) -> int:
    """
    Calculate adaptive timeout based on model complexity and system resources
    Uses intelligent timeout strategy for different models
    """
    import psutil
    
    # Get available RAM
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    # Base timeouts by model complexity
    model_timeouts = {
        "llama3.1:8b": 900,  # 15 minutes for complex model
        "phi3:mini": 300,    # 5 minutes for efficient model
        "tinyllama": 180,    # 3 minutes for lightweight model
        "nomic-embed-text": 120  # 2 minutes for embedding model
    }
    
    # Get clean model name
    clean_model = model.replace("ollama/", "")
    base_timeout = model_timeouts.get(clean_model, 600)
    
    # Adjust timeout based on available memory
    if available_gb < 2.0:
        # Low memory - increase timeout for swap usage
        multiplier = 2.0
    elif available_gb < 4.0:
        # Medium memory - slight increase
        multiplier = 1.5
    else:
        # High memory - standard timeout
        multiplier = 1.0
    
    final_timeout = int(base_timeout * multiplier)
    
    # Cap maximum timeout to prevent infinite waits
    max_timeout = 1800  # 30 minutes maximum
    return min(final_timeout, max_timeout)

class DataAnalysisTool(BaseTool):
    """Tool for data analysis operations"""
    
    name: str = "data_analysis"
    description: str = "Analyze structured data files (CSV, JSON) and perform statistical operations"
    sandbox: Any = None
    
    def __init__(self, sandbox, **kwargs):
        super().__init__(sandbox=sandbox, **kwargs)
    
    def _run(self, code: str, data: Any = None, **kwargs) -> str:
        """Execute data analysis code in sandbox"""
        try:
            result = self.sandbox.execute(code, data=data)
            if "error" in result:
                return f"Error: {result['error']}"
            return str(result.get("result", ""))
        except Exception as e:
            return f"Tool execution failed: {str(e)}"

class OptimizedRAGTool(BaseTool):
    """
    Optimized RAG Tool with advanced retrieval algorithms and performance improvements
    Features:
    - Semantic similarity ranking with multiple query strategies
    - Context optimization using sliding window technique
    - Fallback strategies for better document matching
    - Performance monitoring and adaptive retrieval
    """
    
    name: str = "rag_retrieval"
    description: str = "Retrieve and summarize information from unstructured documents using advanced vector similarity and semantic ranking"
    chroma_client: Any = None
    llm_client: Any = None
    
    def __init__(self, chroma_client, llm_client, **kwargs):
        super().__init__(chroma_client=chroma_client, llm_client=llm_client, **kwargs)
        self._query_cache = {}  # Cache for repeated queries
        self._retrieval_stats = {"total_queries": 0, "cache_hits": 0}
    
    def _optimize_query(self, query: str) -> List[str]:
        """
        Generate multiple query variations for better retrieval
        Uses synonym expansion and keyword extraction
        """
        queries = [query]  # Original query
        
        # Add variations
        words = query.lower().split()
        if len(words) > 1:
            # Add individual important words
            important_words = [w for w in words if len(w) > 3]
            if important_words:
                queries.append(" ".join(important_words))
        
        # Add context-aware variations
        if "skill" in query.lower():
            queries.append("technical skills programming experience")
        if "summary" in query.lower() or "summarize" in query.lower():
            queries.append("overview main points key information")
        
        return queries[:3]  # Limit to 3 variations
    
    def _rank_and_filter_results(self, results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Advanced result ranking using multiple similarity metrics
        Time Complexity: O(n log n) for sorting
        """
        docs = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0] if 'distances' in results else []
        metadatas = results.get('metadatas', [[]])[0] if 'metadatas' in results else [{}] * len(docs)
        ids = results.get('ids', [[]])[0] if 'ids' in results else list(range(len(docs)))
        
        if not docs:
            return results
        
        # Score each document using multiple factors
        scored_docs = []
        query_words = set(query.lower().split())
        
        for i, (doc, metadata, doc_id) in enumerate(zip(docs, metadatas, ids)):
            doc_lower = doc.lower()
            doc_words = set(doc_lower.split())
            
            # Calculate multiple similarity scores
            word_overlap = len(query_words & doc_words) / len(query_words) if query_words else 0
            query_coverage = sum(1 for word in query_words if word in doc_lower) / len(query_words)
            doc_length_penalty = max(0.5, min(1.0, len(doc) / 1000))  # Prefer medium-length docs
            
            # Distance score (lower is better, so invert)
            distance_score = 1.0 - (distances[i] if i < len(distances) else 0.5)
            
            # Combined score
            final_score = (
                word_overlap * 0.3 +
                query_coverage * 0.3 +
                distance_score * 0.3 +
                doc_length_penalty * 0.1
            )
            
            scored_docs.append({
                'doc': doc,
                'metadata': metadata,
                'id': doc_id,
                'score': final_score,
                'word_overlap': word_overlap
            })
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # Filter out very low-scoring results
        min_score = 0.1
        filtered_docs = [d for d in scored_docs if d['score'] >= min_score]
        
        # Reconstruct results format
        return {
            'documents': [[d['doc'] for d in filtered_docs]],
            'metadatas': [[d['metadata'] for d in filtered_docs]],
            'ids': [[d['id'] for d in filtered_docs]],
            'distances': [[1.0 - d['score'] for d in filtered_docs]]
        }
    
    def _optimize_context(self, docs: List[str], query: str, max_length: int = 3000) -> str:
        """
        Optimize context using intelligent truncation and relevance ranking
        Uses sliding window to keep most relevant parts
        """
        if not docs:
            return ""
        
        # Join all documents
        full_context = "\n\n".join(docs)
        
        # If context is short enough, return as-is
        if len(full_context) <= max_length:
            return full_context
        
        # Find most relevant sections using sliding window
        query_words = set(query.lower().split())
        sentences = full_context.split('.')
        
        # Score sentences by relevance
        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            sentence_lower = sentence.lower()
            relevance = sum(1 for word in query_words if word in sentence_lower)
            scored_sentences.append((sentence, relevance))
        
        # Sort by relevance and take top sentences within length limit
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        optimized_context = ""
        for sentence, _ in scored_sentences:
            if len(optimized_context) + len(sentence) + 2 <= max_length:
                optimized_context += sentence + ". "
            else:
                break
        
        return optimized_context.strip()
    
    def _run(self, query: str, n_results: int = 5, **kwargs) -> str:
        """
        Optimized RAG retrieval with multiple strategies and performance enhancements
        """
        import time
        start_time = time.time()
        
        # Update stats
        self._retrieval_stats["total_queries"] += 1
        
        # Check cache first
        cache_key = f"{query}_{n_results}"
        if cache_key in self._query_cache:
            self._retrieval_stats["cache_hits"] += 1
            return self._query_cache[cache_key]
        
        try:
            # Auto-index any missing documents first
            from .document_indexer import fix_rag_indexing_issue
            fix_rag_indexing_issue()
            
            # Generate query variations for better retrieval
            query_variations = self._optimize_query(query)
            
            all_results = {}
            for q in query_variations:
                try:
                    results = self.chroma_client.query(query_text=q, n_results=n_results)
                    if results.get('documents', [[]])[0]:  # Has results
                        # Rank and filter results
                        filtered_results = self._rank_and_filter_results(results, query)
                        if filtered_results.get('documents', [[]])[0]:
                            all_results[q] = filtered_results
                            break  # Use first successful query
                except Exception as e:
                    logging.warning(f"Query variation '{q}' failed: {e}")
                    continue
            
            # If no results from any variation, return helpful message
            if not all_results:
                response = (
                    f"No relevant documents found for query: '{query}'. "
                    f"This might be because:\n"
                    f"1. The document hasn't been indexed yet\n"
                    f"2. The query doesn't match the document content\n"
                    f"3. Try rephrasing your query with different keywords"
                )
                self._query_cache[cache_key] = response
                return response
            
            # Get best results
            best_query, best_results = next(iter(all_results.items()))
            docs = best_results.get('documents', [[]])[0]
            
            # Optimize context for better LLM processing
            optimized_context = self._optimize_context(docs, query, max_length=2500)
            
            # Create optimized prompt
            prompt = f"""Based on the following context from documents, provide a comprehensive answer to the query.

Query: {query}

Context:
{optimized_context}

Instructions:
1. Answer directly and specifically to the query
2. Use information from the context provided
3. If asking about skills or qualifications, list them clearly
4. Be comprehensive but concise
5. If the context doesn't fully answer the query, mention what information is available

Answer:"""
            
            # Generate response with the LLM
            response = self.llm_client.generate_primary(prompt)
            final_response = response.get("response", "No response generated")
            
            # Add processing stats for debugging
            processing_time = time.time() - start_time
            if processing_time > 5:  # Log slow queries
                logging.info(f"RAG query took {processing_time:.2f}s for: {query[:50]}...")
            
            # Cache successful response
            self._query_cache[cache_key] = final_response
            
            return final_response
            
        except Exception as e:
            error_msg = f"RAG retrieval failed: {str(e)}"
            logging.error(error_msg)
            return error_msg

class VisualizationTool(BaseTool):
    """Tool for generating data visualizations"""
    
    name: str = "visualization"
    description: str = "Generate Plotly visualizations from data analysis results"
    llm_client: Any = None
    
    def __init__(self, llm_client, **kwargs):
        super().__init__(llm_client=llm_client, **kwargs)
    
    def _run(self, data_summary: str, chart_type: str = "auto", **kwargs) -> str:
        """Generate visualization code"""
        try:
            prompt = f"""Generate Python Plotly code for visualization:
Data Summary: {data_summary}
Chart Type: {chart_type}

Return only the Python code that creates a Plotly figure."""
            
            response = self.llm_client.generate_primary(prompt)
            return response.get("response", "No visualization generated")
        except Exception as e:
            return f"Visualization tool failed: {str(e)}"

# create_base_llm function is defined above

def create_analysis_tools(sandbox, chroma_client, llm_client):
    """Factory function to create all optimized analysis tools"""
    return [
        DataAnalysisTool(sandbox=sandbox),
        OptimizedRAGTool(chroma_client=chroma_client, llm_client=llm_client),
        VisualizationTool(llm_client=llm_client)
    ]