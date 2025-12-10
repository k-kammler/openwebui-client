"""
Embeddings Example for OpenWebUI Client

Demonstrates:
- Generating text embeddings with bge-m3
- Batch embedding multiple texts
- Calculating similarity between embeddings

Uses: jgu_introduction.pdf (text extracted for embedding)
"""

import sys
from pathlib import Path

# Add parent directory to Python path so we can import openwebui_client
sys.path.insert(0, str(Path(__file__).parent.parent))

from openwebui_client import OpenWebUIClient, EMBEDDING_MODELS

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


def main():
    print("🧪 Embeddings Example")
    print("=" * 50)
    
    # Show available embedding models
    print(f"\n📋 Available embedding models: {EMBEDDING_MODELS}")
    
    # Initialize client
    print("\n🔧 Initializing OpenWebUI client...")
    client = OpenWebUIClient()
    
    # Example 1: Single text embedding
    print("\n" + "=" * 50)
    print("📊 Example 1: Single Text Embedding")
    print("-" * 50)
    
    text = "Johannes Gutenberg University Mainz is a public research university in Mainz, Germany."
    print(f"Text: {text}")
    
    result = client.create_embeddings(text)
    embedding = result['data'][0]['embedding']
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Example 2: Batch embeddings
    print("\n" + "=" * 50)
    print("📊 Example 2: Batch Embeddings")
    print("-" * 50)
    
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "The weather today is sunny and warm.",
        "Natural language processing helps computers understand text."
    ]
    
    print("Texts to embed:")
    for i, t in enumerate(texts):
        print(f"   {i+1}. {t}")
    
    result = client.create_embeddings(texts)
    print(f"\n✅ Generated {len(result['data'])} embeddings")
    print(f"Usage: {result.get('usage', {})}")
    
    # Example 3: Similarity comparison
    print("\n" + "=" * 50)
    print("📊 Example 3: Semantic Similarity")
    print("-" * 50)
    
    embeddings = [item['embedding'] for item in result['data']]
    
    print("\nSimilarity matrix:")
    print("(Higher = more similar)")
    print()
    
    # Print header
    print("     ", end="")
    for i in range(len(texts)):
        print(f"  [{i+1}]  ", end="")
    print()
    
    # Print similarity matrix
    for i, emb1 in enumerate(embeddings):
        print(f"[{i+1}]  ", end="")
        for j, emb2 in enumerate(embeddings):
            sim = cosine_similarity(emb1, emb2)
            print(f" {sim:.3f} ", end="")
        print()
    
    print("\n📝 Notice: ML/AI related texts (1,2,4) have higher")
    print("   similarity to each other than to weather text (3)")
    
    # Example 4: Document chunks (simulating PDF content)
    print("\n" + "=" * 50)
    print("📊 Example 4: Document Embedding (JGU Introduction)")
    print("-" * 50)
    
    # Simulated chunks from jgu_introduction.pdf
    doc_chunks = [
        "JGU Mainz was founded in 1477 and is one of the largest universities in Germany.",
        "The university offers a wide range of programs in natural sciences and humanities.",
        "Research at JGU spans multiple disciplines including physics, medicine, and computer science."
    ]
    
    query = "What research areas does the university focus on?"
    
    print(f"Query: {query}")
    print("\nDocument chunks:")
    for i, chunk in enumerate(doc_chunks):
        print(f"   {i+1}. {chunk}")
    
    # Embed query and chunks
    all_texts = [query] + doc_chunks
    result = client.create_embeddings(all_texts)
    all_embeddings = [item['embedding'] for item in result['data']]
    
    query_embedding = all_embeddings[0]
    chunk_embeddings = all_embeddings[1:]
    
    # Find most relevant chunk
    print("\nRelevance scores:")
    scores = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        sim = cosine_similarity(query_embedding, chunk_emb)
        scores.append((i, sim))
        print(f"   Chunk {i+1}: {sim:.4f}")
    
    best_idx, best_score = max(scores, key=lambda x: x[1])
    print(f"\n🎯 Most relevant chunk: {best_idx + 1} (score: {best_score:.4f})")
    print(f"   \"{doc_chunks[best_idx]}\"")
    
    # Example 5: Read and summarize PDF
    print("\n" + "=" * 50)
    print("📊 Example 5: PDF Reading and Summarization")
    print("-" * 50)
    
    if not PDF_AVAILABLE:
        print("⚠️  PyPDF2 not installed. Install with: pip install PyPDF2")
        print("   Skipping PDF example...")
    else:
        # Path to PDF file
        script_dir = Path(__file__).parent
        pdf_path = script_dir / "jgu_introduction.pdf"
        
        if not pdf_path.exists():
            print(f"⚠️  PDF not found: {pdf_path}")
            print("   Please ensure jgu_introduction.pdf is in the examples/ directory")
        else:
            print(f"📄 Reading PDF: {pdf_path.name}")
            
            # Extract text from PDF
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    num_pages = len(pdf_reader.pages)
                    print(f"   Pages: {num_pages}")
                    
                    # Extract all text
                    full_text = ""
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        full_text += text + "\n"
                    
                    # Clean up the text
                    full_text = full_text.strip()
                    word_count = len(full_text.split())
                    print(f"   Extracted {word_count} words")
                    
                    # Split into chunks (simple split by paragraphs)
                    paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip() and len(p.strip()) > 50]
                    print(f"   Split into {len(paragraphs)} chunks")
                    
                    # Create embeddings for all chunks
                    print("\n🔄 Creating embeddings for document chunks...")
                    embeddings_result = client.create_embeddings(paragraphs)
                    chunk_embeddings = [item['embedding'] for item in embeddings_result['data']]
                    print(f"✅ Generated {len(chunk_embeddings)} embeddings")
                    
                    # Strategy 1: Summarize each chunk, then combine (Map-Reduce)
                    # This processes the full document but requires multiple LLM calls
                    print("\n🤖 Generating summary using map-reduce strategy...")
                    print("   Strategy: Summarize each chunk, then create final summary")
                    print("   Note: This processes the FULL document via multiple LLM calls")
                    
                    # Get available models
                    models = client.get_models()
                    summarization_model = models[0]['id'] if models else "default"
                    print(f"   Using model: {summarization_model}")
                    
                    # Summarize each chunk
                    chunk_summaries = []
                    print(f"\n   Processing {len(paragraphs)} chunks...")
                    
                    for i, chunk in enumerate(paragraphs[:10]):  # Limit to first 10 chunks for demo
                        chunk_summary = client.invoke(
                            f"Summarize this text in 1-2 sentences:\n\n{chunk}",
                            model=summarization_model
                        )
                        chunk_summaries.append(chunk_summary)
                        print(f"   ✓ Chunk {i+1}/{min(len(paragraphs), 10)}")
                    
                    # Combine chunk summaries into final summary
                    combined_summaries = "\n".join(chunk_summaries)
                    
                    final_summary_prompt = f"""Based on these section summaries from a document, create a coherent overall summary in 3-4 sentences:

{combined_summaries}"""
                    
                    print("\n   Creating final combined summary...")
                    final_summary = client.invoke(final_summary_prompt, model=summarization_model)
                    
                    print("\n📝 Full Document Summary (Map-Reduce):")
                    print("-" * 50)
                    print(final_summary)
                    print("-" * 50)
                    
                    # Strategy 2: RAG - Use embeddings to answer specific questions
                    # This is where embeddings REALLY shine - fitting relevant info in context!
                    print("\n" + "=" * 50)
                    print("🔍 RAG (Retrieval-Augmented Generation) Demo")
                    print("-" * 50)
                    print("This shows how embeddings help with context limits!")
                    print("Instead of sending the WHOLE document, we:")
                    print("1. Find most relevant chunks using embeddings")
                    print("2. Send only those chunks to the LLM")
                    print("3. Answer fits in context window!")
                    
                    rag_query = "What are the main research areas at JGU?"
                    print(f"\n❓ Question: {rag_query}")
                    
                    # Embed the query
                    query_result = client.create_embeddings(rag_query)
                    query_embedding = query_result['data'][0]['embedding']
                    
                    # Find top 3 most relevant chunks using embeddings
                    print("\n   Finding most relevant chunks using embeddings...")
                    chunk_scores = []
                    for i, chunk_emb in enumerate(chunk_embeddings):
                        sim = cosine_similarity(query_embedding, chunk_emb)
                        chunk_scores.append((i, sim, paragraphs[i]))
                    
                    # Sort by similarity and take top 3
                    chunk_scores.sort(key=lambda x: x[1], reverse=True)
                    top_chunks = chunk_scores[:3]
                    
                    print(f"   ✓ Retrieved top 3 chunks (out of {len(paragraphs)} total)")
                    for rank, (idx, score, _) in enumerate(top_chunks, 1):
                        print(f"      {rank}. Chunk {idx+1} (similarity: {score:.4f})")
                    
                    # Build context from top chunks
                    context = "\n\n".join([text for _, _, text in top_chunks])
                    
                    # Now ask LLM using only the relevant chunks (RAG!)
                    rag_prompt = f"""Answer the following question based ONLY on the provided context.

Context:
{context}

Question: {rag_query}

Answer:"""
                    
                    print("\n   Generating answer using only relevant chunks...")
                    rag_answer = client.invoke(rag_prompt, model=summarization_model)
                    
                    print("\n💡 RAG Answer (using only 3 relevant chunks):")
                    print("-" * 50)
                    print(rag_answer)
                    print("-" * 50)
                    
                    print("\n✨ Key insight:")
                    print(f"   - Full document: {word_count} words ({len(paragraphs)} chunks)")
                    print(f"   - Sent to LLM: ~{len(context.split())} words (3 chunks)")
                    print(f"   - Reduction: {100 * (1 - len(context.split())/word_count):.1f}% less context needed!")
                    print("   - This is how embeddings help with long documents!")
                    
            except Exception as e:
                print(f"❌ Error reading PDF: {e}")
                print("   Make sure the PDF is not corrupted and is readable")
    
    # Cleanup
    client.close()
    print("\n✅ Embeddings example completed!")


if __name__ == "__main__":
    main()
