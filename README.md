<div align="center">
  <p>
      <img width="100%" src="" alt="Nextits RAG Banner">
  </p>

English | [한국어](./docs/README_ko.md) | [简体中文](./docs/README_zh.md)

<!-- icon -->
![python](https://img.shields.io/badge/python-3.11~3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](./LICENSE)



**Nextits RAG is an advanced Retrieval-Augmented Generation system providing multimodal search, intelligent routing, and context-aware answer generation**

</div>

# Nextits RAG
[![Framework](https://img.shields.io/badge/Python-3.11+-blue)](#)
[![AI](https://img.shields.io/badge/AI-SGLang-orange)](#)
[![Features](https://img.shields.io/badge/Features-Text%20%7C%20Image%20%7C%20Multimodal-green)](#)

> [!TIP]
> Nextits RAG provides a comprehensive RAG system with multimodal capabilities, supporting text and image retrieval, intelligent query routing, and high-quality answer generation.
>
> It efficiently handles complex queries with parallel search, reranking, and context refinement.


**Nextits RAG** is a production-ready RAG (Retrieval-Augmented Generation) system that provides **multimodal search and intelligent answer generation** capabilities. It integrates text and image retrieval with advanced reranking and generation models.

### Core Features

- **Multimodal RAG Pipeline (rag_pipeline.py)**  
  Unified pipeline integrating text search, image search, query routing, generation, evaluation, and refinement for comprehensive RAG workflows.

- **Text Retrieval System (rag_text/)**  
  Vector-based text search with Weaviate integration, semantic search, and advanced reranking for high-precision text retrieval.

- **Image Retrieval System (rag_image/)**  
  Multimodal image search supporting visual similarity, caption-based search, and intelligent image reranking.

- **Intelligent Query Router (router.py)**  
  Smart query classification determining optimal search strategy (text-only, image-only, or multimodal) based on query analysis.

- **SGLang Generator (generator.py)**  
  High-performance answer generation using SGLang with context-aware prompting and efficient GPU memory management.

- **Answer Refiner (refiner.py)**  
  Post-processing module for improving answer quality, coherence, and relevance through iterative refinement.

## 📣 Recent Updates

### 2026.01: Advanced RAG System Release

- **Multimodal Pipeline**:
  - Integrated text and image retrieval
  - Intelligent query routing
  - Parallel search execution
  - Context-aware generation

- **Text Retrieval**:
  - Weaviate vector database integration
  - Semantic and keyword hybrid search
  - Advanced reranking with transformer models
  - Configurable relevance thresholds

- **Image Retrieval**:
  - Visual similarity search
  - Caption and tag-based retrieval
  - Multimodal reranking
  - Image processing and embedding

- **Generation & Refinement**:
  - SGLang-based efficient inference
  - Context-aware prompt engineering
  - Answer quality evaluation
  - Iterative refinement pipeline

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/hnextits/NextitsLM_RAG.git
cd NextitsLM_RAG/backend/notebooklm

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export WEAVIATE_HOST="your-weaviate-host"
export WEAVIATE_PORT="8080"
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_CX_ID="your-cx-id"
export HUGGINGFACE_TOKEN="your-hf-token"
```

### Basic Usage

```python
from rag_pipeline import RAGPipeline

# Initialize RAG pipeline
pipeline = RAGPipeline()

# Query with text search
query = "What is machine learning?"
result = pipeline.run(query, search_type="text")

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")

# Query with multimodal search
query = "Show me images of neural networks"
result = pipeline.run(query, search_type="multimodal")

print(f"Answer: {result['answer']}")
print(f"Text Sources: {len(result['text_results'])}")
print(f"Image Sources: {len(result['image_results'])}")
```

### Advanced Usage

```python
from config import RAGConfig
from router import RAGRouter
from generator import SGLangGenerator

# Custom configuration
config = RAGConfig()
config.TEXT_TOP_K = 
config.IMAGE_TOP_K = 
config.ENABLE_QUERY_REWRITE = True

# Initialize components
router = RAGRouter(config)
generator = SGLangGenerator(config)

# Route query
query = "Explain convolutional neural networks with examples"
search_type = router.route(query)
print(f"Recommended search type: {search_type}")

# Generate answer with custom context
context = "CNNs are neural networks specialized for image processing..."
answer = generator.generate(query, context)
print(f"Generated answer: {answer}")
```

## 📦 Module Structure

```
notebooklm/
├── rag_pipeline.py          # Main RAG pipeline orchestrator
├── config.py                # System configuration and settings
├── router.py                # Intelligent query routing
├── generator.py             # SGLang-based answer generation
├── refiner.py               # Answer refinement and post-processing
├── evaluator.py             # Answer quality evaluation
├── query_rewriter.py        # Query expansion and rewriting
├── parallel_search.py       # Parallel search execution
├── embedding_text.py        # Text embedding generation
├── embedding_image.py       # Image embedding generation
├── image_processor.py       # Image processing utilities
├── weaviate_utils.py        # Weaviate database utilities
├── shared_embedding.py      # Shared embedding model management
├── rag_text/
│   ├── text_search.py       # Text vector search
│   └── text_reranker.py     # Text reranking
└── rag_image/
    ├── image_search.py      # Image vector search
    └── image_reranker.py    # Image reranking
```

## 🔧 Configuration

### Main Configuration (config.py)

```python
class RAGConfig:
    # Weaviate Settings
    WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
    WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
    
    # Model Settings
    EMBEDDING_MODEL = "Model Name"
    RERANKER_MODEL_NAME = "Model Name"
    LLM_MODEL = "Model Name"
    
    # Search Settings
    TEXT_TOP_K = 
    TEXT_FINAL_K = 
    IMAGE_TOP_K = 
    IMAGE_FINAL_K = 
    
    # Generation Settings
    GENERATOR_MAX_TOKENS = 
    GENERATOR_TEMPERATURE = 
    GENERATOR_TOP_P = 
    
    # API Settings
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CX_ID = os.getenv("GOOGLE_CX_ID")
```

### Environment Variables

Create a `.env` file or set system environment variables:

```bash
# Weaviate Configuration
WEAVIATE_HOST=your-weaviate-host
WEAVIATE_PORT=8080
WEAVIATE_URL=http://your-weaviate-host:8080

# API Keys
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CX_ID=your-cx-id
HUGGINGFACE_TOKEN=your-hf-token

# Optional
GENERATE_ENDPOINT=your-endpoint
```

## 🎯 Key Features

### Multimodal RAG Pipeline
- **Unified Interface**: Single pipeline for text, image, and multimodal queries
- **Intelligent Routing**: Automatic query type detection and optimal search strategy selection
- **Parallel Execution**: Concurrent text and image search for faster results
- **Context Integration**: Seamless merging of multimodal search results

### Text Retrieval
- **Vector Search**: Efficient semantic search with Weaviate
- **Hybrid Search**: Combined semantic and keyword-based retrieval
- **Advanced Reranking**: Transformer-based reranking for precision
- **Relevance Filtering**: Configurable thresholds for quality control

### Image Retrieval
- **Visual Similarity**: CLIP-based image embedding and search
- **Caption Search**: Text-to-image retrieval via captions
- **Tag-based Filtering**: Metadata-enhanced search
- **Multimodal Reranking**: Cross-modal relevance scoring

### Generation & Refinement
- **SGLang Integration**: High-performance inference with efficient memory management
- **Context-aware Prompting**: Dynamic prompt construction based on retrieved context
- **Quality Evaluation**: Automatic answer quality assessment
- **Iterative Refinement**: Multi-stage answer improvement

### Query Processing
- **Query Rewriting**: Automatic query expansion and reformulation
- **Intent Detection**: Query type classification (factual, comparative, etc.)
- **Multi-turn Support**: Conversation context management
- **Error Handling**: Robust fallback mechanisms

## 📊 Performance

### Search Performance
- **Text Search**: < 200ms for top-5 retrieval
- **Image Search**: < 300ms for top-3 retrieval
- **Reranking**: < 100ms per batch
- **End-to-End**: < 2s for complete RAG pipeline

### Model Performance
- **Generation Speed**: 50-100 tokens/sec (GPU)
- **Memory Usage**: 8-12GB VRAM (7B model)
- **Batch Processing**: Up to 32 concurrent queries
- **Throughput**: 100+ queries/minute

### Accuracy Metrics
- **Retrieval Precision@5**: > 85%
- **Answer Relevance**: > 90%
- **Multimodal Accuracy**: > 80%
- **User Satisfaction**: > 4.5/5.0

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Test text retrieval
python -m rag_text.text_search --query "test query"

# Test image retrieval
python -m rag_image.image_search --query "test image query"

# Test full pipeline
python rag_pipeline.py --query "What is AI?" --search-type multimodal
```

## 💻 Development

### Requirements
- Python 3.11+
- CUDA 11.8+ (for GPU acceleration)
- Weaviate 1.24+
- 16GB+ RAM
- 12GB+ VRAM (for 7B models)

### GPU Configuration

```python
# Single GPU
config.TEXT_GENERATOR_DEVICE = "cuda:0"
config.RERANKER_DEVICE = "cuda:0"

# Multi-GPU
config.TEXT_GENERATOR_DEVICE = "cuda:0"
config.RERANKER_DEVICE = "cuda:1"
config.EMBEDDING_DEVICE = "cuda:1"
```

### Adding Custom Models

```python
# In config.py
self.EMBEDDING_MODEL = "your-embedding-model"
self.RERANKER_MODEL_NAME = "your-reranker-model"
self.LLM_MODEL = "your-generation-model"
```

## 📝 License

This project is distributed under the Apache 2.0 License. See the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgments

This project was made possible with the help of the following open-source projects:

- **[SGLang](https://github.com/sgl-project/sglang)**: High-performance LLM serving framework
- **[Weaviate](https://github.com/weaviate/weaviate)**: Vector database for knowledge management

## 🎓 Citation

If you use this project in your research, please cite the following papers:

### SGLang
```bibtex
@misc{zheng2023sglang,
  title={SGLang: Efficient Execution of Structured Language Model Programs},
  author={Lianmin Zheng and Liangsheng Yin and Zhiqiang Xie and Jeff Huang and Chuyue Sun and Cody Hao Yu and Shiyi Cao and Christos Kozyrakis and Ion Stoica and Joseph E. Gonzalez and Clark Barrett and Ying Sheng},
  year={2023},
  url={https://github.com/sgl-project/sglang}
}
```

## 🌐 Demo Site

Try out our system at: [https://quantuss.hnextits.com/](https://quantuss.hnextits.com/)

## 👥 Contributors

This project was developed by the following team members:

- **Lim** - [junseung_lim@hnextits.com](mailto:junseung_lim@hnextits.com)
- **Jeong** - [jeongnext@hnextits.com](mailto:jeongnext@hnextits.com)
- **Ryu** - [fbgjungits@hnextits.com](mailto:fbgjungits@hnextits.com)

## 📧 Contact

For questions and feedback, please contact us at the email addresses above or open an issue on GitHub.

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

---

<div align="center">
Made with 🩸💦😭 by Nextits Team
</div>
