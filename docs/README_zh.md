<div align="center">
  <p>
      <img width="100%" src="" alt="Nextits RAG Banner">
  </p>

[English](../README.md) | [한국어](./README_ko.md) | 简体中文

<!-- icon -->
![python](https://img.shields.io/badge/python-3.11~3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](../LICENSE)



**Nextits RAG 是一个先进的检索增强生成系统，提供多模态搜索、智能路由和上下文感知答案生成**

</div>

# Nextits RAG
[![Framework](https://img.shields.io/badge/Python-3.11+-blue)](#)
[![AI](https://img.shields.io/badge/AI-SGLang-orange)](#)
[![Features](https://img.shields.io/badge/Features-Text%20%7C%20Image%20%7C%20Multimodal-green)](#)

> [!TIP]
> Nextits RAG 提供具有多模态功能的综合RAG系统，支持文本和图像检索、智能查询路由和高质量答案生成。
>
> 它通过并行搜索、重排序和上下文精炼高效处理复杂查询。


**Nextits RAG** 是一个生产就绪的RAG（检索增强生成）系统，提供**多模态搜索和智能答案生成**功能。它将文本和图像检索与先进的重排序和生成模型集成。

### 核心功能

- **多模态RAG管道 (rag_pipeline.py)**  
  统一管道，集成文本搜索、图像搜索、查询路由、生成、评估和精炼，实现全面的RAG工作流。

- **文本检索系统 (rag_text/)**  
  基于向量的文本搜索，集成Weaviate、语义搜索和高精度文本检索的高级重排序。

- **图像检索系统 (rag_image/)**  
  支持视觉相似性、基于标题的搜索和智能图像重排序的多模态图像搜索。

- **智能查询路由器 (router.py)**  
  基于查询分析确定最佳搜索策略（纯文本、纯图像或多模态）的智能查询分类。

- **SGLang生成器 (generator.py)**  
  使用SGLang进行高性能答案生成，具有上下文感知提示和高效GPU内存管理。

- **答案精炼器 (refiner.py)**  
  通过迭代精炼提高答案质量、连贯性和相关性的后处理模块。

## 📣 最近更新

### 2026.01: 高级RAG系统发布

- **多模态管道**:
  - 集成文本和图像检索
  - 智能查询路由
  - 并行搜索执行
  - 上下文感知生成

- **文本检索**:
  - Weaviate向量数据库集成
  - 语义和关键词混合搜索
  - 使用Transformer模型的高级重排序
  - 可配置的相关性阈值

- **图像检索**:
  - 视觉相似性搜索
  - 基于标题和标签的检索
  - 多模态重排序
  - 图像处理和嵌入

- **生成和精炼**:
  - 基于SGLang的高效推理
  - 上下文感知提示工程
  - 答案质量评估
  - 迭代精炼管道

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/hnextits/NextitsLM_RAG.git
cd NextitsLM_RAG/backend/notebooklm

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export WEAVIATE_HOST="your-weaviate-host"
export WEAVIATE_PORT="8080"
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_CX_ID="your-cx-id"
export HUGGINGFACE_TOKEN="your-hf-token"
```

### 基本用法

```python
from rag_pipeline import RAGPipeline

# 初始化RAG管道
pipeline = RAGPipeline()

# 使用文本搜索查询
query = "什么是机器学习？"
result = pipeline.run(query, search_type="text")

print(f"答案: {result['answer']}")
print(f"来源: {result['sources']}")

# 使用多模态搜索查询
query = "显示神经网络的图像"
result = pipeline.run(query, search_type="multimodal")

print(f"答案: {result['answer']}")
print(f"文本来源: {len(result['text_results'])}")
print(f"图像来源: {len(result['image_results'])}")
```

### 高级用法

```python
from config import RAGConfig
from router import RAGRouter
from generator import SGLangGenerator

# 自定义配置
config = RAGConfig()
config.TEXT_TOP_K = 
config.IMAGE_TOP_K = 
config.ENABLE_QUERY_REWRITE = True

# 初始化组件
router = RAGRouter(config)
generator = SGLangGenerator(config)

# 路由查询
query = "用示例解释卷积神经网络"
search_type = router.route(query)
print(f"推荐的搜索类型: {search_type}")

# 使用自定义上下文生成答案
context = "CNN是专门用于图像处理的神经网络..."
answer = generator.generate(query, context)
print(f"生成的答案: {answer}")
```

## 📦 模块结构

```
notebooklm/
├── rag_pipeline.py          # 主RAG管道编排器
├── config.py                # 系统配置和设置
├── router.py                # 智能查询路由
├── generator.py             # 基于SGLang的答案生成
├── refiner.py               # 答案精炼和后处理
├── evaluator.py             # 答案质量评估
├── query_rewriter.py        # 查询扩展和重写
├── parallel_search.py       # 并行搜索执行
├── embedding_text.py        # 文本嵌入生成
├── embedding_image.py       # 图像嵌入生成
├── image_processor.py       # 图像处理工具
├── weaviate_utils.py        # Weaviate数据库工具
├── shared_embedding.py      # 共享嵌入模型管理
├── rag_text/
│   ├── text_search.py       # 文本向量搜索
│   └── text_reranker.py     # 文本重排序
└── rag_image/
    ├── image_search.py      # 图像向量搜索
    └── image_reranker.py    # 图像重排序
```

## 🔧 配置

### 主配置 (config.py)

```python
class RAGConfig:
    # Weaviate设置
    WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
    WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
    
    # 模型设置
    EMBEDDING_MODEL = "Model Name"
    RERANKER_MODEL_NAME = "Model Name"
    LLM_MODEL = "Model Name"
    
    # 搜索设置
    TEXT_TOP_K = 
    TEXT_FINAL_K = 
    IMAGE_TOP_K = 
    IMAGE_FINAL_K = 
    
    # 生成设置
    GENERATOR_MAX_TOKENS = 
    GENERATOR_TEMPERATURE = 
    GENERATOR_TOP_P = 
    
    # API设置
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CX_ID = os.getenv("GOOGLE_CX_ID")
```

### 环境变量

创建`.env`文件或设置系统环境变量：

```bash
# Weaviate配置
WEAVIATE_HOST=your-weaviate-host
WEAVIATE_PORT=8080
WEAVIATE_URL=http://your-weaviate-host:8080

# API密钥
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CX_ID=your-cx-id
HUGGINGFACE_TOKEN=your-hf-token

# 可选
GENERATE_ENDPOINT=your-endpoint
```

## 🎯 主要功能

### 多模态RAG管道
- **统一接口**: 用于文本、图像和多模态查询的单一管道
- **智能路由**: 自动查询类型检测和最佳搜索策略选择
- **并行执行**: 并发文本和图像搜索以获得更快的结果
- **上下文集成**: 多模态搜索结果的无缝合并

### 文本检索
- **向量搜索**: 使用Weaviate进行高效语义搜索
- **混合搜索**: 结合语义和基于关键词的检索
- **高级重排序**: 基于Transformer的精确重排序
- **相关性过滤**: 用于质量控制的可配置阈值

### 图像检索
- **视觉相似性**: 基于CLIP的图像嵌入和搜索
- **标题搜索**: 通过标题进行文本到图像检索
- **基于标签的过滤**: 元数据增强搜索
- **多模态重排序**: 跨模态相关性评分

### 生成和精炼
- **SGLang集成**: 具有高效内存管理的高性能推理
- **上下文感知提示**: 基于检索上下文的动态提示构建
- **质量评估**: 自动答案质量评估
- **迭代精炼**: 多阶段答案改进

### 查询处理
- **查询重写**: 自动查询扩展和重构
- **意图检测**: 查询类型分类（事实性、比较性等）
- **多轮支持**: 对话上下文管理
- **错误处理**: 强大的回退机制

## 📊 性能

### 搜索性能
- **文本搜索**: top-5检索 < 200ms
- **图像搜索**: top-3检索 < 300ms
- **重排序**: 每批 < 100ms
- **端到端**: 完整RAG管道 < 2s

### 模型性能
- **生成速度**: 50-100 tokens/秒 (GPU)
- **内存使用**: 8-12GB VRAM (7B模型)
- **批处理**: 最多32个并发查询
- **吞吐量**: 100+ 查询/分钟

### 准确度指标
- **检索Precision@5**: > 85%
- **答案相关性**: > 90%
- **多模态准确度**: > 80%
- **用户满意度**: > 4.5/5.0

## 🧪 测试

```bash
# 运行单元测试
pytest tests/

# 测试文本检索
python -m rag_text.text_search --query "测试查询"

# 测试图像检索
python -m rag_image.image_search --query "测试图像查询"

# 测试完整管道
python rag_pipeline.py --query "什么是AI？" --search-type multimodal
```

## 💻 开发

### 要求
- Python 3.11+
- CUDA 11.8+ (用于GPU加速)
- Weaviate 1.24+
- 16GB+ RAM
- 12GB+ VRAM (用于7B模型)

### GPU配置

```python
# 单GPU
config.TEXT_GENERATOR_DEVICE = "cuda:0"
config.RERANKER_DEVICE = "cuda:0"

# 多GPU
config.TEXT_GENERATOR_DEVICE = "cuda:0"
config.RERANKER_DEVICE = "cuda:1"
config.EMBEDDING_DEVICE = "cuda:1"
```

### 添加自定义模型

```python
# 在config.py中
self.EMBEDDING_MODEL = "your-embedding-model"
self.RERANKER_MODEL_NAME = "your-reranker-model"
self.LLM_MODEL = "your-generation-model"
```

## 📝 许可证

本项目根据 Apache 2.0 许可证分发。详情请参阅 [LICENSE](../LICENSE) 文件。

## 🙏 致谢

本项目得益于以下开源项目的帮助：

- **[SGLang](https://github.com/sgl-project/sglang)**: 高性能LLM服务框架
- **[Weaviate](https://github.com/weaviate/weaviate)**: 用于知识管理的向量数据库

## 🎓 引用

如果您在研究中使用本项目，请引用以下论文：

### SGLang
```bibtex
@misc{zheng2023sglang,
  title={SGLang: Efficient Execution of Structured Language Model Programs},
  author={Lianmin Zheng and Liangsheng Yin and Zhiqiang Xie and Jeff Huang and Chuyue Sun and Cody Hao Yu and Shiyi Cao and Christos Kozyrakis and Ion Stoica and Joseph E. Gonzalez and Clark Barrett and Ying Sheng},
  year={2023},
  url={https://github.com/sgl-project/sglang}
}
```

## 🌐 演示网站

在线试用我们的系统：[https://quantuss.hnextits.com/](https://quantuss.hnextits.com/)

## 👥 开发者

本项目由以下团队成员开发：

- **Lim** - [junseung_lim@hnextits.com](mailto:junseung_lim@hnextits.com)
- **Jeong** - [jeongnext@hnextits.com](mailto:jeongnext@hnextits.com)
- **Ryu** - [fbgjungits@hnextits.com](mailto:fbgjungits@hnextits.com)

## 📧 联系方式

如有问题和反馈，请通过上述电子邮件地址联系我们或在GitHub上提出问题。

## 🤝 贡献

我们欢迎贡献！请随时提交Pull Request。

---

<div align="center">
Made with 🩸💦😭 by Nextits Team
</div>
