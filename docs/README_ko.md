<div align="center">
  <p>
      <img width="100%" src="" alt="Nextits RAG Banner">
  </p>

[English](../README.md) | 한국어 | [简体中文](./README_zh.md)

<!-- icon -->
![python](https://img.shields.io/badge/python-3.11~3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](../LICENSE)



**Nextits RAG는 멀티모달 검색, 지능형 라우팅, 컨텍스트 인식 답변 생성을 제공하는 고급 검색 증강 생성 시스템입니다**

</div>

# Nextits RAG
[![Framework](https://img.shields.io/badge/Python-3.11+-blue)](#)
[![AI](https://img.shields.io/badge/AI-SGLang-orange)](#)
[![Features](https://img.shields.io/badge/Features-Text%20%7C%20Image%20%7C%20Multimodal-green)](#)

> [!TIP]
> Nextits RAG는 텍스트 및 이미지 검색, 지능형 쿼리 라우팅, 고품질 답변 생성을 지원하는 멀티모달 기능을 갖춘 종합 RAG 시스템을 제공합니다.
>
> 병렬 검색, 리랭킹, 컨텍스트 정제를 통해 복잡한 쿼리를 효율적으로 처리합니다.


**Nextits RAG**는 **멀티모달 검색 및 지능형 답변 생성** 기능을 제공하는 프로덕션 준비 완료 RAG(Retrieval-Augmented Generation) 시스템입니다. 텍스트 및 이미지 검색을 고급 리랭킹 및 생성 모델과 통합합니다.

### 핵심 기능

- **멀티모달 RAG 파이프라인 (rag_pipeline.py)**  
  텍스트 검색, 이미지 검색, 쿼리 라우팅, 생성, 평가, 정제를 통합한 종합 RAG 워크플로우를 위한 통합 파이프라인.

- **텍스트 검색 시스템 (rag_text/)**  
  Weaviate 통합, 시맨틱 검색, 고정밀 텍스트 검색을 위한 고급 리랭킹을 갖춘 벡터 기반 텍스트 검색.

- **이미지 검색 시스템 (rag_image/)**  
  시각적 유사성, 캡션 기반 검색, 지능형 이미지 리랭킹을 지원하는 멀티모달 이미지 검색.

- **지능형 쿼리 라우터 (router.py)**  
  쿼리 분석을 기반으로 최적의 검색 전략(텍스트 전용, 이미지 전용 또는 멀티모달)을 결정하는 스마트 쿼리 분류.

- **SGLang 생성기 (generator.py)**  
  컨텍스트 인식 프롬프팅 및 효율적인 GPU 메모리 관리를 통한 SGLang 기반 고성능 답변 생성.

- **답변 정제기 (refiner.py)**  
  반복적인 정제를 통해 답변 품질, 일관성, 관련성을 개선하는 후처리 모듈.

## 📣 최근 업데이트

### 2026.01: 고급 RAG 시스템 공개

- **멀티모달 파이프라인**:
  - 텍스트 및 이미지 검색 통합
  - 지능형 쿼리 라우팅
  - 병렬 검색 실행
  - 컨텍스트 인식 생성

- **텍스트 검색**:
  - Weaviate 벡터 데이터베이스 통합
  - 시맨틱 및 키워드 하이브리드 검색
  - 트랜스포머 모델을 활용한 고급 리랭킹
  - 구성 가능한 관련성 임계값

- **이미지 검색**:
  - 시각적 유사성 검색
  - 캡션 및 태그 기반 검색
  - 멀티모달 리랭킹
  - 이미지 처리 및 임베딩

- **생성 및 정제**:
  - SGLang 기반 효율적인 추론
  - 컨텍스트 인식 프롬프트 엔지니어링
  - 답변 품질 평가
  - 반복적 정제 파이프라인

## 🚀 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/hnextits/NextitsLM_RAG.git
cd NextitsLM_RAG/backend/notebooklm

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
export WEAVIATE_HOST="your-weaviate-host"
export WEAVIATE_PORT="8080"
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_CX_ID="your-cx-id"
export HUGGINGFACE_TOKEN="your-hf-token"
```

### 기본 사용법

```python
from rag_pipeline import RAGPipeline

# RAG 파이프라인 초기화
pipeline = RAGPipeline()

# 텍스트 검색으로 쿼리
query = "머신러닝이란 무엇인가요?"
result = pipeline.run(query, search_type="text")

print(f"답변: {result['answer']}")
print(f"출처: {result['sources']}")

# 멀티모달 검색으로 쿼리
query = "신경망 이미지를 보여주세요"
result = pipeline.run(query, search_type="multimodal")

print(f"답변: {result['answer']}")
print(f"텍스트 출처: {len(result['text_results'])}")
print(f"이미지 출처: {len(result['image_results'])}")
```

### 고급 사용법

```python
from config import RAGConfig
from router import RAGRouter
from generator import SGLangGenerator

# 사용자 정의 설정
config = RAGConfig()
config.TEXT_TOP_K = 
config.IMAGE_TOP_K = 
config.ENABLE_QUERY_REWRITE = True

# 컴포넌트 초기화
router = RAGRouter(config)
generator = SGLangGenerator(config)

# 쿼리 라우팅
query = "합성곱 신경망을 예시와 함께 설명해주세요"
search_type = router.route(query)
print(f"권장 검색 유형: {search_type}")

# 사용자 정의 컨텍스트로 답변 생성
context = "CNN은 이미지 처리에 특화된 신경망입니다..."
answer = generator.generate(query, context)
print(f"생성된 답변: {answer}")
```

## 📦 모듈 구조

```
notebooklm/
├── rag_pipeline.py          # 메인 RAG 파이프라인 오케스트레이터
├── config.py                # 시스템 설정 및 구성
├── router.py                # 지능형 쿼리 라우팅
├── generator.py             # SGLang 기반 답변 생성
├── refiner.py               # 답변 정제 및 후처리
├── evaluator.py             # 답변 품질 평가
├── query_rewriter.py        # 쿼리 확장 및 재작성
├── parallel_search.py       # 병렬 검색 실행
├── embedding_text.py        # 텍스트 임베딩 생성
├── embedding_image.py       # 이미지 임베딩 생성
├── image_processor.py       # 이미지 처리 유틸리티
├── weaviate_utils.py        # Weaviate 데이터베이스 유틸리티
├── shared_embedding.py      # 공유 임베딩 모델 관리
├── rag_text/
│   ├── text_search.py       # 텍스트 벡터 검색
│   └── text_reranker.py     # 텍스트 리랭킹
└── rag_image/
    ├── image_search.py      # 이미지 벡터 검색
    └── image_reranker.py    # 이미지 리랭킹
```

## 🔧 설정

### 메인 설정 (config.py)

```python
class RAGConfig:
    # Weaviate 설정
    WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
    WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
    
    # 모델 설정
    EMBEDDING_MODEL = "Model Name"
    RERANKER_MODEL_NAME = "Model Name"
    LLM_MODEL = "Model Name"
    
    # 검색 설정
    TEXT_TOP_K = 
    TEXT_FINAL_K = 
    IMAGE_TOP_K = 
    IMAGE_FINAL_K = 
    
    # 생성 설정
    GENERATOR_MAX_TOKENS = 
    GENERATOR_TEMPERATURE = 
    GENERATOR_TOP_P = 
    
    # API 설정
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CX_ID = os.getenv("GOOGLE_CX_ID")
```

### 환경 변수

`.env` 파일을 생성하거나 시스템 환경 변수를 설정하세요:

```bash
# Weaviate 설정
WEAVIATE_HOST=your-weaviate-host
WEAVIATE_PORT=8080
WEAVIATE_URL=http://your-weaviate-host:8080

# API 키
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CX_ID=your-cx-id
HUGGINGFACE_TOKEN=your-hf-token

# 선택 사항
GENERATE_ENDPOINT=your-endpoint
```

## 🎯 주요 기능

### 멀티모달 RAG 파이프라인
- **통합 인터페이스**: 텍스트, 이미지, 멀티모달 쿼리를 위한 단일 파이프라인
- **지능형 라우팅**: 자동 쿼리 유형 감지 및 최적 검색 전략 선택
- **병렬 실행**: 더 빠른 결과를 위한 동시 텍스트 및 이미지 검색
- **컨텍스트 통합**: 멀티모달 검색 결과의 원활한 병합

### 텍스트 검색
- **벡터 검색**: Weaviate를 활용한 효율적인 시맨틱 검색
- **하이브리드 검색**: 시맨틱 및 키워드 기반 검색 결합
- **고급 리랭킹**: 정밀도를 위한 트랜스포머 기반 리랭킹
- **관련성 필터링**: 품질 관리를 위한 구성 가능한 임계값

### 이미지 검색
- **시각적 유사성**: CLIP 기반 이미지 임베딩 및 검색
- **캡션 검색**: 캡션을 통한 텍스트-이미지 검색
- **태그 기반 필터링**: 메타데이터 강화 검색
- **멀티모달 리랭킹**: 교차 모달 관련성 점수 매기기

### 생성 및 정제
- **SGLang 통합**: 효율적인 메모리 관리를 통한 고성능 추론
- **컨텍스트 인식 프롬프팅**: 검색된 컨텍스트 기반 동적 프롬프트 구성
- **품질 평가**: 자동 답변 품질 평가
- **반복적 정제**: 다단계 답변 개선

### 쿼리 처리
- **쿼리 재작성**: 자동 쿼리 확장 및 재구성
- **의도 감지**: 쿼리 유형 분류(사실적, 비교적 등)
- **다중 턴 지원**: 대화 컨텍스트 관리
- **오류 처리**: 강력한 폴백 메커니즘

## 📊 성능

### 검색 성능
- **텍스트 검색**: top-5 검색에 < 200ms
- **이미지 검색**: top-3 검색에 < 300ms
- **리랭킹**: 배치당 < 100ms
- **엔드투엔드**: 전체 RAG 파이프라인에 < 2s

### 모델 성능
- **생성 속도**: 50-100 토큰/초 (GPU)
- **메모리 사용량**: 40GB+ VRAM (전체)
- **배치 처리**: 최대 32개 동시 쿼리
- **처리량**: 100+ 쿼리/분

### 정확도 지표
- **검색 Precision@5**: > 85%
- **답변 관련성**: > 90%
- **멀티모달 정확도**: > 80%
- **사용자 만족도**: > 4.5/5.0

## 🧪 테스트

```bash
# 단위 테스트 실행
pytest tests/

# 텍스트 검색 테스트
python -m rag_text.text_search --query "테스트 쿼리"

# 이미지 검색 테스트
python -m rag_image.image_search --query "테스트 이미지 쿼리"

# 전체 파이프라인 테스트
python rag_pipeline.py --query "AI란 무엇인가?" --search-type multimodal
```

## 💻 개발

### 요구사항
- Python 3.11+
- CUDA 11.8+ (GPU 가속용)
- Weaviate 1.24+
- 16GB+ RAM
- 40GB+ VRAM (전체)

### GPU 설정

```python
# 단일 GPU
config.TEXT_GENERATOR_DEVICE = "cuda:0"
config.RERANKER_DEVICE = "cuda:0"

# 멀티 GPU
config.TEXT_GENERATOR_DEVICE = "cuda:0"
config.RERANKER_DEVICE = "cuda:1"
config.EMBEDDING_DEVICE = "cuda:1"
```

### 사용자 정의 모델 추가

```python
# config.py에서
self.EMBEDDING_MODEL = "your-embedding-model"
self.RERANKER_MODEL_NAME = "your-reranker-model"
self.LLM_MODEL = "your-generation-model"
```

## 📝 라이선스

이 프로젝트는 Apache 2.0 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](../LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트들의 도움을 받았습니다:

- **[SGLang](https://github.com/sgl-project/sglang)**: 고성능 LLM 서빙 프레임워크
- **[Weaviate](https://github.com/weaviate/weaviate)**: 지식 관리를 위한 벡터 데이터베이스

## 🎓 Citation

이 프로젝트를 연구에 사용하시는 경우, 다음 논문들을 인용해주세요:

### SGLang
```bibtex
@misc{zheng2023sglang,
  title={SGLang: Efficient Execution of Structured Language Model Programs},
  author={Lianmin Zheng and Liangsheng Yin and Zhiqiang Xie and Jeff Huang and Chuyue Sun and Cody Hao Yu and Shiyi Cao and Christos Kozyrakis and Ion Stoica and Joseph E. Gonzalez and Clark Barrett and Ying Sheng},
  year={2023},
  url={https://github.com/sgl-project/sglang}
}
```

## 🌐 데모 사이트

시스템을 직접 사용해보세요: [https://quantuss.hnextits.com/](https://quantuss.hnextits.com/)

## 👥 기여자

이 프로젝트는 다음 팀원들이 개발했습니다:

- **Lim** - [junseung_lim@hnextits.com](mailto:junseung_lim@hnextits.com)
- **Jeong** - [jeongnext@hnextits.com](mailto:jeongnext@hnextits.com)
- **Ryu** - [fbgjungits@hnextits.com](mailto:fbgjungits@hnextits.com)

## 📧 문의

질문이나 피드백은 위의 이메일 주소로 연락하거나 GitHub에서 이슈를 열어주세요.

## 🤝 기여하기

기여를 환영합니다! Pull Request를 자유롭게 제출해주세요.

---

<div align="center">
Made with 🩸💦😭 by Nextits Team
</div>
