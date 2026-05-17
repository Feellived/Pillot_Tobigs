# Pillot

**Tobig's Conference 프로젝트**

알약 이미지에서 시작해 약물 인식, 속성 분류, 상호작용 해석, 복약 안내까지 연결하는 AI 기반 복약 지원 프로젝트입니다. | 유주형 · 한수영 · 김지우 · 조윤수  

---

## 1. 프로젝트가 해결하려는 문제

여러 약을 함께 복용하는 다약제 환경에서는 약물 간 상호작용, 중복 복용, 복용 시점 혼동이 쉽게 발생합니다. 특히 고령층이나 만성질환자처럼 복용 약물이 많은 사용자에게는 다음 문제가 반복됩니다.

- 약 봉투나 알약 사진만 보고 약을 구분하기 어렵다.
- 약 이름을 알아도 어떤 조합이 위험한지 판단하기 어렵다.
- 의료 정보가 전문용어 위주라 실제 복약 행동으로 이어지지 않는다.

Pillot은 이 문제를 `이미지 기반 약 인식`과 `사용자 친화적 복약 안내`로 연결해 해결하는 것을 목표로 합니다.

---

## 2. 최종적으로 지향하는 서비스 파이프라인

프로젝트의 장기 목표는 아래와 같은 end-to-end 복약 지원 흐름입니다.

```text
알약 이미지 입력
    ↓
알약 탐지 및 crop 생성
    ↓
약물 식별 또는 속성 분류
    ↓
공공/내부 약물 DB 매핑
    ↓
약물 상호작용 및 복약 위험 분석
    ↓
근거 기반 복약 안내 생성
```

핵심 아이디어는 단순히 "사진 속 객체를 맞히는 모델"을 만드는 것이 아니라, 인식 결과를 실제 복약 의사결정에 도움이 되는 정보로 바꾸는 것입니다.

---

## 3. 현재 구현된 범위

이 레포는 아직 전체 에이전트 구현체가 아니라, 위 파이프라인 중 컴퓨터 비전 앞단을 검증하는 실험 저장소입니다. 현재 포함된 범위는 다음과 같습니다.

- `YOLOv11n` 기반 알약 detection baseline
- detection bbox를 분류 친화적 crop으로 바꾸는 crop baseline
- `EfficientNet-B3` 기반 속성 분류 baseline
- Colab + Google Drive/GCS 환경에서 재실행 가능한 실험 노트북

즉, 현재 저장소는 "알약을 어디서 어떻게 잘라서 어떤 표현으로 분류할 것인가"를 검증하는 단계에 가깝습니다.

이 레포에 앞으로 포함될 영역은 다음과 같습니다.

- 최종 약물명 식별 모델
- OCR 기반 각인 인식 파이프라인
- DUR/DrugBank 기반 상호작용 분석 엔진
- LLM/RAG 기반 복약 지도 생성 서비스
- 사용자 인터페이스 또는 배포용 API 서버

---

## 4. 현재 CV 파이프라인

현재 노트북 기준 실험 흐름은 아래와 같습니다.

```text
AI Hub detection manifest 로드
    ↓
YOLOv11n으로 알약 bbox 학습
    ↓
best.pt로 예측 bbox 생성
    ↓
원본 이미지에서 square crop 생성
    ↓
crop 이미지 기반 속성 분류 학습
```

세부적으로는 다음 원칙을 따릅니다.

- detection은 `single pill` subset에서 1-class 객체 탐지로 시작
- crop은 원본 해상도에서 수행하고 `reflect padding`을 사용
- 분류 입력용 resize는 crop 저장 시점이 아니라 classifier 직전에 수행
- classification은 현재 `color`와 `shape` 두 속성을 예측하는 multi-head 구조

이 구조를 택한 이유는, detection 성능과 classification 성능을 분리해서 병목을 분석하기 쉽기 때문입니다.

---

## 5. 저장소 구조

현재 레포의 핵심 구조는 아래와 같습니다.

```text
Pillot/
├── README.md
├── notebooks/
│   ├── YOLO/
│   │   └── jh_yolo_baseline.ipynb
│   ├── CROP/
│   │   └── jh_crop_baseline.ipynb
│   └── EfficientNet-B3/
│       └── jh_classification_baseline.ipynb
├── runs/
│   └── yolo11n_single_baseline/
└── cache/   # 로컬/Drive 실험 산출물, 기본적으로 버전 관리 대상 아님
```

각 폴더의 역할은 다음과 같습니다.

- `notebooks/YOLO`
  - detection baseline 실험
  - `ultralytics` 기반 학습, 검증, 가중치 저장

- `notebooks/CROP`
  - YOLO 또는 GT bbox 기반 crop 생성
  - chunk 단위 추론과 resume 가능한 저장 구조 포함

- `notebooks/EfficientNet-B3`
  - crop 이미지를 사용한 속성 분류 baseline
  - 현재는 `color_class1`, `drug_shape` 2-head 분류 실험 포함

- `runs/`
  - YOLO 학습 결과 예시
  - `best.pt`, `last.pt`, `results.csv`, 시각화 이미지 포함

- `cache/`
  - crop manifest, bbox cache, classification checkpoint 등 중간 산출물
  - 재생성 가능한 실험 캐시 성격

---

## 6. 노트북별 역할

### 6.1 `notebooks/YOLO/jh_yolo_baseline.ipynb`

알약 detection 베이스라인입니다.

- AI Hub `detection_single.csv` manifest를 읽습니다.
- 선택한 zip subset을 train/val로 구성합니다.
- COCO bbox를 YOLO 형식으로 변환합니다.
- `yolo11n.pt`로 1-class pill detector를 학습합니다.

이 노트북의 목적은 "알약 위치를 안정적으로 찾을 수 있는가?"를 확인하는 것입니다.

### 6.2 `notebooks/CROP/jh_crop_baseline.ipynb`

detection 결과를 classification/OCR 친화적인 crop으로 바꾸는 노트북입니다.

- GT bbox 또는 YOLO 예측 bbox를 선택할 수 있습니다.
- 원본 이미지에서 정사각형 margin crop을 만듭니다.
- 경계 밖은 `reflect padding`으로 채웁니다.
- bbox 예측과 crop 저장 모두 중간 재개가 가능하도록 설계되어 있습니다.

이 단계는 단순 전처리가 아니라, 실제 분류 성능을 크게 좌우하는 입력 품질 관리 단계입니다.

### 6.3 `notebooks/EfficientNet-B3/jh_classification_baseline.ipynb`

YOLO crop 기반 속성 분류 baseline입니다.

- `EfficientNet-B3` pretrained backbone을 사용합니다.
- 현재는 약물명 분류가 아니라 `색상`과 `형태`를 예측하는 2-head 분류 구조입니다.
- `StratifiedGroupKFold`를 사용해 image-level leakage를 막도록 split합니다.
- checkpoint는 validation macro-F1 기준으로 저장합니다.

이 노트북은 이후 `fine-grained drug ID classification` 또는 `OCR 결합 모델`로 확장하기 위한 출발점입니다.

---

## 7. 데이터와 의존성

이 저장소는 로컬 단독 실행형 패키지보다는, 다음 외부 자원과 함께 쓰는 실험용 코드 레포입니다.

- AI Hub 기반 detection manifest
- Google Drive / GCS 저장소
- Colab 런타임
- 속성 정보 조회용 약물 DB

그래서 노트북을 그대로 실행하려면 데이터 경로, Drive 구조, DB 접근 환경이 맞아야 합니다. 공개 레포만 clone해서 바로 재현되는 구조가 아닙니다.

---

## 8. 왜 이 프로젝트가 의미 있는가

이 프로젝트의 의의는 세 가지로 정리할 수 있습니다.

- **실용성**
  - 약 봉투, 처방전, 약통이 섞인 현실 환경에서 사용 가능한 복약 지원 인터페이스를 지향합니다.

- **의료 안전성**
  - 단순 분류 정확도보다, 실제로 위험한 복약 조합을 더 잘 드러내는 방향을 목표로 합니다.

- **설명 가능성**
  - 최종 단계에서는 사용자가 이해할 수 있는 복약 설명과 근거 제시까지 연결하려고 합니다.

즉, Pillot은 "알약 이미지를 맞히는 모델"이 아니라, "복약 맥락에서 의미 있는 종합적 판단을 돕는 시스템"을 목표로 합니다.

---

## 9. 현재 한계

현 시점의 저장소는 연구용 베이스라인이기 때문에 다음 한계가 있습니다.

- 실험이 노트북 중심으로 관리됩니다.
- subset 기반 검증이 많아 전체 서비스 성능을 직접 대변하지는 않습니다.
- 속성 분류와 최종 약물 식별은 아직 분리된 단계입니다.
- DB, OCR, RAG, 상호작용 분석은 아직 통합 파이프라인으로 정리되지 않았습니다.

---

## 10. 한눈에 보기

이 레포를 처음 보셨다면, 아래처럼 이해하면 됩니다.

- 이것은 "복약 지도 에이전트" 전체 구현체가 아니라 그 전단계 CV 연구 레포다.
- 현재 핵심은 `YOLO detection -> crop generation -> EfficientNet classification`이다.
- 최종 목표는 약물 인식 결과를 바탕으로 상호작용 해석과 복약 안내까지 제공하는 것이다.

필요하면 `notebooks/YOLO`부터 순서대로 읽는 것이 가장 빠릅니다.
