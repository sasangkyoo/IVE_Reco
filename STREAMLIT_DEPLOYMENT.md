# Streamlit Cloud 배포 가이드

## 📋 배포에 필요한 파일들

### 1. 필수 파일 (GitHub에 업로드)
- `recommender/app_streamlit.py` - 메인 Streamlit 앱
- `requirements.txt` - Python 패키지 의존성
- `.streamlit/config.toml` - Streamlit 설정
- `README.md` - 프로젝트 설명서
- `.gitignore` - Git 무시 파일 설정

### 2. 데이터 파일 (GitHub에 업로드)
- `preprocessed/user_profile.csv` - 사용자 프로필 데이터 (585,000명)
- `preprocessed/ads_profile.csv` - 광고 프로필 데이터 (445,260개)
- `correct_interactions.zip` - 압축된 상호작용 데이터 (54MB)
- `extract_data.py` - 압축 해제 스크립트

### 3. 스크립트 파일 (GitHub에 업로드)
- `create_correct_interactions.py` - 상호작용 데이터 생성 스크립트
- `user/create_user_profile_final.py` - 사용자 프로필 생성 스크립트
- `user/create_ads_profile_final.py` - 광고 프로필 생성 스크립트

## 🚀 배포 단계

### 1. GitHub 저장소 생성
```bash
# Git 저장소 초기화
git init
git add .
git commit -m "Initial commit: IVE 추천시스템"

# GitHub 저장소 연결 및 푸시
git remote add origin https://github.com/yourusername/ive-recommender.git
git branch -M main
git push -u origin main
```

### 2. Streamlit Cloud 배포
1. [Streamlit Cloud](https://share.streamlit.io/) 접속
2. "New app" 클릭
3. GitHub 저장소 연결
4. 메인 파일 경로: `recommender/app_streamlit.py`
5. 배포 실행

## ⚠️ 주의사항

- **압축 파일**: `correct_interactions.zip`은 54MB로 GitHub 제한 내
- **자동 해제**: Streamlit 앱 실행 시 자동으로 압축 해제
- **배포 시간**: 초기 배포 시 압축 해제로 인해 약간의 시간 소요

## 🔧 로컬 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# Streamlit 앱 실행
streamlit run recommender/app_streamlit.py
```

## 📊 최종 프로젝트 구조

```
M4_final_LLM/
├── recommender/
│   ├── app_streamlit.py          # 메인 Streamlit 앱
│   ├── reco_batch.py             # 배치 추천 시스템
│   └── eval_reco.py              # 추천 성능 평가
├── user/
│   ├── create_user_profile_final.py
│   └── create_ads_profile_final.py
├── preprocessed/
│   ├── user_profile.csv          # 사용자 프로필 데이터
│   └── ads_profile.csv           # 광고 프로필 데이터
├── input/save/
│   └── correct_interactions.csv  # 정제된 상호작용 데이터
├── create_correct_interactions.py
├── requirements.txt
├── .streamlit/config.toml
├── .gitignore
├── README.md
└── STREAMLIT_DEPLOYMENT.md
```

## 🎯 배포 후 확인사항

### 1. 기본 기능
- [ ] 앱이 정상적으로 로드되는지 확인
- [ ] 사용자 선택 기능 작동 확인
- [ ] 추천 결과 표시 확인

### 2. 상호작용 분석
- [ ] 상호작용 이력 표시 확인
- [ ] 총 리워드 금액 계산 정확성 확인
- [ ] 상호작용 타입 구분 확인

### 3. 데이터 일관성
- [ ] 총 상호작용 수와 테이블 표시 수 일치
- [ ] 고유 광고 수 정확성 확인
- [ ] 리워드 계산 정확성 확인

## 📈 성능 지표

### 추천 성능
- **정확도**: 코사인 유사도 + 타입/카테고리 가중치
- **속도**: 평균 1-2초
- **규모**: 583K 사용자, 445K 광고

### 데이터 품질
- **상호작용**: 1,477,341개 (정제됨)
- **사용자**: 583,386명
- **광고**: 445,260개

---

**IVE 추천시스템** - Streamlit Cloud 배포 가이드