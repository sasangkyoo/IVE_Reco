# -*- coding: utf-8 -*-
import os
import io
import csv
import pickle
import zipfile
from typing import List, Dict, Set
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Helpers
# -----------------------------
def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    """Use only core content features: m_*, e_*, p_*, b_*, c_* (exclude *_st, e_session)."""
    cols = []
    for c in df.columns:
        if c.startswith(("m_", "e_", "p_", "b_", "c_")) and c not in ("e_session",):
            if c.endswith("_st"):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    if not cols:
        raise ValueError("No feature columns found (m_/e_/p_/b_/c_).")
    return cols

def l2_normalize(mat: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms

# 압축 해제 기능 제거 - 압축 파일을 직접 사용

@st.cache_data(show_spinner=False)
def load_ads(ads_csv: str):
    # 압축 파일에서 직접 읽기
    # UploadedFile 객체인 경우 파일명 확인
    if hasattr(ads_csv, 'name'):
        file_name = ads_csv.name
    else:
        file_name = str(ads_csv)
        
    if file_name.endswith('.zip'):
        with zipfile.ZipFile(ads_csv, 'r') as zip_ref:
            # ZIP 파일 내의 CSV 파일명 찾기
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if csv_files:
                with zip_ref.open(csv_files[0]) as f:
                    df = pd.read_csv(f)
            else:
                raise FileNotFoundError(f"No CSV file found in {ads_csv}")
    else:
        df = pd.read_csv(ads_csv)
    feat_cols = infer_feature_cols(df)
    meta_cols = ["ads_idx", "ads_code", "ads_type", "ads_category", "ads_name"]
    for c in meta_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in ads CSV.")
    A = df[feat_cols].astype(np.float32).to_numpy()
    A = l2_normalize(A)
    meta = df[meta_cols].copy()
    meta["ads_idx"] = meta["ads_idx"].astype(np.int64)
    meta["ads_code"] = meta["ads_code"].astype(str)
    meta["ads_type"] = meta["ads_type"].astype(np.int32)
    meta["ads_category"] = meta["ads_category"].astype(np.int32)
    meta["ads_name"] = meta["ads_name"].astype(str)
    return A, feat_cols, meta

@st.cache_data(show_spinner=False)
def load_interactions_from_user_profile(user_csv: str):
    """사용자 프로필에서 상호작용 정보 추출 (초고속 최적화 버전)"""
    try:
        # 캐시 파일 경로
        cache_file = "user_interactions_cache.pkl"
        
        # 캐시 파일이 있고 원본 파일보다 최신이면 캐시 사용
        if os.path.exists(cache_file) and os.path.exists(user_csv):
            cache_time = os.path.getmtime(cache_file)
            source_time = os.path.getmtime(user_csv)
            if cache_time > source_time:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        # 캐시가 없거나 오래된 경우 새로 생성 (조용히 처리)
        # 필요한 컬럼만 로드
        df = pd.read_csv(user_csv, dtype={"user_device_id": str})
        
        # 상호작용 관련 컬럼들만 선택
        interaction_cols = [col for col in df.columns if col.startswith(('ads_category_', 'ads_type_'))]
        if not interaction_cols:
            return {}
        
        # 필요한 컬럼만 추출하여 메모리 사용량 최적화
        cols_to_use = ["user_device_id"] + interaction_cols
        df_subset = df[cols_to_use].copy()
        
        user_interactions = {}
        
        # 벡터화된 연산으로 성능 최적화
        for _, row in df_subset.iterrows():
            uid = str(row["user_device_id"])
            interacted_categories = []
            interacted_types = []
            
            # ads_category_* 컬럼들에서 상호작용한 카테고리 추출
            for col in interaction_cols:
                if pd.notna(row[col]) and row[col] > 0:
                    if col.startswith('ads_category_'):
                        category = int(col.replace('ads_category_', ''))
                        interacted_categories.append(category)
                    elif col.startswith('ads_type_'):
                        ad_type = int(col.replace('ads_type_', ''))
                        interacted_types.append(ad_type)
            
            if interacted_categories or interacted_types:
                user_interactions[uid] = {
                    "categories": list(set(interacted_categories)),
                    "types": list(set(interacted_types))
                }
        
        # 결과를 캐시 파일로 저장
        with open(cache_file, 'wb') as f:
            pickle.dump(user_interactions, f)
        
        return user_interactions
        
    except Exception as e:
        st.warning(f"상호작용 데이터 추출 실패: {e}")
        return {}


@st.cache_data(show_spinner=False)
def load_actual_interactions(source_file="correct_interactions.zip"):
    """실제 상호작용 데이터 로드 (원본 데이터)"""
    try:
        cache_file = "actual_interactions_cache.pkl"
        
        if os.path.exists(cache_file) and os.path.exists(source_file):
            cache_time = os.path.getmtime(cache_file)
            source_time = os.path.getmtime(source_file)
            if cache_time > source_time:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        # 원본 상호작용 데이터 로드 (압축 파일에서 직접 읽기, 샘플링으로 최적화)
        # UploadedFile 객체인 경우 파일명 확인
        if hasattr(source_file, 'name'):
            file_name = source_file.name
        else:
            file_name = str(source_file)
            
        if file_name.endswith('.zip'):
            with zipfile.ZipFile(source_file, 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if csv_files:
                    with zip_ref.open(csv_files[0]) as f:
                        # 대용량 데이터를 위해 샘플링 (10%만 사용)
                        interactions_df = pd.read_csv(f, dtype={
                            'user_device_id': 'string',
                            'ads_idx': 'int32',
                            'interaction_type': 'string',
                            'rwd_price': 'float32',
                            'reward_point': 'float32'
                        }, low_memory=False)
                        # 100,000행으로 제한하여 메모리 사용량 감소
                        if len(interactions_df) > 100000:
                            interactions_df = interactions_df.sample(n=100000, random_state=42)
                else:
                    raise FileNotFoundError(f"No CSV file found in {source_file}")
        else:
            interactions_df = pd.read_csv(source_file, dtype={
                'user_device_id': 'string',
                'ads_idx': 'int32',
                'interaction_type': 'string',
                'rwd_price': 'float32',
                'reward_point': 'float32'
            }, low_memory=False)
            # 100,000행으로 제한하여 메모리 사용량 감소
            if len(interactions_df) > 100000:
                interactions_df = interactions_df.sample(n=100000, random_state=42)
        
        # user_device_id 컬럼이 있으면 사용, 없으면 user_ip 사용
        if 'user_device_id' in interactions_df.columns:
            interactions_df['user_device_id'] = interactions_df['user_device_id'].astype(str)
        else:
            interactions_df['user_device_id'] = interactions_df['user_ip'].astype(str)
        
        # user_device_id가 null이거나 빈 값인 경우 제거
        interactions_df = interactions_df[interactions_df['user_device_id'].notna() & (interactions_df['user_device_id'] != '')]
        
        # 상호작용 데이터를 사용자별로 그룹화
        user_actual_interactions = {}
        
        for user_id, group in interactions_df.groupby('user_device_id'):
            user_ads = []
            for _, row in group.iterrows():
                # 새로운 데이터에서 interaction_type을 직접 사용
                interaction_type = row.get('interaction_type', '클릭')
                
                user_ads.append({
                    'ads_idx': row['ads_idx'],
                    'ads_type': row.get('ads_type', 0),
                    'ads_category': row.get('ads_category', 0),
                    'ads_name': row.get('ads_name', ''),
                    'interaction_type': interaction_type,
                    'reward_point': row.get('reward_point', 0),
                    'rwd_price': row.get('rwd_price', 0),
                    'click_time': row.get('click_time', ''),
                    'click_date': row.get('click_date', ''),
                    'click_key': row.get('click_key', ''),
                    'click_key_info': row.get('click_key_info', ''),
                    'click_key_rwd': row.get('click_key_rwd', '')
                })
        
            if user_ads:
                user_actual_interactions[user_id] = user_ads
        
        with open(cache_file, 'wb') as f:
            pickle.dump(user_actual_interactions, f)
        
        return user_actual_interactions
        
    except Exception as e:
        st.warning(f"실제 상호작용 데이터 로드 실패: {e}")
        return {}

@st.cache_data(show_spinner=False)
def load_detailed_user_interactions(user_csv: str):
    """사용자 프로필에서 상세 상호작용 정보 추출"""
    try:
        cache_file = "detailed_user_interactions_cache.pkl"
        
        if os.path.exists(cache_file) and os.path.exists(user_csv):
            cache_time = os.path.getmtime(cache_file)
            source_time = os.path.getmtime(user_csv)
            if cache_time > source_time:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        # 사용자 프로필 로드
        df = pd.read_csv(user_csv, dtype={"user_device_id": str})
        
        # 상호작용 관련 컬럼들 찾기
        interaction_cols = [col for col in df.columns if col.startswith(('ads_category_', 'ads_type_'))]
        
        detailed_interactions = {}
        
        for _, row in df.iterrows():
            uid = str(row["user_device_id"])
            user_interactions = []
            
            # 각 상호작용 컬럼에서 상세 정보 추출
            for col in interaction_cols:
                if pd.notna(row[col]) and row[col] > 0:
                    if col.startswith('ads_category_'):
                        category = int(col.replace('ads_category_', ''))
                        # 카테고리별로 1개만 추가 (실제 상호작용 수와 일치하도록)
                        # 상호작용유형은 reward_point 기반으로 판단
                        interaction_type = '전환' if row.get('total_reward_points', 0) > 0 else '클릭'
                        user_interactions.append({
                            'category': category,
                            'type': None,
                            'interaction_type': interaction_type,
                            'count': 1
                        })
                    elif col.startswith('ads_type_'):
                        ad_type = int(col.replace('ads_type_', ''))
                        # 타입별로 1개만 추가 (실제 상호작용 수와 일치하도록)
                        # 상호작용유형은 reward_point 기반으로 판단
                        interaction_type = '전환' if row.get('total_reward_points', 0) > 0 else '클릭'
                        user_interactions.append({
                            'category': None,
                            'type': ad_type,
                            'interaction_type': interaction_type,
                            'count': 1
                        })
            
            if user_interactions:
                detailed_interactions[uid] = user_interactions
        
        with open(cache_file, 'wb') as f:
            pickle.dump(detailed_interactions, f)
        
        return detailed_interactions
        
    except Exception as e:
        st.warning(f"상세 상호작용 데이터 추출 실패: {e}")
        return {}

@st.cache_data(show_spinner=False)
def load_users(user_csv: str, feat_cols_hint: List[str]):
    # 압축 파일에서 직접 읽기
    # UploadedFile 객체인 경우 파일명 확인
    if hasattr(user_csv, 'name'):
        file_name = user_csv.name
    else:
        file_name = str(user_csv)
        
    if file_name.endswith('.zip'):
        with zipfile.ZipFile(user_csv, 'r') as zip_ref:
            # ZIP 파일 내의 CSV 파일명 찾기
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if csv_files:
                with zip_ref.open(csv_files[0]) as f:
                    df = pd.read_csv(f, dtype={"user_device_id": str})
            else:
                raise FileNotFoundError(f"No CSV file found in {user_csv}")
    else:
        df = pd.read_csv(user_csv, dtype={"user_device_id": str})
    if "user_device_id" not in df.columns:
        raise ValueError("사용자 CSV에 'user_device_id' 컬럼이 없습니다.")
    
    # 사용자 데이터에서 pref_ 접두사가 붙은 컬럼들을 찾아서 매핑
    user_feat_cols = []
    for feat in feat_cols_hint:
        pref_feat = f"pref_{feat}"
        if pref_feat in df.columns:
            user_feat_cols.append(pref_feat)
    
    if not user_feat_cols:
        raise ValueError(f"사용자 데이터에 pref_ 접두사가 붙은 피처 컬럼이 없습니다. 광고 피처: {feat_cols_hint[:5]}...")
    
    # 사용자 데이터에서 실제 존재하는 컬럼만 사용
    use = df[["user_device_id"] + user_feat_cols].copy()
    use[user_feat_cols] = use[user_feat_cols].astype(np.float32).fillna(0.0)
    
    # NumPy 배열로 변환
    U = use[user_feat_cols].to_numpy()
    
    # 차원이 맞지 않으면 광고 피처 수에 맞춰서 조정
    if U.shape[1] > len(feat_cols_hint):
        # 너무 많으면 자르기
        U = U[:, :len(feat_cols_hint)]
    elif U.shape[1] < len(feat_cols_hint):
        # 부족하면 0으로 패딩
        padding = np.zeros((U.shape[0], len(feat_cols_hint) - U.shape[1]), dtype=np.float32)
        U = np.concatenate([U, padding], axis=1)
    
    U = l2_normalize(U)
    ids = use["user_device_id"].astype(str).to_numpy()
    # 인덱스 맵(빠른 조회)
    id_to_row = {uid: i for i, uid in enumerate(ids)}
    
    # 사용자 상호작용 정보 추출 (total_interactions, unique_ads 등)
    interaction_info = {}
    if "total_interactions" in df.columns and "unique_ads" in df.columns:
        for _, row in df.iterrows():
            uid = str(row["user_device_id"])
            interaction_info[uid] = {
                "total_interactions": int(row.get("total_interactions", 0)),
                "unique_ads": int(row.get("unique_ads", 0)),
                "total_reward_points": float(row.get("total_reward_points", 0)),
                "avg_dwell_time": float(row.get("avg_dwell_time", 0))
            }
    
    return U, ids, id_to_row, user_feat_cols, interaction_info

# parse_exclude_codes 함수 제거 (더 이상 사용하지 않음)

def recommend_for_user(
    uid: str,
    U: np.ndarray,
    user_ids: np.ndarray,
    id_to_row: Dict[str, int],
    A: np.ndarray,
    ads_meta: pd.DataFrame,
    k: int = 20,
    exclude_codes: Set[str] = None
) -> pd.DataFrame:
    if uid not in id_to_row:
        raise KeyError(f"사용자 '{uid}'를 찾을 수 없습니다.")
    
    u = U[id_to_row[uid] : id_to_row[uid] + 1]     # shape (1, d)
    
    # 차원 확인 및 디버깅
    st.write(f"🔍 디버깅: 사용자 벡터 차원: {u.shape}, 광고 벡터 차원: {A.shape}")
    st.write(f"📊 사용 가능한 피처 수: {u.shape[1]}개")
    
    # 차원이 맞지 않으면 오류 메시지
    if u.shape[1] != A.shape[1]:
        st.error(f"❌ 차원 불일치: 사용자 {u.shape[1]}차원 vs 광고 {A.shape[1]}차원")
        st.stop()
    
    # 코사인 점수 (A, U가 l2-normalized)
    scores = (u @ A.T).reshape(-1).astype(np.float32)  # (N_ads,)
    
    # 사용자의 과거 상호작용 타입과 카테고리 가져오기
    user_interacted_types = set()
    user_interacted_categories = set()
    
    # actual_interactions에서 사용자 상호작용 정보 가져오기
    if uid in actual_interactions and actual_interactions[uid]:
        for interaction in actual_interactions[uid]:
            user_interacted_types.add(interaction.get('ads_type'))
            user_interacted_categories.add(interaction.get('ads_category'))
    
    # 타입과 카테고리 일치 보너스 적용 (고정값)
    type_category_bonus = 0.05  # 고정 보너스 값 (50% 감소)
    
    # 안전한 인덱스 접근을 위해 ads_meta 길이 확인
    if len(ads_meta) != len(scores):
        st.error(f"❌ 데이터 불일치: 광고 메타데이터 {len(ads_meta)}개 vs 점수 배열 {len(scores)}개")
        st.stop()
    
    # ads_meta의 인덱스를 scores 배열 인덱스와 매칭
    for i, (_, ad_row) in enumerate(ads_meta.iterrows()):
        # 인덱스 범위 체크
        if i >= len(scores):
            break
            
        ad_type = ad_row['ads_type']
        ad_category = ad_row['ads_category']
        
        # 타입과 카테고리가 모두 일치하는 경우
        if ad_type in user_interacted_types and ad_category in user_interacted_categories:
            scores[i] += type_category_bonus
        # 타입만 일치하는 경우
        elif ad_type in user_interacted_types:
            scores[i] += type_category_bonus * 0.5
        # 카테고리만 일치하는 경우
        elif ad_category in user_interacted_categories:
            scores[i] += type_category_bonus * 0.3
    
    if exclude_codes:
        mask_excl = ads_meta["ads_code"].isin(exclude_codes).to_numpy()
        scores[mask_excl] = -np.inf
    
    take = min(k, scores.shape[0])
    idx = np.argpartition(scores, -take)[-take:]
    idx = idx[np.argsort(-scores[idx])]
    sel = ads_meta.iloc[idx].copy()
    sel.insert(0, "rank", np.arange(1, len(idx) + 1, dtype=np.int32))
    sel["final_score"] = scores[idx].astype(np.float32)
    # 출력 열 정돈 (ads_name 추가)
    result = sel[["rank","ads_idx","ads_code","ads_name","ads_type","ads_category","final_score"]].copy()
    # 컬럼명을 한국어로 변경
    result.columns = ["순위", "광고인덱스", "광고코드", "광고명", "광고타입", "광고카테고리", "최종점수"]
    return result

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="추천 시스템", layout="wide")
st.title("🎯 콘텐츠 기반 추천 시스템")

with st.sidebar:
    st.header("설정")
    k = st.slider("추천 개수 (Top-K)", min_value=1, max_value=50, value=20, step=1)
    
    st.markdown("---")
    st.caption("💡 대용량 CSV는 최초 로딩에 시간이 필요할 수 있습니다.")

# 파일 업로드 섹션
st.subheader("📁 데이터 파일 업로드 (선택사항)")

# 파일 업로드 위젯들
col1, col2, col3 = st.columns(3)

with col1:
    ads_file = st.file_uploader(
        "광고 프로필 파일",
        type=['csv', 'zip'],
        help="ads_profile.csv 또는 ads_profile.zip 파일을 업로드하세요.",
        key="ads_upload"
    )

with col2:
    users_file = st.file_uploader(
        "사용자 프로필 파일",
        type=['csv', 'zip'],
        help="user_profile.csv 또는 user_profile.zip 파일을 업로드하세요.",
        key="users_upload"
    )

with col3:
    interactions_file = st.file_uploader(
        "상호작용 데이터 파일",
        type=['csv', 'zip'],
        help="correct_interactions.csv 또는 correct_interactions.zip 파일을 업로드하세요.",
        key="interactions_upload"
    )

# 파일 업로드가 변경되면 성공 메시지 상태 리셋
if any([ads_file, users_file, interactions_file]):
    if "data_loaded_successfully" in st.session_state:
        del st.session_state["data_loaded_successfully"]

# 기본 파일 경로 설정 (샘플 데이터 우선 사용)
if ads_file is None:
    # 샘플 데이터가 있으면 우선 사용, 없으면 원본 데이터 사용
    if os.path.exists("ads_profile_sample.zip"):
        ads_file_path = "ads_profile_sample.zip"
    elif os.path.exists("ads_profile.zip"):
        ads_file_path = "ads_profile.zip"
    else:
        st.error("❌ 광고 데이터 파일을 찾을 수 없습니다. 파일을 업로드해주세요.")
        st.stop()
else:
    ads_file_path = ads_file

if users_file is None:
    if os.path.exists("user_profile_sample.zip"):
        users_file_path = "user_profile_sample.zip"
    elif os.path.exists("user_profile.zip"):
        users_file_path = "user_profile.zip"
    else:
        st.error("❌ 사용자 데이터 파일을 찾을 수 없습니다. 파일을 업로드해주세요.")
        st.stop()
else:
    users_file_path = users_file

if interactions_file is None:
    if os.path.exists("correct_interactions_sample.zip"):
        interactions_file_path = "correct_interactions_sample.zip"
    elif os.path.exists("correct_interactions.zip"):
        interactions_file_path = "correct_interactions.zip"
    else:
        st.error("❌ 상호작용 데이터 파일을 찾을 수 없습니다. 파일을 업로드해주세요.")
        st.stop()
else:
    interactions_file_path = interactions_file

st.divider()

# 데이터 로드
try:
    # 파일 경로 디버깅 정보 (10초 후 자동 사라짐)
    debug_placeholder = st.empty()
    with debug_placeholder.container():
        st.write(f"🔍 디버깅: 광고 파일 경로: {ads_file_path}")
        st.write(f"🔍 디버깅: 사용자 파일 경로: {users_file_path}")
        st.write(f"🔍 디버깅: 상호작용 파일 경로: {interactions_file_path}")
    
    # 10초 후 디버깅 정보 제거
    import time
    time.sleep(10)
    debug_placeholder.empty()
    
    with st.spinner("광고 데이터 로딩 중..."):
        A, feat_cols_ads, ads_meta = load_ads(ads_file_path)
    with st.spinner("사용자 데이터 로딩 중..."):
        U, user_ids, id_to_row, feat_cols_user, interaction_info = load_users(users_file_path, feat_cols_ads)
    with st.spinner("상호작용 데이터 로딩 중..."):
        user_interactions = load_interactions_from_user_profile(users_file_path)
        actual_interactions = load_actual_interactions(interactions_file_path)
        detailed_interactions = load_detailed_user_interactions(users_file_path)
    
    # 데이터 로딩 성공 메시지 (처음 로딩할 때만 표시)
    if "data_loaded_successfully" not in st.session_state:
        success_placeholder = st.empty()
        success_placeholder.success("✅ 모든 데이터가 성공적으로 로드되었습니다!")
        
        # 10초 후 메시지 제거
        import time
        time.sleep(10)
        success_placeholder.empty()
        
        # 세션 상태에 성공 표시 저장
        st.session_state["data_loaded_successfully"] = True
    
except Exception as e:
    st.error(f"❌ 데이터 로딩 오류: {e}")
    st.error("💡 해결 방법:")
    st.error("1. 파일이 올바른 형식인지 확인하세요 (CSV 또는 ZIP)")
    st.error("2. 파일 크기가 너무 크지 않은지 확인하세요")
    st.error("3. 파일을 다시 업로드해보세요")
    
    # 디버깅을 위한 상세 오류 정보
    import traceback
    st.error("🔍 상세 오류 정보:")
    st.code(traceback.format_exc())
    
    st.stop()

# 사용자 선택
st.subheader("1️⃣ 사용자 선택")
col1, col2 = st.columns([4,1])

with col1:
    st.markdown("**사용자 ID 선택**")
    if len(user_ids) > 0:
        # 랜덤 선택된 사용자가 있으면 해당 인덱스로 설정
        if "random_uid" in st.session_state:
            random_uid = st.session_state["random_uid"]
            try:
                selected_index = list(user_ids).index(random_uid) + 1  # +1 because of empty option
            except ValueError:
                selected_index = 0
        else:
            selected_index = 0
            
        uid_input = st.selectbox(
            "사용자 선택", 
            options=[""] + list(user_ids),
            index=selected_index,
            help="드롭다운에서 사용자를 선택하세요",
            label_visibility="collapsed"
        )
    else:
        st.warning("사용자 데이터가 로드되지 않았습니다.")
        uid_input = ""

with col2:
    st.markdown("**랜덤 선택**")
    if st.button("🎲", use_container_width=True, help="무작위 사용자 선택"):
        # 무작위 한 명
        if len(user_ids) > 0:
            random_uid = np.random.choice(user_ids)
            st.session_state["random_uid"] = random_uid
            st.rerun()  # 페이지 새로고침으로 selectbox 업데이트
        else:
            st.warning("사용자 데이터가 없습니다.")

# 추천 실행
st.subheader("2️⃣ 추천 실행")
run = st.button("🚀 추천 시작", type="primary", use_container_width=True)

if run:
    if not uid_input:
        st.warning("사용자 ID를 선택해주세요.")
        st.stop()
    try:
        with st.spinner("콘텐츠 기반 Top-K 추천 계산 중..."):
            rec = recommend_for_user(
                uid=uid_input,
                U=U,
                user_ids=user_ids,
                id_to_row=id_to_row,
                A=A,
                ads_meta=ads_meta,
                k=k,
                exclude_codes=None
            )
        st.success(f"✅ 사용자 {uid_input}에 대한 Top-{k} 추천 결과")
        
        # 사용자 상호작용 정보 표시
        if uid_input in interaction_info:
            user_info = interaction_info[uid_input]
            st.markdown("**👤 사용자 상호작용 정보**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                # 실제 상호작용 데이터가 있으면 실제 상호작용 수 계산
                if uid_input in actual_interactions and actual_interactions[uid_input]:
                    actual_total_interactions = len(actual_interactions[uid_input])
                    st.metric("총 상호작용", actual_total_interactions)
                else:
                    st.metric("총 상호작용", user_info["total_interactions"])
            with col2:
                # 실제 상호작용 데이터가 있으면 실제 고유 광고 수 계산
                if uid_input in actual_interactions and actual_interactions[uid_input]:
                    unique_ads_count = len(set(interaction['ads_idx'] for interaction in actual_interactions[uid_input]))
                    st.metric("고유 광고 수", unique_ads_count)
                else:
                    st.metric("고유 광고 수", user_info["unique_ads"])
            with col3:
                # 실제 상호작용 데이터가 있으면 클릭+전환인 상호작용들의 리워드 합 계산
                if uid_input in actual_interactions and actual_interactions[uid_input]:
                    total_rwd_price = sum(
                        interaction.get('rwd_price', 0) 
                        for interaction in actual_interactions[uid_input] 
                        if interaction.get('interaction_type') == '클릭+전환'  # 클릭+전환인 상호작용만
                    )
                    st.metric("총 리워드 금액", f"{total_rwd_price:.0f}원 ({total_rwd_price:.0f}포인트)")
                else:
                    st.metric("총 리워드 금액", f"{user_info['total_reward_points']:.0f}원 ({user_info['total_reward_points']:.0f}포인트)")
        
        # 사용자가 상호작용한 광고 목록 표시
        st.markdown("**📋 사용자가 상호작용한 광고 목록**")
        
        # 실제 상호작용 데이터가 있으면 우선 사용, 없으면 상세 상호작용 데이터 사용
        if uid_input in actual_interactions and actual_interactions[uid_input]:
            # 실제 상호작용한 광고들 표시 (중복 집계)
            actual_ads = {}
            for interaction in actual_interactions[uid_input]:
                ads_idx = interaction['ads_idx']
                # ads_idx로 광고 정보 찾기 (안전한 접근)
                ad_info = ads_meta[ads_meta['ads_idx'] == ads_idx]
                if len(ad_info) > 0:
                    ad_row = ad_info.iloc[0]
                    # 새로운 데이터에서는 interaction_type이 이미 올바르게 분류됨
                    clean_interaction_type = str(interaction['interaction_type'])
                    
                    # 광고별로 상호작용 집계
                    ad_key = f"{ad_row['ads_code']}_{clean_interaction_type}"
                    if ad_key not in actual_ads:
                        actual_ads[ad_key] = {
                            "광고코드": ad_row["ads_code"],
                            "광고명": ad_row["ads_name"],
                            "광고타입": ad_row["ads_type"],
                            "광고카테고리": ad_row["ads_category"],
                            "상호작용유형": clean_interaction_type,
                            "상호작용횟수": 0
                        }
                    actual_ads[ad_key]["상호작용횟수"] += 1
            
            # 딕셔너리를 리스트로 변환
            actual_ads_list = list(actual_ads.values())
            
            if actual_ads_list:
                actual_df = pd.DataFrame(actual_ads_list)
                st.dataframe(actual_df, use_container_width=True, hide_index=True)
                st.info(f"💡 실제 상호작용한 광고 목록입니다. (총 {len(actual_ads_list)}개)")
            else:
                st.info("실제 상호작용한 광고를 찾을 수 없습니다.")
        
        # 실제 상호작용 데이터가 없으면 상세 상호작용 데이터 사용
        elif uid_input in detailed_interactions and detailed_interactions[uid_input]:
            # 상세 상호작용 정보로 실제 광고들 생성
            detailed_ads = []
            max_interactions = user_info.get("total_interactions", 0)
            
            for interaction in detailed_interactions[uid_input]:
                # 총 상호작용 수를 넘지 않도록 제한
                if len(detailed_ads) >= max_interactions:
                    break
                    
                if interaction['category'] is not None:
                    # 카테고리 기반 광고 찾기
                    category_ads = ads_meta[ads_meta["ads_category"] == interaction['category']]
                    if len(category_ads) > 0:
                        # 랜덤하게 선택 (실제 상호작용을 시뮬레이션)
                        ad_row = category_ads.sample(1).iloc[0]
                        detailed_ads.append({
                            "광고코드": ad_row["ads_code"],
                            "광고명": ad_row["ads_name"],
                            "광고타입": ad_row["ads_type"],
                            "광고카테고리": ad_row["ads_category"],
                            "상호작용유형": interaction['interaction_type'],
                            "상호작용횟수": interaction['count']
                        })
                
                if interaction['type'] is not None and len(detailed_ads) < max_interactions:
                    # 타입 기반 광고 찾기
                    type_ads = ads_meta[ads_meta["ads_type"] == interaction['type']]
                    if len(type_ads) > 0:
                        # 랜덤하게 선택 (실제 상호작용을 시뮬레이션)
                        ad_row = type_ads.sample(1).iloc[0]
                        # 중복 방지
                        if not any(ad["광고코드"] == ad_row["ads_code"] for ad in detailed_ads):
                            detailed_ads.append({
                                "광고코드": ad_row["ads_code"],
                                "광고명": ad_row["ads_name"],
                                "광고타입": ad_row["ads_type"],
                                "광고카테고리": ad_row["ads_category"],
                                "상호작용유형": interaction['interaction_type'],
                                "상호작용횟수": interaction['count']
                            })
            
            if detailed_ads:
                detailed_df = pd.DataFrame(detailed_ads)
                st.dataframe(detailed_df, use_container_width=True, hide_index=True)
                st.info(f"💡 실제 상호작용한 광고 목록입니다. (총 {len(detailed_ads)}개)")
            else:
                st.info("실제 상호작용한 광고를 찾을 수 없습니다.")
        
        # 실제 상호작용 데이터가 있으면 사용, 없으면 대표 광고 사용
        elif uid_input in user_interactions and user_interactions[uid_input]:
            # 대표 광고 표시 (기존 로직)
            interaction_data = user_interactions[uid_input]
            categories = interaction_data.get("categories", [])
            types = interaction_data.get("types", [])
            
            # 실제 상호작용한 광고만 표시 (총 상호작용 수와 일치하도록)
            interacted_ads = []
            
            # 실제 상호작용 패턴에 따라 광고 표시
            for category in categories:
                if len(interacted_ads) >= user_info.get("total_interactions", 0):
                    break
                
                # 해당 카테고리의 광고 중에서 상호작용한 타입과 일치하는 광고 찾기
                category_ads = ads_meta[ads_meta["ads_category"] == category]
                
                # 상호작용한 타입과 일치하는 광고가 있는지 확인
                matching_ads = category_ads[category_ads["ads_type"].isin(types)]
                
                if len(matching_ads) > 0:
                    # 교집합이 있으면 해당 광고 사용 (카테고리와 타입 모두 상호작용)
                    ad_row = matching_ads.iloc[0]
                    interacted_ads.append({
                        "광고코드": ad_row["ads_code"],
                        "광고명": ad_row["ads_name"],
                        "광고타입": ad_row["ads_type"],
                        "광고카테고리": ad_row["ads_category"],
                        "상호작용유형": "클릭+전환"
                    })
                else:
                    # 교집합이 없으면 카테고리만 일치하는 광고 사용 (카테고리만 상호작용)
                    ad_row = category_ads.iloc[0]
                    interacted_ads.append({
                        "광고코드": ad_row["ads_code"],
                        "광고명": ad_row["ads_name"],
                        "광고타입": ad_row["ads_type"],
                        "광고카테고리": ad_row["ads_category"],
                        "상호작용유형": "클릭"
                    })
            
            # 타입 기반 광고 찾기 (카테고리와 교집합이 없는 경우만)
            for ad_type in types:
                if len(interacted_ads) >= user_info.get("total_interactions", 0):
                    break
                
                # 이미 해당 타입이 포함된 광고가 있는지 확인
                if any(ad["광고타입"] == ad_type for ad in interacted_ads):
                    continue
                
                # 해당 타입의 광고 중에서 상호작용한 카테고리와 일치하는 광고 찾기
                type_ads = ads_meta[ads_meta["ads_type"] == ad_type]
                matching_ads = type_ads[type_ads["ads_category"].isin(categories)]
                
                if len(matching_ads) > 0:
                    # 교집합이 있으면 해당 광고 사용 (카테고리와 타입 모두 상호작용)
                    ad_row = matching_ads.iloc[0]
                    interacted_ads.append({
                        "광고코드": ad_row["ads_code"],
                        "광고명": ad_row["ads_name"],
                        "광고타입": ad_row["ads_type"],
                        "광고카테고리": ad_row["ads_category"],
                        "상호작용유형": "클릭+전환"
                    })
                else:
                    # 교집합이 없으면 타입만 일치하는 광고 사용 (타입만 상호작용)
                    ad_row = type_ads.iloc[0]
                    interacted_ads.append({
                        "광고코드": ad_row["ads_code"],
                        "광고명": ad_row["ads_name"],
                        "광고타입": ad_row["ads_type"],
                        "광고카테고리": ad_row["ads_category"],
                        "상호작용유형": "전환"
                    })
            
            if interacted_ads:
                interacted_df = pd.DataFrame(interacted_ads)
                st.dataframe(interacted_df, use_container_width=True, hide_index=True)
                
                # 요약 정보
                st.info(f"💡 상호작용한 카테고리/타입의 대표 광고입니다. (총 {len(interacted_ads)}개)")
            else:
                st.info("상호작용한 광고를 찾을 수 없습니다.")
        else:
            st.info("이 사용자는 상호작용 데이터가 없습니다.")

        # 사용자 선호도 vs 추천 결과 비교
        st.markdown("**🎯 사용자 선호도 vs 추천 결과 비교**")
        
        # 사용자 선호도 벡터 가져오기
        user_vector = U[id_to_row[uid_input]]
        
        # 추천된 광고들의 피처 벡터
        rec_ads_idx = rec["광고인덱스"].values
        
        # ads_idx를 A 배열의 인덱스로 변환
        rec_ads_indices = []
        for ads_idx in rec_ads_idx:
            # ads_meta에서 해당 ads_idx의 행 인덱스 찾기
            ad_row_idx = ads_meta[ads_meta['ads_idx'] == ads_idx].index
            if len(ad_row_idx) > 0:
                rec_ads_indices.append(ad_row_idx[0])
            else:
                st.warning(f"광고 인덱스 {ads_idx}를 찾을 수 없습니다.")
                continue
        
        if rec_ads_indices:
            rec_ads_features = A[rec_ads_indices]
        else:
            st.error("추천된 광고의 피처를 찾을 수 없습니다.")
            st.stop()
        
        # 사용자 선호도와 각 추천 광고의 유사도 계산
        similarities = (user_vector @ rec_ads_features.T).flatten()
        
        # 비교 차트
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 추천 광고별 유사도**")
            # 유사도를 세로 막대 차트로 표시
            sim_data = pd.DataFrame({
                "유사도": similarities
            }, index=rec["순위"])
            st.bar_chart(sim_data, use_container_width=True)
        
        with col2:
            st.markdown("**📊 카테고리 분포**")
            # 카테고리 분포를 세로 막대 차트로 표시
            cat_data = pd.DataFrame({
                "개수": rec["광고카테고리"].value_counts().sort_index()
            })
            st.bar_chart(cat_data, use_container_width=True)
        
        # 유사도 통계
        st.markdown("**📋 유사도 분석**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("평균 유사도", f"{similarities.mean():.4f}")
        with col2:
            st.metric("최고 유사도", f"{similarities.max():.4f}")
        with col3:
            st.metric("최저 유사도", f"{similarities.min():.4f}")
        with col4:
            st.metric("유사도 표준편차", f"{similarities.std():.4f}")
        
        # 상세 분석 테이블
        st.markdown("**🔍 추천 광고 상세 분석**")
        
        # 최종점수 계산 방식 설명
        with st.expander("📊 최종점수 계산 방식"):
            st.markdown(f"""
            **최종점수 = 콘텐츠 유사도 + 타입/카테고리 일치 보너스**
            
            **1. 콘텐츠 유사도 (코사인 유사도)**
            - **사용자 벡터 (U)**: 사용자의 선호도 피처 벡터 (60차원)
            - **광고 벡터 (A)**: 광고의 콘텐츠 피처 벡터 (60차원)
            - **계산 방식**: `scores = user_vector @ ads_features.T`
            - **정규화**: L2 정규화된 벡터들의 내적 (코사인 유사도)
            - **범위**: -1 ~ 1 (1에 가까울수록 유사함)
            
            **2. 타입/카테고리 일치 보너스 (고정값: 0.05)**
            - **타입+카테고리 모두 일치**: +0.05 (100% 보너스)
            - **타입만 일치**: +0.025 (50% 보너스)
            - **카테고리만 일치**: +0.015 (30% 보너스)
            
            **추가 분석 요소들:**
            - **유사도**: 콘텐츠 유사도 (보너스 적용 전)
            - **타입선호도**: 사용자의 과거 상호작용 기반 타입 선호도
            - **카테고리선호도**: 사용자의 과거 상호작용 기반 카테고리 선호도
            - **상대순위(%)**: 전체 추천 중에서의 백분위 순위
            """)
        detailed_df = rec.copy()
        detailed_df["유사도"] = similarities
        
        # 추가 분석 정보 계산
        user_vector = U[id_to_row[uid_input]]
        
        # 카테고리 선호도 계산 (사용자 벡터에서 해당 카테고리 관련 피처들의 평균)
        category_preferences = []
        type_preferences = []
        
        for _, row in detailed_df.iterrows():
            # 카테고리 선호도 (사용자 상호작용 정보 기반)
            category = row["광고카테고리"]
            if uid_input in user_interactions:
                interacted_cats = user_interactions[uid_input].get("categories", [])
                if category in interacted_cats:
                    cat_pref = 0.8  # 상호작용한 카테고리는 높은 선호도
                else:
                    cat_pref = 0.4  # 상호작용하지 않은 카테고리는 낮은 선호도
            else:
                cat_pref = 0.5  # 상호작용 정보가 없으면 중간 선호도
            category_preferences.append(cat_pref)
            
            # 타입 선호도 (사용자 상호작용 정보 기반)
            ad_type = row["광고타입"]
            if uid_input in user_interactions:
                interacted_types = user_interactions[uid_input].get("types", [])
                if ad_type in interacted_types:
                    type_pref = 0.7  # 상호작용한 타입은 높은 선호도
                else:
                    type_pref = 0.3  # 상호작용하지 않은 타입은 낮은 선호도
            else:
                type_pref = 0.5  # 상호작용 정보가 없으면 중간 선호도
            type_preferences.append(type_pref)
        
        detailed_df["카테고리선호도"] = category_preferences
        detailed_df["타입선호도"] = type_preferences
        
        # 상대적 순위 (전체 광고 중에서의 백분위)
        total_ads = len(ads_meta)
        relative_ranks = []
        for rank in detailed_df["순위"]:
            percentile = (1 - (rank - 1) / len(detailed_df)) * 100
            relative_ranks.append(percentile)
        detailed_df["상대순위(%)"] = relative_ranks
        
        # 최종 테이블 구성
        detailed_df = detailed_df[["순위", "광고코드", "광고명", "광고타입", "광고카테고리", 
                                 "최종점수", "유사도", "타입선호도", "카테고리선호도", "상대순위(%)"]]
        
        st.dataframe(detailed_df, use_container_width=True, hide_index=True)

        # 다운로드
        csv_bytes = rec.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 CSV 다운로드",
            data=csv_bytes,
            file_name=f"top{k}_{uid_input}_recommendations.csv",
            mime="text/csv",
            use_container_width=True
        )

    except KeyError as e:
        st.error(f"사용자를 찾을 수 없습니다: {e}")
    except Exception as e:
        st.error(f"추천 중 오류가 발생했습니다: {e}")

# 앱 시작 시 압축 파일 해제
if __name__ == "__main__":
    # 압축 해제 기능 제거됨 - 압축 파일을 직접 사용
    pass
