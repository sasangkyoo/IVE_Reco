#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
배치 처리용 사용자 프로필 생성 스크립트
- 청크 단위로 데이터 처리
- 메모리 효율적
- 중간 결과 저장으로 안전한 처리
"""

import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 참조시각/스캔 모드
REF_TIME_MODE = "auto"  # {"now","data_max","auto"}
STALE_THRESHOLD_DAYS = 7  # auto일 때 now - global_max > 7d면 data_max 사용

# A1
RECENCY_HALF_WINDOW_HOURS = 72  # tau_recency 분모
TAU_MAX = 0.4

# A2
SHORT_WINDOWS_HOURS = (72, 168, 336)  # 72h -> 7d -> 14d 적응형
TIME_DECAY_HOURS = 24           # 지수가중 시간 상수
W_CLICK_INFO = 1.0
W_CLICK      = 1.5
W_CLICK_RWD  = 1.5

# A3
NOVELTY_WINDOW_DAYS = 14
LAPLACE_MU = 14.0               # 카테고리 수와 동일하게

# 콘텐츠 키(ads_profile와 동일해야 함)
M_KEYS = ["m_fun","m_social","m_rewards","m_savings","m_trust","m_conv","m_growth","m_status","m_curiosity","m_habit","m_safety"]
E_KEYS = ["e_casual","e_hardcore","e_freq","e_multi","e_retention"]
P_KEYS = ["p_install","p_coupon","p_fomo","p_exclusive","p_trial"]
B_KEYS = ["b_loyalty","b_nostalgia","b_trust","b_award","b_local","b_global"]
C_KEYS = ["c_price","c_premium","c_freq","c_risk","c_recurring","c_big"]
ALL_CONTENT_KEYS = M_KEYS + E_KEYS + P_KEYS + B_KEYS + C_KEYS

CATEGORY_IDS = list(range(14))  # 0..13

def parse_local_ts(s: pd.Series) -> pd.Series:
    """
    원천이 이미 로컬시간(naive)임을 전제로 안전 파싱.
    tz 정보를 붙이거나 변환하지 않는다.
    실패는 NaT로 처리.
    """
    return pd.to_datetime(s, errors="coerce")  # tz 미부여

def percentile_rank_0_1(series: pd.Series) -> pd.Series:
    s = series.astype("float32")
    if s.notna().sum() <= 1:
        return s.fillna(0.5)
    return s.rank(method="average", pct=True).astype("float32")

class BatchUserProfileCreator:
    """배치 처리용 사용자 프로필 생성 클래스"""
    
    def __init__(self, data_path: str = "preprocessed", chunk_size: int = 100000):
        # 현재 스크립트 위치를 기준으로 프로젝트 루트 찾기
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        self.data_path = os.path.join(project_root, data_path)
        self.chunk_size = chunk_size
        self.temp_dir = os.path.join(project_root, "temp_user_profile")
        
        # 임시 디렉토리 생성
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        # 전역 최대 시각
        self._global_max_ts = pd.NaT

        # A1: 유저별 최근 시각
        self._a1_user_last_ts = {}  # user_id -> pd.Timestamp (naive)

        # A2: 창별 가중합 (유저별로 3개 창을 동시 누적)
        self._a2_sum_w = {w: {} for w in SHORT_WINDOWS_HOURS}          # window -> {uid: float}
        self._a2_sum_feat = {w: {} for w in SHORT_WINDOWS_HOURS}       # window -> {uid: np.ndarray(33,)}

        # A3: 유저×카테고리 14일 카운트
        self._a3_cat_counts = {}    # uid -> np.ndarray(14,)
        
        # 참조 시각
        self._ref_time = None
    
    def process_chunks(self):
        """청크 단위로 데이터 처리 (2패스 방식)"""
        print("=== 배치 처리용 사용자 프로필 생성 시작 ===")
        
        # 1패스: 전역 최대 시각 및 사용자별 최근 시각 수집
        print("\n1패스: 전역 최대 시각 및 사용자별 최근 시각 수집...")
        self.pass1_collect_timestamps()
        
        # 참조 시각 결정
        print("\n참조 시각 결정...")
        self.determine_reference_time()
        
        # 2패스: A2/A3 계산 (참조 시각 기준)
        print("\n2패스: A2/A3 계산 (참조 시각 기준)...")
        self.pass2_calculate_a2_a3()
        
        # 1단계: f_ads_rwd_info.csv 청크 단위 처리 (기본 통계)
        print("\n1단계: f_ads_rwd_info.csv 청크 단위 처리...")
        self.process_f_ads_rwd_chunks()
        
        # 2단계: 사용자별 통계 집계
        print("\n2단계: 사용자별 통계 집계...")
        self.aggregate_user_stats()
        
        # 3단계: 광고 특성과 병합
        print("\n3단계: 광고 특성과 병합...")
        self.merge_ads_features()
        
        # 4단계: 최종 프로필 생성
        print("\n4단계: 최종 프로필 생성...")
        self.create_final_profiles()
        
        # 5단계: 파일 정리
        print("\n5단계: 파일 정리...")
        self.cleanup_temp_files()
        
        print("\n=== 배치 처리 완료 ===")
    
    def pass1_collect_timestamps(self):
        """1패스: 전역 최대 시각 및 사용자별 최근 시각 수집"""
        print("1패스: 시간 정보 수집 중...")
        
        # 원본 데이터 파일 사용
        data_file = "f_ads_rwd_info.csv"
        print(f"사용할 데이터 파일: {data_file}")
        
        chunk_count = 0
        for chunk in tqdm(pd.read_csv(
            os.path.join(self.data_path, data_file), 
            chunksize=self.chunk_size * 2
        ), desc="1패스: 시간 정보 수집"):
            
            # user_device_id 생성
            chunk['user_device_id'] = np.where(
                chunk['dvc_idx'] != 0, 
                chunk['dvc_idx'].astype(str), 
                chunk['user_ip']
            )
            
            # 시간 표준화 (naive)
            if "click_dt" in chunk.columns:
                chunk["event_ts"] = parse_local_ts(chunk["click_dt"])
            elif "click_date" in chunk.columns:
                chunk["event_ts"] = parse_local_ts(chunk["click_date"])
            elif "ts" in chunk.columns:
                chunk["event_ts"] = parse_local_ts(chunk["ts"])
            else:
                chunk["event_ts"] = pd.NaT
            
            # 전역 최대 시각 갱신
            chunk_max = chunk["event_ts"].max()
            if pd.notna(chunk_max):
                if pd.isna(self._global_max_ts) or (chunk_max > self._global_max_ts):
                    self._global_max_ts = chunk_max
            
            # 유저별 최근 시각 갱신
            grp_last = chunk.groupby("user_device_id", observed=True)["event_ts"].max()
            for uid, ts in grp_last.items():
                if pd.isna(ts): 
                    continue
                prev = self._a1_user_last_ts.get(uid)
                if (prev is None) or (ts > prev):
                    self._a1_user_last_ts[uid] = ts
            
            chunk_count += 1
        
        print(f"1패스 완료: {chunk_count}개 청크 처리")
        print(f"전역 최대 시각: {self._global_max_ts}")
        print(f"사용자별 최근 시각 수집: {len(self._a1_user_last_ts):,}명")
    
    def determine_reference_time(self):
        """참조 시각 결정"""
        now_local = pd.Timestamp.now()
        
        if REF_TIME_MODE == "data_max":
            self._ref_time = self._global_max_ts
        elif REF_TIME_MODE == "now":
            self._ref_time = now_local
        else:  # "auto"
            if pd.isna(self._global_max_ts):
                self._ref_time = now_local
            else:
                stale = (now_local - self._global_max_ts).days if pd.notna(self._global_max_ts) else 9999
                self._ref_time = self._global_max_ts if stale > STALE_THRESHOLD_DAYS else now_local
        
        print(f"참조 시각 결정: {self._ref_time}")
        print(f"모드: {REF_TIME_MODE}")
    
    def pass2_calculate_a2_a3(self):
        """2패스: A2/A3 계산 (참조 시각 기준)"""
        print("2패스: A2/A3 계산 중...")
        
        # ads_profile 서브셋 준비
        ads_profile_path = os.path.join(self.data_path, "ads_profile.csv")
        ads_profile_subset = None
        if os.path.exists(ads_profile_path):
            ads_profile = pd.read_csv(ads_profile_path)
            available_keys = [k for k in ALL_CONTENT_KEYS if k in ads_profile.columns]
            missing_keys = [k for k in ALL_CONTENT_KEYS if k not in ads_profile.columns]
            if missing_keys:
                print(f"누락된 특성 컬럼들: {missing_keys}")
            ads_profile_subset = ads_profile[['ads_idx','ads_category'] + available_keys].copy()
            for key in missing_keys:
                ads_profile_subset[key] = 0.0
            print(f"ads_profile 서브셋 로드 완료: {len(ads_profile_subset)}개 광고, {len(available_keys)}개 특성")
        
        # 원본 데이터 파일 사용
        data_file = "f_ads_rwd_info.csv"
        print(f"2패스 데이터 파일: {data_file}")
        
        chunk_count = 0
        for chunk in tqdm(pd.read_csv(
            os.path.join(self.data_path, data_file), 
            chunksize=self.chunk_size * 2
        ), desc="2패스: A2/A3 계산"):
            
            # user_device_id 생성
            chunk['user_device_id'] = np.where(
                chunk['dvc_idx'] != 0, 
                chunk['dvc_idx'].astype(str), 
                chunk['user_ip']
            )
            
            # 시간 표준화 (naive)
            if "click_dt" in chunk.columns:
                chunk["event_ts"] = parse_local_ts(chunk["click_dt"])
            elif "click_date" in chunk.columns:
                chunk["event_ts"] = parse_local_ts(chunk["click_date"])
            elif "ts" in chunk.columns:
                chunk["event_ts"] = parse_local_ts(chunk["ts"])
            else:
                chunk["event_ts"] = pd.NaT
            
            # 이벤트 가중치 계산
            ck  = pd.to_numeric(chunk.get("click_key", 0), errors="coerce").fillna(0)
            cki = pd.to_numeric(chunk.get("click_key_info", 0), errors="coerce").fillna(0)
            ckr = pd.to_numeric(chunk.get("click_key_rwd", 0), errors="coerce").fillna(0)
            chunk["event_w"] = np.where((ckr > 0) | (ck > 0), W_CLICK_RWD, np.where(cki > 0, W_CLICK_INFO, 0.0)).astype("float32")
            
            # ads_profile 서브셋 병합
            if ads_profile_subset is not None:
                chunk = chunk.merge(ads_profile_subset, on='ads_idx', how='left')
            
            # A2: 창별(72h/168h/336h) 가중합 동시 누적
            for window_h in SHORT_WINDOWS_HOURS:
                cutoff = self._ref_time - pd.Timedelta(hours=window_h)
                short = chunk[chunk["event_ts"] >= cutoff].copy()
                if short.empty:
                    continue
                
                # 시간 가중 (ref_time 기준)
                dt_hours = (self._ref_time - short["event_ts"]).dt.total_seconds() / 3600.0
                w_time = np.exp(- dt_hours.astype("float32") / float(TIME_DECAY_HOURS)).astype("float32")
                w_final = (short["event_w"].astype("float32") * w_time).astype("float32")
                short["__w_final__"] = w_final

                # 유저별 가중치 합
                wsum = short.groupby("user_device_id", observed=True)["__w_final__"].sum()
                for uid, w in wsum.items():
                    if w <= 0:
                        continue
                    sub = short[short["user_device_id"] == uid]
                    # ALL_CONTENT_KEYS가 있는 컬럼만 처리
                    available_keys = [k for k in ALL_CONTENT_KEYS if k in sub.columns]
                    if not available_keys:
                        continue
                    mat = sub[available_keys].astype("float32").to_numpy(copy=False)
                    wf  = sub["__w_final__"].to_numpy(dtype="float32", copy=False).reshape(-1, 1)
                    vec = np.nansum(mat * wf, axis=0).astype("float32")
                    
                    # 33개 특성에 맞춰 패딩
                    full_vec = np.zeros(33, dtype="float32")
                    for i, key in enumerate(ALL_CONTENT_KEYS):
                        if key in available_keys:
                            key_idx = available_keys.index(key)
                            full_vec[i] = vec[key_idx]

                    self._a2_sum_w[window_h][uid] = float(self._a2_sum_w[window_h].get(uid, 0.0) + float(w))
                    prev = self._a2_sum_feat[window_h].get(uid)
                    self._a2_sum_feat[window_h][uid] = full_vec if prev is None else (prev + full_vec).astype("float32")
            
            # A3: ref_time 기준 최근 14일 카테고리 카운트
            cutoff_exp = self._ref_time - pd.Timedelta(days=NOVELTY_WINDOW_DAYS)
            exp = chunk[chunk["event_ts"] >= cutoff_exp][["user_device_id","ads_category"]].dropna()
            if not exp.empty:
                cnt = exp.groupby(["user_device_id","ads_category"], observed=True).size()
                for (uid, cat), v in cnt.items():
                    if pd.isna(cat):
                        continue
                    c = int(cat)
                    if (c < 0) or (c >= len(CATEGORY_IDS)):
                        continue
                    vec = self._a3_cat_counts.get(uid)
                    if vec is None:
                        vec = np.zeros(len(CATEGORY_IDS), dtype="float32")
                    vec[c] += float(v)
                    self._a3_cat_counts[uid] = vec
            
            chunk_count += 1
        
        print(f"2패스 완료: {chunk_count}개 청크 처리")
        print(f"A2 창별 데이터: {[len(self._a2_sum_w[w]) for w in SHORT_WINDOWS_HOURS]}")
        print(f"A3 카테고리 데이터: {len(self._a3_cat_counts)}명")
    
    def process_f_ads_rwd_chunks(self):
        """f_ads_rwd_info.csv를 청크 단위로 처리 (기본 통계만)"""
        chunk_num = 0
        user_chunks = []
        
        # 원본 데이터 파일 사용
        data_file = "f_ads_rwd_info.csv"
        print(f"사용할 데이터 파일: {data_file}")
        
        for chunk in tqdm(pd.read_csv(
            os.path.join(self.data_path, data_file), 
            chunksize=self.chunk_size * 2
        ), desc=f"{data_file} 처리"):
            
            # user_device_id 생성 (벡터화)
            chunk['user_device_id'] = np.where(
                chunk['dvc_idx'] != 0, 
                chunk['dvc_idx'].astype(str), 
                chunk['user_ip']
            )
            
            # 기존 가중치 계산 (벡터화) - 수정됨
            chunk['weight'] = np.where(
                chunk['click_key_rwd'].notna(),  # 리워드 받은 클릭
                1.5,  # 리워드 받은 클릭 (1.5배 가중치)
                np.where(
                    chunk['click_key'].notna(),  # 클릭만 (click_key)
                    1.5,  # 클릭만 (1.5배 가중치)
                    np.where(
                        chunk['click_key_info'].notna(),  # 정보만 클릭
                        1.0,  # 정보만 클릭 (1.0배 가중치)
                        1.0   # 기본 가중치
                    )
                )
            )
            
            # click_date에서 시간 추출 (시간 패턴 분석용)
            chunk['click_datetime'] = pd.to_datetime(chunk['click_date'], errors='coerce')
            chunk['click_hour'] = chunk['click_datetime'].dt.hour
            
            # 사용자별 기본 통계 계산 (최적화 + 추가 정보)
            user_stats_chunk = chunk.groupby('user_device_id').agg({
                'ads_idx': ['count', 'nunique'],  # 총 상호작용 수, 고유 광고 수
                'reward_point': ['sum', 'mean', 'max'],  # 리워드 포인트 총합, 평균, 최대
                'show_price': ['mean', 'max'],  # 광고 표시 가격 평균, 최대
                'adv_price': ['mean', 'max'],  # 광고주 가격 평균, 최대
                'rwd_price': ['mean', 'max'],  # 리워드 가격 평균, 최대
                'ctit': 'mean',
                'click_hour': 'mean',  # 평균 클릭 시간 (시간 패턴 분석용)
                'weight': 'sum'
            }).reset_index()
            
            # 컬럼명 정리
            user_stats_chunk.columns = [
                'user_device_id', 'total_interactions', 'unique_ads', 
                'total_reward_points', 'avg_reward_points', 'max_reward_points',
                'avg_show_price', 'max_show_price',
                'avg_adv_price', 'max_adv_price', 
                'avg_rwd_price', 'max_rwd_price',
                'avg_dwell_time', 'avg_click_hour', 'total_weight'
            ]
            
            # 데이터 타입 최적화
            user_stats_chunk['total_weight'] = user_stats_chunk['total_weight'].astype(np.float32)
            user_stats_chunk['avg_dwell_time'] = user_stats_chunk['avg_dwell_time'].astype(np.float32)
            user_stats_chunk['avg_click_hour'] = user_stats_chunk['avg_click_hour'].astype(np.float32)
            
            # 청크 저장
            chunk_file = os.path.join(self.temp_dir, f"user_stats_chunk_{chunk_num}.pkl")
            with open(chunk_file, 'wb') as f:
                pickle.dump(user_stats_chunk, f)
            
            user_chunks.append(chunk_file)
            chunk_num += 1
        
        # 청크 파일 목록 저장
        with open(os.path.join(self.temp_dir, "chunk_files.pkl"), 'wb') as f:
            pickle.dump(user_chunks, f)
        
        print(f"총 {chunk_num}개 청크 처리 완료")
    
    def aggregate_user_stats(self):
        """사용자별 통계 집계 (최적화 + 시간 패턴 분석)"""
        # 청크 파일 목록 로드
        with open(os.path.join(self.temp_dir, "chunk_files.pkl"), 'rb') as f:
            chunk_files = pickle.load(f)
        
        # 모든 청크를 하나로 합치기 (메모리 효율적)
        all_user_stats = []
        
        for chunk_file in tqdm(chunk_files, desc="청크 통합"):
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
                all_user_stats.append(chunk_data)
        
        # 통합된 데이터로 최종 집계 (벡터화)
        print("데이터 통합 중...")
        combined_df = pd.concat(all_user_stats, ignore_index=True)
        
        # 사용자별 최종 통계 (최적화)
        print("사용자별 통계 계산 중...")
        final_user_stats = combined_df.groupby('user_device_id').agg({
            'total_interactions': 'sum',  # 총 상호작용 수
            'unique_ads': 'sum',  # 총 고유 광고 수
            'total_reward_points': 'sum',  # 총 리워드 포인트 (실제 받은 금액)
            'avg_reward_points': 'mean',  # 평균 리워드 포인트 (리워드 민감도 계산용)
            'max_reward_points': 'max',  # 최대 리워드 포인트
            'avg_show_price': 'mean',  # 평균 광고 표시 가격
            'max_show_price': 'max',  # 최대 광고 표시 가격
            'avg_adv_price': 'mean',  # 평균 광고주 가격
            'max_adv_price': 'max',  # 최대 광고주 가격
            'avg_rwd_price': 'mean',  # 평균 리워드 가격
            'max_rwd_price': 'max',  # 최대 리워드 가격
            'avg_dwell_time': 'mean',  # 평균 체류 시간
            'avg_click_hour': 'mean',  # 평균 클릭 시간 (시간 패턴 분석용)
            'total_weight': 'sum'  # 총 가중치
        }).reset_index()
        
        # 광고 다양성 계산
        final_user_stats['ad_diversity'] = (
            final_user_stats['unique_ads'] / final_user_stats['total_interactions']
        ).fillna(0)
        
        # 추가 계산 (벡터화)
        final_user_stats['reward_rate'] = (
            final_user_stats['total_reward_points'] / final_user_stats['total_interactions']
        ).fillna(0)
        
        # 시간 패턴 분석 추가 (click_date에서 추출한 시간 기준)
        print("시간 패턴 분석 중...")
        # 시간대별 활동 패턴 (click_date에서 추출한 실제 클릭 시간 기준)
        # Morning: 06:00–12:00, Afternoon: 12:00–18:00, Evening: 18:00–21:00, Night: 21:00–06:00
        final_user_stats['time_pattern_segment'] = pd.cut(
            final_user_stats['avg_click_hour'],
            bins=[0, 6, 12, 18, 21, 24],
            labels=['night', 'morning', 'afternoon', 'evening', 'night'],
            ordered=False
        )
        
        # 세그먼트 생성 (최적화)
        final_user_stats['activity_level'] = pd.cut(
            final_user_stats['total_interactions'],
            bins=[0, 5, 20, 100, float('inf')],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # 리워드 민감도 개선: 평균 리워드 금액 기준으로 계산
        # 총 누계가 아닌 평균 리워드 금액으로 민감도 측정
        final_user_stats['reward_sensitivity'] = pd.cut(
            final_user_stats['avg_reward_points'],
            bins=[0, 50, 150, 300, float('inf')],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # 추가: 광고 가격 민감도 (평균 광고 가격 기준)
        final_user_stats['price_sensitivity'] = pd.cut(
            final_user_stats['avg_show_price'],
            bins=[0, 200, 500, 1000, float('inf')],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # 데이터 타입 최적화
        price_cols = ['avg_reward_points', 'max_reward_points', 'avg_show_price', 'max_show_price', 
                     'avg_adv_price', 'max_adv_price', 'avg_rwd_price', 'max_rwd_price']
        for col in ['total_interactions', 'unique_ads', 'total_reward_points', 'avg_dwell_time', 
                   'avg_click_hour', 'total_weight', 'reward_rate', 'ad_diversity'] + price_cols:
            if col in final_user_stats.columns:
                final_user_stats[col] = final_user_stats[col].astype(np.float32)
        
        # 저장
        with open(os.path.join(self.temp_dir, "final_user_stats.pkl"), 'wb') as f:
            pickle.dump(final_user_stats, f)
        
        print(f"사용자별 통계 집계 완료: {len(final_user_stats):,}명")
    
    def merge_ads_features(self):
        """광고 특성과 병합 (실제 사용자 상호작용 기반 가중치 적용)"""
        print("실제 사용자 상호작용 기반 선호도 계산...")
        
        # 사용자 통계에서 user_device_id 목록 가져오기
        with open(os.path.join(self.temp_dir, "final_user_stats.pkl"), 'rb') as f:
            user_stats = pickle.load(f)
        
        user_ids = user_stats['user_device_id'].tolist()
        print(f"처리할 사용자 수: {len(user_ids):,}명")
        
        # 광고 특성 데이터 로드
        print("광고 특성 데이터 로드 중...")
        json_total = pd.read_csv(os.path.join(self.data_path, "json_total.csv"))
        ads_info = pd.read_csv(os.path.join(self.data_path, "preprocessed_ads_list.csv"))
        
        # motivation_status_display를 m_status로 매핑
        if 'motivation_status_display' in json_total.columns:
            json_total = json_total.copy()
            json_total['m_status'] = json_total['motivation_status_display']
            print("motivation_status_display를 m_status로 매핑 완료")
        
        # 특성 컬럼 선택 (m_*, e_*, p_*, b_*, c_*) - e_session 제외 (문자열, 별도 처리)
        feature_cols = [col for col in json_total.columns 
                       if col.startswith(('m_', 'e_', 'p_', 'b_', 'c_')) and col != 'ads_idx' and col != 'e_session']
        
        # e_session은 별도로 처리 (문자열 특성)
        session_col = 'e_session' if 'e_session' in json_total.columns else None
        print(f"선택된 특성: {len(feature_cols)}개")
        
        # 광고 정보와 특성 병합 (e_session 포함)
        merge_cols = ['ads_idx'] + feature_cols
        if session_col:
            merge_cols.append(session_col)
        
        ads_with_features = ads_info.merge(json_total[merge_cols], on='ads_idx', how='left')
        
        # f_ads_rwd_info.csv를 청크 단위로 다시 읽어서 사용자별 선호도 계산
        print("사용자별 실제 상호작용 기반 선호도 계산 중...")
        
        # 사용자별 선호도 초기화
        user_preferences = {}
        for user_id in user_ids:
            user_preferences[user_id] = {
                'total_weight': 0.0,
                'weighted_features': {col: 0.0 for col in feature_cols},
                'category_weights': {}
            }
        
        # 청크 단위로 상호작용 데이터 처리
        data_file = "f_ads_rwd_info.csv"
        print(f"선호도 계산용 데이터 파일: {data_file}")
        
        chunk_count = 0
        for chunk in tqdm(pd.read_csv(
            os.path.join(self.data_path, data_file), 
            chunksize=self.chunk_size * 2
        ), desc="사용자별 선호도 계산"):
            
            # user_device_id 생성
            chunk['user_device_id'] = np.where(
                chunk['dvc_idx'] != 0, 
                chunk['dvc_idx'].astype(str), 
                chunk['user_ip']
            )
            
            # 가중치 계산
            chunk['weight'] = np.where(
                chunk['click_key_rwd'].notna(),  # 리워드 받은 클릭
                1.5,  # 리워드 받은 클릭 (1.5배 가중치)
                np.where(
                    chunk['click_key'].notna(),  # 클릭만 (click_key)
                    1.5,  # 클릭만 (1.5배 가중치)
                    np.where(
                        chunk['click_key_info'].notna(),  # 정보만 클릭
                        1.0,  # 정보만 클릭 (1.0배 가중치)
                        1.0   # 기본 가중치
                    )
                )
            )
            
            # 광고 특성과 병합
            chunk_with_features = chunk.merge(ads_with_features, on='ads_idx', how='left')
            
            # 특성 컬럼 데이터 타입 변환 (안전하게)
            for col in feature_cols:
                if col in chunk_with_features.columns:
                    # 문자열 특성은 제외하고 숫자형만 변환
                    if col != 'e_session':  # e_session은 문자열 특성
                        chunk_with_features[col] = pd.to_numeric(chunk_with_features[col], errors='coerce')
            
            # 사용자별 가중치 적용된 특성 계산
            for _, row in chunk_with_features.iterrows():
                user_id = row['user_device_id']
                weight = row['weight']
                
                if user_id in user_preferences:
                    user_preferences[user_id]['total_weight'] += weight
                    
                    # 특성별 가중합 계산
                    for col in feature_cols:
                        if pd.notna(row[col]) and pd.notna(weight):
                            try:
                                feature_value = float(row[col])
                                user_preferences[user_id]['weighted_features'][col] += feature_value * weight
                            except (ValueError, TypeError):
                                # 변환 실패 시 무시
                                continue
                    
                    # e_session 별도 처리 (문자열 특성)
                    if session_col and session_col in row and pd.notna(row[session_col]):
                        session_value = row[session_col]
                        if 'e_session' not in user_preferences[user_id]['category_weights']:
                            user_preferences[user_id]['category_weights']['e_session'] = {}
                        if session_value not in user_preferences[user_id]['category_weights']['e_session']:
                            user_preferences[user_id]['category_weights']['e_session'][session_value] = 0.0
                        user_preferences[user_id]['category_weights']['e_session'][session_value] += weight
                    
                    # 카테고리/타입별 가중합 계산
                    for col in ['ads_type', 'ads_category']:
                        if col in row and pd.notna(row[col]):
                            key = f'{col}_{row[col]}'
                            if key not in user_preferences[user_id]['category_weights']:
                                user_preferences[user_id]['category_weights'][key] = 0.0
                            user_preferences[user_id]['category_weights'][key] += weight
            
            chunk_count += 1
            if chunk_count % 10 == 0:
                print(f"  처리된 청크: {chunk_count}개")
        
        # 가중평균 계산 및 DataFrame 생성
        print("가중평균 계산 중...")
        all_preferences = {'user_device_id': user_ids}
        
        # 특성별 선호도 (가중평균)
        for col in feature_cols:
            pref_values = []
            for user_id in user_ids:
                if user_preferences[user_id]['total_weight'] > 0:
                    avg_value = user_preferences[user_id]['weighted_features'][col] / user_preferences[user_id]['total_weight']
                else:
                    # 상호작용이 없는 사용자는 전체 평균값 사용
                    avg_value = json_total[col].mean()
                pref_values.append(avg_value)
            all_preferences[f'pref_{col}'] = np.array(pref_values, dtype=np.float32)
        
        # 가중평균 후 percentile ranking 적용 (ads_profile과 일관성 유지)
        print("특성별 percentile ranking 적용 중...")
        for col in feature_cols:
            if f'pref_{col}' in all_preferences:
                # percentile ranking 적용
                pref_series = pd.Series(all_preferences[f'pref_{col}'])
                all_preferences[f'pref_{col}'] = pref_series.rank(pct=True).fillna(0).astype(np.float32)
        
        # e_session 별도 처리 (문자열 특성)
        if session_col:
            session_values = set()
            for user_id in user_ids:
                if 'e_session' in user_preferences[user_id]['category_weights']:
                    session_values.update(user_preferences[user_id]['category_weights']['e_session'].keys())
            
            # 가장 많이 상호작용한 session 값 선택
            for user_id in user_ids:
                if 'e_session' in user_preferences[user_id]['category_weights']:
                    session_weights = user_preferences[user_id]['category_weights']['e_session']
                    if session_weights:
                        most_common_session = max(session_weights, key=session_weights.get)
                        all_preferences.setdefault('e_session', []).append(most_common_session)
                    else:
                        all_preferences.setdefault('e_session', []).append('unknown')
                else:
                    all_preferences.setdefault('e_session', []).append('unknown')
        
        # 카테고리/타입별 선호도 (가중합)
        all_categories = set()
        for user_id in user_ids:
            all_categories.update(user_preferences[user_id]['category_weights'].keys())
        
        # e_session 제외하고 처리
        for category in all_categories:
            if category != 'e_session':  # e_session은 별도 처리
                pref_values = []
                for user_id in user_ids:
                    weight = user_preferences[user_id]['category_weights'].get(category, 0.0)
                    pref_values.append(weight)
                all_preferences[category] = np.array(pref_values, dtype=np.float32)
        
        # DataFrame 생성
        final_pref_df = pd.DataFrame(all_preferences)
        
        # 메모리 최적화
        for col in final_pref_df.columns:
            if col != 'user_device_id' and pd.api.types.is_numeric_dtype(final_pref_df[col]):
                final_pref_df[col] = final_pref_df[col].astype(np.float32)
        
        print(f"실제 상호작용 기반 선호도 생성 완료: {len(final_pref_df):,}명, {len(final_pref_df.columns)}개 컬럼")
        
        # 저장
        with open(os.path.join(self.temp_dir, "user_preferences.pkl"), 'wb') as f:
            pickle.dump(final_pref_df, f)
    
    def create_final_profiles(self):
        """최종 프로필 생성 (최적화)"""
        # 사용자 통계 로드
        with open(os.path.join(self.temp_dir, "final_user_stats.pkl"), 'rb') as f:
            user_stats = pickle.load(f)
        
        # 사용자 선호도 로드
        with open(os.path.join(self.temp_dir, "user_preferences.pkl"), 'rb') as f:
            user_preferences = pickle.load(f)
        
        # 병합 (최적화)
        print("최종 프로필 병합 중...")
        final_profiles = user_stats.merge(user_preferences, on='user_device_id', how='left')
        
        # 4-1) A1: last_interaction_ts, tau_recency (ref_time 기준)
        print("A1: 최근 상호작용 시각 및 recency 계산 중...")
        
        last_ts_s = pd.Series(self._a1_user_last_ts, name="last_interaction_ts")
        
        # ISO8601 문자열(타임존 없이)
        last_ts_str = last_ts_s.apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S") if pd.notna(x) else "")
        
        def _tau(ts: pd.Timestamp) -> float:
            if pd.isna(ts) or pd.isna(self._ref_time):
                return 0.0
            hours = (self._ref_time - ts).total_seconds() / 3600.0
            tau_raw = float(np.exp(- hours / float(RECENCY_HALF_WINDOW_HOURS)))
            return float(min(TAU_MAX, tau_raw))
        
        tau_s = last_ts_s.apply(_tau).astype("float32").rename("tau_recency")
        
        a1_df = pd.concat([last_ts_str, tau_s], axis=1)
        final_profiles = final_profiles.merge(a1_df, left_on="user_device_id", right_index=True, how="left")
        final_profiles["last_interaction_ts"] = final_profiles["last_interaction_ts"].fillna("")
        final_profiles["tau_recency"] = final_profiles["tau_recency"].fillna(0.0).astype("float32")
        
        # 4-2) A2: 33개 _st (적응형 창 선택)
        print("A2: 단기 선호도 계산 중...")
        
        # 창별로 u_short 후보 구성
        cand_frames = []
        for window_h in SHORT_WINDOWS_HOURS:
            wmap = self._a2_sum_w[window_h]
            fmap = self._a2_sum_feat[window_h]
            if not wmap:
                continue
            rows = []
            for uid, w in wmap.items():
                vec = fmap.get(uid)
                if (w > 0.0) and (vec is not None):
                    rows.append((uid, (vec / w)))
            if rows:
                dfw = pd.DataFrame({"user_device_id":[r[0] for r in rows]})
                for i,k in enumerate(ALL_CONTENT_KEYS):
                    dfw[k+"_st_"+str(window_h)] = [r[1][i] for r in rows]
                dfw.set_index("user_device_id", inplace=True)
                cand_frames.append((window_h, dfw))
        
        # 창 우선순위: 72h > 168h > 336h
        if cand_frames:
            # 유저별로 가장 짧은 창의 값을 선택
            base = pd.DataFrame(index=final_profiles["user_device_id"]).set_index("user_device_id")
            sel = base
            for window_h, dfw in cand_frames:
                sel = sel.join(dfw, how="left")

            # 최종 열 생성
            for k in ALL_CONTENT_KEYS:
                c72 = k+"_st_72"
                c168 = k+"_st_168"
                c336 = k+"_st_336"
                # 우선순위 선택
                final_profiles[k+"_st"] = (
                    sel.get(c72)
                    .fillna(sel.get(c168))
                    .fillna(sel.get(c336))
                )

            # 퍼센타일[0,1]
            for k in ALL_CONTENT_KEYS:
                col = k+"_st"
                final_profiles[col] = percentile_rank_0_1(pd.to_numeric(final_profiles[col], errors="coerce"))

        # 결측 → 장기값 복사
        for k in ALL_CONTENT_KEYS:
            st = k + "_st"
            if st not in final_profiles.columns:
                # 기존 컬럼이 있으면 복사, 없으면 0으로 채우기
                if k in final_profiles.columns:
                    final_profiles[st] = final_profiles[k].astype("float32")
                else:
                    final_profiles[st] = 0.0
            else:
                # 기존 컬럼이 있으면 fillna, 없으면 그대로 유지
                if k in final_profiles.columns:
                    final_profiles[st] = final_profiles[st].fillna(final_profiles[k]).astype("float32")
                else:
                    final_profiles[st] = final_profiles[st].fillna(0.0).astype("float32")
        
        # 4-3) A3: exp_cat_0..13 (라플라스 평활)
        print("A3: 카테고리 노출 비율 계산 중...")
        if len(self._a3_cat_counts) > 0:
            rows = [(uid, vec) for uid, vec in self._a3_cat_counts.items()]
            exp_df = pd.DataFrame(rows, columns=["user_device_id","cnts"]).set_index("user_device_id")

            def _shares(v: np.ndarray) -> np.ndarray:
                total = float(np.nansum(v))
                if not np.isfinite(total):
                    total = 0.0
                return ((v + (LAPLACE_MU / len(CATEGORY_IDS))) / (total + LAPLACE_MU)).astype("float32")

            exp_arr = np.stack([_shares(v) for v in exp_df["cnts"].to_numpy()], axis=0)
            for i,c in enumerate(CATEGORY_IDS):
                exp_df[f"exp_cat_{c}"] = exp_arr[:, i].astype("float32")
            exp_df.drop(columns=["cnts"], inplace=True)

            final_profiles = final_profiles.merge(exp_df, left_on="user_device_id", right_index=True, how="left")

        # 결측 → 균등 분포
        for c in CATEGORY_IDS:
            col = f"exp_cat_{c}"
            if col in final_profiles.columns:
                final_profiles[col] = final_profiles[col].fillna(1.0/len(CATEGORY_IDS)).astype("float32")
            else:
                final_profiles[col] = (1.0/len(CATEGORY_IDS))
        
        # 메모리 최적화
        print("메모리 최적화 중...")
        for col in final_profiles.columns:
            if col != 'user_device_id' and pd.api.types.is_numeric_dtype(final_profiles[col]):
                final_profiles[col] = final_profiles[col].astype(np.float32)
        
        # 저장 (최적화)
        output_path = os.path.join(self.data_path, "user_profile.csv")
        print("파일 저장 중...")
        final_profiles.to_csv(output_path, index=False)
        
        print(f"\n=== 최종 프로필 생성 완료 ===")
        print(f"파일 저장: {output_path}")
        print(f"총 {len(final_profiles):,} 명의 사용자 프로필")
        print(f"총 {len(final_profiles.columns)} 개의 특성")
        
        # 파일 크기
        file_size = os.path.getsize(output_path) / (1024**2)
        print(f"파일 크기: {file_size:.2f} MB")
        
        # 요약 통계
        print(f"\n평균 상호작용 수: {final_profiles['total_interactions'].mean():.2f}")
        print(f"평균 리워드 포인트: {final_profiles['total_reward_points'].mean():.2f} (총 누계)")
        print(f"평균 리워드 금액: {final_profiles['avg_reward_points'].mean():.2f} (광고당 평균)")
        print(f"평균 광고 가격: {final_profiles['avg_show_price'].mean():.2f}")
        print(f"활동 수준 분포:")
        print(final_profiles['activity_level'].value_counts())
        print(f"리워드 민감도 분포 (평균 리워드 금액 기준):")
        print(final_profiles['reward_sensitivity'].value_counts())
        print(f"가격 민감도 분포 (평균 광고 가격 기준):")
        print(final_profiles['price_sensitivity'].value_counts())
        
        # 카테고리 선호도 컬럼 수
        category_cols = [col for col in final_profiles.columns if col.startswith(('ads_type_', 'ads_category_'))]
        print(f"카테고리/타입 선호도 컬럼: {len(category_cols)}개")
        
        # A1/A2/A3 컬럼 수
        a1_cols = [col for col in final_profiles.columns if col in ['last_interaction_ts', 'tau_recency']]
        a2_cols = [col for col in final_profiles.columns if col.endswith('_st')]
        a3_cols = [col for col in final_profiles.columns if col.startswith('exp_cat_')]
        print(f"A1 컬럼 (최근 상호작용): {len(a1_cols)}개")
        print(f"A2 컬럼 (단기 선호도): {len(a2_cols)}개")
        print(f"A3 컬럼 (카테고리 노출): {len(a3_cols)}개")
        
        # A1 통계
        if 'tau_recency' in final_profiles.columns:
            print(f"평균 tau_recency: {final_profiles['tau_recency'].mean():.4f}")
            print(f"tau_recency 분포: 0={sum(final_profiles['tau_recency'] == 0):,}명, >0={sum(final_profiles['tau_recency'] > 0):,}명")
        
        # A2 통계 (단기 선호도가 있는 사용자)
        if a2_cols:
            st_users = final_profiles[a2_cols].notna().any(axis=1).sum()
            print(f"단기 선호도 데이터가 있는 사용자: {st_users:,}명")
        
        # A3 통계 (카테고리 노출 데이터가 있는 사용자)
        if a3_cols:
            exp_users = final_profiles[a3_cols].notna().any(axis=1).sum()
            print(f"카테고리 노출 데이터가 있는 사용자: {exp_users:,}명")
    
    def cleanup_temp_files(self):
        """파일 정리"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print("파일 정리 완료")

if __name__ == "__main__":
    creator = BatchUserProfileCreator()
    creator.process_chunks()
