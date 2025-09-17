#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최적화된 광고 프로필 생성기
- 모든 광고 데이터 포함
- 불필요한 컬럼 제거
- 메모리 최적화
- 파일 크기 최소화
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class OptimizedAdsProfileCreator:
    def __init__(self, data_path: str = "preprocessed", sample_size: int = None):
        # 현재 스크립트 위치를 기준으로 프로젝트 루트 찾기
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        self.data_path = os.path.join(project_root, data_path)
        self.sample_size = sample_size
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터 로드 (상호작용 데이터 제외)"""
        print("데이터 로드 중...")
        
        # 광고 정보 로드
        ads_info_path = os.path.join(self.data_path, "preprocessed_ads_list.csv")
        ads_info = pd.read_csv(ads_info_path, nrows=self.sample_size)
        print(f"광고 정보 로드 완료: {len(ads_info):,}개")
        
        # 광고 세부 특성 로드 (json_total.csv)
        json_total_path = os.path.join(self.data_path, "json_total.csv")
        json_total = pd.read_csv(json_total_path, nrows=self.sample_size)
        print(f"광고 세부 특성 로드 완료: {len(json_total):,}개")
        
        return ads_info, json_total
    
    
    def calculate_scores(self, ads_info: pd.DataFrame) -> pd.DataFrame:
        """점수 계산 (광고 자체 특성만, 상호작용 데이터 완전 제외)"""
        print("\n점수 계산 중...")
        
        # 수익성 점수 (show_price - rwd_price 기반, 광고주 비용 - 사용자 혜택)
        profitability = ads_info['show_price'] - ads_info['rwd_price']
        ads_info['profitability_score'] = profitability.rank(pct=True).fillna(0)
        
        # 리워드 가격 점수 (rwd_price 기반)
        ads_info['reward_price_score'] = ads_info['rwd_price'].rank(pct=True).fillna(0)
        
        # 광고 가격 점수 (show_price 기반)
        ads_info['ad_price_score'] = ads_info['show_price'].rank(pct=True).fillna(0)
        
        # 노출 순위 점수 (ads_ranking 기반, 높을수록 좋으므로 역순 정규화)
        ads_info['ranking_score'] = (1 - ads_info['ads_ranking'].rank(pct=True)).fillna(0)
        
        # M/E/P/B/C 특성들도 percentile ranking으로 정규화 (user_profile과 일관성 유지)
        feature_groups = ['m_', 'e_', 'p_', 'b_', 'c_']
        for group in feature_groups:
            group_cols = [col for col in ads_info.columns if col.startswith(group) and col != 'e_session']
            for col in group_cols:
                if col in ads_info.columns and pd.api.types.is_numeric_dtype(ads_info[col]):
                    ads_info[f'{col}_score'] = ads_info[col].rank(pct=True).fillna(0)
        
        return ads_info
    
    def create_segments(self, ads_info: pd.DataFrame) -> pd.DataFrame:
        """세그먼트 생성 (광고 자체 특성만)"""
        print("\n세그먼트 생성 중...")
        
        # 수익성 세그먼트
        ads_info['profitability_segment'] = pd.cut(
            ads_info['profitability_score'],
            bins=[0, 0.33, 0.67, 1.0],
            labels=['low', 'medium', 'high']
        )
        
        # 리워드 가격 세그먼트
        ads_info['reward_price_segment'] = pd.cut(
            ads_info['reward_price_score'],
            bins=[0, 0.33, 0.67, 1.0],
            labels=['low', 'medium', 'high']
        )
        
        
        
        
        return ads_info
    
    def optimize_features(self, ads_info: pd.DataFrame, json_total: pd.DataFrame) -> pd.DataFrame:
        """광고 특성 최적화 (json_total.csv 포함)"""
        print("\n광고 특성 최적화 중...")
        
        # ads_info에서 중요한 컬럼만 선택 (ads_code 포함)
        important_cols = [
            'ads_idx', 'ads_code', 'ads_type', 'ads_category', 'ads_name', 
            'ads_age_min', 'ads_age_max', 'ads_os_type', 'show_price', 'rwd_price', 'ads_ranking'
        ]
        
        # 존재하는 컬럼만 선택
        available_important_cols = [col for col in important_cols if col in ads_info.columns]
        
        # json_total에서 특성 컬럼들 (m_, e_, p_, b_, c_) - e_session 제외 (문자열, 별도 처리)
        feature_cols = [col for col in json_total.columns 
                       if col.startswith(('m_', 'e_', 'p_', 'b_', 'c_')) and col != 'e_session']
        
        # e_session은 별도로 처리 (문자열 특성)
        session_col = 'e_session' if 'e_session' in json_total.columns else None
        
        # 낮은 분산 특성 제거
        low_variance_cols = []
        for col in feature_cols:
            if col in json_total.columns:
                if pd.api.types.is_numeric_dtype(json_total[col]):
                    if json_total[col].var() < 0.001:
                        low_variance_cols.append(col)
                else:
                    if json_total[col].nunique() <= 1:
                        low_variance_cols.append(col)
        
        # 최종 특성 컬럼 선택
        final_feature_cols = [col for col in feature_cols if col not in low_variance_cols]
        
        # motivation_status_display를 m_status로 매핑
        if 'motivation_status_display' in json_total.columns:
            json_total = json_total.copy()
            json_total['m_status'] = json_total['motivation_status_display']
            if 'm_status' not in final_feature_cols:
                final_feature_cols.append('m_status')
            print("motivation_status_display를 m_status로 매핑 완료")
        
        # ads_info와 json_total 병합 (e_session 포함, ads_code는 ads_info에서 가져옴)
        merge_cols = ['ads_idx'] + final_feature_cols
        if session_col:
            merge_cols.append(session_col)
        
        ads_with_features = ads_info[available_important_cols].merge(
            json_total[merge_cols], 
            on='ads_idx', 
            how='left'
        )
        
        print(f"특성 최적화 완료: {len(ads_with_features.columns)}개 컬럼")
        print(f"  - 기본 정보: {len(available_important_cols)}개")
        print(f"  - 세부 특성: {len(final_feature_cols)}개")
        if low_variance_cols:
            print(f"제거된 낮은 분산 특성: {len(low_variance_cols)}개")
        
        return ads_with_features
    
    def finalize_profiles(self, ads_info: pd.DataFrame) -> pd.DataFrame:
        """데이터 최종화 (상호작용 데이터 완전 제외)"""
        print("\n데이터 최종화 중...")
        
        # 세그먼트는 결측값을 'low'로 채우기 (모든 광고가 세그먼트를 가져야 함)
        segment_columns = ['profitability_segment', 'reward_price_segment']
        for col in segment_columns:
            if col in ads_info.columns:
                # Categorical 타입인 경우 categories에 'low' 추가 (이미 있으면 추가하지 않음)
                if pd.api.types.is_categorical_dtype(ads_info[col]):
                    if 'low' not in ads_info[col].cat.categories:
                        ads_info[col] = ads_info[col].cat.add_categories(['low'])
                ads_info[col] = ads_info[col].fillna('low')
        
        # 데이터 타입 최적화
        ads_info['ads_idx'] = ads_info['ads_idx'].astype('int32')
        
        numeric_cols = ads_info.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'ads_idx':
                ads_info[col] = ads_info[col].astype('float32')
        
        return ads_info
    
    def save_profiles(self, profiles: pd.DataFrame) -> None:
        """프로필 저장"""
        output_path = os.path.join(self.data_path, "ads_profile.csv")
        profiles.to_csv(output_path, index=False)
        
        print(f"\n광고 프로필 저장 완료: {output_path}")
        print(f"총 {len(profiles):,} 개의 광고 프로필이 생성되었습니다.")
        print(f"총 {len(profiles.columns)} 개의 특성이 포함되었습니다.")
        
        # 파일 크기 확인
        file_size = os.path.getsize(output_path) / (1024**2)
        print(f"파일 크기: {file_size:.2f} MB")
        
        # 요약 통계
        print("\n=== 광고 프로필 요약 통계 ===")
        print(f"총 광고 수: {len(profiles):,}")
        print(f"평균 수익성 점수: {profiles['profitability_score'].mean():.3f}")
        print(f"평균 리워드 가격 점수: {profiles['reward_price_score'].mean():.3f}")
        print(f"평균 광고 가격 점수: {profiles['ad_price_score'].mean():.3f}")
        print(f"평균 노출 순위 점수: {profiles['ranking_score'].mean():.3f}")
        
        # 세그먼트 분포
        print("\n=== 수익성 세그먼트 분포 ===")
        print(profiles['profitability_segment'].value_counts())
        
        print("\n=== 리워드 가격 세그먼트 분포 ===")
        print(profiles['reward_price_segment'].value_counts())
    
    def run(self) -> None:
        """메인 실행 함수"""
        print("=== 최적화된 광고 프로필 생성 시작 ===")
        
        # 데이터 로드 (상호작용 데이터 제외)
        ads_info, json_total = self.load_data()
        
        # 광고 특성 최적화 (json_total.csv 포함)
        ads_info_optimized = self.optimize_features(ads_info, json_total)
        
        # 점수 계산 (광고 자체 특성만)
        ads_info_with_scores = self.calculate_scores(ads_info_optimized)
        
        # 세그먼트 생성 (광고 자체 특성만)
        ads_info_with_segments = self.create_segments(ads_info_with_scores)
        
        # 데이터 최종화 (상호작용 데이터 완전 제외)
        final_profiles = self.finalize_profiles(ads_info_with_segments)
        
        # 프로필 저장
        self.save_profiles(final_profiles)
        
        print("\n=== 광고 프로필 생성 완료 ===")

if __name__ == "__main__":
    creator = OptimizedAdsProfileCreator()
    creator.run()
