# -*- coding: utf-8 -*-
"""
광고 필터링 모듈 (Advertisement Filtering Module)

광고의 재참여 타입, 시작일, 종료일을 기반으로 추천 가능한 광고를 필터링합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Set, Optional
import warnings
warnings.filterwarnings('ignore')

class AdsFilter:
    """광고 필터링 클래스"""
    
    def __init__(self):
        self.today = date.today()
        
    def filter_ads_by_date_and_rejoin(
        self, 
        ads_df: pd.DataFrame, 
        user_interactions: Optional[pd.DataFrame] = None,
        user_device_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        광고를 날짜와 재참여 타입으로 필터링합니다.
        
        Args:
            ads_df: 광고 프로필 데이터프레임
            user_interactions: 사용자 상호작용 데이터 (선택사항)
            user_device_id: 사용자 디바이스 ID (선택사항)
            
        Returns:
            필터링된 광고 데이터프레임
        """
        if ads_df.empty:
            return ads_df
            
        # 원본 데이터 복사
        filtered_df = ads_df.copy()
        
        # 1. 날짜 필터링
        filtered_df = self._filter_by_date(filtered_df)
        
        # 2. 재참여 타입 필터링
        if user_interactions is not None and user_device_id is not None:
            filtered_df = self._filter_by_rejoin_type(
                filtered_df, user_interactions, user_device_id
            )
        
        return filtered_df
    
    def _filter_by_date(self, ads_df: pd.DataFrame) -> pd.DataFrame:
        """
        광고 시작일과 종료일로 필터링합니다.
        
        필터링 규칙:
        - ads_sdate: 최근 2년 내에 시작된 광고만 (null이면 제외)
        - ads_edate: 오늘보다 이후이거나 null (무기한 광고)
        """
        filtered_df = ads_df.copy()
        
        # 날짜 컬럼을 datetime으로 변환
        if 'ads_sdate' in filtered_df.columns:
            filtered_df['ads_sdate'] = pd.to_datetime(filtered_df['ads_sdate'], errors='coerce')
            
        if 'ads_edate' in filtered_df.columns:
            filtered_df['ads_edate'] = pd.to_datetime(filtered_df['ads_edate'], errors='coerce')
        
        # 시작일 필터링: 최근 2년 내에 시작된 광고만 (null 제외)
        if 'ads_sdate' in filtered_df.columns:
            from datetime import timedelta
            two_years_ago = self.today - timedelta(days=730)  # 2년 전
            
            start_mask = (
                filtered_df['ads_sdate'].notna() &  # null이 아닌 것만
                (filtered_df['ads_sdate'].dt.date >= two_years_ago) &  # 2년 내 시작
                (filtered_df['ads_sdate'].dt.date <= self.today)  # 오늘보다 이전
            )
            filtered_df = filtered_df[start_mask]
        
        # 종료일 필터링: 오늘보다 이후이거나 null (무기한 광고)
        if 'ads_edate' in filtered_df.columns:
            end_mask = (
                filtered_df['ads_edate'].isna() |  # null이면 무기한 광고로 간주
                (filtered_df['ads_edate'].dt.date >= self.today)  # 오늘보다 이후
            )
            filtered_df = filtered_df[end_mask]
        
        return filtered_df
    
    def _filter_by_rejoin_type(
        self, 
        ads_df: pd.DataFrame, 
        user_interactions: pd.DataFrame, 
        user_device_id: str
    ) -> pd.DataFrame:
        """
        재참여 타입에 따라 광고를 필터링합니다.
        
        재참여 타입 규칙:
        - NONE: 재참여 불가 (사용자가 이미 상호작용한 광고는 제외)
        - ADS_CODE_DAILY_UPDATE: 매일 재참여 가능 (날짜별로 체크)
        - REJOINABLE: 계속 재참여 가능 (제한 없음)
        """
        if 'ads_rejoin_type' not in ads_df.columns:
            return ads_df
            
        # 사용자의 과거 상호작용 광고 목록
        user_ads = user_interactions[
            user_interactions['user_device_id'] == user_device_id
        ]['ads_idx'].unique()
        
        # 재참여 타입별 필터링
        rejoin_mask = pd.Series(True, index=ads_df.index)
        
        # NONE: 재참여 불가 - 사용자가 이미 상호작용한 광고 제외
        none_mask = ads_df['ads_rejoin_type'] == 'NONE'
        if none_mask.any():
            rejoin_mask[none_mask] = ~ads_df.loc[none_mask, 'ads_idx'].isin(user_ads)
        
        # ADS_CODE_DAILY_UPDATE: 매일 재참여 가능
        # (현재는 단순히 허용, 실제로는 날짜별 체크 필요)
        daily_mask = ads_df['ads_rejoin_type'] == 'ADS_CODE_DAILY_UPDATE'
        # 일단 모든 daily 광고 허용 (향후 날짜별 로직 추가 가능)
        
        # REJOINABLE: 계속 재참여 가능 - 모든 광고 허용
        rejoinable_mask = ads_df['ads_rejoin_type'] == 'REJOINABLE'
        # 모든 rejoinable 광고 허용
        
        return ads_df[rejoin_mask]
    
    def get_filter_stats(self, original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> Dict:
        """
        필터링 통계를 반환합니다.
        """
        total_ads = len(original_df)
        filtered_ads = len(filtered_df)
        filtered_ratio = (filtered_ads / total_ads * 100) if total_ads > 0 else 0
        
        stats = {
            'total_ads': total_ads,
            'filtered_ads': filtered_ads,
            'filtered_ratio': round(filtered_ratio, 2),
            'removed_ads': total_ads - filtered_ads
        }
        
        # 재참여 타입별 통계
        if 'ads_rejoin_type' in filtered_df.columns:
            rejoin_stats = filtered_df['ads_rejoin_type'].value_counts().to_dict()
            stats['rejoin_type_distribution'] = rejoin_stats
        
        return stats

def create_ads_filter() -> AdsFilter:
    """AdsFilter 인스턴스를 생성합니다."""
    return AdsFilter()

# 사용 예시 함수
def filter_ads_for_recommendation(
    ads_df: pd.DataFrame,
    user_interactions: Optional[pd.DataFrame] = None,
    user_device_id: Optional[str] = None
) -> tuple[pd.DataFrame, Dict]:
    """
    추천을 위한 광고 필터링을 수행합니다.
    
    Returns:
        tuple: (필터링된 광고 데이터프레임, 필터링 통계)
    """
    filter_instance = create_ads_filter()
    
    # 필터링 전 원본 데이터 저장
    original_df = ads_df.copy()
    
    # 필터링 수행
    filtered_df = filter_instance.filter_ads_by_date_and_rejoin(
        ads_df, user_interactions, user_device_id
    )
    
    # 통계 생성
    stats = filter_instance.get_filter_stats(original_df, filtered_df)
    
    return filtered_df, stats

if __name__ == "__main__":
    # 테스트 코드
    print("광고 필터링 모듈 테스트")
    
    # 샘플 데이터 생성
    sample_ads = pd.DataFrame({
        'ads_idx': [1, 2, 3, 4, 5],
        'ads_name': ['광고1', '광고2', '광고3', '광고4', '광고5'],
        'ads_rejoin_type': ['NONE', 'ADS_CODE_DAILY_UPDATE', 'REJOINABLE', 'NONE', 'REJOINABLE'],
        'ads_sdate': ['2020-01-01', '2020-01-01', '2020-01-01', '2020-01-01', '2020-01-01'],
        'ads_edate': [None, None, None, '2023-01-01', None]
    })
    
    # 필터링 테스트
    filtered_ads, stats = filter_ads_for_recommendation(sample_ads)
    
    print(f"원본 광고 수: {stats['total_ads']}")
    print(f"필터링된 광고 수: {stats['filtered_ads']}")
    print(f"필터링 비율: {stats['filtered_ratio']}%")
    print(f"제거된 광고 수: {stats['removed_ads']}")
