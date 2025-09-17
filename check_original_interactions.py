#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
원본 실제 상호작용 데이터 분석
"""

import pandas as pd

def check_original_interactions():
    """원본 실제 상호작용 데이터 분석"""
    print("=== 원본 실제 상호작용 데이터 분석 ===")
    
    # 원본 상호작용 데이터 로드
    df = pd.read_csv('input/save/correct_interactions.csv')
    
    print(f"총 상호작용 수: {len(df):,}개")
    print(f"고유한 사용자 수: {df['user_device_id'].nunique():,}명")
    print(f"고유한 광고 수: {df['ads_idx'].nunique():,}개")
    print(f"사용자당 평균 상호작용: {len(df) / df['user_device_id'].nunique():.1f}개")
    print(f"광고당 평균 상호작용: {len(df) / df['ads_idx'].nunique():.1f}개")
    
    print("\n=== 사용자별 상호작용 분포 ===")
    user_counts = df['user_device_id'].value_counts()
    print(f"최대 상호작용: {user_counts.max()}개")
    print(f"최소 상호작용: {user_counts.min()}개")
    print(f"중간값: {user_counts.median():.1f}개")
    print(f"표준편차: {user_counts.std():.1f}개")
    
    print("\n=== 상위 10명 사용자 ===")
    print(user_counts.head(10))
    
    print("\n=== 하위 10명 사용자 ===")
    print(user_counts.tail(10))
    
    print("\n=== 상호작용 타입별 분포 ===")
    type_counts = df['interaction_type'].value_counts()
    print(type_counts)
    
    print("\n=== 광고별 상호작용 분포 ===")
    ad_counts = df['ads_idx'].value_counts()
    print(f"최대 상호작용: {ad_counts.max()}개")
    print(f"최소 상호작용: {ad_counts.min()}개")
    print(f"중간값: {ad_counts.median():.1f}개")
    
    print("\n=== 상위 10개 광고 ===")
    print(ad_counts.head(10))
    
    print("\n=== 하위 10개 광고 ===")
    print(ad_counts.tail(10))

if __name__ == "__main__":
    check_original_interactions()
