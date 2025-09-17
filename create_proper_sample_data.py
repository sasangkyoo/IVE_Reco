#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
올바른 샘플 데이터 생성 스크립트
- 상호작용 데이터를 먼저 샘플링하고, 해당 광고들만 광고 데이터에서 추출
"""

import pandas as pd
import zipfile
import numpy as np

def create_proper_sample_data():
    """올바른 순서로 샘플 데이터 생성"""
    print("🚀 올바른 샘플 데이터 생성 시작...")
    
    # 1. 원본 상호작용 데이터에서 먼저 샘플링 (10,000개)
    print("📊 1단계: 상호작용 데이터 샘플링...")
    
    # 원본 상호작용 데이터 로드
    with zipfile.ZipFile("correct_interactions.zip", 'r') as zip_ref:
        with zip_ref.open("correct_interactions.csv") as f:
            interactions_df = pd.read_csv(f)
    
    print(f"📊 원본 상호작용: {len(interactions_df):,}개")
    
    # 10,000개 샘플링
    sample_interactions = interactions_df.sample(n=10000, random_state=42)
    print(f"📊 샘플 상호작용: {len(sample_interactions):,}개")
    
    # 2. 샘플 상호작용에 포함된 광고 인덱스들 추출
    sample_ads_indices = set(sample_interactions['ads_idx'].tolist())
    print(f"📊 샘플 상호작용에 포함된 광고: {len(sample_ads_indices)}개")
    
    # 3. 원본 광고 데이터에서 해당 광고들만 추출
    print("📊 2단계: 광고 데이터 필터링...")
    
    with zipfile.ZipFile("ads_profile.zip", 'r') as zip_ref:
        with zip_ref.open("ads_profile.csv") as f:
            ads_df = pd.read_csv(f)
    
    print(f"📊 원본 광고: {len(ads_df):,}개")
    
    # 샘플 상호작용에 포함된 광고들만 필터링
    sample_ads = ads_df[ads_df['ads_idx'].isin(sample_ads_indices)]
    print(f"📊 샘플 광고: {len(sample_ads):,}개")
    
    # 4. 샘플 상호작용에 포함된 사용자들 추출
    print("📊 3단계: 사용자 데이터 필터링...")
    
    sample_user_ids = set(sample_interactions['user_device_id'].tolist())
    print(f"📊 샘플 상호작용에 포함된 사용자: {len(sample_user_ids)}개")
    
    with zipfile.ZipFile("user_profile.zip", 'r') as zip_ref:
        with zip_ref.open("user_profile.csv") as f:
            users_df = pd.read_csv(f)
    
    print(f"📊 원본 사용자: {len(users_df):,}개")
    
    # 샘플 상호작용에 포함된 사용자들만 필터링
    sample_users = users_df[users_df['user_device_id'].isin(sample_user_ids)]
    print(f"📊 샘플 사용자: {len(sample_users):,}개")
    
    # 5. 샘플 데이터 저장
    print("📊 4단계: 샘플 데이터 저장...")
    
    # 광고 데이터 저장
    with zipfile.ZipFile("ads_profile_sample.zip", 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_ref:
        zip_ref.writestr("ads_profile.csv", sample_ads.to_csv(index=False))
    
    # 사용자 데이터 저장
    with zipfile.ZipFile("user_profile_sample.zip", 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_ref:
        zip_ref.writestr("user_profile.csv", sample_users.to_csv(index=False))
    
    # 상호작용 데이터 저장
    with zipfile.ZipFile("correct_interactions_sample.zip", 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_ref:
        zip_ref.writestr("correct_interactions.csv", sample_interactions.to_csv(index=False))
    
    print("✅ 모든 샘플 데이터 생성 완료!")
    print(f"📊 최종 결과:")
    print(f"   - 광고: {len(sample_ads):,}개")
    print(f"   - 사용자: {len(sample_users):,}개") 
    print(f"   - 상호작용: {len(sample_interactions):,}개")

if __name__ == "__main__":
    create_proper_sample_data()
