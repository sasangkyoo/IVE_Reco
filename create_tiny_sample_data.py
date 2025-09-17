#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
매우 작은 샘플 데이터 생성 스크립트
- Streamlit Cloud에서 안정적으로 작동하도록 최소한의 데이터만 생성
"""

import pandas as pd
import zipfile
import numpy as np

def create_tiny_sample_data():
    """매우 작은 샘플 데이터 생성"""
    print("🚀 매우 작은 샘플 데이터 생성 시작...")
    
    # 1. 광고 데이터 (100개)
    print("📊 광고 데이터 생성 중...")
    
    # 가상의 광고 데이터 생성
    np.random.seed(42)
    
    ads_data = []
    for i in range(100):
        ad = {
            'ads_idx': i + 1,
            'ads_code': f'AD{i+1:03d}',
            'ads_name': f'광고 {i+1}',
            'ads_type': np.random.randint(1, 5),
            'ads_category': np.random.randint(1, 6),
            # 피처 컬럼들 (간단하게)
            'm_click_rate': np.random.uniform(0.01, 0.1),
            'm_conversion_rate': np.random.uniform(0.001, 0.01),
            'e_engagement': np.random.uniform(0.1, 0.9),
            'p_price': np.random.uniform(1000, 100000),
            'b_brand_score': np.random.uniform(0.1, 1.0),
            'c_category_score': np.random.uniform(0.1, 1.0)
        }
        ads_data.append(ad)
    
    ads_df = pd.DataFrame(ads_data)
    
    # 2. 사용자 데이터 (50명)
    print("👥 사용자 데이터 생성 중...")
    
    users_data = []
    for i in range(50):
        user = {
            'user_device_id': f'user_{i+1:03d}',
            'total_interactions': np.random.randint(5, 20),
            'unique_ads': np.random.randint(3, 15),
            'total_reward_points': np.random.uniform(100, 2000),
            # 피처 컬럼들
            'm_avg_click_rate': np.random.uniform(0.01, 0.1),
            'm_avg_conversion_rate': np.random.uniform(0.001, 0.01),
            'e_avg_engagement': np.random.uniform(0.1, 0.9),
            'p_avg_price_preference': np.random.uniform(1000, 100000),
            'b_brand_preference': np.random.uniform(0.1, 1.0),
            'c_category_preference': np.random.uniform(0.1, 1.0)
        }
        users_data.append(user)
    
    users_df = pd.DataFrame(users_data)
    
    # 3. 상호작용 데이터 (500개)
    print("🔄 상호작용 데이터 생성 중...")
    
    interactions_data = []
    for i in range(500):
        user_id = f'user_{np.random.randint(1, 51):03d}'
        ads_idx = np.random.randint(1, 101)
        interaction_type = np.random.choice(['클릭', '클릭+전환'], p=[0.8, 0.2])
        
        if interaction_type == '클릭+전환':
            reward_point = np.random.uniform(100, 1000)
            rwd_price = reward_point
        else:
            reward_point = 0
            rwd_price = 0
        
        interaction = {
            'user_device_id': user_id,
            'ads_idx': ads_idx,
            'ads_code': f'AD{ads_idx:03d}',
            'ads_name': f'광고 {ads_idx}',
            'ads_type': np.random.randint(1, 5),
            'ads_category': np.random.randint(1, 6),
            'interaction_type': interaction_type,
            'reward_point': reward_point,
            'rwd_price': rwd_price
        }
        interactions_data.append(interaction)
    
    interactions_df = pd.DataFrame(interactions_data)
    
    # 4. 압축 파일로 저장
    print("📦 압축 파일 생성 중...")
    
    # 광고 데이터
    with zipfile.ZipFile("ads_profile_tiny.zip", 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_ref:
        zip_ref.writestr("ads_profile.csv", ads_df.to_csv(index=False))
    
    # 사용자 데이터
    with zipfile.ZipFile("user_profile_tiny.zip", 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_ref:
        zip_ref.writestr("user_profile.csv", users_df.to_csv(index=False))
    
    # 상호작용 데이터
    with zipfile.ZipFile("correct_interactions_tiny.zip", 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_ref:
        zip_ref.writestr("correct_interactions.csv", interactions_df.to_csv(index=False))
    
    print("✅ 매우 작은 샘플 데이터 생성 완료!")
    print(f"📊 최종 결과:")
    print(f"   - 광고: {len(ads_df)}개")
    print(f"   - 사용자: {len(users_df)}개")
    print(f"   - 상호작용: {len(interactions_df)}개")
    
    # 파일 크기 확인
    import os
    files = ["ads_profile_tiny.zip", "user_profile_tiny.zip", "correct_interactions_tiny.zip"]
    total_size = 0
    
    for filename in files:
        size_kb = os.path.getsize(filename) / 1024
        total_size += size_kb
        print(f"   - {filename}: {size_kb:.1f} KB")
    
    print(f"📊 총 크기: {total_size:.1f} KB")

if __name__ == "__main__":
    create_tiny_sample_data()
