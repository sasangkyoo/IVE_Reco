#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
매칭되는 상호작용 데이터 생성 스크립트
- 샘플 광고 데이터에 맞는 가상의 상호작용 데이터 생성
"""

import pandas as pd
import zipfile
import numpy as np

def create_matching_interactions():
    """샘플 광고 데이터에 맞는 상호작용 데이터 생성"""
    print("🔧 매칭되는 상호작용 데이터 생성 중...")
    
    # 1. 샘플 광고 데이터 로드
    with zipfile.ZipFile("ads_profile_sample.zip", 'r') as zip_ref:
        with zip_ref.open("ads_profile.csv") as f:
            ads_df = pd.read_csv(f)
    
    print(f"📊 샘플 광고: {len(ads_df):,}개")
    sample_ads_indices = ads_df['ads_idx'].tolist()
    
    # 2. 샘플 사용자 데이터 로드
    with zipfile.ZipFile("user_profile_sample.zip", 'r') as zip_ref:
        with zip_ref.open("user_profile.csv") as f:
            users_df = pd.read_csv(f)
    
    print(f"📊 샘플 사용자: {len(users_df):,}개")
    sample_user_ids = users_df['user_device_id'].tolist()
    
    # 3. 가상의 상호작용 데이터 생성
    print("📊 가상 상호작용 데이터 생성 중...")
    
    np.random.seed(42)  # 재현 가능한 결과를 위해
    
    # 각 사용자당 평균 20개의 상호작용 생성
    interactions = []
    
    for user_id in sample_user_ids:
        # 사용자당 10-30개의 상호작용 (평균 20개)
        num_interactions = np.random.randint(10, 31)
        
        # 랜덤하게 광고 선택
        selected_ads = np.random.choice(sample_ads_indices, size=num_interactions, replace=True)
        
        for ads_idx in selected_ads:
            # 상호작용 타입 결정 (80% 클릭, 20% 클릭+전환)
            interaction_type = np.random.choice(['클릭', '클릭+전환'], p=[0.8, 0.2])
            
            # 리워드 포인트 계산
            if interaction_type == '클릭+전환':
                reward_point = np.random.uniform(100, 1000)  # 100-1000 포인트
                rwd_price = reward_point
            else:
                reward_point = 0
                rwd_price = 0
            
            # 광고 정보 가져오기
            ad_info = ads_df[ads_df['ads_idx'] == ads_idx].iloc[0]
            
            interaction = {
                'user_device_id': user_id,
                'ads_idx': ads_idx,
                'ads_code': ad_info['ads_code'],
                'ads_name': ad_info['ads_name'],
                'ads_type': ad_info['ads_type'],
                'ads_category': ad_info['ads_category'],
                'interaction_type': interaction_type,
                'reward_point': reward_point,
                'rwd_price': rwd_price
            }
            interactions.append(interaction)
    
    # DataFrame으로 변환
    interactions_df = pd.DataFrame(interactions)
    print(f"📊 생성된 상호작용: {len(interactions_df):,}개")
    
    # 4. 상호작용 데이터 저장
    with zipfile.ZipFile("correct_interactions_sample.zip", 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_ref:
        zip_ref.writestr("correct_interactions.csv", interactions_df.to_csv(index=False))
    
    print("✅ 매칭되는 상호작용 데이터 생성 완료!")
    
    # 통계 출력
    print(f"📊 최종 통계:")
    print(f"   - 총 상호작용: {len(interactions_df):,}개")
    print(f"   - 클릭: {len(interactions_df[interactions_df['interaction_type'] == '클릭']):,}개")
    print(f"   - 클릭+전환: {len(interactions_df[interactions_df['interaction_type'] == '클릭+전환']):,}개")
    print(f"   - 총 리워드: {interactions_df['reward_point'].sum():.0f} 포인트")

if __name__ == "__main__":
    create_matching_interactions()
