#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실제 데이터에서 랜덤으로 샘플 데이터 생성
- 실제 상호작용 데이터에서 랜덤 사용자 선택
- 선택된 사용자들의 실제 상호작용 추출
- 상호작용한 광고들의 실제 광고 데이터 추출
- 추천용 추가 광고 랜덤 선택하여 추가
"""

import pandas as pd
import numpy as np
import zipfile
import os
from typing import Tuple, List, Set

def load_original_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """원본 데이터 로드 (기존 샘플 데이터 사용)"""
    print("📊 원본 데이터 로딩 중...")
    
    # 광고 데이터 로드 (기존 샘플 데이터 사용)
    with zipfile.ZipFile("ads_profile_sample.zip", 'r') as zip_ref:
        with zip_ref.open("ads_profile.csv") as f:
            ads_df = pd.read_csv(f, dtype={
                'ads_idx': 'int32',
                'ad_type': 'string',
                'ad_category': 'string',
                'ad_subcategory': 'string',
                'ad_title': 'string',
                'ad_description': 'string',
                'ad_price': 'float32',
                'ad_rating': 'float32',
                'ad_review_count': 'int32',
                'ad_click_count': 'int32',
                'ad_conversion_count': 'int32'
            })
    
    # 사용자 데이터 로드 (기존 샘플 데이터 사용)
    with zipfile.ZipFile("user_profile_sample.zip", 'r') as zip_ref:
        with zip_ref.open("user_profile.csv") as f:
            users_df = pd.read_csv(f, dtype={
                'user_device_id': 'string',
                'user_age': 'int32',
                'user_gender': 'string',
                'user_region': 'string',
                'user_income': 'float32',
                'user_interests': 'string',
                'user_behavior_score': 'float32',
                'user_activity_level': 'string',
                'user_device_type': 'string',
                'user_os': 'string',
                'total_interactions': 'int32',
                'total_reward': 'float32'
            })
    
    # 상호작용 데이터 로드 (기존 샘플 데이터 사용)
    with zipfile.ZipFile("correct_interactions_sample.zip", 'r') as zip_ref:
        with zip_ref.open("correct_interactions.csv") as f:
            interactions_df = pd.read_csv(f, dtype={
                'user_device_id': 'string',
                'ads_idx': 'int32',
                'interaction_type': 'string',
                'rwd_price': 'float32',
                'reward_point': 'float32'
            })
    
    print(f"✅ 광고 데이터: {len(ads_df):,}개")
    print(f"✅ 사용자 데이터: {len(users_df):,}명")
    print(f"✅ 상호작용 데이터: {len(interactions_df):,}개")
    
    return ads_df, users_df, interactions_df

def select_random_users(users_df: pd.DataFrame, interactions_df: pd.DataFrame, n_users: int = 500) -> List[str]:
    """상호작용이 있는 사용자 중에서 랜덤 선택"""
    print(f"🎯 상호작용이 있는 사용자 중에서 {n_users}명 랜덤 선택 중...")
    
    # 상호작용이 있는 사용자만 필터링
    active_users = interactions_df['user_device_id'].unique()
    print(f"📊 상호작용이 있는 사용자: {len(active_users):,}명")
    
    # 랜덤 선택
    selected_users = np.random.choice(active_users, size=min(n_users, len(active_users)), replace=False)
    
    print(f"✅ 선택된 사용자: {len(selected_users)}명")
    return selected_users.tolist()

def extract_user_interactions(interactions_df: pd.DataFrame, selected_users: List[str]) -> pd.DataFrame:
    """선택된 사용자들의 상호작용 데이터 추출"""
    print("📊 선택된 사용자들의 상호작용 데이터 추출 중...")
    
    user_interactions = interactions_df[interactions_df['user_device_id'].isin(selected_users)].copy()
    
    print(f"✅ 추출된 상호작용: {len(user_interactions):,}개")
    return user_interactions

def extract_interacted_ads(ads_df: pd.DataFrame, user_interactions: pd.DataFrame) -> pd.DataFrame:
    """상호작용한 광고들의 광고 데이터 추출"""
    print("📊 상호작용한 광고들의 광고 데이터 추출 중...")
    
    interacted_ads_idx = user_interactions['ads_idx'].unique()
    interacted_ads = ads_df[ads_df['ads_idx'].isin(interacted_ads_idx)].copy()
    
    print(f"✅ 상호작용한 광고: {len(interacted_ads):,}개")
    return interacted_ads

def add_recommendation_ads(ads_df: pd.DataFrame, interacted_ads: pd.DataFrame, n_additional: int = 500) -> pd.DataFrame:
    """추천용 추가 광고 랜덤 선택"""
    print(f"📊 추천용 추가 광고 {n_additional}개 랜덤 선택 중...")
    
    # 이미 상호작용한 광고 제외
    interacted_ads_idx = set(interacted_ads['ads_idx'])
    available_ads = ads_df[~ads_df['ads_idx'].isin(interacted_ads_idx)]
    
    # 랜덤 선택
    additional_ads = available_ads.sample(n=min(n_additional, len(available_ads)), random_state=42)
    
    print(f"✅ 추가 광고: {len(additional_ads)}개")
    return additional_ads

def create_sample_users(users_df: pd.DataFrame, selected_users: List[str], user_interactions: pd.DataFrame) -> pd.DataFrame:
    """샘플 사용자 데이터 생성"""
    print("📊 샘플 사용자 데이터 생성 중...")
    
    sample_users = users_df[users_df['user_device_id'].isin(selected_users)].copy()
    
    # 상호작용 통계 재계산
    print("📊 사용자별 상호작용 통계 재계산 중...")
    for user_id in selected_users:
        user_interactions_subset = user_interactions[user_interactions['user_device_id'] == user_id]
        sample_users.loc[sample_users['user_device_id'] == user_id, 'total_interactions'] = len(user_interactions_subset)
        sample_users.loc[sample_users['user_device_id'] == user_id, 'total_reward'] = user_interactions_subset['reward_point'].sum()
    
    print(f"✅ 샘플 사용자: {len(sample_users)}명")
    return sample_users

def create_sample_ads(interacted_ads: pd.DataFrame, additional_ads: pd.DataFrame) -> pd.DataFrame:
    """샘플 광고 데이터 생성"""
    print("📊 샘플 광고 데이터 생성 중...")
    
    # 상호작용한 광고 + 추가 광고
    sample_ads = pd.concat([interacted_ads, additional_ads], ignore_index=True)
    
    print(f"✅ 샘플 광고: {len(sample_ads)}개 (상호작용: {len(interacted_ads)}개 + 추가: {len(additional_ads)}개)")
    return sample_ads

def create_sample_interactions(user_interactions: pd.DataFrame, sample_users: pd.DataFrame, sample_ads: pd.DataFrame) -> pd.DataFrame:
    """샘플 상호작용 데이터 생성"""
    print("📊 샘플 상호작용 데이터 생성 중...")
    
    # 샘플 사용자와 샘플 광고에 해당하는 상호작용만 필터링
    sample_user_ids = set(sample_users['user_device_id'])
    sample_ads_idx = set(sample_ads['ads_idx'])
    
    sample_interactions = user_interactions[
        (user_interactions['user_device_id'].isin(sample_user_ids)) &
        (user_interactions['ads_idx'].isin(sample_ads_idx))
    ].copy()
    
    print(f"✅ 샘플 상호작용: {len(sample_interactions):,}개")
    return sample_interactions

def save_sample_data(sample_users: pd.DataFrame, sample_ads: pd.DataFrame, sample_interactions: pd.DataFrame):
    """샘플 데이터 저장"""
    print("💾 샘플 데이터 저장 중...")
    
    # 사용자 데이터 저장
    with zipfile.ZipFile("user_profile_real_sample.zip", 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.writestr("user_profile.csv", sample_users.to_csv(index=False))
    
    # 광고 데이터 저장
    with zipfile.ZipFile("ads_profile_real_sample.zip", 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.writestr("ads_profile.csv", sample_ads.to_csv(index=False))
    
    # 상호작용 데이터 저장
    with zipfile.ZipFile("correct_interactions_real_sample.zip", 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.writestr("correct_interactions.csv", sample_interactions.to_csv(index=False))
    
    print("✅ 샘플 데이터 저장 완료!")
    print(f"📁 user_profile_real_sample.zip: {os.path.getsize('user_profile_real_sample.zip') / 1024 / 1024:.1f}MB")
    print(f"📁 ads_profile_real_sample.zip: {os.path.getsize('ads_profile_real_sample.zip') / 1024 / 1024:.1f}MB")
    print(f"📁 correct_interactions_real_sample.zip: {os.path.getsize('correct_interactions_real_sample.zip') / 1024 / 1024:.1f}MB")

def main():
    """메인 함수"""
    print("🚀 실제 데이터 기반 샘플 데이터 생성 시작!")
    
    # 원본 데이터 로드
    ads_df, users_df, interactions_df = load_original_data()
    
    # 랜덤 사용자 선택
    selected_users = select_random_users(users_df, interactions_df, n_users=500)
    
    # 선택된 사용자들의 상호작용 추출
    user_interactions = extract_user_interactions(interactions_df, selected_users)
    
    # 상호작용한 광고들의 광고 데이터 추출
    interacted_ads = extract_interacted_ads(ads_df, user_interactions)
    
    # 추천용 추가 광고 선택
    additional_ads = add_recommendation_ads(ads_df, interacted_ads, n_additional=500)
    
    # 샘플 데이터 생성
    sample_users = create_sample_users(users_df, selected_users, user_interactions)
    sample_ads = create_sample_ads(interacted_ads, additional_ads)
    sample_interactions = create_sample_interactions(user_interactions, sample_users, sample_ads)
    
    # 샘플 데이터 저장
    save_sample_data(sample_users, sample_ads, sample_interactions)
    
    print("\n🎉 실제 데이터 기반 샘플 데이터 생성 완료!")
    print(f"📊 최종 통계:")
    print(f"   - 사용자: {len(sample_users):,}명")
    print(f"   - 광고: {len(sample_ads):,}개")
    print(f"   - 상호작용: {len(sample_interactions):,}개")

if __name__ == "__main__":
    main()

