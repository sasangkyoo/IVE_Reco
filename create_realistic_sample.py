#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실제 데이터 분포에 맞춘 현실적인 샘플 데이터 생성
- 평균 2.5개, 중간값 1개, 표준편차 5.6개
- 파레토 분포 (80-20 법칙) 적용
- 실제 광고 플랫폼 패턴 모방
"""

import pandas as pd
import numpy as np
import zipfile
import os
from typing import List, Tuple

def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """기존 샘플 데이터 로드"""
    print("📊 기존 샘플 데이터 로딩 중...")
    
    # 광고 데이터 로드
    with zipfile.ZipFile("ads_profile_expanded_sample.zip", 'r') as zip_ref:
        with zip_ref.open("ads_profile.csv") as f:
            ads_df = pd.read_csv(f)
    
    # 사용자 데이터 로드
    with zipfile.ZipFile("user_profile_sample.zip", 'r') as zip_ref:
        with zip_ref.open("user_profile.csv") as f:
            users_df = pd.read_csv(f)
    
    # 상호작용 데이터 로드
    with zipfile.ZipFile("correct_interactions_sample.zip", 'r') as zip_ref:
        with zip_ref.open("correct_interactions.csv") as f:
            interactions_df = pd.read_csv(f)
    
    print(f"✅ 광고: {len(ads_df):,}개, 사용자: {len(users_df):,}명, 상호작용: {len(interactions_df):,}개")
    return ads_df, users_df, interactions_df

def create_realistic_interaction_distribution(n_users: int = 500, n_ads: int = 1000) -> List[Tuple[str, int, int]]:
    """실제 데이터 분포에 맞춘 상호작용 분포 생성"""
    print("🎯 실제 데이터 분포에 맞춘 상호작용 분포 생성 중...")
    
    # 실제 데이터 분포: 평균 2.5, 중간값 1, 표준편차 5.6
    # 파레토 분포를 사용하여 현실적인 분포 생성
    
    user_interactions = []
    
    for i in range(n_users):
        user_id = f"user_{i:06d}"
        
        # 파레토 분포로 상호작용 수 결정 (실제 데이터 패턴)
        if np.random.random() < 0.05:  # 5%는 매우 활성 사용자 (50-200개)
            interactions = np.random.randint(50, 201)
        elif np.random.random() < 0.15:  # 10%는 활성 사용자 (10-50개)
            interactions = np.random.randint(10, 51)
        elif np.random.random() < 0.35:  # 20%는 보통 사용자 (3-10개)
            interactions = np.random.randint(3, 11)
        else:  # 65%는 비활성 사용자 (1-3개)
            interactions = np.random.randint(1, 4)
        
        user_interactions.append((user_id, interactions))
    
    # 광고별 인기도 생성 (파레토 분포)
    # 20%의 광고가 80%의 상호작용을 받음
    ad_popularity = []
    for i in range(n_ads):
        ad_idx = i + 1
        if np.random.random() < 0.05:  # 5%는 매우 인기 광고
            popularity = np.random.randint(50, 200)  # 높은 인기도
        elif np.random.random() < 0.15:  # 10%는 인기 광고
            popularity = np.random.randint(10, 50)   # 보통 인기도
        elif np.random.random() < 0.35:  # 20%는 보통 광고
            popularity = np.random.randint(3, 10)    # 낮은 인기도
        else:  # 50%는 비인기 광고
            popularity = np.random.randint(1, 3)     # 매우 낮은 인기도
        
        ad_popularity.append((ad_idx, popularity))
    
    # 통계 계산
    total_interactions = sum(count for _, count in user_interactions)
    avg_interactions = total_interactions / n_users
    median_interactions = np.median([count for _, count in user_interactions])
    std_interactions = np.std([count for _, count in user_interactions])
    
    print(f"✅ 사용자 분포: 매우활성 {sum(1 for _, count in user_interactions if count >= 50)}명, 활성 {sum(1 for _, count in user_interactions if 10 <= count < 50)}명, 보통 {sum(1 for _, count in user_interactions if 3 <= count < 10)}명, 비활성 {sum(1 for _, count in user_interactions if count < 3)}명")
    print(f"✅ 광고 분포: 매우인기 {sum(1 for _, pop in ad_popularity if pop >= 50)}개, 인기 {sum(1 for _, pop in ad_popularity if 10 <= pop < 50)}개, 보통 {sum(1 for _, pop in ad_popularity if 3 <= pop < 10)}개, 비인기 {sum(1 for _, pop in ad_popularity if pop < 3)}개")
    print(f"✅ 통계: 평균 {avg_interactions:.1f}개, 중간값 {median_interactions:.1f}개, 표준편차 {std_interactions:.1f}개")
    
    return user_interactions, ad_popularity

def generate_realistic_interactions(user_interactions: List[Tuple[str, int]], ad_popularity: List[Tuple[int, int]], 
                                  original_interactions: pd.DataFrame) -> pd.DataFrame:
    """현실적인 상호작용 데이터 생성"""
    print("📊 현실적인 상호작용 데이터 생성 중...")
    
    interactions_list = []
    
    for user_id, user_count in user_interactions:
        # 사용자의 상호작용 수만큼 광고 선택
        # 인기도에 따라 가중치 적용
        ad_weights = [pop for _, pop in ad_popularity]
        ad_indices = [idx for idx, _ in ad_popularity]
        
        # 가중치 기반 샘플링
        selected_ads = np.random.choice(
            ad_indices, 
            size=min(user_count, len(ad_indices)), 
            replace=False, 
            p=np.array(ad_weights) / sum(ad_weights)
        )
        
        for ad_idx in selected_ads:
            # 상호작용 타입 결정 (실제 데이터: 52% 클릭+전환, 48% 클릭)
            if np.random.random() < 0.52:
                interaction_type = "클릭+전환"
                reward_point = np.random.uniform(10, 100)  # 리워드 포인트
            else:
                interaction_type = "클릭"
                reward_point = 0.0
            
            # 원본 데이터에서 비슷한 패턴 찾기
            similar_interaction = original_interactions.sample(1).iloc[0]
            
            interaction = {
                'user_device_id': user_id,
                'ads_idx': ad_idx,
                'interaction_type': interaction_type,
                'rwd_price': similar_interaction['rwd_price'] if interaction_type == "클릭+전환" else 0.0,
                'reward_point': reward_point
            }
            
            interactions_list.append(interaction)
    
    interactions_df = pd.DataFrame(interactions_list)
    
    print(f"✅ 생성된 상호작용: {len(interactions_df):,}개")
    print(f"✅ 클릭: {len(interactions_df[interactions_df['interaction_type'] == '클릭']):,}개")
    print(f"✅ 클릭+전환: {len(interactions_df[interactions_df['interaction_type'] == '클릭+전환']):,}개")
    
    return interactions_df

def create_realistic_users(users_df: pd.DataFrame, user_interactions: List[Tuple[str, int]], 
                          interactions_df: pd.DataFrame) -> pd.DataFrame:
    """현실적인 사용자 데이터 생성"""
    print("📊 현실적인 사용자 데이터 생성 중...")
    
    realistic_users = []
    
    for i, (user_id, interaction_count) in enumerate(user_interactions):
        # 원본 사용자 데이터에서 랜덤 선택
        base_user = users_df.sample(1).iloc[0]
        
        # 사용자 ID 변경
        user_data = base_user.copy()
        user_data['user_device_id'] = user_id
        
        # 상호작용 통계 업데이트
        user_interactions_subset = interactions_df[interactions_df['user_device_id'] == user_id]
        user_data['total_interactions'] = len(user_interactions_subset)
        user_data['total_reward'] = user_interactions_subset['reward_point'].sum()
        
        realistic_users.append(user_data)
    
    realistic_users_df = pd.DataFrame(realistic_users)
    
    print(f"✅ 생성된 사용자: {len(realistic_users_df):,}명")
    print(f"✅ 평균 상호작용: {realistic_users_df['total_interactions'].mean():.1f}개")
    print(f"✅ 최대 상호작용: {realistic_users_df['total_interactions'].max()}개")
    print(f"✅ 최소 상호작용: {realistic_users_df['total_interactions'].min()}개")
    print(f"✅ 중간값: {realistic_users_df['total_interactions'].median():.1f}개")
    print(f"✅ 표준편차: {realistic_users_df['total_interactions'].std():.1f}개")
    
    return realistic_users_df

def save_realistic_data(users_df: pd.DataFrame, interactions_df: pd.DataFrame):
    """현실적인 데이터 저장"""
    print("💾 현실적인 데이터 저장 중...")
    
    # 사용자 데이터 저장
    with zipfile.ZipFile("user_profile_realistic.zip", 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.writestr("user_profile.csv", users_df.to_csv(index=False))
    
    # 상호작용 데이터 저장
    with zipfile.ZipFile("correct_interactions_realistic.zip", 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.writestr("correct_interactions.csv", interactions_df.to_csv(index=False))
    
    print("✅ 현실적인 데이터 저장 완료!")
    print(f"📁 user_profile_realistic.zip: {os.path.getsize('user_profile_realistic.zip') / 1024 / 1024:.1f}MB")
    print(f"📁 correct_interactions_realistic.zip: {os.path.getsize('correct_interactions_realistic.zip') / 1024 / 1024:.1f}MB")

def main():
    """메인 함수"""
    print("🚀 실제 데이터 분포에 맞춘 현실적인 샘플 데이터 생성 시작!")
    
    # 샘플 데이터 로드
    ads_df, users_df, original_interactions = load_sample_data()
    
    # 현실적인 분포 생성
    user_interactions, ad_popularity = create_realistic_interaction_distribution(500, 1000)
    
    # 현실적인 상호작용 데이터 생성
    interactions_df = generate_realistic_interactions(user_interactions, ad_popularity, original_interactions)
    
    # 현실적인 사용자 데이터 생성
    realistic_users_df = create_realistic_users(users_df, user_interactions, interactions_df)
    
    # 데이터 저장
    save_realistic_data(realistic_users_df, interactions_df)
    
    print("\n🎉 현실적인 데이터 생성 완료!")
    print(f"📊 최종 통계:")
    print(f"   - 사용자: {len(realistic_users_df):,}명")
    print(f"   - 상호작용: {len(interactions_df):,}개")
    print(f"   - 평균 상호작용: {len(interactions_df) / len(realistic_users_df):.1f}개")

if __name__ == "__main__":
    main()
