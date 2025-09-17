#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실제 데이터처럼 자연스러운 상호작용 분포 생성
- 파레토 분포 (80-20 법칙) 적용
- 일부 사용자는 많이, 대부분은 적게 상호작용
- 실제 광고 플랫폼 패턴 모방
"""

import pandas as pd
import numpy as np
import zipfile
import os
from typing import List, Tuple

def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """샘플 데이터 로드"""
    print("📊 샘플 데이터 로딩 중...")
    
    # 광고 데이터 로드
    with zipfile.ZipFile("ads_profile_expanded_sample.zip", 'r') as zip_ref:
        with zip_ref.open("ads_profile.csv") as f:
            ads_df = pd.read_csv(f)
    
    # 사용자 데이터 로드
    with zipfile.ZipFile("user_profile_expanded_sample.zip", 'r') as zip_ref:
        with zip_ref.open("user_profile.csv") as f:
            users_df = pd.read_csv(f)
    
    # 상호작용 데이터 로드
    with zipfile.ZipFile("correct_interactions_expanded_sample.zip", 'r') as zip_ref:
        with zip_ref.open("correct_interactions.csv") as f:
            interactions_df = pd.read_csv(f)
    
    print(f"✅ 광고: {len(ads_df):,}개, 사용자: {len(users_df):,}명, 상호작용: {len(interactions_df):,}개")
    return ads_df, users_df, interactions_df

def create_natural_interaction_distribution(n_users: int = 500, n_ads: int = 1000) -> List[Tuple[str, int, int]]:
    """자연스러운 상호작용 분포 생성 (파레토 분포)"""
    print("🎯 자연스러운 상호작용 분포 생성 중...")
    
    # 사용자별 상호작용 수 생성 (파레토 분포)
    # 80%의 사용자는 적게, 20%의 사용자는 많이 상호작용
    user_interactions = []
    
    for i in range(n_users):
        user_id = f"user_{i:06d}"
        
        # 파레토 분포로 상호작용 수 결정 (더 현실적으로)
        if np.random.random() < 0.1:  # 10%는 활성 사용자
            # 활성 사용자: 8-15개 상호작용
            interactions = np.random.randint(8, 16)
        elif np.random.random() < 0.3:  # 20%는 보통 사용자
            # 보통 사용자: 3-8개 상호작용
            interactions = np.random.randint(3, 9)
        else:  # 70%는 비활성 사용자
            # 비활성 사용자: 1-3개 상호작용
            interactions = np.random.randint(1, 4)
        
        user_interactions.append((user_id, interactions))
    
    # 광고별 인기도 생성 (파레토 분포)
    # 20%의 광고가 80%의 상호작용을 받음
    ad_popularity = []
    for i in range(n_ads):
        ad_idx = i + 1
        if np.random.random() < 0.2:  # 20%는 인기 광고
            popularity = np.random.randint(8, 25)  # 높은 인기도
        elif np.random.random() < 0.5:  # 30%는 보통 광고
            popularity = np.random.randint(3, 9)   # 보통 인기도
        else:  # 50%는 비인기 광고
            popularity = np.random.randint(1, 4)   # 낮은 인기도
        
        ad_popularity.append((ad_idx, popularity))
    
    print(f"✅ 사용자 분포: 활성 {sum(1 for _, count in user_interactions if count >= 8)}명, 보통 {sum(1 for _, count in user_interactions if 3 <= count < 8)}명, 비활성 {sum(1 for _, count in user_interactions if count < 3)}명")
    print(f"✅ 광고 분포: 인기 {sum(1 for _, pop in ad_popularity if pop >= 8)}개, 보통 {sum(1 for _, pop in ad_popularity if 3 <= pop < 8)}개, 비인기 {sum(1 for _, pop in ad_popularity if pop < 3)}개")
    
    return user_interactions, ad_popularity

def generate_natural_interactions(user_interactions: List[Tuple[str, int]], ad_popularity: List[Tuple[int, int]], 
                                original_interactions: pd.DataFrame) -> pd.DataFrame:
    """자연스러운 상호작용 데이터 생성"""
    print("📊 자연스러운 상호작용 데이터 생성 중...")
    
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
            # 상호작용 타입 결정 (80% 클릭, 20% 클릭+전환)
            if np.random.random() < 0.8:
                interaction_type = "클릭"
                reward_point = 0.0
            else:
                interaction_type = "클릭+전환"
                reward_point = np.random.uniform(10, 100)  # 리워드 포인트
            
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

def create_natural_users(users_df: pd.DataFrame, user_interactions: List[Tuple[str, int]], 
                        interactions_df: pd.DataFrame) -> pd.DataFrame:
    """자연스러운 사용자 데이터 생성"""
    print("📊 자연스러운 사용자 데이터 생성 중...")
    
    natural_users = []
    
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
        
        natural_users.append(user_data)
    
    natural_users_df = pd.DataFrame(natural_users)
    
    print(f"✅ 생성된 사용자: {len(natural_users_df):,}명")
    print(f"✅ 평균 상호작용: {natural_users_df['total_interactions'].mean():.1f}개")
    print(f"✅ 최대 상호작용: {natural_users_df['total_interactions'].max()}개")
    print(f"✅ 최소 상호작용: {natural_users_df['total_interactions'].min()}개")
    
    return natural_users_df

def save_natural_data(users_df: pd.DataFrame, interactions_df: pd.DataFrame):
    """자연스러운 데이터 저장"""
    print("💾 자연스러운 데이터 저장 중...")
    
    # 사용자 데이터 저장
    with zipfile.ZipFile("user_profile_natural.zip", 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.writestr("user_profile.csv", users_df.to_csv(index=False))
    
    # 상호작용 데이터 저장
    with zipfile.ZipFile("correct_interactions_natural.zip", 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.writestr("correct_interactions.csv", interactions_df.to_csv(index=False))
    
    print("✅ 자연스러운 데이터 저장 완료!")
    print(f"📁 user_profile_natural.zip: {os.path.getsize('user_profile_natural.zip') / 1024 / 1024:.1f}MB")
    print(f"📁 correct_interactions_natural.zip: {os.path.getsize('correct_interactions_natural.zip') / 1024 / 1024:.1f}MB")

def main():
    """메인 함수"""
    print("🚀 자연스러운 상호작용 데이터 생성 시작!")
    
    # 샘플 데이터 로드
    ads_df, users_df, original_interactions = load_sample_data()
    
    # 자연스러운 분포 생성
    user_interactions, ad_popularity = create_natural_interaction_distribution(500, 1000)
    
    # 자연스러운 상호작용 데이터 생성
    interactions_df = generate_natural_interactions(user_interactions, ad_popularity, original_interactions)
    
    # 자연스러운 사용자 데이터 생성
    natural_users_df = create_natural_users(users_df, user_interactions, interactions_df)
    
    # 데이터 저장
    save_natural_data(natural_users_df, interactions_df)
    
    print("\n🎉 자연스러운 데이터 생성 완료!")
    print(f"📊 최종 통계:")
    print(f"   - 사용자: {len(natural_users_df):,}명")
    print(f"   - 상호작용: {len(interactions_df):,}개")
    print(f"   - 평균 상호작용: {len(interactions_df) / len(natural_users_df):.1f}개")

if __name__ == "__main__":
    main()
