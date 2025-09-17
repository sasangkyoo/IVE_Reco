#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 샘플 데이터를 확장하여 5,000개의 광고가 포함된 샘플 데이터 생성
- 실제 상호작용 데이터 유지
- 광고 데이터를 5,000개로 확장
- 추천 시스템에 적합한 데이터셋 생성
"""

import pandas as pd
import numpy as np
import zipfile
import os
from typing import Tuple, List, Set

def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """기존 샘플 데이터 로드"""
    print("📊 기존 샘플 데이터 로딩 중...")
    
    # 광고 데이터 로드
    with zipfile.ZipFile("ads_profile_sample.zip", 'r') as zip_ref:
        with zip_ref.open("ads_profile.csv") as f:
            ads_df = pd.read_csv(f, dtype={
                'ads_idx': 'int32',
                'ads_code': 'string',
                'ads_type': 'float32',
                'ads_category': 'float32',
                'ads_name': 'string',
                'ads_age_min': 'float32',
                'ads_age_max': 'float32',
                'ads_os_type': 'float32',
                'show_price': 'float32',
                'rwd_price': 'float32',
                'ads_ranking': 'float32'
            })
    
    # 사용자 데이터 로드
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
    
    # 상호작용 데이터 로드
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

def expand_ads_data(ads_df: pd.DataFrame, target_count: int = 5000) -> pd.DataFrame:
    """광고 데이터를 5,000개로 확장"""
    print(f"📊 광고 데이터를 {target_count:,}개로 확장 중...")
    
    # 기존 광고 데이터 복사
    expanded_ads = ads_df.copy()
    
    # 필요한 추가 광고 수 계산
    additional_count = target_count - len(ads_df)
    
    if additional_count > 0:
        print(f"📊 추가 광고 {additional_count:,}개 생성 중...")
        
        # 기존 광고의 패턴을 기반으로 추가 광고 생성
        additional_ads_list = []
        
        for i in range(additional_count):
            # 기존 광고에서 랜덤 선택하여 변형
            base_ad = ads_df.sample(1).iloc[0]
            
            # 새로운 광고 인덱스 (기존 최대값 + 1부터 시작)
            new_ads_idx = len(ads_df) + i + 1
            
            # 광고 데이터 생성 (기존 패턴 기반)
            new_ad = {
                'ads_idx': new_ads_idx,
                'ads_code': f"EXT{new_ads_idx:06d}",
                'ads_type': base_ad['ads_type'],
                'ads_category': base_ad['ads_category'],
                'ads_name': f"{base_ad['ads_name']} (추천 {i+1})",
                'ads_age_min': base_ad['ads_age_min'],
                'ads_age_max': base_ad['ads_age_max'],
                'ads_os_type': base_ad['ads_os_type'],
                'show_price': base_ad['show_price'] * np.random.uniform(0.8, 1.2),  # 가격 변동
                'rwd_price': base_ad['rwd_price'] * np.random.uniform(0.8, 1.2),  # 리워드 가격 변동
                'ads_ranking': max(1, min(100, base_ad['ads_ranking'] + np.random.randint(-10, 11)))  # 랭킹 변동
            }
            
            # 나머지 컬럼들도 기존 패턴 기반으로 생성
            for col in base_ad.index:
                if col not in new_ad:
                    try:
                        # 숫자형 컬럼인지 확인
                        if pd.api.types.is_numeric_dtype(base_ad[col]):
                            # 숫자형 컬럼은 약간의 변동 추가
                            new_ad[col] = base_ad[col] * np.random.uniform(0.9, 1.1)
                        else:
                            # 문자열이나 다른 타입은 그대로 복사
                            new_ad[col] = base_ad[col]
                    except:
                        # 오류 발생 시 그대로 복사
                        new_ad[col] = base_ad[col]
            
            additional_ads_list.append(new_ad)
        
        # 추가 광고를 DataFrame으로 변환
        additional_ads_df = pd.DataFrame(additional_ads_list)
        
        # 기존 광고와 추가 광고 합치기
        expanded_ads = pd.concat([expanded_ads, additional_ads_df], ignore_index=True)
    
    print(f"✅ 확장된 광고 데이터: {len(expanded_ads):,}개")
    return expanded_ads

def save_expanded_sample_data(users_df: pd.DataFrame, expanded_ads: pd.DataFrame, interactions_df: pd.DataFrame):
    """확장된 샘플 데이터 저장"""
    print("💾 확장된 샘플 데이터 저장 중...")
    
    # 사용자 데이터 저장
    with zipfile.ZipFile("user_profile_expanded_sample.zip", 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.writestr("user_profile.csv", users_df.to_csv(index=False))
    
    # 광고 데이터 저장
    with zipfile.ZipFile("ads_profile_expanded_sample.zip", 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.writestr("ads_profile.csv", expanded_ads.to_csv(index=False))
    
    # 상호작용 데이터 저장 (기존 유지)
    with zipfile.ZipFile("correct_interactions_expanded_sample.zip", 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        zip_ref.writestr("correct_interactions.csv", interactions_df.to_csv(index=False))
    
    print("✅ 확장된 샘플 데이터 저장 완료!")
    print(f"📁 user_profile_expanded_sample.zip: {os.path.getsize('user_profile_expanded_sample.zip') / 1024 / 1024:.1f}MB")
    print(f"📁 ads_profile_expanded_sample.zip: {os.path.getsize('ads_profile_expanded_sample.zip') / 1024 / 1024:.1f}MB")
    print(f"📁 correct_interactions_expanded_sample.zip: {os.path.getsize('correct_interactions_expanded_sample.zip') / 1024 / 1024:.1f}MB")

def main():
    """메인 함수"""
    print("🚀 확장된 샘플 데이터 생성 시작!")
    
    # 기존 샘플 데이터 로드
    ads_df, users_df, interactions_df = load_sample_data()
    
    # 광고 데이터를 5,000개로 확장
    expanded_ads = expand_ads_data(ads_df, target_count=5000)
    
    # 확장된 샘플 데이터 저장
    save_expanded_sample_data(users_df, expanded_ads, interactions_df)
    
    print("\n🎉 확장된 샘플 데이터 생성 완료!")
    print(f"📊 최종 통계:")
    print(f"   - 사용자: {len(users_df):,}명")
    print(f"   - 광고: {len(expanded_ads):,}개")
    print(f"   - 상호작용: {len(interactions_df):,}개")
    print(f"   - 상호작용한 광고: {len(ads_df):,}개")
    print(f"   - 추가 광고: {len(expanded_ads) - len(ads_df):,}개")

if __name__ == "__main__":
    main()
