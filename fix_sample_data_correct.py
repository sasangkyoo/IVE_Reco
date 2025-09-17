#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
샘플 데이터 올바른 수정 스크립트
- 샘플 광고 데이터에 포함된 광고들만 상호작용하도록 수정
"""

import pandas as pd
import zipfile

def fix_sample_interactions_correct():
    """샘플 상호작용 데이터를 샘플 광고 데이터와 올바르게 매칭"""
    print("🔧 샘플 상호작용 데이터 올바른 수정 중...")
    
    # 샘플 광고 데이터의 실제 인덱스들 확인
    with zipfile.ZipFile("ads_profile_sample.zip", 'r') as zip_ref:
        with zip_ref.open("ads_profile.csv") as f:
            ads_df = pd.read_csv(f)
    
    # 샘플 광고에 포함된 ads_idx들
    sample_ads_indices = set(ads_df['ads_idx'].tolist())
    print(f"📊 샘플 광고 인덱스 개수: {len(sample_ads_indices)}개")
    print(f"📊 샘플 광고 인덱스 범위: {min(sample_ads_indices)} ~ {max(sample_ads_indices)}")
    
    # 기존 샘플 상호작용 데이터 로드
    with zipfile.ZipFile("correct_interactions_sample.zip", 'r') as zip_ref:
        with zip_ref.open("correct_interactions.csv") as f:
            interactions_df = pd.read_csv(f)
    
    print(f"📊 수정 전 상호작용: {len(interactions_df):,}개")
    
    # 샘플 광고에 포함된 광고들만 상호작용 필터링
    filtered_df = interactions_df[interactions_df['ads_idx'].isin(sample_ads_indices)]
    
    print(f"📊 수정 후 상호작용: {len(filtered_df):,}개")
    
    # 만약 필터링 후 데이터가 너무 적으면 샘플링 조정
    if len(filtered_df) < 1000:
        print("⚠️ 필터링 후 데이터가 부족합니다. 샘플 광고 데이터를 다시 생성해야 할 수 있습니다.")
        # 최대한 많은 데이터 사용
        target_size = min(len(filtered_df), 10000)
        if len(filtered_df) > target_size:
            filtered_df = filtered_df.sample(n=target_size, random_state=42)
    else:
        # 10,000개로 샘플링
        if len(filtered_df) > 10000:
            filtered_df = filtered_df.sample(n=10000, random_state=42)
    
    print(f"📊 최종 상호작용: {len(filtered_df):,}개")
    
    # 수정된 데이터를 새 파일로 저장
    with zipfile.ZipFile("correct_interactions_sample_fixed.zip", 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_ref:
        zip_ref.writestr("correct_interactions.csv", filtered_df.to_csv(index=False))
    
    print("✅ correct_interactions_sample_fixed.zip 생성 완료")
    
    # 기존 파일 교체
    import shutil
    shutil.move("correct_interactions_sample_fixed.zip", "correct_interactions_sample.zip")
    print("✅ 기존 파일 교체 완료")

if __name__ == "__main__":
    fix_sample_interactions_correct()
