#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
샘플 데이터 수정 스크립트
- 기존 샘플 상호작용 데이터를 샘플 광고 데이터와 호환되도록 수정
"""

import pandas as pd
import zipfile

def fix_sample_interactions():
    """샘플 상호작용 데이터를 샘플 광고 데이터와 호환되도록 수정"""
    print("🔧 샘플 상호작용 데이터 수정 중...")
    
    # 샘플 광고 데이터의 인덱스 범위 확인
    with zipfile.ZipFile("ads_profile_sample.zip", 'r') as zip_ref:
        with zip_ref.open("ads_profile.csv") as f:
            ads_df = pd.read_csv(f)
    
    min_ads_idx = ads_df['ads_idx'].min()
    max_ads_idx = ads_df['ads_idx'].max()
    print(f"📊 샘플 광고 인덱스 범위: {min_ads_idx} ~ {max_ads_idx}")
    
    # 기존 샘플 상호작용 데이터 로드
    with zipfile.ZipFile("correct_interactions_sample.zip", 'r') as zip_ref:
        with zip_ref.open("correct_interactions.csv") as f:
            interactions_df = pd.read_csv(f)
    
    print(f"📊 수정 전 상호작용: {len(interactions_df):,}개")
    
    # 샘플 광고 인덱스 범위에 맞는 상호작용만 필터링
    filtered_df = interactions_df[
        (interactions_df['ads_idx'] >= min_ads_idx) & 
        (interactions_df['ads_idx'] <= max_ads_idx)
    ]
    
    print(f"📊 수정 후 상호작용: {len(filtered_df):,}개")
    
    # 수정된 데이터를 새 파일로 저장
    with zipfile.ZipFile("correct_interactions_sample_fixed.zip", 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_ref:
        zip_ref.writestr("correct_interactions.csv", filtered_df.to_csv(index=False))
    
    print("✅ correct_interactions_sample_fixed.zip 생성 완료")
    
    # 기존 파일 교체
    import shutil
    shutil.move("correct_interactions_sample_fixed.zip", "correct_interactions_sample.zip")
    print("✅ 기존 파일 교체 완료")

if __name__ == "__main__":
    fix_sample_interactions()
