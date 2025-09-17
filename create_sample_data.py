#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
샘플 데이터 생성 스크립트
- 더 작은 크기의 샘플 데이터를 생성하여 빠른 테스트 가능
"""

import pandas as pd
import numpy as np
import zipfile
import os

def create_sample_ads_profile():
    """광고 프로필 샘플 데이터 생성 (1,000개)"""
    print("📊 광고 프로필 샘플 데이터 생성 중...")
    
    # 원본 데이터 로드
    with zipfile.ZipFile("ads_profile.zip", 'r') as zip_ref:
        with zip_ref.open("ads_profile.csv") as f:
            df = pd.read_csv(f)
    
    # 1,000개 샘플링
    sample_df = df.sample(n=1000, random_state=42)
    
    # 압축 파일로 저장
    with zipfile.ZipFile("ads_profile_sample.zip", 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_ref:
        zip_ref.writestr("ads_profile.csv", sample_df.to_csv(index=False))
    
    print(f"✅ ads_profile_sample.zip 생성 완료 ({len(sample_df):,}개 광고)")

def create_sample_user_profile():
    """사용자 프로필 샘플 데이터 생성 (500개)"""
    print("👥 사용자 프로필 샘플 데이터 생성 중...")
    
    # 원본 데이터 로드
    with zipfile.ZipFile("user_profile.zip", 'r') as zip_ref:
        with zip_ref.open("user_profile.csv") as f:
            df = pd.read_csv(f)
    
    # 500개 샘플링
    sample_df = df.sample(n=500, random_state=42)
    
    # 압축 파일로 저장
    with zipfile.ZipFile("user_profile_sample.zip", 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_ref:
        zip_ref.writestr("user_profile.csv", sample_df.to_csv(index=False))
    
    print(f"✅ user_profile_sample.zip 생성 완료 ({len(sample_df):,}개 사용자)")

def create_sample_interactions():
    """상호작용 데이터 샘플 생성 (10,000개)"""
    print("🔄 상호작용 데이터 샘플 생성 중...")
    
    # 원본 데이터 로드
    with zipfile.ZipFile("correct_interactions.zip", 'r') as zip_ref:
        with zip_ref.open("correct_interactions.csv") as f:
            df = pd.read_csv(f)
    
    # 10,000개 샘플링
    sample_df = df.sample(n=10000, random_state=42)
    
    # 압축 파일로 저장
    with zipfile.ZipFile("correct_interactions_sample.zip", 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_ref:
        zip_ref.writestr("correct_interactions.csv", sample_df.to_csv(index=False))
    
    print(f"✅ correct_interactions_sample.zip 생성 완료 ({len(sample_df):,}개 상호작용)")

def main():
    """메인 함수"""
    print("🚀 샘플 데이터 생성 시작...")
    
    try:
        create_sample_ads_profile()
        create_sample_user_profile()
        create_sample_interactions()
        
        print("\n✅ 모든 샘플 데이터 생성 완료!")
        print("\n📁 생성된 파일들:")
        print("- ads_profile_sample.zip (1,000개 광고)")
        print("- user_profile_sample.zip (500개 사용자)")
        print("- correct_interactions_sample.zip (10,000개 상호작용)")
        
        # 파일 크기 확인
        print("\n📊 파일 크기:")
        for filename in ["ads_profile_sample.zip", "user_profile_sample.zip", "correct_interactions_sample.zip"]:
            if os.path.exists(filename):
                size_mb = os.path.getsize(filename) / (1024 * 1024)
                print(f"- {filename}: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
