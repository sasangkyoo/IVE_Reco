#!/usr/bin/env python3
"""
올바른 상호작용 데이터 생성 스크립트

규칙:
- click_key가 있으면 클릭 + 전환
- click_key_rwd가 있으면 클릭 + 전환  
- click_key_info가 있으면 클릭
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def create_correct_interactions():
    """올바른 상호작용 데이터 생성"""
    
    print("📊 상호작용 데이터 생성 시작...")
    
    # 1. f_ads_rwd_info.csv 로드 (실제 상호작용 데이터)
    print("1️⃣ f_ads_rwd_info.csv 로드 중...")
    f_ads_rwd = pd.read_csv('preprocessed/f_ads_rwd_info.csv', low_memory=False)
    print(f"   - 총 {len(f_ads_rwd):,}개 상호작용 로드")
    
    # 2. preprocessed_ads_list.csv 로드 (광고 메타데이터)
    print("2️⃣ preprocessed_ads_list.csv 로드 중...")
    ads_list = pd.read_csv('preprocessed/preprocessed_ads_list.csv', low_memory=False)
    print(f"   - 총 {len(ads_list):,}개 광고 로드")
    
    # 3. 상호작용 타입 결정
    print("3️⃣ 상호작용 타입 결정 중...")
    
    def determine_interaction_type(row):
        """상호작용 타입 결정"""
        click_key = row.get('click_key', '')
        click_key_rwd = row.get('click_key_rwd', '')
        click_key_info = row.get('click_key_info', '')
        
        # click_key 또는 click_key_rwd가 있으면 클릭 + 전환
        if pd.notna(click_key) and str(click_key).strip() != '':
            return '클릭+전환'
        elif pd.notna(click_key_rwd) and str(click_key_rwd).strip() != '':
            return '클릭+전환'
        # click_key_info가 있으면 클릭만
        elif pd.notna(click_key_info) and str(click_key_info).strip() != '':
            return '클릭'
        else:
            return '클릭'  # 기본값
    
    def adjust_reward_for_interaction_type(row):
        """상호작용 타입에 따라 리워드 조정"""
        interaction_type = row.get('interaction_type', '클릭')
        
        # 클릭만 있는 경우 리워드를 0으로 설정
        if interaction_type == '클릭':
            return 0.0, 0.0  # reward_point, rwd_price
        else:
            # 클릭+전환인 경우
            reward_point = row.get('reward_point', 0)
            rwd_price = row.get('rwd_price', 0)
            
            # reward_point가 nan이거나 0이면 rwd_price를 reward_point로 사용
            if pd.isna(reward_point) or reward_point == 0:
                reward_point = rwd_price
            
            return reward_point, rwd_price
    
    # 상호작용 타입 추가
    f_ads_rwd['interaction_type'] = f_ads_rwd.apply(determine_interaction_type, axis=1)
    
    # 리워드 조정 (클릭만 있는 경우 리워드 0으로 설정)
    print("4️⃣ 리워드 조정 중...")
    reward_adjustments = f_ads_rwd.apply(adjust_reward_for_interaction_type, axis=1)
    f_ads_rwd['reward_point'] = [adj[0] for adj in reward_adjustments]
    f_ads_rwd['rwd_price'] = [adj[1] for adj in reward_adjustments]
    
    # 5. 광고 메타데이터와 병합
    print("5️⃣ 광고 메타데이터와 병합 중...")
    
    # ads_list에서 필요한 컬럼만 선택 (컬럼명 충돌 방지)
    ads_meta = ads_list[['ads_idx', 'ads_type', 'ads_category', 'ads_name', 'rwd_price']].copy()
    ads_meta = ads_meta.rename(columns={
        'ads_type': 'ads_type_meta',
        'ads_category': 'ads_category_meta', 
        'ads_name': 'ads_name_meta',
        'rwd_price': 'rwd_price_meta'
    })
    
    # f_ads_rwd에서 rwd_price 컬럼 제거 (ads_meta의 rwd_price를 사용)
    f_ads_rwd_clean = f_ads_rwd.drop(columns=['rwd_price'], errors='ignore')
    
    # 상호작용 데이터와 광고 메타데이터 병합
    print(f"   - 병합 전 상호작용 데이터: {len(f_ads_rwd_clean):,}개")
    print(f"   - 병합 전 광고 메타데이터: {len(ads_meta):,}개")
    
    interactions = f_ads_rwd_clean.merge(ads_meta, on='ads_idx', how='left')
    print(f"   - 병합 후 데이터: {len(interactions):,}개")
    
    # 병합된 컬럼들을 원래 이름으로 변경
    interactions = interactions.rename(columns={
        'ads_type_meta': 'ads_type',
        'ads_category_meta': 'ads_category',
        'ads_name_meta': 'ads_name', 
        'rwd_price_meta': 'rwd_price'
    })
    
    # 6. 최종 상호작용 데이터 생성
    print("6️⃣ 최종 상호작용 데이터 생성 중...")
    
    # 사용 가능한 컬럼 확인
    print(f"   - 사용 가능한 컬럼: {list(interactions.columns)}")
    
    # 필요한 컬럼만 선택하고 정리
    required_columns = [
        'user_ip',           # 사용자 IP (백업용)
        'dvc_idx',           # 실제 사용자 디바이스 ID
        'ads_idx',           # 광고 ID
        'ads_type',          # 광고 타입
        'ads_category',      # 광고 카테고리
        'ads_name',          # 광고 이름
        'click_time',        # 클릭 시간
        'click_date',        # 클릭 날짜
        'reward_point',      # 리워드 포인트
        'rwd_price',         # 리워드 가격
        'interaction_type',  # 상호작용 타입
        'click_key',         # 클릭 키
        'click_key_rwd',     # 클릭 키 리워드
        'click_key_info'     # 클릭 키 정보
    ]
    
    # 존재하는 컬럼만 선택
    available_columns = [col for col in required_columns if col in interactions.columns]
    missing_columns = [col for col in required_columns if col not in interactions.columns]
    
    if missing_columns:
        print(f"   ⚠️ 누락된 컬럼: {missing_columns}")
    
    final_interactions = interactions[available_columns].copy()
    
    # 7. 사용자 ID 매핑 (dvc_idx 우선, 없으면 user_ip 사용)
    print("7️⃣ 사용자 ID 매핑 중...")
    
    def set_user_device_id(row):
        """dvc_idx가 있으면 dvc_idx 사용, 없으면 user_ip 사용"""
        dvc_idx = row.get('dvc_idx')
        user_ip = row.get('user_ip')
        
        # dvc_idx가 있고 null이 아니면 dvc_idx 사용
        if pd.notna(dvc_idx) and str(dvc_idx).strip() != '':
            return str(dvc_idx).strip()
        # dvc_idx가 없거나 null이면 user_ip 사용
        elif pd.notna(user_ip) and str(user_ip).strip() != '':
            return str(user_ip).strip()
        else:
            return 'unknown'  # 둘 다 없으면 unknown
    
    final_interactions['user_device_id'] = final_interactions.apply(set_user_device_id, axis=1)
    
    # user_device_id가 'unknown'인 경우 제거하지 않고 유지
    print(f"   - 매핑 전 총 상호작용: {len(final_interactions):,}개")
    print(f"   - 매핑 후 총 상호작용: {len(final_interactions):,}개")
    
    # dvc_idx 사용 통계
    dvc_used = final_interactions[final_interactions['user_device_id'] != 'unknown'].shape[0]
    unknown_count = (final_interactions['user_device_id'] == 'unknown').sum()
    print(f"   - dvc_idx 또는 user_ip 사용: {dvc_used:,}개")
    print(f"   - unknown: {unknown_count:,}개")
    
    # 8. 데이터 정리
    print("8️⃣ 데이터 정리 중...")
    
    # user_ip 컬럼 제거 (user_device_id로 대체됨)
    if 'user_ip' in final_interactions.columns:
        final_interactions = final_interactions.drop(columns=['user_ip'])
        print("   - user_ip 컬럼 제거 완료")
    
    # dvc_idx 컬럼 제거 (user_device_id로 대체됨)
    if 'dvc_idx' in final_interactions.columns:
        final_interactions = final_interactions.drop(columns=['dvc_idx'])
        print("   - dvc_idx 컬럼 제거 완료")
    
    # 결측값 처리
    final_interactions['ads_type'] = final_interactions['ads_type'].fillna(0).astype(int)
    final_interactions['ads_category'] = final_interactions['ads_category'].fillna(0).astype(int)
    final_interactions['reward_point'] = final_interactions['reward_point'].fillna(0)
    final_interactions['rwd_price'] = final_interactions['rwd_price'].fillna(0)
    
    # 9. 통계 출력
    print("\n📈 상호작용 데이터 통계:")
    print(f"   - 총 상호작용 수: {len(final_interactions):,}")
    print(f"   - 고유 사용자 수: {final_interactions['user_device_id'].nunique():,}")
    print(f"   - 고유 광고 수: {final_interactions['ads_idx'].nunique():,}")
    
    print("\n📊 상호작용 타입별 분포:")
    type_counts = final_interactions['interaction_type'].value_counts()
    for interaction_type, count in type_counts.items():
        print(f"   - {interaction_type}: {count:,} ({count/len(final_interactions)*100:.1f}%)")
    
    print("\n💰 리워드 통계:")
    reward_interactions = final_interactions[final_interactions['reward_point'] > 0]
    print(f"   - 리워드가 있는 상호작용: {len(reward_interactions):,}")
    print(f"   - 총 리워드 포인트: {reward_interactions['reward_point'].sum():,.0f}")
    print(f"   - 평균 리워드 포인트: {reward_interactions['reward_point'].mean():.1f}")
    
    # 10. 파일 저장
    print("\n💾 파일 저장 중...")
    output_file = 'input/save/correct_interactions.csv'
    final_interactions.to_csv(output_file, index=False)
    print(f"   - 저장 완료: {output_file}")
    
    # 11. 샘플 데이터 출력
    print("\n🔍 샘플 데이터 (처음 5개):")
    print(final_interactions.head().to_string())
    
    return final_interactions

if __name__ == "__main__":
    try:
        interactions = create_correct_interactions()
        print("\n✅ 상호작용 데이터 생성 완료!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
