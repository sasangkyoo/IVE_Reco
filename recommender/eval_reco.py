#!/usr/bin/env python3
"""
추천 시스템 평가기 (Recommendation System Evaluator)

추천 결과를 평가하여 다양한 메트릭을 계산합니다.
- Precision@K, Recall@K, F1@K
- nDCG@K, HitRate@K
- Coverage, Bias Ratio

사용법:
    python recommender/eval_reco.py \
        --reco_csv topn_all_users.csv \
        --gt_csv ground_truth_interactions.csv \
        --ads_csv preprocessed/ads_profile.csv \
        --k_list 10 20 \
        --out_report metrics_summary.json
"""

import argparse
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
import warnings
warnings.filterwarnings('ignore')

# 타입과 카테고리 매핑 정의
TYPE_MAPPING = {
    1: "설치형",
    2: "실행형", 
    3: "참여형",
    4: "클릭형",
    5: "페북",
    6: "트위터", 
    7: "인스타",
    8: "노출형",
    9: "퀘스트",
    10: "유튜브",
    11: "네이버",
    12: "CPS(물건구매)"
}

CATEGORY_MAPPING = {
    0: "카테고리 선택안함",
    1: "앱(간편적립)",
    2: "경험하기(게임적립)/앱(간편적립) - cpi,cpe",
    3: "구독(간편적립)",
    4: "간편미션-퀘즈(간편적립)",
    5: "경험하기(게임적립) - cpa",
    6: "멀티보상(게임적립)",
    7: "금융(참여적립)",
    8: "무료참여(참여적립)",
    10: "유료참여(참여적립)",
    11: "쇼핑-상품별카테고리(쇼핑적립)",
    12: "제휴몰(쇼핑적립)",
    13: "간편미션(간편적립)"
}

def get_type_name(type_num: int) -> str:
    """타입 번호를 실제 이름으로 변환"""
    return TYPE_MAPPING.get(type_num, f"타입{type_num}")

def get_category_name(category_num: int) -> str:
    """카테고리 번호를 실제 이름으로 변환"""
    return CATEGORY_MAPPING.get(category_num, f"카테고리{category_num}")


class RecommendationEvaluator:
    """추천 시스템 평가기"""
    
    def __init__(self, 
                 reco_csv: str,
                 gt_csv: str,
                 ads_csv: str,
                 k_list: List[int],
                 out_report: str):
        """
        평가기 초기화
        
        Args:
            reco_csv: 추천 결과 CSV 파일
            gt_csv: 정답 데이터 CSV 파일
            ads_csv: 광고 프로필 CSV 파일
            k_list: 평가할 K 값 목록
            out_report: 출력 리포트 JSON 파일
        """
        self.reco_csv = reco_csv
        self.gt_csv = gt_csv
        self.ads_csv = ads_csv
        self.k_list = k_list
        self.out_report = out_report
        
        # 데이터 저장소
        self.recommendations = None
        self.ground_truth = None
        self.ads_profiles = None
        
    def load_data(self) -> None:
        """데이터 로드"""
        print("데이터 로딩 중...")
        
        # 추천 결과 로드
        print(f"추천 결과 로딩: {self.reco_csv}")
        self.recommendations = pd.read_csv(self.reco_csv, dtype={'user_device_id': str})
        print(f"추천 수: {len(self.recommendations):,}개")
        
        # 정답 데이터 로드
        print(f"정답 데이터 로딩: {self.gt_csv}")
        self.ground_truth = pd.read_csv(self.gt_csv, dtype={'user_device_id': str})
        print(f"정답 상호작용 수: {len(self.ground_truth):,}개")
        
        # 광고 프로필 로드
        print(f"광고 프로필 로딩: {self.ads_csv}")
        self.ads_profiles = pd.read_csv(self.ads_csv)
        print(f"광고 수: {len(self.ads_profiles):,}개")
        
        # dtype normalization (README 규칙: ads_idx를 Int64로 강제 변환)
        self.recommendations['ads_idx'] = pd.to_numeric(self.recommendations['ads_idx'], errors="coerce").astype("Int64")
        self.ground_truth['ads_idx'] = pd.to_numeric(self.ground_truth['ads_idx'], errors="coerce").astype("Int64")
        self.ads_profiles['ads_idx'] = pd.to_numeric(self.ads_profiles['ads_idx'], errors="coerce").astype("Int64")
        
        # inventory match rate 계산 (unique-level)
        inv_ids = set(self.ads_profiles['ads_idx'].dropna().unique().tolist())
        reco_ids = set(self.recommendations['ads_idx'].dropna().unique().tolist())
        self.inventory_match_rate = (len(reco_ids & inv_ids) / max(1, len(reco_ids))) if len(reco_ids) > 0 else 0.0
        print(f"Inventory match rate: {self.inventory_match_rate:.4f}")
        
        # 데이터 검증
        self._validate_data()
        
    def _validate_data(self) -> None:
        """데이터 검증"""
        # 필수 컬럼 확인
        required_reco_cols = ['user_device_id', 'ads_idx', 'rank']
        missing_cols = [col for col in required_reco_cols if col not in self.recommendations.columns]
        if missing_cols:
            raise ValueError(f"추천 결과에 필수 컬럼이 없습니다: {missing_cols}")
        
        required_gt_cols = ['user_device_id', 'ads_idx']
        missing_cols = [col for col in required_gt_cols if col not in self.ground_truth.columns]
        if missing_cols:
            raise ValueError(f"정답 데이터에 필수 컬럼이 없습니다: {missing_cols}")
        
        required_ads_cols = ['ads_idx', 'ads_category']
        missing_cols = [col for col in required_ads_cols if col not in self.ads_profiles.columns]
        if missing_cols:
            raise ValueError(f"광고 프로필에 필수 컬럼이 없습니다: {missing_cols}")
        
        # 중복 제거
        self.ground_truth = self.ground_truth.drop_duplicates(subset=['user_device_id', 'ads_idx'])
        print(f"중복 제거 후 정답 상호작용 수: {len(self.ground_truth):,}개")
        
    def _get_user_ground_truth(self) -> Dict[str, Set[int]]:
        """사용자별 정답 데이터"""
        user_gt = {}
        for _, row in self.ground_truth.iterrows():
            user_id = row['user_device_id']
            ads_idx = int(row['ads_idx'])
            if user_id not in user_gt:
                user_gt[user_id] = set()
            user_gt[user_id].add(ads_idx)
        return user_gt
    
    def _get_user_recommendations(self) -> Dict[str, List[int]]:
        """사용자별 추천 결과"""
        user_reco = {}
        for _, row in self.recommendations.iterrows():
            user_id = row['user_device_id']
            ads_idx = int(row['ads_idx'])
            rank = int(row['rank'])
            
            if user_id not in user_reco:
                user_reco[user_id] = {}
            user_reco[user_id][rank] = ads_idx
        
        # 순위별로 정렬된 리스트로 변환
        for user_id in user_reco:
            ranks = sorted(user_reco[user_id].keys())
            user_reco[user_id] = [user_reco[user_id][r] for r in ranks]
        
        return user_reco
    
    def _compute_precision_at_k(self, user_gt: Dict[str, Set[int]], 
                               user_reco: Dict[str, List[int]], k: int) -> float:
        """Precision@K 계산"""
        precisions = []
        
        for user_id, gt_items in user_gt.items():
            if user_id not in user_reco:
                continue
            
            reco_items = user_reco[user_id][:k]
            hits = len(set(reco_items) & gt_items)
            precision = hits / k if k > 0 else 0.0
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def _compute_recall_at_k(self, user_gt: Dict[str, Set[int]], 
                            user_reco: Dict[str, List[int]], k: int) -> float:
        """Recall@K 계산"""
        recalls = []
        
        for user_id, gt_items in user_gt.items():
            if len(gt_items) == 0:  # 정답이 없는 사용자는 제외
                continue
            
            if user_id not in user_reco:
                recalls.append(0.0)
                continue
            
            reco_items = user_reco[user_id][:k]
            hits = len(set(reco_items) & gt_items)
            recall = hits / len(gt_items)
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def _compute_f1_at_k(self, user_gt: Dict[str, Set[int]], 
                        user_reco: Dict[str, List[int]], k: int) -> float:
        """F1@K 계산"""
        f1_scores = []
        
        for user_id, gt_items in user_gt.items():
            if user_id not in user_reco:
                continue
            
            reco_items = user_reco[user_id][:k]
            hits = len(set(reco_items) & gt_items)
            
            precision = hits / k if k > 0 else 0.0
            recall = hits / len(gt_items) if len(gt_items) > 0 else 0.0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            f1_scores.append(f1)
        
        return np.mean(f1_scores) if f1_scores else 0.0
    
    def _compute_ndcg_at_k(self, user_gt: Dict[str, Set[int]], 
                          user_reco: Dict[str, List[int]], k: int) -> float:
        """nDCG@K 계산 (이진 관련성)"""
        ndcgs = []
        
        for user_id, gt_items in user_gt.items():
            if user_id not in user_reco:
                continue
            
            reco_items = user_reco[user_id][:k]
            
            # DCG@K 계산
            dcg = 0.0
            for i, ads_idx in enumerate(reco_items):
                if ads_idx in gt_items:
                    dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
            
            # IDCG@K 계산
            idcg = 0.0
            min_len = min(len(gt_items), k)
            for i in range(min_len):
                idcg += 1.0 / np.log2(i + 2)
            
            # nDCG@K 계산
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
            
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0
    
    def _compute_hit_rate_at_k(self, user_gt: Dict[str, Set[int]], 
                              user_reco: Dict[str, List[int]], k: int) -> float:
        """HitRate@K 계산"""
        hit_rates = []
        
        for user_id, gt_items in user_gt.items():
            if user_id not in user_reco:
                hit_rates.append(0.0)
                continue
            
            reco_items = user_reco[user_id][:k]
            hits = len(set(reco_items) & gt_items)
            hit_rate = 1.0 if hits > 0 else 0.0
            hit_rates.append(hit_rate)
        
        return np.mean(hit_rates) if hit_rates else 0.0
    
    def _compute_coverage(self) -> float:
        """Coverage 계산 (README 규칙: inner-join으로 인벤토리에 있는 광고만)"""
        # join before any coverage/bias math
        rec_inv = self.recommendations.merge(
            self.ads_profiles[['ads_idx']], 
            on='ads_idx', 
            how='inner'
        )
        
        if len(rec_inv) == 0:
            print("[WARN] No recommended ads matched inventory on ads_idx; set Coverage/Bias=0. Check ID mapping.")
            return 0.0
        
        # 추천된 고유 광고 수 (인벤토리에 있는 것만)
        unique_reco_ads = rec_inv['ads_idx'].nunique()
        
        # 전체 인벤토리의 고유 광고 수
        unique_inventory_ads = self.ads_profiles['ads_idx'].nunique()
        
        # 커버리지 계산
        coverage = unique_reco_ads / unique_inventory_ads if unique_inventory_ads > 0 else 0.0
        
        return coverage
    
    def _compute_bias_ratio(self) -> Dict:
        """Bias Ratio 계산 (README 규칙: inner-join으로 인벤토리에 있는 광고만)"""
        # join before any coverage/bias math
        rec_inv = self.recommendations.merge(
            self.ads_profiles[['ads_idx', 'ads_category']], 
            on='ads_idx', 
            how='inner',
            suffixes=('', '_inv')
        )
        
        if len(rec_inv) == 0:
            print("[WARN] No recommended ads matched inventory on ads_idx; set Coverage/Bias=0. Check ID mapping.")
            return {
                'macro_ratio': 0.0,
                'max_ratio': 0.0,
                'top_biased_categories': []
            }
        
        # 추천에서의 카테고리 분포 (인벤토리에 있는 광고만)
        rec_share = rec_inv['ads_category'].value_counts(normalize=True).sort_index()
        
        # 인벤토리의 카테고리 분포
        inv_share = self.ads_profiles['ads_category'].value_counts(normalize=True).sort_index()
        
        # 비율 계산 (0으로 나누기 방지)
        bias = (rec_share / inv_share).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        
        # 상위 5개 편향된 카테고리
        top_biased = [{"category": int(k), "ratio": float(v)} for k, v in bias.head(5).items()]
        
        return {
            'macro_ratio': float(bias.mean()) if len(bias) > 0 else 0.0,
            'max_ratio': float(bias.max()) if len(bias) > 0 else 0.0,
            'top_biased_categories': top_biased
        }
    
    def evaluate(self) -> Dict:
        """전체 평가 실행"""
        print("평가 시작...")
        
        # 사용자별 데이터 준비
        user_gt = self._get_user_ground_truth()
        user_reco = self._get_user_recommendations()
        
        # 평가할 사용자 수
        users_evaluated = len(set(user_gt.keys()) & set(user_reco.keys()))
        print(f"평가 대상 사용자 수: {users_evaluated:,}명")
        
        # 각 K에 대한 메트릭 계산
        precision_at_k = {}
        recall_at_k = {}
        f1_at_k = {}
        ndcg_at_k = {}
        hit_rate_at_k = {}
        
        for k in self.k_list:
            print(f"K={k} 메트릭 계산 중...")
            
            precision_at_k[str(k)] = self._compute_precision_at_k(user_gt, user_reco, k)
            recall_at_k[str(k)] = self._compute_recall_at_k(user_gt, user_reco, k)
            f1_at_k[str(k)] = self._compute_f1_at_k(user_gt, user_reco, k)
            ndcg_at_k[str(k)] = self._compute_ndcg_at_k(user_gt, user_reco, k)
            hit_rate_at_k[str(k)] = self._compute_hit_rate_at_k(user_gt, user_reco, k)
        
        # Coverage 계산
        print("Coverage 계산 중...")
        coverage = self._compute_coverage()
        
        # Bias Ratio 계산
        print("Bias Ratio 계산 중...")
        bias_ratio = self._compute_bias_ratio()
        
        # 결과 구성
        results = {
            'users_evaluated': users_evaluated,
            'K_list': self.k_list,
            'Precision@K': precision_at_k,
            'Recall@K': recall_at_k,
            'F1@K': f1_at_k,
            'nDCG@K': ndcg_at_k,
            'HitRate@K': hit_rate_at_k,
            'inventory_match_rate': self.inventory_match_rate,
            'Coverage': float(coverage),
            'BiasRatio': bias_ratio
        }
        
        return results
    
    def save_results(self, results: Dict) -> None:
        """결과 저장"""
        print(f"결과 저장 중: {self.out_report}")
        
        with open(self.out_report, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("결과 저장 완료")
    
    def print_summary(self, results: Dict) -> None:
        """결과 요약 출력"""
        print("\n=== 평가 결과 요약 ===")
        print(f"평가 대상 사용자 수: {results['users_evaluated']:,}명")
        print(f"Coverage: {results['Coverage']:.4f}")
        print(f"Bias Ratio (Macro): {results['BiasRatio']['macro_ratio']:.4f}")
        print(f"Bias Ratio (Max): {results['BiasRatio']['max_ratio']:.4f}")
        
        print("\n=== K별 메트릭 ===")
        for k in results['K_list']:
            k_str = str(k)
            print(f"K={k}:")
            print(f"  Precision: {results['Precision@K'][k_str]:.4f}")
            print(f"  Recall:    {results['Recall@K'][k_str]:.4f}")
            print(f"  F1:        {results['F1@K'][k_str]:.4f}")
            print(f"  nDCG:      {results['nDCG@K'][k_str]:.4f}")
            print(f"  HitRate:   {results['HitRate@K'][k_str]:.4f}")
        
        print("\n=== 상위 편향된 카테고리 ===")
        for i, cat_info in enumerate(results['BiasRatio']['top_biased_categories'], 1):
            print(f"{i}. 카테고리 {cat_info['category']}: {cat_info['ratio']:.4f}")
    
    def run(self) -> None:
        """평가 실행"""
        print("=== 추천 시스템 평가 시작 ===")
        
        # 데이터 로드
        self.load_data()
        
        # 평가 실행
        results = self.evaluate()
        
        # 결과 저장
        self.save_results(results)
        
        # 결과 출력
        self.print_summary(results)
        
        print("\n=== 평가 완료 ===")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='추천 시스템 평가기')
    
    # 필수 인자
    parser.add_argument('--reco_csv', required=True, help='추천 결과 CSV 파일')
    parser.add_argument('--gt_csv', required=True, help='정답 데이터 CSV 파일')
    parser.add_argument('--ads_csv', required=True, help='광고 프로필 CSV 파일')
    parser.add_argument('--k_list', nargs='+', type=int, required=True, help='평가할 K 값 목록')
    parser.add_argument('--out_report', required=True, help='출력 리포트 JSON 파일')
    
    args = parser.parse_args()
    
    # 평가기 생성 및 실행
    evaluator = RecommendationEvaluator(
        reco_csv=args.reco_csv,
        gt_csv=args.gt_csv,
        ads_csv=args.ads_csv,
        k_list=args.k_list,
        out_report=args.out_report
    )
    
    evaluator.run()


if __name__ == '__main__':
    main()
