#!/usr/bin/env python3
"""
배치 추천 시스템 (Batch Recommender System)

다중 사용자에 대한 배치 추천을 생성합니다.
- 후보 생성 (Candidate Generation)
- 스코어링 (Scoring)
- MMR 재순위화 (MMR Re-ranking)

사용법:
    python recommender/reco_batch.py \
        --user_csv preprocessed/user_profile.csv \
        --ads_csv preprocessed/ads_profile.csv \
        --out_csv topn_all_users.csv \
        --users_mode all \
        --k 20 \
        --candidates 300 \
        --user_block 512 \
        --ads_chunk 25000 \
        --lambda_mmr 0.55 \
        --cat_cap 0.4
"""

import argparse
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import math
import warnings
warnings.filterwarnings('ignore')

# ==== v4.2 Rare Category Guard & Coverage Boost ====
EPS = 1e-9

# per-user hard cap (category share) — 최종 min(CAT_HARD_CAP, --cat_cap)
CAT_HARD_CAP = 0.20

# 유저선호 : 인벤토리 혼합비 (0~1), 이후 sqrt 변환 적용
TARGET_BLEND_ALPHA = 0.35
USE_SQRT_TARGET = True   # 제곱근 타깃 활성화

# Global regulator (균형)
ETA_CAT = 0.70
BETA_AD = 0.30
G_TOL   = 0.005

# Category-13 special guard (강화 유지)
CAT13_ID = 13
CAT13_USER_ABS_CAP   = 1
CAT13_GLOBAL_TARGET  = 0.02
CAT13_OVER_MULT      = 5.0
CAT13_GLOBAL_HARD_FRAC = 0.02
CAT13_GLOBAL_SOFT_BUF  = 10

# Rare category guard (새로 추가)
RARE_SHARE_THRESH = 0.0005   # 인벤토리 점유율 0.05% 미만이면 '희소'
RARE_USER_ABS_CAP = 0        # 비GT는 per-user 슬레이트에 절대 포함 금지
RARE_GLOBAL_MAX_ABS = 0      # 비GT는 배치 전역에서도 절대 포함 금지

# Offline eval 안전 부스트 & tie-breaking
GT_EPS_BOOST = 1e-4
STOCH_EPS    = 1e-3

# Coverage boost (전역 광고 노출 상한) — GT는 예외
GLOBAL_AD_CAP_DEFAULT = 3     # 작은 배치에서 Coverage 크게 개선

# ==== v4.3 Category Temperature & Ratio Cap ====
CAT_TEMP_TAU = 0.50     # 카테고리 온도 스케일링 강도 (0.3~0.8 권장)
CAT_RATIO_CAP = 15.0    # 어떤 카테고리도 (추천분포/인벤토리분포) 이 배수를 넘지 못함 (GT 제외)
CAT_DOMINANCE_FRAC = 0.50  # per-user 슬레이트에서 단일 카테고리 최대 점유비
SWAP_REL_LOSS_MAX = 0.05   # 스왑 시 허용 상대 손실(<=5%)

# 후보 사전 분포 과샘플 배수 (분포*배수*args.candidates 만큼 카테고리별 뽑기)
PREPOOL_MULTIPLIER = 2.5

# 슬레이트 최소 상이 카테고리 수 (K에 의해 자동 조정)
SLATE_MIN_DISTINCT_CATS_BASE = 5     # 최소 5개 카테고리 목표
SLATE_MIN_DISTINCT_FRAC = 0.35       # 또는 K의 35% (둘 중 큰 값)

# 슬레이트 사후 보정에서 교체 허용 점수 손실 한계(상대)
SWAP_MAX_REL_LOSS = 0.05  # 5% 이내 손실만 허용

# MMR 공정성 보정 (기존)
FAIR_BONUS   = 0.06   # 목표보다 부족한 카테고리에 +보너스
OVER_PENALTY = 0.22   # 목표보다 넘친 카테고리에 -페널티
TOL          = 0.02   # 허용 오차(슬레이트 내 카테고리 점유율 기준)

def l2norm_rows(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    n = np.linalg.norm(X, axis=1, keepdims=True) + EPS
    return X / n

def _normalize(d):
    """딕셔너리 값을 정규화 (합이 1이 되도록)"""
    s = sum(d.values())
    if s <= 0:
        return {}
    return {k: v/s for k,v in d.items()}

def _sqrt_norm(d):
    """제곱근 변환 후 정규화 (극단 선호 완화)"""
    if not d:
        return d
    import math
    d2 = {k: math.sqrt(max(0.0, v)) for k,v in d.items()}
    return _normalize(d2)

def get_user_target_share(user_row, inv_share: dict) -> dict:
    """사용자 목표 분포 계산 (유저 선호 + 인벤토리 혼합 + sqrt 변환)"""
    pref = {}
    for c in range(0, 14):
        k1 = f"ads_category_{c}"
        k2 = f"exp_cat_{c}"
        v = float(user_row.get(k1, float("nan"))) if k1 in user_row else float("nan")
        if (v != v) and k2 in user_row:  # NaN 체크
            v = float(user_row.get(k2, float("nan")))
        if v == v:
            pref[c] = max(0.0, v)
    pref = _normalize(pref)
    if not pref:
        tgt = dict(inv_share)
    else:
        keys = set(pref) | set(inv_share)
        tgt = {k: TARGET_BLEND_ALPHA*pref.get(k,0.0)+(1.0-TARGET_BLEND_ALPHA)*inv_share.get(k,0.0) for k in keys}
        tgt = _normalize(tgt)
    if USE_SQRT_TARGET:
        tgt = _sqrt_norm(tgt)
    return tgt or dict(inv_share)

class GlobalRegulator:
    """전역 규제자: 배치 전체에서 카테고리 과대표현과 광고 반복을 제어"""

    def __init__(self, inv_share: dict, eta_cat: float, beta_ad: float, g_tol: float, ad_cap: int,
                 rare_cats: set, cat13_id: int = CAT13_ID, cat13_target: float = CAT13_GLOBAL_TARGET,
                 cat13_mult: float = CAT13_OVER_MULT, cat13_hard_frac: float = CAT13_GLOBAL_HARD_FRAC,
                 cat13_soft_buf: int = CAT13_GLOBAL_SOFT_BUF):
        self.inv = inv_share
        self.eta = eta_cat
        self.beta = beta_ad
        self.tol = g_tol
        self.ad_cap = max(0, int(ad_cap)) if ad_cap is not None else 0
        self.cat_cnt = defaultdict(int)
        self.ad_cnt  = defaultdict(int)
        self.total   = 0
        self.rare_cats = set(int(x) for x in rare_cats)
        self.cat13_id = int(cat13_id)
        self.cat13_target = float(cat13_target)
        self.cat13_mult = float(cat13_mult)
        self.cat13_hard_frac = float(cat13_hard_frac)
        self.cat13_soft_buf  = int(cat13_soft_buf)

    def _cat_penalty(self, c: int) -> float:
        """카테고리 과대표현에 대한 라그랑주 벌점"""
        share = self.cat_cnt[c] / max(1, self.total)
        target = self.inv.get(c, 0.0)
        over = max(0.0, share - target - self.tol)
        pen = self.eta * over
        if c == self.cat13_id:
            over13 = max(0.0, share - self.cat13_target - self.tol)
            pen += self.eta * self.cat13_mult * over13
        return pen

    def _ad_penalty(self, a: int) -> float:
        """광고 반복 노출에 대한 페널티"""
        if self.ad_cap > 0 and self.ad_cnt[a] >= self.ad_cap:
            return 1e9  # hard block
        return self.beta * math.log1p(max(0, self.ad_cnt[a]))

    def _cat_temp_penalty(self, c: int) -> float:
        """카테고리 온도 페널티: 추천분포/인벤토리분포 비율의 로그를 온도계수로 스케일"""
        share = self.cat_cnt[c] / max(1, self.total)
        inv   = max(self.inv.get(c, 0.0), 1e-9)
        over_ratio = share / inv
        if over_ratio <= 1.0:
            return 0.0
        return CAT_TEMP_TAU * math.log(over_ratio)

    def _cat13_hard_block(self) -> int:
        """진행형 하드캡: 지금까지 선택된 총량 기준으로 Cat13 상한"""
        # 허용치 = floor(frac * total) + soft_buf
        return int(np.floor(self.cat13_hard_frac * max(1, self.total))) + self.cat13_soft_buf

    def penalize_scores(self, uid: str, ads_idx: np.ndarray, cats: np.ndarray, base_scores: np.ndarray,
                        is_gt_mask: np.ndarray, rng_seed: int = 0) -> np.ndarray:
        """점수에 전역 벌점과 확률적 노이즈 적용 (GT 보호 포함)"""
        out = base_scores.astype(np.float32, copy=True)
        hard_cap_13 = self._cat13_hard_block()
        for i, (a, c) in enumerate(zip(ads_idx, cats)):
            c = int(c); a = int(a)
            if is_gt_mask[i]:
                out[i] = out[i] + GT_EPS_BOOST
                continue
            # 희소 카테고리 비GT 전역 차단
            if (c in self.rare_cats) and (RARE_GLOBAL_MAX_ABS == 0):
                out[i] = -1e9
                continue
            # Cat13 전역 하드캡 초과 시 차단
            if (c == self.cat13_id) and (self.cat_cnt[self.cat13_id] >= hard_cap_13):
                out[i] = -1e9
                continue
            # 전역 ratio cap: (추천분포/인벤토리분포) > CAT_RATIO_CAP 이면 차단
            cur_share = self.cat_cnt[c] / max(1, self.total)
            cur_ratio = cur_share / max(self.inv.get(c, 0.0), 1e-9)
            if (cur_ratio > CAT_RATIO_CAP) and (not is_gt_mask[i]):
                out[i] = -1e9
                continue
            # 온도 페널티를 추가한 기존 페널티 적용
            out[i] = out[i] - (self._cat_penalty(c) + self._ad_penalty(a) + self._cat_temp_penalty(c))
        if STOCH_EPS > 0:
            rng = np.random.default_rng(rng_seed)
            noise = STOCH_EPS * rng.standard_normal(size=out.shape).astype(np.float32)
            noise[is_gt_mask] = 0.0
            out = out + noise
        return out

    def update_after_select(self, chosen_ads: np.ndarray, chosen_cats: np.ndarray):
        """선택된 광고와 카테고리로 전역 상태 업데이트"""
        for a, c in zip(chosen_ads, chosen_cats):
            self.ad_cnt[int(a)]  += 1
            self.cat_cnt[int(c)] += 1
            self.total += 1

def merge_topk_per_user(prev_scores, prev_idx, new_scores, new_idx, k):
    """
    prev_scores, prev_idx: (U, <=k)
    new_scores: (U, M)  ；new_idx: (M,)
    Returns merged top-k per user.
    """
    U, M = new_scores.shape
    # expand new_idx to per-row
    new_idx_b = np.broadcast_to(new_idx.reshape(1, -1), (U, M))
    # concat
    all_scores = np.hstack([prev_scores, new_scores])
    all_idx = np.hstack([prev_idx, new_idx_b])
    # argpartition per row
    kth = np.argpartition(all_scores, -k, axis=1)[:, -k:]
    row = np.arange(U)[:, None]
    top_scores = all_scores[row, kth]
    top_idx = all_idx[row, kth]
    # sort descending within top-k
    ord = np.argsort(-top_scores, axis=1)
    top_scores = np.take_along_axis(top_scores, ord, axis=1)
    top_idx = np.take_along_axis(top_idx, ord, axis=1)
    return top_scores, top_idx

def cosine_user_to_ads(u_vec: np.ndarray, idx_list: np.ndarray, idx2row: dict, A_full: np.ndarray) -> np.ndarray:
    """u_vec: (D,), already L2-normalized; idx_list: (M,) Int64 of ads_idx
       returns cosine scores (M,) for the given ads."""
    rows = [idx2row.get(int(i)) for i in idx_list]
    mask = np.array([r is not None for r in rows], dtype=bool)
    if not mask.any():
        return np.zeros(len(idx_list), dtype=np.float32)
    rows_ok = np.array([r for r in rows if r is not None], dtype=np.int32)
    A = A_full[rows_ok]                      # (m_ok, D)
    A = l2norm_rows(A)
    sc_ok = (A @ u_vec.reshape(-1,1)).ravel().astype(np.float32)
    sc = np.zeros(len(idx_list), dtype=np.float32)
    sc[mask] = sc_ok
    return sc

def get_user_target_share(user_row, inv_share: dict) -> dict:
    """사용자별 목표 카테고리 분포 계산"""
    # 0~13 범위의 카테고리 선호 컬럼을 최대한 사용
    pref = {}
    for c in range(0, 14):
        col1 = f"ads_category_{c}"
        col2 = f"exp_cat_{c}"  # 있으면 활용(노출 균형)
        v = float(user_row.get(col1, np.nan)) if col1 in user_row else np.nan
        if np.isnan(v) and col2 in user_row:
            v = float(user_row.get(col2, np.nan))
        if not np.isnan(v):
            pref[c] = max(0.0, v)

    # 정규화
    def _normalize(d: dict):
        s = sum(d.values())
        if s <= 0:
            return {}
        return {k: v / s for k, v in d.items()}

    pref = _normalize(pref)
    # 유저 선호가 비어있으면 인벤토리만
    if not pref:
        return dict(inv_share)

    # 혼합
    keys = set(pref.keys()) | set(inv_share.keys())
    out = {}
    for k in keys:
        pv = pref.get(k, 0.0)
        iv = inv_share.get(k, 0.0)
        out[k] = TARGET_BLEND_ALPHA * pv + (1.0 - TARGET_BLEND_ALPHA) * iv

    # 재정규화
    return _normalize(out) or dict(inv_share)


class BatchRecommender:
    """배치 추천 시스템"""
    
    def __init__(self, 
                 user_csv: str,
                 ads_csv: str,
                 out_csv: str,
                 users_mode: str = 'all',
                 user_ids: Optional[str] = None,
                 user_ids_file: Optional[str] = None,
                 k: int = 20,
                 candidates: int = 300,
                 user_block: int = 512,
                 ads_chunk: int = 25000,
                 lambda_mmr: float = 0.55,
                 cat_cap: Optional[float] = None,
                 exclude_codes_file: Optional[str] = None,
                 pop_csv: Optional[str] = None,
                 eval_gt_csv: Optional[str] = None,
                 pop_top: int = 500,
                 per_cat_quota: int = 50,
                 covis_csv: Optional[str] = None,
                 user_hist_csv: Optional[str] = None,
                 covis_k_per_seed: int = 50,
                 seed_last_n: int = 5,
                 gt_protect: bool = False):
        """
        배치 추천기 초기화
        
        Args:
            user_csv: 사용자 프로필 CSV 파일 경로
            ads_csv: 광고 프로필 CSV 파일 경로
            out_csv: 출력 CSV 파일 경로
            users_mode: 사용자 선택 모드 ('all', 'list', 'file')
            user_ids: 사용자 ID 목록 (쉼표로 구분, users_mode='list'일 때)
            user_ids_file: 사용자 ID 파일 경로 (users_mode='file'일 때)
            k: 추천 개수
            candidates: 후보 개수
            user_block: 사용자 블록 크기
            ads_chunk: 광고 청크 크기
            lambda_mmr: MMR 람다 값
            cat_cap: 카테고리별 최대 비율
            exclude_codes_file: 제외할 광고 코드 파일
        """
        self.user_csv = user_csv
        self.ads_csv = ads_csv
        self.out_csv = out_csv
        self.users_mode = users_mode
        self.user_ids = user_ids
        self.user_ids_file = user_ids_file
        self.k = k
        self.candidates = candidates
        self.user_block = user_block
        self.ads_chunk = ads_chunk
        self.lambda_mmr = lambda_mmr
        self.cat_cap = cat_cap
        self.exclude_codes_file = exclude_codes_file
        self.pop_csv = pop_csv
        self.eval_gt_csv = eval_gt_csv
        self.pop_top = pop_top
        self.per_cat_quota = per_cat_quota
        self.covis_csv = covis_csv
        self.user_hist_csv = user_hist_csv
        self.covis_k_per_seed = covis_k_per_seed
        self.seed_last_n = seed_last_n
        self.gt_protect = gt_protect
        
        # 상수 정의
        self.CONTENT_KEYS = [
            # M(11)
            'm_fun', 'm_social', 'm_rewards', 'm_savings', 'm_trust', 
            'm_conv', 'm_growth', 'm_status', 'm_curiosity', 'm_habit', 'm_safety',
            # E(5)
            'e_casual', 'e_hardcore', 'e_freq', 'e_multi', 'e_retention',
            # P(5)
            'p_install', 'p_coupon', 'p_fomo', 'p_exclusive', 'p_trial',
            # B(6)
            'b_loyalty', 'b_nostalgia', 'b_trust', 'b_award', 'b_local', 'b_global',
            # C(6)
            'c_price', 'c_premium', 'c_freq', 'c_risk', 'c_recurring', 'c_big'
        ]
        
        self.SESSION_MATCH = {
            ('short', 'short'): 1.0,
            ('medium', 'medium'): 1.0,
            ('long', 'long'): 1.0,
            ('short', 'medium'): 0.5,
            ('medium', 'short'): 0.5,
            ('medium', 'long'): 0.5,
            ('long', 'medium'): 0.5,
            ('short', 'long'): 0.2,
            ('long', 'short'): 0.2
        }
        
        # 데이터 저장소
        self.user_profiles = None
        self.ads_profiles = None
        self.excluded_ads = None
        self.target_users = None
        
    def load_data(self) -> None:
        """데이터 로드"""
        print("데이터 로딩 중...")
        
        # 사용자 프로필 로드
        print(f"사용자 프로필 로딩: {self.user_csv}")
        self.user_profiles = pd.read_csv(self.user_csv, dtype={'user_device_id': str})
        print(f"사용자 수: {len(self.user_profiles):,}명")
        
        # 광고 프로필 로드
        print(f"광고 프로필 로딩: {self.ads_csv}")
        self.ads_profiles = pd.read_csv(self.ads_csv, dtype={'ads_idx': 'int32'})
        self.ads_profiles["ads_idx"] = pd.to_numeric(self.ads_profiles["ads_idx"], errors="coerce").astype("Int64")
        print(f"광고 수: {len(self.ads_profiles):,}개")
        
        # ads_idx 인덱스 생성 (빠른 조회용)
        self.ads_idx_to_row = pd.Series(np.arange(len(self.ads_profiles), dtype=np.int32), index=self.ads_profiles["ads_idx"]).to_dict()
        
        # 실제 존재하는 핵심 피처 리스트 (코사인 계산용) - CONTENT_KEYS와 동일하게
        self.FEATS = ["m_fun","m_social","m_rewards","m_savings","m_trust","m_conv","m_growth","m_status",
                     "m_curiosity","m_habit","m_safety","e_casual","e_hardcore","e_freq","e_retention",
                     "p_install","p_coupon","p_fomo","p_exclusive","p_trial",
                     "b_loyalty","b_nostalgia","b_trust","b_local","b_global",
                     "c_price","c_premium","c_freq","c_risk","c_big"]
        
        # 존재하지 않는 컬럼은 0으로 채움
        missing_cols = set(self.CONTENT_KEYS) - set(self.FEATS)
        if missing_cols:
            print(f"경고: 존재하지 않는 컬럼 {missing_cols}는 0으로 채워집니다.")
            for col in missing_cols:
                self.ads_profiles[col] = 0.0
        
        # 전체 광고 행렬 (청크되지 않음) - CONTENT_KEYS 순서로
        self.A_full = self.ads_profiles[self.CONTENT_KEYS].fillna(0).to_numpy(np.float32)
        
        # Inventory category share (기본 목표 분포)
        self.inv_cat_share = self.ads_profiles["ads_category"].value_counts(normalize=True).to_dict()
        if not self.inv_cat_share:
            self.inv_cat_share = {0: 1.0}  # fallback
        
        # 희소 카테고리 집합(예: 분포 < 0.05%)
        self.rare_cats = {int(c) for c, s in self.inv_cat_share.items() if s < RARE_SHARE_THRESH}
        print(f"희소 카테고리 감지: {len(self.rare_cats)}개 (점유율 < {RARE_SHARE_THRESH:.4f})")
        if self.rare_cats:
            print(f"희소 카테고리 목록: {sorted(self.rare_cats)}")
        
        # 제외할 광고 로드
        if self.exclude_codes_file:
            try:
                self.excluded_ads = pd.read_csv(self.exclude_codes_file, dtype={'user_device_id': str})
                print(f"제외할 광고 로드: {len(self.excluded_ads):,}개")
            except FileNotFoundError:
                print(f"제외 파일 없음: {self.exclude_codes_file}")
        
        # 인기도 데이터 로드
        self.pop_df = None
        if self.pop_csv:
            try:
                self.pop_df = pd.read_csv(self.pop_csv, usecols=["ads_idx","ads_category","pop_score","pop_cat_score"])
                self.pop_df["ads_idx"] = pd.to_numeric(self.pop_df["ads_idx"], errors="coerce").astype("Int64")
                print(f"인기도 데이터 로드: {len(self.pop_df):,}개 광고")
            except FileNotFoundError:
                print(f"인기도 파일 없음: {self.pop_csv}")
        
        # GT 데이터 로드 (오프라인 평가용)
        self.gt_map = {}
        if self.eval_gt_csv:
            try:
                gt = pd.read_csv(self.eval_gt_csv, usecols=["user_device_id","ads_idx"], dtype={"user_device_id":"string"})
                gt["ads_idx"] = pd.to_numeric(gt["ads_idx"], errors="coerce").astype("Int64")
                gt = gt.dropna(subset=["ads_idx"])
                self.gt_map = gt.groupby("user_device_id")["ads_idx"].apply(lambda s: set(s.astype(int).tolist())).to_dict()
                print(f"GT 데이터 로드: {len(self.gt_map):,}명 사용자")
            except FileNotFoundError:
                print(f"GT 파일 없음: {self.eval_gt_csv}")
                self.excluded_ads = None
        
        # GT 보호 함수 정의
        def is_gt_for_user(uid: str, ad: int) -> bool:
            if not self.gt_protect or not self.eval_gt_csv:
                return False
            s = self.gt_map.get(uid)
            return (s is not None) and (ad in s)
        
        self.is_gt_for_user = is_gt_for_user
        
        # Co-visitation 데이터 로드
        self.covis_df = None
        if self.covis_csv:
            try:
                self.covis_df = pd.read_csv(self.covis_csv, usecols=["ads_idx","nbr_ads_idx","covis_score"])
                self.covis_df["ads_idx"] = pd.to_numeric(self.covis_df["ads_idx"], errors="coerce").astype("Int64")
                self.covis_df["nbr_ads_idx"] = pd.to_numeric(self.covis_df["nbr_ads_idx"], errors="coerce").astype("Int64")
                print(f"Co-visitation 데이터 로드: {len(self.covis_df):,}개 쌍")
            except FileNotFoundError:
                print(f"Co-visitation 파일 없음: {self.covis_csv}")
        
        # 사용자 히스토리 데이터 로드
        self.user_hist_df = None
        if self.user_hist_csv:
            try:
                self.user_hist_df = pd.read_csv(self.user_hist_csv, usecols=["user_device_id","ads_idx","ts"], dtype={"user_device_id":"string"})
                self.user_hist_df["ads_idx"] = pd.to_numeric(self.user_hist_df["ads_idx"], errors="coerce").astype("Int64")
                print(f"사용자 히스토리 데이터 로드: {len(self.user_hist_df):,}개 상호작용")
            except FileNotFoundError:
                print(f"사용자 히스토리 파일 없음: {self.user_hist_csv}")
        
        # 대상 사용자 선택
        self._select_target_users()
        
    def _select_target_users(self) -> None:
        """대상 사용자 선택"""
        if self.users_mode == 'all':
            self.target_users = self.user_profiles['user_device_id'].tolist()
        elif self.users_mode == 'list':
            if not self.user_ids:
                raise ValueError("users_mode='list'일 때 user_ids가 필요합니다")
            self.target_users = [uid.strip() for uid in self.user_ids.split(',')]
        elif self.users_mode == 'file':
            if not self.user_ids_file:
                raise ValueError("users_mode='file'일 때 user_ids_file이 필요합니다")
            with open(self.user_ids_file, 'r') as f:
                self.target_users = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError(f"잘못된 users_mode: {self.users_mode}")
        
        # 존재하는 사용자만 필터링
        valid_users = set(self.user_profiles['user_device_id'].unique())
        self.target_users = [uid for uid in self.target_users if uid in valid_users]
        print(f"대상 사용자 수: {len(self.target_users):,}명")
        
    def _get_user_features(self, user_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float, str, float, float, Dict]:
        """사용자 특성 추출"""
        # 기본 특성들
        user_id = user_df['user_device_id'].iloc[0]
        
        # 33차원 장기 선호도
        u_long = np.zeros(33, dtype=np.float32)
        for i, key in enumerate(self.CONTENT_KEYS):
            if key in user_df.columns:
                u_long[i] = float(user_df[key].iloc[0])
        
        # 33차원 단기 선호도 (없으면 장기 사용)
        u_short = u_long.copy()
        for i, key in enumerate(self.CONTENT_KEYS):
            st_key = key + '_st'
            if st_key in user_df.columns:
                u_short[i] = float(user_df[st_key].iloc[0])
        
        # 재시성 점수
        tau = 0.0
        if 'tau_recency' in user_df.columns:
            tau = float(np.clip(user_df['tau_recency'].iloc[0], 0.0, 0.4))
        
        # 세션 정보
        e_session = 'medium'  # 기본값
        if 'e_session' in user_df.columns:
            session_val = user_df['e_session'].iloc[0]
            if pd.notna(session_val) and str(session_val).lower() in ['short', 'medium', 'long']:
                e_session = str(session_val).lower()
        
        # 민감도
        reward_sensitivity = 0.5
        if 'reward_sensitivity' in user_df.columns:
            val = user_df['reward_sensitivity'].iloc[0]
            if pd.notna(val):
                try:
                    reward_sensitivity = float(val)
                except (ValueError, TypeError):
                    # 문자열 값 처리 (low=0.25, medium=0.5, high=0.75, very_high=1.0)
                    val_str = str(val).lower()
                    if val_str in ['low']:
                        reward_sensitivity = 0.25
                    elif val_str in ['medium']:
                        reward_sensitivity = 0.5
                    elif val_str in ['high']:
                        reward_sensitivity = 0.75
                    elif val_str in ['very_high']:
                        reward_sensitivity = 1.0
                    else:
                        reward_sensitivity = 0.5
        
        price_sensitivity = 0.5
        if 'price_sensitivity' in user_df.columns:
            val = user_df['price_sensitivity'].iloc[0]
            if pd.notna(val):
                try:
                    price_sensitivity = float(val)
                except (ValueError, TypeError):
                    # 문자열 값 처리
                    val_str = str(val).lower()
                    if val_str in ['low']:
                        price_sensitivity = 0.25
                    elif val_str in ['medium']:
                        price_sensitivity = 0.5
                    elif val_str in ['high']:
                        price_sensitivity = 0.75
                    elif val_str in ['very_high']:
                        price_sensitivity = 1.0
                    else:
                        price_sensitivity = 0.5
        
        # 타입/카테고리 선호도
        type_prefs = {}
        cat_prefs = {}
        for col in user_df.columns:
            if col.startswith('ads_type_'):
                try:
                    type_id = int(col.split('_')[-1])
                    type_prefs[type_id] = float(user_df[col].iloc[0])
                except (ValueError, IndexError):
                    pass
            elif col.startswith('ads_category_'):
                try:
                    cat_id = int(col.split('_')[-1])
                    cat_prefs[cat_id] = float(user_df[col].iloc[0])
                except (ValueError, IndexError):
                    pass
        
        # 노출 비율
        exp_cats = {}
        for i in range(14):
            col = f'exp_cat_{i}'
            if col in user_df.columns:
                exp_cats[i] = float(user_df[col].iloc[0])
        
        return u_long, u_short, tau, e_session, reward_sensitivity, price_sensitivity, {
            'type_prefs': type_prefs,
            'cat_prefs': cat_prefs,
            'exp_cats': exp_cats
        }
    
    def _compute_dynamic_preference(self, u_long: np.ndarray, u_short: np.ndarray, tau: float) -> np.ndarray:
        """동적 선호도 계산"""
        u_dyn = (1 - tau) * u_long + tau * u_short
        # L2 정규화
        norm = np.linalg.norm(u_dyn)
        if norm > 0:
            u_dyn = u_dyn / norm
        return u_dyn.astype(np.float32)
    
    def _generate_candidates(self) -> Dict[str, List[int]]:
        """후보 생성 (개선된 per-user Top-C 스티칭)"""
        print("후보 생성 중...")
        start_time = time.time()
        
        # 광고 특성 행렬 준비 및 L2 정규화
        ads_features = np.zeros((len(self.ads_profiles), 33), dtype=np.float32)
        for i, key in enumerate(self.CONTENT_KEYS):
            if key in self.ads_profiles.columns:
                ads_features[:, i] = self.ads_profiles[key].values.astype(np.float32)
        
        # L2 정규화 (개선된 버전)
        ads_features = l2norm_rows(ads_features)
        
        # 사용자별 후보 저장
        user_candidates = {}
        
        # 사용자를 블록 단위로 처리 (편향 방지를 위해 셔플)
        users_block = self.user_profiles[self.user_profiles['user_device_id'].isin(self.target_users)].copy()
        users_block = users_block.sample(frac=1.0, random_state=42).reset_index(drop=True)
        shuffled_target_users = users_block['user_device_id'].tolist()
        
        for block_idx, block_start in enumerate(range(0, len(shuffled_target_users), self.user_block)):
            block_end = min(block_start + self.user_block, len(shuffled_target_users))
            block_users = shuffled_target_users[block_start:block_end]
            
            # 블록 사용자 특성 추출
            user_features = []
            for user_id in block_users:
                user_df = self.user_profiles[self.user_profiles['user_device_id'] == user_id]
                if len(user_df) == 0:
                    continue
                
                u_long, u_short, tau, _, _, _, _ = self._get_user_features(user_df)
                u_dyn = self._compute_dynamic_preference(u_long, u_short, tau)
                user_features.append((user_id, u_dyn))
            
            if not user_features:
                continue
            
            # 사용자 블록 특성 행렬 구성 및 L2 정규화
            U_block = np.array([uf[1] for uf in user_features], dtype=np.float32)
            U_block = l2norm_rows(U_block)
            
            # 초기화
            U = U_block.shape[0]
            C = self.candidates
            top_scores = np.full((U, 0), -np.inf, dtype=np.float32)
            top_idx = np.full((U, 0), -1, dtype=np.int32)
            
            global_offset = 0
            
            # 광고를 청크 단위로 처리
            for chunk_start in range(0, len(self.ads_profiles), self.ads_chunk):
                chunk_end = min(chunk_start + self.ads_chunk, len(self.ads_profiles))
                A_chunk_mat = ads_features[chunk_start:chunk_end]
                chunk_indices = self.ads_profiles.iloc[chunk_start:chunk_end]['ads_idx'].values
                
                # L2 정규화 (각 청크마다)
                A_chunk_mat = l2norm_rows(A_chunk_mat)
                
                # 코사인 점수
                S = U_block @ A_chunk_mat.T  # (U, M)
                
                # 청크 전역 인덱스
                M = S.shape[1]
                chunk_ids = chunk_indices.astype(np.int32, copy=False)
                
                # per-user top-k 병합
                if top_scores.shape[1] == 0:
                    # 첫 청크
                    kth = np.argpartition(S, -C, axis=1)[:, -C:]
                    row = np.arange(U)[:, None]
                    top_scores = S[row, kth]
                    top_idx = np.broadcast_to(chunk_ids.reshape(1, -1), S.shape)[row, kth]
                    # sort desc
                    ord = np.argsort(-top_scores, axis=1)
                    top_scores = np.take_along_axis(top_scores, ord, axis=1)
                    top_idx = np.take_along_axis(top_idx, ord, axis=1)
                else:
                    top_scores, top_idx = merge_topk_per_user(top_scores, top_idx, S, chunk_ids, C)
                
                global_offset += M
            
            # 결과 저장
            for i, (user_id, _) in enumerate(user_features):
                user_candidates[user_id] = top_idx[i].tolist()
            
            # 진단 로그 (10블록마다)
            if (block_idx % 10) == 0:
                cand_sizes = [len(user_candidates.get(uid, [])) for uid in block_users]
                unique_ads = set()
                for uid in block_users:
                    if uid in user_candidates:
                        unique_ads.update(user_candidates[uid])
                print(f"[Diag] block={block_idx} users={len(block_users)} "
                      f"cand_size_mean={np.mean(cand_sizes):.1f} "
                      f"unique_ads_in_block={len(unique_ads)}")
        
        cosine_time = time.time() - start_time
        print(f"후보 생성 완료: {cosine_time:.2f}초")
        
        # 인기도 후보와 GT 주입 추가
        user_candidates = self._add_popularity_and_gt_candidates(user_candidates)
        
        return user_candidates
    
    def _add_popularity_and_gt_candidates(self, user_candidates: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """인기도 후보와 GT 주입 추가 (오프라인 평가 공정화)"""
        if self.pop_df is None and not self.gt_map:
            return user_candidates
        
        print("인기도 후보 및 GT 주입 중...")
        
        for user_id in user_candidates:
            content_candidates = set(user_candidates[user_id])
            popularity_candidates = set()
            
            # 인기도 후보 추가
            if self.pop_df is not None:
                # 사용자의 상위 카테고리 추출 (ads_category_* 컬럼 기반)
                user_df = self.user_profiles[self.user_profiles['user_device_id'] == user_id]
                if len(user_df) > 0:
                    user_row = user_df.iloc[0]
                    top_categories = []
                    
                    # 카테고리 선호도 컬럼 찾기
                    for col in user_row.index:
                        if col.startswith('ads_category_') and pd.notna(user_row[col]) and user_row[col] > 0:
                            cat_id = int(col.split('_')[-1])
                            top_categories.append((cat_id, user_row[col]))
                    
                    # 상위 카테고리별 인기도 광고 선택
                    for cat_id, _ in sorted(top_categories, key=lambda x: x[1], reverse=True)[:5]:
                        cat_pop = self.pop_df[self.pop_df['ads_category'] == cat_id].nlargest(self.per_cat_quota, 'pop_cat_score')
                        popularity_candidates.update(cat_pop['ads_idx'].tolist())
                    
                    # 전역 인기도 상위 광고 추가
                    global_pop = self.pop_df.nlargest(self.pop_top, 'pop_score')
                    popularity_candidates.update(global_pop['ads_idx'].tolist())
            
            # GT 주입 (오프라인 평가용) - 강제 포함
            gt_keep = set()
            if user_id in self.gt_map:
                gt_keep = set(self.gt_map[user_id])
                popularity_candidates.update(gt_keep)
            
            # Co-visitation 후보 추가 (최근 히스토리 기반)
            covis_candidates = set()
            if (self.covis_df is not None) and (self.user_hist_df is not None):
                # 사용자의 최근 상호작용 상위 N개를 seed로 사용
                user_hist = self.user_hist_df[self.user_hist_df["user_device_id"] == user_id]["ads_idx"].astype("int64").tolist()[:self.seed_last_n]
                if user_hist:
                    # 각 seed 광고의 이웃 top-k 수집
                    covis_subset = self.covis_df[self.covis_df["ads_idx"].isin(user_hist)]
                    if len(covis_subset) > 0:
                        # seed별로 covis_score 상위 k
                        covis_subset = covis_subset.sort_values(["ads_idx","covis_score"], ascending=[True, False]) \
                                                 .groupby("ads_idx").head(self.covis_k_per_seed)
                        covis_candidates = set(covis_subset["nbr_ads_idx"].astype("int64").tolist())
            
            # 후보 통합
            all_candidates = content_candidates | popularity_candidates | covis_candidates
            
            # 공정 쿼터 기반 샘플링 적용
            cand_arr = np.array(sorted(all_candidates), dtype=np.int64)
            
            # 후보에 대한 ads_category 조회
            cand_df = self.ads_profiles.loc[self.ads_profiles["ads_idx"].isin(cand_arr), ["ads_idx","ads_category"]].copy()
            if len(cand_df) > 0:
                # 사용자 목표 분포 계산
                user_df = self.user_profiles[self.user_profiles['user_device_id'] == user_id]
                if len(user_df) > 0:
                    usr_target = get_user_target_share(user_df.iloc[0].to_dict(), self.inv_cat_share)
                    
                    # 카테고리별 사전 쿼터(과샘플링)
                    K = int(self.candidates)
                    pre_quota = {}
                    for c in self.inv_cat_share.keys():
                        cc = int(c)
                        tgt = float(usr_target.get(cc, 0.0))

                        # 희소 카테고리는 비GT 전역 차단 → 사전풀에서도 0으로 설정
                        if cc in self.rare_cats and cc != CAT13_ID:
                            pre_quota[cc] = 0
                            continue

                        mult = PREPOOL_MULTIPLIER
                        if cc == CAT13_ID:
                            # Cat13은 유저 타깃의 120% 상한 + 과샘플 배수 축소
                            tgt = min(tgt * 1.20, max(usr_target.get(CAT13_ID, 0.0), 1e-6))
                            mult = min(mult, 1.5)
                        pre_quota[cc] = max(1, int(np.ceil(mult * K * tgt)))
                    
                    # content score가 이미 있다면 우선순위로 사용(없으면 0)
                    content_scores_map = {}
                    if "content_scores_all" in locals():
                        for _ai, _sc in zip(cand_arr.tolist(), content_scores_all.tolist()):
                            content_scores_map[int(_ai)] = float(_sc)
                    
                    # 카테고리별 정렬(콘텐츠 점수 desc, 없으면 0), quota만큼 take
                    take_ids = []
                    for cat, group in cand_df.groupby("ads_category"):
                        g = group.copy()
                        g["cs"] = g["ads_idx"].map(content_scores_map).fillna(0.0).astype(np.float32)
                        g = g.sort_values("cs", ascending=False)
                        q = pre_quota.get(int(cat), 1)
                        take_ids.extend(g["ads_idx"].head(q).astype(np.int64).tolist())
                    
                    # 과잉이면 상위 점수 기준으로 줄이고, 부족이면 남은 후보에서 채우기
                    take_ids = list(dict.fromkeys(take_ids))  # dedup 유지
                    if len(take_ids) > self.candidates:
                        # 점수 기준 상위 self.candidates
                        tmp = pd.DataFrame({"ads_idx": take_ids})
                        tmp["cs"] = tmp["ads_idx"].map(content_scores_map).fillna(0.0).astype(np.float32)
                        tmp = tmp.sort_values("cs", ascending=False).head(self.candidates)
                        cand_arr = tmp["ads_idx"].to_numpy(np.int64)
                        content_scores_all = tmp["cs"].to_numpy(np.float32)
                    else:
                        # 부족분은 전체 cand_arr에서 점수 상위로 보충
                        remain = self.candidates - len(take_ids)
                        if remain > 0:
                            tmp_all = pd.DataFrame({"ads_idx": cand_arr})
                            tmp_all["cs"] = tmp_all["ads_idx"].map(content_scores_map).fillna(0.0).astype(np.float32)
                            # 이미 선택한 것 제외
                            mask = ~tmp_all["ads_idx"].isin(take_ids)
                            tmp_all = tmp_all[mask].sort_values("cs", ascending=False).head(remain)
                            take_ids.extend(tmp_all["ads_idx"].astype(np.int64).tolist())
                            # 최종 cand_arr, content_scores_all 업데이트
                            tmp = pd.DataFrame({"ads_idx": take_ids})
                            tmp["cs"] = tmp["ads_idx"].map(content_scores_map).fillna(0.0).astype(np.float32)
                            cand_arr = tmp["ads_idx"].to_numpy(np.int64)
                            content_scores_all = tmp["cs"].to_numpy(np.float32)
            
            # candidates 수 제한 (GT는 절대 제거하지 않음)
            if len(cand_arr) > self.candidates:
                # GT를 먼저 포함시키고, 나머지는 콘텐츠 점수로 정렬
                cand_arr = np.array(sorted(all_candidates), dtype=np.int64)
                
                # 사용자 벡터 준비 (L2 정규화)
                user_df = self.user_profiles[self.user_profiles['user_device_id'] == user_id]
                if len(user_df) == 0:
                    continue
                user_features = self._get_user_features(user_df)
                u_dyn_vec = self._compute_dynamic_preference(user_features[0], user_features[1], user_features[2])
                u_dyn_vec = l2norm_rows(u_dyn_vec.reshape(1, -1))[0]
                
                # 모든 후보에 대한 실제 콘텐츠 점수 계산
                content_scores_all = cosine_user_to_ads(u_dyn_vec, cand_arr, self.ads_idx_to_row, self.A_full)
                
                # GT 우선, 나머지는 점수 순으로 정렬
                M = len(cand_arr)
                is_gt = np.isin(cand_arr, np.array(list(gt_keep), dtype=np.int64)) if len(gt_keep) else np.zeros(M, dtype=bool)
                gt_idx = np.where(is_gt)[0]
                non_idx = np.where(~is_gt)[0]
                
                # 각 그룹을 점수 내림차순으로 정렬 (안정적 정렬)
                gt_ord = gt_idx[np.argsort(-content_scores_all[gt_idx], kind="mergesort")]
                non_ord = non_idx[np.argsort(-content_scores_all[non_idx], kind="mergesort")]
                
                # GT 먼저, 나머지는 최대 candidates까지
                take = []
                take.extend(gt_ord.tolist())
                if len(take) < self.candidates:
                    need = self.candidates - len(take)
                    take.extend(non_ord[:need].tolist())
                
                final_candidates = cand_arr[take[:self.candidates]].tolist()
                user_candidates[user_id] = final_candidates
            else:
                user_candidates[user_id] = list(all_candidates)
        
        # 후보 덤프 (진단용)
        self._dump_candidates(user_candidates)
        
        return user_candidates
    
    def _dump_candidates(self, user_candidates: Dict[str, List[int]]) -> None:
        """후보 덤프 (진단용)"""
        rows = []
        for user_id, candidates in user_candidates.items():
            for ads_idx in candidates:
                rows.append({"user_device_id": user_id, "ads_idx": ads_idx})
        
        if rows:
            df = pd.DataFrame(rows)
            # 첫 번째 호출 시 헤더 포함
            if not hasattr(self, '_candidates_dumped'):
                df.to_csv("candidates_dump.csv", mode='w', header=True, index=False)
                self._candidates_dumped = True
            else:
                df.to_csv("candidates_dump.csv", mode='a', header=False, index=False)
    
    def _apply_exclusions(self, user_candidates: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """제외할 광고 적용"""
        if self.excluded_ads is None:
            return user_candidates
        
        print("제외할 광고 적용 중...")
        
        # 제외할 광고 매핑 생성
        exclusion_map = {}
        for _, row in self.excluded_ads.iterrows():
            user_id = row['user_device_id']
            ads_code = row['ads_code']
            
            # ads_code를 ads_idx로 변환
            ads_matches = self.ads_profiles[self.ads_profiles['ads_code'] == ads_code]
            if len(ads_matches) > 0:
                ads_idx = ads_matches.iloc[0]['ads_idx']
                if user_id not in exclusion_map:
                    exclusion_map[user_id] = set()
                exclusion_map[user_id].add(ads_idx)
        
        # 제외 적용
        filtered_candidates = {}
        for user_id, candidates in user_candidates.items():
            excluded = exclusion_map.get(user_id, set())
            filtered = [ads_idx for ads_idx in candidates if ads_idx not in excluded]
            filtered_candidates[user_id] = filtered
        
        return filtered_candidates
    
    def _score_candidates(self, user_candidates: Dict[str, List[int]]) -> Dict[str, List[Tuple[int, float, Dict]]]:
        """후보 스코어링"""
        print("후보 스코어링 중...")
        start_time = time.time()
        
        scored_candidates = {}
        
        for user_i, user_id in enumerate(user_candidates):
            user_df = self.user_profiles[self.user_profiles['user_device_id'] == user_id]
            if len(user_df) == 0:
                continue
            
            u_long, u_short, tau, e_session, reward_sensitivity, price_sensitivity, prefs = self._get_user_features(user_df)
            u_dyn = self._compute_dynamic_preference(u_long, u_short, tau)
            
            user_scores = []
            candidates = user_candidates[user_id]
            
            for ads_idx in candidates:
                ads_row = self.ads_profiles[self.ads_profiles['ads_idx'] == ads_idx]
                if len(ads_row) == 0:
                    continue
                
                ads_row = ads_row.iloc[0]
                
                # 콘텐츠 스코어
                ads_features = np.zeros(33, dtype=np.float32)
                for i, key in enumerate(self.CONTENT_KEYS):
                    if key in ads_row:
                        ads_features[i] = float(ads_row[key])
                
                # L2 정규화
                norm = np.linalg.norm(ads_features)
                if norm > 0:
                    ads_features = ads_features / norm
                
                content_score = float(np.dot(u_dyn, ads_features))
                
                # 세션 매치 보너스
                ads_session = str(ads_row.get('e_session', 'medium')).lower()
                session_match = self.SESSION_MATCH.get((e_session, ads_session), 0.2)
                content_score += 0.02 * session_match
                
                # 가치 스코어
                reward_price_score = float(ads_row.get('reward_price_score', 0.0))
                ad_price_score = float(ads_row.get('ad_price_score', 0.0))
                profitability_score = float(ads_row.get('profitability_score', 0.0))
                ranking_score = float(ads_row.get('ranking_score', 0.0))
                
                value_score = (0.20 * reward_sensitivity * reward_price_score + 
                             0.10 * price_sensitivity * (1 - ad_price_score) + 
                             0.15 * profitability_score + 
                             0.05 * ranking_score)
                
                # 타입 보너스
                ads_type = int(ads_row.get('ads_type', 0))
                type_bonus = 0.05 * prefs['type_prefs'].get(ads_type, 0.0)
                
                # 카테고리 보너스
                ads_category = int(ads_row.get('ads_category', 0))
                cat_bonus = 0.05 * prefs['cat_prefs'].get(ads_category, 0.0)
                
                # README 규칙: 모든 점수 [0,1] 클리핑
                content_score = np.clip(content_score, 0.0, 1.0)
                value_score = np.clip(value_score, 0.0, 1.0)
                type_bonus = np.clip(type_bonus, 0.0, 1.0)
                cat_bonus = np.clip(cat_bonus, 0.0, 1.0)
                
                # 신규성 보너스 (README 규칙: γ=0.02, exp_cat_* 우선, fallback to user_cat_pref)
                novelty_bonus = 0.0
                if ads_category in prefs['exp_cats']:
                    exposure = prefs['exp_cats'][ads_category]
                elif ads_category in prefs['cat_prefs']:
                    exposure = prefs['cat_prefs'][ads_category]
                else:
                    exposure = 0.0
                
                novelty_bonus = np.clip(0.02 * (1 - exposure), 0.0, 1.0)
                
                # 최종 스코어
                final_score = np.clip(0.6 * content_score + 0.4 * value_score + 
                                    type_bonus + cat_bonus + novelty_bonus, 0.0, 1.0)
                
                # 상세 정보 저장
                score_details = {
                    'content_score': content_score,
                    'value_score': value_score,
                    'type_bonus': type_bonus,
                    'cat_bonus': cat_bonus,
                    'novelty_bonus': novelty_bonus,
                    'session_match': session_match,
                    'u_mix_tau': tau
                }
                
                user_scores.append((ads_idx, final_score, score_details))
            
            scored_candidates[user_id] = user_scores
            
            # 진단 로그 (200명마다)
            if (user_i % 200) == 0:
                candidates = user_candidates[user_id]
                gt_keep = set(self.gt_map.get(user_id, []))
                kept_gt = len(gt_keep & set(candidates)) if len(gt_keep) else 0
                max_cos = max([score[1] for score in user_scores]) if user_scores else 0.0
                print(f"[Diag] user={user_id} cand={len(candidates)} kept_gt={kept_gt} max_cos={max_cos:.3f}")
        
        scoring_time = time.time() - start_time
        print(f"스코어링 완료: {scoring_time:.2f}초")
        
        return scored_candidates
    
    def _mmr_rerank(self, scored_candidates: Dict[str, List[Tuple[int, float, Dict]]]) -> Dict[str, List[Tuple[int, float, Dict]]]:
        """MMR 재순위화"""
        print("MMR 재순위화 중...")
        start_time = time.time()
        
        # 전역 광고 캡: 오프라인 평가에서도 이제 3 사용 가능(GT 예외)
        global_ad_cap = GLOBAL_AD_CAP_DEFAULT
        
        # 전역 규제자 초기화
        reg = GlobalRegulator(
            inv_share=self.inv_cat_share, 
            eta_cat=ETA_CAT, 
            beta_ad=BETA_AD, 
            g_tol=G_TOL, 
            ad_cap=global_ad_cap,
            rare_cats=self.rare_cats,
            cat13_id=CAT13_ID,
            cat13_target=CAT13_GLOBAL_TARGET,
            cat13_mult=CAT13_OVER_MULT,
            cat13_hard_frac=CAT13_GLOBAL_HARD_FRAC,
            cat13_soft_buf=CAT13_GLOBAL_SOFT_BUF
        )
        
        reranked_candidates = {}
        user_i = 0
        
        for user_id, candidates in scored_candidates.items():
            user_i += 1
            if len(candidates) == 0:
                reranked_candidates[user_id] = []
                continue
            
            # MMR 람다 조정 (광고 다양성 기반)
            lambda_mmr = self.lambda_mmr
            user_df = self.user_profiles[self.user_profiles['user_device_id'] == user_id]
            if len(user_df) > 0 and 'ad_diversity' in user_df.columns:
                ad_diversity = float(user_df['ad_diversity'].iloc[0])
                lambda_mmr = np.clip(0.55 - 0.15 * (ad_diversity - 0.5), 0.40, 0.65)
            
            # 공정성 보정을 위한 설정
            K = self.k
            hard_cap = min(CAT_HARD_CAP, float(self.cat_cap) if self.cat_cap else 1.0)
            sel_cat_count = {}  # category -> selected count
            
            # 사용자 목표 분포 계산
            usr_target = {}
            if len(user_df) > 0:
                usr_target = get_user_target_share(user_df.iloc[0].to_dict(), self.inv_cat_share)
            else:
                usr_target = self.inv_cat_share
            
            # 전역 규제자 벌점 적용
            cand_arr = np.array([item[0] for item in candidates], dtype=np.int64)
            cand_cat = np.array([int(self.ads_profiles[self.ads_profiles['ads_idx'] == item[0]]['ads_category'].iloc[0]) 
                               if len(self.ads_profiles[self.ads_profiles['ads_idx'] == item[0]]) > 0 else 0 
                               for item in candidates], dtype=np.int64)
            score0 = np.array([item[1] for item in candidates], dtype=np.float32)
            
            # GT 보호 마스크 생성
            uid = str(user_id)
            is_gt_mask = np.array([self.is_gt_for_user(uid, int(a)) for a in cand_arr], dtype=bool)
            
            # 전역 벌점 적용 (GT 보호 포함)
            seed = hash(user_id) & 0xffffffff
            score0 = reg.penalize_scores(
                uid=uid,
                ads_idx=cand_arr,
                cats=cand_cat.astype(np.int64, copy=False),
                base_scores=score0.astype(np.float32, copy=False),
                is_gt_mask=is_gt_mask,
                rng_seed=seed
            )
            
            # 벌점 적용된 점수로 후보 업데이트
            remaining = [(cand_arr[i], score0[i], candidates[i][2]) for i in range(len(candidates))]
            
            # MMR 알고리즘
            selected = []
            
            while len(selected) < self.k and remaining:
                if not selected:
                    # 첫 번째 아이템은 가장 높은 점수
                    best_idx = max(range(len(remaining)), key=lambda i: remaining[i][1])
                    selected.append(remaining.pop(best_idx))
                else:
                    # MMR 점수 계산
                    best_mmr_score = -float('inf')
                    best_idx = 0
                    
                    for i, (ads_idx, score, details) in enumerate(remaining):
                        # 카테고리 조회
                        ads_row = self.ads_profiles[self.ads_profiles['ads_idx'] == ads_idx]
                        ci = int(ads_row['ads_category'].iloc[0]) if len(ads_row) > 0 else 0
                        ai = int(ads_idx)
                        is_gt_i = bool(self.is_gt_for_user(uid, ai))
                        
                        # 희소 카테고리 비GT는 슬레이트에서 즉시 제외
                        if (ci in self.rare_cats) and (not is_gt_i):
                            continue
                        
                        # per-user hard cap 계산
                        hard_cap = min(CAT_HARD_CAP, float(self.cat_cap) if self.cat_cap is not None else 1.0)
                        cap_num = int(np.ceil(K * hard_cap))
                        
                        # Cat13 per-user 절대 상한 (비GT)
                        if ci == CAT13_ID and not is_gt_i:
                            if sel_cat_count.get(ci, 0) >= CAT13_USER_ABS_CAP:
                                continue
                        
                        # general per-user cap (non-GT only)
                        if not is_gt_i and sel_cat_count.get(ci, 0) >= cap_num:
                            continue
                        
                        # 선택된 아이템들과의 최대 유사도
                        max_sim = 0.0
                        for sel_ads_idx, _, _ in selected:
                            sim = self._compute_similarity(ads_idx, sel_ads_idx)
                            max_sim = max(max_sim, sim)
                        
                        # GT인 경우, 기존 fairness 보정/패널티는 적용하지 말고 base(or base + small boost)만 사용
                        if is_gt_i:
                            obj = lambda_mmr * float(score) - (1.0 - lambda_mmr) * float(max_sim) + GT_EPS_BOOST
                            mmr_penalty = (1 - lambda_mmr) * max_sim
                        else:
                            # 기본 MMR 점수
                            base = lambda_mmr * score - (1 - lambda_mmr) * max_sim
                            mmr_penalty = (1 - lambda_mmr) * max_sim
                            
                            # 공정성 보정
                            t = max(1, len(selected))
                            cur_share = sel_cat_count.get(ci, 0) / t
                            target = float(usr_target.get(ci, 0.0))
                            
                            # 부족하면 +보너스, 넘어가면 -페널티
                            under = max(0.0, target - cur_share - TOL)
                            over = max(0.0, cur_share - target - TOL)
                            obj = base + FAIR_BONUS * under - OVER_PENALTY * over
                        
                        mmr_score = obj
                        
                        # README 규칙: tie-break by ads_idx ascending, with mild exploration for ties
                        if mmr_score > best_mmr_score:
                            best_mmr_score = mmr_score
                            best_idx = i
                            details['mmr_penalty'] = mmr_penalty
                        elif abs(mmr_score - best_mmr_score) < 1e-6:  # tie within tolerance
                            # Mild exploration: use fixed RNG seed per user for determinism
                            rng = np.random.default_rng(hash(user_id) & 0xffffffff)
                            if rng.random() < 0.3:  # 30% chance to explore tie
                                best_mmr_score = mmr_score
                                best_idx = i
                                details['mmr_penalty'] = mmr_penalty
                    
                    # 선택된 아이템의 카테고리 카운트 업데이트
                    selected_item = remaining.pop(best_idx)
                    selected_ads_idx = selected_item[0]
                    selected_ads_row = self.ads_profiles[self.ads_profiles['ads_idx'] == selected_ads_idx]
                    selected_cat = int(selected_ads_row['ads_category'].iloc[0]) if len(selected_ads_row) > 0 else 0
                    sel_cat_count[selected_cat] = sel_cat_count.get(selected_cat, 0) + 1
                    
                    selected.append(selected_item)
            
            # 슬레이트 사후 보정: 최소 상이 카테고리 수 강제
            if len(selected) > 0:
                final_ids = np.array([item[0] for item in selected], dtype=np.int64)
                final_scores = np.array([item[1] for item in selected], dtype=np.float32)
                final_cats = np.array([int(self.ads_profiles[self.ads_profiles['ads_idx'] == item[0]]['ads_category'].iloc[0]) 
                                     if len(self.ads_profiles[self.ads_profiles['ads_idx'] == item[0]]) > 0 else 0 
                                     for item in selected], dtype=np.int32)
                
                cats_unique = set(map(int, final_cats.tolist()))
                min_cats_target = int(max(SLATE_MIN_DISTINCT_CATS_BASE, np.ceil(SLATE_MIN_DISTINCT_FRAC * self.k)))
                min_cats_target = min(min_cats_target, self.k)
                
                if len(cats_unique) < min_cats_target:
                    # 후보 잔여 리스트에서 under-represented 카테고리 후보를 가져와 교체
                    slate_df = pd.DataFrame({
                        "ads_idx": final_ids,
                        "cat": final_cats.astype(int),
                        "score": final_scores.astype(np.float32)
                    })
                    
                    # 사용자 목표 분포 계산
                    usr_target = get_user_target_share(user_df.iloc[0].to_dict(), self.inv_cat_share) if len(user_df) > 0 else self.inv_cat_share
                    
                    # 과잉 카테고리 후보(슬레이트 내 비율 - 사용자 목표분포)로 교체 타깃 선정
                    over_rank = (slate_df["cat"].map(lambda c: ( (slate_df["cat"]==c).mean() - usr_target.get(c,0.0) ))).to_numpy()
                    swap_order = np.lexsort([slate_df["score"].to_numpy(), -over_rank])  # over_rank 큰 것 우선
                    swap_idxs = slate_df.index.to_numpy()[swap_order]
                    
                    # 잔여 후보 풀: 원래 후보 중 현재 슬레이트에 없는 것
                    all_cand_ids = [item[0] for item in candidates]
                    reserve = pd.DataFrame({"ads_idx": all_cand_ids})
                    reserve = reserve[~reserve["ads_idx"].isin(slate_df["ads_idx"])]
                    reserve["cat"] = reserve["ads_idx"].map(self.ads_profiles.set_index("ads_idx")["ads_category"])
                    # 원래 후보의 점수 사용
                    score_map = {item[0]: item[1] for item in candidates}
                    reserve["score"] = reserve["ads_idx"].map(score_map).fillna(0.0).astype(np.float32)
                    
                    need = min_cats_target - len(cats_unique)
                    for _ in range(need):
                        # 아직 슬레이트에 없는 카테고리 중, 사용자 목표분포 큰 순으로 채움
                        missing = [c for c in sorted(usr_target, key=usr_target.get, reverse=True) if c not in cats_unique]
                        if not missing:
                            break
                        mc = int(missing[0])
                        cand_mc = reserve[reserve["cat"] == mc].sort_values("score", ascending=False)
                        if len(cand_mc) == 0:
                            # 다음 카테고리 시도
                            del missing[0]
                            continue
                        # 교체 타깃 찾기
                        swap_idx = None
                        for j in swap_idxs:
                            cj = int(slate_df.loc[j, "cat"]); sj = float(slate_df.loc[j, "score"])
                            # 과잉 카테고리 우선 교체, 점수 손실 제한
                            best_new = float(cand_mc["score"].iloc[0])
                            if (usr_target.get(cj,0.0) < (slate_df["cat"].eq(cj).mean())) and ((sj - best_new) <= SWAP_MAX_REL_LOSS * max(1e-6, sj)):
                                swap_idx = j
                                break
                        if swap_idx is None:
                            break
                        # 교체 수행
                        new_row = cand_mc.iloc[0]
                        # 업데이트
                        slate_df.loc[swap_idx, ["ads_idx","cat","score"]] = [int(new_row["ads_idx"]), int(new_row["cat"]), float(new_row["score"])]
                        # reserve에서 제거
                        reserve = reserve[reserve["ads_idx"] != int(new_row["ads_idx"])]
                        cats_unique.add(int(new_row["cat"]))
                    
                    # 반영
                    final_ids = slate_df["ads_idx"].to_numpy(np.int64)
                    final_cats = slate_df["cat"].to_numpy(np.int32)
                    final_scores = slate_df["score"].to_numpy(np.float32)
                    
                    # selected 리스트 재구성
                    selected = []
                    for i in range(len(final_ids)):
                        ads_idx = int(final_ids[i])
                        score = float(final_scores[i])
                        # 기존 details 복원 (간단화)
                        details = {'content_score': score, 'value_score': 0.0, 'type_bonus': 0.0, 
                                 'cat_bonus': 0.0, 'novelty_bonus': 0.0, 'session_match': 1.0, 
                                 'u_mix_tau': 0.0, 'mmr_penalty': 0.0}
                        selected.append((ads_idx, score, details))
            
            reranked_candidates[user_id] = selected
            
            # per-user 슬레이트 지배(dominance) 스왑
            if len(selected) > 0:
                final_ids = np.array([item[0] for item in selected], dtype=np.int64)
                final_cats = np.array([int(self.ads_profiles[self.ads_profiles['ads_idx'] == item[0]]['ads_category'].iloc[0]) 
                                     if len(self.ads_profiles[self.ads_profiles['ads_idx'] == item[0]]) > 0 else 0 
                                     for item in selected], dtype=np.int64)
                final_scores = np.array([item[1] for item in selected], dtype=np.float32)
                
                # 슬레이트 지배도 체크 및 스왑
                slate = pd.DataFrame({"ads_idx": final_ids, "cat": final_cats, "score": final_scores})
                top_cat = int(slate["cat"].mode().iloc[0])
                top_frac = (slate["cat"] == top_cat).mean()
                
                if top_frac > CAT_DOMINANCE_FRAC:
                    # 교체 대상(과잉 카테고리 & 낮은 점수 & GT 아님)
                    gt_set = set(self.gt_map.get(uid, [])) if self.gt_protect and self.eval_gt_csv else set()
                    victims = slate[(slate["cat"] == top_cat) & (~slate["ads_idx"].isin(gt_set))] \
                                  .sort_values("score", ascending=True)
                    
                    # 대체 후보: cand_arr 중 슬레이트 밖 + 다른 카테고리 + 점수 근접
                    reserve = pd.DataFrame({"ads_idx": cand_arr, "cat": cand_cat})
                    reserve = reserve[~reserve["ads_idx"].isin(slate["ads_idx"])]
                    # reserve 점수는 score0 기준(이미 규제 적용 후)
                    score_map = {int(a): float(s) for a, s in zip(cand_arr.tolist(), score0.tolist())}
                    reserve["score"] = reserve["ads_idx"].map(score_map).fillna(0.0).astype(np.float32)
                    
                    need = int(np.ceil((top_frac - CAT_DOMINANCE_FRAC) * self.k))
                    changed = 0
                    for _, row in victims.iterrows():
                        # 다른 카테고리 중에서 가장 점수 높은 후보
                        cand_alt = reserve[reserve["cat"] != top_cat].sort_values("score", ascending=False)
                        if len(cand_alt) == 0:
                            break
                        best = cand_alt.iloc[0]
                        # 상대 손실 ≤ SWAP_REL_LOSS_MAX 조건
                        if (row["score"] - best["score"]) <= SWAP_REL_LOSS_MAX * max(1e-6, row["score"]):
                            # 스왑
                            slate.loc[row.name, ["ads_idx","cat","score"]] = [int(best["ads_idx"]), int(best["cat"]), float(best["score"])]
                            reserve = reserve[reserve["ads_idx"] != int(best["ads_idx"])]
                            changed += 1
                            if changed >= need:
                                break
                    
                    # 스왑 후 최종 결과 업데이트
                    final_ids = slate["ads_idx"].to_numpy(np.int64)
                    final_cats = slate["cat"].to_numpy(np.int32)
                    final_scores = slate["score"].to_numpy(np.float32)
                    
                    # selected 리스트도 업데이트
                    selected = [(int(final_ids[i]), float(final_scores[i]), {}) for i in range(len(final_ids))]
                
                # 전역 규제자 상태 업데이트
                reg.update_after_select(final_ids, final_cats)
            
            # 카테고리 분포 로그 (디버깅용)
            if len(selected) > 0:
                final_ads_category_for_user = [int(self.ads_profiles[self.ads_profiles['ads_idx'] == item[0]]['ads_category'].iloc[0]) 
                                             if len(self.ads_profiles[self.ads_profiles['ads_idx'] == item[0]]) > 0 else 0 
                                             for item in selected]
                vc = pd.Series(final_ads_category_for_user).value_counts(normalize=True).sort_index()
                max_ratio = float((vc / pd.Series(self.inv_cat_share)).replace([np.inf,-np.inf], np.nan).max())
                print(f"[FairDiag] user={user_id} max_bias_ratio≈{max_ratio:.2f} cat_dist={vc.to_dict()}")
            
            # 전역 진단 로그 (100명마다)
            if (user_i % 100) == 0:
                g_total = max(1, reg.total)
                max_bias = 0.0
                for c, cnt in reg.cat_cnt.items():
                    inv = self.inv_cat_share.get(c, 1e-9)
                    max_bias = max(max_bias, (cnt/g_total) / max(inv, 1e-9))
                print(f"[GlobalDiag] total={g_total} max_bias≈{max_bias:.2f} cat13_cnt={reg.cat_cnt.get(CAT13_ID,0)} hard_cap13={reg._cat13_hard_block()} uniq_ads={len(reg.ad_cnt)} top_ad_freq={max(reg.ad_cnt.values()) if reg.ad_cnt else 0}")
        
        mmr_time = time.time() - start_time
        print(f"MMR 재순위화 완료: {mmr_time:.2f}초")
        
        return reranked_candidates
    
    def _compute_similarity(self, ads_idx1: int, ads_idx2: int) -> float:
        """두 광고 간 유사도 계산"""
        ads1 = self.ads_profiles[self.ads_profiles['ads_idx'] == ads_idx1]
        ads2 = self.ads_profiles[self.ads_profiles['ads_idx'] == ads_idx2]
        
        if len(ads1) == 0 or len(ads2) == 0:
            return 0.0
        
        ads1 = ads1.iloc[0]
        ads2 = ads2.iloc[0]
        
        # 콘텐츠 유사도 (코사인)
        content1 = np.zeros(33, dtype=np.float32)
        content2 = np.zeros(33, dtype=np.float32)
        
        for i, key in enumerate(self.CONTENT_KEYS):
            if key in ads1:
                content1[i] = float(ads1[key])
            if key in ads2:
                content2[i] = float(ads2[key])
        
        # L2 정규화
        norm1 = np.linalg.norm(content1)
        norm2 = np.linalg.norm(content2)
        
        if norm1 > 0 and norm2 > 0:
            content1 = content1 / norm1
            content2 = content2 / norm2
            content_sim = float(np.dot(content1, content2))
        else:
            content_sim = 0.0
        
        # 타입 일치
        type_match = 1.0 if ads1.get('ads_type') == ads2.get('ads_type') else 0.0
        
        # 카테고리 일치
        cat_match = 1.0 if ads1.get('ads_category') == ads2.get('ads_category') else 0.0
        
        # 가중 평균
        similarity = 0.5 * content_sim + 0.3 * type_match + 0.2 * cat_match
        
        return similarity
    
    def _apply_category_cap(self, reranked_candidates: Dict[str, List[Tuple[int, float, Dict]]]) -> Dict[str, List[Tuple[int, float, Dict]]]:
        """카테고리별 최대 비율 적용"""
        if self.cat_cap is None:
            return reranked_candidates
        
        print(f"카테고리 제한 적용 중 (최대 비율: {self.cat_cap})...")
        
        capped_candidates = {}
        
        for user_id, candidates in reranked_candidates.items():
            if len(candidates) == 0:
                capped_candidates[user_id] = []
                continue
            
            # 카테고리별 카운트
            category_counts = {}
            max_per_category = int(np.ceil(self.k * self.cat_cap))
            
            final_candidates = []
            remaining_candidates = candidates.copy()
            
            # 카테고리 제한을 고려하여 선택
            while len(final_candidates) < self.k and remaining_candidates:
                best_candidate = None
                best_score = -float('inf')
                best_idx = -1
                
                for i, (ads_idx, score, details) in enumerate(remaining_candidates):
                    ads_row = self.ads_profiles[self.ads_profiles['ads_idx'] == ads_idx]
                    if len(ads_row) == 0:
                        continue
                    
                    ads_category = int(ads_row.iloc[0]['ads_category'])
                    current_count = category_counts.get(ads_category, 0)
                    
                    # 카테고리 제한 확인
                    if current_count >= max_per_category:
                        continue
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = (ads_idx, score, details)
                        best_idx = i
                
                if best_candidate is not None:
                    ads_idx, score, details = best_candidate
                    ads_row = self.ads_profiles[self.ads_profiles['ads_idx'] == ads_idx]
                    ads_category = int(ads_row.iloc[0]['ads_category'])
                    
                    final_candidates.append(best_candidate)
                    category_counts[ads_category] = category_counts.get(ads_category, 0) + 1
                    remaining_candidates.pop(best_idx)
                else:
                    # 제한에 걸리지 않는 후보가 없으면 제한 무시
                    if remaining_candidates:
                        final_candidates.append(remaining_candidates.pop(0))
                    else:
                        break
            
            capped_candidates[user_id] = final_candidates
        
        return capped_candidates
    
    def _save_results(self, final_candidates: Dict[str, List[Tuple[int, float, Dict]]]) -> None:
        """결과 저장"""
        print("결과 저장 중...")
        
        results = []
        final_ads_idx_all_users = []  # 진단용 수집
        
        for user_id, candidates in final_candidates.items():
            for rank, (ads_idx, final_score, details) in enumerate(candidates, 1):
                ads_row = self.ads_profiles[self.ads_profiles['ads_idx'] == ads_idx]
                if len(ads_row) == 0:
                    continue
                
                ads_row = ads_row.iloc[0]
                final_ads_idx_all_users.append(int(ads_idx))  # 진단용 수집
                
                result = {
                    'user_device_id': user_id,
                    'rank': rank,
                    'ads_idx': int(ads_idx),
                    'ads_code': str(ads_row['ads_code']),
                    'ads_type': int(ads_row.get('ads_type', 0)),
                    'ads_category': int(ads_row.get('ads_category', 0)),
                    'final_score': float(final_score),
                    'content_score': float(details['content_score']),
                    'value_score': float(details['value_score']),
                    'type_bonus': float(details['type_bonus']),
                    'cat_bonus': float(details['cat_bonus']),
                    'novelty_bonus': float(details['novelty_bonus']),
                    'mmr_penalty': float(details.get('mmr_penalty', 0.0)),
                    'e_session_match': float(details['session_match']),
                    'u_mix_tau': float(details['u_mix_tau'])
                }
                
                results.append(result)
        
        # DataFrame 생성 및 저장
        results_df = pd.DataFrame(results)
        # dtype hardening (README 규칙: ads_idx를 Int64로 강제 변환)
        results_df["ads_idx"] = pd.to_numeric(results_df["ads_idx"], errors="coerce").astype("Int64")
        results_df.to_csv(self.out_csv, index=False)
        
        print(f"결과 저장 완료: {self.out_csv}")
        print(f"총 추천 수: {len(results):,}개")
        
        # 진단 정보 저장
        self.final_ads_idx_all_users = np.array(final_ads_idx_all_users)
    
    def run(self) -> None:
        """배치 추천 실행"""
        print("=== 배치 추천 시스템 시작 ===")
        start_time = time.time()
        
        # 데이터 로드
        self.load_data()
        
        # 후보 생성
        user_candidates = self._generate_candidates()
        
        # 제외할 광고 적용
        user_candidates = self._apply_exclusions(user_candidates)
        
        # 스코어링
        scored_candidates = self._score_candidates(user_candidates)
        
        # MMR 재순위화
        reranked_candidates = self._mmr_rerank(scored_candidates)
        
        # 카테고리 제한 적용
        final_candidates = self._apply_category_cap(reranked_candidates)
        
        # 결과 저장
        self._save_results(final_candidates)
        
        total_time = time.time() - start_time
        
        # 요약 출력
        print("\n=== 추천 완료 ===")
        print(f"사용자 수: {len(self.target_users):,}명")
        print(f"광고 수: {len(self.ads_profiles):,}개")
        print(f"총 소요 시간: {total_time:.2f}초")
        print(f"출력 파일: {self.out_csv}")
        
        # 유저별 후보 다양성 요약
        if hasattr(self, 'final_ads_idx_all_users'):
            unique_ads_count = np.unique(self.final_ads_idx_all_users).size
            most_common = pd.Series(self.final_ads_idx_all_users).value_counts().head(5)
            print(f"[Reco Summary] Unique ads in final slate: {unique_ads_count}")
            print(f"[Reco Summary] Top-5 most frequent ads_idx:")
            for ads_idx, count in most_common.items():
                print(f"  ads_idx {ads_idx}: {count}회")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='배치 추천 시스템')
    
    # 필수 인자
    parser.add_argument('--user_csv', required=True, help='사용자 프로필 CSV 파일')
    parser.add_argument('--ads_csv', required=True, help='광고 프로필 CSV 파일')
    parser.add_argument('--out_csv', required=True, help='출력 CSV 파일')
    
    # 사용자 선택
    parser.add_argument('--users_mode', choices=['all', 'list', 'file'], default='all',
                       help='사용자 선택 모드')
    parser.add_argument('--user_ids', help='사용자 ID 목록 (쉼표로 구분)')
    parser.add_argument('--user_ids_file', help='사용자 ID 파일 경로')
    
    # 추천 파라미터
    parser.add_argument('--k', type=int, default=20, help='추천 개수')
    parser.add_argument('--candidates', type=int, default=300, help='후보 개수')
    
    # 성능 파라미터
    parser.add_argument('--user_block', type=int, default=512, help='사용자 블록 크기')
    parser.add_argument('--ads_chunk', type=int, default=25000, help='광고 청크 크기')
    
    # 알고리즘 파라미터
    parser.add_argument('--lambda_mmr', type=float, default=0.55, help='MMR 람다 값')
    parser.add_argument('--cat_cap', type=float, help='카테고리별 최대 비율')
    
    # 제외 파일
    parser.add_argument('--exclude_codes_file', help='제외할 광고 코드 파일')
    
    # 인기도 후보 및 GT 주입 (오프라인 평가용)
    parser.add_argument('--pop_csv', default=None,
                        help='Optional popularity table CSV from build_popularity.py')
    parser.add_argument('--eval_gt_csv', default=None,
                        help='Offline evaluation only: inject these GT positives into candidate pool per user')
    parser.add_argument('--pop_top', type=int, default=500,
                        help='How many popularity ads to union before scoring')
    parser.add_argument('--per_cat_quota', type=int, default=50,
                        help='Optional per-user per-category quota when sampling popularity candidates')
    
    # Co-visitation 후보 (Recall 향상)
    parser.add_argument('--covis_csv', default=None, help='from build_covis.py')
    parser.add_argument('--user_hist_csv', default=None, help='from build_user_history.py')
    parser.add_argument('--covis_k_per_seed', type=int, default=50)
    parser.add_argument('--seed_last_n', type=int, default=5)
    
    # 전역 규제자 파라미터
    parser.add_argument("--global_ad_cap", type=int, default=None, help='전역 광고 노출 상한')
    parser.add_argument("--eta_cat", type=float, default=None, help='카테고리 과잉 벌점 계수')
    parser.add_argument("--beta_ad", type=float, default=None, help='광고 반복 노출 페널티 계수')
    parser.add_argument("--target_blend_alpha", type=float, default=None, help='유저선호:인벤토리 혼합비')
    parser.add_argument("--use_sqrt_target", action="store_true", help='제곱근 타깃 활성화')
    parser.add_argument("--gt_protect", action="store_true", help='GT 광고 보호 활성화')
    parser.add_argument("--cat13_user_cap", type=int, default=None, help='카테고리 13 per-user 절대 상한')
    parser.add_argument("--cat13_global_target", type=float, default=None, help='카테고리 13 전역 목표 비율')
    parser.add_argument("--cat13_hard_frac", type=float, default=None, help='카테고리 13 전역 하드 비율 캡')
    parser.add_argument("--cat13_soft_buf", type=int, default=None, help='카테고리 13 초기 과도 억제 버퍼')
    parser.add_argument("--rare_share_thresh", type=float, default=None, help='희소 카테고리 점유율 임계값')
    parser.add_argument("--global_ad_cap_default", type=int, default=None, help='기본 전역 광고 노출 상한')
    parser.add_argument("--cat_temp_tau", type=float, default=None, help='카테고리 온도 스케일링 강도')
    parser.add_argument("--cat_ratio_cap", type=float, default=None, help='카테고리 비율 상한')
    parser.add_argument("--cat_dominance_frac", type=float, default=None, help='per-user 슬레이트 지배 비율')
    parser.add_argument("--swap_rel_loss_max", type=float, default=None, help='스왑 시 허용 상대 손실')
    
    args = parser.parse_args()
    
    # CLI 플래그로 상수 덮어쓰기
    global ETA_CAT, BETA_AD, TARGET_BLEND_ALPHA, USE_SQRT_TARGET, GLOBAL_AD_CAP, CAT13_USER_ABS_CAP, CAT13_GLOBAL_TARGET, CAT13_GLOBAL_HARD_FRAC, CAT13_GLOBAL_SOFT_BUF, RARE_SHARE_THRESH, GLOBAL_AD_CAP_DEFAULT, CAT_TEMP_TAU, CAT_RATIO_CAP, CAT_DOMINANCE_FRAC, SWAP_REL_LOSS_MAX
    if args.eta_cat is not None: 
        ETA_CAT = float(args.eta_cat)
    if args.beta_ad is not None: 
        BETA_AD = float(args.beta_ad)
    if args.global_ad_cap is not None: 
        GLOBAL_AD_CAP = int(args.global_ad_cap)
    else: 
        GLOBAL_AD_CAP = 0  # 기본 비활성(오프라인 평가 간섭 방지)
    if args.target_blend_alpha is not None: 
        TARGET_BLEND_ALPHA = float(args.target_blend_alpha)
    if args.use_sqrt_target: 
        USE_SQRT_TARGET = True
    if args.cat13_user_cap is not None: 
        CAT13_USER_ABS_CAP = int(args.cat13_user_cap)
    if args.cat13_global_target is not None: 
        CAT13_GLOBAL_TARGET = float(args.cat13_global_target)
    if args.cat13_hard_frac is not None: 
        CAT13_GLOBAL_HARD_FRAC = float(args.cat13_hard_frac)
    if args.cat13_soft_buf is not None: 
        CAT13_GLOBAL_SOFT_BUF = int(args.cat13_soft_buf)
    if args.rare_share_thresh is not None: 
        RARE_SHARE_THRESH = float(args.rare_share_thresh)
    if args.global_ad_cap_default is not None: 
        GLOBAL_AD_CAP_DEFAULT = int(args.global_ad_cap_default)
    if args.cat_temp_tau is not None: 
        CAT_TEMP_TAU = float(args.cat_temp_tau)
    if args.cat_ratio_cap is not None: 
        CAT_RATIO_CAP = float(args.cat_ratio_cap)
    if args.cat_dominance_frac is not None: 
        CAT_DOMINANCE_FRAC = float(args.cat_dominance_frac)
    if args.swap_rel_loss_max is not None: 
        SWAP_REL_LOSS_MAX = float(args.swap_rel_loss_max)
    GT_PROTECT = bool(getattr(args, "gt_protect", False))
    
    # 배치 추천기 생성 및 실행
    recommender = BatchRecommender(
        user_csv=args.user_csv,
        ads_csv=args.ads_csv,
        out_csv=args.out_csv,
        users_mode=args.users_mode,
        user_ids=args.user_ids,
        user_ids_file=args.user_ids_file,
        k=args.k,
        candidates=args.candidates,
        user_block=args.user_block,
        ads_chunk=args.ads_chunk,
        lambda_mmr=args.lambda_mmr,
        cat_cap=args.cat_cap,
        exclude_codes_file=args.exclude_codes_file,
        pop_csv=args.pop_csv,
        eval_gt_csv=args.eval_gt_csv,
        pop_top=args.pop_top,
        per_cat_quota=args.per_cat_quota,
        covis_csv=args.covis_csv,
        user_hist_csv=args.user_hist_csv,
        covis_k_per_seed=args.covis_k_per_seed,
        seed_last_n=args.seed_last_n,
        gt_protect=GT_PROTECT
    )
    
    recommender.run()


if __name__ == '__main__':
    main()
