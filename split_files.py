import pandas as pd
import os

def split_large_files():
    """대용량 파일을 25MB 이하로 분할합니다."""
    
    # ads_profile.csv 분할
    print("ads_profile.csv 분할 중...")
    df_ads = pd.read_csv('preprocessed/ads_profile.csv')
    chunk_size = 100000  # 약 20MB 정도
    
    for i in range(0, len(df_ads), chunk_size):
        part_num = i // chunk_size + 1
        filename = f'ads_profile_part_{part_num}.csv'
        df_ads.iloc[i:i+chunk_size].to_csv(filename, index=False)
        file_size = os.path.getsize(filename) / (1024*1024)  # MB
        print(f"  {filename}: {file_size:.1f}MB")
    
    # user_profile.csv 분할
    print("\nuser_profile.csv 분할 중...")
    df_user = pd.read_csv('preprocessed/user_profile.csv')
    chunk_size = 50000  # 약 20MB 정도
    
    for i in range(0, len(df_user), chunk_size):
        part_num = i // chunk_size + 1
        filename = f'user_profile_part_{part_num}.csv'
        df_user.iloc[i:i+chunk_size].to_csv(filename, index=False)
        file_size = os.path.getsize(filename) / (1024*1024)  # MB
        print(f"  {filename}: {file_size:.1f}MB")
    
    print("\n분할 완료!")

if __name__ == "__main__":
    split_large_files()
