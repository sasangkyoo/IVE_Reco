import pandas as pd
import os

def split_correct_interactions():
    """correct_interactions.csv를 20MB 이하로 분할합니다."""
    
    print("correct_interactions.csv 분할 중...")
    
    # 압축 해제
    import zipfile
    with zipfile.ZipFile('correct_interactions.zip', 'r') as zip_ref:
        zip_ref.extractall('temp/')
    
    # CSV 파일 읽기
    df = pd.read_csv('temp/correct_interactions.csv')
    print(f"총 행 수: {len(df):,}")
    
    # 청크 크기 계산 (약 15MB 정도)
    chunk_size = 50000  # 약 15MB 정도
    
    for i in range(0, len(df), chunk_size):
        part_num = i // chunk_size + 1
        filename = f'correct_interactions_part_{part_num}.csv'
        df.iloc[i:i+chunk_size].to_csv(filename, index=False)
        file_size = os.path.getsize(filename) / (1024*1024)  # MB
        print(f"  {filename}: {file_size:.1f}MB")
    
    # 임시 파일 삭제
    os.remove('temp/correct_interactions.csv')
    os.rmdir('temp')
    
    print("\ncorrect_interactions 분할 완료!")

if __name__ == "__main__":
    split_correct_interactions()
