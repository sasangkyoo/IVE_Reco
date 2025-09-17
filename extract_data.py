#!/usr/bin/env python3
"""
데이터 압축 해제 스크립트
GitHub에서 다운로드한 압축 파일을 해제합니다.
"""

import os
import zipfile
import shutil
from pathlib import Path

def extract_data():
    """압축된 데이터 파일을 해제합니다."""
    
    # 압축 파일 경로
    zip_file = "correct_interactions.zip"
    extract_to = "input/save/"
    
    # 디렉토리 생성
    os.makedirs(extract_to, exist_ok=True)
    
    # 압축 해제
    if os.path.exists(zip_file):
        print(f"압축 파일 해제 중: {zip_file}")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"압축 해제 완료: {extract_to}")
        
        # 파일 존재 확인
        extracted_file = os.path.join(extract_to, "correct_interactions.csv")
        if os.path.exists(extracted_file):
            file_size = os.path.getsize(extracted_file)
            print(f"파일 크기: {file_size / (1024*1024):.2f} MB")
            print("✅ 데이터 파일이 성공적으로 해제되었습니다!")
        else:
            print("❌ 파일 해제에 실패했습니다.")
    else:
        print(f"❌ 압축 파일을 찾을 수 없습니다: {zip_file}")

if __name__ == "__main__":
    extract_data()
