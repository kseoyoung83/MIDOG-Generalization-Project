"""
01_parse_metadata.py
MIDOG.json (COCO format) 파싱 및 메타데이터 CSV 생성
"""

import json
import pandas as pd
from pathlib import Path
import sys

sys.path.append('/workspace')
from myscripts import config

def parse_midog_json(json_path, output_csv_path):
    """COCO 형식 MIDOG.json 파싱"""
    print(f"Loading {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # COCO format: images, annotations
    images = {img['id']: img for img in data['images']}
    
    # Annotation 개수 집계
    mitosis_counts = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in mitosis_counts:
            mitosis_counts[img_id] = []
        mitosis_counts[img_id].append(ann['bbox'])
    
    # 메타데이터 생성
    records = []
    for img_id, img_info in images.items():
        filename = img_info['file_name']
        
        # 파일명에서 번호 추출 (001.tiff -> 1)
        file_num = int(filename.split('.')[0])
        
        # 스캐너 매핑
        scanner = None
        for scanner_name, scanner_info in config.SCANNERS.items():
            start, end = scanner_info['range']
            if start <= file_num <= end:
                scanner = scanner_name
                break
        
        # 유사분열 개수
        num_mitoses = len(mitosis_counts.get(img_id, []))
        mitosis_coords = str(mitosis_counts.get(img_id, []))
        
        records.append({
            'image_id': img_id,
            'filename': filename,
            'file_number': file_num,
            'scanner': scanner,
            'scanner_label': config.SCANNERS.get(scanner, {}).get('label', 'Unknown'),
            'num_mitoses': num_mitoses,
            'mitosis_coords': mitosis_coords,
            'width': img_info.get('width', 0),
            'height': img_info.get('height', 0)
        })
    
    df = pd.DataFrame(records)
    df = df.sort_values(['scanner', 'file_number'])
    df.to_csv(output_csv_path, index=False)
    
    print(f"✓ Metadata saved to {output_csv_path}")
    print(f"\n=== Summary ===")
    print(f"Total images: {len(df)}")
    print(f"\nImages per scanner:")
    print(df['scanner'].value_counts())
    print(f"\nTotal mitoses: {df['num_mitoses'].sum()}")
    print(f"\nMitoses per scanner:")
    print(df.groupby('scanner')['num_mitoses'].sum())
    
    return df

def main():
    json_path = config.RAW_DATA_DIR / "MIDOG.json"
    output_csv_path = config.METADATA_DIR / "metadata.csv"
    
    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if not json_path.exists():
        print(f"ERROR: {json_path} not found!")
        return
    
    df = parse_midog_json(json_path, output_csv_path)
    
    # 스캐너별 CSV
    for scanner_name in config.SCANNERS.keys():
        scanner_df = df[df['scanner'] == scanner_name]
        scanner_csv = config.METADATA_DIR / f"scanner_{scanner_name}.csv"
        scanner_df.to_csv(scanner_csv, index=False)
        print(f"✓ {scanner_name}: {len(scanner_df)} images")

if __name__ == "__main__":
    main()