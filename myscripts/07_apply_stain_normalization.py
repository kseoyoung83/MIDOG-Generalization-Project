import sys, json, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from tiatoolbox.tools.stainnorm import get_normalizer

sys.path.append('/workspace')
from myscripts import config

targets = json.load(open(config.RESULTS_DIR / "target_image_info.json"))
methods = ['macenko', 'vahadane', 'reinhard']
scanners = ['Hamamatsu-XR', 'Hamamatsu-S360', 'Aperio-CS2']
stats = []

for method in methods:
    print(f"\n[{method.upper()}]")
    for scanner in scanners:
        print(f"{scanner}:", end=' ')
        target_path = config.RAW_DATA_DIR / targets[scanner]['filename']
        
        # Fit
        with Image.open(target_path) as img:
            img = img.convert('RGB')
            w, h = img.size
            crop = img.crop((w//2-512, h//2-512, w//2+512, h//2+512))
            arr = np.array(crop, dtype=np.uint8)
        
        normalizer = get_normalizer(method)
        normalizer.fit(arr)
        print("fitted,", end=' ')
        
        # Transform
        in_dir = config.PROCESSED_DIR / "patches" / scanner
        out_dir = config.PROCESSED_DIR / "patches_normalized" / method / scanner
        
        for cls in ['positive', 'negative']:
            (out_dir / cls).mkdir(parents=True, exist_ok=True)
            patches = list((in_dir / cls).glob("*.png"))
            
            for i, p in enumerate(patches):
                if i % 500 == 0:
                    print(f"{cls[0]}{i}", end=' ')
                img = Image.open(p).convert('RGB')
                norm = normalizer.transform(np.array(img, dtype=np.uint8))
                Image.fromarray(norm).save(out_dir / cls / p.name)
        
        pos = len(list((out_dir / 'positive').glob("*.png")))
        neg = len(list((out_dir / 'negative').glob("*.png")))
        stats.append({'method': method, 'scanner': scanner, 'pos': pos, 'neg': neg})
        print(f"✓ {pos}+{neg}")

pd.DataFrame(stats).to_csv(config.RESULTS_DIR / "normalization_stats.csv", index=False)
print("\n✓ Done")