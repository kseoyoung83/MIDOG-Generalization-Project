"""
MIDOG 2021 Research Configuration
재현 가능성을 위한 모든 설정 중앙 관리
Project: MIDOG-Generalization-Project
User: ksy @ Geonungui-MacBookPro-2
"""

import os
import random
import numpy as np
import torch
from pathlib import Path

# ============= 재현성 설정 =============
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """모든 난수 생성기 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# ============= 경로 설정 =============
# Docker 내부 경로
PROJECT_ROOT = Path("/workspace")

# 데이터 디렉토리 (실제 구조 반영)
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "MIDOG_2021_DATA"
METADATA_DIR = DATA_DIR / "metadata"
PROCESSED_DIR = DATA_DIR / "processed"

# 결과 디렉토리
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# 스크립트 디렉토리
SCRIPTS_DIR = PROJECT_ROOT / "myscripts"

# ============= MIDOG 2021 데이터 설정 =============
# 스캐너 정보
SCANNERS = {
    'Hamamatsu-XR': {
        'range': (1, 50),
        'label': 'Scanner_A',
        'files': [f'{i:03d}.tiff' for i in range(1, 51)]
    },
    'Hamamatsu-S360': {
        'range': (51, 100),
        'label': 'Scanner_B',
        'files': [f'{i:03d}.tiff' for i in range(51, 101)]
    },
    'Aperio-CS2': {
        'range': (101, 150),
        'label': 'Scanner_C',
        'files': [f'{i:03d}.tiff' for i in range(101, 151)]
    },
    'Leica-GT450': {
        'range': (151, 200),
        'label': 'Scanner_D',
        'files': [f'{i:03d}.tiff' for i in range(151, 201)],
        'unlabeled': True
    }
}

# LOSO Folds (Leica 제외)
LOSO_FOLDS = [
    {
        'name': 'Fold1_TestA',
        'test_scanner': 'Hamamatsu-XR',
        'train_scanners': ['Hamamatsu-S360', 'Aperio-CS2']
    },
    {
        'name': 'Fold2_TestB',
        'test_scanner': 'Hamamatsu-S360',
        'train_scanners': ['Hamamatsu-XR', 'Aperio-CS2']
    },
    {
        'name': 'Fold3_TestC',
        'test_scanner': 'Aperio-CS2',
        'train_scanners': ['Hamamatsu-XR', 'Hamamatsu-S360']
    }
]

# ============= 전처리 설정 =============
TISSUE_MASK_PARAMS = {
    'method': 'Otsu',
    'gray_level': True,
    'min_region_size': 500
}

QC_THRESHOLDS = {
    'min_tissue_ratio': 0.3,
    'max_artifact_ratio': 0.2
}

# ============= 패치 추출 설정 =============
PATCH_SIZE = 224
PATCH_RESOLUTION = 0.5
PATCH_FORMAT = 'png'
POSITIVE_PATCH_MARGIN = 10
NEGATIVE_SAMPLING_RATIO = 3

# ============= Stain Normalization =============
STAIN_NORM_METHODS = ['Macenko', 'Vahadane', 'Reinhard']
TARGET_IMAGE = '025.tiff'  # 실험 후 조정 가능

# ============= 모델 설정 =============
MODEL_CONFIG = {
    'architecture': 'resnet18',
    'pretrained': True,
    'num_classes': 2,
    'freeze_backbone': False
}

TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'optimizer': 'Adam',
    'lr_scheduler': 'ReduceLROnPlateau',
    'early_stopping_patience': 10,
    'num_workers': 4,
    'pin_memory': False
}

# ============= 평가 지표 =============
METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'auroc', 'auprc']
BOOTSTRAP_CONFIG = {'n_iterations': 1000, 'confidence_level': 0.95}

# ============= 통계 검정 =============
STATISTICS_CONFIG = {
    'alpha': 0.05,
    'h1_test': 'one_way_anova',
    'h1_posthoc': 'tukey_hsd',
    'h2_test': 'wilcoxon_signed_rank',
    'h2_effect_size': 'cohens_d'
}

# ============= 하드웨어 설정 =============
DEVICE = 'cpu'
NUM_THREADS = 8
torch.set_num_threads(NUM_THREADS)

# ============= 로깅 =============
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s'
}

set_seed(RANDOM_SEED)