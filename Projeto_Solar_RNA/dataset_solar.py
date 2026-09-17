import json
from pathlib import Path
import numpy as np
from PIL import Image

CLASSES = [
    'No-Anomaly',
    'Cell',
    'Cell-Multi',
    'Cracking',
    'Hot-Spot',
    'Hot-Spot-Multi',
    'Shadowing',
    'Diode',
    'Diode-Multi',
    'Vegetation',
    'Soiling',
    'Offline-Module'
]

# Semantic domain grouping for PV solar anomalies
CLASS_GROUPS = {
    'Normal': ['No-Anomaly'],
    'Célula': ['Cell', 'Cell-Multi', 'Cracking', 'Hot-Spot', 'Hot-Spot-Multi'],
    'Elétrica': ['Diode', 'Diode-Multi', 'Offline-Module'],
    'Externa': ['Shadowing', 'Vegetation', 'Soiling']
}

GROUP_COLORS = {
    'Normal': '#2ecc71',     # Green
    'Célula': '#e74c3c',     # Crimson / Red
    'Elétrica': '#f39c12',   # Amber / Orange
    'Externa': '#9b59b6'     # Purple
}

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASSES)}

# Get group name for a class
def get_class_group(cls_name):
    for grp, cls_list in CLASS_GROUPS.items():
        if cls_name in cls_list:
            return grp
    return 'Outros'


class SolarDataset:
    def __init__(self, base_path=None):
        if base_path is None:
            root = Path(__file__).resolve().parent.parent
            base_path = root / 'Dataset' / '2020-02-14_InfraredSolarModules' / 'InfraredSolarModules'
        else:
            base_path = Path(base_path)
            
        self.base_path = base_path
        self.metadata_path = base_path / 'module_metadata.json'
        self.images_dir = base_path / 'images'
        
        self.raw_metadata = {}
        self.image_keys = []
        self.labels = []
        self.filepaths = []
        self.cached_images = {}  # key -> np.ndarray (40, 24) float32 [0..1]
        
        self.train_indices = np.array([], dtype=np.int64)
        self.val_indices = np.array([], dtype=np.int64)
        self.test_indices = np.array([], dtype=np.int64)
        
        self._load_metadata()

    def _load_metadata(self):
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadados não encontrados em: {self.metadata_path}")
            
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.raw_metadata = json.load(f)
            
        for key, info in self.raw_metadata.items():
            cls_name = info.get('anomaly_class')
            if cls_name in CLASS_TO_IDX:
                rel_path = info.get('image_filepath')
                full_path = self.base_path / rel_path
                if full_path.exists():
                    self.image_keys.append(key)
                    self.labels.append(CLASS_TO_IDX[cls_name])
                    self.filepaths.append(full_path)
                    
        self.labels = np.array(self.labels, dtype=np.int64)

    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        rng = np.random.default_rng(seed)
        train_idx, val_idx, test_idx = [], [], []
        
        for cls_idx in range(len(CLASSES)):
            cls_matches = np.where(self.labels == cls_idx)[0]
            rng.shuffle(cls_matches)
            
            n_cls = len(cls_matches)
            n_train = int(n_cls * train_ratio)
            n_val = int(n_cls * val_ratio)
            
            train_idx.extend(cls_matches[:n_train])
            val_idx.extend(cls_matches[n_train:n_train + n_val])
            test_idx.extend(cls_matches[n_train + n_val:])
            
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)
        
        self.train_indices = np.array(train_idx, dtype=np.int64)
        self.val_indices = np.array(val_idx, dtype=np.int64)
        self.test_indices = np.array(test_idx, dtype=np.int64)
        
        return self.train_indices, self.val_indices, self.test_indices

    def load_image(self, index, cache=True):
        key = self.image_keys[index]
        if cache and key in self.cached_images:
            return self.cached_images[key]
            
        img_path = self.filepaths[index]
        img = Image.open(img_path).convert('L')  # Convert to Grayscale
        arr = np.array(img, dtype=np.float32) / 255.0  # Shape (40, 24)
        
        if cache:
            self.cached_images[key] = arr
        return arr

    def get_feature_vector(self, index, mode='grayscale'):
        img_2d = self.load_image(index)
        vec = img_2d.flatten()  # 960 elements
        
        if mode == 'bipolar':
            return np.where(vec >= 0.5, 1.0, -1.0)
        elif mode == 'binary':
            return (vec >= 0.5).astype(np.float32)
        else:
            return vec

    def preload_subset(self, indices, max_count=1000):
        sub = indices[:max_count]
        for idx in sub:
            self.load_image(idx, cache=True)

    def get_class_counts(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        res = {CLASSES[idx]: 0 for idx in range(len(CLASSES))}
        for idx, count in zip(unique, counts):
            res[CLASSES[idx]] = int(count)
        return res

    # --- SYNTHETIC FAULT INJECTION (For Interactive PV Simulation) ---
    @staticmethod
    def generate_synthetic_anomaly(anomaly_type='Hot-Spot', shape=(40, 24)):
        """Generates realistic thermal pattern on 24x40 grid."""
        grid = np.full(shape, 0.25, dtype=np.float32)  # Nominal cool background
        
        if anomaly_type == 'Hot-Spot':
            # Single bright cell hotspot
            r, c = np.random.randint(10, 30), np.random.randint(5, 19)
            grid[r-2:r+3, c-2:c+3] = 0.95
            
        elif anomaly_type == 'Diode':
            # 1/3 of module heated due to active bypass diode (rows 0 to 13)
            grid[0:13, :] = 0.85
            
        elif anomaly_type == 'Offline-Module':
            # Entire module overheated
            grid[:, :] = 0.90
            
        elif anomaly_type == 'Shadowing':
            # Diagonal corner shadow mask
            for r in range(15):
                for c in range(15 - r):
                    grid[r, c] = 0.05
                    
        elif anomaly_type == 'Vegetation':
            # Irregular cold obstruction block at bottom
            grid[28:40, 0:16] = 0.02
            
        elif anomaly_type == 'Soiling':
            # Dust noise across surface
            noise = np.random.uniform(-0.1, 0.1, size=shape).astype(np.float32)
            grid = np.clip(grid + noise, 0.1, 0.4)

        return grid
