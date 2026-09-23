import json
from pathlib import Path
import numpy as np
from PIL import Image
from Projeto_Solar_RNA.network_config import load_network_config

IMAGE_SHAPE = (40, 24)
NUM_PIXELS = 960
NUM_FEATURES = load_network_config().num_inputs

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
    def __init__(self, base_path=None, config=None):
        self.config = config if config is not None else load_network_config()
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
        with Image.open(img_path) as img:
            arr = np.array(img.convert('L'), dtype=np.float32) / 255.0
        if arr.shape != IMAGE_SHAPE:
            raise ValueError(f"Imagem {key}: formato {arr.shape}, esperado {IMAGE_SHAPE}")
        
        if cache:
            self.cached_images[key] = arr
        return arr

    def get_feature_vector(self, index, mode='grayscale'):
        return self.build_feature_vector(self.load_image(index), mode=mode, config=self.config)

    @staticmethod
    def build_feature_vector(image, mode='grayscale', config=None):
        config = config if config is not None else load_network_config()
        image = np.asarray(image, dtype=np.float32)
        if image.shape != IMAGE_SHAPE:
            raise ValueError(f"Formato {image.shape}, esperado {IMAGE_SHAPE}")
        if mode == 'bipolar':
            image = np.where(image >= 0.5, 1.0, -1.0).astype(np.float32)
        elif mode == 'binary':
            image = (image >= 0.5).astype(np.float32)
        mask_rows, mask_columns = config.mask_shape
        grid_rows, grid_columns = config.mask_grid
        masks = image.reshape(grid_rows, mask_rows, grid_columns, mask_columns).mean(axis=(1, 3))
        return np.concatenate((image.ravel(), image.mean(axis=1), image.mean(axis=0), masks.ravel()))

    def export_intensity_statistics(self, output_path=None):
        if not self.image_keys:
            raise ValueError("Nenhuma imagem disponivel para calcular estatisticas")
        if output_path is None:
            output_path = Path(__file__).resolve().parent / 'resultados' / 'estatisticas_imagens.json'
        output_path = Path(output_path)
        records = []
        for index, key in enumerate(self.image_keys):
            image = self.load_image(index, cache=False)
            minimum = int(round(float(image.min()) * 255))
            maximum = int(round(float(image.max()) * 255))
            records.append({
                'id': key,
                'class': IDX_TO_CLASS[int(self.labels[index])],
                'min': minimum,
                'max': maximum,
                'min_normalized': minimum / 255.0,
                'max_normalized': maximum / 255.0,
            })
        darkest = min(record['min'] for record in records)
        lightest = max(record['max'] for record in records)
        statistics = {
            'image_count': len(records),
            'image_shape': list(IMAGE_SHAPE),
            'intensity_scale': [0, 255],
            'darkest': {
                'value': darkest,
                'normalized_value': darkest / 255.0,
                'images': [{'id': record['id'], 'class': record['class']}
                           for record in records if record['min'] == darkest],
            },
            'lightest': {
                'value': lightest,
                'normalized_value': lightest / 255.0,
                'images': [{'id': record['id'], 'class': record['class']}
                           for record in records if record['max'] == lightest],
            },
            'images': records,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as output_file:
            json.dump(statistics, output_file, indent=2, ensure_ascii=False)
            output_file.write('\n')
        return output_path

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
