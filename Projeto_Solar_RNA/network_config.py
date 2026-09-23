import json
from dataclasses import dataclass
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name('network_config.json')


@dataclass(frozen=True)
class NetworkConfig:
    image_shape: tuple
    mask_shape: tuple
    hidden_grid: tuple
    output_grid: tuple
    activation: str
    seed: int

    def __post_init__(self):
        for name in ('image_shape', 'mask_shape', 'hidden_grid', 'output_grid'):
            shape = getattr(self, name)
            if len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
                raise ValueError(f'{name}: esperado [linhas, colunas] com inteiros positivos')
        if self.image_shape != (40, 24):
            raise ValueError('O banco de imagens requer image_shape [40, 24]')
        if any(size % mask for size, mask in zip(self.image_shape, self.mask_shape)):
            raise ValueError('mask_shape deve dividir image_shape sem sobras')
        if self.output_grid != (12, 1):
            raise ValueError('As 12 classes requerem output_grid [12, 1]')
        if self.activation not in ('relu', 'sigmoid'):
            raise ValueError('activation deve ser relu ou sigmoid')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed deve ser um inteiro nao negativo')

    @property
    def mask_grid(self):
        return tuple(size // mask for size, mask in zip(self.image_shape, self.mask_shape))

    @property
    def num_inputs(self):
        rows, columns = self.image_shape
        mask_rows, mask_columns = self.mask_grid
        return rows * columns + rows + columns + mask_rows * mask_columns

    @property
    def num_hidden(self):
        return self.hidden_grid[0] * self.hidden_grid[1]

    @property
    def num_classes(self):
        return self.output_grid[0] * self.output_grid[1]


def load_network_config(path=None):
    with Path(path if path is not None else CONFIG_PATH).open(encoding='utf-8') as config_file:
        values = json.load(config_file)
    for name in ('image_shape', 'mask_shape', 'hidden_grid', 'output_grid'):
        if not isinstance(values.get(name), list):
            raise ValueError(f'{name}: esperado um array de duas dimensoes')
        values[name] = tuple(values[name])
    return NetworkConfig(**values)