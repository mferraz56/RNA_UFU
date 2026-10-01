"""Executa o Trabalho 06 e exporta predicoes, pesos e graficos do ajuste."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mlp import INPUTS, TARGETS, EpochResult, MLPRegressor


def plot_results(model: MLPRegressor, history: list[EpochResult]):
    """Compara a curva da MLP aos pontos conhecidos e mostra erro e residuos."""
    import matplotlib.pyplot as plt

    grid = np.linspace(0, 1, 501)
    predictions = model.predict(INPUTS)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
    figure.suptitle(f'Trabalho 06 | Aproximacao funcional | MLP 1-{model.hidden_size}-1')
    axes[0].plot(grid, model.predict(grid), color='#087f70', label='Aproximacao da MLP')
    axes[0].scatter(INPUTS, TARGETS, color='#c64b64', marker='o', zorder=3, label='11 pontos fornecidos')
    axes[0].set(title='Ajuste no intervalo [0, 1]', xlabel='x', ylabel='t / saida da rede')
    axes[0].legend(fontsize=8)
    epochs = [entry.epoch for entry in history]
    losses = [max(entry.mse, np.finfo(float).tiny) for entry in history]
    axes[1].semilogy(epochs, losses, color='#394b9a')
    axes[1].set(title='EQM nos pontos de treino', xlabel='Epoca', ylabel='Erro quadratico medio (log)')
    axes[2].axhline(0, color='#555555', linewidth=1)
    axes[2].bar(INPUTS, predictions - TARGETS, width=0.06, color='#c64b64')
    axes[2].set(title='Residuos nos pontos fornecidos', xlabel='x', ylabel='Saida da rede - alvo')
    for axis in axes:
        axis.grid(alpha=0.25)
    return figure


def run_experiment(output: Path, hidden_size: int = 10, learning_rate: float = 0.05,
                   epochs: int = 100000, tolerance: float = 1e-4, seed: int = 42,
                   show: bool = True) -> dict:
    """Treina com os 11 pontos; as metricas exportadas nao sao de teste independente."""
    model = MLPRegressor(hidden_size, learning_rate, seed)
    initial_parameters = [parameter.tolist() for parameter in model.parameters]
    history = model.fit(INPUTS, TARGETS, epochs, tolerance)
    predictions = model.predict(INPUTS)
    residuals = predictions - TARGETS
    mse = float(np.mean(residuals ** 2))
    converged = mse <= tolerance
    summary = {
        'architecture': [1, hidden_size, 1],
        'hidden_activation': 'tanh', 'output_activation': 'linear',
        'input_transform': '2*x-1', 'optimizer': 'batch_gradient_descent',
        'learning_rate': learning_rate, 'seed': seed,
        'max_epochs': epochs, 'epochs_completed': history[-1].epoch,
        'target_mse': tolerance, 'target_reached': converged,
        'stop_reason': 'target_mse' if converged else 'max_epochs',
        'evaluation': 'training_points', 'independent_test': False,
        'initial_mse': history[0].mse, 'final_mse': mse,
        'rmse': float(np.sqrt(mse)), 'mae': float(np.mean(np.abs(residuals))),
        'max_absolute_error': float(np.max(np.abs(residuals))),
        'sse': float(np.sum(residuals ** 2)),
        'inputs': INPUTS.tolist(), 'targets': TARGETS.tolist(),
        'parameter_order': ['hidden_weights', 'hidden_bias', 'output_weights', 'output_bias'],
        'initial_parameters': initial_parameters,
        'parameters': {name: parameter.tolist() for name, parameter in zip(
            ('hidden_weights', 'hidden_bias', 'output_weights', 'output_bias'), model.parameters)},
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'predicoes.csv').open('w', newline='', encoding='utf-8') as destination:
        writer = csv.writer(destination)
        writer.writerow(['x', 'target', 'prediction', 'residual', 'squared_error'])
        writer.writerows(zip(INPUTS, TARGETS, predictions, residuals, residuals ** 2))
    with (output / 'historico.csv').open('w', newline='', encoding='utf-8') as destination:
        writer = csv.DictWriter(destination, fieldnames=['epoch', 'mse'])
        writer.writeheader()
        writer.writerows(asdict(entry) for entry in history)
    grid = np.linspace(0, 1, 501)
    with (output / 'curva.csv').open('w', newline='', encoding='utf-8') as destination:
        writer = csv.writer(destination)
        writer.writerow(['x', 'prediction'])
        writer.writerows(zip(grid, model.predict(grid)))
    (output / 'modelo.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    figure = plot_results(model, history)
    figure.savefig(output / 'resultado.png', dpi=160)
    print(f'MLP: 1 -> {hidden_size} -> 1 | tanh -> linear | gradiente descendente em lote')
    print(f'Epocas: {history[-1].epoch}/{epochs} | taxa: {learning_rate} | semente: {seed}')
    print(f'EQM inicial: {history[0].mse:.8f} | EQM final: {mse:.8f}')
    print(f'RMSE: {summary["rmse"]:.8f} | maior erro absoluto: {summary["max_absolute_error"]:.8f}')
    print(f'Meta de EQM {tolerance:g}: {"atingida" if converged else "NAO atingida (limite de epocas)"}')
    print('\nReavaliacao dos pontos de treino (nao e teste independente)')
    print('   x    |    alvo    |  previsao  |   residuo')
    for value, target, prediction, residual in zip(INPUTS, TARGETS, predictions, residuals):
        print(f'{value:7.2f} | {target:+10.6f} | {prediction:+10.6f} | {residual:+10.6f}')
    print(f'\nResultados: {output.resolve()}')
    import matplotlib.pyplot as plt

    if show:
        plt.show()
    plt.close(figure)
    return summary


def main() -> None:
    """Oferece execucao grafica ou headless e parametros reproduziveis pelo terminal."""
    parser = argparse.ArgumentParser(description='Trabalho 06: aproximacao funcional com MLP.')
    parser.add_argument('--ocultos', type=int, default=10)
    parser.add_argument('--taxa', type=float, default=0.05)
    parser.add_argument('--epocas', type=int, default=100000)
    parser.add_argument('--tolerancia', type=float, default=1e-4)
    parser.add_argument('--semente', type=int, default=42)
    parser.add_argument('--saida', type=Path, default=Path(__file__).resolve().parent / 'resultados')
    parser.add_argument('--sem-janela', action='store_true', help='Salva o grafico sem abrir uma janela.')
    args = parser.parse_args()
    if args.sem_janela:
        import matplotlib

        matplotlib.use('Agg')
    try:
        run_experiment(args.saida, args.ocultos, args.taxa, args.epocas, args.tolerancia,
                       args.semente, show=not args.sem_janela)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Erro: {error}\n')


if __name__ == '__main__':
    main()