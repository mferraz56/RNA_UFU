import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from adaline import Adaline, load_dataset, split_indices


def plot_results(features, targets, train, test, model, history, matrix, independent):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    figure.suptitle('Trabalho 5 / Adaline / Base B2', fontsize=16)
    epochs = [entry['epoch'] for entry in history]
    axes[0].plot(epochs, [entry['online_sse'] for entry in history],
                 label='Acumulado nas atualizacoes (aula)', color='#c64b64')
    axes[0].plot(epochs, [entry['sse'] for entry in history],
                 label='Recalculado ao fim da epoca', color='#087f70')
    axes[0].set(title='Erro quadratico total / treino', xlabel='Epoca', ylabel='Soma dos erros ao quadrado')
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    for label, color in ((-1, '#c64b64'), (1, '#087f70')):
        indices = train[targets[train] == label]
        axes[1].scatter(features[indices, 0], features[indices, 1], color=color,
                        marker='o', label=f'Treino {label:+d}')
        if independent:
            indices = test[targets[test] == label]
            axes[1].scatter(features[indices, 0], features[indices, 1], color=color,
                            marker='X', s=85, label=f'Teste {label:+d}')
    lower = features.min(axis=0) - 0.3
    upper = features.max(axis=0) + 0.3
    horizontal, vertical = np.meshgrid(np.linspace(lower[0], upper[0], 180),
                                       np.linspace(lower[1], upper[1], 180))
    scores = model.decision_function(np.column_stack((horizontal.ravel(), vertical.ravel()))).reshape(horizontal.shape)
    if scores.min() < 0 < scores.max():
        axes[1].contour(horizontal, vertical, scores, levels=[0], colors=['#394b9a'], linewidths=2)
    axes[1].set(title='Fronteira: w1*s1 + w2*s2 + b = 0', xlabel='s1', ylabel='s2',
                xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[2].imshow(matrix, cmap='Blues', vmin=0)
    title = 'Teste separado' if independent else 'Reavaliacao do treino (nao independente)'
    axes[2].set(title=title, xlabel='Classe prevista', ylabel='Classe esperada',
                xticks=[0, 1], yticks=[0, 1], xticklabels=['-1', '+1'], yticklabels=['-1', '+1'])
    for row in range(2):
        for column in range(2):
            axes[2].text(column, row, str(matrix[row, column]), ha='center', va='center',
                         fontsize=20, color='white' if matrix[row, column] > matrix.max() / 2 else 'black')
    return figure


def run_experiment(base, output, learning_rate=0.01, epochs=200, test_size=0.3, seed=42, show=True):
    features, targets = load_dataset(base)
    train, test = split_indices(targets, test_size, seed)
    model = Adaline(learning_rate=learning_rate, seed=seed)
    initial_weights, initial_bias = model.weights.copy(), model.bias
    initial_sse = float(np.sum((targets[train] - model.decision_function(features[train])) ** 2))
    if show:
        from interface import run_training

        history = run_training(model, features[train], targets[train], epochs)
        if history is None:
            print('Treinamento encerrado sem salvar. Resultados anteriores foram preservados.')
            return None
    else:
        history = model.fit(features[train], targets[train], epochs=epochs)
    linear_outputs = model.decision_function(features[test])
    predictions = model.predict(features[test])
    accuracy = float(np.mean(predictions == targets[test]))
    train_accuracy = float(np.mean(model.predict(features[train]) == targets[train]))
    matrix = confusion_matrix(targets[test], predictions, labels=[-1, 1])
    independent = test_size > 0
    evaluation = 'Teste separado' if independent else 'Reavaliacao do treino (nao e teste independente)'
    print(f'Base: {Path(base).name} | treino: {len(train)} | avaliacao: {len(test)}')
    print(f'Taxa: {learning_rate} | epocas: {epochs} | semente: {seed}')
    print(f'Pesos: {model.weights} | bias: {model.bias:.6f}')
    print(f'EQT inicial: {initial_sse:.6f} | EQT final: {history[-1]["sse"]:.6f}')
    print(f'EQT acumulado na ultima epoca: {history[-1]["online_sse"]:.6f}')
    print(f'Acuracia no treino: {train_accuracy:.2%}')
    print(f'\n{evaluation}\nLinha CSV |   s1    |   s2    | esperado | saida linear | previsto | acerto')
    rows = []
    for index, score, prediction in zip(test, linear_outputs, predictions):
        correct = bool(prediction == targets[index])
        rows.append({'csv_line': int(index + 2), 's1': float(features[index, 0]),
                     's2': float(features[index, 1]), 'target': int(targets[index]),
                     'linear_output': float(score), 'prediction': int(prediction), 'correct': correct})
        print(f'{index + 2:9d} | {features[index, 0]:7.3f} | {features[index, 1]:7.3f} | '
              f'{targets[index]:+8d} | {score:+12.6f} | {prediction:+8d} | {"sim" if correct else "nao"}')
    print(f'Acuracia da avaliacao: {accuracy:.2%} ({np.count_nonzero(predictions == targets[test])}/{len(test)})')
    print(f'Matriz de confusao (ordem -1, +1; linhas=real, colunas=previsto):\n{matrix}')
    summary = {'dataset': str(Path(base).resolve()), 'seed': seed, 'learning_rate': learning_rate,
               'epochs': epochs, 'test_fraction': test_size, 'independent_test': independent,
               'input_transform': 'none; continuous s1 and s2', 'threshold': 0.0,
               'initial_weights': initial_weights.tolist(), 'initial_bias': initial_bias,
               'weights': model.weights.tolist(), 'bias': float(model.bias),
               'initial_sse': initial_sse, 'final_sse': history[-1]['sse'],
               'train_accuracy': train_accuracy, 'evaluation_accuracy': accuracy,
               'confusion_labels': [-1, 1], 'confusion_matrix': matrix.tolist(),
               'train_csv_lines': (train + 2).tolist(), 'test_csv_lines': (test + 2).tolist()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'predicoes.csv').open('w', newline='', encoding='utf-8') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'historico.csv').open('w', newline='', encoding='utf-8') as destination:
        writer = csv.writer(destination)
        writer.writerow(['epoch', 'online_sse', 'sse', 'w1', 'w2', 'bias'])
        for entry in history:
            writer.writerow([entry['epoch'], entry['online_sse'], entry['sse'], *entry['weights'], entry['bias']])
    (output / 'modelo.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    figure = plot_results(features, targets, train, test, model, history, matrix, independent)
    figure.savefig(output / 'resultado.png', dpi=160)
    print(f'\nArquivos salvos em: {output.resolve()}')
    import matplotlib.pyplot as plt

    if show:
        plt.show()
    plt.close(figure)
    return summary


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Trabalho 5: classificacao de padroes usando Adaline.')
    parser.add_argument('--base', type=Path, default=folder / 'Basedados_B2.csv')
    parser.add_argument('--saida', type=Path, default=folder / 'resultados')
    parser.add_argument('--taxa', type=float, default=0.01)
    parser.add_argument('--epocas', type=int, default=200)
    parser.add_argument('--fracao-teste', type=float, default=0.3,
                        help='Fracao reservada para teste; 0 reavalia o proprio treino.')
    parser.add_argument('--semente', type=int, default=42)
    parser.add_argument('--sem-janela', action='store_true', help='Salva o grafico sem abrir uma janela.')
    args = parser.parse_args()
    if args.sem_janela:
        import matplotlib

        matplotlib.use('Agg')
    try:
        run_experiment(args.base, args.saida, args.taxa, args.epocas, args.fracao_teste,
                       args.semente, show=not args.sem_janela)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Erro: {error}\n')


if __name__ == '__main__':
    main()