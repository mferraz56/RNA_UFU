# Projeto Solar RNA V2

Versao do projeto que utiliza uma CNN pequena em PyTorch para classificar as 12 classes termicas do dataset InfraredSolarModules.

## Estrutura

```text
Projeto_Solar_RNA-V2/
├── experiment.json
├── data.py
├── model.py
├── metrics.py
├── train.py
├── baseline.py
├── gradcam.py
├── test_v2.py
└── resultados/
```

## Configuracao

O arquivo `experiment.json` controla seed, formato das imagens, divisao dos dados, hiperparametros e augmentation. O parser rejeita JSON invalido, dimensoes incorretas, razoes de divisao inconsistentes e configuracoes fora do contrato.

O treinamento usa:

- divisao estratificada em 70% treino, 15% validacao e 15% teste;
- seed fixa;
- normalizacao dos pixels em [0, 1];
- pesos de classe calculados pelo desbalanceamento;
- Adam;
- early stopping pela perda de validacao;
- restauracao dos melhores pesos;
- Grad-CAM para explicar a predicao.

## Instalacao

Na raiz de `RNA_UFU`, com o ambiente virtual ativo:

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r '.\Projeto_Solar_RNA-V2\requirements.txt'
```

## Testes sinteticos

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s '.\Projeto_Solar_RNA-V2' -p 'test_v2.py' -v
```

## Baseline linear

```powershell
& '.\.venv\Scripts\python.exe' '.\Projeto_Solar_RNA-V2\baseline.py'
```

## Treinamento curto para validacao

```powershell
& '.\.venv\Scripts\python.exe' '.\Projeto_Solar_RNA-V2\train.py' --epochs 2 --max-train-samples 500
```

## Treinamento completo

```powershell
& '.\.venv\Scripts\python.exe' '.\Projeto_Solar_RNA-V2\train.py'
```

Os resultados completos ficam em `resultados/`: checkpoint, historico, contagens por classe e metricas do teste.

## Grad-CAM

```powershell
& '.\.venv\Scripts\python.exe' '.\Projeto_Solar_RNA-V2\gradcam.py' --checkpoint '.\Projeto_Solar_RNA-V2\resultados\cnn_solar_v2.pt' --index 0
```
