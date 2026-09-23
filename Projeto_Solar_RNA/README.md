# RNA Solar - Diagnóstico de Falhas em Módulos Fotovoltaicos

## Visão geral

Este projeto implementa uma aplicação interativa para classificação de falhas térmicas em módulos fotovoltaicos a partir de imagens infravermelhas. A ideia central é usar redes neurais artificiais para analisar imagens térmicas de painéis solares e identificar anomalias como:

- ausência de defeito
- células defeituosas
- hotspots
- diodos com falha
- sombreamento
- vegetação ou obstrução externa
- sujeira/soiling
- módulos offline

O projeto combina:

- carregamento e organização do dataset
- pré-processamento das imagens
- modelos de redes neurais
- visualização espacial da rede
- interface gráfica para inspeção e treinamento
- métricas de avaliação e exportação de resultados

O objetivo didático é permitir visualizar como uma rede neural interpreta padrões espaciais em uma imagem térmica e como as decisões são tomadas por classe.

---

## Contexto do problema

Painéis fotovoltaicos podem sofrer falhas que alteram a distribuição de temperatura na superfície do módulo. Essas anomalias aparecem como regiões mais quentes ou mais frias em imagens térmicas. Identificar corretamente o tipo de falha é importante para:

- manutenção preditiva
- redução de perdas de geração
- prevenção de falhas estruturais
- diagnóstico rápido em inspeções de campo

A rede neural do projeto aprende a separar padrões térmicos e associá-los a classes conhecidas de falhas.

---

## Dataset

O projeto usa um conjunto de imagens térmicas do dataset InfraredSolarModules, localizado em:

- `Dataset/2020-02-14_InfraredSolarModules/InfraredSolarModules`

O dataset contém:

- metadados em `module_metadata.json`
- imagens em `images/`
- informações de classe por imagem, como `anomaly_class`

Cada imagem está em formato térmico e é normalizada para uma matriz 40x24, com valores entre 0 e 1.

### Classes disponíveis

O conjunto de classes é:

1. `No-Anomaly`
2. `Cell`
3. `Cell-Multi`
4. `Cracking`
5. `Hot-Spot`
6. `Hot-Spot-Multi`
7. `Shadowing`
8. `Diode`
9. `Diode-Multi`
10. `Vegetation`
11. `Soiling`
12. `Offline-Module`

Além disso, o código agrupa semanticamente as classes em grupos:

- `Normal`: `No-Anomaly`
- `Célula`: `Cell`, `Cell-Multi`, `Cracking`, `Hot-Spot`, `Hot-Spot-Multi`
- `Elétrica`: `Diode`, `Diode-Multi`, `Offline-Module`
- `Externa`: `Shadowing`, `Vegetation`, `Soiling`

Esses agrupamentos ajudam a interpretar o diagnóstico em níveis mais gerais, em vez de apenas classes muito específicas.

---

## Estrutura do projeto

A estrutura principal do projeto é:

```text
RNA_UFU/
├── Dataset/
│   └── 2020-02-14_InfraredSolarModules/
│       └── InfraredSolarModules/
│           ├── images/
│           └── module_metadata.json
├── Projeto_Solar_RNA/
│   ├── __init__.py
│   ├── dataset_solar.py
│   ├── neural_network.py
│   ├── interface_solar.py
│   ├── visualization_solar.py
│   ├── test_solar_rna.py
│   └── README.md
└── .venv/
```

### Arquivos principais

#### `dataset_solar.py`
Responsável por:

- carregar os metadados do dataset
- validar as imagens existentes
- transformar cada imagem em matriz 40x24
- criar vetores de entrada para a rede
- dividir os dados em treino, validação e teste
- gerar falhas sintéticas para simulação interativa

#### `neural_network.py`
Responsável pelo modelo de rede neural:

- `SingleLayerPerceptron`
- `MLPSolarNetwork`
- funções auxiliares como softmax, ReLU e funções de derivada
- função `evaluate_network()` para cálculo de acurácia e perda

#### `interface_solar.py`
Responsável pela interface gráfica em Tkinter.

Ela oferece:

- painel térmico para visualização da imagem
- seleção de amostra do dataset
- categoria de parte do dataset (treino, validação ou teste)
- classificação em tempo real
- treinamento interativo
- gráfico de perda e acurácia
- matriz de confusão
- explorador de campos receptivos
- exportação de dados

#### `visualization_solar.py`
Responsável pela visualização da arquitetura da rede:

- entrada espacial em grade 24x40
- camada oculta configurável, inicialmente 64x1 (64 neurônios)
- camada de saída 12x1 com 12 classes
- médias das 15 máscaras 8x8 em uma grade 5x3
- conexões entre camadas
- seleção visual de pixel, neurônio oculto e classe de saída

#### `test_solar_rna.py`
Contém testes unitários para garantir que:

- o dataset é carregado corretamente
- as classes estão consistentes
- as falhas sintéticas têm o formato esperado
- o perceptron funciona
- o MLP funciona
- a avaliação retorna valores válidos

---

## Como o processamento de imagem funciona

Cada imagem térmica do dataset é convertida em escala de cinza e depois normalizada.

### Transformação

- imagem original: matriz 2D com intensidade térmica
- conversão para escala cinza: `convert('L')`
- normalização: `float32 / 255.0`
- resultado: matriz 40x24 com valores entre 0 e 1

### Vetorização

A entrada tem 1.039 valores `float32`, nesta ordem:

- posições 0 a 959: os 960 pixels, linha por linha
- posições 960 a 999: as 40 médias das linhas
- posições 1000 a 1023: as 24 médias das colunas
- posições 1024 a 1038: as 15 médias de máscaras 8x8

As máscaras não se sobrepõem: são 3 na horizontal e 5 na vertical, cobrindo
os 960 pixels. Cada média usa os 64 pixels do bloco. A ordem é da esquerda
para a direita, depois de cima para baixo: bloco (0, 0), (0, 1), (0, 2),
(1, 0), ..., (4, 2). Não há padding nem descarte de bordas.

As médias são calculadas sobre os pixels normalizados, sem normalização
individual por imagem. Treino, avaliação, desenho e simulação usam a mesma
função `SolarDataset.build_feature_vector()`.

A entrada pode ser usada em três modos:

- `grayscale`: valores contínuos em [0, 1]
- `binary`: valores binários conforme limiar 0.5
- `bipolar`: valores em {-1, 1}

No código, o modo padrão é `grayscale`. Nos modos `binary` e `bipolar`, os
pixels são convertidos antes do cálculo das médias; as médias continuam
podendo ter valores fracionários.

### Estatísticas de tonalidade

Ao abrir a interface, o banco inteiro é percorrido e o arquivo
[resultados/estatisticas_imagens.json](resultados/estatisticas_imagens.json)
é atualizado. Esse processamento pode acrescentar alguns segundos à abertura.

- `darkest` e `lightest`: valor extremo global, valor normalizado e lista
	de todos os IDs e classes que apresentam esse extremo, incluindo empates.
- `images`: uma entrada por imagem, com `id`, `class`, `min`, `max`,
	`min_normalized` e `max_normalized`.
- `image_count`, `image_shape` e `intensity_scale`: quantidade, formato e escala.

Os extremos são intensidades em escala de cinza de 0 a 255, não temperaturas
físicas nem médias de brilho. A normalização continua sendo divisão por 255;
as estatísticas globais não são usadas para ajustar os dados de treino.

Para gerar o JSON sem abrir a interface, a partir da raiz do workspace:

```powershell
.\.venv\Scripts\python.exe -c "from Projeto_Solar_RNA.dataset_solar import SolarDataset; print(SolarDataset().export_intensity_statistics())"
```

---

## Modelos implementados

### Configuração compartilhada

O arquivo [network_config.json](network_config.json) é a fonte de configuração
do dataset, da rede e do dashboard:

```json
{
	"image_shape": [40, 24],
	"mask_shape": [8, 8],
	"hidden_grid": [64, 1],
	"output_grid": [12, 1],
	"activation": "relu",
	"seed": 42
}
```

Todas as dimensões seguem `[linhas, colunas]`. A quantidade de entradas é
calculada automaticamente pela soma dos pixels e das médias das linhas,
colunas e máscaras. Não é preciso informar `num_inputs` manualmente.

- `hidden_grid`: muda a quantidade de neurônios e sua disposição visual.
- `mask_shape`: dimensões dos blocos; devem dividir 40x24 sem sobras.
- `activation`: `relu` ou `sigmoid` para a camada oculta.
- `seed`: inicialização dos pesos e divisão dos dados.
- `image_shape` e `output_grid`: ficam em `[40, 24]` e `[12, 1]`, pois
	correspondem ao formato e às 12 classes deste banco. Valores incompatíveis
	são rejeitados.

Após editar o JSON, feche e reabra o aplicativo. A configuração é lida na
inicialização; reset e troca de modelo reutilizam essa configuração. Alterar
a arquitetura exige novo treinamento. O JSON de configuração é separado
do relatório de estatísticas das imagens.

### 1. Perceptron simples

O `SingleLayerPerceptron` faz a classificação direta de uma imagem em uma das 12 classes de saída. Ele não tem camada oculta.

#### Estrutura

- Entrada: 1.039 valores
- Saída: 12 neurônios

#### Processo

- cálculo dos scores
- aplicação de `softmax`
- escolha da classe com maior probabilidade
- atualização dos pesos quando há erro

Esse modelo serve como referência didática de classificação linear.

### 2. MLP com camada oculta espacial

O `MLPSolarNetwork` usa uma arquitetura mais rica:

- entrada: 1.039 valores (960 pixels + 40 médias de linhas + 24 médias de colunas + 15 médias de máscaras)
- camada oculta: grade 64x1 = 64 neurônios, com ativação ReLU
- saída: grade 12x1 = 12 classes

#### Observações

- A grade 64x1 é uma organização visual dos 64 neurônios
- cada neurônio oculto está conectado a todas as 1.039 entradas
- a camada de saída produz a probabilidade de cada classe

As matrizes de pesos têm formatos `(64, 1039)` e `(12, 64)`. Os campos
receptivos 40x24 incorporam os pesos das médias, dividindo o peso de cada
linha entre seus 24 pixels, o de cada coluna entre seus 40 pixels e o de cada
máscara entre seus 64 pixels.

O dashboard mostra uma coluna com 64 neurônios, com espaçamento fixo e
rolagem vertical. O hover mostra a ativação e o clique seleciona o neurônio.
Os limites das máscaras são destacados na imagem e suas médias aparecem
em uma grade 5x3 abaixo dela. Para grades com mais de 256 neurônios, a
visualização usa um mapa de cores e limita as conexões destacadas ao neurônio
selecionado. As barras de rolagem mantêm o diagrama acessível.

Grades maiores aumentam o custo de memória e treinamento. Para reduzir a sobrecarga de threads em
operações pequenas, pode-se executar `$env:OPENBLAS_NUM_THREADS = "1"` antes
de iniciar a aplicação no PowerShell.

A nova arquitetura precisa ser treinada e comparada em validação/teste;
a adição das médias não garante melhora de acurácia.

---

## Função de perda e treinamento

O treinamento usa perda de cross-entropy.

A lógica principal é:

- calcular logits no forward pass
- obter probabilidades por softmax
- calcular perda para a classe correta
- atualizar pesos por gradiente descendente

A atualização inclui:

- `W1`, `b1` para a camada oculta
- `W2`, `b2` para a camada final

No perceptron simples, a atualização também é feita pela regra de correção de erro, com o ajuste dos pesos para a classe correta e a classe predita.

---

## Simulador de falhas térmicas

Uma funcionalidade muito importante do projeto é a injeção de falhas sintéticas, implementada por `SolarDataset.generate_synthetic_anomaly()`.

### Falhas suportadas

- `Hot-Spot`: ponto quente em uma região local
- `Diode`: aquecimento de um terço do módulo
- `Offline-Module`: módulo aquecido em toda a extensão
- `Shadowing`: sombra diagonal em uma região da superfície
- `Vegetation`: bloqueio frio em área inferior
- `Soiling`: ruído semelhante a poeira e contaminação

Isso permite testar o modelo de forma interativa mesmo sem acessar uma instância real de falha de campo.

---

## Interface gráfica

A interface foi desenvolvida em Tkinter e apresenta um dashboard com três colunas principais:

### Coluna esquerda

- painel térmico IR
- ajustes de intensidade e borracha
- botão de limpeza
- simulação de falhas térmicas
- navegação do dataset real
- seleção de classe e partição

### Coluna central

- seleção de arquitetura do modelo
- painel de visualização da rede
- inspeção de neurônios ocultos
- inspeção de classes de saída
- visualização de conexões entre camadas

### Coluna direita

- gráficos de perda
- gráficos de acurácia
- matriz de confusão
- auditoria de campo receptivo
- exportação dos pesos e do histórico

---

## Treinamento interativo

A aplicação possui uma rotina de treinamento controlada pela interface.

### Controles disponíveis

- `Passo (Step)`: realiza uma etapa do processo de treino
- `Treinar (Play)`: executa o treinamento em sequência
- `Reiniciar Pesos`: reseta os parâmetros da rede
- taxa de aprendizado (`η`)
- número de épocas
- delay entre ciclos
- velocidade de execução

Durante o treinamento:

- a imagem atual é carregada
- a rede executa forward
- calcula erro
- atualiza pesos
- atualiza métricas e visualizações

---

## Visualização da rede

A classe `SolarNetworkView` desenha três elementos principais:

1. matriz espacial de entrada (24x40 pixels)
2. camada oculta (grade configurada no JSON, inicialmente 64x1)
3. camada de saída (12 classes)

Ela também destaca:

- pixels selecionados
- neurônios ocultos ativos
- classe vencedora
- linha de conexão de pesos
- campo receptivo de cada classe

Essa abordagem é útil para explicar o comportamento da rede, porque a imagem é interpretada como um padrão espacial e não apenas como uma lista de números sem contexto.

---

## Métricas e auditoria

A interface também exibe métricas importantes:

- perda média na fase de treino
- acurácia em treino e validação
- matriz de confusão
- campo receptivo dos neurônios de saída

### Campo receptivo

A função `get_receptive_field()` ou `get_hidden_neuron_receptive_field()` foi projetada para mostrar quais regiões da imagem influenciam mais uma classe ou neurônio específico.

Isso pode ser usado para responder perguntas como:

- qual área da imagem a rede está observando para decidir que é um hotspot?
- a classe de diodo está sendo ativada por um padrão localizado ou por uma região ampla?
- o neurônio oculto está focado em um detalhe espacial específico?

---

## Exportação de dados

A aplicação permite exportar:

- pesos em formato CSV
- histórico de treinamento em JSON

### Exportação CSV

Os dados exportados incluem:

- índice da classe
- nome da classe
- linha e coluna do pixel
- valor do peso associado

### Exportação JSON

O JSON registra:

- arquitetura escolhida
- época atual
- número de atualizações
- histórico de métricas

---

## Execução do projeto

### Requisitos

O projeto depende de:

- Python 3.x
- NumPy
- Pillow
- Matplotlib
- scikit-learn
- Tkinter

### Ambiente virtual recomendado

No Windows, normalmente usa-se:

```powershell
cd C:\Github\Mente-Meio-Maquina\Obisidian-MMM\RNA_UFU
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m Projeto_Solar_RNA.interface_solar
```

No Linux/macOS:

```bash
cd RNA_UFU
source .venv/bin/activate
python -m Projeto_Solar_RNA.interface_solar
```

---

## Como o projeto é usado na prática

### Fluxo típico

1. o usuário abre a aplicação
2. o sistema carrega o dataset de imagens térmicas
3. o usuário navega entre amostras
4. pode selecionar uma imagem real ou injetar uma falha sintética
5. a rede realiza inferência
6. a aplicação mostra a classe prevista e a probabilidade
7. o usuário pode treinar a rede e acompanhar métricas
8. pode explorar o campo receptivo e a arquitetura visualmente

---

## Testes

Os testes unitários cobrem os componentes principais:

- carregamento do dataset
- identificação correta de classes
- geração de anomalias sintéticas
- funcionamento do perceptron
- funcionamento do MLP
- avaliação geral da rede

Os testes ficam em:

- `Projeto_Solar_RNA/test_solar_rna.py`

Para executar os testes:

```powershell
cd RNA_UFU
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m unittest Projeto_Solar_RNA.test_solar_rna -v
```

---

## Observações importantes

### 1. Foco didático
O projeto foi desenvolvido com ênfase em visualização e compreensão do comportamento da rede, e não apenas em performance máxima.

### 2. Dados térmicos como entrada espacial
A imagem térmica não é tratada como um dado genérico; ela é vista como uma matriz espacial de temperatura, o que é essencial para detectar padrões geo-localizados.

### 3. Arquitetura explicável
A camada oculta e os campos receptivos permitem observar como a rede se concentra em regiões específicas da imagem, reforçando a ideia de “explicabilidade visual”.

### 4. Aplicação experimental
O projeto funciona como laboratório para estudo de redes neurais e inspeção termográfica em energias renováveis.

---

## Possíveis extensões

Com base no código atual, várias melhorias futuras são possíveis:

- adicionar treinamento em lote (batch training)
- incluir validação por conjunto de classes com métricas mais detalhadas
- usar arquitetura CNN para melhor extração de padrões espaciais
- ampliar a base de dados com mais exemplos reais
- implementar comparação entre modelos
- adicionar salvamento e carregamento do modelo treinado
- incluir métricas por classe e recall/F1-score

---

## Conclusão

Este projeto combina redes neurais, processamento de imagens térmicas, visualização interativa e análise de falhas em painéis fotovoltaicos. Ele é uma excelente referência para quem deseja entender:

- como uma rede neurais processa imagens em formato espacial
- como classificar defeitos em energia solar
- como visualizar internamente o que a rede aprende
- como construir uma interface educacional para ciência de dados e IA aplicada

Em resumo, ele é tanto um laboratório educacional quanto uma ferramenta de diagnóstico visual para inspeção de módulos fotovoltaicos.
