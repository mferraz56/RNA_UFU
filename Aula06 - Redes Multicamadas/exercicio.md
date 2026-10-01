1) treinar uma rede neural multicamada(MLP) para aproximar uma função a partir dos pontos am

Pontos amostrados:

x = [   0        .1         .2         .3       .4        .5        .6         .7       .8        .9       1.0   ];
t = [-.9602 -.5770 -.0729  .3771  .6405  .6600  .4609 .1336 -.2013 -.4344 -.5000 ];

## Resolução

A solução implementada está em [EXERCICIO/README.md](EXERCICIO/README.md),
com código, equações da retropropagação, tabela de previsões e gráfico.

Foi treinada uma MLP **1 → 10 → 1**, com ativação `tanh` na camada oculta
e saída linear, por gradiente descendente em lote. Com taxa 0,05 e semente
42, atingiu EQM menor ou igual a 0,0001 em **40.414 épocas**.
O RMSE nos 11 pontos foi aproximadamente **0,0100**.

![Curva aproximada, erro e resíduos](EXERCICIO/resultados/resultado.png)

As métricas são calculadas nos pontos usados para treinar. O enunciado não
fornece a expressão da função nem pontos de teste independentes; portanto,
o ajuste não comprova o erro entre as amostras ou fora do intervalo [0, 1].