O problema de classificação é treinavel mas acredito estar com resultados ruins. 

quero que implemente da seguinte maneira para testarmos. 

1 - buscar a tonalidade mais clara e mais escura do banco de imagem e salvar em um json qual o id da imagem e classificação e os valores. 

2 - adicionar ao mesmo json os valores maximos e minimos de cada imagem seu id e sua classe. 

3 - Para a rede neural usaremos como parametro de input:
    -- todos960 pixeis de cada imagem
    -- o valor médio dos pixeis de cada linha
    -- o valor médio dos pixeis de cada coluna

portanto na primeira camada serão 1024 neuronios, na segunda camada 64 neuronios por fim o resultado da classificação

---------------------------------

4 - mascara de 8x8 


clusterização de calor por proximidade de pixeis por painel
---- buscar semelhança nos clusters e relacionar a problema

----------------------------------

