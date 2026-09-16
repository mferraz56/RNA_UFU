// implementação da Regra Delta para treinamento do adaline
// Levantamento da fronteira de separação
// aplicação: funções lógicas  bipolares
// data: 14/04/2026
// autor: Keiji Yamanaka

clc; // limpa area de comando
clear;// limpa as variáveis da area de trabalho

// dados de treinamento
// tabela verdade
//  x1  x2
x=[ 1  1   
   -1  1
    1 -1
   -1 -1]; 
t=[ 1  
    1  
    1 
   -1];

// gera gráfico dos pontos
clf(); // limpa a janela de gráficos
subplot(2,2,1)
gca.grid=[1 1 1]
set(gca(),"auto_scale", "on");
set(gca(),"data_bounds", [-3,-2;3,4]);
title("Dados");
xlabel("x1");
ylabel("x2");
//da=gda();
//da.y_location="left";
//da.x_location="bottom";
for ponto=1:4
    if t(ponto)==1
        plot(x(ponto,1), x(ponto,2),'bd');
    else
        plot(x(ponto,1), x(ponto,2),'rd');
    end
end
//pause;
/// ------treinamento do adaline ------

// inicialização das variáveis e dos parâmetros
want=0.5- rand(1,2,"uniform"); // inicialização dos pesos
bant=0.5- rand();
teta = 0;// limiar  da função de ativação degrau(rede treinada)
alfa = 0.05; // taxa de aprendizagem(0<alfa<=1)
numciclos=50;  // número  total de ciclos de treinamento
ciclos=0;  // conta o número de vezes que os dados foram apresentados
subplot(2,2,2)
title('Curva do Erro Quadratico total')
xlabel('ciclos')
ylabel('Erro quadratico total')
// treinar enquanto condição de parada não for satisfeita
mprintf("Treinamento do Adaline\n");

while ciclos<=numciclos // limite de treinamento
    erroquadratico=0; // cálculo do erro quadratico total
    ciclos=ciclos+1; //  conta número de ciclos de treinamento
    mprintf("ciclos = %d\n", ciclos);
     for entrada =1:4 // apresenta todos os padrões de entrada
         yliquido = want(1)*x(entrada,1)+ want(2)*x(entrada,2)+bant;
         // função de ativação: linear
         y=yliquido; 
         // cálculo do erro quadrático
         erroquadratico= erroquadratico+(t(entrada)-y)^2;
         // atualização dos pesos
         wnovo(1)= want(1)+alfa*(t(entrada)-y)*x(entrada,1);
         wnovo(2)= want(2)+alfa*(t(entrada)-y)*x(entrada,2);
         bnovo=bant+alfa*(t(entrada)-y);
         // salva os pesos para a próxima atualização
         want=wnovo;
         bant=bnovo;
         
     end   
     mprintf('erroquadratico= %f\n',erroquadratico);
          plot(ciclos, erroquadratico, 'r*'); 
end

// ----- teste da rede treinada ------
mprintf('Teste da rede treinada\n\n');
for entrada =1:4 // apresenta todos os padrões de entrada
    yliquido = wnovo(1)*x(entrada,1)+ wnovo(2)*x(entrada,2)+bnovo;
    // função de ativação para o teste: degrau
    if yliquido >= teta
      y=1;
    else
      y=-1;
    end
    mprintf('t(%d)= %d   y(%d):  %d\n',entrada, t(entrada),entrada, y);
 //  mprintf('t(%d)= %d   y(%d):  %f\n',entrada, t(entrada),entrada, yliquido);

end
subplot(2,2,3)
gca.grid=[1 1 1]
title('Fronteira de separação')

//imprimindo a fronteira de seção
for ponto=1:4
    if t(ponto)==1
        plot(x(ponto,1), x(ponto,2),'bd');
    else
        plot(x(ponto,1), x(ponto,2),'rd');
    end
end

 for abcissa=-3:0.1:3
    ordenada = (-abcissa*wnovo(1)-bnovo)/wnovo(2);
    plot(abcissa, ordenada,'g.') ;
 end
