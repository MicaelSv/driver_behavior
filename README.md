# Classificação de Comportamento de Motoristas

Este repositório contém o código utilizado para classificar o comportamento de motoristas com base em dados de sensores. A classificação é realizada utilizando dois modelos de aprendizado de máquina: Random Forest e Long Short-Term Memory (LSTM). Os dados incluem leituras de acelerômetro e giroscópio de veículos, categorizadas em três classes: lento, normal e agressivo. Este trabalho combina aprendizado de máquina com medidas de teoria da informação para melhorar a precisão dos modelos.

## Dados

Os conjuntos de dados são divididos em conjuntos de treinamento e teste, com diferentes tamanhos de janela e características. As características brutas incluem:

- Dados de acelerômetro: 'AccX', 'AccY', 'AccZ'
- Dados de giroscópio: 'GyroX', 'GyroY', 'GyroZ'

As características informacionais incluem medidas de entropia e complexidade derivadas dos dados de sensores brutos.

## Scripts

- 'mymodels.py': Contém funções para construir e avaliar os modelos Random Forest e LSTM.
- 'mygraphics.py': Carrega os resultados das métricas de avaliação dos modelos a partir de arquivos JSON e gera gráficos comparativos de desempenho.
