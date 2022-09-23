from codigo_algoritmo_de_kohonen import Kohonen
import pandas as pd

# O algoritmo implantado entende os dados de treino e de teste como arrays,
# dessa forma se torna preciso um pré-processamento nos dados para transforma-los em array.
# Os dados arquivos CSV usados não serão disponibilizados, logo, o usuário deverá usar seus próprios dados.

# 1.1 - DADOS DE TREINO.

dados = pd.read_csv("3-navios_treino.csv")
del dados['a']

#Transformando os dados de treino em array:
dados_treino = dados.to_records(index=False)

# 1.2 - DADOS DE TESTE.

dados2 = pd.read_csv("3-navios_teste.csv")
rotulos = dados2['a']
del dados2['a']

#Transformando os dados de teste em array:
dados_teste = dados2.to_records(index=False)

# Estabelecendo os parâmetros de taxa de aprendizagem e épocas de treinamento:
tx_de_aprendizagem = 0.05
epocas = 1

# 1.3 CLASSIFICADOR.

# Criando o objeto:
kohonen = Kohonen(10,10,dados_treino,epocas,tx_de_aprendizagem,1)

# Executando o Algoritmo:
kohonen.kohonen(dados_teste,rotulos)






