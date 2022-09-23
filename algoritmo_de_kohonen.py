import numpy as np
import math
from math import floor
import pandas as pd

class _Neuronio:
    def __init__(self,range_entrada,linha,coluna):
        np.random.seed(70)
        self._peso = np.random.rand(range_entrada)
        self._linha_neuronio = linha
        self._coluna_neuronio = coluna

    def _get_peso(self):
        return self._peso.tolist()

    def _set_peso(self,novo_peso):
        peso = np.array(novo_peso)
        self._peso = peso

    def _get_linha(self):
        return self._linha_neuronio

    def _get_coluna(self):
        return self._coluna_neuronio

class Kohonen():
    def __init__(self,quantidade_de_linhas_mapa,quantidade_de_colunas_mapa,dados_treino,epocas,taxa_aprendizagem,raio_de_vizinhanca):
        self._dados_treino = dados_treino
        self._range_entrada = len(dados_treino[0])
        self._epocas = epocas
        self._taxa_aprendizagem = taxa_aprendizagem
        self._raio_de_vizinhanca = raio_de_vizinhanca
        self._qtd_linhas = quantidade_de_linhas_mapa
        self._qtd_colunas = quantidade_de_colunas_mapa
        self._mapa_som = None

    def _criando_mapa_som(self):
        matriz = []

        i = 0
        while i < self._qtd_linhas:
            linha = []
            for j in range(self._qtd_colunas):
                neuronio = _Neuronio(self._range_entrada,i,j)
                linha.append(neuronio)
            matriz.append(linha)
            i += 1
        mapa = np.array(matriz)
        self._mapa_som = mapa

    @property
    def mapa(self):
        return self._mapa_som

    def _distancia_euclidiana(self,wi,xi):
        return math.sqrt((wi - xi) ** 2)

    def _amostra(self,n_interacoes):
        amostra = self._dados_treino[n_interacoes]
        return amostra

    def neuronio_vencedor(self,k,dados):

        matriz_de_distancias = np.zeros((self._qtd_linhas, self._qtd_colunas))
        amostra = dados[k]

        for coluna in range(self._qtd_colunas):
            for linha in range(self._qtd_linhas):
                neuronio = self._mapa_som[linha][coluna]
                peso_neuronio = neuronio._get_peso()
                resultado = []
                for indice in range(len(peso_neuronio)):
                    calculo_distancia = self._distancia_euclidiana(peso_neuronio[indice],(amostra[indice]))
                    resultado.append(calculo_distancia)

                distancia_final_neuronio = sum(resultado)
                matriz_de_distancias[linha][coluna] = distancia_final_neuronio

        indice_neuronio_vencedor = np.argmin(matriz_de_distancias)
        coluna = indice_neuronio_vencedor % self._qtd_linhas
        linha = floor(indice_neuronio_vencedor % self._qtd_linhas)
        neuronio_vencedor = self._mapa_som[linha][coluna]

        return neuronio_vencedor

    def _atualizacao_pesos(self,neuronio_vencedor,k):
        peso_neuronio = neuronio_vencedor._get_peso()
        pesos_atualizados = []

        for indice in range(len(peso_neuronio)):
            peso_atualizado = peso_neuronio[indice] + self._taxa_aprendizagem * ((self._amostra(k)[indice]) - peso_neuronio[indice])
            pesos_atualizados.append(peso_atualizado)

        neuronio_vencedor._set_peso(pesos_atualizados)

        self._atualizacao_vizinhanca(neuronio_vencedor,k)

    def _atualizar_vizinho(self,neuronio_vizinho,k):
        peso_neuronio_vizinho = neuronio_vizinho._get_peso()
        pesos_atualizados_neuronio_vizinho = []

        for indice in range(len(peso_neuronio_vizinho)):
            peso_atualizado = peso_neuronio_vizinho[indice] + (self._taxa_aprendizagem / 2) * ((self._amostra(k)[indice]) - peso_neuronio_vizinho[indice])
            pesos_atualizados_neuronio_vizinho.append(peso_atualizado)

        neuronio_vizinho._set_peso(pesos_atualizados_neuronio_vizinho)

    def _atualizacao_vizinhanca(self,neuronio_vencedor,k):
        for coluna in range(self._qtd_colunas):
            for linha in range(self._qtd_linhas):

                neuronio_vizinho = self._mapa_som[linha][coluna]
                distancia_linha_neuronio = self._distancia_euclidiana(neuronio_vencedor._get_linha(),neuronio_vizinho._get_linha())
                distancia_coluna_neuronio = self._distancia_euclidiana(neuronio_vencedor._get_coluna(),neuronio_vizinho._get_coluna())
                distancia_linha_e_coluna_neuronio = [distancia_linha_neuronio,distancia_coluna_neuronio]

                if distancia_linha_neuronio <= self._raio_de_vizinhanca and distancia_coluna_neuronio <= self._raio_de_vizinhanca and distancia_linha_e_coluna_neuronio != [0.0,0.0]:
                    self._atualizar_vizinho(neuronio_vizinho,k)

    def _classificador(self,dados,rotulos):
        amostras_clasificadas = []

        for indice in range(len(dados)):
            amostra = dados[indice]
            neuronio = self.neuronio_vencedor(indice,dados)

            amostra_classificada = [amostra, f'{[neuronio._get_linha()]}{[neuronio._get_coluna()]}',rotulos[indice]]
            amostras_clasificadas.append(amostra_classificada)

        dataframe = pd.DataFrame(amostras_clasificadas, columns=['Amostra','Cluster','Classes'])
        agrupamento_por_cluster = dataframe["Cluster"].value_counts()
        quantidade_classes = dataframe["Classes"].value_counts()
        agrupamento_por_classe_em_cluster = dataframe.groupby('Cluster')['Classes'].value_counts()

        print("Amostras Classificadas:")
        print(dataframe)
        print("\n")
        print("Classificação por Cluster:")
        print(agrupamento_por_cluster)
        print("\n")
        print("Classes atribuídas por Cluster:")
        print(agrupamento_por_classe_em_cluster)
        print("\n")
        print("Quantidade de Classes nos dados de teste:")
        print(quantidade_classes)

    def kohonen(self,dados_para_classificar,rotulos):
        self._criando_mapa_som()

        contador = 0

        while contador < self._epocas:
            for k in range(len(self._dados_treino)):
                neuronio_vencedor = self.neuronio_vencedor(k, self._dados_treino)
                self._atualizacao_pesos(neuronio_vencedor, k)

            contador += 1

        self._classificador(dados_para_classificar,rotulos)

