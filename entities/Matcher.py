import numpy as np
from typing import *
import hnswlib
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
import pandas as pd

MatcherTypes = Literal['cosine', 'euclidean', 'manhattan', 'mean', 'knn']

class Matcher:
  '''Classe responsável por comparar embeddings e retornar os pares dado um threshold. Pode ser comparado como maior ou menor que, 
  dependendo da métrica utilizada.'''
  def __init__(self, embeddings: np.ndarray, persistence:bool=False):
    self.embeddings = embeddings
    self.similarityCheckers = [cosine_similarity]
    self.distanceCheckers = [euclidean_distances, manhattan_distances]
    self.matrixes = []
    self.persistence = persistence
  
  def setEmbeddings(self, embeddings: np.ndarray) -> None:
    '''Define os embeddings a serem utilizados.'''
    self.embeddings = embeddings
  
  def saveSimilarityTable(self, path: str, similarityMatrix:np.ndarray) -> None:
    '''Salva a tabela de similaridade em um arquivo sqlite. Cria o banco, a tabela e insere os dados.'''
    # transforma em index0, index1 e similarity
    if not self.persistence:
      return
    
    quantidade = similarityMatrix.shape[0] * similarityMatrix.shape[1]
    index = 0
    import sqlite3
    from datetime import datetime
    
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    # cria o banco
    cursor.execute('''CREATE TABLE IF NOT EXISTS similarity (index0 INTEGER, index1 INTEGER, similarity REAL)''')

    print(f"Proccess starting in {datetime.now().isoformat().split('T')[1].split('.')[0]}")
    
    print(f"Salvando {index}/{quantidade} - {((index/quantidade) * 100):.2f}", end='\r')
    for i in range(similarityMatrix.shape[0]):
      for j in range(similarityMatrix.shape[1]):
        cursor.execute('''INSERT INTO similarity (index0, index1, similarity) VALUES (?, ?, ?)''', (i, j, similarityMatrix[i][j]))
        index += 1
        if index % 100000 == 0:
          print(f"Salvando {index}/{quantidade} - {((index/quantidade) * 100):.2f}", end='\r')
    conn.commit()
    conn.close()
    
  def __getPairsCosine(self, threshold: float) -> List[Tuple[int, int, float]]:
    '''Calcula a similaridade de cosseno e retorna os pares que possuem uma similaridade maior ou igual ao threshold'''
    similarityMatrix = cosine_similarity(self.embeddings)
    pairs = np.argwhere(similarityMatrix >= threshold)
    # self.saveSimilarityTable('similarityCosine.sqlite', similarityMatrix)
    # salva com a similaridade
    pairs = [(p[0], p[1], similarityMatrix[p[0], p[1]]) for p in pairs if p[0] != p[1]]
    return pairs
  
  def __getPairsEuclidean(self, threshold: float) -> List[Tuple[int, int, float]]:
    '''Calcula a distância euclidiana e retorna os pares que possuem uma distância menor ou igual ao threshold'''
    distanceMatrix = euclidean_distances(self.embeddings)
    pairs = np.argwhere(distanceMatrix <= threshold)
    pairs = [(p[0], p[1], distanceMatrix[p[0], p[1]]) for p in pairs if p[0] != p[1]]
    self.saveSimilarityTable('similarityEuclidean.sqlite', distanceMatrix)
    return pairs
  
  def __getPairsManhattan(self, threshold: float) -> List[Tuple[int, int, float]]:
    '''Calcula a distância de manhattan e retorna os pares que possuem uma distância menor ou igual ao threshold'''
    distanceMatrix = manhattan_distances(self.embeddings)
    pairs = np.argwhere(distanceMatrix <= threshold)
    pairs = [(p[0], p[1], distanceMatrix[p[0], p[1]]) for p in pairs if p[0] != p[1]]
    self.saveSimilarityTable('similarityManhattan.sqlite', distanceMatrix)
    return pairs
  
  def __getPairsMean(self, threshold: float) -> List[Tuple[int, int, float]]:
    '''Calcula a média das distâncias e similaridades e retorna os pares que possuem uma similaridade maior ou igual ao threshold.
    Método mais custoso computacionalmente, por precisar de 3 matrizes e realizar a normalização de cada uma.'''
    cosineMatrix = cosine_similarity(self.embeddings)
    euclideanMatrix = euclidean_distances(self.embeddings)
    manhattanMatrix = manhattan_distances(self.embeddings)
    
    euclideanMatrix = 1 / (1 + euclideanMatrix)
    manhattanMatrix = 1 / (1 + manhattanMatrix)
    
    cosineMatrix = MinMaxScaler().fit_transform(cosineMatrix)
    euclideanMatrix = RobustScaler().fit_transform(euclideanMatrix)
    manhattanMatrix = RobustScaler().fit_transform(manhattanMatrix)
    
    meanMatrix = np.mean([cosineMatrix, euclideanMatrix, manhattanMatrix], axis=0)
    del cosineMatrix
    del euclideanMatrix
    del manhattanMatrix
    pairs = np.argwhere(meanMatrix >= threshold)
    pairs = [(p[0], p[1], meanMatrix[p[0], p[1]]) for p in pairs if p[0] != p[1]]
    self.saveSimilarityTable('similarityMean.sqlite', meanMatrix)
    return pairs
  
  # def __knn_search(self, embeddings1: np.array ,embeddings2: np.array, ids: np.array, k: int, seed: int, metric="cosine", dim=300) -> Tuple[np.array, np.array]:
  #   '''Calcula os K vizinhos mais próximos de embeddings2 em relação a embeddings1. TESTANDO AINDA.'''
  #   # printa o shape
  #   print(len(embeddings1), len(embeddings2))
  #   print(len(embeddings1[0]), len(embeddings2[0]))
  #   index = hnswlib.Index(space=metric, dim=dim)
  #   index.init_index(max_elements=len(embeddings1), ef_construction=200, M=32, random_seed=seed)
  #   index.add_items(embeddings1, ids)
  #   index.set_ef(400)
  #   I, D = index.knn_query(embeddings2, k=k)
  #   return I, D
  
  # def __getPairsKnn(self, threshold: float) -> List[Tuple[int, int]]:
  #   '''Calcula os pares de instâncias que são vizinhos próximos, vulgo KNN. TESTANDO AINDA.'''
  #   if self.k_neighbors is None or self.seed is None or self.metric is None or self.dim is None:
  #     raise Exception("You need to configure the KNN before using it. Use the configureKNN method.")
  #   pairs = []
  #   ids = np.arange(0, len(self.embeddings))
  #   I1, D1 = self.__knn_search(self.embeddings, self.embeddings, ids, self.k_neighbors, self.seed, self.metric, self.dim)
  #   pairs = [(p, vi) for p, v, d, in zip(ids, I1, D1) for vi, di in zip(v, d) if di >= threshold]
  #   pairs = [(p[0], p[1]) for p in pairs if p[0] != p[1]]
    
  #   pairsSimilarity = [(p, vi, d, di) for p, v, d, in zip(ids, I1, D1) for vi, di in zip(v, d) if di <= threshold]
  #   # Transforma uma lista de id1, id2, sim em uma matriz de similaridade
  #   # similarityMatrix = np.zeros((len(self.embeddings), len(self.embeddings)))
  #   # for p in pairsSimilarity:
  #   #   similarityMatrix[p[0], p[1]] = p[3]
  #   # self.saveSimilarityTable('similarityKnn.sqlite', similarityMatrix)
      
  #   return pairs
  
  # def configureKNN(self, k_neighbors: int=5, seed: int=42, metric: str='cosine', dim: int=300) -> None:
  #   '''Configura os parâmetros para o KNN.'''
  #   self.k_neighbors = k_neighbors
  #   self.seed = seed
  #   self.metric = metric
  #   self.dim = dim
  
  def getPairs(self, threshold: float, by: MatcherTypes='cosine') -> List[Tuple[int, int, float]]:
    '''Retorna os pares de instâncias que possuem uma similaridade maior que o threshold. os médotos dispiníveis são:
    - cosine: Similaridade de cosseno: Quanto mais próximo de 1, mais similar. Utilizado de padrão.
    - euclidean: Distância euclidiana: Quanto mais próximo de 0, mais similar.
    - manhattan: Distância de manhattan: Quanto mais próximo de 0, mais similar.
    - mean: Média das 3 métricas anteriores: Quanto mais próximo de 1, mais similar. Mais custoso computacionalmente.
    - knn: K-Nearest Neighbors: Retorna os pares que são vizinhos próximos. TESTANDO AINDA.
    '''

    if by == 'cosine':
      return self.__getPairsCosine(threshold)
    elif by == 'euclidean':
      return self.__getPairsEuclidean(threshold)
    elif by == 'manhattan':
      return self.__getPairsManhattan(threshold)
    elif by == 'mean':
      return self.__getPairsMean(threshold)
    # elif by == 'knn':
    #   return self.__getPairsKnn(threshold)
    else:
      raise Exception("Invalid method. Use 'cosine', 'euclidean', 'manhattan', 'mean' or 'knn'.")
    