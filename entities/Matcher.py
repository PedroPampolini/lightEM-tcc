import numpy as np
from typing import *
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
import pandas as pd
from args.args import Args

MatcherTypes = Literal['cosine', 'euclidean', 'manhattan', 'mean']

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
    
  def __getPairsCosine(self, threshold: float, embedds1: np.ndarray, embedds2: np.ndarray) -> List[Tuple[int, int, float]]:
    '''Calcula a similaridade de cosseno e retorna os pares que possuem uma similaridade maior ou igual ao threshold'''
    similarityMatrix = cosine_similarity(embedds1, embedds2)
    pairs = np.argwhere(similarityMatrix >= threshold)
    # self.saveSimilarityTable('similarityCosine.sqlite', similarityMatrix)
    # salva com a similaridade
    pairs = [(p[0], p[1], similarityMatrix[p[0], p[1]]) for p in pairs if p[0] != p[1]]
    return pairs
  
  def __getPairsEuclidean(self, threshold: float, embedds1: np.ndarray, embedds2: np.ndarray) -> List[Tuple[int, int, float]]:
    '''Calcula a distância euclidiana e retorna os pares que possuem uma distância menor ou igual ao threshold'''
    distanceMatrix = euclidean_distances(embedds1, embedds2)
    pairs = np.argwhere(distanceMatrix <= threshold)
    pairs = [(p[0], p[1], distanceMatrix[p[0], p[1]]) for p in pairs if p[0] != p[1]]
    self.saveSimilarityTable('similarityEuclidean.sqlite', distanceMatrix)
    return pairs
  
  def __getPairsManhattan(self, threshold: float, embedds1: np.ndarray, embedds2: np.ndarray) -> List[Tuple[int, int, float]]:
    '''Calcula a distância de manhattan e retorna os pares que possuem uma distância menor ou igual ao threshold'''
    distanceMatrix = manhattan_distances(embedds1, embedds2)
    pairs = np.argwhere(distanceMatrix <= threshold)
    pairs = [(p[0], p[1], distanceMatrix[p[0], p[1]]) for p in pairs if p[0] != p[1]]
    self.saveSimilarityTable('similarityManhattan.sqlite', distanceMatrix)
    return pairs
  
  def __getPairsMean(self, threshold: float, embedds1: np.ndarray, embedds2: np.ndarray) -> List[Tuple[int, int, float]]:
    '''Calcula a média das distâncias e similaridades e retorna os pares que possuem uma similaridade maior ou igual ao threshold.
    Método mais custoso computacionalmente, por precisar de 3 matrizes e realizar a normalização de cada uma.'''
    cosineMatrix = cosine_similarity(embedds1, embedds2)
    euclideanMatrix = euclidean_distances(embedds1, embedds2)
    manhattanMatrix = manhattan_distances(embedds1, embedds2)
    
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
  
  def __runInBatch(self, threshold: float, getPairsFunc: Callable[[float, np.ndarray, np.ndarray], List[Tuple[int, int, float]]]) -> List[Tuple[int, int, float]]:
    '''Executa a função getPairsFunc em batch, dividindo a matriz de embeddings em partes menores para evitar estouro de memória.'''
    print("Running in batch")
    pairs = []
    num_embeddings = len(self.embeddings)
    batch_size = 1000
    for i in range(0, num_embeddings, batch_size):
      batch_embedds = self.embeddings[i:i+batch_size]
      batch_pairs = getPairsFunc(threshold, batch_embedds, self.embeddings)
      
      # Ajusta os índices dos pares do batch para o índice global
      adjusted_pairs = [(p[0] + i, p[1], p[2]) for p in batch_pairs]
      pairs.extend(adjusted_pairs)
      
    return pairs
  
  def getPairs(self, threshold: float, by: MatcherTypes='cosine') -> List[Tuple[int, int, float]]:
    '''Retorna os pares de instâncias que possuem uma similaridade maior que o threshold. os médotos dispiníveis são:
    - cosine: Similaridade de cosseno: Quanto mais próximo de 1, mais similar. Utilizado de padrão.
    - euclidean: Distância euclidiana: Quanto mais próximo de 0, mais similar.
    - manhattan: Distância de manhattan: Quanto mais próximo de 0, mais similar.
    - mean: Média das 3 métricas anteriores: Quanto mais próximo de 1, mais similar. Mais custoso computacionalmente.
    '''
    if Args.runLowMemory:
      match (by):
        case 'cosine':
          return self.__runInBatch(threshold, self.__getPairsCosine)
        case 'euclidean':
          return self.__runInBatch(threshold, self.__getPairsEuclidean)
        case 'manhattan':
          return self.__runInBatch(threshold, self.__getPairsManhattan)
        case 'mean':
          return self.__runInBatch(threshold, self.__getPairsMean) # avaliar remoção
        case _:
          raise Exception("Invalid method. Use 'cosine', 'euclidean', 'manhattan' or 'mean'.")
    else:
      match (by):
        case 'cosine':
          return self.__getPairsCosine(threshold, self.embeddings, self.embeddings)
        case 'euclidean':
          return self.__getPairsEuclidean(threshold, self.embeddings, self.embeddings)
        case 'manhattan':
          return self.__getPairsManhattan(threshold, self.embeddings, self.embeddings)
        case 'mean':
          return self.__getPairsMean(threshold, self.embeddings, self.embeddings) # avaliar remoção
        case _:
          raise Exception("Invalid method. Use 'cosine', 'euclidean', 'manhattan' or 'mean'.")
    