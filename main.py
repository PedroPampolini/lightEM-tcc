from EntityMatcher import EntityMatcher
import time, json, os

METRICS_PATH = 'metrics.json'

def main(matcherType, embedderType, databasePath, databaseColumns):
  initTime = time.time()
  entityMatcher = EntityMatcher(databasePath, databaseColumns, matcherType=matcherType, embedderType=embedderType)
  clusters = entityMatcher.pipeline()
  print(f"Got {len(clusters)} clusters.")
  finalTime = time.time()
  
  # salva as métricas
  with open(METRICS_PATH, 'r') as f:
    metrics = json.load(f)
  
  if databasePath not in metrics:
    metrics[databasePath] = {}
  if embedderType not in metrics[databasePath]:
    metrics[databasePath][embedderType] = {}
  if matcherType not in metrics[databasePath][embedderType]:
    metrics[databasePath][embedderType][matcherType] = {}
  if 'time' not in metrics[databasePath][embedderType][matcherType]:
    metrics[databasePath][embedderType][matcherType]['time'] = []
  if 'clusters' not in metrics[databasePath][embedderType][matcherType]:
    metrics[databasePath][embedderType][matcherType]['clusters'] = []
  metrics[databasePath][embedderType][matcherType]['time'].append(finalTime -initTime)
  metrics[databasePath][embedderType][matcherType]['clusters'].append(len(clusters))
  
  
  with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)
    
  os.makedirs('clustersTeste', exist_ok=True)
  with open(f'clustersTeste/clusters_{databasePath.replace("/", "_")}_{embedderType}_{matcherType}.txt', 'w') as f:
    # salva o tempo
    f.write(f"Total time: {(finalTime - initTime):.2f} seconds\n")
    for cluster in clusters:
      f.write(f"{cluster}\n")
      
  print(f"Total time: {(finalTime - initTime):.2f} seconds")

if __name__ == '__main__':
  # checa se o arquivo de métricas já existe
  if not os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, 'w') as f:
      json.dump({}, f)
  
  embedders = [
    'glove',
    'sentence-transformers'
  ]
  matchers = [
    'cosine',
    'euclidean',
    'manhattan',
    'mean'
  ]
  databases = {
    'bases/Geo': {
      'name': 1,
      'latitude': 1,
      'longtitude': 1,
    },
    # 'bases/Music-20': {
    #   'title': 1,
    #   'artist': 1,
    #   'album': 1,
    # },
    # 'bases/Music-200': {
    #   'title': 1,
    #   'artist': 1,
    #   'album': 1,
    # },
    # 'bases/Music-2000': {
    #   'title': 1,
    #   'artist': 1,
    #   'album': 1,
    # },
    # 'bases/Shopee': {
    #   'title': 1,
    # },
  }
  for database, databaseColumns in databases.items():
    for embedder in embedders:
      for matcher in matchers:
        print(f"Running with {embedder} embedder and {matcher} matcher...")
        #try:
        main(matcher, embedder, database, databaseColumns=databaseColumns)
        # except Exception as e:
        #   print(f"Error while running with {embedder} embedder and {matcher} matcher")
        #   continue