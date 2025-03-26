from EntityMatcher import EntityMatcher
import time

def runWithGlove():
  path = 'bases/Music-20'
  entityMatcher = EntityMatcher(path, ['title','artist'], embedderType='glove')
  clusters = entityMatcher.pipeline()
  with open('clustersGlove.txt', 'w') as f:
    for cluster in clusters:
      f.write(f"{cluster}\n")
      
def runWithFasttext():
  path = 'bases/Music-20'
  entityMatcher = EntityMatcher(path, ['title','artist'], embedderType='fasttext')
  clusters = entityMatcher.pipeline()
  with open('clustersFastText.txt', 'w') as f:
    for cluster in clusters:
      f.write(f"{cluster}\n")

def main(matcherType, embedderType):
  initTime = time.time()
  path = 'bases/Music-20'
  entityMatcher = EntityMatcher(path, ['title','artist'], matcherType=matcherType, embedderType=embedderType)
  clusters = entityMatcher.pipeline()
  print(f"Got {len(clusters)} clusters.")
  finalTime = time.time()
  
  with open(f'clustersTeste/clusters_{embedderType}_{matcherType}.txt', 'w') as f:
    # salva o tempo
    f.write(f"Total time: {(finalTime - initTime):.2f} seconds\n")
    for cluster in clusters:
      f.write(f"{cluster}\n")
      
  print(f"Total time: {(finalTime - initTime):.2f} seconds")

if __name__ == '__main__':
  # embedders = ['glove', 'fasttext']
  # matchers = ['cosine', 'euclidean', 'manhattan', 'mean']
  # for embedder in embedders:
  #   for matcher in matchers:
  #     print(f"Running with {embedder} embedder and {matcher} matcher...")
  #     try:
  #       main(matcher, embedder)
  #     except Exception as e:
  #       print(f"Error while running with {embedder} embedder and {matcher} matcher")
  #       continue
  embedder = 'fasttext'
  matcher = 'euclidean'
  print(f"Running with {embedder} embedder and {matcher} matcher...")
  try:
    main(matcher, embedder)
  except Exception as e:
    print(f"Error while running with {embedder} embedder and {matcher} matcher: {e}")