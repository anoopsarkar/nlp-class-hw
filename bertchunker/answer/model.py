from default import *
chunker = FinetuneTagger('data/chunker', '.pt', 'distilbert-base-uncased')
print(chunker.model_str())
