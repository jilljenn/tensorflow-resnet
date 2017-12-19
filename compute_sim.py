from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json


with open('hashes.json') as f:
    hashes = json.load(f)

filenames, X = zip(*hashes.items())
print(filenames)
print(cosine_similarity(X))
