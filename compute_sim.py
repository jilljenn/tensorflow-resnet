from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os.path


# with open('hashes.json') as f:
#     hashes = json.load(f)

# # filenames, X = zip(*hashes.items())  # All
# LIMIT = ['data/castle.jpg', 'data/resize.jpg', 'data/sample.jpg', 'data/224.jpg', 'data/result.jpg', 'data/sampled.jpg']
# filenames = LIMIT
# X = [hashes[filename] for filename in filenames]
# print(filenames)
# print(cosine_similarity(X))
#print(hashes['data/224.jpg'])

filenames = open('filenames.txt').read().splitlines()
X = np.load('hashes.npy')
sim = cosine_similarity(X)
DEMO = '/Users/jin/Desktop/New Year/resized/IMG_20180104_191930.jpg'
index = filenames.index(DEMO)
neighbor_ids = sim[index, :].argsort()[::-1][:10]
with open('results.html', 'w') as f:
    f.write('<h1>Plus proches voisins de {}</h1>'.format(DEMO))
    f.write('<img src="{}" style="border: 2px solid red" /><br />'.format(DEMO))
    for index in neighbor_ids:
        f.write('<img src="{}" />'.format(filenames[index]))
