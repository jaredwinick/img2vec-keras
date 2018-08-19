import sys
import os
sys.path.append("..")  # Adds higher directory to python modules path.

from img_to_vec import Img2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

img2vec = Img2Vec()
cat1_vec = img2vec.get_vec('images/cat1.jpg')
cat2_vec = img2vec.get_vec('images/cat2.jpg')
dog1_vec = img2vec.get_vec('images/dog1.jpg')
dog2_vec = img2vec.get_vec('images/dog2.jpg')

X = np.stack([cat1_vec, cat2_vec, dog1_vec, dog2_vec])
Y = X
similarity_matrix = cosine_similarity(X, Y)

print(similarity_matrix)
