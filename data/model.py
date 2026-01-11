from sklearn.neighbours import NearestNeighbours
import numpy as np

nn = NearestNeighbours(metric="cosine")
embeddings = np.load("data/embeddings.npy")
nn.fit(embeddings)