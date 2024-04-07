import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

dataset = load_dataset('parquet', data_files={'train': 'data/output.parquet'}, split='train')
dataset = dataset.with_format('jax')

params_vectors = []

for i in range(len(dataset)):
    params = dataset[i]['params']
    params_vectors.append(params)

# Convert the list of params vectors to a numpy array
params_array = np.array(params_vectors)

# Apply t-SNE to reduce the dimensionality of the params vectors
tsne = TSNE(n_components=2, random_state=42, learning_rate=100.0, perplexity=40.0)
tsne_embeddings = tsne.fit_transform(params_array)

plt.figure(figsize=(8, 6))
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1])
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization of Params Vectors')
plt.show()
plt.close()
