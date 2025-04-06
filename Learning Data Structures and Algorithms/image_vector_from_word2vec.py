import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load model GloVe
model = api.load("glove-twitter-200")

# Chọn vài từ để vẽ
words = ["king", "queen", "man", "woman", "apple", "orange", "cat", "dog", "paris", "france"]

# Lấy vector 200 chiều cho các từ trên
vectors = np.array([model[word] for word in words])

# Giảm chiều về 2D bằng PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

# Vẽ
plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    x, y = reduced[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, word)

plt.title("Word vectors (reduced from 200D to 2D using PCA)")
plt.grid(True)
plt.show()
