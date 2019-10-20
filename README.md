# img2vec-keras
Image to dense vector embedding. This library uses the ResNet50 model in TensorFlow Keras, pre-trained on Imagenet, to generate image embeddings via https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer. Basically a clone of https://github.com/christiansafka/img2vec for TensorFlow Keras users. 

# Install
`img2vec_keras` uses the `keras` module shipped with `tensorflow`. To install `img2vec_keras` and its dependencies

```pip install git+git://github.com/jaredwinick/img2vec-keras.git```

# Usage
```python
from img2vec_keras import Img2Vec
img2vec = Img2Vec()
x = img2vec.get_vec('/path/to/image/dog1.jpg')
```

# Examples

[Basic example with cosine similarity of vectors](https://github.com/jaredwinick/img2vec-keras/blob/master/examples/similarity.py)

[Colab notebook using t-SNE to visualize image vectors](https://colab.research.google.com/drive/14OvmH6KvoQJ41jb6QRL3FgwI61vq-UAJ)

# Contributors
* Thanks to @gmgeorg for upgrading to TensorFlow 2.0
