from gensim.models import KeyedVectors
import time
import numpy as np
model_dir = './entity_vector/entity_vector.model.bin'
model = KeyedVectors.load_word2vec_format(model_dir, binary=True)
# model_dir = './jawiki_word_vectors_300d/word_vectors.txt'
# model = KeyedVectors.load_word2vec_format(model_dir, binary=False)
print(type(model))

# print(len(model.get_vector('九州')))
start = time.time()
print('九州' in model)
end = time.time()
print(end-start)
# print(model(u'九州大学'))

# results = model.most_similar(u'九州大学')
# for result in results:
#     print(result)
#
# print('\n')
#


model_dir = './entity_vector/entity_vector.model.bin'
model = KeyedVectors.load_word2vec_format(model_dir, binary=True)
similar_list = model.most_similar(positive=[u'[東京]',u'[日本]'],negative=[u'[アメリカ]'])
for similar_set in similar_list:
    print(similar_set)

vector = model.get_vector("東京")
print(vector)
print(type(vector))
print(vector.shape)
sim_vector = np.zeros(200)
sim_vector[1:200] = vector[1:200]
sim_vector[0] = vector[0] * 0.9
print(sim_vector)
print(model.similar_by_vector(sim_vector)[0][0])

def cosine_similarity(X, Y):
    """
    行列X,Yの列ベクトルと行ベクトルのコサイン類似度をまとめて計算し，コサイン類似度を並べたリストを出力
    """
    return (X @ Y.T) / np.sqrt(np.nansum(np.power(X, 2), axis=1) * np.nansum(np.power(Y, 2), axis=1))   # .Tは転置行列

def word_vectors(word, model):
    """
    引数のwordは単語，modelはword2vecのモデル(下の_main_参照)
    word2vecが引数のwordに対応していればその単語に対応する200次元wordベクトルを，なければ０が200個並ぶベクトルを返す
    """
    if word in model:
        return model.get_vector(word)
    else:
        return np.zeros(200)

# v = np.asarray([word_vectors(word, model) for word in model])
# print(len(v))


