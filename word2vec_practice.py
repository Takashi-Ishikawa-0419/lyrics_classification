from gensim.models import KeyedVectors
import time
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

