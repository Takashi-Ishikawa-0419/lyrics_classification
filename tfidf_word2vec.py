import mojimoji
import MeCab
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
# import get_textbooks
import scipy.stats as ss
from scipy import sparse
# from tqdm import tqdm
# import word2vec_analytics as wa
import json


def clean_text(text):
    """
    半角カタカナを全角に，全角英数字を半角に，大文字を小文字にする
    """
    text = mojimoji.han_to_zen(text, digit=False, ascii=False)
    text = mojimoji.zen_to_han(text, kana=False)
    text = text.lower()
    return text

def text2wakati(text):
    """
    1テキストを単語に分かちします。不要な品詞は除外、活用形は原形にする
    """
    tagger = MeCab.Tagger('-Ochasen')
    parsed_text = tagger.parse(text)
    # 除外する品詞
    stop_parts = (
    '名詞-接尾-形容動詞語幹', 'その他', 'フィラー', '副詞', '助動詞', '助詞', '動詞-接尾', '動詞-非自立', '名詞-動詞非自立的', '名詞-特殊-助動詞語幹', '名詞-接尾-サ変接続',
    '名詞-接尾-副詞可能', '名詞-接尾-人名', '名詞-接尾-助動詞語幹', '名詞-接尾-形容動詞語幹', '名詞-接尾-特殊', '名詞-非自立', '感動詞', '接続詞', '接頭詞-動詞接続',
    '接頭詞-形容詞接続', '形容詞-接尾', '形容詞-非自立', '記号-一般', '記号-句点', '記号-括弧閉', '記号-括弧開', '記号-空白', '記号-読点', '連体詞')
    return '' if not parsed_text else ' '.join([y[2] for y in [x.split('\t') for x in parsed_text.splitlines()[:-1]] if
                                                (len(y) == 6) and (not y[3].startswith(stop_parts))])


def get_tfidf_and_feature_names(corpus):
    """
    引数のcorpusは本文のテキストデータのリスト．実行すると,テキストデータに含まれる全単語数のリストと，corpusの要素(テキストデータ)数×テキストデータに含まれる全単語数の，要素に各テキストと各単語のTFIDFの値を持つ行列が出力される
    """
    vectorizer = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    return vectorizer.fit_transform([text2wakati(clean_text(text)) for text in corpus]), vectorizer.get_feature_names()

def word_vectors(word, model):
    """
    引数のwordは単語，modelはword2vecのモデル(下の_main_参照)
    word2vecが引数のwordに対応していればその単語に対応する200次元wordベクトルを，なければ０が200個並ぶベクトルを返す
    """
    if word in model:
        return model.get_vector(word)
    else:
        return np.zeros(200)

# def flask_analytics(words):
#     url = 'http://192.168.100.100:20000'
#     data = {'words':words}
#     headers = {
#         'Content-Type': 'application/json',
#     }
#
#     import urllib
#     req = urllib.request.Request(url, json.dumps(data).encode(), headers)
#     with urllib.request.urlopen(req, timeout=3600) as res:
#         matrix_list = json.loads(res.read())
#     return matrix_list

def query_text_to_vector(query, model):
    """
    関数word_vectorを用いて，設問文をベクトル化する．設問文中の単語の単語ベクトルの平均を取る
    """
    words = text2wakati(clean_text(query)).split()
    vector = np.asarray([word_vectors(word, model) for word in words])
    return np.nanmean(vector, axis=0).reshape(1, -1)

def cosine_similarity(X, Y):
    """
    行列X,Yの列ベクトルと行ベクトルのコサイン類似度をまとめて計算し，コサイン類似度を並べたリストを出力
    """
    return (X @ Y.T) / np.sqrt(np.nansum(np.power(X, 2), axis=1) * np.nansum(np.power(Y, 2), axis=1))   # .Tは転置行列


def cosine_similarity_vector(v1, v2):
    """
    ベクトル同士のコサイン類似度を計算
    """
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def soft_max(a):
    """
    ソフトマックス関数
    """
    # 一番大きい値を取得
    c = np.max(a)
    # 各要素から一番大きな値を引く（オーバーフロー対策）
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    # 要素の値/全体の要素の合計
    y = exp_a / sum_exp_a

    return y
# def similarity_matrix(matrix):
#     """
#     item-feature 行列が与えられた際に
#     item 間コサイン類似度行列を求める関数
#     """
#     d = matrix @ matrix.T  # item-vector 同士の内積を要素とする行列
#
#     # コサイン類似度の分母に入れるための、各 item-vector の大きさの平方根
#     norm = (matrix * matrix.T).sum(axis=1, keepdims=True) ** .5
#
#     # それぞれの item の大きさの平方根で割っている（なんだかスマート！）
#     return d / norm / norm.T


def bias_vector(corpus, query):
    """
    queryから単語抽出して、corpus(1つのスライド)の各ページに(完全一致で)含まれているqueryの単語数を要素としたベクトルBをつくる
    """
    words_in_query = text2wakati(clean_text(query)).split()
    vector_b = []
    for text in corpus:
        words_count = 0
        for word in words_in_query:
            words_count += text.count(word)
        vector_b.append(words_count)
    return np.asarray(vector_b)


def calculate_relation_vector(matrix_s, vector_r, vector_b, alpha):
    """
    ランキングベクトル(島田先生の論文ITSE-05-2018-0026（掲載論文）参照，詳しくは島田先生に)を計算
    """
    before_r = np.zeros(len(vector_r))
    while np.linalg.norm(before_r - vector_r) > 0.0001:
        before_r = vector_r
        vector_r = alpha * (matrix_s @ vector_r) + (1 - alpha) * vector_b
        # print("チェック")
        # print(np.linalg.norm(before_r - vector_r))
    return vector_r


def calculate_relation_vector_without_S(vector_r, vector_b, alpha):
    """
    ランキングベクトル計算式のSを除いてRを計算
    """
    before_r = np.zeros(len(vector_r))
    while np.linalg.norm(before_r - vector_r) > 0.0001:
        before_r = vector_r
        vector_r = alpha * vector_r + (1 - alpha) * vector_b
    return vector_r




if __name__ == "__main__":
    # lecture_to_pages = get_textbooks.LectureToPages("2019年度春学期・火3・サイバーセキュリティ基礎論（島田　敬士）")
    # lecture_to_pages = get_textbooks.LectureToPages("2019年度前期・木1・情報科学（林　政喜）")
    # pageslist = get_textbooks.LectureToPagesList(lecture_to_pages)
    # pageslist = get_textbooks.BookToPagesList(lecture_to_pages[1])
    kinggnu_df = pd.read_csv('list_kinggnu.csv', encoding='cp932')
    kinggnu_df_list = kinggnu_df.values.tolist()
    print(kinggnu_df)
    print(kinggnu_df_list)
    kinggnu_lyrics_list = []
    for i in range(len(kinggnu_df_list)):
        kinggnu_lyrics_list.append(kinggnu_df_list[i][1])
    print(kinggnu_lyrics_list)
    for i in range(len(kinggnu_lyrics_list)):
        text = kinggnu_lyrics_list[i].replace('\u3000', '')
        kinggnu_lyrics_list[i] = text
    print(kinggnu_lyrics_list)
    w, feature_names = get_tfidf_and_feature_names(kinggnu_lyrics_list)

    model_dir = './entity_vector/entity_vector.model.bin'
    model = KeyedVectors.load_word2vec_format(model_dir, binary=True)
    v = np.asarray([word_vectors(word, model) for word in feature_names])

    ws = w.sum(axis=1)
    doc_vectors = (w @ v) / ws
    print(doc_vectors)
    print(doc_vectors.shape())
#########################

    # query = '次の相関に関する説明の中から最も適切なものを選べ。データの相関を分析することで、2種類の量の間の関係性をある程度知ることができる。'
    # query_vector = query_text_to_vector(query, model)
    # S0 = np.asarray([cosine_similarity(doc_vectors[1:], vector) for vector in doc_vectors[1:]])[:, :, 0]
    # S = S0 / np.sum(S0, axis=0)
    #
    #
    # similarities = cosine_similarity(doc_vectors, query_vector)
    # print("R[0:]")
    # print(len(np.array(similarities.T)[0].tolist()))
    # print(np.array(similarities.T)[0].tolist())
    # R = np.array(similarities.T)[0][1:].tolist()
    # print("R[1:]")
    # print(len(R))
    # print(R)
    #
    # R_rank = ss.rankdata(R)
    #
    # result_indexes = np.argsort(R_rank).tolist()  # 類似度の順(低い順)にソート
    # result_inverse = result_indexes[::-1]
    # print(result_inverse)
    #
    # B0 = bias_vector(pageslist, query)[1:]
    # B = B0 / np.sum(B0)
    #
    #
    # print("R_calculated")
    # R2 = calculate_relation_vector(S, R, B, 0.5)
    #
    # result_indexes2 = np.argsort(R2).tolist()  # 類似度の順(低い順)にソート
    # result_inverse2 = result_indexes2[::-1]
    # print(result_inverse2)










