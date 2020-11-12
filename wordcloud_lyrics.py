from wordcloud import WordCloud

text_file = open('wakati_list.txt', encoding='utf-8')
text = text_file.read()

#日本語のフォントパス
fpath = '/System/Library/Fonts/AquaKana.ttc'

#無意味そうな単語除去
stop_words = ['そう', 'ない', 'いる', 'する', 'まま', 'よう', 'てる', 'なる', 'こと', 'もう', 'いい', 'ある', 'ゆく', 'れる']

wordcloud = WordCloud(background_color='white',
    font_path=fpath, width=800, height=600, stopwords=set(stop_words)).generate(text)

#画像はwordcloud.pyファイルと同じディレクトリにpng保存
wordcloud.to_file('./wordcloud.png')