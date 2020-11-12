import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

#スクレイピングしたデータを入れる表を作成
list_df = pd.DataFrame(columns=['歌詞'])

for page in range(1, 3):
    #曲ページ先頭アドレス
    base_url = 'https://www.uta-net.com'

    #歌詞一覧ページ
    url = 'https://www.uta-net.com/artist/3891/0/' + str(page) + '/' # 嵐
    url = 'https://www.uta-net.com/artist/23343/0/' + str(page) + '/' # King Gnu
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    links = soup.find_all('td', class_='side td1')
    for link in links:
        a = base_url + (link.a.get('href'))

        #歌詞詳細ページ
        response = requests.get(a)
        soup = BeautifulSoup(response.text, 'lxml')
        song_lyrics = soup.find('div', itemprop='lyrics')
        song_lyric = song_lyrics.text
        song_lyric = song_lyric.replace('\n','')
        #サーバーに負荷を与えないため1秒待機
        time.sleep(1)

        #取得した歌詞を表に追加
        tmp_se = pd.DataFrame([song_lyric], index=list_df.columns).T
        list_df = list_df.append(tmp_se)

print(list_df)

#csv保存
# list_df.to_csv('list.csv', mode = 'a', encoding='cp932')
list_df.to_csv('list_kinggnu.csv', mode = 'a', encoding='cp932')