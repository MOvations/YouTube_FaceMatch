import json
import pandas as pd
import sys
from matplotlib.pyplot import imshow
from pytube import YouTube
from youtube_search import YoutubeSearch
import warnings
import time
import os
import pickle

warnings.filterwarnings("ignore")
from myFuncs import YTCompare, addFaces

pd.set_option("display.max_columns", None)


##### MAIN - Edit Search Term for New results #####
search_term = "roku stock"
max_results = 30

if os.path.isfile("yt_DataFrame.p"):
    df = pickle.load(open("yt_DataFrame.p", "rb"))

else:
    if os.path.isfile("links_list.p"):
        links = pickle.load(open("links_list.p", "rb"))
    else:
        results = YoutubeSearch(search_term, max_results=max_results).to_json()
        res = json.loads(results)["videos"]
        print(res)  # DEBUG
        links = [res[i]["id"] for i in range(len(res))]
        pickle.dump(links, open("links_list.p", "wb"))

    print(links)

    ytdatalist = []
    for i, _ in enumerate(links):
        time.sleep(1)
        try:
            print(f"Now looking up information for: {links[i]}")
            yt = YouTube(f"http://youtube.com//watch?v={links[i]}")
            ytdata = {
                "title": yt.title,
                "thurl": yt.thumbnail_url,
                "length": yt.length,
                "views": yt.views,
                "author": yt.author,
            }
            ytdatalist.append(ytdata)
        except:
            pass

    df = pd.DataFrame(ytdatalist)
    df["faces"] = 0
    df["face_eyes"] = 0
    df["smile"] = 0
    pickle.dump(df, open("yt_DataFrame.p", "wb"))

print(df.head())

for f, _ in enumerate(df):
    imagePath = df.iloc[f]["thurl"]
    df["faces"].iloc[f] = int(addFaces(imagePath)[0])
    df["smile"].iloc[f] = 1 if int(addFaces(imagePath)[1]) > 0 else 0
    df["face_eyes"].iloc[f] = int(addFaces(imagePath)[2] / 2)

df["SimScores"] = 0
df["SimScores"] = [YTCompare(search_term, df.iloc[i]["title"]) for i in range(len(df))]
df.sort_values(by="views", ascending=False).to_excel(f"{search_term}.xlsx")
