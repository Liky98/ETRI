import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
pd.set_option('display.max_columns', None)

neutral_IBI = []
happy_IBI = []
surprise_IBI = []
angry_IBI = []
disqust_IBI = []
sad_IBI = []
fear_IBI = []

for file in os.listdir("../KEMDy20_v1_1/annotation"):
    file_annotation = pd.read_csv(f"../KEMDy20_v1_1/annotation/{file}")
    for segmentID in file_annotation["Segment ID"][1:]:
        Emotion = file_annotation["Total Evaluation"][file_annotation.index[(file_annotation['Segment ID']==segmentID)]].item()

        file_IBI = csv.reader(open(f"../KEMDy20_v1_1/IBI/Session{segmentID.split('_')[0][-2:]}/{segmentID[:-4]}.csv"))
        file_IBI_list = [row for row in file_IBI if len(row) == 4]
        IBI_score = [row[1] for row in file_IBI_list]

        IBI_score = np.array(IBI_score, dtype=float)

        emotions = str(Emotion).split(';')
        for i in range(len(emotions)):
            globals()[f"{emotions[i]}_IBI"].append(np.mean(IBI_score))

emotions = {'Neutral': neutral_IBI, 'Happy': happy_IBI, 'Surprise': surprise_IBI,
            'Angry': angry_IBI, 'Disgust': disqust_IBI, 'Sad': sad_IBI, 'Fear': fear_IBI}

for emotion, eda in emotions.items():
    print(f'{emotion} 일때 IBI의 평균 값 : {np.nanmean(eda)}')
    print(f'{emotion} 일때 IBI의 분산 값 : {np.nanvar(eda)}')
    print(f'{emotion} 일때 IBI의 표준편차 값 : {np.nanstd(eda)}')
    print()

# 데이터
emotions = ['Neutral', 'Happy', 'Surprise', 'Angry', 'Disqust', 'Sad', 'Fear']
means = [np.nanmean(neutral_IBI), np.nanmean(happy_IBI), np.nanmean(surprise_IBI), np.nanmean(angry_IBI), np.nanmean(disqust_IBI), np.nanmean(sad_IBI), np.nanmean(fear_IBI)]
stds = [np.nanstd(neutral_IBI), np.nanstd(happy_IBI), np.nanstd(surprise_IBI), np.nanstd(angry_IBI), np.nanstd(disqust_IBI), np.nanstd(sad_IBI), np.nanstd(fear_IBI)]

# 그래프 그리기
fig, ax = plt.subplots()
ax.bar(emotions, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_xlabel('Emotions')
ax.set_ylabel('IBI Mean')
ax.set_title('Emotion IBI Means with Error Bars')
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()
