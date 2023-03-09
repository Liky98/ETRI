import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
pd.set_option('display.max_columns', None)

neutral_EDA = []
happy_EDA = []
surprise_EDA = []
angry_EDA = []
disqust_EDA = []
sad_EDA = []
fear_EDA = []

for file in os.listdir("../KEMDy20_v1_1/annotation"):
    file_annotation = pd.read_csv(f"../KEMDy20_v1_1/annotation/{file}")
    for segmentID in file_annotation["Segment ID"][1:]:
        Emotion = file_annotation["Total Evaluation"][file_annotation.index[(file_annotation['Segment ID']==segmentID)]].item()

        file_EDA = csv.reader(open(f"../KEMDy20_v1_1/EDA/Session{segmentID.split('_')[0][-2:]}/{segmentID[:-4]}.csv"))
        file_EDA_list = [row for row in file_EDA if len(row) == 3]
        EDA_score = [row[0] for row in file_EDA_list]

        EDA_score = np.array(EDA_score, dtype=float)

        emotions = str(Emotion).split(';')
        for i in range(len(emotions)):
            globals()[f"{emotions[i]}_EDA"].append(np.mean(EDA_score))

emotions = {'Neutral': neutral_EDA, 'Happy': happy_EDA, 'Surprise': surprise_EDA,
            'Angry': angry_EDA, 'Disgust': disqust_EDA, 'Sad': sad_EDA, 'Fear': fear_EDA}

for emotion, eda in emotions.items():
    print(f'{emotion} 일때 EDA의 평균 값 : {np.nanmean(eda)}')
    print(f'{emotion} 일때 EDA의 분산 값 : {np.nanvar(eda)}')
    print(f'{emotion} 일때 EDA의 표준편차 값 : {np.nanstd(eda)}')
    print()

# 데이터
emotions = ['Neutral', 'Happy', 'Surprise', 'Angry', 'Disqust', 'Sad', 'Fear']
means = [np.nanmean(neutral_EDA), np.nanmean(happy_EDA), np.nanmean(surprise_EDA), np.nanmean(angry_EDA), np.nanmean(disqust_EDA), np.nanmean(sad_EDA), np.nanmean(fear_EDA)]
stds = [np.nanstd(neutral_EDA), np.nanstd(happy_EDA), np.nanstd(surprise_EDA), np.nanstd(angry_EDA), np.nanstd(disqust_EDA), np.nanstd(sad_EDA), np.nanstd(fear_EDA)]

# 그래프 그리기
fig, ax = plt.subplots()
ax.bar(emotions, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_xlabel('Emotions')
ax.set_ylabel('EDA Mean')
ax.set_title('Emotion EDA Means with Error Bars')
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()
