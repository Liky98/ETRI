import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
pd.set_option('display.max_columns', None)

neutral_VALENCE = []
happy_VALENCE = []
surprise_VALENCE = []
angry_VALENCE = []
disqust_VALENCE = []
sad_VALENCE = []
fear_VALENCE = []

for file in os.listdir("../KEMDy20_v1_1/annotation"):
    file_annotation = pd.read_csv(f"../KEMDy20_v1_1/annotation/{file}")
    for segmentID in file_annotation["Segment ID"][1:]:
        Emotion = file_annotation["Total Evaluation"][file_annotation.index[(file_annotation['Segment ID']==segmentID)]].item()
        Valence = file_annotation[" .1"][file_annotation.index[(file_annotation['Segment ID']==segmentID)]].item()

        emotions = str(Emotion).split(';')
        for i in range(len(emotions)):
            globals()[f"{emotions[i]}_VALENCE"].append(float(Valence))

emotions = {'Neutral': neutral_VALENCE, 'Happy': happy_VALENCE, 'Surprise': surprise_VALENCE,
            'Angry': angry_VALENCE, 'Disgust': disqust_VALENCE, 'Sad': sad_VALENCE, 'Fear': fear_VALENCE}

for emotion, eda in emotions.items():
    print(f'{emotion} 일때 VALENCE의 평균 값 : {np.nanmean(eda)}')
    print(f'{emotion} 일때 VALENCE의 분산 값 : {np.nanvar(eda)}')
    print(f'{emotion} 일때 VALENCE의 표준편차 값 : {np.nanstd(eda)}')
    print()

# 데이터
emotions = ['Neutral', 'Happy', 'Surprise', 'Angry', 'Disqust', 'Sad', 'Fear']
means = [np.nanmean(neutral_VALENCE), np.nanmean(happy_VALENCE), np.nanmean(surprise_VALENCE), np.nanmean(angry_VALENCE), np.nanmean(disqust_VALENCE), np.nanmean(sad_VALENCE), np.nanmean(fear_VALENCE)]
stds = [np.nanstd(neutral_VALENCE), np.nanstd(happy_VALENCE), np.nanstd(surprise_VALENCE), np.nanstd(angry_VALENCE), np.nanstd(disqust_VALENCE), np.nanstd(sad_VALENCE), np.nanstd(fear_VALENCE)]

# 그래프 그리기
fig, ax = plt.subplots()
ax.bar(emotions, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_xlabel('Emotions')
ax.set_ylabel('VALENCE Mean')
ax.set_title('Emotion VALENCE Means with Error Bars')
ax.yaxis.grid(True)
ax.set_ylim([1.5, 4.5])
plt.tight_layout()
plt.show()
