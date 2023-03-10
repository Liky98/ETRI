import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
pd.set_option('display.max_columns', None)

neutral_AROUSAL = []
happy_AROUSAL = []
surprise_AROUSAL = []
angry_AROUSAL = []
disqust_AROUSAL = []
sad_AROUSAL = []
fear_AROUSAL = []

for file in os.listdir("../KEMDy20_v1_1/annotation"):
    file_annotation = pd.read_csv(f"../KEMDy20_v1_1/annotation/{file}")
    for segmentID in file_annotation["Segment ID"][1:]:
        Emotion = file_annotation["Total Evaluation"][file_annotation.index[(file_annotation['Segment ID']==segmentID)]].item()
        Valence = file_annotation[" .2"][file_annotation.index[(file_annotation['Segment ID']==segmentID)]].item()

        emotions = str(Emotion).split(';')
        for i in range(len(emotions)):
            globals()[f"{emotions[i]}_AROUSAL"].append(float(Valence))

emotions = {'Neutral': neutral_AROUSAL, 'Happy': happy_AROUSAL, 'Surprise': surprise_AROUSAL,
            'Angry': angry_AROUSAL, 'Disgust': disqust_AROUSAL, 'Sad': sad_AROUSAL, 'Fear': fear_AROUSAL}

for emotion, eda in emotions.items():
    print(f'{emotion} 일때 AROUSAL의 평균 값 : {np.nanmean(eda)}')
    print(f'{emotion} 일때 AROUSAL의 분산 값 : {np.nanvar(eda)}')
    print(f'{emotion} 일때 AROUSAL의 표준편차 값 : {np.nanstd(eda)}')
    print()

# 데이터
emotions = ['Neutral', 'Happy', 'Surprise', 'Angry', 'Disqust', 'Sad', 'Fear']
means = [np.nanmean(neutral_AROUSAL), np.nanmean(happy_AROUSAL), np.nanmean(surprise_AROUSAL), np.nanmean(angry_AROUSAL), np.nanmean(disqust_AROUSAL), np.nanmean(sad_AROUSAL), np.nanmean(fear_AROUSAL)]
stds = [np.nanstd(neutral_AROUSAL), np.nanstd(happy_AROUSAL), np.nanstd(surprise_AROUSAL), np.nanstd(angry_AROUSAL), np.nanstd(disqust_AROUSAL), np.nanstd(sad_AROUSAL), np.nanstd(fear_AROUSAL)]

# 그래프 그리기
fig, ax = plt.subplots()
ax.bar(emotions, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_xlabel('Emotions')
ax.set_ylabel('AROUSAL Mean')
ax.set_title('Emotion AROUSAL Means with Error Bars')
ax.yaxis.grid(True)
ax.set_ylim([2.5, 4.0])
plt.tight_layout()
plt.show()
