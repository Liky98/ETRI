import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
pd.set_option('display.max_columns', None)

neutral_TEMP = []
happy_TEMP = []
surprise_TEMP = []
angry_TEMP = []
disqust_TEMP = []
sad_TEMP = []
fear_TEMP = []

for file in os.listdir("../KEMDy20_v1_1/annotation"):
    file_annotation = pd.read_csv(f"../KEMDy20_v1_1/annotation/{file}")
    for segmentID in file_annotation["Segment ID"][1:]:
        Emotion = file_annotation["Total Evaluation"][file_annotation.index[(file_annotation['Segment ID']==segmentID)]].item()

        file_TEMP = csv.reader(open(f"../KEMDy20_v1_1/TEMP/Session{segmentID.split('_')[0][-2:]}/{segmentID[:-4]}.csv"))
        file_TEMP_list = [row for row in file_TEMP if len(row) == 3]
        TEMP_score = [row[0] for row in file_TEMP_list]

        TEMP_score = np.array(TEMP_score, dtype=float)

        emotions = str(Emotion).split(';')
        for i in range(len(emotions)):
            globals()[f"{emotions[i]}_TEMP"].append(np.mean(TEMP_score))

emotions = {'Neutral': neutral_TEMP, 'Happy': happy_TEMP, 'Surprise': surprise_TEMP,
            'Angry': angry_TEMP, 'Disgust': disqust_TEMP, 'Sad': sad_TEMP, 'Fear': fear_TEMP}

for emotion, eda in emotions.items():
    print(f'{emotion} 일때 TEMP의 평균 값 : {np.nanmean(eda)}')
    print(f'{emotion} 일때 TEMP의 분산 값 : {np.nanvar(eda)}')
    print(f'{emotion} 일때 TEMP의 표준편차 값 : {np.nanstd(eda)}')
    print()

# 데이터
emotions = ['Neutral', 'Happy', 'Surprise', 'Angry', 'Disqust', 'Sad', 'Fear']
means = [np.nanmean(neutral_TEMP), np.nanmean(happy_TEMP), np.nanmean(surprise_TEMP), np.nanmean(angry_TEMP), np.nanmean(disqust_TEMP), np.nanmean(sad_TEMP), np.nanmean(fear_TEMP)]
stds = [np.nanstd(neutral_TEMP), np.nanstd(happy_TEMP), np.nanstd(surprise_TEMP), np.nanstd(angry_TEMP), np.nanstd(disqust_TEMP), np.nanstd(sad_TEMP), np.nanstd(fear_TEMP)]

# 그래프 그리기
fig, ax = plt.subplots()
ax.bar(emotions, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_xlabel('Emotions')
ax.set_ylabel('TEMP Mean')
ax.set_title('Emotion TEMP Means with Error Bars')
ax.yaxis.grid(True)
ax.set_ylim([30,35])
plt.tight_layout()
plt.show()
