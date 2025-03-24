import numpy as np
import pandas as pd
import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# csv파일 가져오기
df = pd.read_csv("emotions.csv")

# 특성과 라벨 분리
X = df.drop(columns=["label"]) # EEG 수치 데이터
y = df["label"] # 감정 라벨

# 라벨 인코딩 (문자 -> 숫자)
le = LabelEncoder()
y_encoded = le.fit_transform(y)


# 훈련 / 테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.4, random_state=random.randint(1, 100))

# 결정 트리 분류기 선언 및 학습
clf = DecisionTreeClassifier(random_state=random.randint(1, 100))
clf.fit(X_train, y_train)

# 예측 및 평가가
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df.round(3))