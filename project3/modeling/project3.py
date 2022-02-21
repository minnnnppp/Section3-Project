import os
import pandas as pd
import sqlite3

from sklearn.metrics import f1_score, accuracy_score, plot_confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from category_encoders import OrdinalEncoder
from catboost import CatBoostClassifier

conn = sqlite3.connect("/Users/minky/Desktop/sec3/DB/project.db")
cur = conn.cursor()

cur.execute("SELECT * FROM lungcancer")
rows = cur.fetchall()
cols = [column[0] for column in cur.description]
df = pd.DataFrame.from_records(data=rows, columns=cols)

conn.close

# 결측치 확인
print(df.isna().sum())

# 컬럼 형태 확인하기
print('df: \n', df.dtypes)

# EDA
df['GENDER'] = df['GENDER'].replace(['M', 'F'], [1, 0])

# baseline
print(df['LUNG_CANCER'].value_counts(normalize=True))


# target 설정
target = 'LUNG_CANCER'
features = df.drop(columns=[target]).columns

# train/val/test split
train, test = train_test_split(df, random_state=2)
train, val = train_test_split(train, random_state=2)

# 독립/종속변수 지정
X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]
X_val = val[features]
y_val = val[target]

# catboost model
pipe_cat = make_pipeline(
    OrdinalEncoder(),
    CatBoostClassifier(random_state=2))

pipe_cat.fit(X_train, y_train)
y_predc = pipe_cat.predict(X_val)

# train/val 평가지표
print('catboost 훈련 정확도', pipe_cat.score(X_train, y_train))
print('catboost 검증 정확도', pipe_cat.score(X_val, y_val))
print('catboost_val Report \n',classification_report(y_val, y_predc))
print('catboost_val f1 스코어',f1_score(y_val, y_predc))
print('catboost_val auc점수 : ', roc_auc_score(y_val, y_predc))

# test 데이터 결과
y_test_pred = pipe_cat.predict(X_test)
print('test_Report \n',classification_report(y_test, y_test_pred))

# test 데이터 평가지표
auc_score = roc_auc_score(y_test, y_test_pred)
print('테스트 정확도', pipe_cat.score(X_test, y_test))
print('test_f1 스코어',f1_score(y_test, y_test_pred))
print('test_auc점수 : ', auc_score)

# 모델 피클링
import pickle

with open("model_cat.pkl", "wb") as file:
    pickle.dump(pipe_cat, file)