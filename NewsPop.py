#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from time import time
from pathlib import Path
import joblib as jl
import matplotlib.pyplot as plt
import cufflinks as cf
import plotly.offline

cf.go_offline()
cf.set_config_file(offline=True)

rngseed = 1
shares_cutoff = 1400
new_col_name = 'popular'

basedf = pd.read_csv('OnlineNewsPopularity.csv')
basedf.rename(columns=lambda x: x.strip(), inplace=True)
# newsdf.describe()


newsdf = basedf
newsdf[new_col_name] = np.where(newsdf.shares < shares_cutoff, 0, 1)
newsdf.drop(labels='shares', axis=1, inplace=True)
X = newsdf.drop(labels=[new_col_name, 'url', 'timedelta'], axis=1)
y = newsdf.loc[:, newsdf.columns == new_col_name]

X.corr().iplot(kind='heatmap', colorscale="Blues", title="Feature Correlation Matrix")

corr_cols = ['n_tokens_content', 'title_subjectivity', 'min_positive_polarity', 'kw_avg_min']
# corr_cols = attributes[11:14].url
newsdf.loc[:, corr_cols].corr().iplot(kind='heatmap', colorscale="Blues", title="Feature Correlation Matrix")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rngseed, stratify=y)
X_train_scaled = StandardScaler().fit_transform(X_train.astype(np.float64))
X_test_scaled = (X_test - np.mean(X_train)) / np.std(X_train)

rnd_clf = RandomForestClassifier(max_features='sqrt', n_jobs=-1, random_state=rngseed)
ext_clf = ExtraTreesClassifier(max_features='sqrt', n_jobs=-1, random_state=rngseed)
sgd_clf = SGDClassifier(loss='log', penalty='elasticnet', max_iter=5000, n_jobs=-1, random_state=rngseed)

rnd_p_grid = {"n_estimators": [20, 40, 60, 80],
              "max_depth": [10, 12, 14, 16, 18],
              }

ext_p_grid = {"n_estimators": [170, 200, 250],
              "max_depth": [12, 14, 16, 18],
              }

sgd_p_grid = {"alpha": [.0001, 0.0005, 0.001],
              "l1_ratio": [0, 0.1, 0.5, 0.9, 1],
              #              "n_iter_no_change" :[5,10, 15],
              "tol": [1e-3, 1e-4]}

cv_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=rngseed)

rnd_cv = GridSearchCV(estimator=rnd_clf, param_grid=rnd_p_grid, cv=cv_folds, iid=False, error_score=np.nan)

ext_cv = GridSearchCV(estimator=ext_clf, param_grid=ext_p_grid, cv=cv_folds, iid=False, error_score=np.nan)

sgd_cv = GridSearchCV(estimator=sgd_clf, param_grid=sgd_p_grid, cv=cv_folds, iid=False, error_score=np.nan)

t1 = time()
ignore_files = True
vclf = VotingClassifier(estimators=[
    ('rnd', rnd_cv),
    ('ext', ext_cv),
    ('sgd', sgd_cv),
]
    , voting='soft',
    flatten_transform=False)
file = Path("Voting.pickle")
if file.exists() and not ignore_files:
    vclf = jl.load(file)
else:
    vclf.fit(X_train_scaled, np.ravel(y_train))
    jl.dump(vclf, file, compress=True)
print("%0.6fs." % (time() - t1))

# Prediction and Soft Voting
y_pred = vclf.predict(X_test_scaled)

accuracy_score(y_test, y_pred)

confusion_matrix(y_test, y_pred)

proba = [pd.DataFrame(x) for x in vclf.transform(X_test_scaled)]

# Hard voting
prob = pd.concat([x.idxmax(axis=1) for x in proba]).groupby(level=0).agg(pd.Series.mode)
accuracy_score(y_test, prob)

## ROC Curve
class_score = [roc_curve(y_test, df.loc[:, 1]) for df in proba]

estimator_classes = [vclf.named_estimators[x].estimator.__class__.__name__
                     for x in vclf.named_estimators.keys()]


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()


plt.figure(figsize=(16, 9))
for clf, clfname in zip(class_score, estimator_classes):
    plot_roc_curve(clf[0], clf[1], label=clfname)
plt.show()
