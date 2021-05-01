
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

with open('clean_final.csv')as f:
    X_clean=pd.read_csv(f)
print(X_clean[:-2])    
  
model_type='Non-balanced'  
cat_features = ['term','amt_difference', 'grade', 'home_ownership', 'verification_status', 'purpose', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 'pub_rec_cat', 'initial_list_status']
Array = X_clean.to_numpy()
X=Array[:,:19]
print(X)
y = X_clean['target']
X_scaled = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.4, random_state=0)

clfs = {'GradientBoosting': GradientBoostingClassifier(max_depth=6, n_estimators=100, max_features=0.3),
            'LogisticRegression': LogisticRegression(),
            # 'GaussianNB': GaussianNB(),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=10)
            }
cols = ['model', 'matthews_corrcoef', 'roc_auc_score', 'precision_score', 'recall_score', 'f1_score']
models_report = pd.DataFrame(columns=cols)
conf_matrix = dict()

for clf, clf_name in zip(clfs.values(), clfs.keys()):
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]
    name=clf_name+'.joblib'
    joblib.dump(clf,'Final'+name)
    print('computing {} - {} '.format(clf_name, model_type))