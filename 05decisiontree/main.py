import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from DecisionTree import DecisionTree
import numpy as np
import os

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/titanic_train.csv"))
    X = df.loc[:, :].drop(['Survived', 'PassengerId'], axis=1).values
    y = df.loc[:, 'Survived'].values

    dt = DecisionTree(criterion='ig', max_depth=5)
    dt.train(X, y, df.columns.drop(['Survived', 'PassengerId']))


    df_test = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/titanic_test.csv"))
    X_test = df_test.loc[:, :].drop(['Survived', 'PassengerId'], axis=1).values
    y_test = df_test.loc[:, 'Survived'].values
    predicts = []
    for x in X_test:
        predicts.append(dt.predict(x))
        # print("Predicted:", predicts, "Actual:", y_test[len(predicts)-1])
    predicts = np.asarray(predicts)
    print("Accuracy:", len(predicts[predicts == y_test])/len(predicts))
    # sai 1 test ở dòng 170 

    dt_sk = DecisionTreeClassifier(max_depth=5)
    X[X[:, 7] == 'male', 7] = 1
    X[X[:, 7] == 'female', 7] = 0

    X_test[X_test[:, 7] == 'male', 7] = 1
    X_test[X_test[:, 7] == 'female', 7] = 0
    dt_sk.fit(X, y)
    y_pred = dt_sk.predict(X_test)
    print("Accuracy of Sk-learn:", len(y_pred[y_pred == y_test]) / len(y_pred))
