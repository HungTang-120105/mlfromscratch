import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
from Knn import KNN
import os
from sklearn.preprocessing import LabelEncoder

def load_csv(filename):
    """Hàm giúp đọc file CSV từ thư mục chứa file script đang chạy"""
    file_path = os.path.join(os.path.dirname(__file__), filename)
    return pd.read_csv(file_path)

def experiment(X, y, X_test, y_test):
    print("--- Experiment ---")
    ks = [1, 3, 5, 7, 9, 11]
    metrics = ['manhattan', 'euclidean', 'cosine']
    for metric in metrics:
        for k in ks:
            knn = KNN( X, y,k, metric=metric)

            y_pred = knn.predict(X_test)

            print("KNN with K = %d and metric = %s | Accuracy: %f" % (k, metric, len(y_test[y_pred == y_test]) / len(y_test)))

        print("-"*50)

def main():
    df = load_csv("data/train.csv")
    X_train = df.iloc[:, :].values
    y_train = load_csv("data/trainDirection.csv").iloc[:, 0].values
    print("X shape:", X_train.shape)
    print("y shape:", y_train.shape)

    df_test = load_csv("data/testing.csv")
    X_test = df_test.drop('Direction', axis=1).iloc[:, 1:].values
    y_test = df_test.loc[:, 'Direction'].values

    print("X test shape:", X_test.shape)
    print("y test shape:", y_test.shape)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)  # Chuyển 'Up' thành 1, 'Down' thành 0
    y_test = label_encoder.transform(y_test)  # Chuyển đổi cho tập test


    debug = True

    if debug:
        experiment(X_train, y_train, X_test, y_test)
        return

    k = 3
    y_train = np.array(y_train, dtype=np.float64)
    knn = KNN(X_train, y_train,k, metric='manhattan')

    X_test = np.array(X_test, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)
    pred = knn.predict(X_test)

    print("My KNN accuracy:", len(y_test[pred == y_test]) / len(y_test))

    # Check with Sk learn KNN.
    sk_knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    sk_knn.fit(X_train, y_train)

    y_sk = sk_knn.predict(X_test)

    print("Sk-learn KNN accuracy:", len(y_test[y_sk == y_test]) / len(y_test))


if __name__ == '__main__':
    main()