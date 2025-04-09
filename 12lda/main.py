import LDA as LDA
from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

model = LDA.LDA()
model.fit(X_train, y_train)


# plt.scatter(X_lda[:, 0], X_lda[:, 1] if X_lda.shape[1] > 1 else np.zeros_like(X_lda[:, 0]), c=y_test)
# plt.title("LDA Projection")
# plt.show()

model_nonlda = LogisticRegression(max_iter=1000)
model_nonlda.fit(X_train, y_train)
y_pred = model_nonlda.predict(X_test)
print("Accuracy without LDA: ", model_nonlda.score(X_test, y_test))


#test model with lda
X_train_lda = model.transform(X_train)
X_test_lda = model.transform(X_test)
model_withlda = LogisticRegression(max_iter=1000)
model_withlda.fit(X_train_lda, y_train)
y_pred = model_withlda.predict(X_test_lda)
print("Accuracy with LDA: ", model_withlda.score(X_test_lda, y_test))



