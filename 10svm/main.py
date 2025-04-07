import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from svm import SVM  
from sklearn import svm  

if __name__ == "__main__":
    # Tạo dữ liệu giả lập với 2 lớp không tuyến tính
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

    # Khởi tạo mô hình SVM tùy chỉnh
    custom_svm = SVM(kernel='rbf', C=1.0, is_saved=False)
    custom_svm.fit(X_train, y_train)
    y_pred_custom = custom_svm.predict(X_test)

    # Khởi tạo mô hình SVM từ thư viện sklearn
    sklearn_svm = svm.SVC(kernel='rbf', C=1.0)
    sklearn_svm.fit(X_train, y_train)
    y_pred_sklearn = sklearn_svm.predict(X_test)

    # Đánh giá độ chính xác của cả hai mô hình
    accuracy_custom = accuracy_score(y_test, y_pred_custom)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

    print(f"Độ chính xác của mô hình SVM tùy chỉnh: {accuracy_custom:.2f}")
    print(f"Độ chính xác của mô hình SVM từ thư viện sklearn: {accuracy_sklearn:.2f}")
