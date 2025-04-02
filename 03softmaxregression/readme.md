# Cách tải dữ liệu MNIST bằng PyTorch

Trong phần này, vì không thể import bộ dữ liệu MNIST từ `libs.mnist_lib.py`, chúng ta sẽ sử dụng PyTorch để lấy bộ dữ liệu MNIST.

## Cách cấu hình và import trong Python

### 1. Cấu hình `sys.path` để import file từ các package khác

Để gọi file trong các package khác nhau, cần phải có file `__init__.py` trong mỗi thư mục của package. Ngoài ra, ta cũng cần thêm đường dẫn thư mục cha (project/) vào `sys.path` để đảm bảo việc import thành công.

Ví dụ, thêm đoạn mã sau vào đầu file của bạn để cấu hình lại `sys.path`:

```python
import os
import sys 

# Thêm đường dẫn thư mục cha (project/) vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  

```

### 2. Ngoài ra cần chú ý khi sử dụng module libs, thì cần phải điền đúng thứ tự tham số khi nhập vào hoặc tạo model
