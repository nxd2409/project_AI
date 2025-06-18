# Dự án Anti-Spoofing

## Cấu trúc thư mục
- `cnn1.h5, cnn2.h5`: Mô hình đã được huấn luyện.
- `script.py`: Script thực hiện dự đoán ảnh đơn lẻ, đánh giá folder ảnh cùng với detect và nhận biết giả mạo khuôn mặt qua camera.
- `cnn_1.ipynb, cnn_2.ipynb`: Notebook tương tác với cấu hình, tiền xử lý dữ liệu, huấn luyện mô hình.
- `anti-spoofing_data/`: Chứa dữ liệu gốc dùng để huấn luyện mô hình và file CSV (anti-spoofing.csv).
- `graph/`: Lưu trữ đồ thị (confusion matrix, roc curve, training history).
- `internet_collect_test_data/`: Chứa dữ liệu ảnh dùng để kiểm tra đơn lẻ.
- `test_data/`: Chứa dữ liệu kiểm thử theo folder.

## Cài đặt
pip install -r requirements.txt