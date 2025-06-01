# Dự án Anti-Spoofing

## Cấu trúc thư mục
- `anti_spoofing_model.h5`: Mô hình đã được huấn luyện.
- `improved_cnn_model.py`: Script thực hiện dự đoán ảnh đơn lẻ và đánh giá folder ảnh.
- `improved_cnn.ipynb`: Notebook tương tác với cấu hình, tiền xử lý dữ liệu, huấn luyện mô hình.
- `anti-spoofing_data/`: Chứa dữ liệu gốc dùng để huấn luyện mô hình và file CSV (anti-spoofing.csv).
- `graph/`: Lưu trữ đồ thị (confusion matrix, roc curve, training history).
- `internet_collect_test_data/`: Chứa dữ liệu ảnh dùng để kiểm tra đơn lẻ.
- `test_data/`: Chứa dữ liệu kiểm thử theo folder.
