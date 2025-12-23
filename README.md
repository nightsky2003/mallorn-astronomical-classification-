# Mallorn Astronomical Classification Challenge

## 1. Giới thiệu
Dự án này được thực hiện nhằm tham gia cuộc thi **Mallorn Astronomical Classification Challenge** trên Kaggle.

Mục tiêu của dự án là phân loại các đối tượng thiên văn dựa trên dữ liệu **light curve đa băng tần** (u, g, r, i, z, y). Nhóm sử dụng cách tiếp cận truyền thống gồm trích xuất đặc trưng thủ công từ chuỗi thời gian, sau đó huấn luyện các mô hình học máy để thực hiện phân loại.

---

## 2. Dữ liệu
Dữ liệu được cung cấp bao gồm:
- Dữ liệu light curve được chia thành nhiều thư mục (`split_01` đến `split_20`)
- File `train_log.csv` và `test_log.csv` chứa thông tin meta của từng object

Để thuận tiện cho việc xử lý và huấn luyện mô hình, dữ liệu từ các split được gộp lại và lưu dưới dạng **Parquet**.

---

## 3. Tiền xử lý dữ liệu (`merge.py`)
File `merge.py` thực hiện các bước:
- Gộp dữ liệu light curve từ các thư mục split
- Ghép dữ liệu light curve với thông tin meta từ file log
- Áp dụng hiệu chỉnh extinction theo công thức Fitzpatrick (1999)
- Xuất dữ liệu đã xử lý ra các file:
  - `master_train_corrected.parquet`
  - `master_test_corrected.parquet`

Bước này giúp chuẩn hóa dữ liệu trước khi trích xuất đặc trưng.

---

## 4. Trích xuất đặc trưng (`feature.py`)
Từ dữ liệu light curve đã được tiền xử lý, nhóm trích xuất các đặc trưng bao gồm:
- Đặc trưng thống kê: mean, max, min, standard deviation
- Đặc trưng hình dạng light curve: biên độ (amplitude), rise time, fall time
- Đặc trưng độ biến thiên: Neumann statistic, số lượng đỉnh (peaks)
- Đặc trưng đa băng tần: color index, tỉ lệ flux giữa các filter
- Đặc trưng hoạt động trước và sau đỉnh sáng

Các đặc trưng được lưu ra các file:
- `train_features.csv`
- `test_features.csv`

---

## 5. Huấn luyện mô hình (`train_model.py`)
Dự án sử dụng các mô hình học máy truyền thống như:
- XGBoost
- LightGBM
- CatBoost (tùy chọn)

Các mô hình được huấn luyện bằng **Stratified K-Fold Cross Validation** để đảm bảo cân bằng giữa các lớp dữ liệu.

Sau khi huấn luyện, các mô hình được kết hợp bằng phương pháp **Voting** để cải thiện độ ổn định của kết quả. Ngưỡng phân loại được lựa chọn dựa trên F1-score.

---

## 6. Kết quả
- Mô hình cho kết quả ổn định trên tập validation
- File `submission.csv` được tạo và nộp lên hệ thống Kaggle
- Kết quả leaderboard được dùng để đánh giá hiệu quả mô hình

---

## 7. Cách chạy lại dự án
```bash
python merge.py
python feature.py
python train_model.py