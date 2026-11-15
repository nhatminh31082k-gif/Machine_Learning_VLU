# BÀI THỰC HÀNH 7: ĐÁNH GIÁ MÔ HÌNH (ACCURACY & METRICS)
**Môn học:** Học máy ứng dụng (Machine Learning)  
**Giảng viên:** ThS. Nguyễn Thái Anh  
**Đơn vị:** Khoa Công nghệ Thông tin - Trường Đại học Văn Lang  

---

## Cấu trúc thư mục

```text
07. Accuracy_Metrics/
├── data/
│   ├── gia_nha.csv                  # Dữ liệu giá nhà (60 căn, dùng lại cho Bài tập 6)
│   └── sinh_vien_bo_hoc.csv         # Dữ liệu sinh viên bỏ học (1200 mẫu, ~10% lớp 1)
├── code/                            # 6 tệp mã nguồn hướng dẫn trong tài liệu
│   ├── h1_do_bua.py                 # Bước 1: Accuracy đánh lừa trên dữ liệu lệch lớp
│   ├── h2_ma_tran.py                # Bước 2: Tự đếm 4 ô ma trận nhầm lẫn
│   ├── h3_precision_recall.py       # Bước 3: Precision, recall, F1 và classification_report
│   ├── h4_nguong.py                 # Bước 4: Đổi ngưỡng và chọn ngưỡng qua cross_val_predict
│   ├── h5_roc_auc.py                # Bước 5: Đường ROC, AUC và so sánh các mô hình
│   └── h6_can_bang.py               # Bước 6: Kiểm định chéo nhiều thước đo & class_weight
├── baitap07/                        # Thư mục 6 bài tập tự thực hành
│   ├── bai1.py                      # Bài 1: Tự viết hàm danh_gia chỉ dùng numpy
│   ├── bai2.py                      # Bài 2: classification_report cho SVM chữ số viết tay
│   ├── bai3.py                      # Bài 3: Chọn ngưỡng lớn nhất đạt recall >= 0.8 trên tập học
│   ├── bai4.py                      # Bài 4: Vẽ đường ROC (Logistic vs Random Forest)
│   ├── bai5.py                      # Bài 5: So sánh class_weight="balanced" cho RF và SVC
│   ├── bai6.py                      # Bài 6: Ảnh hưởng của 1 dự đoán hỏng (+5 tỷ) tới MAE, RMSE, R2
│   ├── tao_anh_chup.py              # Script tự động chạy và chụp màn hình terminal
│   ├── roc.png                      # Biểu đồ đường ROC của Bài tập 4
│   └── screenshot_bai1..6.png       # Ảnh chụp màn hình kết quả chạy 6 bài tập
├── roc.png                          # Bản sao biểu đồ đường ROC tại thư mục gốc
├── Lab07_DanhGiaMoHinh.pdf          # Tài liệu thực hành & đề bài
└── README.md                        # Báo cáo tổng hợp kết quả
```

---

## Tóm tắt kết quả 6 bài tập

### Bài tập 1: Tự viết hàm đánh giá (`baitap07/bai1.py`)
- Hàm `danh_gia(y_that, y_doan)` tính `accuracy`, `precision`, `recall`, `F1` chỉ bằng `numpy`, khớp hoàn toàn với `scikit-learn`:
  - **Hồi quy logistic:** `accuracy = 0.9467`, `precision = 0.8500`, `recall = 0.5667`, `F1 = 0.6800`.
  - **Mô hình lười (`DummyClassifier(strategy="most_frequent")`):** `accuracy = 0.9000`, `precision = 0.0000`, `recall = 0.0000`, `F1 = 0.0000`.
- **Giải thích:** Với mô hình lười, mô hình luôn đoán lớp `0` nên `TP = 0` và `FP = 0`, mẫu số `TP + FP = 0`. Về toán học, `precision = 0 / 0` là không xác định. Hàm `danh_gia` kiểm tra mẫu số `> 0` trước khi chia và trả về `0.0` khi mẫu số bằng `0` (tương đương `zero_division=0` trong `scikit-learn`).

### Bài tập 2: Chữ số viết tay (`baitap07/bai2.py`)
- Chạy lại `SVC(kernel="rbf")` trên `load_digits` (`450` ảnh kiểm tra): `accuracy = 0.9911`, `macro avg F1 = 0.9911`, `weighted avg F1 = 0.9911`.
- **Chữ số có recall thấp nhất:** chữ số **`8`** (`0.9767`).
- **Chữ số có precision thấp nhất:** chữ số **`8`** (`0.9767`).
- **Vì sao `macro avg` và `weighted avg` gần như bằng nhau?** Vì số mẫu thật (`support`) của 10 chữ số rất đồng đều (từ `43` đến `46` ảnh mỗi lớp), nên trọng số theo tỷ lệ mẫu trong `weighted avg` gần như bằng `1/10` giống như `macro avg`.

### Bài tập 3: Chọn ngưỡng theo yêu cầu (`baitap07/bai3.py`)
- Dùng `cross_val_predict(..., cv=5, method="predict_proba")` trên tập học (`900` sinh viên, `89` bỏ học):
  - Với lưới bước `0.05` như mục 5, ngưỡng lớn nhất đạt `recall >= 0.8` là **`0.15`** (`recall CV = 0.8315`). *(Tham khảo: ở bước mịn `0.01`, ngưỡng lớn nhất là `0.17` với `recall CV = 0.8090`).*
- Áp ngưỡng `0.15` lên tập kiểm tra **đúng một lần**:
  - **Recall:** `0.8333` (`25/30` bạn bỏ học) $\rightarrow$ **Đạt** yêu cầu `>= 0.8`.
  - **Precision:** `0.5102` (`25/49` lời báo đúng).
  - **Số bạn bị báo:** `49` bạn (`TP = 25`, `FP = 24`, bỏ sót `FN = 5`).

### Bài tập 4: Tự vẽ đường ROC (`baitap07/bai4.py`)
- Xuất hình `baitap07/roc.png` (và `roc.png`):
  - **Hồi quy logistic:** `AUC = 0.9449`.
  - **Rừng ngẫu nhiên (200 cây):** `AUC = 0.9249`.
- **Đường nào ôm sát góc trên bên trái hơn?** Đường của **Hồi quy logistic** ôm sát góc trên bên trái hơn nhìn chung.
- **Có đoạn nào hai đường cắt nhau không?** **Có**, tại đoạn đầu rất nhỏ (`FPR` từ `0.0000` đến `0.0074`, ở `FPR = 0.0037` Rừng đạt `TPR = 0.5333` vượt lên trên Logistic `0.4000`) và ở đoạn cuối (`FPR` từ `0.4741` đến `0.7296`, khi Rừng nối thẳng từ `(0.4741, 0.9667)` lên `(1.0, 1.0)` còn Logistic giữ `0.9667` tới `FPR = 0.7296`).

### Bài tập 5: Cân bằng lớp cho mô hình khác (`baitap07/bai5.py`)
| Mô hình | Trạng thái | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: |
| `RandomForestClassifier(200)` | Trước cân bằng | `0.7727` | `0.5667` | `0.6538` |
| `RandomForestClassifier(200)` | `class_weight="balanced"` | `0.7500` | `0.4000` | `0.5217` |
| `SVC` (`RBF + StandardScaler`) | Trước cân bằng | `0.8824` | `0.5000` | `0.6383` |
| `SVC` (`RBF + StandardScaler`) | `class_weight="balanced"` | `0.4464` | `0.8333` | `0.5814` |

- **Mô hình thay đổi nhiều nhất:** **`SVC`**, với `recall` tăng vọt `+0.3333` (từ `0.5000` lên `0.8333`) và `precision` giảm `-0.4359` (từ `0.8824` xuống `0.4464`).

### Bài tập 6: MAE và RMSE với một lần đoán hỏng (`baitap07/bai6.py`)
- Trên dữ liệu `gia_nha.csv` (mô hình hồi quy tuyến tính theo `dien_tich`, `test_size=0.2, random_state=42`):
  - **Trước khi sửa:** `MAE = 0.2578`, `RMSE = 0.3730`, `R² = 0.9622`.
  - **Sau khi cộng `+5` tỷ vào 1 căn kiểm tra:** `MAE = 0.6745` (gấp `2.62` lần), `RMSE = 1.5505` (gấp `4.16` lần), `R² = 0.3475` (giảm `-0.6148`).
- **Kết luận:** Giữa hai thước đo sai số, **RMSE** thay đổi nhiều hơn hẳn **MAE** (và `R²` cũng sụt giảm nghiêm trọng) vì cả RMSE lẫn `R²` đều **bình phương sai số** trước khi lấy trung bình, nên một sai số lớn (`~5` tỷ $\rightarrow$ bình phương `~25`) bị phạt nặng hơn rất nhiều so với cách lấy trị tuyệt đối tuyến tính của MAE.
