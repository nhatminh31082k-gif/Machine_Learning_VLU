# BÀI THỰC HÀNH 5: CÂY QUYẾT ĐỊNH VÀ RỪNG NGẪU NHIÊN
**Môn học:** Học máy ứng dụng (Machine Learning)  
**Giảng viên:** ThS. Nguyễn Thái Anh  
**Đơn vị:** Khoa Công nghệ Thông tin - Trường Đại học Văn Lang  

---

## 📁 Cấu trúc thư mục

```text
05. DecisionTree_RandomForest/
├── data/
│   ├── gia_nha.csv                                # Dữ liệu bài toán hồi quy (60 căn nhà)
│   └── khach_vay.csv                              # Dữ liệu phân loại rủi ro tín dụng (600 khách)
├── code/                                          # Mã nguồn 7 ví dụ thực hành trong giáo trình
│   ├── f1_gini.py                                 # Bước 1: Tự tính Gini và chọn câu hỏi tốt nhất
│   ├── f2_cay_nam_la.py                           # Bước 2: Học và đọc cây 5 lá
│   ├── f3_do_sau.py                               # Bước 3: Cây học vẹt và chọn độ sâu bằng 5-fold CV
│   ├── f4_khong_chuan_hoa.py                      # Bước 4: Kiểm chứng cây không cần chuẩn hoá
│   ├── f5_rung.py                                 # Bước 5: Trồng rừng ngẫu nhiên (Random Forest)
│   ├── f6_do_quan_trong.py                        # Bước 6: Đọc độ quan trọng của đặc trưng (MDI)
│   └── f7_cay_hoi_quy.py                          # Bước 7: Cây quyết định cho hồi quy giá nhà
├── baitap05/                                      # Mã nguồn và sản phẩm 6 bài tập thực hành
│   ├── bai1.py                                    # Bài 1: Vẽ cây 5 lá & cây sâu 2, giải thích chia nút
│   ├── bai2.py                                    # Bài 2: Tự tìm câu hỏi gốc trên 450 khách tập học
│   ├── bai3.py                                    # Bài 3: So sánh thước đo Gini vs Entropy
│   ├── bai4.py                                    # Bài 4: Cắt tỉa bằng min_samples_leaf
│   ├── bai5.py                                    # Bài 5: Phân tích độ quan trọng xáo trộn (PFI)
│   ├── bai6.py                                    # Bài 6: Rừng ngẫu nhiên hồi quy & vấn đề ngoại suy
│   ├── tao_anh_chup.py                            # Script tự động chạy & chụp ảnh màn hình terminal
│   ├── cay_nam_la.png                             # Hình vẽ cây 5 lá
│   ├── cay_sau_2.png                              # Hình vẽ cây độ sâu 2
│   ├── hoi_quy_gia_nha.png                        # Đồ thị so sánh dự đoán hồi quy & ngoại suy
│   ├── screenshot_bai1.png                        # Ảnh chụp màn hình kết quả Bài 1
│   ├── screenshot_bai2.png                        # Ảnh chụp màn hình kết quả Bài 2
│   ├── screenshot_bai3.png                        # Ảnh chụp màn hình kết quả Bài 3
│   ├── screenshot_bai4.png                        # Ảnh chụp màn hình kết quả Bài 4
│   ├── screenshot_bai5.png                        # Ảnh chụp màn hình kết quả Bài 5
│   └── screenshot_bai6.png                        # Ảnh chụp màn hình kết quả Bài 6
├── cay_nam_la.png                                 # Bản sao cây 5 lá tại thư mục gốc
├── Lab05_CayQuyetDinh_RungNgauNhien.pdf           # Đề bài và tài liệu hướng dẫn Lab 5
└── README.md                                      # Báo cáo tổng hợp chi tiết bài thực hành
```

---

## 📌 Báo cáo chi tiết 6 bài tập

### 📝 Bài tập 1: Tự vẽ cây
* **Mã nguồn:** `baitap05/bai1.py`
* **Lệnh chạy:**
  ```powershell
  python baitap05/bai1.py
  ```
* **Hình ảnh tạo ra:** `baitap05/cay_nam_la.png`, `baitap05/cay_sau_2.png`.
* **Cây sâu 2 (dạng chữ):**
  ```text
  |--- thu_nhap <= 15.050
  |   |--- ty_le_no <= 0.615
  |   |   |--- weights: [68.000, 53.000] class: 0
  |   |--- ty_le_no >  0.615
  |   |   |--- weights: [114.000, 14.000] class: 0
  |--- thu_nhap >  15.050
  |   |--- ty_le_no <= 1.020
  |   |   |--- weights: [27.000, 145.000] class: 1
  |   |--- ty_le_no >  1.020
  |   |   |--- weights: [20.000, 9.000] class: 0
  ```
* **Câu hỏi & Trả lời:**
  1. *Hai lá bên trái của cây này có cùng kết luận không?*
     * **CÓ**, cả hai lá con bên trái đều đưa ra kết luận là lớp `0` (trễ hạn):
       * Lá 1 (`ty_le_no <= 0.615`): có 68 khách trễ hạn, 53 khách đúng hạn $\rightarrow$ lớp đa số là 0.
       * Lá 2 (`ty_le_no > 0.615`): có 114 khách trễ hạn, 14 khách đúng hạn $\rightarrow$ lớp đa số là 0.
  2. *Nếu có, vì sao cây vẫn chia nút đó ra làm hai?*
     * Thuật toán cây quyết định CART phân chia nút dựa trên tiêu chí **giảm độ bất thuần khiết (Gini Impurity Gain)**, chứ không căn cứ vào việc nhãn đa số có thay đổi hay không:
       * Tại nút cha (`thu_nhap <= 15.05`): có 249 khách (182 trễ, 67 đúng), $G_{cha} = 1 - (182/249)^2 - (67/249)^2 \approx 0.3926$.
       * Sau khi chia theo `ty_le_no <= 0.615`:
         * Nhánh trái: 121 khách [68 trễ, 53 đúng], $G_{trai} \approx 0.4921$ (lẫn lộn nhiều).
         * Nhánh phải: 128 khách [114 trễ, 14 đúng], $G_{phai} \approx 0.1947$ (rất thuần khiết, xác suất trễ hạn tới 89.06%).
         * Gini trung bình có trọng số sau chia: $G_{chia} = \frac{121}{249} \times 0.4921 + \frac{128}{249} \times 0.1947 \approx 0.3392$.
         * Độ giảm Gini: $\Delta G = 0.3926 - 0.3392 = 0.0534 > 0$.
     * Nhờ phân chia, cây tách được một nhóm 128 khách có xác suất trễ hạn gần như tuyệt đối (89%), làm tăng độ tin cậy xác suất dự đoán (`predict_proba`) và giảm mạnh độ bất định.

---

### 📝 Bài tập 2: Tự tìm câu hỏi gốc trên dữ liệu thật
* **Mã nguồn:** `baitap05/bai2.py`
* **Lệnh chạy:**
  ```powershell
  python baitap05/bai2.py
  ```
* **Bảng kết quả thử mọi đặc trưng trên 450 khách tập học:**

| Tên cột | Số lượng ngưỡng thử | Ngưỡng tốt nhất | Gini sau chia | Mức Gini giảm ($\Delta G$) |
| :--- | :---: | :---: | :---: | :---: |
| **thu_nhap** | **207** | **15.0500** | **0.3777** | **0.1221 (Cao nhất)** |
| ty_le_no | 111 | 0.6050 | 0.4412 | 0.0586 |
| co_viec_on_dinh | 1 | 0.5000 | 0.4669 | 0.0329 |
| so_nguoi_phu_thuoc | 4 | 2.5000 | 0.4764 | 0.0235 |

* **Đối chiếu với Scikit-learn:**
  * Hàm tự viết: chọn cột `thu_nhap`, ngưỡng `15.0500`, $\Delta G = 0.1221$.
  * `DecisionTreeClassifier(max_depth=1)`: chọn cột `thu_nhap`, ngưỡng `15.0500`.
  * **Kết luận:** Kết quả tìm thủ công trùng khớp 100% với scikit-learn và tài liệu giáo trình.

---

### 📝 Bài tập 3: Gini hay Entropy
* **Mã nguồn:** `baitap05/bai3.py`
* **Lệnh chạy:**
  ```powershell
  python baitap05/bai3.py
  ```
* **Bảng so sánh chi tiết:**

| Tiêu chí | Gini (Mục 4) | Entropy (Bài 3) |
| :--- | :---: | :---: |
| **Độ sâu tốt nhất ($d_{tot}$)** | **5** | **5** |
| Điểm CV 5-fold cao nhất | 0.8333 | 0.8222 |
| Số lá của cây tối ưu | 27 lá | 23 lá |
| **Accuracy trên tập kiểm tra** | **0.8533** (128/150) | **0.8467** (127/150) |
| Số mẫu kiểm tra dự đoán khác nhau | \multicolumn{2}{c|}{7 / 150 mẫu (~4.7%)} |

* **Nhận xét một câu:**  
  > *"Đổi thước đo từ Gini sang Entropy không làm kết quả khác đi nhiều: cả hai đều tìm ra cùng độ sâu tối ưu $d = 5$ và cho độ chính xác trên tập kiểm tra gần như tương đương nhau (0.8533 so với 0.8467, chỉ lệch đúng 7 khách)."*
* **Giải thích:** Cả Gini và Entropy đều là hàm đo độ hỗn loạn đối xứng, đạt giá trị 0 tại nút thuần khiết và đạt cực đại tại phân bố đều 50-50. Vì hình dạng đường cong rất tương đồng nên trong hơn 95% trường hợp, hai thước đo này chọn cùng câu hỏi phân chia tại mỗi nút. Gini thường được mặc định sử dụng vì tính toán số học nhanh hơn, không phải tính hàm logarit.

---

### 📝 Bài tập 4: Cắt tỉa bằng số khách tối thiểu ở lá (min_samples_leaf)
* **Mã nguồn:** `baitap05/bai4.py`
* **Lệnh chạy:**
  ```powershell
  python baitap05/bai4.py
  ```
* **Bảng khảo sát `min_samples_leaf`:**

| `min_samples_leaf` | Độ sâu | Số lá | Accuracy tập học | CV (5-fold) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 13 | 92 | 1.0000 | 0.7733 |
| 2 | 11 | 72 | 0.9467 | 0.7822 |
| 5 | 10 | 46 | 0.8956 | 0.8067 |
| **10** | **8** | **30** | **0.8600** | **0.8089 (Tốt nhất)** |
| 20 | 7 | 17 | 0.8422 | 0.8089 |
| 40 | 4 | 8 | 0.7822 | 0.7133 |

* **So sánh hai phương pháp cắt tỉa:**

| Phương pháp cắt tỉa | Độ sâu | Số lá | Điểm CV | Accuracy kiểm tra |
| :--- | :---: | :---: | :---: | :---: |
| Cắt tỉa bằng `min_samples_leaf = 10` | 8 | 30 | 0.8089 | 0.8467 |
| **Giới hạn độ sâu `max_depth = 5` (Mục 4)** | **5** | **27** | **0.8333** | **0.8533** |

* **Trả lời câu hỏi:**
  * **Cách cắt tỉa bằng giới hạn độ sâu `max_depth = 5` cho kết quả TỐT HƠN** cả trên tập học (CV 0.8333 vs 0.8089) lẫn trên tập kiểm tra (0.8533 vs 0.8467).
  * **Giải thích:** Giới hạn `max_depth = 5` khống chế toàn diện chiều dài đường đi từ gốc tới lá, ngăn chặn triệt để các chuỗi câu hỏi quá sâu và loại bỏ các quy tắc học vẹt trên toàn bộ cấu trúc cây. Ngược lại, `min_samples_leaf` chỉ khống chế cỡ mẫu tối thiểu ở nút lá nên cây vẫn mọc sâu tới 8 tầng, dẫn tới nguy cơ vẫn tồn tại những nhánh phức tạp thừa thãi.

---

### 📝 Bài tập 5: Độ quan trọng bằng cách xáo trộn (Permutation Importance)
* **Mã nguồn:** `baitap05/bai5.py`
* **Lệnh chạy:**
  ```powershell
  python baitap05/bai5.py
  ```
* **Bảng so sánh thứ hạng đặc trưng:**

| Đặc trưng | MDI (`feature_importances_`) | Hạng MDI | PFI mean (Xáo trộn) | Hạng PFI |
| :--- | :---: | :---: | :---: | :---: |
| **thu_nhap** | 0.4269 | 1 | 0.3073 | **1** |
| **ty_le_no** | 0.2696 | 2 | 0.1993 | **2** |
| **co_viec_on_dinh** | 0.0773 | 4 | 0.1117 | **3** |
| **so_nguoi_phu_thuoc** | 0.0764 | 5 | 0.0497 | **4** |
| **so_ngau_nhien** | **0.1499** | **3 (Ảo giác)** | **0.0017** | **5 (Cuối cùng)** |

* **Trả lời câu hỏi:**
  1. *Cột ngẫu nhiên xếp thứ mấy theo Permutation Importance?*
     * Xếp thứ **5 / 5 (hạng cuối cùng)** với mức giảm accuracy chỉ $0.0017 \approx 0$.
  2. *So với thứ hạng của nó theo `feature_importances_`:*
     * Theo `feature_importances_` (MDI), cột ngẫu nhiên xếp **thứ 3/5** (chiếm tới 15.0% độ quan trọng, vượt qua cả 2 cột thật là `co_viec_on_dinh` và `so_nguoi_phu_thuoc`).
     * Theo `permutation_importance` (PFI), nó rơi thẳng xuống **hạng cuối cùng (5/5)**.
  3. *Giải thích sự khác nhau:*
     * **MDI đo trên tập học:** `so_ngau_nhien` là biến liên tục có 450 giá trị duy nhất (high cardinality). Cây quyết định mọc sâu luôn có thể tìm ra những ngưỡng ngẫu nhiên giúp tách biệt một vài mẫu trên tập học để giảm Gini, khiến chỉ số MDI bị thổi phồng giả tạo (hiện tượng overfitting đặc trưng).
     * **PFI đo trên tập kiểm tra:** PFI xáo trộn dữ liệu của cột trên tập kiểm tra độc lập và xem accuracy bị giảm đi bao nhiêu. Vì cột ngẫu nhiên thực chất không mang tri thức dự đoán thật sự, nên việc xáo trộn nó không ảnh hưởng gì tới kết quả phân loại ($\Delta \approx 0$), phản ánh khách quan và chính xác sự vô giá trị của cột này.

---

### 📝 Bài tập 6: Rừng ngẫu nhiên cho bài toán hồi quy & vấn đề ngoại suy
* **Mã nguồn:** `baitap05/bai6.py`
* **Lệnh chạy:**
  ```powershell
  python baitap05/bai6.py
  ```
* **Hình ảnh tạo ra:** `baitap05/hoi_quy_gia_nha.png`
* **So sánh MAE trên tập kiểm tra:**

| Mô hình | MAE tập kiểm tra (tỷ đồng) |
| :--- | :---: |
| **Đường thẳng (`LinearRegression`)** | **0.2578 (Tốt nhất)** |
| Rừng 200 cây (`RandomForestRegressor`) | 0.4559 |
| Cây sâu 3 (`DecisionTreeRegressor`) | 0.5039 |

* **Dự đoán giá cho các căn nhà mới:**

| Diện tích | Đường thẳng (Linear) | Cây sâu 3 (Decision Tree) | Rừng 200 cây (Random Forest) |
| :---: | :---: | :---: | :---: |
| 60 m² | 5.12 tỷ | 4.92 tỷ | 5.09 tỷ |
| 100 m² | 8.24 tỷ | 8.32 tỷ | 7.89 tỷ |
| 150 m² | 12.13 tỷ | 9.40 tỷ | 9.20 tỷ |
| **250 m²** | **19.91 tỷ** | **9.40 tỷ** | **9.20 tỷ** |

* **Trả lời câu hỏi:**
  1. *So sánh MAE:* Đường thẳng cho sai số thấp nhất ($0.2578$ tỷ), tốt hơn nhiều so với Rừng ($0.4559$ tỷ) và Cây ($0.5039$ tỷ), vì giá nhà thực tế tăng tuyến tính liên tục theo diện tích.
  2. *Dự đoán căn 250 m²:*
     * Hồi quy tuyến tính: **19.91 tỷ**
     * Cây sâu 3: **9.40 tỷ**
     * Rừng 200 cây: **9.20 tỷ**
  3. *Rừng có ngoại suy được không?*
     * **HOÀN TOÀN KHÔNG NGOẠI SUY ĐƯỢC.** Giá nhà dự đoán bị chặn cứng ở mức ~9.20 tỷ đồng dù diện tích tăng vọt lên 250 m².
  4. *Giải thích bằng cách lá của mỗi cây đưa ra dự đoán:*
     * Tại mỗi nút lá của cây quyết định hồi quy, giá trị dự đoán là **trung bình cộng** giá trị nhãn $y$ của các mẫu học rơi vào lá đó: $\hat{y}_{leaf} = \frac{1}{|S|}\sum_{i \in S} y_i$. Do đó, kết quả dự đoán của bất kỳ nút lá nào luôn bị chặn trên bởi giá trị cực đại trong tập học: $\hat{y} \le \max(y_{train}) = 9.74$ tỷ.
     * Khi gặp căn nhà 250 m² (nằm ngoài miền diện tích học $[37.6, 117.5]\text{ m}^2$), mẫu này luôn đi theo nhánh rẽ phải ngoài cùng của mọi cây và rơi vào chiếc lá tận cùng bên phải.
     * Tại chiếc lá đó, mỗi cây chỉ có thể trả về giá trung bình của các căn hộ lớn nhất mà nó thấy trong mẫu con bootstrap (quanh 9.2 - 9.4 tỷ).
     * Kết quả dự đoán của rừng là trung bình cộng của 200 cây, nên giá trị dự đoán bị "san phẳng" ở mức 9.20 tỷ, không thể ngoại suy theo xu hướng tăng trưởng tuyến tính như `LinearRegression`.

---

## 📸 Ảnh chụp màn hình kết quả chạy thực tế

Toàn bộ ảnh chụp màn hình được lưu trong thư mục `baitap05/`:
- `screenshot_bai1.png`: Kết quả chạy Bài tập 1.
- `screenshot_bai2.png`: Kết quả chạy Bài tập 2.
- `screenshot_bai3.png`: Kết quả chạy Bài tập 3.
- `screenshot_bai4.png`: Kết quả chạy Bài tập 4.
- `screenshot_bai5.png`: Kết quả chạy Bài tập 5.
- `screenshot_bai6.png`: Kết quả chạy Bài tập 6.
- `cay_nam_la.png`: Biểu đồ trực quan cây quyết định 5 lá.
- `cay_sau_2.png`: Biểu đồ trực quan cây quyết định độ sâu 2.
- `hoi_quy_gia_nha.png`: Biểu đồ thể hiện hạn chế ngoại suy của Cây và Rừng hồi quy so với Đường thẳng.
