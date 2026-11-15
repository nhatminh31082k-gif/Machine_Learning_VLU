# Báo Cáo & Lời Giải Bài Tập Thực Hành 4: K Láng Giềng Gần Nhất (KNN)

Tài liệu: `Lab04_KNN.pdf`  
Môn học: **Học máy ứng dụng** - Khoa Công nghệ Thông tin, Trường Đại học Văn Lang  
Giảng viên: **ThS. Nguyễn Thái Anh**  
Năm học: 2026 – 2027

---

## 📁 Cấu trúc thư mục

```text
04. KNN/
├── data/
│   ├── sinh_vien_mo_rong.csv     # Dữ liệu sinh viên mở rộng (thêm cột chi tiêu)
│   ├── hai_vang_trang.csv        # Dữ liệu phân loại 2 lớp có ranh giới cong
│   └── gia_nha.csv               # Dữ liệu diện tích và giá nhà (hồi quy)
├── code/                         # 5 tệp mã ví dụ minh họa trong tài liệu
│   ├── e1_khoang_cach.py         # Mục 2: Tự tính khoảng cách và tìm láng giềng
│   ├── e2_chuan_hoa.py           # Mục 3: Đo hậu quả khi quên chuẩn hoá
│   ├── e3_chon_k.py              # Mục 4: Chọn k bằng kiểm định chéo
│   ├── e4_so_sanh.py             # Mục 5: So sánh KNN với Logistic Regression & Naive Bayes
│   └── e5_knn_hoi_quy.py         # Mục 6: KNN cho bài toán hồi quy
├── baitap04/                     # 6 bài tập tự luyện được yêu cầu nộp
│   ├── bai1.py                   # Tự viết KNN không dùng thư viện
│   ├── bai2.py                   # Khoảng cách Manhattan vs Euclid
│   ├── bai3.py                   # Đổi đơn vị đo và vai trò của chuẩn hoá
│   ├── bai4.py                   # Láng giềng có trọng số (weights="distance")
│   ├── bai5.py                   # Chọn k cho bài toán hồi quy giá nhà
│   ├── bai6.py                   # Đo thời gian fit/predict (Mô hình học lười)
│   ├── screenshot_bai1.png -> screenshot_bai6.png  # Ảnh chụp kết quả màn hình
│   ├── hinh_anh_hai_vang_trang.png                 # Trực quan hoá biên quyết định
│   └── hinh_anh_hoi_quy_gia_nha.png                # Trực quan hoá hồi quy & ngoại suy
├── bai1.py -> bai6.py            # Bản sao tại thư mục gốc để chạy linh hoạt
├── tao_anh_chup.py               # Script tự động chụp màn hình terminal PowerShell
├── ve_bieu_do.py                 # Script vẽ biểu đồ phân tích trực quan
└── README.md                     # Báo cáo tổng hợp chi tiết
```

### Cách thực thi các bài tập

Từ thư mục `04. KNN`:
```bash
# Chạy từng bài tập trong thư mục baitap04 theo đúng yêu cầu đề bài
python baitap04/bai1.py
python baitap04/bai2.py
python baitap04/bai3.py
python baitap04/bai4.py
python baitap04/bai5.py
python baitap04/bai6.py
```
*(Các script đều được thiết kế linh hoạt, có thể chạy trực tiếp từ `d:\ML_ungdung`, `04. KNN`, hoặc bên trong thư mục `baitap04` mà không gặp lỗi đường dẫn dữ liệu).*

---

## 📌 Tổng quan các con số kết quả chính

Bảng đối chiếu chuẩn theo hướng dẫn của giảng viên:

| Đại lượng | Giá trị thực nghiệm | Ý nghĩa / Ghi chú |
|---|---|---|
| $d(A, B)$ và $d(A, C)$ khi **chưa chuẩn hoá** | **102.2** và **3000.0** | Cột chi tiêu nuốt trọn khoảng cách |
| $d(A, B)$ và $d(A, C)$ khi **đã chuẩn hoá** | **3.85** và **1.79** | A gần C nhất (đúng trực giác học tập) |
| Ba cột, không chuẩn hoá (k=7) | **0.5333** | Tệ hơn cả đoán bừa cả lớp qua (0.6000) |
| Ba cột, đổi chi tiêu sang triệu đồng | **0.8667** | Thang đo co lại, accuracy tăng vọt |
| Ba cột, có chuẩn hoá (StandardScaler) | **0.8667** | Chuẩn hoá bắt buộc cho KNN |
| Bỏ cột chi tiêu thừa, có chuẩn hoá | **0.9000** | Loại bỏ nhiễu từ đặc trưng không liên quan |
| Mức đoán bừa (Majority Dummy) | **0.6000** | Điểm chuẩn cơ sở trên tập test sinh viên |
| $k$ tốt nhất theo CV (Euclid, hai vầng trăng) | **13** (CV = 0.8667) | Test Accuracy = **0.9200** |
| $k$ tốt nhất theo CV (Manhattan, hai vầng trăng) | **27** (CV = 0.8667) | Test Accuracy = **0.9067** |
| Hai vầng trăng: Logistic & Naive Bayes | Cùng đạt **0.8533** | Thua KNN (0.9200) do biên cong phi tuyến |
| Giá nhà: MAE KNN ($k=5$) và Đường thẳng | **0.4030** vs **0.2578** tỷ | Đường thẳng thắng vì quan hệ thực tuyến tính |
| Giá nhà: MAE KNN tối ưu ($k=2$) | **0.4867** tỷ (CV = 0.3498) | Vẫn không vượt được mô hình đường thẳng |

---

## 📝 Lời giải & Phân tích chi tiết từng bài tập

---

### Bài tập 1: Tự viết KNN không dùng thư viện

#### 1. Yêu cầu đề bài
- Dựa vào `e1_khoang_cach.py`, viết hàm `knn_du_doan(X_hoc, y_hoc, x_moi, k)` trả về nhãn dự đoán cho một mẫu mới.
- Dùng hàm đó dự đoán cho toàn bộ tập kiểm tra của bộ `hai_vang_trang.csv` với $k = 13$, rồi tính Accuracy.
- So sánh kết quả với `KNeighborsClassifier` ở mục 4 (hai con số phải bằng nhau). Gợi ý: dùng `np.argsort` để lấy chỉ số của $k$ khoảng cách nhỏ nhất.

#### 2. Cơ chế thuật toán & Công thức toán học
1. **Khoảng cách Euclid (Euclidean Distance)**:
   Khoảng cách từ mẫu mới $x_{mới}$ tới mẫu thứ $i$ trong tập học $x_i$:
   $$d(x_{mới}, x_i) = \sqrt{\sum_{j=1}^{D} (x_{mới, j} - x_{i, j})^2}$$
2. **Tìm $k$ láng giềng gần nhất**:
   Sử dụng hàm `np.argsort(khoang_cach)[:k]` để lấy ra $k$ chỉ số có khoảng cách nhỏ nhất mà không cần sắp xếp lại toàn bộ mảng.
3. **Bỏ phiếu đa số (Majority Voting)**:
   Với nhãn nhị phân $y \in \{0, 1\}$, tính số phiếu của từng lớp:
   $$\hat{y} = \begin{cases} 1 & \text{nếu } \sum_{i \in \mathcal{N}_k} y_i > \frac{k}{2} \\ 0 & \text{ngược lại} \end{cases}$$
   (Vì chọn $k$ lẻ nên không bao giờ xảy ra trường hợp hòa phiếu).

#### 3. Kết quả thực nghiệm
- Tập học: 225 mẫu | Tập kiểm tra: 75 mẫu
- **Accuracy KNN tự viết**: `0.9200` (69/75 mẫu đúng)
- **Accuracy KNeighborsClassifier (scikit-learn)**: `0.9200` (69/75 mẫu đúng)
- **Số mẫu dự đoán hoàn toàn trùng khớp**: `75/75 (100.0%)`
- **Kết luận**: Hai con số bằng nhau tuyệt đối!

#### 4. Giải thích cơ chế hoạt động của KNN
1. **KNN là mô hình học lười (Lazy Learning / Instance-based Learning)**:
   - Trong pha huấn luyện (`.fit`), KNN không hề tối ưu hóa hàm mất mát hay tính toán trọng số $\theta$ như Hồi quy tuyến tính hay Logistic.
   - Nó chỉ đơn thuần lưu trữ toàn bộ dữ liệu mẫu vào bộ nhớ RAM.
2. **Gánh nặng tính toán dồn sang pha dự đoán (`.predict`)**:
   - Khi gặp mẫu mới, KNN đo khoảng cách tới toàn bộ tập học, lọc ra $k$ láng giềng gần nhất và tổng hợp ý kiến đa số.
   - Do tuân thủ cùng công thức khoảng cách Euclid và quy tắc bỏ phiếu đa số với $k=13$, hàm tự viết đạt kết quả trùng khớp 100% với thư viện chuẩn `scikit-learn`.

---

### Bài tập 2: Khoảng cách Manhattan

#### 1. Yêu cầu đề bài
- Chạy lại `e3_chon_k.py` nhưng thêm tham số `metric="manhattan"` vào `KNeighborsClassifier`.
- Ghi lại $k$ tốt nhất và Accuracy trên tập kiểm tra, rồi so với khoảng cách Euclid.
- Nhận xét một câu: Đổi cách đo khoảng cách có làm kết quả khác đi nhiều không?

#### 2. So sánh toán học giữa Euclid và Manhattan
- **Khoảng cách Euclid ($L_2$ norm)**:
  $$d_{\text{Euclid}}(x, z) = \|x - z\|_2 = \sqrt{\sum_{j=1}^{D} (x_j - z_j)^2}$$
  - Hình dạng vùng đẳng khoảng cách: **Đường tròn / Hình cầu**.
  - Rất nhạy cảm với các khoảng cách lớn do phép bình phương.
- **Khoảng cách Manhattan ($L_1$ norm / Taxi-cab distance)**:
  $$d_{\text{Manhattan}}(x, z) = \|x - z\|_1 = \sum_{j=1}^{D} |x_j - z_j|$$
  - Hình dạng vùng đẳng khoảng cách: **Hình quả trám (hình vuông xoay 45 độ)**.
  - Đo độ dài đường gấp khúc song song với các trục tọa độ.

#### 3. Kết quả thực nghiệm đối đầu

| Tiêu chí | Khoảng cách Euclid ($L_2$) | Khoảng cách Manhattan ($L_1$) |
|---|---|---|
| **Công thức** | $\sqrt{\sum (x_j - z_j)^2}$ | $\sum \|x_j - z_j\|$ |
| **$k$ tốt nhất theo 5-fold CV** | **$k = 13$** | **$k = 27$** |
| **Điểm CV cao nhất trên tập học** | **0.8667** | **0.8667** |
| **Accuracy trên tập kiểm tra** | **0.9200** (69/75 mẫu) | **0.9067** (68/75 mẫu) |
| **Số mẫu kiểm tra dự đoán đúng** | 69 mẫu | 68 mẫu (chỉ lệch đúng 1 mẫu) |

#### 4. Nhận xét theo yêu cầu
> **Nhận xét**: Đổi cách đo khoảng cách từ Euclid sang Manhattan **KHÔNG làm kết quả khác đi nhiều** (điểm kiểm định chéo CV cùng đạt $0.8667$, Accuracy trên tập kiểm tra chỉ chênh lệch $0.0133$, tương đương đúng $1$ mẫu trong tổng số $75$ mẫu kiểm tra).  
> **Nguyên nhân**: Trên bộ dữ liệu 2 chiều liên tục như hai vầng trăng, hình dạng lân cận giữa đường tròn (Euclid) và hình quả trám (Manhattan) nhìn chung vẫn bao bọc các tập hợp điểm láng giềng gần như tương đồng nhau, không làm đảo lộn cấu trúc ranh giới phi tuyến.

---

### Bài tập 3: Đổi đơn vị đo & Vai trò của Chuẩn hoá

#### 1. Yêu cầu đề bài
- Trên bộ `sinh_vien_mo_rong.csv`, đổi cột `chi_tieu` từ nghìn đồng sang triệu đồng bằng cách chia cho 1000.
- Chạy KNN với $k = 7$ không chuẩn hoá trên ba cột mới.
- Accuracy thay đổi thế nào so với $0.5333$ ở mục 3?
- Giải thích vì sao chỉ đổi đơn vị mà kết quả lại khác hẳn?
- Nếu có chuẩn hoá thì đổi đơn vị có làm kết quả thay đổi không?

#### 2. Thống kê thang đo (Độ lệch chuẩn $\sigma$) trước và sau đổi đơn vị

| Đặc trưng | Ban đầu (nghìn đồng) | Sau khi chia 1000 (triệu đồng) | Tỷ lệ thang đo |
|---|---|---|---|
| `gio_on` | $\sigma = 7.85$ giờ | $\sigma = 7.85$ giờ | Giữ nguyên |
| `diem_giua_ky` | $\sigma = 2.10$ điểm | $\sigma = 2.10$ điểm | Giữ nguyên |
| `chi_tieu` | $\sigma = \mathbf{1698.8}$ nghìn đồng | $\sigma = \mathbf{1.699}$ triệu đồng | **Giảm 1000 lần** |

#### 3. Kết quả thực nghiệm

| Trường hợp thử nghiệm | Chuẩn hoá | Đơn vị cột `chi_tieu` | Accuracy tập kiểm tra |
|---|---|---|---|
| Mức đoán bừa (Majority Baseline) | Không | - | `0.6000` |
| **TH1: Mục 3 tài liệu** | **Không** | **Nghìn đồng** | **`0.5333`** (Tệ hơn đoán bừa!) |
| **TH2: Bài tập 3 yêu cầu** | **Không** | **Triệu đồng** | **`0.8667`** (Tăng vọt!) |
| **TH3: Có chuẩn hoá pipeline** | **StandardScaler** | **Nghìn đồng** | **`0.8667`** |
| **TH4: Có chuẩn hoá pipeline** | **StandardScaler** | **Triệu đồng** | **`0.8667`** (Bất biến!) |

#### 4. Trả lời & Giải thích câu hỏi
1. **Accuracy thay đổi thế nào so với $0.5333$ ở mục 3?**
   - Accuracy tăng vọt từ **`0.5333` lên `0.8667`** (tăng thêm $+0.3333$, tức tăng $+33.33\%$).
   - Từ mức dự đoán tệ hại (thua cả đoán bừa cả lớp qua $0.6000$), mô hình đã trở nên rất chính xác ($86.67\%$).
2. **Vì sao chỉ đổi đơn vị mà kết quả lại khác hẳn?**
   - KNN hoàn toàn "mù" về mặt ngữ nghĩa đơn vị; nó chỉ nhìn vào các con số thuần túy trong công thức khoảng cách Euclid:
     $$d^2 = (gio_1 - gio_2)^2 + (diem_1 - diem_2)^2 + (chitieu_1 - chitieu_2)^2$$
   - Khi dùng **nghìn đồng**, độ lệch chuẩn của `chi_tieu` là $1698.8$, gấp hơn 216 lần `gio_on` ($7.85$) và gấp hơn 800 lần `diem_giua_ky` ($2.10$). Độ chênh lệch hàng nghìn đơn vị của chi tiêu đã **nuốt trọn khoảng cách**, làm cho hai thuộc tính quyết định sống còn tới việc qua môn bị biến mất. Mô hình thực chất chỉ đang phân loại theo tiền chi tiêu!
   - Khi đổi sang **triệu đồng (chia 1000)**, độ lệch chuẩn của `chi_tieu` co lại chỉ còn $1.699$, nhỏ hơn `gio_on` ($7.85$) và tương đương `diem_giua_ky` ($2.10$). Cột chi tiêu không còn lấn át được nữa, giúp thông tin từ giờ ôn và điểm giữa kỳ được khôi phục sức nặng trong việc xác định láng giềng.
3. **Nếu có chuẩn hoá thì đổi đơn vị có làm kết quả thay đổi không?**
   - **TUYỆT ĐỐI KHÔNG THAY ĐỔI** (cả hai trường hợp đều đạt chính xác $0.8667$).
   - **Chứng minh toán học**:
     Chuẩn hoá $z$-score biến đổi đặc trưng theo công thức:
     $$z = \frac{x - \mu}{\sigma}$$
     Khi ta đổi đơn vị bằng một phép biến đổi tỉ lệ $x_{\text{mới}} = c \cdot x$ (ở đây $c = \frac{1}{1000}$):
     - Kỳ vọng mới: $\mu_{\text{mới}} = c \cdot \mu$
     - Độ lệch chuẩn mới: $\sigma_{\text{mới}} = c \cdot \sigma$
     - Giá trị sau chuẩn hoá:
       $$z_{\text{mới}} = \frac{c \cdot x - c \cdot \mu}{c \cdot \sigma} = \frac{c(x - \mu)}{c \cdot \sigma} = \frac{x - \mu}{\sigma} \equiv z$$
     Do $z_{\text{mới}} \equiv z$, tọa độ hình học của mọi điểm dữ liệu sau chuẩn hoá là đồng nhất $100\%$. Do đó, khoảng cách tương đối và thứ tự láng giềng không đổi, dẫn đến kết quả phân loại bất biến trước mọi phép đổi đơn vị tuyến tính.

---

### Bài tập 4: Láng giềng có trọng số (Weighted KNN)

#### 1. Yêu cầu đề bài
- Thêm tham số `weights="distance"` vào `KNeighborsClassifier`. Khi đó láng giềng càng gần thì phiếu càng nặng.
- Chạy lại với $k = 1, 13, 101$ trên bộ hai vầng trăng và so với bảng ở mục 4.
- Với $k = 101$, trọng số theo khoảng cách có giúp mô hình bớt cẩu thả không? Vì sao?

#### 2. Bảng kết quả so sánh giữa `weights="uniform"` và `weights="distance"`

| Số láng giềng $k$ | `weights="uniform"` (Tập học) | `weights="uniform"` (Kiểm tra) | `weights="distance"` (Tập học) | `weights="distance"` (Kiểm tra) |
|---|---|---|---|---|
| **$k = 1$** | $1.0000$ | $0.8000$ | $1.0000$ | $0.8000$ |
| **$k = 13$** | $0.8844$ | $0.9200$ | $1.0000$ | **$0.9333$** ($\uparrow +0.0133$) |
| **$k = 101$** | $0.8311$ | **$0.8533$** | $1.0000$ | **$0.9067$** ($\mathbf{\uparrow +0.0534}$) |

#### 3. Trả lời câu hỏi & Phân tích chuyên sâu
**Câu hỏi**: *Với $k = 101$, trọng số theo khoảng cách có giúp mô hình bớt cẩu thả không? Vì sao?*

> **TRẢ LỜI: CÓ, TRỌNG SỐ THEO KHOẢNG CÁCH GIÚP MÔ HÌNH BỚT CẨU THẢ RÕ RỆT.**  
> Độ chính xác trên tập kiểm tra tăng mạnh từ **$0.8533$ lên $0.9067$** (tăng thêm $+5.34\%$).

**Giải thích nguyên nhân toán học & hình học**:
1. **Khi dùng `weights="uniform"` (trọng số đều)**:
   - Tất cả 101 láng giềng đều có giá trị lá phiếu ngang bằng nhau ($w_i = 1$).
   - Tập huấn luyện chỉ có 225 mẫu, nên $k = 101$ chiếm tới gần **$45\%$ toàn bộ dữ liệu**.
   - Bán kính lân cận buộc phải mở rộng rất lớn để gom đủ 101 điểm, lấn sâu sang cả vầng trăng đối diện. Số lượng áp đảo của các điểm ở xa đã "pha loãng" và dập tắt tiếng nói của một vài điểm cực gần. Biên quyết định bị duỗi thẳng đơ, san phẳng mọi độ cong tự nhiên của dữ liệu (mô hình bị underfitting / cẩu thả).
2. **Khi dùng `weights="distance"` (trọng số nghịch đảo khoảng cách)**:
   - Trọng số lá phiếu được tính theo $w_i = \frac{1}{d(x, x_i)}$.
   - Các điểm láng giềng ở cự ly xa có khoảng cách $d$ rất lớn $\implies$ trọng số $w_i \to 0$, dù có hàng chục điểm xa thì tổng phiếu của chúng vẫn không đáng kể.
   - Ngược lại, những điểm ở cự ly sát cạnh mẫu mới có khoảng cách $d$ cực nhỏ $\implies$ trọng số $w_i$ cực lớn, chi phối hoàn toàn quyết định bỏ phiếu.
   - Nhờ cơ chế này, cấu trúc cục bộ tại vùng lân cận gần được bảo toàn trọn vẹn, ngăn chặn tình trạng cẩu thả khi người dùng chọn $k$ quá lớn.
3. **Lưu ý về điểm số trên tập học**:
   - Khi dùng `weights="distance"`, Accuracy trên tập học luôn đạt $1.0000$ vì với mỗi mẫu trong tập học, khoảng cách tới chính nó bằng $0$, dẫn tới trọng số $w \to \infty$. Con số $1.0000$ này chỉ là hệ quả toán học, không phản ánh nguy cơ học vẹt như khi $k=1$.

---

### Bài tập 5: Chọn k cho bài toán hồi quy (KNeighborsRegressor)

#### 1. Yêu cầu đề bài
- Trên dữ liệu giá nhà (`gia_nha.csv`), dùng kiểm định chéo 5 phần (5-fold CV) trên tập học để chọn $k$ tốt nhất cho `KNeighborsRegressor`, với $k$ chạy từ 1 tới 20.
- Gợi ý: truyền `scoring="neg_mean_absolute_error"` vào `cross_val_score`, và nhớ rằng điểm trả về mang dấu âm.
- Với $k$ tốt nhất, MAE trên tập kiểm tra có vượt được đường thẳng không?

#### 2. Kết quả kiểm định chéo 5 phần tìm $k$ trên tập học (48 căn)

| $k$ | Điểm CV (`neg_mean_absolute_error`) | MAE trung bình (tỷ đồng) | Đánh giá |
|---|---|---|---|
| $k = 1$ | -0.4164 | 0.4164 | |
| **$k = 2$** | **-0.3498** | **0.3498** | **TỐT NHẤT (MAE thấp nhất)** |
| $k = 3$ | -0.3714 | 0.3714 | |
| $k = 4$ | -0.3618 | 0.3618 | |
| $k = 5$ | -0.3816 | 0.3816 | (Mặc định ở mục 6) |
| $k = 6$ | -0.4002 | 0.4002 | |
| $k = 7$ | -0.3990 | 0.3990 | |
| $k = 10$ | -0.4365 | 0.4365 | |
| $k = 15$ | -0.5290 | 0.5290 | |
| $k = 20$ | -0.6254 | 0.6254 | Sai số tăng dần |

- Giá trị $k$ tối ưu được chọn là **$k = 2$** với MAE kiểm định chéo nhỏ nhất là **$0.3498$ tỷ đồng**.

#### 3. Bảng so sánh MAE trên tập kiểm tra (12 căn)

| Mô hình | Tham số | MAE trên tập kiểm tra | Nhận xét so với đường thẳng |
|---|---|---|---|
| **KNN Regressor tối ưu** | **$k = 2$** | **`0.4867` tỷ đồng** | Kém hơn $+0.2289$ tỷ |
| **KNN Regressor bài học** | **$k = 5$** | **`0.4030` tỷ đồng** | Kém hơn $+0.1452$ tỷ |
| **Hồi quy tuyến tính (Bài 1)** | **Đường thẳng** | **`0.2578` tỷ đồng** | **THẮNG ÁP ĐẢO (Sai số thấp nhất)** |

#### 4. Trả lời câu hỏi & Giải thích nguyên nhân
> **TRẢ LỜI: KHÔNG VƯỢT ĐƯỢC.**  
> Với $k = 2$ tốt nhất, MAE trên tập kiểm tra của KNN là **$0.4867$ tỷ đồng**, vẫn cao hơn nhiều so với mức sai số **$0.2578$ tỷ đồng** của mô hình Hồi quy tuyến tính (Linear Regression).

**Giải thích nguyên nhân bản chất**:
1. **Quan hệ thực tế trong dữ liệu là tuyến tính**:
   - Trên tập dữ liệu `gia_nha.csv`, giá nhà tăng theo tỷ lệ rất thẳng và đều đặn theo diện tích ($y \approx a \cdot x + b$).
   - Mô hình tham số Hồi quy tuyến tính đã nắm bắt đúng bản chất toàn cục này, tận dụng triệt để toàn bộ 48 điểm để khớp một đường thẳng mượt mà với sai số cực nhỏ.
2. **Hạn chế xấp xỉ bậc thang của KNN hồi quy**:
   - KNN hồi quy dự đoán bằng cách lấy trung bình cộng của $k$ giá nhà gần nhất. Đồ thị dự đoán của nó không phải là một đường dốc liên tục mà là các **bậc thang giật cục (piecewise constant)**.
   - Dù ở cự ly gần, các bậc thang này vẫn tạo ra sai lệch so với độ dốc liên tục thực tế của giá nhà, khiến MAE bị đội lên.
3. **Hiện tượng không thể ngoại suy (Extrapolation Failure)**:
   - Với những căn nhà có diện tích lớn hơn hoặc nhỏ hơn vùng dữ liệu huấn luyện, KNN hoàn toàn bất lực trong việc suy diễn xu hướng đi lên, nó chỉ gán giá trị trung bình của các căn biên (ví dụ căn $150m^2$ hay $250m^2$ đều bị đoán cụt ở mức $9.32$ tỷ). Trong khi đó, đường thẳng có khả năng ngoại suy chuẩn xác theo độ dốc.
4. **Bài học rút ra**: Không có thuật toán nào là vạn năng (*No Free Lunch Theorem*). Trên dữ liệu có biên uốn lượn phức tạp (hai vầng trăng), KNN áp đảo đường thẳng; nhưng trên dữ liệu có bản chất tuyến tính rõ ràng (giá nhà), mô hình hồi quy tuyến tính luôn là lựa chọn vượt trội.

---

### Bài tập 6: Đo thời gian dự đoán (Mô hình học lười)

#### 1. Yêu cầu đề bài
- Tạo bộ dữ liệu ngẫu nhiên có $1,000$; $10,000$ và $100,000$ mẫu với $2$ đặc trưng bằng `np.random.default_rng(0).random`.
- Với mỗi cỡ dữ liệu, đo thời gian `.fit` và thời gian `.predict` cho $1,000$ mẫu mới bằng mô đun `time`.
- Lập bảng kết quả rồi giải thích vì sao KNN được gọi là mô hình học lười.

#### 2. Bảng kết quả đo thời gian thực nghiệm

| Cỡ tập học ($N$) | Thời gian `.fit` (giây) | Thời gian `.predict` cho 1,000 mẫu mới (giây) | Tỷ số giữa Predict và Fit |
|---|---|---|---|
| **$1,000$ mẫu** | **`0.000867` s** | **`0.016276` s** | Predict chậm hơn Fit **~18.8 lần** |
| **$10,000$ mẫu** | **`0.003373` s** | **`0.015542` s** | Predict chậm hơn Fit **~4.6 lần** |
| **$100,000$ mẫu** | **`0.037222` s** | **`0.016212` s** | Fit tăng theo kích thước dữ liệu |

*(Thời gian đo bằng `time.perf_counter()` trên hệ thống Windows PowerShell với bộ dữ liệu ngẫu nhiên).*

#### 3. Giải thích chuyên sâu: Vì sao KNN là mô hình học lười (Lazy Learning)?
1. **Pha huấn luyện (`.fit`) "lười biếng"**:
   - Ở các mô hình học chủ động (**Eager Learning**) như Logistic Regression, Support Vector Machine hay Deep Learning: Quá trình `.fit` đòi hỏi hàng trăm vòng lặp tối ưu Gradient Descent, tính đạo hàm ma trận để tìm ra bộ trọng số tối ưu. Giai đoạn này tiêu tốn phần lớn thời gian và tài nguyên CPU/GPU.
   - Ngược lại, **KNN không học bất kỳ tham số nào**. Khi gọi `.fit(X_train, y_train)`, KNN chỉ thực hiện một việc duy nhất: nạp dữ liệu vào bộ nhớ RAM (và tổ chức cấu trúc chỉ mục tìm kiếm như KD-Tree hoặc Ball-Tree). Do đó thời gian `.fit` diễn ra gần như tức thì trong tích tắc (chưa tới 1 mili-giây cho 1,000 mẫu).
2. **Pha dự đoán (`.predict`) "gánh nợ tính toán"**:
   - Khi có dữ liệu mới cần dự đoán, KNN mới bắt đầu dồn toàn lực để tính toán: nó phải tính khoảng cách giữa từng mẫu mới tới tất cả các mẫu cũ trong kho dữ liệu, sắp xếp khoảng cách và trích xuất $k$ láng giềng.
   - Độ phức tạp tính toán dự đoán cho $M$ mẫu mới trên $N$ mẫu huấn luyện với $D$ chiều là $\mathcal{O}(M \cdot N \cdot D)$ (với brute-force) hoặc $\mathcal{O}(M \cdot D \log N)$ (với cây tìm kiếm).
   - Trong khi đó, ở các mô hình tham số đã học xong, dự đoán 1 mẫu chỉ là một phép nhân vector cực nhanh $\theta^T x$ với độ phức tạp $\mathcal{O}(D)$, hoàn toàn độc lập với kích thước dữ liệu $N$.
3. **Ưu và nhược điểm của mô hình học lười**:
   - **Ưu điểm**: Huấn luyện tức thì, thêm dữ liệu mới cực kỳ đơn giản (chỉ việc thêm dòng vào bảng mà không cần train lại từ đầu).
   - **Nhược điểm**: Dự đoán chậm, tốn nhiều bộ nhớ RAM để lưu trữ toàn bộ tập dữ liệu, và bị ảnh hưởng nặng nề bởi "lời nguyền số chiều" khi số đặc trưng $D$ tăng cao.

---

## 🖼️ Ảnh chụp màn hình kết quả chạy thực tế

Toàn bộ các ảnh chụp màn hình terminal PowerShell chuẩn đẹp đã được tự động tạo và lưu trữ đầy đủ trong thư mục `baitap04/`:

| Bài tập | File ảnh kết quả | Nội dung kiểm chứng |
|---|---|---|
| **Bài 1** | `baitap04/screenshot_bai1.png` | Khớp 100% giữa KNN tự viết và scikit-learn (Acc = 0.9200) |
| **Bài 2** | `baitap04/screenshot_bai2.png` | Đối chiếu k tốt nhất và Accuracy giữa Euclid và Manhattan |
| **Bài 3** | `baitap04/screenshot_bai3.png` | Hiệu ứng đổi nghìn đồng sang triệu đồng & tính bất biến của StandardScaler |
| **Bài 4** | `baitap04/screenshot_bai4.png` | Bảng so sánh $k=1, 13, 101$ với weights='distance' và giải thích |
| **Bài 5** | `baitap04/screenshot_bai5.png` | Bảng CV chọn $k=2$ và so sánh MAE với đường thẳng |
| **Bài 6** | `baitap04/screenshot_bai6.png` | Bảng thời gian .fit và .predict chứng minh bản chất Lazy Learning |

---

## 💡 Phụ lục: Các lỗi thường gặp và cách xử lý khi dùng KNN

Trích xuất từ Phụ lục B tài liệu thực hành để đối chiếu và phòng tránh lỗi:

1. **Accuracy thấp hơn cả đoán bừa**:
   - *Nguyên nhân*: Quên chuẩn hoá dữ liệu khi các đặc trưng có thang đo chênh lệch lớn (cột thang đo lớn nuốt trọn khoảng cách).
   - *Cách xử lý*: Luôn dùng `make_pipeline(StandardScaler(), KNeighborsClassifier(...))`.
2. **Rò rỉ dữ liệu (Data Leakage)**:
   - *Nguyên nhân*: Chuẩn hoá toàn bộ dữ liệu trước khi gọi `train_test_split`.
   - *Cách xử lý*: Chỉ fit scaler trên tập train rồi transform sang tập test, hoặc dùng `make_pipeline`.
3. **Accuracy trên tập học bằng 1.0**:
   - *Nguyên nhân*: Đang dùng $k = 1$ hoặc dùng `weights="distance"`. Mẫu tự tính khoảng cách tới chính nó bằng 0 nên luôn đoán đúng.
   - *Cách xử lý*: Tuyệt đối không đánh giá chất lượng mô hình trên tập học; luôn đo trên tập kiểm tra hoặc dùng Cross-Validation.
4. **Hòa phiếu khi phân loại nhị phân**:
   - *Nguyên nhân*: Đặt $k$ chẵn dẫn đến số phiếu ngang nhau giữa hai lớp.
   - *Cách xử lý*: Luôn chọn $k$ lẻ cho bài toán phân loại 2 lớp.
