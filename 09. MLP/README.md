# BÀI THỰC HÀNH 9: MẠNG NƠ-RON NHIỀU LỚP (MLP)
**Môn học:** Học máy ứng dụng (Machine Learning)  
**Giảng viên:** ThS. Nguyễn Thái Anh  
**Đơn vị:** Khoa Công nghệ Thông tin - Trường Đại học Văn Lang  

---

## Cấu trúc thư mục

```text
09. MLP/
├── data/
│   ├── gia_nha.csv                  # Dữ liệu 60 căn hộ (dien_tich, so_phong, tuoi_nha, gia)
│   ├── hai_vang_trang.csv           # Dữ liệu 300 điểm hai vầng trăng (x1, x2, nhan)
│   ├── sinh_vien_mo_rong.csv        # Dữ liệu sinh viên có cột chi_tieu hàng nghìn đồng
│   └── xor.csv                      # Dữ liệu 400 điểm bài toán XOR (x1, x2, nhan)
├── code/                            # 5 tệp mã nguồn hướng dẫn trong tài liệu
│   ├── m1_lan_truyen_tien.py        # Bước 1: Tự tính lan truyền tiến bằng numpy
│   ├── m2_xor.py                    # Bước 2: Một nơ-ron không giải được XOR, một lớp ẩn thì được
│   ├── m3_do_lon_mang.py            # Bước 3: Độ lớn mạng và điều chuẩn alpha bằng CV
│   ├── m4_chuan_hoa.py              # Bước 4: Mạng nơ-ron khi quên chuẩn hoá
│   └── m5_chu_so.py                 # Bước 5: Nhận dạng chữ số viết tay bằng MLPClassifier
├── baitap09/                        # Thư mục 6 bài tập tự thực hành
│   ├── bai1.py                      # Bài 1: Tự dự đoán bằng trọng số đã học (numpy vs predict_proba)
│   ├── bai2.py                      # Bài 2: Kiểm chứng & giải thích mạng XOR |x1 - x2|
│   ├── bai3.py                      # Bài 3: So sánh activation (logistic, tanh, relu)
│   ├── bai4.py                      # Bài 4: Ảnh hưởng của learning_rate_init & biểu đồ loss_curve_
│   ├── bai5.py                      # Bài 5: Dò lưới GridSearchCV cho MLP trên bộ chữ số
│   ├── bai6.py                      # Bài 6: MLPRegressor trên dữ liệu giá nhà & khả năng ngoại suy
│   ├── tao_anh_chup.py              # Script tự động chạy và chụp màn hình terminal
│   ├── toc_do_hoc.png               # Biểu đồ 4 đường loss_curve_ theo learning_rate_init
│   └── screenshot_bai1..6.png       # Ảnh chụp màn hình kết quả chạy 6 bài tập
├── toc_do_hoc.png                   # Bản sao biểu đồ tại thư mục gốc
├── Lab09_MLP.pdf                    # Tài liệu thực hành & đề bài
└── README.md                        # Báo cáo tổng hợp kết quả
```

---

## Tóm tắt kết quả 6 bài tập

### Bài tập 1: Tự dự đoán bằng trọng số đã học (`baitap09/bai1.py`)
- Huấn luyện lại `MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs", max_iter=5000, random_state=0)` trên `data/xor.csv` (`test_size=0.25, random_state=17, stratify=y`), đạt accuracy toàn tập kiểm tra `0.9100`.
- Trích xuất `coefs_` (`(2, 8)`, `(8, 1)`) và `intercepts_` (`(8,)`, `(1,)`), tự tính lan truyền tiến bằng `numpy` (`ReLU` ở lớp ẩn, `sigmoid` ở lớp ra) cho 5 điểm đầu của tập kiểm tra và đối chiếu với `mlp.predict_proba(...)[:, 1]`:
  - Điểm `1` (`0.3319, -0.5595`, nhãn `0`): tự tính `0.017557985373` vs `0.017557985373` (sai số `0.000e+00`)
  - Điểm `2` (`-0.7016, -0.8251`, nhãn `1`): `0.999434831959` vs `0.999434831959` (sai số `0.000e+00`)
  - Điểm `3` (`0.5155, -0.0052`, nhãn `0`): `0.794331056599` vs `0.794331056599` (sai số `0.000e+00`)
  - Điểm `4` (`-0.6036, 0.2726`, nhãn `0`): `0.134400889648` vs `0.134400889648` (sai số `2.776e-17`)
  - Điểm `5` (`0.2748, 0.3515`, nhãn `1`): `0.957805718116` vs `0.957805718116` (sai số `0.000e+00`)
- **Kết luận:** Hai kết quả khớp hoàn toàn (sai lệch tối đa `2.776e-17` do làm tròn `float64`).

### Bài tập 2: Một mạng XOR khác (`baitap09/bai2.py`)
- Kiểm chứng bằng `numpy` mạng `h1 = max(0, x1 - x2)`, `h2 = max(0, x2 - x1)`, `y = h1 + h2` trên 4 đầu vào `(0, 0), (0, 1), (1, 0), (1, 1)` cho ra đúng `[0., 1., 1., 0.]` (`4/4`).
- **Giải thích:**
  - Về đại số: `h1 + h2 = max(0, x1 - x2) + max(0, x2 - x1) = |x1 - x2|`. Với hai bit nhị phân `{0, 1}`, giá trị tuyệt đối hiệu bằng `0` khi `x1 = x2` và bằng `1` khi `x1 != x2` — đúng là phép XOR.
  - Về hình học: đường `x1 = x2` chứa cả `(0, 0)` và `(1, 1)` (nơi cả `h1 = h2 = 0`); `h1` chỉ bật ở nửa mặt phẳng `x1 > x2` chứa `(1, 0)`, còn `h2` chỉ bật ở nửa mặt phẳng đối diện `x2 > x1` chứa `(0, 1)`. Nhờ ReLU bẻ gập mặt phẳng thành hình chữ V tại `x1 = x2`, mạng phân tách được XOR.

### Bài tập 3: So các hàm kích hoạt (`baitap09/bai3.py`)
| `activation` | Accuracy tập học | Accuracy tập kiểm tra | Kiểm định chéo 5-fold |
| :--- | :---: | :---: | :---: |
| `"logistic"` | `0.8400` | `0.8533` | `0.8356` |
| `"tanh"` | `0.8711` | `0.9067` | `0.8444` |
| `"relu"` | **`0.8756`** | **`0.9067`** | **`0.8578`** |

- **Kết luận:** `"relu"` và `"tanh"` cùng đạt accuracy cao nhất trên tập kiểm tra (`0.9067`), vượt xa `"logistic"` (`0.8533`). Xét tổng thể cả tập học (`0.8756`) và kiểm định chéo 5-fold (`0.8578`), **`"relu"`** cho kết quả tốt nhất ở đây.

### Bài tập 4: Tốc độ học (`baitap09/bai4.py`)
| `learning_rate_init` | Số vòng (`n_iter_`) | Mất mát cuối | Accuracy kiểm tra |
| :---: | :---: | :---: | :---: |
| `0.0001` | `582` | `0.018928` | `0.9889` |
| `0.001` | `144` | `0.005210` | `0.9844` |
| `0.01` | `38` | `0.001077` | `0.9778` |
| `0.1` | `30` | `0.120373` | `0.9511` |

- Biểu đồ 4 đường `loss_curve_` lưu tại [`baitap09/toc_do_hoc.png`](baitap09/toc_do_hoc.png).
- **Nhận xét:** Tốc độ học quá nhỏ (`0.0001`) làm mạng cập nhật rất chậm, tốn tới `582` vòng mới hội tụ. Tốc độ học quá lớn (`0.1`) làm bước nhảy vọt qua đáy cực tiểu, đường mất mát rung lắc mạnh và dừng sớm ở mức mất mát cao (`0.1204` sau `30` vòng), khiến accuracy tụt xuống `0.9511`.

### Bài tập 5: Dò lưới cho mạng nơ-ron (`baitap09/bai5.py`)
- Dùng `GridSearchCV` (5-fold) trên `load_digits` với `hidden_layer_sizes` trong `(10,), (50,), (100,), (50, 50)` và `alpha` trong `0.0001, 0.01, 1`:
  - **Cặp tham số tốt nhất:** `hidden_layer_sizes=(50,)`, `alpha=1` (điểm CV 5-fold `0.9785`).
  - **Accuracy trên tập kiểm tra:** MLP tốt nhất đạt **`0.9822`** (`442/450` ảnh), **không vượt được** `SVC()` kernel RBF (**`0.9911`**, `446/450` ảnh).

### Bài tập 6: Mạng nơ-ron cho hồi quy (`baitap09/bai6.py`)
- Trên `data/gia_nha.csv` (`X = [["dien_tich"]]`, `test_size=0.2, random_state=42`):
  - **Hồi quy tuyến tính (`LinearRegression`):** `MAE = 0.2578` tỷ đồng (`R2 = 0.9622`), dự đoán căn `250 m2` là **`19.9139` tỷ đồng**.
  - **`MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0)` có `StandardScaler`:** `MAE = 0.5008` tỷ đồng (`R2 = 0.8557`), dự đoán căn `250 m2` là **`28.1758` tỷ đồng** (lệch `+8.26` tỷ đồng).
- **Kết luận:** Mạng nơ-ron **không ngoại suy tốt như đường thẳng**. Ngoài biên miền dữ liệu học `[35.5, 117.5] m2`, các nơ-ron ReLU chỉ nối tiếp các đoạn thẳng ở vùng rìa nên dễ bẻ lệch độ dốc và dự đoán sai rất mạnh.
