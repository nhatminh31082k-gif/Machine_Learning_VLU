# BÀI THỰC HÀNH 8: PHÂN CỤM K-MEANS
**Môn học:** Học máy ứng dụng (Machine Learning)  
**Giảng viên:** ThS. Nguyễn Thái Anh  
**Đơn vị:** Khoa Công nghệ Thông tin - Trường Đại học Văn Lang  

---

## Cấu trúc thư mục

```text
08. K-Means/
├── data/
│   ├── khach_hang.csv               # Dữ liệu 210 khách hàng (tuoi, thu_nhap, diem_chi_tieu)
│   └── hai_vang_trang.csv           # Dữ liệu 300 điểm hai vầng trăng (x1, x2, nhan)
├── code/                            # 5 tệp mã nguồn hướng dẫn trong tài liệu
│   ├── k1_tu_viet.py                # Bước 1: Tự viết K-Means bằng numpy trên 7 điểm
│   ├── k2_khach_hang.py             # Bước 2: Phân khúc khách hàng bằng K-Means
│   ├── k3_chon_k.py                 # Bước 3: Chọn số cụm K bằng khuỷu tay và silhouette
│   ├── k4_ba_cai_bay.py             # Bước 4: Ba cái bẫy (quên chuẩn hoá, khởi tạo xấu, cụm không tròn)
│   └── k5_nen_anh.py                # Bước 5: Nén màu ảnh china.jpg bằng K-Means
├── baitap08/                        # Thư mục 6 bài tập tự thực hành
│   ├── bai1.py                      # Bài 1: Tự viết hàm silhouette_mot_diem bằng numpy
│   ├── bai2.py                      # Bài 2: Phân cụm trên cả 3 cột (tuoi, thu_nhap, diem_chi_tieu)
│   ├── bai3.py                      # Bài 3: So sánh init="random" và init="k-means++" (50 lần)
│   ├── bai4.py                      # Bài 4: Lưu ảnh nén 16 màu và xử lý PNG 4 kênh RGBA
│   ├── bai5.py                      # Bài 5: Thử DBSCAN trên dữ liệu hai vầng trăng
│   ├── bai6.py                      # Bài 6: Phân cụm chữ số viết tay load_digits (K=10)
│   ├── tao_anh_chup.py              # Script tự động chạy và chụp màn hình terminal
│   ├── anh_goc.png                  # Ảnh china.jpg gốc lưu dạng PNG
│   ├── anh_16_mau.png               # Ảnh china.jpg nén còn 16 màu
│   ├── anh_cua_em.png               # Ảnh cá nhân 4 kênh RGBA PNG
│   ├── anh_cua_em_16_mau.png        # Ảnh cá nhân sau khi bỏ kênh 4 và nén còn 16 màu
│   └── screenshot_bai1..6.png       # Ảnh chụp màn hình kết quả chạy 6 bài tập
├── Lab08_KMeans.pdf                 # Tài liệu thực hành & đề bài
└── README.md                        # Báo cáo tổng hợp kết quả
```

---

## Tóm tắt kết quả 6 bài tập

### Bài tập 1: Tự tính silhouette (`baitap08/bai1.py`)
- Hàm `silhouette_mot_diem(X, nhan, i)` tự tính `a`, `b`, `s` chỉ bằng `numpy` cho toàn bộ `210` khách hàng (`K = 5`):
  - **Silhouette trung bình tự tính:** `0.722706` — khớp hoàn toàn với `silhouette_score` của `scikit-learn` (`0.722706`, sai lệch lớn nhất từng điểm `9.99e-16`).
- **Khách hàng có `s` nhỏ nhất:** khách hàng thứ `113` (dòng dữ liệu thứ `114`: `tuoi = 38`, `thu_nhap = 25.4` triệu, `diem_chi_tieu = 41`), thuộc cụm `2` (thu nhập vừa, chi vừa), có `a = 0.9178`, `b = 0.9448`, `s = 0.0286`.
- **Vị trí trên Hình 3:** Trên Hình 3 (cụm giữa màu xanh ngọc được đánh số là Cụm 3), khách này nằm lệch hẳn sang rìa bên phải – phía dưới của cụm giữa, trôi ra khoảng trống hướng về phía cụm thu nhập cao, chi ít (Cụm 4 màu cam, tâm `(30.1, 20.0)`), nên khoảng cách tới cụm mình (`a`) gần bằng khoảng cách tới cụm hàng xóm (`b`), làm `s` rất gần `0`.

### Bài tập 2: Thêm cột tuổi (`baitap08/bai2.py`)
- Chuẩn hoá cả 3 cột `tuoi`, `thu_nhap`, `diem_chi_tieu` và quét `K` từ `2` tới `10`:
  - **`K` tốt nhất vẫn là `5`** với `silhouette = 0.4808` (giảm so với `0.723` khi chỉ dùng 2 cột do thêm chiều `tuoi` làm các điểm trong cụm phân tán rộng hơn trong không gian 3D).
- **So sánh tâm cụm với mục 3:**
  - Hai nhóm chi ít giữ nguyên hoàn toàn: `(8.0, 18.9)` (`40` khách, tuổi `44.9`) và `(30.1, 20.0)` (`35` khách, tuổi `42.9`).
  - Nhóm thu nhập thấp, chi nhiều nhận thêm `4` khách trẻ (`35 -> 39` khách), tâm dịch từ `(7.5, 80.2)` thành `(8.5, 77.6)`, tuổi trung bình `25.1`.
  - Nhóm thu nhập vừa, chi vừa giảm từ `60` xuống `57` khách, tâm `(17.6, 49.2)`, tuổi trung bình tăng từ `40.4` lên `41.6`.
  - Nhóm thu nhập cao, chi nhiều giảm nhẹ `1` khách (`40 -> 39`), tâm gần như giữ nguyên `(30.3, 82.0)`, tuổi trung bình `32.1`.

### Bài tập 3: K-Means++ và khởi tạo ngẫu nhiên (`baitap08/bai3.py`)
- Chạy `50` lần với `n_init=1`, `random_state` từ `0` tới `49` trên dữ liệu khách hàng đã chuẩn hoá (`K = 5`):
  - **`init="random"`:** `30/50` lần (`60%`) đạt inertia `23.0`; `20/50` lần (`40%`) mắc kẹt ở cực tiểu cục bộ xấu (inertia từ `72.4` lên tới `204.05`).
  - **`init="k-means++"`:** `50/50` lần (`100%`) đạt đúng inertia `23.02`, không có lần nào rơi vào nghiệm xấu.
- **Kết luận:** `k-means++` đáng tin hơn hẳn vì chọn các tâm ban đầu trải xa nhau theo xác suất tỷ lệ thuận với bình phương khoảng cách tới tâm gần nhất, giúp tránh tình trạng nhiều tâm rơi cùng vào một cụm dẫn đến cắt đôi cụm này và gộp chung cụm khác.

### Bài tập 4: Lưu ảnh đã nén (`baitap08/bai4.py`)
- **Ảnh mẫu `china.jpg` (`427 x 640 x 3`):**
  - Ảnh gốc PNG (`baitap08/anh_goc.png`): `537.3 KB`.
  - Ảnh 16 màu PNG (`baitap08/anh_16_mau.png`): `131.2 KB` (nhỏ hơn **`4.09` lần**, giảm `75.6%`).
- **Ảnh cá nhân RGBA PNG (`baitap08/anh_cua_em.png`):**
  - Đọc bằng `plt.imread` có shape `(427, 640, 4)` (`4` kênh RGBA); bỏ kênh alpha thứ tư còn `(427, 640, 3)` trước khi phân cụm `16` màu.
  - Dung lượng giảm từ `409.5 KB` xuống `54.3 KB` (`baitap08/anh_cua_em_16_mau.png`, nhỏ hơn **`7.55` lần**, giảm `86.7%`).

### Bài tập 5: DBSCAN trên hai vầng trăng (`baitap08/bai5.py`)
| `eps` | Số cụm tìm được | Số điểm nhiễu (`-1`) | Tỷ lệ khớp nhãn thật |
| :---: | :---: | :---: | :---: |
| `0.2` | `1` | `40` | `0.4367` (gộp 1 cụm) |
| `0.3` | `1` | `10` | `0.4933` (gộp 1 cụm) |
| `0.4` | `1` | `3` | `0.4967` (gộp 1 cụm) |

- **Kết luận:** Không có giá trị nào trong `{0.2, 0.3, 0.4}` tách được hai vầng trăng tốt hơn K-Means (`0.7500`). Do dữ liệu `hai_vang_trang.csv` có độ nhiễu cao, khi `eps >= 0.2` (với `min_samples=5`), bán kính lân cận đủ lớn để bắc cầu qua phần rìa giữa hai vầng trăng, gộp cả hai thành một cụm duy nhất. Tăng `eps` làm giảm số điểm nhiễu nhưng càng dễ dính cụm; ngược lại giảm `eps` quá nhỏ sẽ làm cụm vỡ vụn thành nhiều cụm nhỏ và nhiều nhiễu.

### Bài tập 6: Phân cụm chữ số viết tay (`baitap08/bai6.py`)
- Chạy `KMeans(n_clusters=10, n_init=10, random_state=0)` trên `load_digits` (`1797` ảnh, `64` chiều) rồi lập bảng chéo `pd.crosstab(nhan_that, cum)`:
  - **Chữ số được gom gọn nhất vào một cụm riêng:** số **`0`** (`176/178 = 98.9%` ở cụm `0`), số **`6`** (`177/181 = 97.8%` ở cụm `5`), và số **`7`** (`174/179 = 97.2%` ở cụm `2`); ngoài ra còn có **`4`** (`91.2%`), **`3`** (`84.2%`), **`2`** (`83.6%`).
  - **Chữ số hay bị gom chung:** số **`1`** và số **`8`** bị gom chung nhiều nhất ở cụm `1` (`100` mẫu số `1` và `100` mẫu số `8`); đồng thời ở cụm `8` (`139` mẫu số `9`) bị lẫn thêm `41` mẫu số `5` và `52` mẫu số `8`.
