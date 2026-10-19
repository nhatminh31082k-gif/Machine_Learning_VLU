# Báo Cáo & Lời Giải Bài Tập Thực Hành 3: Naive Bayes

Tài liệu: `Lab03_Naive_Bayes (1).pdf`  
Môn học: **Học máy ứng dụng** - Khoa Công nghệ Thông tin, Trường Đại học Văn Lang  
Giảng viên: **ThS. Nguyễn Thái Anh**

---

## Cấu trúc thư mục

```text
03. Naive_Bayes/
├── data/
│   ├── sinh_vien.csv
│   └── tin_nhan.csv
├── code/
│   ├── d1_bayes_tay.py
│   ├── d2_tui_tu.py
│   ├── d3_loc_thu_rac.py
│   ├── d4_lam_tron.py
│   └── d5_gaussian.py
└── baitap03/
    ├── bai1.py
    ├── bai2.py
    ├── bai3.py
    ├── bai4.py
    ├── bai5.py
    ├── bai6.py
    └── README.md
```

Cách thực thi từ thư mục gốc `03. Naive_Bayes`:
```bash
python baitap03/bai1.py
python baitap03/bai2.py
python baitap03/bai3.py
python baitap03/bai4.py
python baitap03/bai5.py
python baitap03/bai6.py
```

---

## Bài tập 1: Định lý Bayes với ba nguyên nhân

### 1. Đề bài
Một lớp có ba nhóm sinh viên:
- 50% học đều, tỷ lệ qua môn: 0.95
- 30% học dồn, tỷ lệ qua môn: 0.70
- 20% ít học, tỷ lệ qua môn: 0.30

Gặp một bạn đã qua môn. Tính xác suất bạn đó thuộc từng nhóm. Ba con số cộng lại phải bằng 1 để tự kiểm tra.

### 2. Phương pháp & Công thức
Áp dụng công thức Bayes tổng quát cho hệ đầy đủ gồm 3 biến cố $A_1, A_2, A_3$:
$$P(A_i \mid B) = \frac{P(B \mid A_i) \cdot P(A_i)}{\sum_{j=1}^{3} P(B \mid A_j) \cdot P(A_j)}$$

Trong đó:
- $B$: biến cố sinh viên qua môn.
- $A_1$: sinh viên thuộc nhóm Học đều ($P(A_1) = 0.50$, $P(B \mid A_1) = 0.95$).
- $A_2$: sinh viên thuộc nhóm Học dồn ($P(A_2) = 0.30$, $P(B \mid A_2) = 0.70$).
- $A_3$: sinh viên thuộc nhóm Ít học ($P(A_3) = 0.20$, $P(B \mid A_3) = 0.30$).

Mẫu số (xác suất toàn phần qua môn):
$$P(B) = 0.50 \times 0.95 + 0.30 \times 0.70 + 0.20 \times 0.30 = 0.475 + 0.210 + 0.060 = 0.745$$

Xác suất hậu nghiệm:
- $P(\text{Học đều} \mid \text{Qua môn}) = \frac{0.475}{0.745} \approx \mathbf{0.6376}$ (63.76%)
- $P(\text{Học dồn} \mid \text{Qua môn}) = \frac{0.210}{0.745} \approx \mathbf{0.2819}$ (28.19%)
- $P(\text{Ít học} \mid \text{Qua môn}) = \frac{0.060}{0.745} \approx \mathbf{0.0805}$ (8.05%)

Tổng xác suất hậu nghiệm:
$$0.63758389 + 0.28187919 + 0.08053691 = \mathbf{1.000000}$$

---

## Bài tập 2: Từ nào tố cáo mạnh nhất

### 1. Đề bài
Huấn luyện `MultinomialNB` trên toàn bộ 300 tin nhắn. Dùng thuộc tính `feature_log_prob_` để tính hiệu giữa hai lớp cho từng từ, in ra 5 từ tố cáo tin rác mạnh nhất và 5 từ cho thấy tin thường mạnh nhất.

### 2. Kết quả thực nghiệm
- Hiệu log-prob: $\Delta = \log P(\text{từ} \mid \text{rác}) - \log P(\text{từ} \mid \text{thường})$

**Top 5 từ tố cáo tin rác mạnh nhất (hiệu dương lớn nhất):**
| STT | Từ | log P(từ \| rác) | log P(từ \| thường) | Hiệu log |
|---|---|---|---|---|
| 1 | **mãi** | -3.4032 | -6.6067 | **+3.2034** |
| 2 | **tay** | -3.0079 | -5.9135 | **+2.9056** |
| 3 | **ngay** | -3.0285 | -5.6904 | **+2.6618** |
| 4 | **miễn** | -3.2108 | -5.6904 | **+2.4795** |
| 5 | **trúng**| -3.3162 | -5.6904 | **+2.3742** |

**Top 5 từ cho thấy tin thường mạnh nhất (hiệu âm sâu nhất):**
| STT | Từ | log P(từ \| rác) | log P(từ \| thường) | Hiệu log |
|---|---|---|---|---|
| 1 | **lớp** | -5.8011 | -3.1889 | **-2.6122** |
| 2 | **em** | -5.8011 | -3.2394 | **-2.5618** |
| 3 | **bài** | -5.2903 | -2.9957 | **-2.2946** |
| 4 | **báo** | -5.5134 | -3.2394 | **-2.2741** |
| 5 | **phòng** | -5.5134 | -3.2394 | **-2.2741** |

---

## Bài tập 3: Đổi tham số làm trơn Laplace (alpha)

### 1. Đề bài
Chạy lại bộ lọc với `alpha` lần lượt bằng `0.001`, `0.1`, `1.0`, `10.0`, `100.0`. In ra Accuracy, Precision, Recall trên tập kiểm tra. Nhận xét chuyện gì xảy ra khi alpha quá lớn và giải thích nguyên nhân.

### 2. Kết quả thực nghiệm
| alpha | Accuracy | Precision | Recall |
|---|---|---|---|
| 0.001 | 0.9467 | 1.0000 | 0.8667 |
| 0.100 | 0.9467 | 1.0000 | 0.8667 |
| 1.000 | 0.9467 | 1.0000 | 0.8667 |
| 10.000 | 0.9467 | 1.0000 | 0.8667 |
| 100.000 | 0.9467 | 1.0000 | 0.8667 |
| *1000.0* | 0.6000 | 0.0000 | 0.0000 |

### 3. Nhận xét & Giải thích
1. **Hiện tượng khi alpha quá lớn:**
   - Ở mức $\alpha$ từ $0.001$ đến $100.0$, trên tập dữ liệu này các nhãn dự đoán vẫn ổn định.
   - Nhưng khi $\alpha$ tăng lên cực lớn ($\alpha \ge 1000$), mô hình bị thoái hóa: Precision và Recall của lớp rác sụt về $0$, mô hình đoán toàn bộ tin nhắn thành tin thường (lớp chiếm đa số trong tập huấn luyện).
2. **Nguyên nhân toán học:**
   - Công thức làm trơn Laplace:
     $$P(\text{từ}_j \mid \text{lớp}_k) = \frac{\text{đếm}_{jk} + \alpha}{\text{tổng}_k + \alpha V}$$
   - Khi $\alpha \to \infty$, giá trị $\alpha$ áp đảo hoàn toàn tần suất xuất hiện thực tế $\text{đếm}_{jk}$ và $\text{tổng}_k$. Cả tử số và mẫu số tiến tới:
     $$\lim_{\alpha \to \infty} \frac{\text{đếm}_{jk} + \alpha}{\text{tổng}_k + \alpha V} = \frac{\alpha}{\alpha V} = \frac{1}{V}$$
   - Xác suất của mọi từ đều bằng nhau và bằng phẳng ở cả hai lớp ($1/V$). Do đó, thông tin ngữ nghĩa từ các từ bị triệt tiêu hoàn toàn, mô hình chỉ còn dựa vào xác suất tiên nghiệm $P(y)$ để gán nhãn, dẫn đến việc chọn lớp đa số.

---

## Bài tập 4: Tự viết bộ lọc không dùng thư viện

### 1. Đề bài
Dựa vào `d4_lam_tron.py`, viết bộ lọc hoàn chỉnh cho toàn bộ 300 tin nhắn. Tự đếm từ, tự làm trơn Laplace ($\alpha = 1.0$), tự tính điểm bằng tổng loga thay vì tích. Chia dữ liệu bằng `train_test_split(random_state=17)` và so sánh với `MultinomialNB`.

### 2. Kết quả so sánh
- **Accuracy bộ lọc tự viết**: `0.9467`
- **Accuracy MultinomialNB (scikit-learn)**: `0.9467`
- **Độ lệch Accuracy**: `0.000000`
- **Số mẫu dự đoán hoàn toàn trùng khớp**: `75/75 (100.0%)`
- **Confusion Matrix**: `TN = 45, FP = 0, FN = 4, TP = 26`

### 3. Giải thích cơ chế
1. **Vì sao tính tổng loga thay vì tích:**
   - Xác suất của từng từ là số dương rất nhỏ (thường $< 0.05$). Phép nhân liên tiếp nhiều số thực nhỏ sẽ dẫn đến hiện tượng **tràn dưới (underflow)**, khiến máy tính làm tròn thành $0.0000000000$ và tê liệt việc so sánh.
   - Khi lấy loga: $\log(A \times B) = \log A + \log B$, phép nhân biến thành phép cộng. Vì hàm loga đồng biến ngặt, thứ tự so sánh giữa hai lớp được bảo toàn tuyệt đối mà không gặp lỗi số học.
2. **Vì sao khớp tuyệt đối với scikit-learn:**
   - Bộ lọc tự viết tuân thủ đúng 100% công thức chuẩn của Multinomial Naive Bayes: tiên nghiệm $P(y) = N_k / N$, làm trơn Laplace với vocabulary size $V$, và quy tắc quyết định $\arg\max_k [\log P(y=k) + \sum x_j \log P(w_j \mid k)]$.

---

## Bài tập 5: Chọn sai biến thể Naive Bayes

### 1. Đề bài
Chạy `MultinomialNB` trên tập `sinh_vien.csv` với 2 cột `gio_on` và `diem_giua_ky`. So sánh accuracy với `GaussianNB`. Chương trình có báo lỗi không? Accuracy bằng bao nhiêu? Giải thích vì sao việc này là sai về nguyên tắc dù code vẫn chạy.

### 2. Kết quả thực nghiệm
- `GaussianNB` (đúng biến thể): **accuracy = 0.9000** (90.00%)
- `MultinomialNB` (sai biến thể): **accuracy = 0.8000** (80.00%)

### 3. Trả lời & Giải thích
1. **Chương trình có báo lỗi không?**
   - **KHÔNG báo lỗi.** `MultinomialNB` chỉ ném lỗi `ValueError: Negative values in data` khi dữ liệu có giá trị âm ($< 0$). Do `gio_on` và `diem_giua_ky` đều $\ge 0$, scikit-learn vẫn nhận dữ liệu và thực thi tính toán bình thường.
2. **Accuracy bằng bao nhiêu?**
   - Đạt **0.8000** (80.00%), thấp hơn hẳn mức **0.9000** của `GaussianNB`.
3. **Giải thích sai lầm về nguyên tắc:**
   - `MultinomialNB` được thiết kế cho **dữ liệu đếm rời rạc** (tần suất từ $0, 1, 2...$). Nó coi giá trị của thuộc tính là số lần diễn ra của sự kiện.
   - Nhưng `gio_on` (12.5 giờ) và `diem_giua_ky` (7.7 điểm) là **biến số thực liên tục**. Giá trị $12.5$ là một điểm trên trục số thực chứ không phải là sự kiện xảy ra 12.5 lần.
   - Biến liên tục tuân theo phân phối chuẩn (hình chuông), có độ lệch chuẩn và trung bình. Dùng `GaussianNB` mới mô hình hóa đúng hàm mật độ xác suất thực tế của sinh viên.
   - Đây là ví dụ điển hình về **lỗi im lặng (silent bug)** trong học máy: code chạy mượt mà nhưng vi phạm giả định lý thuyết, khiến hiệu năng mô hình giảm sút.

---

## Bài tập 6: Lớp hiếm và xác suất tiên nghiệm (Cái bẫy Accuracy)

### 1. Đề bài
Tạo tập dữ liệu mới mất cân bằng nặng: 180 tin thường và chỉ lấy 20 tin rác đầu tiên (tỷ lệ rác giảm từ 40% xuống còn 10%). Huấn luyện lại và chấm điểm. In ra Accuracy và Recall của lớp rác. Nhận xét Accuracy có còn phản ánh đúng chất lượng bộ lọc không?

### 2. Kết quả thực nghiệm
- Tập train: 150 mẫu (135 thường, 15 rác)
- Tập test: 50 mẫu (45 thường, 5 rác)
- **Accuracy**: `0.9800` (98.00%)
- **Recall (lớp rác, pos_label=1)**: `0.8000` (80.00%)
- **Precision (lớp rác, pos_label=1)**: `1.0000` (100.00%)
- **Ma trận nhầm lẫn**: `TN = 45, FP = 0, FN = 1, TP = 4`

### 3. Nhận xét & Giải thích
1. **Accuracy có phản ánh đúng chất lượng bộ lọc không?**
   - **HOÀN TOÀN KHÔNG.** Đây chính là **cái bẫy Accuracy (Accuracy Paradox)**.
   - Con số 98.00% trông rất ấn tượng, cao hơn cả tập gốc (94.67%), nhưng đó là ảo giác do lớp thường chiếm tới 90% tập kiểm tra.
   - Một mô hình ngớ ngẩn (Dummy Classifier) chỉ cần luôn đoán là "tin thường" cũng đã đạt Accuracy = 90.00% (45/50 mẫu đúng), dù không bắt được một tin rác nào (Recall = 0%).
2. **Cơ chế xác suất tiên nghiệm của Naive Bayes:**
   - Naive Bayes ra quyết định theo:
     $$\hat{y} = \arg\max_k P(y=k) \prod_j P(x_j \mid y=k)$$
   - Ở tập mới, xác suất tiên nghiệm của lớp rác bị thu hẹp chỉ còn $P(\text{rác}) = 0.10$, trong khi $P(\text{thường}) = 0.90$ (lớn gấp 9 lần).
   - Khoản chênh lệch tiên nghiệm này trở thành một rào cản rất lớn: mô hình trở nên cực kỳ dè dặt khi gán nhãn rác, đòi hỏi bằng chứng từ từ ngữ phải vô cùng mạnh. Kết quả là mô hình dễ bỏ lọt tin rác (Recall giảm từ 86.67% xuống 80.00%, bỏ lọt 1/5 tin rác trong tập test).
3. **Kết luận:**
   - Khi dữ liệu mất cân bằng nghiêm trọng, Accuracy mất hoàn toàn giá trị đánh giá. Bắt buộc phải dùng **Recall, Precision và F1-Score của lớp thiểu số** để đo lường độ tin cậy thực tế.
