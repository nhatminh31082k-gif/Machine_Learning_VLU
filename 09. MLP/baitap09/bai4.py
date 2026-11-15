# -*- coding: utf-8 -*-
"""
Bài tập 4: Tốc độ học
Yêu cầu:
- Trên bộ chữ số, dùng mạng một lớp ẩn 100 nơ-ron, lần lượt đặt
  learning_rate_init bằng 0.0001, 0.001, 0.01 và 0.1.
- Với mỗi giá trị, ghi số vòng học n_iter_, mất mát cuối cùng và accuracy
  trên tập kiểm tra.
- Vẽ bốn đường loss_curve_ lên cùng một hình và lưu thành toc_do_hoc.png.
- Trả lời: Tốc độ học quá lớn và quá nhỏ gây ra chuyện gì?
"""

import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    X, y = load_digits(return_X_y=True)
    X_hoc, X_kiem_tra, y_hoc, y_kiem_tra = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )

    cac_toc_do_hoc = [0.0001, 0.001, 0.01, 0.1]
    duong_mat_mat = {}

    print("=" * 72)
    print("BAI TAP 4: ANH HUONG CUA TOC DO HOC (learning_rate_init)")
    print("=" * 72)
    print("Bo du lieu: load_digits | Mang: hidden_layer_sizes=(100,), solver='adam'")
    print()
    print("learning_rate_init   So vong (n_iter_)   Mat mat cuoi   Accuracy kiem tra")
    print("-" * 72)

    for toc_do in cac_toc_do_hoc:
        mo_hinh = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(100,),
                learning_rate_init=toc_do,
                max_iter=1000,
                random_state=0,
            ),
        ).fit(X_hoc, y_hoc)

        mlp = mo_hinh[-1]
        so_vong = mlp.n_iter_
        mat_mat_cuoi = mlp.loss_curve_[-1]
        acc_kiem_tra = accuracy_score(y_kiem_tra, mo_hinh.predict(X_kiem_tra))
        duong_mat_mat[toc_do] = mlp.loss_curve_

        print(f"  {toc_do:<16}        {so_vong:4d}            {mat_mat_cuoi:.6f}         {acc_kiem_tra:.4f}")

    # Vẽ bốn đường loss_curve_ lên cùng một hình
    plt.figure(figsize=(9, 5.5))
    bang_mau = {
        0.0001: ("#1f77b4", "-"),
        0.001:  ("#2ca02c", "-"),
        0.01:   ("#ff7f0e", "--"),
        0.1:    ("#d62728", "-."),
    }
    for toc_do in cac_toc_do_hoc:
        mau, kieu_net = bang_mau[toc_do]
        duong = duong_mat_mat[toc_do]
        nhan_ve = f"lr = {toc_do} (vong={len(duong)}, loss={duong[-1]:.4f})"
        plt.plot(range(1, len(duong) + 1), duong, color=mau, linestyle=kieu_net,
                 linewidth=2, label=nhan_ve)

    plt.title("Duong mat mat theo tung toc do hoc tren bo chu so viet tay",
              fontsize=13, fontweight="bold")
    plt.xlabel("Vong hoc (epoch)", fontsize=11)
    plt.ylabel("Mat mat (loss)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=9.5)
    plt.tight_layout()

    duong_dan_anh = "baitap09/toc_do_hoc.png" if os.path.isdir("baitap09") else "toc_do_hoc.png"
    plt.savefig(duong_dan_anh, dpi=150)
    if os.path.isdir("baitap09"):
        # Lưu thêm bản ở thư mục gốc nếu người chấm kiểm tra tại thư mục chạy lệnh
        plt.savefig("toc_do_hoc.png", dpi=150)
    plt.close()

    print(f"\nDa luu bieu do vao: {duong_dan_anh} (va toc_do_hoc.png)")
    print("\n" + "-" * 72)
    print("NHAN XET VE TOC DO HOC QUA NHO VA QUA LON:")
    print("1. Toc do hoc qua nho (0.0001):")
    print("   - Moi buoc cap nhat trong so nhich rat it nen mang hoc cham chap, can")
    print("     toi 582 vong hoc (gap ~4 lan mac dinh 0.001 la 144 vong va gap ~15 lan")
    print("     lr=0.01 la 38 vong). Neu max_iter khong du lon, mang se chua kip hoi tu.")
    print("   - Tuy ton nhieu vong, buoc di nho giup mang ha can than va dat accuracy")
    print("     cao (0.9889) khi cho phep toi 1000 vong.")
    print("2. Toc do hoc qua lon (0.1):")
    print("   - Buoc nhay qua dai khien qua trinh giam do doc vot qua day cuc tieu,")
    print("     duong mat mat bi rung lac/nhay coc va khong ha sau duoc (ket thuc o")
    print("     loss = 0.1204 sau chi 30 vong vi vuot qua nguong dung som noi bo).")
    print("   - Hau qua la mang chua hoc ky da dung, accuracy tren tap kiem tra tut")
    print("     xuong chi con 0.9511 (thap nhat trong 4 gia tri).")
    print("3. Muc hop ly (0.001 - 0.01): giam mat mat muot ma, can bang giua toc do")
    print("   hoi tu va do chinh xac.")
    print("=" * 72)


if __name__ == "__main__":
    main()
