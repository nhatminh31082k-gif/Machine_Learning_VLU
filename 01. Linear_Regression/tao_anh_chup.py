# -*- coding: utf-8 -*-
"""
Script tự động chạy từng bài tập và tạo ảnh chụp màn hình (screenshot) dạng terminal Windows PowerShell.
Sử dụng font Consolas chuẩn của Windows.
"""

import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_SIZE = 16
LINE_HEIGHT = 22
PADDING_LEFT = 24
PADDING_TOP = 50
PADDING_BOTTOM = 25
BG_COLOR = (12, 12, 12)
HEADER_COLOR = (31, 31, 31)
TEXT_COLOR = (220, 220, 220)
PROMPT_COLOR = (255, 255, 255)
COMMAND_COLOR = (255, 215, 0)
TITLE_COLOR = (200, 200, 200)

font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
font_bold = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", FONT_SIZE)
font_title = ImageFont.truetype(FONT_PATH, 13)


def render_terminal_screenshot(command_str, output_lines, save_path):
    """Vẽ một cửa sổ terminal chứa dòng lệnh đã chạy và toàn bộ kết quả in ra."""
    all_lines = [("prompt", "PS D:\\LAB1> " + command_str)]
    for line in output_lines:
        all_lines.append(("text", line))

    # Tính toán kích thước ảnh dựa trên nội dung
    max_len = max(len(l[1]) for l in all_lines)
    img_width = max(860, int(max_len * 9.8) + PADDING_LEFT * 2)
    img_height = PADDING_TOP + len(all_lines) * LINE_HEIGHT + PADDING_BOTTOM

    img = Image.new("RGB", (img_width, img_height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    # 1. Vẽ thanh tiêu đề cửa sổ (Title bar)
    draw.rectangle([0, 0, img_width, 36], fill=HEADER_COLOR)
    draw.text((16, 10), "Windows PowerShell - D:\\LAB1", font=font_title, fill=TITLE_COLOR)

    # Vẽ nút điều khiển cửa sổ (Minimize, Maximize, Close)
    # Nút thu nhỏ
    draw.line([img_width - 110, 18, img_width - 98, 18], fill=(180, 180, 180), width=1)
    # Nút phóng to
    draw.rectangle([img_width - 75, 12, img_width - 63, 24], outline=(180, 180, 180), width=1)
    # Nút đóng
    draw.line([img_width - 35, 12, img_width - 23, 24], fill=(220, 80, 80), width=2)
    draw.line([img_width - 35, 24, img_width - 23, 12], fill=(220, 80, 80), width=2)

    # 2. Vẽ nội dung dòng lệnh và kết quả
    y_pos = PADDING_TOP
    for kind, text in all_lines:
        if kind == "prompt":
            draw.text((PADDING_LEFT, y_pos), "PS D:\\LAB1> ", font=font_bold, fill=(90, 200, 255))
            prompt_len = draw.textlength("PS D:\\LAB1> ", font=font_bold)
            draw.text((PADDING_LEFT + prompt_len, y_pos), command_str, font=font_bold, fill=COMMAND_COLOR)
        else:
            # Màu sắc điểm nhấn cho một số loại thông báo
            text_color = TEXT_COLOR
            if "===" in text:
                text_color = (100, 220, 100)
            elif "CANH BAO" in text:
                text_color = (255, 140, 0)
            elif "MSE" in text or "R2" in text:
                text_color = (240, 240, 240)
            draw.text((PADDING_LEFT, y_pos), text, font=font, fill=text_color)
        y_pos += LINE_HEIGHT

    img.save(save_path)
    print(f"Da tao anh chup: {save_path}")


def chay_va_chup(cmd, file_ten):
    res = subprocess.run(
        cmd,
        cwd=r"D:\LAB1",
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    lines = res.stdout.strip().split("\n")
    if res.stderr.strip():
        lines.extend(["STDERR:"] + res.stderr.strip().split("\n"))

    output_dir = r"D:\LAB1\baitap01"
    save_path = os.path.join(output_dir, file_ten)
    render_terminal_screenshot(cmd, lines, save_path)


if __name__ == "__main__":
    chay_va_chup("python baitap01/bai1.py", "screenshot_bai1.png")
    chay_va_chup("python baitap01/bai2.py", "screenshot_bai2.png")
    chay_va_chup("python baitap01/bai3.py", "screenshot_bai3.png")
    chay_va_chup("python baitap01/bai4.py", "screenshot_bai4.png")
    chay_va_chup("python baitap01/bai5.py 0.001", "screenshot_bai5_lr0001.png")
    chay_va_chup("python baitap01/bai5.py 1.02", "screenshot_bai5_lr102.png")
    chay_va_chup("python baitap01/bai5.py", "screenshot_bai5_tonghop.png")
    chay_va_chup("python baitap01/bai6.py", "screenshot_bai6.png")
