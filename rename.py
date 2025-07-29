import os
import tkinter as tk
from tkinter import filedialog

# Tạo cửa sổ ẩn để chọn thư mục
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(title="Chọn thư mục chứa ảnh Anomaly cần đổi tên thành 4 chữ số")

if not folder_path:
    print("❌ Không chọn thư mục. Thoát.")
    exit()

count = 0
skipped = 0

# Duyệt các file trong thư mục
for filename in os.listdir(folder_path):
    if filename.endswith(".png") and "-" in filename:
        parts = filename[:-4].split("-")  # bỏ .png rồi tách dấu -
        if len(parts) == 2 and parts[1].isdigit():
            new_name = parts[1].zfill(4) + ".png"  # đảm bảo 4 số
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"✅ {filename} → {new_name}")
                count += 1
            else:
                print(f"⚠️ Bỏ qua (đã tồn tại): {new_name}")
                skipped += 1

print(f"\n🏁 Hoàn tất: {count} file đổi tên, {skipped} bị bỏ qua.")
