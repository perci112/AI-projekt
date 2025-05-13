import cv2
import pandas as pd
import os
import numpy as np
from collections import defaultdict

# Ścieżki
csv_path = "output.csv"
image_folder = "Subject1"
output_folder = "AI"
train_folder = os.path.join(output_folder, "train")
test_folder = os.path.join(output_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

class_counts = defaultdict(int)

# Wczytaj CSV
df_raw = pd.read_csv(csv_path, header=None)
df_split = df_raw[0].str.split(",", expand=True)
df_split.columns = df_split.iloc[0]
df = df_split[1:].copy()
df = df.dropna(how='all')
df.reset_index(drop=True, inplace=True)

# Znajdź kolumny dla rąk
hand_columns = []
for col in df.columns:
    if col.endswith("_VFR"):
        prefix = col[:-4]
        left_col = f"{prefix}_VFL"
        if left_col in df.columns:
            hand_columns.append((col, left_col))

# Funkcja: maska koloru skóry
def get_skin_mask(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    # Morfologia: usunięcie szumów i białej otoczki
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    return mask

# Funkcja: przetwórz i zapisz
def crop_and_save(img, coords, side, column_label, filename):
    x, y, w, h = coords
    crop = img[y:y + h, x:x + w]

    # Skala szarości


    # Maska skóry
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Morfologia – oczyszczanie maski
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Największy kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    else:
        clean_mask = mask

    # Nałożenie maski na obraz w szarości
    hand_only = cv2.bitwise_and(gray_crop, gray_crop, mask=clean_mask)

    # Zapis
    name = f"{os.path.splitext(filename)[0]}_{side.upper()}_{column_label.upper()}.png"
    class_counts[column_label] += 1
    subfolder = test_folder if class_counts[column_label] % 5 == 0 else train_folder
    output_path = os.path.join(subfolder, name)
    cv2.imwrite(output_path, hand_only)
    print(f"✅ {filename} ➝ {name}")


# Główna pętla
for idx, row in df.iterrows():
    raw_path = str(row["rgb"]).replace("\\", "/").replace("..", "").strip("/")
    filename = os.path.basename(raw_path)
    image_path = os.path.join(image_folder, filename)

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Nie znaleziono zdjęcia: {image_path}")
        continue

    found = False
    for right_col, left_col in hand_columns:
        right_val = str(row[right_col])
        left_val = str(row[left_col])

        if right_val != '[0 0 0 0]' and left_val != '[0 0 0 0]':
            try:
                right_coords = list(map(int, right_val.strip('[]').split()))
                left_coords = list(map(int, left_val.strip('[]').split()))

                crop_and_save(image, right_coords, "right", right_col, filename)
                crop_and_save(image, left_coords, "left", left_col, filename)

                found = True
                break
            except Exception as e:
                print(f"⚠️ Błąd w {filename}: {e}")
                break

    if not found:
        print(f"❗️ Brak aktywnych kolumn dla {filename}")

