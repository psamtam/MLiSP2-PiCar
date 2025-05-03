import cv2
import os
import csv
from datetime import datetime
import pandas as pd
import numpy as np

timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # Format: YYYYMMDD_HHMM


image_dir = "/mnt/c/Users/psamt/OneDrive - The University of Nottingham/_Spring/PHYS4036_Machine Learning in Science Part II/Project/Captured_data/processed/T_left"
unprocessed_csv = f"{image_dir}/_labels_unprocessed.csv"
unprocesssed_df = pd.read_csv(unprocessed_csv).set_index("image_id")
old_csv = f"{image_dir}/_labels_old.csv"
old_df = pd.read_csv(old_csv).set_index("image_id")
old_index = list(old_df.index)

key_to_change = None
if "left" in image_dir:
    key_to_change = "left_arrow"
elif "right" in image_dir:
    key_to_change = "right_arrow"

output_csv = f"{image_dir}/_labels_{timestamp}.csv"
labels = {
    "0": "no_sign",
    "1": "has_sign",
}  # Edit for each task

csv_dir = image_dir
last_csv = None
start_from = 1

def get_img_path(image_id):
    return os.path.join(image_dir, f"{image_id}.png")

image_window = "Labelling"
text_window = "Labels"
cv2.namedWindow(image_window)
cv2.namedWindow(text_window)
cv2.moveWindow(image_window, 1024, 300)  # Image window position
cv2.moveWindow(text_window, 900, 100)  # Text window to the right
cv2.resizeWindow(image_window, 600, 600)  # Image window size
cv2.resizeWindow(text_window, 400, 100)  # Small text window


with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id"] + list(unprocesssed_df.columns))

    # exit()

    for i in list(unprocesssed_df.index):        
        if i in old_index:
            writer.writerow([i] + list(old_df.loc[i]))
            f.flush()
            continue

        angle = unprocesssed_df.loc[i]["angle"]
        speed = unprocesssed_df.loc[i]["speed"]
        # if speed == 1:
        #     continue

        text = f"ImageID: {i} | Angle: {angle} | Speed: {int(speed)}"

        img = cv2.imread(get_img_path(i))
        cv2.imshow(image_window, img)

        text_img = np.zeros((100, 1000, 3), dtype="uint8")  # Black background
        if speed == 0:
            text_img[:, :, 2] = 255
        elif speed == 1:
            text_img[:, :, 1] = 255            


        cv2.putText(
            text_img,
            text,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255) if speed == 0 else (0, 0, 0),
            2,
            cv2.LINE_AA,  # White text
        )
        cv2.imshow(text_window, text_img)  # Text in separate window

        key = cv2.waitKey(0) & 0xFF

        print(chr(key))

        row = unprocesssed_df.loc[i]

        row[key_to_change] = int(chr(key))

        writer.writerow([i] + list(row))

        f.flush()


        # if label == "multi":
        #     key1 = cv2.waitKey(0) & 0xFF
        #     key2 = cv2.waitKey(0) & 0xFF
        #     label1, label2 = labels.get(chr(key1), "unknown"), labels.get(chr(key2), "unknown")
        #     writer.writerow([i, label1, label2])
        # else:
        #     writer.writerow([i, label])
        # f.flush()  # Force save to disk
        print(f"Labeled {i} as {int(chr(key))}")
cv2.destroyAllWindows()
