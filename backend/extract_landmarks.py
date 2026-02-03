import cv2
import os
import mediapipe as mp
import numpy as np
import csv


DATASET_PATH = r"C:\Users\junai\Documents\asl-translator\backend\asl_alphabet_train"
print("Looking for dataset at:", os.path.abspath(DATASET_PATH))
OUTPUT_FILE = "asl_landmarks.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)

    # header
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}"]
    header.append("label")
    writer.writerow(header)

    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(label_path):
            continue

        print(f"Processing {label}...")

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                row = []

                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y])

                row.append(label)
                writer.writerow(row)

hands.close()
print("Landmark extraction completed.")
