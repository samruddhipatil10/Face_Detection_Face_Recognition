import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



import cv2
import numpy as np
import os
import sqlite3
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()

def get_embedding(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype("float32")
    return embedder.embeddings([face])[0]

def save_embedding(name, embedding):
    conn = sqlite3.connect("database/face_recognition.db")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (name, embedding) VALUES (?, ?)",
        (name, pickle.dumps(embedding))
    )
    conn.commit()
    conn.close()

DATASET = "dataset"

for person in os.listdir(DATASET):
    person_path = os.path.join(DATASET, person)
    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img = cv2.imread(os.path.join(person_path, img_name))
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        for f in faces:
            x, y, w, h = f["box"]
            x, y = max(0, x), max(0, y)
            face = rgb[y:y+h, x:x+w]
            if face.size == 0:
                continue

            emb = get_embedding(face)
            save_embedding(person, emb)

    print(f"Saved: {person}")

print("DONE")
