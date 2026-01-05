import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



import cv2
import numpy as np
import sqlite3
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

detector = MTCNN()
embedder = FaceNet()

def get_embedding(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype("float32")
    return embedder.embeddings([face])[0]

def recognize(embedding):
    conn = sqlite3.connect("database/face_recognition.db")
    cur = conn.cursor()
    cur.execute("SELECT name, embedding FROM users")
    data = cur.fetchall()
    conn.close()

    for name, emb in data:
        db_emb = pickle.loads(emb)
        if cosine_similarity([embedding], [db_emb])[0][0] > 0.8:
            return name
    return "Unknown"

test_images = [
    "test_images/test1.jpeg",
    "test_images/test2.jpeg",
    "test_images/test3.jpeg",
    "test_images/test4.jpeg"
]

for img_path in test_images:
    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb)

    for f in faces:
        x, y, w, h = f["box"]
        x, y = max(0, x), max(0, y)
        face = rgb[y:y+h, x:x+w]
        if face.size == 0:
            continue

        name = recognize(get_embedding(face))
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, name, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Result", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

