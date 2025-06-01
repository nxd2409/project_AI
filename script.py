import os
import sys
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# Cấu hình mặc định
IMAGE_SIZE = (224, 224)
MODEL_PATH = 'cnn1.h5'


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Chuyển BGR->RGB, resize và chuẩn hóa về [0,1]"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    return img.astype(np.float32) / 255.0


def load_antispoofing_model(model_path: str) -> keras.Model:
    """Load model .h5 mà không compile lại (compile=False)"""
    if not os.path.exists(model_path):
        print(f"[Error] Model file '{model_path}' không tồn tại.")
        sys.exit(1)
    model = keras.models.load_model(model_path, compile=False)
    print(f"Loaded model from {model_path}")
    return model


def predict_image_array(x: np.ndarray) -> tuple:
    """Dự đoán một ảnh đã preprocess trả về nhãn và độ tin cậy"""
    score = model.predict(np.expand_dims(x, axis=0), verbose=0)[0][0]
    if score > 0.5:
        return 'Fake', score
    else:
        return 'Real', 1 - score


def predict_single_image(image_path: str):
    if not os.path.exists(image_path):
        print(f"[Error] File {image_path} không tồn tại.")
        return
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] Không đọc được ảnh {image_path}.")
        return

    x = preprocess_image(img)
    label, conf = predict_image_array(x)
    print(f"{Path(image_path).name}: {label} (confidence: {conf*100:.2f}%)")


def evaluate_folder(folder_path: str, show_details: bool=True):
    if not os.path.isdir(folder_path):
        print(f"[Error] Folder {folder_path} không tồn tại hoặc không phải folder.")
        return
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = sorted([f for f in Path(folder_path).iterdir() if f.suffix.lower() in exts])
    if not files:
        print(f"[Warning] Không tìm thấy ảnh trong {folder_path}.")
        return

    total, correct = 0, 0
    for f in files:
        stem = f.stem.lower()
        if stem.endswith('_real'):
            true_label = 'Real'
        elif stem.endswith('_fake'):
            true_label = 'Fake'
        else:
            continue

        img = cv2.imread(str(f))
        if img is None:
            print(f"[Warning] Không đọc được ảnh {f.name}, bỏ qua.")
            continue
        x = preprocess_image(img)
        pred_label, _ = predict_image_array(x)
        if pred_label == true_label:
            correct += 1
        total += 1
        if show_details:
            _, conf = predict_image_array(x)
            print(f"{f.name}: Pred={pred_label} ({conf*100:.2f}%), True={true_label}")

    if total == 0:
        print("[Info] Không có ảnh hợp lệ để đánh giá.")
    else:
        acc = correct / total * 100
        print(f"\nProcessed: {total} images → Accuracy: {acc:.2f}% ({correct}/{total})")


def realtime_face_detection():
    """Dùng webcam để detect khuôn mặt và phân loại real/fake"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Không mở được webcam.")
        return

    print("Bắt đầu webcam. Nhấn 'q' để thoát.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            x_proc = preprocess_image(face_roi)
            label, conf = predict_image_array(x_proc)
            # Vẽ bounding box và label
            color = (0, 255, 0) if label == 'Real' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} {conf*100:.1f}%", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Anti-Spoofing - Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def interactive_menu():
    while True:
        print("\n=== Anti-Spoofing Inference ===")
        print("1. Predict single image")
        print("2. Evaluate folder of test images")
        print("3. Real-time webcam face detection")
        print("Q. Quit")
        choice = input("Select [1/2/3/Q]: ").strip().lower()
        if choice == '1':
            path = input("Enter image path: ").strip()
            predict_single_image(path)
        elif choice == '2':
            path = input("Enter folder path: ").strip()
            evaluate_folder(path)
        elif choice == '3':
            realtime_face_detection()
        elif choice == 'q':
            print("Exiting.")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == '__main__':
    model = load_antispoofing_model(MODEL_PATH)
    # Khởi chạy menu mà không truyền tham số model
    interactive_menu()
