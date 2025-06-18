import os
import sys
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from mtcnn import MTCNN

# Cấu hình mặc định
IMAGE_SIZE = (224, 224)
MODEL_PATH = 'cnn2.h5'
THRESHOLD = 0.5

# Khởi tạo bộ phát hiện khuôn mặt MTCNN
detector = MTCNN()


def preprocess_image(image: np.ndarray) -> np.ndarray:
    faces = detector.detect_faces(image)
    if faces:
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        face = image[y:y+h, x:x+w]
    else:
        face = image

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, IMAGE_SIZE)
    return face.astype(np.float32) / 255.0


def load_antispoofing_model(model_path: str) -> keras.Model:
    if not os.path.exists(model_path):
        print(f"[Error] Model file '{model_path}' không tồn tại.")
        sys.exit(1)
    model = keras.models.load_model(model_path, compile=False)
    print(f"Loaded model from {model_path}")
    return model


def predict_image_array(model: keras.Model, x: np.ndarray) -> tuple:
    """
    Dự đoán một ảnh đã preprocess, trả về nhãn và độ tin cậy.
    """
    score = model.predict(np.expand_dims(x, axis=0), verbose=0)[0][0]
    if score > THRESHOLD:
        return 'Fake', score
    else:
        return 'Real', 1 - score


def predict_faces_in_image(model: keras.Model, image_path: str):
    """
    Load ảnh tĩnh, phát hiện mặt với MTCNN, dự đoán và hiển thị kết quả.
    Cửa sổ hiển thị được mở ở chế độ resizable và tự phóng to vừa màn hình.
    """
    if not os.path.exists(image_path):
        print(f"[Error] File {image_path} không tồn tại.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] Không đọc được ảnh {image_path}.")
        return

    orig_h, orig_w = img.shape[:2]

    faces = detector.detect_faces(img)
    if not faces:
        print("Không tìm thấy khuôn mặt nào!")
    else:
        for face_info in faces:
            x, y, w, h = face_info['box']
            x, y = max(0, x), max(0, y)
            face_roi = img[y:y+h, x:x+w]
            x_proc = preprocess_image(face_roi)
            label, conf = predict_image_array(model, x_proc)
            color = (0, 255, 0) if label == 'Real' else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img,
                        f"{label} {conf*100:.1f}%",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2)

    # Mở cửa sổ ở chế độ resizable và phóng to vừa với kích thước gốc của ảnh
    window_name = 'Anti-Spoofing - Detected Faces'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, orig_w, orig_h)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def evaluate_folder(model: keras.Model, folder_path: str, show_details: bool=True):
    """
    Duyệt thư mục ảnh, dự đoán từng ảnh dựa trên tên file _real/_fake và tính accuracy.
    """
    if not os.path.isdir(folder_path):
        print(f"[Error] Folder {folder_path} không tồn tại hoặc không phải folder.")
        return

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = sorted([f for f in Path(folder_path).iterdir() if f.suffix.lower() in exts])
    if not files:
        print(f"[Warning] Không tìm thấy ảnh trong {folder_path}.")
        return

    total = correct = 0
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
        pred_label, conf = predict_image_array(model, x)

        if pred_label == true_label:
            correct += 1
        total += 1

        if show_details:
            print(f"{f.name}: Pred={pred_label} ({conf*100:.2f}%), True={true_label}")

    if total == 0:
        print("[Info] Không có ảnh hợp lệ để đánh giá.")
    else:
        acc = correct / total * 100
        print(f"\nProcessed: {total} images → Accuracy: {acc:.2f}% ({correct}/{total})")


def realtime_face_detection(model: keras.Model):
    """
    Dùng webcam, phát hiện mặt theo MTCNN, dự đoán real/fake theo thời gian thực.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Không mở được webcam.")
        return

    print("Bắt đầu webcam. Nhấn 'q' để thoát.")

    # Tạo cửa sổ resizable
    window_name = 'Anti-Spoofing - Webcam'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for face_info in detector.detect_faces(frame):
            x, y, w, h = face_info['box']
            x, y = max(0, x), max(0, y)
            face_roi = frame[y:y+h, x:x+w]

            x_proc = preprocess_image(face_roi)
            label, conf = predict_image_array(model, x_proc)

            color = (0, 255, 0) if label == 'Real' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame,
                        f"{label} {conf*100:.1f}%",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def interactive_menu(model: keras.Model):
    """
    Menu CLI để chọn từng chức năng.
    """
    while True:
        print("\n=== Anti-Spoofing Inference ===")
        print("1. Detect & predict faces in image")
        print("2. Evaluate folder of test images")
        print("3. Real-time webcam face detection")
        print("Q. Quit")
        choice = input("Select [1/2/3/Q]: ").strip().lower()

        if choice == '1':
            path = input("Enter image path: ").strip()
            predict_faces_in_image(model, path)
        elif choice == '2':
            path = input("Enter folder path: ").strip()
            evaluate_folder(model, path)
        elif choice == '3':
            realtime_face_detection(model)
        elif choice == 'q':
            print("Exiting.")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == '__main__':
    model = load_antispoofing_model(MODEL_PATH)
    interactive_menu(model)
