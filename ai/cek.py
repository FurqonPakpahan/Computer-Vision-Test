import cv2
import numpy as np
from screeninfo import get_monitors

# Load model dan configurasi
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# List  class labels
class_labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                 "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                 "train", "tvmonitor"]

# Video file
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Ukuran untuk layar penuh
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Inisialisasi untuk melacak deteksi orang
detection_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Menyiapkan bingkai untuk deteksi objek
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(300, 300), mean=(0, 0, 0))
    net.setInput(blob)
    detections = net.forward()

    # Menghitung orang dalam bingkai
    person_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.1:  
            class_id = int(detections[0, 0, i, 1])
            if class_labels[class_id] == 'person':
                person_count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Melacak orang yang terdeteksi bingkau
    detection_frames.append(person_count)
    if len(detection_frames) > 120: 
        detection_frames.pop(0)

    # Peringatan jika lebih dari 4 orang
    if all(count > 4 for count in detection_frames):
        cv2.putText(frame, "PERINGATAN: Lebih dari 4 orang terdeteksi terus menerus!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Menampilkan jumlah orang yang terdeteksi
    cv2.putText(frame, f'Jumlah Orang: {person_count}', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mengubah ukuran frame agar sesuai dengan ukuran laya
    frame_height, frame_width = frame.shape[:2]
    aspect_ratio = frame_width / frame_height
    new_width = screen_width
    new_height = int(new_width / aspect_ratio)

    if new_height > screen_height:
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Membuat layar hitam untuk menampung frame yang diubah ukurannya 
    black_screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    x_offset = (screen_width - new_width) // 2
    y_offset = (screen_height - new_height) // 2

    black_screen[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

    # Menampilkan frame
    cv2.imshow("Frame", black_screen)

    # Berhenti tekan q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
