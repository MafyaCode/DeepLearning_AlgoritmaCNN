import cv2
from deepface import DeepFace
import numpy as np

# Inisialisasi penangkap video dari webcam (biasanya indeks 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak bisa membuka webcam.")
    exit()

# Loop utama untuk membaca frame dari webcam
while True:
    # Baca satu frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak bisa menerima frame (stream berakhir?). Keluar...")
        break 

    # Salin frame untuk diproses, agar frame asli tidak termodifikasi oleh DeepFace secara langsung
    frame_proses = frame.copy()

    try:
        # Analisis ekspresi menggunakan DeepFace
        # actions = ['emotion'] akan fokus hanya pada analisis emosi
        # enforce_detection=False agar tidak error jika tidak ada wajah terdeteksi
        hasil_analisis = DeepFace.analyze(
            img_path=frame_proses,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv' # atau 'ssd', 'mtcnn', 'dlib' - opencv seringkali paling cepat
        )

        # DeepFace.analyze mengembalikan list, bahkan jika hanya 1 wajah
        # Kita ambil hasil analisis pertama (jika ada wajah terdeteksi)
        if isinstance(hasil_analisis, list) and len(hasil_analisis) > 0:
            analisis = hasil_analisis[0]
            emosi = analisis['dominant_emotion']
            # Dapatkan kotak pembatas (bounding box) wajah
            region = analisis['region'] # {'x': ..., 'y': ..., 'w': ..., 'h': ...}
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            # Gambar kotak di sekitar wajah pada frame asli
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Kotak hijau

            # Tulis teks emosi di atas kotak
            teks_emosi = f"Emosi: {emosi}"
            cv2.putText(frame, teks_emosi, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    except Exception as e:
        # Tangani error jika tidak ada wajah terdeteksi atau masalah lain
        # print(f"Tidak ada wajah terdeteksi atau error: {e}") # Uncomment untuk debugging
        pass # Abaikan saja jika tidak ada wajah

    # Tampilkan frame yang sudah diproses (dengan kotak dan teks)
    cv2.imshow('Analisis Ekspresi Wajah', frame)

    # Hentikan loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Setelah loop selesai, lepaskan penangkap video dan tutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()

print("Aplikasi ditutup.")