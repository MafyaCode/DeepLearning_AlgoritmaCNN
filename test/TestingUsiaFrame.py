import cv2
import mediapipe as mp # Import MediaPipe
from deepface import DeepFace
import numpy as np
import time # Opsional: untuk menghitung FPS

# --- Pengaturan Optimasi ---
PROCESS_EVERY_N_FRAMES = 3 # Jalankan analisis setiap 3 frame (sesuaikan sesuai kebutuhan)

# --- Inisialisasi MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# --- Inisialisasi OpenCV VideoCapture ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened(): 
    print("Error: Tidak bisa membuka webcam.")
    exit()

print(f"Mulai analisis (proses setiap {PROCESS_EVERY_N_FRAMES} frame)... Tekan 'q' untuk keluar.")

# --- Variabel untuk menyimpan hasil terakhir & counter ---
frame_counter = 0
last_known_face_results = [] # Simpan hasil analisis wajah terakhir
last_known_hand_landmarks = None # Simpan landmark tangan terakhir

# --- Variabel untuk FPS ---
prev_frame_time = 0
new_frame_time = 0

# --- Loop Utama ---
while True:
    ret, frame_original = cap.read()
    if not ret:
        print("Error: Tidak bisa menerima frame. Keluar...")
        break

    # Balik frame secara horizontal
    frame = cv2.flip(frame_original, 1)
    frame_height, frame_width, _ = frame.shape # Dapatkan dimensi frame

    # Tingkatkan frame counter
    frame_counter += 1

    # --- Lakukan Analisis Hanya Setiap N Frame ---
    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
        # Reset hasil sementara untuk frame ini
        current_face_results = []
        current_hand_landmarks = None

        # 1. Analisis Wajah & Usia (DeepFace)
        try:
            hasil_analisis_wajah = DeepFace.analyze(
                img_path=frame.copy(), # Berikan salinan frame
                actions=['emotion', 'age'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            # Simpan hasil jika valid
            if isinstance(hasil_analisis_wajah, list) and len(hasil_analisis_wajah) > 0:
                current_face_results = hasil_analisis_wajah

        except Exception as e:
            # print(f"Error DeepFace di frame {frame_counter}: {e}")
            pass # Abaikan error analisis

        # 2. Pelacakan Tangan (MediaPipe)
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results_tangan = hands.process(frame_rgb)
            frame.flags.writeable = True # Set writeable kembali setelah proses

            # Simpan hasil jika valid
            if results_tangan.multi_hand_landmarks:
                current_hand_landmarks = results_tangan.multi_hand_landmarks

        except Exception as e:
            # print(f"Error MediaPipe di frame {frame_counter}: {e}")
            pass # Abaikan error analisis

        # Update hasil terakhir yang diketahui
        last_known_face_results = current_face_results
        last_known_hand_landmarks = current_hand_landmarks

    # --- Gambar Hasil (Dilakukan Setiap Frame) ---
    # Gunakan hasil terakhir yang diketahui untuk menggambar

    # 1. Gambar Hasil Analisis Wajah
    if last_known_face_results:
        for analisis_wajah in last_known_face_results:
            emosi_str = analisis_wajah.get('dominant_emotion', 'N/A')
            usia = analisis_wajah.get('age', 'N/A')
            usia_str = str(usia)
            region = analisis_wajah.get('region', None)

            if region:
                # Pastikan koordinat tidak keluar batas frame
                x = max(0, region['x'])
                y = max(0, region['y'])
                w = region['w']
                h = region['h']
                x2 = min(frame_width, x + w)
                y2 = min(frame_height, y + h)

                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                teks_info_wajah = f"Emosi: {emosi_str} | Usia: {usia_str}"
                # Atur posisi teks agar tidak keluar frame atas
                text_y = y - 10 if y > 20 else y + h + 20
                cv2.putText(frame, teks_info_wajah, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 2. Gambar Hasil Pelacakan Tangan
    if last_known_hand_landmarks:
        for hand_landmarks in last_known_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS)

    # --- Hitung dan Tampilkan FPS ---
    new_frame_time = time.time()
    # Hindari pembagian dengan nol jika frame time sama
    time_diff = new_frame_time - prev_frame_time
    if time_diff > 0:
        fps = 1 / time_diff
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # --- Tampilkan Frame Hasil ---
    cv2.imshow('Analisis Wajah, Usia, dan Tangan - Tekan Q untuk Keluar', frame)

    # --- Kondisi Keluar ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Tombol 'q' ditekan, keluar...")
        break

# --- Cleanup ---
print("Melepaskan sumber dsaya...")
cap.release() 
hands.close()
cv2.destroyAllWindows()
print("Aplikasi ditutup.")
# ```
#
# **Penjelasan Perubahan Kunci:**
#
# 1.  **`PROCESS_EVERY_N_FRAMES`**: Konstanta baru di awal untuk menentukan seberapa sering analisis dilakukan (misalnya, `3` berarti analisis dilakukan pada frame 1, 4, 7, dst.). Anda bisa mengubah angka ini untuk menyeimbangkan antara kehalusan update analisis dan beban CPU/GPU. Angka lebih besar = lebih ringan tapi update analisis lebih jarang.
# 2.  **`frame_counter`**: Menghitung jumlah frame yang sudah dilewati.
# 3.  **Variabel Penyimpan Hasil Terakhir:** `last_known_face_results` dan `last_known_hand_landmarks` digunakan untuk menyimpan hasil analisis dari frame terakhir yang *diproses*.
# 4.  **Blok Analisis Bersyarat:** Kode untuk `DeepFace.analyze()` dan `hands.process()` sekarang hanya dijalankan jika `frame_counter % PROCESS_EVERY_N_FRAMES == 0`. Hasilnya disimpan ke variabel `current_...` lalu disalin ke `last_known_...`.
# 5.  **Blok Menggambar Selalu Jalan:** Kode untuk menggambar kotak wajah, teks, dan landmark tangan sekarang berjalan di *setiap* iterasi loop, tetapi menggunakan data dari `last_known_face_results` dan `last_known_hand_landmarks`. Ini memastikan bahwa meskipun analisis tidak dilakukan di setiap frame, tampilan visual (kotak, teks, landmark) tetap ada berdasarkan deteksi terakhir.
# 6.  **Penyesuaian Koordinat Teks:** Menambahkan logika sederhana agar teks info wajah tidak keluar dari batas atas frame.
# 7.  **Perhitungan FPS:** Sedikit diperbaiki untuk menghindari potensi pembagian dengan nol.
# 
# Dengan cara ini, beban pemrosesan utama (DeepFace dan MediaPipe) berkurang secara signifikan, memungkinkan aplikasi berjalan lebih lancar pada perangkat yang mungkin tidak terlalu kuat, sambil tetap menampilkan visualisasi yang relatif upda