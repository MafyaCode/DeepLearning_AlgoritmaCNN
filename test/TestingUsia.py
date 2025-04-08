import cv2
import mediapipe as mp # Import MediaPipe
from deepface import DeepFace
import numpy as np
import time # Opsional: untuk menghitung FPS

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
# max_num_hands=2 : Deteksi maksimal 2 tangan
# min_detection_confidence : Tingkat kepercayaan minimum agar deteksi tangan dianggap berhasil
# min_tracking_confidence : Tingkat kepercayaan minimum agar landmark tangan bisa dilacak antar frame
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils # Utilitas untuk menggambar landmark

# Inisialisasi OpenCV VideoCapture
# Menggunakan indeks 0 (webcam default) dan backend CAP_DSHOW (sering lebih stabil di Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Tidak bisa membuka webcam.")
    exit()

print("Mulai analisis... Tekan 'q' untuk keluar.")

# Opsional: Untuk menghitung FPS
prev_frame_time = 0
new_frame_time = 0

# Loop utama
while True:
    # Baca frame
    ret, frame_original = cap.read()
    if not ret:
        print("Error: Tidak bisa menerima frame. Keluar...")
        break

    # Balik frame secara horizontal (seperti cermin) agar lebih intuitif
    # Pemrosesan dan penggambaran akan dilakukan pada frame yang sudah dibalik ini
    frame = cv2.flip(frame_original, 1)

    # --- Analisis Wajah & Usia (DeepFace) ---
    usia_str = "N/A" # Nilai default jika tidak terdeteksi
    emosi_str = "N/A" # Nilai default jika tidak terdeteksi
    try:
        # Analisis ekspresi DAN usia
        # DeepFace menerima frame BGR dari OpenCV secara langsung
        hasil_analisis_wajah = DeepFace.analyze(
            img_path=frame,
            actions=['emotion', 'age'], # Minta analisis emosi dan usia
            enforce_detection=False, # Jangan error jika tidak ada wajah
            detector_backend='opencv' # Backend deteksi wajah (cepat)
        )

        # Hasilnya adalah list, ambil elemen pertama jika ada
        if isinstance(hasil_analisis_wajah, list) and len(hasil_analisis_wajah) > 0:
            analisis_wajah = hasil_analisis_wajah[0]
            # Gunakan .get() untuk keamanan jika key tidak ada
            emosi_str = analisis_wajah.get('dominant_emotion', 'N/A')
            usia = analisis_wajah.get('age', 'N/A')
            usia_str = str(usia) # Konversi usia ke string
            region = analisis_wajah.get('region', None) # Dapatkan bounding box wajah

            # Gambar kotak dan teks jika region ditemukan
            if region:
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Kotak hijau
                teks_info_wajah = f"Emosi: {emosi_str} | Usia: {usia_str}"
                # Taruh teks sedikit di atas kotak
                cv2.putText(frame, teks_info_wajah, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    except Exception as e:
        # Jika ada error saat analisis (misal tidak ada wajah), abaikan saja
        # print(f"Error DeepFace: {e}") # Aktifkan untuk debugging
        pass

    # --- Pelacakan Tangan (MediaPipe) ---
    # Konversi warna frame dari BGR (OpenCV) ke RGB (MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Optimasi: Tandai frame sebagai tidak bisa ditulis untuk mempercepat proses MediaPipe
    frame_rgb.flags.writeable = False

    # Proses frame untuk mendeteksi tangan
    results_tangan = hands.process(frame_rgb)

    # Kembalikan flag writeable ke True agar bisa digambari oleh OpenCV
    frame.flags.writeable = True

    # Gambar landmark tangan jika terdeteksi
    if results_tangan.multi_hand_landmarks:
        # Loop untuk setiap tangan yang terdeteksi
        for hand_landmarks in results_tangan.multi_hand_landmarks:
            # Gambar titik landmark dan garis penghubungnya ke frame asli (yang sudah dibalik)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                # Anda bisa menambahkan style drawing di sini jika mau
                # landmark_drawing_spec=mp_drawing.DrawingSpec(...),
                # connection_drawing_spec=mp_drawing.DrawingSpec(...)
                )

    # --- Hitung dan Tampilkan FPS (Opsional) ---
    new_frame_time = time.time()
    fps = 20 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) # Teks biru

    # --- Tampilkan Hasil ---
    cv2.imshow('Analisis Wajah, Usia, dan Tangan - Tekan Q untuk Keluar', frame)

    # --- Kondisi Keluar ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Tombol 'q' ditekan, keluar...")
        break

# --- Cleanup ---
print("Melepaskan sumber daya...")
cap.release() # Lepaskan webcam
hands.close() # Tutup objek MediaPipe Hands
cv2.destroyAllWindows() # Tutup semua jendela OpenCV
print("Aplikasi ditutup.")

# ```

# **Penjelasan Perubahan:**
#
# 1.  **Import `mediapipe`:** Menambahkan `import mediapipe as mp`.
# 2.  **Inisialisasi MediaPipe Hands:** Membuat objek `mp_hands.Hands()` sebelum loop utama untuk mendeteksi tangan. Objek `mp_drawing` juga dibuat untuk membantu menggambar hasilnya.
# 3.  **Analisis Usia DeepFace:** Di dalam blok `try` untuk `DeepFace.analyze()`, parameter `actions` diubah menjadi `['emotion', 'age']` untuk meminta analisis usia juga. Hasil usia diambil dari dictionary `analisis_wajah` menggunakan `.get('age', 'N/A')` dan ditambahkan ke teks yang ditampilkan.
# 4.  **Pemrosesan Tangan MediaPipe:**
#     * Frame dikonversi dari BGR ke RGB karena MediaPipe memerlukan format RGB (`cv2.cvtColor`).
#     * Frame diproses dengan `hands.process(frame_rgb)`.
#     * Dilakukan pengecekan apakah tangan terdeteksi (`if results_tangan.multi_hand_landmarks:`).
#     * Jika terdeteksi, dilakukan loop untuk setiap tangan dan `mp_drawing.draw_landmarks()` digunakan untuk menggambar 21 titik landmark dan garis penghubungnya pada frame.
# 5.  **Flip Frame:** Frame dibalik secara horizontal (`cv2.flip(frame_original, 1)`) di awal loop agar tampilan di layar seperti cermin, yang biasanya lebih intuitif untuk interaksi pengguna. Semua proses dan penggambaran dilakukan pada `frame` yang sudah dibalik ini.
# 6.  **FPS (Opsional):** Menambahkan perhitungan dan tampilan FPS sederhana untuk memantau performa.
# 7.  **Cleanup:** Menambahkan `hands.close()` saat program berakhir untuk melepaskan sumber daya MediaPipe.
#
# Sekarang, jika Anda menjalankan kode ini (dan webcam Anda berfungsi), Anda seharusnya melihat feed video dengan kotak di sekitar wajah yang terdeteksi, teks yang menampilkan emosi dominan dan perkiraan usia, serta landmark yang digambar pada tangan yang terdeteksi di depan kamera. Tekan 'q' untuk kelu