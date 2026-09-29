# MazeTrack Desktop v2.0 🐾
**Alternatif Sederhana, Kuat, & Mandiri dari ANY-maze untuk Analisis Perilaku Hewan Laboratorium**

**MazeTrack Desktop** adalah aplikasi desktop lokal (*native Windows client*) berbasis Python, OpenCV, dan PyQt5 yang dirancang khusus untuk memantau, melacak, dan menganalisis pergerakan hewan laboratorium (mencit/tikus) dalam uji perilaku neurosains (*behavioral neuroscience*).

---

## 🚀 Cara Menjalankan Aplikasi

Cukup lakukan salah satu dari dua cara berikut:

### Cara 1: Sekali Klik (Rekomendasi)
Klik dua kali (*double-click*) file:
```
run.bat
```

### Cara 2: Lewat Command Prompt / Terminal
Buka folder `e:\tttt` dan jalankan:
```bash
python main.py
```

---

## ✨ Fitur-Fitur Terbaru (v2.0)

### 1. 🎯 Kendali Penuh Tracking di Tangan Pengguna
- **Tidak Otomatis Berjalan Sendiri**: Saat aplikasi dibuka, sistem berada dalam **Mode Setup / Persiapan (Idle)**. Video dalam posisi siap (*preview*), timer dan kalkulasi metrik belum berjalan.
- **Tombol Kontrol Eksperimen Jelas**:
  - **`▶ Mulai Tracking (Start Trial)`** (Hijau): Mulai merekam lintasan dan menghitung metrik saat Anda sudah siap.
  - **`⏸ Jeda (Pause)`** (Kuning/Amber): Menjeda perekaman sementara.
  - **`⏹ Selesai / Hentikan Tracking`** (Merah): Menghentikan percobaan, membekukan data metrik, dan siap diekspor.
  - **`🔄 Reset Percobaan`**: Mengembalikan timer dan data ke 0.
- **Status Banner**: Indikator status visual di bagian atas: `[ 🛠️ MODE SETUP / PERSIAPAN ]` vs `[ 🔴 SEDANG TRACKING AKTIF ]` vs `[ 🏁 SELESAI ]`.

### 2. 📐 Pengaturan Batas Arena yang Sangat Mudah (Interactive Drag & Resize)
- **8 Handle Titik Tarik Visual**: Batas kotak arena Open Field kini memiliki 8 titik handle (4 sudut dan 4 tepi) di atas video.
  - Cukup klik dan tarik (*drag*) titik sudut mana saja untuk memperbesar/memperkecil arena.
  - Klik dan tarik bagian tengah untuk memindahkan (*move*) seluruh arena.
  - Kursor mouse otomatis berubah (*diagonal resize, horizontal/vertical resize, move*) saat diarahkan ke handle.
- **Tombol "🖱️ Gambar Ulang Arena (1-Drag)"**: Klik tombol ini lalu klik-dan-tarik kotak baru di atas video dari sudut ke sudut untuk menentukan batas arena seketika (hanya butuh 1 detik).

### 3. ⭐ Custom Free Zones (Zona Bebas Kustom)
Tersedia tab khusus **Custom Zones** untuk menandai area bebas di luar grid:
- **Pilihan Bentuk**:
  - **Kotak (Rectangle)**: Klik & tarik kotak di layar.
  - **Lingkaran (Circle)**: Klik titik tengah & tarik radius.
  - **Poligon Bebas (Polygon)**: Klik titik-titik sudut, klik kanan untuk selesai.
- **Nama & Warna Kustom**: Beri nama zona bebas (misal: `"Objek A"`, `"Objek B"`, `"Shelter"`, `"Makanan"`).
- **Statistik Khusus**: Waktu tinggal (*duration & %*), jumlah masuk (*entries*), dan latensi pertama kali masuk otomatis tercatat di HUD, tabel ringkasan, dan ekspor CSV.

### 4. 📹 Koneksi Kamera yang Andal (Auto-Discovery DirectShow)
- **Dialog Deteksi Otomatis**: Klik tombol **"📹 Sambungkan Kamera"** untuk membuka jendela pemilihan kamera.
- Sistem otomatis mendeteksi seluruh kamera perangkat (seperti kamera bawaan laptop `ACER HD User Facing` maupun webcam USB overhead eksternal).
- **Pilihan Resolusi**: Mendukung resolusi standar `640x480` atau `1280x720 HD`.
- Menggunakan engine DirectShow streaming native yang stabil dan bebas hambatan indeks.

### 5. 🏁 Open Field Grid Kotak (N x M)
- Tentukan pembagian petak bebas: **3x3**, **4x4**, **5x5**, dll.
- **⚡ Auto-Label 1-Klik**: Menandai otomatis 4 sudut sebagai **Corner** (Merah), sisi dinding luar sebagai **Periphery** (Kuning), dan ruang dalam sebagai **Center** (Hijau).
- **🖌️ Kuas Petak Interaktif**: Klik petak mana saja di atas video untuk mengganti labelnya secara manual.

### 6. 🧪 Presets Arena Neurosains Lainnya
- **Morris Water Maze (MWM)**: 4 Kuadran (*NW, NE, SW, SE*) dan lingkaran *Target Platform*.
- **Elevated Plus Maze (EPM)**: Lengan terbuka (*Open Arms*), lengan tertutup (*Closed Arms*), dan *Center*.

### 7. 📊 Ekspor Data Lengkap
- **Ekspor Ringkasan CSV**: Termasuk metrik keseluruhan, metrik per zona grid, dan metrik **Custom Free Zones**.
- **Ekspor Koordinat Mentah CSV**: Data per-frame (Frame, Detik, X, Y, Kecepatan, Petak, Zona, Custom_Zones, Freezing).
- **Ekspor Gambar PNG**: Gambar resolusi tinggi untuk lintasan (*trajectory*) dan peta panas (*heatmap*).

---

## 📁 Struktur File

```
e:/tttt/
├── run.bat             # Peluncur cepat (klik dua kali untuk membuka)
├── main.py             # Titik masuk aplikasi
├── gui_main.py         # Antarmuka desktop PyQt5 lengkap (Dark Scientific Theme)
├── arena_grid.py       # Logika Grid Open Field (N x M, auto-label, drag bounds)
├── arena_presets.py    # Presets Morris Water Maze & Elevated Plus Maze
├── custom_zones.py     # Logika Custom Free Zones (poligon, kotak, lingkaran)
├── camera_manager.py   # Deteksi dan streaming kamera DirectShow/OpenCV
├── tracker_engine.py   # Mesin computer vision OpenCV (Centroid, Smoothing, Freezing, Heatmap)
├── simulation.py       # Generator simulasi mencit virtual realistis
├── export_manager.py   # Ekspor laporan CSV dan gambar PNG
├── test_core.py        # Pengujian backend otomatis
└── README.md           # Panduan lengkap
```
