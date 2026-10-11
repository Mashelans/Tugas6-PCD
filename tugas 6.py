```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==================================================
# 1. MEMBACA GAMBAR GRAYSCALE
# ==================================================

nama_file = "gambar 2.jpg"

if not os.path.exists(nama_file):
    nama_file = "gambar 2.jpeg"

img = cv2.imread(nama_file, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(
        "File gambar 2.jpg atau gambar 2.jpeg tidak ditemukan!"
    )

print("Gambar berhasil dibaca.")


# ==================================================
# 2. MEMBUAT FOLDER OUTPUT
# ==================================================

os.makedirs("hasil_3x3", exist_ok=True)
os.makedirs("hasil_5x5", exist_ok=True)


# ==================================================
# 3. FUNGSI FILTER LPF DAN HPF SOBEL
# ==================================================

def proses_filter(img, kernel):

    # ------------------------------
    # LOW-PASS FILTER (LPF)
    # ------------------------------

    # Average Blur
    blur = cv2.blur(img, (kernel, kernel))

    # Gaussian Blur
    gaussian = cv2.GaussianBlur(
        img, (kernel, kernel), 0
    )

    # Median Blur
    median = cv2.medianBlur(img, kernel)


    # ------------------------------
    # HIGH-PASS FILTER (HPF)
    # SOBEL X DAN SOBEL Y
    # ------------------------------

    # Sobel X
    gx = cv2.Sobel(
        img,
        cv2.CV_64F,
        1,
        0,
        ksize=kernel
    )

    sobel_x = cv2.convertScaleAbs(gx)

    # Sobel Y
    gy = cv2.Sobel(
        img,
        cv2.CV_64F,
        0,
        1,
        ksize=kernel
    )

    sobel_y = cv2.convertScaleAbs(gy)


    # ------------------------------
    # SOBEL MAGNITUDE
    # ------------------------------

    # Menghitung kekuatan gradien
    magnitude_raw = cv2.magnitude(
        gx.astype(np.float32),
        gy.astype(np.float32)
    )

    # Konversi ke citra 8-bit
    # Nilai absolut, skala default 1, lalu dibatasi 0-255
    magnitude = cv2.convertScaleAbs(magnitude_raw)

    # Mengembalikan tujuh gambar
    return [
        img,
        blur,
        gaussian,
        median,
        sobel_x,
        sobel_y,
        magnitude
    ]


# ==================================================
# 4. JUDUL GAMBAR
# ==================================================

judul = [
    "Gambar Asli",
    "LPF - Blur",
    "LPF - Gaussian Blur",
    "LPF - Median Blur",
    "HPF - Sobel X",
    "HPF - Sobel Y",
    "HPF - Magnitude"
]


# ==================================================
# 5. FUNGSI MENYIMPAN GAMBAR GABUNGAN
# ==================================================

def simpan_gabungan(hasil, kernel, nama_output):

    fig, axes = plt.subplots(
        2, 4, figsize=(16, 8)
    )

    axes = axes.flatten()

    for i in range(7):
        axes[i].imshow(
            hasil[i],
            cmap="gray",
            vmin=0,
            vmax=255
        )

        axes[i].set_title(judul[i])
        axes[i].axis("off")

    # Mengosongkan subplot kedelapan
    axes[7].axis("off")

    fig.suptitle(
        f"Hasil Pengolahan Citra Kernel {kernel}x{kernel}",
        fontsize=14
    )

    plt.tight_layout()

    # Simpan gambar gabungan
    fig.savefig(
        nama_output,
        dpi=300,
        bbox_inches="tight"
    )

    # Tampilkan gambar gabungan
    plt.show()

    plt.close(fig)


# ==================================================
# 6. PROSES KERNEL 3x3
# ==================================================

hasil_3x3 = proses_filter(img, 3)

nama_file_3x3 = [
    "gambar_asli.png",
    "blur_3x3.png",
    "gaussian_3x3.png",
    "median_3x3.png",
    "sobel_x_3x3.png",
    "sobel_y_3x3.png",
    "magnitude_3x3.png"
]

# Simpan tujuh gambar individual
for i in range(7):

    path = os.path.join(
        "hasil_3x3",
        nama_file_3x3[i]
    )

    cv2.imwrite(path, hasil_3x3[i])

# Simpan gambar gabungan 3x3
simpan_gabungan(
    hasil_3x3,
    3,
    "hasil_3x3/gabungan_3x3.png"
)


# ==================================================
# 7. PROSES KERNEL 5x5
# ==================================================

hasil_5x5 = proses_filter(img, 5)

nama_file_5x5 = [
    "gambar_asli.png",
    "blur_5x5.png",
    "gaussian_5x5.png",
    "median_5x5.png",
    "sobel_x_5x5.png",
    "sobel_y_5x5.png",
    "magnitude_5x5.png"
]

# Simpan tujuh gambar individual
for i in range(7):

    path = os.path.join(
        "hasil_5x5",
        nama_file_5x5[i]
    )

    cv2.imwrite(path, hasil_5x5[i])

# Simpan gambar gabungan 5x5
simpan_gabungan(
    hasil_5x5,
    5,
    "hasil_5x5/gabungan_5x5.png"
)


# ==================================================
# 8. ANALISIS HASIL FILTER
# ==================================================

print("\n========== ANALISIS HASIL FILTER ==========")

print("""
1. Gambar asli
   Gambar grayscale digunakan sebagai acuan
   untuk membandingkan hasil setiap metode filter.

2. LPF - Blur
   Merata-ratakan nilai piksel di sekitarnya
   sehingga gambar menjadi lebih halus.

3. LPF - Gaussian Blur
   Menghaluskan gambar menggunakan pembobotan
   Gaussian untuk mengurangi detail kecil.

4. LPF - Median Blur
   Mengganti nilai piksel dengan median dari
   lingkungan sekitarnya.

5. HPF - Sobel X
   Mendeteksi perubahan intensitas arah horizontal
   sehingga tepi vertikal cenderung terlihat jelas.

6. HPF - Sobel Y
   Mendeteksi perubahan intensitas arah vertikal
   sehingga tepi horizontal cenderung terlihat jelas.

7. HPF - Magnitude
   Menggabungkan gradien Sobel X dan Y untuk
   menunjukkan kekuatan tepi dari berbagai arah.
   Hasil dikonversi ke citra 8-bit menggunakan
   cv2.convertScaleAbs().

8. Perbandingan kernel 3x3 dan 5x5
   Kernel 3x3 menggunakan area yang lebih kecil,
   sedangkan kernel 5x5 menggunakan area lebih luas.
   Ukuran kernel memengaruhi penghalusan gambar
   dan respons deteksi tepi.

Kesimpulan:
   LPF digunakan untuk menghaluskan gambar,
   sedangkan Sobel digunakan untuk mendeteksi tepi.
   Hasil kedua ukuran kernel dapat dibandingkan
   melalui gambar gabungan yang disimpan terpisah.
""")

print("\nSemua proses selesai.")
print("Folder hasil_3x3 berisi tujuh gambar individual")
print("dan satu gambar gabungan.")
print("Folder hasil_5x5 berisi tujuh gambar individual")
print("dan satu gambar gabungan.")


# ==================================================
# 9. MENUTUP JENDELA OPENCV
# ==================================================

cv2.waitKey(0)
cv2.destroyAllWindows()
```

