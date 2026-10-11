
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
# 3. FUNGSI UNTUK MELAKUKAN FILTER
# ==================================================

def proses_filter(img, kernel):

    # Low-Pass Filter (LPF)

    # Mean Blur
    blur = cv2.blur(img, (kernel, kernel))

    # Gaussian Blur
    gaussian = cv2.GaussianBlur(
        img, (kernel, kernel), 0
    )

    # Median Blur
    median = cv2.medianBlur(img, kernel)


    # High-Pass Filter (HPF) menggunakan Sobel

    # Sobel Derivative X
    gx = cv2.Sobel(
        img, cv2.CV_64F, 1, 0, ksize=kernel
    )

    sobel_x = cv2.convertScaleAbs(gx)

    # Sobel Derivative Y
    gy = cv2.Sobel(
        img, cv2.CV_64F, 0, 1, ksize=kernel
    )

    sobel_y = cv2.convertScaleAbs(gy)

    # Sobel Magnitude
    magnitude_raw = np.sqrt(gx**2 + gy**2)

    magnitude = cv2.normalize(
        magnitude_raw,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Mengembalikan 7 gambar
    hasil = [
        img,
        blur,
        gaussian,
        median,
        sobel_x,
        sobel_y,
        magnitude
    ]

    return hasil


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
# 5. FUNGSI MENAMPILKAN DAN MENYIMPAN GAMBAR GABUNGAN
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

    # Mematikan subplot kedelapan
    axes[7].axis("off")

    fig.suptitle(
        f"Hasil Pengolahan Citra Kernel {kernel}x{kernel}",
        fontsize=14
    )

    plt.tight_layout()

    # Menyimpan gambar gabungan
    fig.savefig(
        nama_output,
        dpi=300,
        bbox_inches="tight"
    )

    # Menampilkan gambar gabungan
    plt.show()

    plt.close(fig)


# ==================================================
# 6. PROSES KERNEL 3x3
# ==================================================

hasil_3x3 = proses_filter(img, 3)

# Nama file individual khusus kernel 3x3
nama_file_3x3 = [
    "gambar_asli.png",
    "blur_3x3.png",
    "gaussian_3x3.png",
    "median_3x3.png",
    "sobel_x_3x3.png",
    "sobel_y_3x3.png",
    "magnitude_3x3.png"
]

# Menyimpan 7 gambar individual
for i in range(7):

    path = os.path.join(
        "hasil_3x3",
        nama_file_3x3[i]
    )

    cv2.imwrite(path, hasil_3x3[i])

# Menyimpan satu gambar gabungan 3x3
simpan_gabungan(
    hasil_3x3,
    3,
    "hasil_3x3/gabungan_3x3.png"
)

# Menampilkan gambar individual menggunakan OpenCV
for i in range(7):
    cv2.imshow(
        "3x3 - " + judul[i],
        hasil_3x3[i]
    )


# ==================================================
# 7. PROSES KERNEL 5x5
# ==================================================

hasil_5x5 = proses_filter(img, 5)

# Hanya menyimpan satu gambar gabungan 5x5
# Tidak menyimpan gambar 5x5 secara individual
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
   Gambar grayscale menjadi acuan untuk membandingkan
   hasil dari setiap metode filter.

2. LPF - Blur
   Merata-ratakan nilai piksel di sekitarnya sehingga
   gambar menjadi lebih halus. Kernel 5x5 biasanya
   menghasilkan penghalusan lebih kuat dibandingkan 3x3.

3. LPF - Gaussian Blur
   Menghaluskan gambar menggunakan pembobotan Gaussian.
   Kernel 5x5 biasanya mengurangi detail kecil lebih banyak
   daripada kernel 3x3.

4. LPF - Median Blur
   Mengganti nilai piksel dengan median dari lingkungan
   sekitarnya. Kernel 5x5 mempertimbangkan area lebih luas
   dibandingkan kernel 3x3.

5. HPF - Sobel X
   Mendeteksi perubahan intensitas pada arah X.
   Tepi vertikal cenderung terlihat lebih jelas.

6. HPF - Sobel Y
   Mendeteksi perubahan intensitas pada arah Y.
   Tepi horizontal cenderung terlihat lebih jelas.

7. HPF - Magnitude
   Menggabungkan kekuatan gradien X dan Y untuk
   menunjukkan tepi dari berbagai arah.

8. Perbandingan kernel 3x3 dan 5x5
   Kernel 3x3 menggunakan area yang lebih kecil,
   sedangkan kernel 5x5 menggunakan area lebih luas.
   Pada LPF, kernel yang lebih besar biasanya memberikan
   penghalusan lebih kuat. Pada Sobel, ukuran kernel
   memengaruhi respons terhadap tepi dan detail gambar.

Kesimpulan:
   LPF digunakan untuk menghaluskan gambar, sedangkan
   Sobel digunakan untuk mendeteksi tepi. Perbandingan
   kernel 3x3 dan 5x5 menunjukkan pengaruh ukuran kernel
   terhadap hasil pengolahan citra.
""")

print("\nSemua proses selesai!")
print("\nFolder hasil_3x3:")
print("- gabungan_3x3.png")
print("- 7 file gambar individual")

print("\nFolder hasil_5x5:")
print("- gabungan_5x5.png saja")


# ==================================================
# 9. MENUTUP JENDELA OPENCV
# ==================================================

cv2.waitKey(0)
cv2.destroyAllWindows()
