# rubric_engine.py
"""
Modul: rubric_engine.py
Deskripsi: Komponen Penilai Berbasis Rubrik (Pillar 3) dalam Sistem Fusi Hibrida Adaptif.
           Berfungsi menghitung rata-rata skalar dari kriteria penilaian instruksional,
           baik menggunakan data acuan RPS (statis/fallback) maupun parameter dinamis dari client.
"""

# Matriks Acuan Standar RPS Struktur Data - Quis 1
# Digunakan sebagai jangkar statistik di Bab IV dan fallback jika client tidak mengirimkan rubrik kustom
RUBRIK_DINAMIS_SOAL = {
    1: {
        "landasan_teori_memori": 85.0, 
        "analisis_kompleksitas": 80.0, 
        "kesimpulan_perbandingan": 80.0
    },
    2: {
        "kondisi_keunggulan": 90.0, 
        "alasan_teoritis_pointer": 80.0
    },
    3: {
        "anatomi_node": 85.0, 
        "dampak_overhead_memori": 75.0, 
        "fleksibilitas_traversal": 80.0
    },
    4: {
        "perbedaan_teoritis_cincin": 90.0, 
        "kesesuaian_use_case": 85.0
    },
    5: {
        "over_allocation": 80.0, 
        "proses_penyalinan": 80.0, 
        "manajemen_memori_lama": 75.0
    }
}

def calculate_rubric_score(soal_no: int, custom_rubric: dict = None) -> float:
    try:
        # Jika dosen mengirimkan Rubrik Dinamis kustom via Client (ReactJS / Postman)
        if custom_rubric and isinstance(custom_rubric, dict) and len(custom_rubric) > 0:
            # 🌟 PERBAIKAN KRUSIAL: Konversi paksa setiap nilai menjadi float 
            # untuk mengantisipasi jika React mengirimkan format string numerik ("85.0")
            clean_scores = []
            for val in custom_rubric.values():
                if isinstance(val, str):
                    # Hilangkan spasi jika ada, lalu ubah ke float
                    clean_scores.append(float(val.strip()))
                else:
                    clean_scores.append(float(val))
            
            score = sum(clean_scores) / len(clean_scores)
            return round(float(score), 2)
            
        # Fallback ke Matriks Standar RPS (Jika custom_rubric Kosong/None)
        matriks_rubrik = RUBRIK_DINAMIS_SOAL.get(soal_no)
        if not matriks_rubrik:
            return 0.0

        score = sum(matriks_rubrik.values()) / len(matriks_rubrik)
        return round(float(score), 2)

    except (ZeroDivisionError, ValueError, TypeError) as e:
        print(f"[RUBRIC ERROR] Gagal kalkulasi rubrik pada soal {soal_no}: {str(e)}")
        return 0.0

def get_rubric_details(soal_no: int) -> dict:
    """
    Mengambil rincian kriteria beserta nilai bawaan (default) berdasarkan nomor soal.
    Berguna untuk menyediakan visualisasi feedback awal/transparansi pada antarmuka dashboard.
    
    Parameters:
    -----------
    soal_no : int
        Nomor urut soal yang dicari.
        
    Returns:
    --------
    dict
        Kamus berisi sub-kriteria beserta bobot bawaannya. Mengembalikan dictionary kosong jika tidak ditemukan.
    """
    return RUBRIK_DINAMIS_SOAL.get(soal_no, {})