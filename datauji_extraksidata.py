import pandas as pd
import requests
import os
import re

def clean_and_format_raw_url(repo_url):
    """
    Mengonversi URL browser GitHub mahasiswa menjadi URL mentah (Raw) 
    dan membersihkan noise seperti ekstensi .git, trailing slash, atau path blob.
    """
    url = repo_url.strip()
    if url.endswith('.git'): url = url[:-4]
    if url.endswith('/'): url = url[:-1]
    if '/blob/' in url: url = url.split('/blob/')[0]
    
    raw_url = url.replace("github.com", "raw.githubusercontent.com")
    return raw_url

def fetch_readme_content(base_raw_url, anon_id):
    """Mencoba mengunduh file README.md dari branch main atau master."""
    for branch in ["main", "master"]:
        target_url = f"{base_raw_url}/{branch}/README.md"
        try:
            res = requests.get(target_url, timeout=10)
            if res.status_code == 200:
                return res.text, f"SUCCESS ({branch})"
        except Exception: continue
    return None, "FAILED (404/PRIVATE/TIMEOUT)"

def split_readme_into_answers(readme_text):
    """
    Memecah teks utuh README.md menjadi potongan jawaban terpisah 1-5
    menggunakan toleransi Regex Multi-Format untuk mengantisipasi variasi penulisan mahasiswa.
    """
    answers = {}
    
    # POLA REGEX MULTI-FORMAT (Mendeteksi format angka, tanda kurung, bold, atau judul RPS)
    patterns = [
        r'(?:^|\n)(?:\*\*|\*|)?\[?1[\.\)]\s*(?:Karakteristik|Karakteristik Memori)?',
        r'(?:^|\n)(?:\*\*|\*|)?\[?2[\.\)]\s*(?:Analisis|Analisis Efisiensi)?',
        r'(?:^|\n)(?:\*\*|\*|)?\[?3[\.\)]\s*(?:Konsep|Konsep Doubly)?',
        r'(?:^|\n)(?:\*\*|\*|)?\[?4[\.\)]\s*(?:Mekanisme|Mekanisme Circular)?',
        r'(?:^|\n)(?:\*\*|\*|)?\[?5[\.\)]\s*(?:Array|Array Dinamis)?'
    ]
    
    combined_pattern = '|'.join(patterns)
    segments = re.split(combined_pattern, readme_text, flags=re.IGNORECASE)
    matches = re.findall(combined_pattern, readme_text, flags=re.IGNORECASE)
    
    start_idx = 1 if len(segments) > len(matches) else 0
    
    for i, seg in enumerate(segments[start_idx:]):
        if i < len(matches):
            match_text = matches[i]
            num_match = re.search(r'[1-5]', match_text)
            if num_match:
                num = int(num_match.group())
                answers[num] = seg.strip()
                
    # LAPISAN PENYELAMAT SECONDARY FALLBACK
    for i in range(1, 6):
        if i not in answers or len(answers[i]) < 5:
            secondary_fallback = re.search(f"{i}[\.\)]\s*(.*)", readme_text, re.DOTALL | re.IGNORECASE)
            if secondary_fallback:
                raw_extracted = secondary_fallback.group(1)
                next_num = i + 1
                if next_num <= 5:
                    raw_extracted = re.split(f"{next_num}[\.\)]", raw_extracted, flags=re.IGNORECASE)[0]
                answers[i] = raw_extracted.strip()
            else:
                answers[i] = "Mahasiswa tidak memberikan jawaban atau format penulisan tidak dikenali oleh sistem Parser."
                
    return answers

if __name__ == "__main__":
    print("="*90)
    print(f"{'PROSES INTEGRASI BATCH JAWABAN VERSI REGEX ROBUST (DIREKTORI BARU)':^90}")
    print("="*90)
    
    # PEMBARUAN JALUR FILE SESUAI INSTRUKSI DOSEN
    INPUT_FILE = "datauji/source/source.csv"
    OUTPUT_FILE = "datauji/dataset_uji.csv"
    
    # Antisipasi otomatis pembuatan folder pendukung jika belum ada
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Folder '{output_dir}' otomatis dibuat oleh sistem.")
    
    # 1. Validasi Keberadaan Berkas Input
    if not os.path.exists(INPUT_FILE):
        print(f"❌ ERROR: Berkas '{INPUT_FILE}' tidak ditemukan.")
        print(f"💡 Silakan letakkan file rekap URL Anda di folder: {INPUT_FILE}")
        exit()
        
    df_input = pd.read_csv(INPUT_FILE)
    print(f"📊 Berhasil memuat berkas source. Terdeteksi {len(df_input)} repositori mahasiswa.\n")
    
    all_extracted_records = []
    
    # 2. Loop Penarikan Data Berdasarkan IDMHS & URLGITHUB
    for idx, row in df_input.iterrows():
        # Membaca kolom sesuai berkas mentah Anda
        id_mhs_original = row['IDMHS']
        original_url = row['URLGITHUB']
        
        # Konversi ke format penomoran ID Anonim untuk Tesis (MHS_01, MHS_02, dst.)
        anon_id = f"MHS_{idx+1:02d}"
        
        print(f"⏳ [{idx+1}/{len(df_input)}] Mengekstrak {anon_id} (Source ID: {id_mhs_original})...")
        base_raw_url = clean_and_format_raw_url(original_url)
        readme_text, status_message = fetch_readme_content(base_raw_url, anon_id)
        
        if "SUCCESS" in status_message:
            print(f"   ✔️ Berhasil mengunduh berkas README.md.")
            parsed_answers = split_readme_into_answers(readme_text)
            
            for nomor_soal in range(1, 6):
                all_extracted_records.append({
                    'student_id': anon_id,
                    'soal_no': nomor_soal,
                    'course': "Struktur Data", # Dikunci sesuai penamaan masa training model Anda
                    'student_answer': parsed_answers[nomor_soal],
                    'github_url_source': original_url
                })
        else:
            print(f"   ❌ {status_message}")
            for nomor_soal in range(1, 6):
                all_extracted_records.append({
                    'student_id': anon_id,
                    'soal_no': nomor_soal,
                    'course': "Struktur Data",
                    'student_answer': "[Error] Gagal mengekstrak data dari GitHub.",
                    'github_url_source': original_url
                })
        print("-" * 90)

    # 3. Ekspor Hasil Ekstraksi Presisi ke Target Direktori Akhir
    if all_extracted_records:
        df_output = pd.DataFrame(all_extracted_records)
        df_output.to_csv(OUTPUT_FILE, index=False)
        print("\n" + "="*90)
        print(f"🎉 SELESAI! Berkas dataset uji hasil ekstraksi disimpan di: {OUTPUT_FILE}")
        print(f"📊 Total record tersimpan: {len(df_output)} baris ($12 \\text{{ mhs}} \\times 5 \\text{{ soal}}$).")
        print("="*90)