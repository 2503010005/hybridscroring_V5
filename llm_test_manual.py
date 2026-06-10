from ollama import chat
import re

nilai = 0
prompt = f"""
Anda adalah seorang dosen yang sedang memberikan nilai untuk jawaban mahasiswa. Berikut adalah soal, kunci jawaban, dan jawaban mahasiswa pada jurusan Ilmu komputer/Teknik Informatika. Berikan penilaian terhadap jawaban mahasiswa berikut dengan skor antara 10-100 berdasarkan kriteria berikut:
1. Kesesuaian logika konsep (40%)
2. Kelengkapan jawaban (30%)
3. Relevansi dengan kunci (20%)
4. Struktur penjelasan (10%)

Mata Kuliah:
Basis Data

Soal:
Jelaskan perbedaan antara SQL dan NoSQL

Kunci:
SQL adalah bahasa query untuk basis data relasional, sedangkan NoSQL adalah jenis basis data yang tidak menggunakan tabel relasional.

Jawaban:
relasional, relasional. SQL data tabel untuk basis sedangkan tidak digunakan

Instruktor_scoring:
58

intruksi:
beri point 5-100 untuk jawaban di atas beserta alasan singkatnya
"""

response = chat(
    model='all-minilm:l6-v2',  # Ganti dengan model yang Anda gunakan
    messages=[{'role': 'user', 'content': prompt}],
)
text = response.message.content
match = re.search(r'\d+', text)
if match:
    score = int(match.group())
    nilai = max(0, min(score, 100))

print(f"LLM Response: {text}")
print(f"LLM Score: {nilai}")
