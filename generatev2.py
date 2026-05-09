import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

random.seed(42)
np.random.seed(42)

# =====================================
# DATABASE DATASET (Bisa dipindah ke JSON/CSV)
# =====================================
# Struktur ini memudahkan penambahan matakuliah baru tanpa menyentuh logika kode
KNOWLEDGE_BASE = {
    "Struktur Data": [
        {
            "question": "Jelaskan pengertian linked list",
            "reference_answer": "Linked list adalah struktur data linear yang terdiri dari node dimana setiap node menyimpan data dan pointer ke node berikutnya."
        },
        {
            "question": "Apa fungsi stack dalam struktur data",
            "reference_answer": "Stack adalah struktur data LIFO dimana elemen terakhir masuk akan keluar pertama."
        },
        {
            "question": "Jelaskan perbedaan antara array dan linked list",
            "reference_answer": "Array memiliki ukuran tetap dan elemen disimpan secara berurutan, sedangkan linked list memiliki ukuran dinamis dan elemen disimpan secara acak dengan pointer."
        },
        {
            "question": "Apa itu queue dalam struktur data?",
            "reference_answer": "Queue adalah struktur data FIFO dimana elemen pertama masuk akan keluar pertama."
        },
        {
            "question": "Jelaskan konsep tree dalam struktur data",
            "reference_answer": "Tree adalah struktur data hierarkis yang terdiri dari node dengan satu atau lebih anak, dimana setiap node memiliki satu induk kecuali root yang tidak memiliki induk." 
        },
        {
            "question": "Apa itu graph dalam struktur data?",
            "reference_answer": "Graph adalah struktur data yang terdiri dari simpul (node) dan sisi (edge) yang menghubungkan simpul-simpul tersebut, digunakan untuk merepresentasikan hubungan antar objek."
        },
        {
            "question": "Jelaskan perbedaan antara binary tree dan binary search tree",
            "reference_answer": "Binary tree adalah struktur data dimana setiap node memiliki maksimal dua anak, sedangkan binary search tree adalah binary tree yang memenuhi aturan dimana nilai anak kiri lebih kecil dari induk dan nilai anak kanan lebih besar dari induk."   
        },
        {
            "question": "Apa itu hash table dalam struktur data?",
            "reference_answer": "Hash table adalah struktur data yang menggunakan fungsi hash untuk memetakan kunci ke indeks dalam array, memungkinkan pencarian, penyisipan, dan penghapusan yang cepat."
        },
        {
            "question": "Jelaskan konsep heap dalam struktur data",
            "reference_answer": "Heap adalah struktur data berbentuk pohon yang memenuhi sifat heap dimana setiap node memiliki nilai yang lebih besar (max heap) atau lebih kecil (min heap) daripada anaknya."
        },
        {
            "question": "Apa itu linked list ganda?",
            "reference_answer": "Linked list ganda adalah jenis linked list dimana setiap node memiliki dua pointer, satu untuk node berikutnya dan satu untuk node sebelumnya."
        }
    ],
    "Web Programming": [
        {
            "question": "Apa fungsi HTML",
            "reference_answer": "HTML digunakan untuk membangun struktur halaman web."
        },
        {
            "question": "Jelaskan fungsi CSS",
            "reference_answer": "CSS digunakan untuk mengatur tampilan dan layout halaman web."
        },
        {
            "question": "Apa itu JavaScript dalam web programming?",
            "reference_answer": "JavaScript adalah bahasa pemrograman yang digunakan untuk membuat halaman web interaktif."
        },
        {
            "question": "Jelaskan perbedaan antara frontend dan backend development",
            "reference_answer": "Frontend development fokus pada tampilan dan interaksi pengguna, sedangkan backend development fokus pada logika server, database, dan integrasi sistem."
        },
        {
            "question": "Apa itu API dalam konteks web programming?",
            "reference_answer": "API (Application Programming Interface) adalah seperangkat aturan yang memungkinkan aplikasi untuk berkomunikasi satu sama lain."
        },
        {
            "question": "Jelaskan konsep responsive design dalam web programming",
            "reference_answer": "Responsive design adalah pendekatan dalam web design yang membuat halaman web dapat menyesuaikan tampilannya dengan berbagai ukuran layar dan perangkat."
        },
        {
            "question": "Apa itu framework dalam web programming?",
            "reference_answer": "Framework adalah kerangka kerja yang menyediakan struktur dan alat untuk mempermudah pengembangan aplikasi web, seperti React, Angular, atau Django."  
        },
        {
            "question": "Jelaskan perbedaan antara REST dan GraphQL",
            "reference_answer": "REST adalah arsitektur API yang menggunakan HTTP untuk komunikasi, sedangkan GraphQL adalah bahasa query untuk API yang memungkinkan klien untuk meminta data yang spesifik dan mengurangi jumlah permintaan yang diperlukan."  
        },
        {
            "question": "Apa itu AJAX dalam web programming?",
            "reference_answer": "AJAX (Asynchronous JavaScript and XML) adalah teknik untuk membuat halaman web interaktif dengan memuat data secara asinkron tanpa perlu memuat ulang halaman."
        }   
    ],
    "Basis Data": [ 
        { 
            "question": "Apa itu basis data?", 
            "reference_answer": "Basis data adalah sekumpulan data yang sistematis dan terorganisir yang disimpan dalam komputer." 
        },
        {
            "question": "Jelaskan perbedaan antara SQL dan NoSQL",
            "reference_answer": "SQL adalah bahasa query untuk basis data relasional, sedangkan NoSQL adalah jenis basis data yang tidak menggunakan tabel relasional."
        }
    ], 
    "Algoritma": [
        {
            "question": "Apa itu algoritma sorting?",
            "reference_answer": "Algoritma sorting adalah metode untuk mengurutkan elemen dalam sebuah list atau array."
        },
        {
            "question": "Jelaskan perbedaan antara algoritma greedy dan dynamic programming",
            "reference_answer": "Algoritma greedy membuat keputusan optimal pada setiap langkah, sedangkan dynamic programming menyelesaikan masalah dengan memecahnya menjadi submasalah yang lebih kecil dan menyimpan hasilnya."
        },
        {
            "question": "Apa itu algoritma divide and conquer?",
            "reference_answer": "Algoritma divide and conquer adalah teknik pemecahan masalah yang membagi masalah menjadi submasalah yang lebih kecil, menyelesaikannya secara rekursif, dan menggabungkan hasilnya."
        },
        {
            "question": "Jelaskan konsep algoritma backtracking",
            "reference_answer": "Algoritma backtracking adalah teknik untuk menyelesaikan masalah dengan mencoba solusi secara sistematis dan mundur jika solusi tersebut tidak berhasil."
        },
        {
            "question": "Apa itu algoritma brute force?",
            "reference_answer": "Algoritma brute force adalah metode untuk menyelesaikan masalah dengan mencoba semua kemungkinan solusi secara sistematis."
        },
        {
            "question": "Jelaskan perbedaan antara algoritma rekursif dan iteratif",
            "reference_answer": "Algoritma rekursif memanggil dirinya sendiri untuk menyelesaikan masalah, sedangkan algoritma iteratif menggunakan loop untuk menyelesaikan masalah." 
        },
        {
            "question": "Apa itu algoritma greedy?",
            "reference_answer": "Algoritma greedy adalah metode untuk menyelesaikan masalah dengan membuat keputusan optimal pada setiap langkah tanpa mempertimbangkan konsekuensi jangka panjang."
        },
        {
            "question": "Jelaskan konsep algoritma dynamic programming",
            "reference_answer": "Algoritma dynamic programming adalah teknik untuk menyelesaikan masalah dengan memecahnya menjadi submasalah yang lebih kecil, menyimpan hasilnya, dan menghindari perhitungan ulang."
        },
        {
            "question": "Apa itu algoritma graph traversal?",
            "reference_answer": "Algoritma graph traversal adalah metode untuk mengunjungi semua node dalam sebuah graph, seperti Depth-First Search (DFS) dan Breadth-First Search (BFS)."
        }
    ],
    "Jaringan Komputer": [
        {
            "question": "Apa itu protokol TCP/IP?",
            "reference_answer": "TCP/IP adalah protokol komunikasi yang digunakan untuk menghubungkan perangkat di internet."
        },
        {
            "question": "Jelaskan fungsi DNS dalam jaringan komputer",
            "reference_answer": "DNS (Domain Name System) berfungsi untuk menerjemahkan nama domain menjadi alamat IP yang dapat dipahami oleh komputer."
        },
        {
            "question": "Apa itu firewall dalam jaringan komputer?",
            "reference_answer": "Firewall adalah sistem keamanan jaringan yang memantau dan mengontrol lalu lintas jaringan berdasarkan aturan keamanan yang telah ditetapkan."
        },
        {
            "question": "Jelaskan perbedaan antara IPv4 dan IPv6",
            "reference_answer": "IPv4 menggunakan alamat 32-bit yang memungkinkan sekitar 4,3 miliar alamat, sedangkan IPv6 menggunakan alamat 128-bit yang memungkinkan jumlah alamat yang jauh lebih besar."
        },
        {
            "question": "Apa itu VPN dalam jaringan komputer?",
            "reference_answer": "VPN (Virtual Private Network) adalah teknologi yang memungkinkan koneksi aman ke jaringan lain melalui internet dengan mengenkripsi data yang dikirimkan."
        },
        {
            "question": "Jelaskan konsep subnetting dalam jaringan komputer",
            "reference_answer": "Subnetting adalah teknik untuk membagi jaringan IP menjadi subnet yang lebih kecil untuk meningkatkan efisiensi dan keamanan jaringan."
        },
        {
            "question": "Apa itu router dalam jaringan komputer?",
            "reference_answer": "Router adalah perangkat jaringan yang menghubungkan beberapa jaringan dan meneruskan data antar jaringan tersebut."
        },
        {
            "question": "Jelaskan perbedaan antara switch dan hub",
            "reference_answer": "Switch adalah perangkat jaringan yang menghubungkan perangkat dalam jaringan dan mengirimkan data hanya ke perangkat yang dituju, sedangkan hub mengirimkan data ke semua perangkat dalam jaringan."
        },
        {
            "question": "Apa itu protokol HTTP?",
            "reference_answer": "HTTP (Hypertext Transfer Protocol) adalah protokol komunikasi yang digunakan untuk mentransfer data di web, memungkinkan komunikasi antara klien dan server."
        }
    ],
    "Pemrograman Berorientasi Objek": [
        {
            "question": "Apa itu kelas dalam OOP?",
            "reference_answer": "Kelas adalah blueprint atau template untuk membuat objek dalam pemrograman berorientasi objek."
        },
        {
            "question": "Jelaskan konsep inheritance dalam OOP",
            "reference_answer": "Inheritance adalah konsep dimana sebuah kelas dapat mewarisi sifat dan perilaku dari kelas lain."
        },
        {
            "question": "Apa itu polymorphism dalam OOP?",
            "reference_answer": "Polymorphism adalah kemampuan objek untuk mengambil banyak bentuk, memungkinkan metode yang sama untuk digunakan pada objek yang berbeda."
        },
        {
            "question": "Jelaskan perbedaan antara class dan object",
            "reference_answer": "Class adalah blueprint untuk membuat objek, sedangkan object adalah instance dari class yang memiliki data dan perilaku."
        },
        {
            "question": "Apa itu encapsulation dalam OOP?",
            "reference_answer": "Encapsulation adalah konsep untuk menyembunyikan data dan hanya memberikan akses melalui metode tertentu untuk melindungi integritas data."
        },
        {
            "question": "Jelaskan konsep abstraction dalam OOP",
            "reference_answer": "Abstraction adalah proses menyembunyikan detail implementasi dan hanya menampilkan fitur penting dari suatu objek."
        },
        {
            "question": "Apa itu method overloading dalam OOP?",
            "reference_answer": "Method overloading adalah kemampuan untuk memiliki beberapa metode dengan nama yang sama tetapi dengan parameter yang berbeda dalam satu kelas."
        },
        {
            "question": "Jelaskan perbedaan antara interface dan abstract class",
            "reference_answer": "Interface hanya mendefinisikan metode tanpa implementasi, sedangkan abstract class dapat memiliki metode dengan atau tanpa implementasi."
        },
        {
            "question": "Apa itu constructor dalam OOP?",
            "reference_answer": "Constructor adalah metode khusus yang dipanggil saat objek dibuat untuk menginisialisasi nilai-nilai awal dari objek tersebut."
        }
    ],
    "Sistem Operasi": [
        {
            "question": "Apa fungsi dari sistem operasi?",
            "reference_answer": "Sistem operasi adalah perangkat lunak yang mengelola perangkat keras komputer dan menyediakan layanan untuk program komputer."
        },
        {
            "question": "Jelaskan perbedaan antara proses dan thread",
            "reference_answer": "Proses adalah program yang sedang berjalan, sedangkan thread adalah unit terkecil dari proses yang dapat dijalankan secara independen."
        },
        {
            "question": "Apa itu virtual memory dalam sistem operasi?",
            "reference_answer": "Virtual memory adalah teknik yang memungkinkan komputer untuk menggunakan ruang penyimpanan sebagai perpanjangan dari RAM, memungkinkan program untuk berjalan meskipun memori fisik terbatas."
        },
        {
            "question": "Jelaskan konsep deadlock dalam sistem operasi",
            "reference_answer": "Deadlock adalah situasi dimana dua atau lebih proses saling menunggu sumber daya yang sedang digunakan oleh proses lain, sehingga tidak ada proses yang dapat melanjutkan."
        },
        {
            "question": "Apa itu scheduling dalam sistem operasi?",
            "reference_answer": "Scheduling adalah proses penjadwalan eksekusi proses atau thread oleh sistem operasi untuk memastikan penggunaan sumber daya yang efisien."
        },
        {
            "question": "Jelaskan perbedaan antara preemptive dan non-preemptive scheduling",
            "reference_answer": "Preemptive scheduling memungkinkan sistem operasi untuk menghentikan proses yang sedang berjalan untuk memberikan kesempatan kepada proses lain, sedangkan non-preemptive scheduling tidak mengizinkan interupsi pada proses yang sedang berjalan."
        },
        {
            "question": "Apa itu file system dalam sistem operasi?",
            "reference_answer": "File system adalah metode dan struktur data yang digunakan oleh sistem operasi untuk mengelola dan menyimpan file di media penyimpanan."
        },
        {
            "question": "Jelaskan konsep paging dalam sistem operasi",
            "reference_answer": "Paging adalah teknik manajemen memori dimana memori fisik dibagi menjadi blok-blok kecil yang disebut halaman, dan memori virtual dibagi menjadi blok-blok yang sama, memungkinkan program untuk menggunakan lebih banyak memori daripada yang tersedia secara fisik."
        },
        {
            "question": "Apa itu kernel dalam sistem operasi?",
            "reference_answer": "Kernel adalah inti dari sistem operasi yang bertanggung jawab untuk mengelola sumber daya komputer dan menyediakan layanan dasar untuk aplikasi."
        }
    ]
}

# =====================================
# VARIATION GENERATOR
# =====================================
def generate_student_answer(reference):
    words = reference.split()
    mode = random.choice(["excellent", "good", "medium", "poor", "copy"])

    if mode == "excellent":
        extra = random.choice([
            "Contohnya sering digunakan pada implementasi aplikasi.",
            "Struktur ini sangat penting dalam pemrograman.",
            "Biasanya digunakan dalam sistem komputer."
        ])
        return reference + " " + extra
    elif mode == "good":
        keep = random.sample(words, max(3, len(words) - 3))
        return " ".join(keep)
    elif mode == "medium":
        keep = random.sample(words, max(3, len(words) // 2))
        typo = random.choice(["jadi", "digunakan", "biasanya"])
        return " ".join(keep) + " " + typo
    elif mode == "poor":
        return random.choice(["kurang paham", "struktur data penting", "digunakan di program", "untuk coding"])
    else:
        return reference

# =====================================
# DATASET ENGINE (PERUBAHAN UTAMA)
# =====================================
def generate_dataset_for_course(course_name, num_samples_per_question=5):
    """
    Menghasilkan dataset hanya untuk satu mata kuliah tertentu.
    """
    if course_name not in KNOWLEDGE_BASE:
        print(f"Mata kuliah {course_name} tidak ditemukan dalam database.")
        return None

    questions_list = KNOWLEDGE_BASE[course_name]
    generated_data = []

    for item in questions_list:
        q = item['question']
        ref = item['reference_answer']

        for _ in range(num_samples_per_question):
            student_ans = generate_student_answer(ref)
            nscore = final_scoring(q, ref, student_ans)
            generated_data.append({
                "course": course_name,
                "question": q,
                "reference_answer": ref,
                "student_answer": student_ans,
                "instructor_score": nscore  # Placeholder untuk skor LLM nanti
            })

    return pd.DataFrame(generated_data)

# =====================================
# FEATURE SCORING
# =====================================
def similarity_score(reference, essay):
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform([reference, essay])

    sim = cosine_similarity(
        tfidf[0:1],
        tfidf[1:2]
    )[0][0]

    return sim * 100


def rubric_score(question, reference, essay):
    essay_words = essay.lower().split()
    ref_words = reference.lower().split()
    q_words = question.lower().split()

    overlap = len(
        set(essay_words) &
        set(ref_words)
    )

    coverage = len(
        set(essay_words) &
        set(q_words)
    )

    vocab_diversity = (
        len(set(essay_words)) /
        max(len(essay_words), 1)
    ) * 100

    length_score = min(
        len(essay_words) * 3,
        100
    )

    score = (
        overlap * 5 +
        coverage * 10 +
        vocab_diversity * 0.2 +
        length_score * 0.2
    )

    return min(score, 100)


def final_scoring(question, reference, essay):
    sim = similarity_score(reference, essay)

    rub = rubric_score(
        question,
        reference,
        essay
    )

    coverage = min(
        len(
            set(question.lower().split()) &
            set(essay.lower().split())
        ) * 20,
        100
    )

    length = min(
        len(essay.split()) * 4,
        100
    )

    final = (
        0.4 * sim +
        0.3 * rub +
        0.2 * coverage +
        0.1 * length
    )

    noise = random.uniform(-3, 3)

    return max(0, min(100, final + noise))

# =====================================
# IMPLEMENTASI EKSEKUSI + SAVE CSV
# =====================================

import os
import math

# =====================================
# CONFIG
# =====================================

OUTPUT_DIR = "data"

# Jumlah data per mata kuliah
JML_DATA = 1000

# =====================================
# CREATE FOLDER
# =====================================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# =====================================
# GENERATE DATASET PER COURSE
# =====================================

for course in KNOWLEDGE_BASE.keys():

    print(f"\nGenerating dataset for: {course}")

    questions_count = len(KNOWLEDGE_BASE[course])

    # Hitung jumlah sample per pertanyaan
    sample_per_question = math.ceil(
        JML_DATA / questions_count
    )

    df_course = generate_dataset_for_course(
        course_name=course,
        num_samples_per_question=sample_per_question
    )

    # Batasi tepat JML_DATA
    df_course = df_course.iloc[:JML_DATA]

    # Nama file aman
    safe_name = (
        course.lower()
        .replace(" ", "-")
        .replace("/", "-")
    )

    file_path = os.path.join(
        OUTPUT_DIR,
        f"dataset-{safe_name}.csv"
    )

    # Save CSV
    df_course.to_csv(
        file_path,
        index=False
    )

    print(f"Saved: {file_path}")
    print(f"Total rows: {len(df_course)}")

# =====================================
# DONE
# =====================================

print("\n====================================")
print("ALL DATASETS GENERATED SUCCESSFULLY")
print("====================================")
print(f"Dataset per course: {JML_DATA} rows")