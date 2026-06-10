import numpy as np
from collections import Counter
import requests  # Untuk memanggil Ollama API secara lokal
from ollama import chat  # Pastikan Anda sudah menginstal Ollama Python SDK 
import config  # File konfigurasi untuk menyimpan pengaturan model dan API

config = Config()

def get_llm_paraphrase(text):
    """
    Memanggil LLM (Gemma) menggunakan library resmi ollama.
    """
    try:
        # Menggunakan model gemma sesuai konfigurasi Anda
        response = ollama.chat(model=config.LLM_MODEL, messages=[
            {
                'role': 'user',
                'content': f"Parafrase kalimat berikut dalam Bahasa Indonesia tanpa mengubah maknanya: {text}",
            },
        ])
        # Mengambil konten pesan dari objek respons
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Error saat augmentasi LLM: {e}")
        return text

def hybrid_augmentation_logic(texts, numeric_features, labels, threshold=5):
    """
    Mengintegrasikan SMOTE (Satria 2023) dalam bentuk Generative Augmentation.
    """
    balanced_texts = list(texts)
    balanced_numeric = list(numeric_features)
    balanced_labels = list(labels)
    
    # Hitung distribusi skor
    counts = Counter(labels)
    
    for label, count in counts.items():
        if count < threshold:
            # Hitung berapa banyak data baru yang dibutuhkan
            needed = threshold - count
            
            # Ambil sampel dari data yang sudah ada dengan label tersebut
            indices = [i for i, x in enumerate(labels) if x == label]
            
            for _ in range(needed):
                idx = np.random.choice(indices)
                original_text = texts[idx]
                
                # Prosedur Augmentasi Generatif (Adaptasi Teknik LLM)
                augmented_text = get_llm_paraphrase(original_text)
                
                balanced_texts.append(augmented_text)
                balanced_numeric.append(numeric_features[idx]) # Fitur numerik mengikuti
                balanced_labels.append(label)
                
    return np.array(balanced_texts), np.array(balanced_numeric), np.array(balanced_labels)