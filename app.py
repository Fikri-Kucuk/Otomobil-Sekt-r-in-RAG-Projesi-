# =====================================================
# 🚗 Otomotiv RAG Sistemi — FAISS + Gemini (CSV versiyonu)
# =====================================================
# Bu proje, otomotiv sektörü için bir Retrieval-Augmented Generation (RAG) sistemini göstermektedir.
# Sistem, etkili benzerlik araması için FAISS'i ve alınan bilgilere dayanarak yanıtlar oluşturmak için Gemini modelini kullanır.
# Sistem, veri kaynağı olarak bir CSV dosyası kullanmaktadır.
# 📦 Gerekli kütüphaneler
# Kütüphaneler not defterinde zaten yüklü, uygulama dosyasında tekrar yüklemeye gerek yok

import gradio as gr
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
import os
from dotenv import load_dotenv
from tqdm import tqdm

file_path = os.path.join(os.getcwd(), "Automobile.csv")  # Geçerli çalışma dizininde ("current working directory") bulunan "Automobile.csv" dosyasının tam dosya yolunu oluşturur.
df = pd.read_csv(file_path)  # Belirtilen dosya yolundaki CSV dosyasını okur ve bir pandas DataFrame'e yükler.
gerekli_sutunlar = ["name", "horsepower", "weight", "model_year", "origin"]  # Veri setinde analiz veya işlem için gerekli olan sütunları tanımlar.
df = df.dropna(subset=gerekli_sutunlar)  # Belirtilen sütunlarda eksik (NaN) değer içeren satırları veri setinden kaldırır.

# =====================================================
# 2️⃣ Veriyi metin haline getir
# =====================================================
# DataFrame'in her satırını, aracı açıklayan bir metin dizesine dönüştürün.
# Bu metin, RAG sistemi için embedding oluşturmak için kullanılacaktır.

def row_to_text(row):
    return (
        f"Araç adı: {row['name']}, Beygir Gücü: {row['horsepower']} HP, "
        f"Ağırlık: {row['weight']} kg, Üretim Yılı: {row['model_year']}, "
        f"Ülke: {row['origin']}."
    )
documents = [row_to_text(row) for _, row in df.iterrows()]

# =====================================================
# 3️⃣ Gemini API yapılandırması
# =====================================================

load_dotenv() # .env dosyasını yükler ve değişkenleri os.environ'a ekler
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# =====================================================
# 4️⃣ Embedding oluştur (Gemini Embeddings)
# =====================================================

embed_model = "text-embedding-004"  # Kullanılacak embedding (gömüntü) modelinin adını belirler.
embeddings = []  # Oluşturulacak embedding'leri saklamak için boş bir liste oluşturur.
for doc in tqdm(documents, desc="Embedding oluşturuluyor"):  # documents listesindeki her belge için döngü oluşturur ve ilerlemeyi gösterir.
    try:
        result = genai.embed_content(model=embed_model, content=doc)  # Belgeyi seçilen model ile vektör (embedding) haline getirir.
        embeddings.append(result["embedding"])  # Oluşan embedding'i listeye ekler.
    except Exception as e:
        tqdm(f"hata: {e}")  # Eğer bir hata oluşursa, hatayı ekrana yazdırır.
        embeddings.append(np.zeros(768))  # Hata durumunda 768 boyutunda sıfır vektörü ekler.
embeddings = np.array(embeddings).astype("float32")  # Tüm embedding listesini numpy array'e çevirir ve veri tipini float32 yapar.

# =====================================================
# 5️⃣ FAISS Index oluştur
# =====================================================

dimension = len(embeddings[0]) if len(embeddings) > 0 else 0  # Embeddinglerin boyutunu alır; liste boşsa 0 olarak ayarlar.
index = None  # FAISS index değişkenini başlatır (şimdilik None).
if dimension == 0:
    raise ValueError("Embedding oluşturulamadı.")  # Eğer embedding yoksa hata verir.
index = faiss.IndexFlatL2(dimension)  # FAISS kütüphanesi ile L2 (Euclidean) mesafeye göre bir düz (flat) index oluşturur.
index.add(embeddings)  # Oluşturulan embedding'leri FAISS index'e ekler.

# =====================================================
# 6️⃣ RAG sistemi
# =====================================================
# Gemini modeli için prompt şablonunu tanımlayın.
# Bu şablon, modele bir otomotiv uzmanı gibi davranmasını ve sağlanan bağlamı
# kullanarak kullanıcının sorusuna yanıt vermesini talimat verir.

prompt_template = """
Sen otomotiv sektöründe uzman bir yapay zekâ asistanısın.

Kullanıcının sorusuna aşağıdaki bağlamı (context) kullanarak:
-Otomobil gazetecisi gibi
-Doğru
-Anlaşılır
bir yanıt ver.

Eğer bağlamda yeterli bilgi yoksa "Bu konuda elimde yeterli bilgi yok." de.

Bağlam:
{context}

Soru:
{question}

Yanıt:
"""

def retrieve_context(query, top_k=3):
    """
    Kullanıcının sorgusuna en uygun bağlamları döndürür.

    Args:
        query (str): Kullanıcının sorusu veya sorgusu.
        top_k (int): Kaç adet en benzer bağlamın döndürüleceği.

    Returns:
        list: En benzer belgelerin listesi.
    """
    if not isinstance(embeddings, np.ndarray) or embeddings.size == 0 or index is None:
        print("⚠️ Embeddingler veya FAISS indexi mevcut değil.") 
        return []

    try:
        # Sorgu için bir embedding oluşturun
        q_embed = genai.embed_content(model=embed_model, content=query)["embedding"]
        q_embed = np.array([q_embed]).astype("float32")

        # En benzer belgeleri FAISS indexinde arayın
        distances, indices = index.search(q_embed, top_k)

        # En iyi 'top_k' benzer belgelerin içeriğini alın
        results = [documents[i] for i in indices[0]]

        return results

    except Exception as e:
        print(f"Sorgu için bağlam alınırken hata oluştu: {query[:50]}... Hata: {e}") 
        return []


# Kullanıcının sorusuna, alınan bağlam ve Gemini modeli ile yanıt oluşturur
def generate_answer(question):
    # Sorguya en uygun bağlam belgelerini alın
    context_docs = retrieve_context(question)

    # Eğer bağlam bulunamazsa, kullanıcıya bunu belirten mesaj döndür
    if not context_docs:
        return "Bu konuda elimde yeterli bilgi yok."

    # Bağlam belgelerini tek bir metin hâline getirin
    context = "\n".join(context_docs)

    # Prompt'u bağlam ve soruya göre biçimlendir
    prompt = prompt_template.format(context=context, question=question)

    # Yanıt üretmek için Gemini modelini seçin
    model = genai.GenerativeModel("gemini-2.5-flash")  # Gerekiyorsa başka bir desteklenen modelle değiştirin

    try:
        # Modeli kullanarak yanıtı oluştur
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        # Hata durumunda kullanıcıya bilgi ver ve logla
        print(f"Soru için yanıt oluşturulurken hata oluştu: {question[:50]}... Hata: {e}")
        return "Yanıt oluşturulurken bir hata oluştu."


# =====================================================
# 7️⃣ Gradio Uygulaması
# =====================================================
# RAG sistemi ile etkileşim kurmak için bir Gradio arayüzü oluşturun.
# Arayüz, bir metin girişi (kullanıcının sorusu) alır ve bir metin çıktısı (oluşturulan yanıt) sağlar.

iface = gr.Interface(
    fn=generate_answer,
    inputs=gr.Textbox(label="Arabalar Hakkında Bir Soru Sorun"),
    outputs=gr.Textbox(label="Yanıt: ", lines=10), # lines parametresini ekledik ve 10 yaptık
    title= "OTOMOTİV RAG SİSTEMİ",
    allow_flagging="never"
)
iface.launch(debug=True, share=True)