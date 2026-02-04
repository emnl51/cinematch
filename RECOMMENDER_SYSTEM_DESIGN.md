# Hibrit Film & Dizi Öneri Sistemi Tasarımı (Üretime Hazır)

Bu doküman, web uygulaması için **uçtan uca hibrit öneri sistemi** kurulumunu teknik ve uygulanabilir şekilde tanımlar. Teorik ayrıntı minimumda tutulmuş, kararlar gerekçelendirilmiştir.

---

## 1) Genel Mimari

```
Veri Kaynakları
  ├─ Explicit: Rating, beğeni
  ├─ Implicit: İzleme, süre, zaman
  ├─ İçerik: Özet, tür, oyuncu, yönetmen, etiket
  └─ Meta: Yıl, ülke, dil, süre
        |
        v
Feature + Embedding Store
  ├─ User profile (long-term + session)
  ├─ Item profile (content + metadata)
  └─ Interaction features (recency, frequency)
        |
        +--------------------+------------------+
        |                    |                  |
        v                    v                  v
Offline Model Train     Offline Eval        Batch Inference
  ├─ CF (ALS/SVD)         ├─ Precision@K       ├─ Daily/Hourly
  ├─ NCF / Sequence       ├─ Recall@K          └─ Cache
  └─ NLP Embedding        └─ NDCG
        |
        v
Model Registry + Online Serving
  ├─ Hybrid Scoring
  ├─ Re-ranking
  └─ Real-time API
```

**Karar:** Offline eğitim doğruluğu maksimize eder; online servis düşük gecikme ve cache ile ölçeklenir. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L1-L34】

---

## 2) Veri Akışı (Pipeline)

1. **ETL/ELT:** event log → normalized interaction table  
   - `user_id, item_id, event_type, timestamp, watch_time`
2. **Feature Engineering:**  
   - **Recency, frequency, session_length**  
   - **TF-IDF / Sentence-BERT embedding**  
3. **Model Eğitim:** CF + NLP + Sequence  
4. **Model Registry:** versioning + A/B  
5. **Serving:** online scoring + cache + rerank

**Karar:** Recency + session bilgisi kısa vadeli ilgiyi yakalamak için zorunludur. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L38-L51】

---

## 3) Model Katmanları (Rol Bazlı)

### 3.1 İçerik Tabanlı Filtreleme
- **Rol:** Yeni içerik (cold-start) ve açıklanabilirlik.  
- **Model:** TF-IDF + cosine / Sentence-BERT embedding.  
- **Çıktı:** `content_similarity(item, item')`.

**Karar:** Etkileşim verisi olmayan item’larda tek güvenilir sinyal içeriktir. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L57-L64】

### 3.2 İşbirlikçi Filtreleme (CF)
- **Rol:** Davranış üzerinden örtük benzerlik.  
- **Model:** Implicit ALS (izleme), SVD (rating).  
- **Çıktı:** `p_user · q_item`.

**Karar:** CF, içerik bilinmese bile keşif gücü sağlar. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L66-L72】

### 3.3 Bilgi Tabanlı / Kural
- **Rol:** Yeni kullanıcı için güvenli başlangıç + iş kuralları.  
- **Model:** Onboarding tercihleri + kural tabanı.  
- **Örnek:** `dil=TR AND tür=Drama`.

**Karar:** İlk oturumda kural tabanlı öneri, kullanıcıyı hızlı aktive eder. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L74-L80】

### 3.4 Derin Öğrenme (NCF + Sequence)
- **Rol:** Kısa dönem ilgi ve sıralı davranış.  
- **Model:** NCF + Transformer (sequence-aware).  
- **Özellik:** Son izleme sırası + zaman aralığı.

**Karar:** Session sinyalleri trend ve anlık ilgiyi yakalar. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L82-L88】

---

## 4) Hibritleştirme Stratejisi

**Late Fusion (Weighted Ensemble)**

```
score(u,i) =
  w_cf*s_cf(u,i) +
  w_content*s_content(u,i) +
  w_seq*s_seq(u,i) +
  w_rule*s_rule(u,i)
```

**Karar:** Late fusion modülerdir, model bağımsız geliştirilebilir ve ağırlıklar A/B test ile optimize edilir. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L92-L104】

---

## 5) Teknoloji ve Algoritma Önerileri

**NLP / Embedding**
- TF-IDF (baseline, hızlı)
- Sentence-BERT (çok dilli semantik)
- LLM embedding (yüksek doğruluk, maliyetli)

**CF**
- implicit-ALS (implicit lib)
- LightFM (hybrid CF + content)

**Deep**
- TensorFlow Recommenders
- PyTorch + Transformers4Rec

**Karar:** ALS büyük veri için hızlı ve stabil; LightFM hibrit geçişi kolaylaştırır. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L108-L123】

---

## 6) Değerlendirme

**Offline**
- Precision@K
- Recall@K
- NDCG@K

**Online**
- CTR
- Ortalama izleme süresi
- 7/30 gün retention

**Karar:** Offline metrikler hızlı iterasyon sağlar; gerçek etki A/B test ile ölçülür. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L127-L136】

---

## 7) Web Uygulaması Entegrasyonu

### API Tasarımı
**GET** `/recommendations?user_id=123&limit=20`
```json
{
  "user_id": 123,
  "recommendations": [
    {"item_id": 42, "score": 0.87, "reason": "Benzer tür + son izleme"}
  ]
}
```

### Gerçek Zamanlı Akış
- Event → Kafka / Redis Streams  
- Online feature update (recency, session context)

### Ölçeklenebilirlik
- Redis cache (popüler öneriler)
- Batch inference (nightly + incremental)

**Karar:** Cache ve batch inference, online latency ve maliyeti düşürür. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L140-L164】

---

## 8) Yol Haritası (MVP → Production)

**MVP (2–4 hafta)**
1. Implicit ALS  
2. TF-IDF içerik benzerliği  
3. Basit API  

**Production (2–3 ay)**
1. Sentence-BERT embedding  
2. NCF + Transformer  
3. A/B test altyapısı  
4. Cache + batch inference  

**Karar:** MVP hızlı değer sağlar; production aşaması doğruluk + ölçeklenebilirlik getirir. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L168-L185】

---

## 9) Kritik Riskler ve Çözümler

- **Cold-start:** içerik + kural tabanlı öneri  
- **Sparsity:** implicit feedback + ALS  
- **Filter bubble:** exploration oranı + çeşitlilik rerank  
- **Performans vs açıklanabilirlik:** baseline modelleri açıklama için koru  

**Karar:** Keşif + açıklanabilirlik uzun vadede retention için zorunlu. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L189-L197】

---

## 10) Online Scoring Pseudocode

```
def recommend(user_id, limit=20):
    cf_scores = cf_model.predict(user_id)
    content_scores = content_model.similar_items(user_profile[user_id])
    seq_scores = seq_model.predict_next(user_id)
    rule_scores = rule_engine.apply(user_profile[user_id])

    hybrid = w_cf*cf_scores + w_content*content_scores + w_seq*seq_scores + w_rule*rule_scores
    reranked = diversity_rerank(hybrid)
    return top_k(reranked, limit)
```

**Karar:** Rerank katmanı filter bubble riskini azaltır. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L201-L215】

