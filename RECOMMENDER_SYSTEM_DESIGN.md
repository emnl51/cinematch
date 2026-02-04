# Hibrit Film & Dizi Öneri Sistemi Tasarımı (Üretime Hazır)

> Hedef: Çoklu sinyali (explicit + implicit + içerik + meta) birlikte kullanarak yüksek doğruluk, düşük gecikme ve ölçeklenebilirlik sağlayan hibrit öneri sistemi.

## 1) Genel Mimari

```
      ┌─────────────── Veri Kaynakları ────────────────┐
      │ explicit (rating) | implicit (watch + time)    │
      │ içerik (özet, tür, oyuncu) | meta (yıl, dil)   │
      └────────────────────────┬───────────────────────┘
                               v
                 ┌──────── Feature/Embedding Store ────────┐
                 │ user profili | item profili | context   │
                 └───────────┬───────────────┬─────────────┘
                             v               v
                   Offline Model Train   Batch Inference
                   (CF/NCF/Seq/NLP)      (nightly cache)
                             v               v
                 ┌────────────── Model Registry ───────────┐
                 │ versioning | A/B test | rollback         │
                 └───────────┬─────────────────────────────┘
                             v
                         Online Serving
                 (hybrid scoring + rerank)
```

**Gerekçe:** Offline modeller doğruluk için ağır eğitim yapar; online katman hızlı cevap ve cache ile ölçeklenir. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L1-L28】

---

## 2) Veri Akışı (Pipeline)

**Adımlar**
1. **ETL/ELT:** log → temizlenmiş etkileşim tablosu (user_id, item_id, event_type, timestamp).
2. **Feature Engineering:**
   - Recency, frequency, watch_time, session_length
   - TF-IDF / Sentence-BERT embedding
3. **Model Eğitim:** CF + NLP + Sequence
4. **Model Registry:** versioning + A/B
5. **Serving:** online scoring + cache + re-rank

**Karar:** recency + session bilgisi, kısa dönem ilgiye hızla adapte olmak için zorunlu. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L32-L46】

---

## 3) Model Katmanları (Rol ve Kullanım)

### 3.1 İçerik Tabanlı
- **Rol:** Yeni içerik için cold-start ve açıklanabilirlik.
- **Model:** TF-IDF + cosine, Sentence-BERT embedding.
- **Çıktı:** `content_similarity(item, item')`

**Neden?** Yeni item’ın etkileşim sinyali yokken içerik embedding’i tek güçlü sinyal. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L52-L58】

### 3.2 İşbirlikçi Filtreleme (CF)
- **Rol:** Kullanıcı davranışı üzerinden örtük benzerlik.
- **Model:** ALS (implicit), SVD (explicit).
- **Çıktı:** `p_user • q_item` latent skor.

**Neden?** İçerik semantiği bilmeden sürpriz öneri üretebilir. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L60-L66】

### 3.3 Bilgi Tabanlı / Kural
- **Rol:** Yeni kullanıcı için ilk öneri + işletme hedefleri.
- **Model:** Onboarding tercihleri + basit kurallar.
- **Örnek:** `dil=TR AND tür=Drama`.

**Neden?** İlk oturumda kullanıcı etkileşimi yoksa güvenli başlangıç. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L68-L73】

### 3.4 Derin Öğrenme (NCF + Sequence)
- **Rol:** Kısa dönem ilgi ve kompleks etkileşim.
- **Model:** NCF + Transformer (sequence-aware).
- **Özellik:** Son izleme sırası, zaman aralığı.

**Neden?** “Son izlenen içerik” etkisini yakalar ve trendleri yakalar. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L75-L80】

---

## 4) Hibritleştirme Stratejisi (Late Fusion)

**Formül**
```
score(u,i) = w_cf*s_cf + w_content*s_content + w_seq*s_seq + w_rule*s_rule
```

**Karar:** Late fusion → her model bağımsız geliştirilebilir, ağırlık A/B ile güncellenir. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L86-L92】

---

## 5) Algoritma ve Teknoloji Önerileri

**NLP / Embedding**
- TF-IDF (baseline)
- Sentence-BERT (çok dilli)
- LLM embedding (yüksek doğruluk, yüksek maliyet)

**CF**
- implicit-ALS (implicit lib)
- LightFM (hybrid)

**Deep**
- TensorFlow Recommenders
- PyTorch + Transformers4Rec

**Karar:** ALS büyük veride hızlıdır, LightFM hibrit için kolaydır. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L98-L112】

---

## 6) Değerlendirme

**Offline**
- Precision@K
- Recall@K
- NDCG@K

**Online**
- CTR
- Watch-time
- 7/30 gün retention

**Karar:** Offline metrik hızlı iterasyon sağlar; gerçek etki A/B ile ölçülür. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L116-L124】

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
- Kullanıcı etkileşimi → Kafka / Redis Streams
- Online feature update (recency, session context)

### Ölçeklenebilirlik
- Redis cache (popüler öneriler)
- Batch inference (nightly + incremental)

**Karar:** Cache + batch inference, online latency ve maliyeti azaltır. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L128-L150】

---

## 8) MVP → Production Yol Haritası

**MVP (2–4 hafta)**
1. Baseline implicit ALS
2. TF-IDF içerik benzerliği
3. Basit API

**Production (2–3 ay)**
1. Sentence-BERT embedding
2. NCF + Transformer (sequence)
3. A/B altyapısı
4. Cache + batch inference

**Karar:** MVP hızlı değer sağlar; production aşaması doğruluk + ölçeklenebilirlik getirir. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L156-L171】

---

## 9) Kritik Riskler ve Çözümler

- **Cold-start:** içerik + kural
- **Sparsity:** implicit feedback + ALS
- **Filter bubble:** exploration oranı + çeşitlilik ölçütü
- **Performans vs açıklanabilirlik:** baseline modelleri açıklama için koru

**Karar:** Keşif çeşitliliği ve açıklanabilirlik uzun vadede retention için zorunlu. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L175-L183】

---

## 10) Basit Pseudocode (Online Scoring)

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

**Karar:** Rerank katmanı, filter bubble riskini azaltır. 【F:RECOMMENDER_SYSTEM_DESIGN.md†L187-L201】

