# NusantaraCare AI Assistant — RAG API

Backend FastAPI asisten GenAI berbasis RAG atas dokumen
**Panduan Operasional Layanan Internal NusantaraCare v2.0**.
Final Project AI Engineer Intermediate — Ricky Rinaldy.

## 1. Problem & Success Criteria
**Problem:** Karyawan sulit menemukan jawaban di dokumen SOP yang panjang.

**Kriteria sukses:**
- Jawaban 100% grounded + kutipan sumber (`sources`).
- Menolak pertanyaan di luar cakupan & menangkal prompt injection.
- Hanya memakai aturan v2.0; arsip v1.4 tidak boleh bocor.
- Respons selalu memuat `answer`, `confidence_label`, `reason_code`.

**Batasan:** satu dokumen acuan; tidak mencakup area medis, hukum, gaji, kinerja personal.

## 2. Knowledge Base Understanding
Dokumen `NC-OPS-001`, versi "2.0", berlaku efektif 2026-07-01, `is_active: true`,
owner: Direktorat Operasi dan Layanan Internal.
Isi: definisi & 6 peran, kanal & jam operasional, klasifikasi P1/P2/P3,
3 SOP (Akses & Akun, Gangguan & Eskalasi, Fasilitas & Perlengkapan),
kebijakan data & kerahasiaan, status tiket & SLA, FAQ, matriks keputusan, riwayat + arsip.

**v2.0 vs v1.4 (krusial):**
| Ketentuan | v1.4 (NONAKTIF) | v2.0 (AKTIF) |
|---|---|---|
| Email | Saluran setara portal | Hanya saat portal down, subjek `[DARURAT-PORTAL]` |
| Lead time perlengkapan | 3 hari kerja | **5 hari kerja** |

## 3. RAG Design & Data Preparation
- **Chunking:** blok paragraf + deteksi judul section; 1000 karakter, overlap 200
  (mencegah syarat kumulatif terpotong). Hasil: **91 chunks**.
- **Metadata per chunk:** `chunk_id`, `section_title`, `doc_version`, `is_active`.
- **Vector database:** in-memory index + cosine similarity.
  *Justifikasi:* KB kecil (91 chunks) → vector DB eksternal (Chroma/FAISS) menambah
  dependency berat tanpa manfaat; deterministic dan ramah deployment serverless.
- **Embedding:** Gemini Embedding REST API dengan auto-detection model;
  fallback TF-IDF lokal bila API gagal.
- **Retrieval:** top-k = 5, threshold 0.30, **hard filter `is_active=true`**
  (chunk v1.4 tidak pernah masuk konteks).
- **Prompt:** jawab HANYA dari konteks; output JSON; anti-injection; temperature 0.1.
- **Dokumen nonaktif:** state machine menandai chunk arsip
  ("Arsip Kebijakan v1.4" s.d. "Pengganti Aktif v2.0") sebagai `is_active=false`.

## 4. Arsitektur
```
POST /api/v1/query → FastAPI → Agent →
  1) retrieve: embed query → cosine search (filter is_active)
  2) gate: skor < threshold → tolak tanpa memanggil LLM
  3) generate: konteks + pertanyaan → LLM → JSON terstruktur
```

## 5. Kontrak API
`POST /api/v1/query` · Request: `{"question": "..."}`
Respons: `answer`, `confidence_label` (high/medium/low),
`reason_code` (answered / no_relevant_context / out_of_scope),
`sources` (chunk_id, section_title, doc_version, score).
`GET /` = health check · `GET /docs` = Swagger UI.

## 6. Cara Menjalankan Lokal
```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # isi OPENAI_API_KEY
uvicorn app.main:app --reload
```

## 7. Deployment
FastAPI Cloud: hubungkan repo GitHub (branch main), entry point `main.py` di root.
Secret env vars: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `CHAT_MODEL`.
Log startup wajib menampilkan `Index siap: 91 chunks`.

## 8. Keterbatasan
- Cold start 20–60 detik (pembangunan index + embedding).
- Ketersediaan model embedding bergantung migrasi Google (ada fallback TF-IDF).
- Tabel matriks di-chunk sebagai teks, bukan parsing tabel penuh.

## 9. Kesimpulan & Rekomendasi
**Hasil uji:** jebakan v1.4 dijawab benar ("5 hari kerja", high, sumber v2.0);
out-of-scope ditolak; prompt injection tidak bocor. Sistem memenuhi kriteria sukses.

**Rekomendasi:** hybrid search + reranker, cache index (cold start < 5 dtk),
endpoint `/feedback`, ChromaDB bila KB multi-dokumen, streaming respons.