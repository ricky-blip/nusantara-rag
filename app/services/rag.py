import os
import re
import math
import hashlib
from dataclasses import dataclass
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DOC_PATH = os.path.join("data", "raw_docs", "nusantaracare_panduan_operasional_internal_v2.md")
HEADER_RE = re.compile(r"^#{1,6}\s+(.*)$")
TOKEN_RE = re.compile(r"[a-z0-9]+")

# Urutan kandidat model embedding Gemini (dari yang terbaru)
EMBEDDING_CANDIDATES = [
    "gemini-embedding-001",
    "gemini-embedding-2",
    "text-embedding-004",
    "embedding-001",
]

@dataclass
class Chunk:
    chunk_id: str
    text: str
    section_title: str
    doc_version: str
    is_active: bool

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

def _is_header(block: str) -> str | None:
    first = block.splitlines()[0].strip()
    m = HEADER_RE.match(first)
    if m:
        return m.group(1).strip()
    if (len(block) < 90 and "\n" not in block and not block.endswith(".")
            and not block.startswith("|") and ":" not in block and not block.endswith("?")):
        return first
    return None

def split_into_chunks(text: str, max_size: int = 1000, overlap: int = 200) -> list[Chunk]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks: list[Chunk] = []
    current, section, prev_tail = "", "Umum", ""

    def flush(content: str, sec: str):
        nonlocal prev_tail
        if not content.strip():
            return
        cid = f"chunk_{len(chunks):03d}"
        chunks.append(Chunk(cid, prev_tail + content.strip(), sec, "2.0", True))
        prev_tail = content[-overlap:] + "\n" if len(content) > overlap else ""

    for block in blocks:
        h = _is_header(block)
        if h:
            flush(current, section)
            current = block      # judul ikut masuk ke chunk (penting untuk penanda v1.4)
            section = h
            continue
        if len(current) + len(block) > max_size:
            flush(current, section)
            current = block
        else:
            current = current + "\n\n" + block if current else block
    flush(current, section)

    # STATE MACHINE v2.0 vs v1.4
    active = True
    for c in chunks:
        if "Arsip Kebijakan v1.4" in c.text:
            active = False
        if "Pengganti Aktif v2.0" in c.text:
            active = True
        c.is_active = active
        c.doc_version = "2.0" if active else "1.4"
    return chunks

class LocalLexicalEmbedder:
    """Fallback TF-IDF lokal jika API embedding tidak tersedia."""
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []

    def fit(self, texts: list[str]):
        df: dict[str, int] = {}
        for t in texts:
            for tok in set(TOKEN_RE.findall(t.lower())):
                df[tok] = df.get(tok, 0) + 1
        for i, tok in enumerate(sorted(df)):
            self.vocab[tok] = i
        self.idf = [math.log((len(texts) + 1) / (df[tok] + 1)) + 1 for tok in sorted(df)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            v = [0.0] * len(self.vocab)
            counts: dict[str, int] = {}
            for tok in TOKEN_RE.findall(t.lower()):
                if tok in self.vocab:
                    counts[tok] = counts.get(tok, 0) + 1
            for tok, c in counts.items():
                v[self.vocab[tok]] = (1 + math.log(c)) * self.idf[self.vocab[tok]]
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / norm for x in v])
        return vecs

class RAGPipeline:
    def __init__(self):
        self.chat_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.chat_model = os.getenv("CHAT_MODEL", "gemini-2.0-flash")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.embed_backend = None          # "gemini:<model>" atau "local"
        self.gemini_model = None
        self.local = LocalLexicalEmbedder()
        self.min_score = 0.30
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []

    def _embed_gemini(self, texts: list[str], model: str) -> list[list[float]]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
        payload = {"requests": [
            {"model": f"models/{model}", "content": {"parts": [{"text": t}]}} for t in texts
        ]}
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload, headers={"x-goog-api-key": self.api_key})
            resp.raise_for_status()
            return [e["values"] for e in resp.json()["embeddings"]]

    def build_index(self):
        with open(DOC_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        self.chunks = split_into_chunks(text)

        # --- Deteksi otomatis backend embedding ---
        for model in EMBEDDING_CANDIDATES:
            try:
                self._embed_gemini([self.chunks[0].text], model)
                self.gemini_model = model
                self.embed_backend = f"gemini:{model}"
                self.min_score = 0.30
                print(f"--- Embedding backend: Gemini API ({model}) ---")
                break
            except Exception as e:
                print(f"  - {model}: tidak tersedia ({type(e).__name__}), coba kandidat lain...")

        if self.embed_backend is None:
            self.embed_backend = "local"
            self.min_score = 0.08
            print("⚠️  API embedding Gemini tidak tersedia -> fallback: lexical embedding lokal (TF-IDF).")

        # --- Bangun vektor ---
        if self.embed_backend.startswith("gemini"):
            for i in range(0, len(self.chunks), 5):
                batch = [c.text for c in self.chunks[i:i + 5]]
                self.vectors.extend(self._embed_gemini(batch, self.gemini_model))
                print(f"  ✓ Embedded {min(i + 5, len(self.chunks))}/{len(self.chunks)} chunks")
        else:
            self.local.fit([c.text for c in self.chunks])
            self.vectors = self.local.embed([c.text for c in self.chunks])

        n_inactive = sum(1 for c in self.chunks if not c.is_active)
        print(f"--- Index yg ready: {len(self.chunks)} chunks --- "
              f"({len(self.chunks) - n_inactive} aktif v2.0, {n_inactive} arsip v1.4)")

    def retrieve(self, query: str, top_k: int = 5):
        if not self.vectors:
            return []
        if self.embed_backend.startswith("gemini"):
            qv = self._embed_gemini([query], self.gemini_model)[0]
        else:
            qv = self.local.embed([query])[0]
        scored = [(cosine(qv, v), c) for v, c in zip(self.vectors, self.chunks) if c.is_active]
        scored = [(s, c) for s, c in scored if s >= self.min_score]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

rag_pipeline = RAGPipeline()