import gzip
import json
import os
import pickle
import re
import time
import unicodedata

import bm25s
import numpy as np
import onnxruntime as ort
import scipy.sparse as sp
from tokenizers import Tokenizer

BODY_WEIGHT = 1.0
WORD_WEIGHT = 0.35
DENSE_WEIGHT = 1.55
TITLE_WEIGHT = 0.05
RRF_K = 60
CANDIDATES = 100
DENSE_CHUNKS = 600
DENSE_DOCS = 150
TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}")


def normalize_text(text):
    folded = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", plain)).strip()


def minmax(scores):
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high <= low:
        return {doc: 1.0 for doc in scores}
    span = high - low
    return {doc: (score - low) / span for doc, score in scores.items()}


class Retriever:
    def __init__(self, index_dir):
        with open(os.path.join(index_dir, "doc_ids.json"), encoding="utf-8") as handle:
            self.doc_ids = json.load(handle)
        self.body_index = bm25s.BM25.load(os.path.join(index_dir, "bm25_body"), load_corpus=False)
        self.title_index = bm25s.BM25.load(os.path.join(index_dir, "bm25_title"), load_corpus=False)

        word_matrix = sp.load_npz(os.path.join(index_dir, "word12.npz")).tocsc()
        self.word_data = word_matrix.data
        self.word_indices = word_matrix.indices
        self.word_indptr = word_matrix.indptr
        self.word_count = word_matrix.shape[0]
        with gzip.open(os.path.join(index_dir, "word12_terms.pkl.gz"), "rb") as handle:
            terms, self.word_idf, self.word_scale = pickle.load(handle)
        self.word_vocab = {term: column for column, term in enumerate(terms)}

        chunk_quantized = np.load(os.path.join(index_dir, "chunk_vectors.npy"))
        chunk_scale = np.load(os.path.join(index_dir, "chunk_scales.npy")).astype(np.float32)
        self.chunk_vectors = chunk_quantized.astype(np.float32) * chunk_scale[:, None]
        self.chunk_doc = np.load(os.path.join(index_dir, "chunk_doc.npy"))

        self.tokenizer = Tokenizer.from_file(os.path.join(index_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.encoder = ort.InferenceSession(
            os.path.join(index_dir, "model_quantized.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def _word_scores(self, normalized):
        tokens = TOKEN_PATTERN.findall(normalized)
        terms = tokens + [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        counts = {}
        for term in terms:
            column = self.word_vocab.get(term)
            if column is not None:
                counts[column] = counts.get(column, 0) + 1
        if not counts:
            return {}
        weights = {column: (1.0 + np.log(count)) * self.word_idf[column] for column, count in counts.items()}
        length = np.sqrt(sum(weight * weight for weight in weights.values()))
        scores = np.zeros(self.word_count, dtype=np.float32)
        for column, weight in weights.items():
            start = self.word_indptr[column]
            end = self.word_indptr[column + 1]
            scores[self.word_indices[start:end]] += (weight / length) * self.word_data[start:end]
        scores *= self.word_scale
        top = np.argsort(-scores)[:CANDIDATES]
        return {int(doc): float(scores[doc]) for doc in top if scores[doc] > 0}

    def _dense_scores(self, normalized):
        encoded = self.tokenizer.encode(normalized)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention = np.array([encoded.attention_mask], dtype=np.int64)
        hidden = self.encoder.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention, "token_type_ids": np.zeros_like(input_ids)},
        )[0]
        pooled = (hidden * attention[..., None]).sum(axis=1) / attention.sum(axis=1, keepdims=True)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)
        similarity = self.chunk_vectors @ pooled[0]
        cutoff = min(DENSE_CHUNKS, len(similarity) - 1)
        top = np.argpartition(similarity, -cutoff)[-cutoff:]
        best = {}
        for chunk in top:
            doc = int(self.chunk_doc[chunk])
            score = float(similarity[chunk])
            if score > best.get(doc, -2.0):
                best[doc] = score
        order = sorted(best, key=best.get, reverse=True)[:DENSE_DOCS]
        return {doc: best[doc] for doc in order}

    def search(self, query, k=3):
        normalized = normalize_text(query)
        if not normalized:
            return []
        tokens = bm25s.tokenize([normalized], stopwords="en", show_progress=False)
        body_ids, body_scores = self.body_index.retrieve(tokens, k=CANDIDATES, show_progress=False)
        title_ids, _ = self.title_index.retrieve(tokens, k=CANDIDATES, show_progress=False)
        body = {int(doc): float(score) for doc, score in zip(body_ids[0], body_scores[0]) if score > 0}
        title = {int(doc): 1.0 / (RRF_K + rank) for rank, doc in enumerate(title_ids[0])}
        word = self._word_scores(normalized)
        dense = self._dense_scores(normalized)
        fused = {}
        for scores, weight in ((body, BODY_WEIGHT), (word, WORD_WEIGHT), (dense, DENSE_WEIGHT), (title, TITLE_WEIGHT)):
            for doc, score in minmax(scores).items():
                fused[doc] = fused.get(doc, 0.0) + weight * score
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.doc_ids[doc] for doc in best]