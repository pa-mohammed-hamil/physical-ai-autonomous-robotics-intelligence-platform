#!/usr/bin/env python3
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: scripts/build_rag_index.py

Build a secure, tenant-scoped retrieval-augmented generation
(RAG) index for robotics manuals, maintenance documents,
fault codes, safety procedures, and simulation knowledge.

Features
--------
- PDF, TXT, Markdown, HTML, JSON, CSV ingestion
- Optional DOCX support
- Recursive directory discovery
- SHA-256 document identification
- Content normalization and cleaning
- Token-aware overlapping chunking
- Document metadata and source citations
- Local sentence-transformer embeddings
- FAISS vector index
- BM25 lexical index
- Hybrid dense + sparse retrieval
- Reciprocal Rank Fusion (RRF)
- Optional cross-encoder reranking
- Tenant and document-level access control
- Incremental updates and deleted-file handling
- Atomic index generation and manifest promotion
- Index integrity verification
- Search and evaluation CLI
- No physical robot command execution

Install
-------
pip install numpy sentence-transformers faiss-cpu \
    rank-bm25 pypdf beautifulsoup4

Optional:
    pip install python-docx

Examples
--------
python scripts/build_rag_index.py build \
    --input data/knowledge \
    --output storage/rag_index \
    --tenant demo \
    --collection robotics

python scripts/build_rag_index.py search \
    --output storage/rag_index \
    --tenant demo \
    --collection robotics \
    --query "What does fault code E102 mean?"

python scripts/build_rag_index.py verify \
    --output storage/rag_index \
    --tenant demo \
    --collection robotics

python scripts/build_rag_index.py evaluate \
    --output storage/rag_index \
    --tenant demo \
    --collection robotics \
    --questions data/rag/evaluation.json

Security notes
--------------
- Use a private, trusted local storage directory.
- Do not ingest secrets or private documents without consent.
- Search access is enforced by tenant and document ACL.
- This script does not execute instructions found in documents.
- Index artifacts should not be loaded from untrusted sources.
- Retrieval results are evidence, not authorization to act.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
import uuid

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


LOGGER = logging.getLogger("build_rag_index")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SCHEMA_VERSION = "1.0.0"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".json",
    ".csv",
    ".docx",
}

DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEFAULT_RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_DOCUMENT_CHARACTERS = 5_000_000
MAX_CHUNKS_PER_DOCUMENT = 20_000


# ============================================================
# Exceptions
# ============================================================

class RAGIndexError(Exception):
    pass


class DocumentParsingError(RAGIndexError):
    pass


class IndexIntegrityError(RAGIndexError):
    pass


class IndexPermissionError(RAGIndexError):
    pass


class IndexConfigurationError(RAGIndexError):
    pass


# ============================================================
# Utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def validate_identifier(
    value: str,
    name: str,
) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise IndexConfigurationError(
            f"Invalid {name}: {value!r}"
        )

    return value


def safe_json_dump(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

            stream.flush()
            os.fsync(stream.fileno())

            temporary = Path(stream.name)

        os.replace(temporary, path)

    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
    ]


def canonical_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def ensure_private_directory(path: Path) -> None:
    path.mkdir(
        parents=True,
        exist_ok=True,
        mode=0o700,
    )

    if os.name == "posix":
        path.chmod(0o700)


# ============================================================
# Configuration
# ============================================================

@dataclass
class RAGIndexConfig:
    input_directory: str
    output_directory: str = "storage/rag_index"

    tenant_id: str = "demo"
    collection: str = "robotics"

    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    chunk_size: int = 350
    chunk_overlap: int = 60

    embedding_batch_size: int = 32

    dense_weight: float = 0.65
    sparse_weight: float = 0.35

    rrf_k: int = 60

    top_k: int = 5

    recursive: bool = True
    incremental: bool = True

    allowed_roles: list[str] = field(
        default_factory=lambda: [
            "admin",
            "engineer",
            "researcher",
        ]
    )

    max_file_bytes: int = MAX_FILE_BYTES

    def validate(self) -> None:
        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.collection,
            "collection",
        )

        if self.chunk_size < 32:
            raise IndexConfigurationError(
                "Chunk size must be at least 32"
            )

        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise IndexConfigurationError(
                "Chunk overlap must be smaller than chunk size"
            )

        if self.embedding_batch_size < 1:
            raise IndexConfigurationError(
                "Embedding batch size must be positive"
            )

        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise IndexConfigurationError(
                "Hybrid retrieval weights cannot be negative"
            )

        if self.dense_weight + self.sparse_weight <= 0:
            raise IndexConfigurationError(
                "At least one retrieval weight must be positive"
            )

        if self.rrf_k < 1:
            raise IndexConfigurationError(
                "RRF constant must be positive"
            )

        if self.top_k < 1:
            raise IndexConfigurationError(
                "top_k must be positive"
            )

        if self.max_file_bytes < 1:
            raise IndexConfigurationError(
                "File size limit must be positive"
            )

        if not self.allowed_roles:
            raise IndexConfigurationError(
                "At least one allowed role is required"
            )

        for role in self.allowed_roles:
            validate_identifier(role, "role")


# ============================================================
# Document Models
# ============================================================

@dataclass
class ParsedDocument:
    document_id: str
    source_path: str
    source_sha256: str

    tenant_id: str
    collection: str

    title: str
    content: str

    mime_type: str
    metadata: dict[str, Any]

    allowed_roles: list[str]

    created_at: str


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str

    tenant_id: str
    collection: str

    source_path: str
    title: str

    chunk_index: int
    text: str

    token_count: int

    allowed_roles: list[str]

    metadata: dict[str, Any]

    citation: str


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str

    title: str
    source_path: str

    text: str
    citation: str

    score: float
    rank: int

    metadata: dict[str, Any]


# ============================================================
# Document Parsers
# ============================================================

class DocumentParser:
    MIME_TYPES = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
        ".json": "application/json",
        ".csv": "text/csv",
        ".docx": (
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
    }

    def parse(
        self,
        path: Path,
        *,
        root: Path,
        tenant_id: str,
        collection: str,
        allowed_roles: Sequence[str],
        max_file_bytes: int,
    ) -> ParsedDocument:
        path = path.resolve()
        root = root.resolve()

        if not path.is_relative_to(root):
            raise DocumentParsingError(
                "Document path escapes input root"
            )

        if path.is_symlink():
            raise DocumentParsingError(
                "Symlinked documents are not supported"
            )

        if not path.is_file():
            raise DocumentParsingError(
                f"Not a regular file: {path}"
            )

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise DocumentParsingError(
                f"Unsupported file extension: {extension}"
            )

        if path.stat().st_size > max_file_bytes:
            raise DocumentParsingError(
                f"Document exceeds size limit: {path}"
            )

        raw_hash = sha256_file(path)

        if extension == ".pdf":
            content = self._parse_pdf(path)

        elif extension in {
            ".txt",
            ".md",
            ".markdown",
        }:
            content = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        elif extension in {
            ".html",
            ".htm",
        }:
            content = self._parse_html(path)

        elif extension == ".json":
            content = self._parse_json(path)

        elif extension == ".csv":
            content = self._parse_csv(path)

        elif extension == ".docx":
            content = self._parse_docx(path)

        else:
            raise DocumentParsingError(
                "Unsupported document format"
            )

        content = normalize_text(content)

        if not content:
            raise DocumentParsingError(
                f"No extractable text: {path}"
            )

        if len(content) > MAX_DOCUMENT_CHARACTERS:
            raise DocumentParsingError(
                "Extracted document text exceeds limit"
            )

        relative_path = path.relative_to(
            root
        ).as_posix()

        document_id = sha256_bytes(
            (
                f"{tenant_id}:{collection}:{relative_path}"
            ).encode("utf-8")
        )[:32]

        return ParsedDocument(
            document_id=document_id,
            source_path=relative_path,
            source_sha256=raw_hash,
            tenant_id=tenant_id,
            collection=collection,
            title=path.stem,
            content=content,
            mime_type=self.MIME_TYPES[extension],
            metadata={
                "extension": extension,
                "size_bytes": path.stat().st_size,
            },
            allowed_roles=list(allowed_roles),
            created_at=utc_now(),
        )

    @staticmethod
    def _parse_pdf(path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentParsingError(
                "Install pypdf for PDF support"
            ) from exc

        reader = PdfReader(str(path))

        parts = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            text = page.extract_text() or ""

            if text.strip():
                parts.append(
                    f"[Page {page_number}]\n{text}"
                )

        return "\n\n".join(parts)

    @staticmethod
    def _parse_html(path: Path) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise DocumentParsingError(
                "Install beautifulsoup4 for HTML support"
            ) from exc

        content = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        soup = BeautifulSoup(
            content,
            "html.parser",
        )

        for element in soup(
            ["script", "style", "noscript"]
        ):
            element.decompose()

        return soup.get_text(
            separator="\n",
            strip=True,
        )

    @staticmethod
    def _parse_json(path: Path) -> str:
        payload = load_json(path)

        return json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )

    @staticmethod
    def _parse_csv(path: Path) -> str:
        rows = []

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream)

            for index, row in enumerate(reader):
                if index >= 100_000:
                    raise DocumentParsingError(
                        "CSV row limit exceeded"
                    )

                rows.append(
                    " | ".join(
                        f"{key}: {value}"
                        for key, value in row.items()
                    )
                )

        return "\n".join(rows)

    @staticmethod
    def _parse_docx(path: Path) -> str:
        try:
            from docx import Document
        except ImportError as exc:
            raise DocumentParsingError(
                "Install python-docx for DOCX support"
            ) from exc

        document = Document(str(path))

        parts = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        for table in document.tables:
            for row in table.rows:
                parts.append(
                    " | ".join(
                        cell.text.strip()
                        for cell in row.cells
                    )
                )

        return "\n".join(parts)


# ============================================================
# File Discovery
# ============================================================

def discover_documents(
    root: Path,
    *,
    recursive: bool,
) -> list[Path]:
    root = root.resolve()

    if not root.is_dir():
        raise DocumentParsingError(
            f"Input directory not found: {root}"
        )

    iterator = (
        root.rglob("*")
        if recursive
        else root.glob("*")
    )

    files = []

    for path in iterator:
        if path.is_symlink():
            continue

        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        files.append(path)

    return sorted(
        files,
        key=lambda path: str(path),
    )


# ============================================================
# Chunking
# ============================================================

class TokenChunker:
    def __init__(
        self,
        chunk_size: int,
        overlap: int,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(
                "Chunk size must be positive"
            )

        if not 0 <= overlap < chunk_size:
            raise ValueError(
                "Invalid chunk overlap"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        document: ParsedDocument,
    ) -> list[DocumentChunk]:
        # Preserve source text while using approximate
        # word/punctuation token boundaries.
        matches = list(
            TOKEN_PATTERN.finditer(
                document.content
            )
        )

        if not matches:
            return []

        chunks = []

        stride = (
            self.chunk_size - self.overlap
        )

        for index, start in enumerate(
            range(0, len(matches), stride)
        ):
            end = min(
                start + self.chunk_size,
                len(matches),
            )

            first_character = matches[start].start()
            last_character = matches[end - 1].end()

            text = document.content[
                first_character:last_character
            ].strip()

            if not text:
                continue

            chunk_id = sha256_bytes(
                (
                    f"{document.document_id}:"
                    f"{document.source_sha256}:"
                    f"{index}:"
                    f"{text}"
                ).encode("utf-8")
            )[:40]

            citation = (
                f"{document.source_path}"
                f"#chunk-{index + 1}"
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    collection=document.collection,
                    source_path=document.source_path,
                    title=document.title,
                    chunk_index=index,
                    text=text,
                    token_count=end - start,
                    allowed_roles=list(
                        document.allowed_roles
                    ),
                    metadata=dict(
                        document.metadata
                    ),
                    citation=citation,
                )
            )

            if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
                raise DocumentParsingError(
                    "Document chunk limit exceeded"
                )

            if end == len(matches):
                break

        return chunks


# ============================================================
# Embedding Model
# ============================================================

class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int,
    ) -> None:
        try:
            from sentence_transformers import (
                SentenceTransformer,
            )
        except ImportError as exc:
            raise RAGIndexError(
                "Install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size

        LOGGER.info(
            "Loading embedding model: %s",
            model_name,
        )

        self.model = SentenceTransformer(
            model_name
        )

    def encode(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        if not texts:
            raise RAGIndexError(
                "Cannot embed an empty text collection"
            )

        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise RAGIndexError(
                "Embedding model returned invalid shape"
            )

        if not np.isfinite(embeddings).all():
            raise RAGIndexError(
                "Embedding model returned non-finite values"
            )

        return embeddings


# ============================================================
# FAISS Dense Index
# ============================================================

class DenseVectorIndex:
    def __init__(self) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RAGIndexError(
                "Install faiss-cpu"
            ) from exc

        self.faiss = faiss
        self.index = None

    def build(
        self,
        embeddings: np.ndarray,
    ) -> None:
        embeddings = np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise RAGIndexError(
                "Embeddings must be two-dimensional"
            )

        dimension = embeddings.shape[1]

        index = self.faiss.IndexFlatIP(
            dimension
        )

        index.add(
            embeddings
        )

        self.index = index

    def save(self, path: Path) -> None:
        if self.index is None:
            raise RAGIndexError(
                "Dense index has not been built"
            )

        self.faiss.write_index(
            self.index,
            str(path),
        )

    def load(self, path: Path) -> None:
        if not path.is_file():
            raise IndexIntegrityError(
                "FAISS index file missing"
            )

        self.index = self.faiss.read_index(
            str(path)
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[tuple[int, float]]:
        if self.index is None:
            raise RAGIndexError(
                "Dense index is not loaded"
            )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        ).reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            min(
                top_k,
                self.index.ntotal,
            ),
        )

        return [
            (int(index), float(score))
            for index, score in zip(
                indices[0],
                scores[0],
            )
            if index >= 0
        ]


# ============================================================
# BM25 Sparse Index
# ============================================================

class SparseBM25Index:
    def __init__(
        self,
        texts: Sequence[str],
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RAGIndexError(
                "Install rank-bm25"
            ) from exc

        self.corpus = [
            tokenize(text)
            for text in texts
        ]

        self.model = BM25Okapi(
            self.corpus
        )

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[int, float]]:
        query_tokens = tokenize(
            query
        )

        if not query_tokens:
            return []

        scores = np.asarray(
            self.model.get_scores(
                query_tokens
            ),
            dtype=np.float64,
        )

        indices = np.argsort(
            -scores,
            kind="stable",
        )[:top_k]

        return [
            (int(index), float(scores[index]))
            for index in indices
            if scores[index] > 0
        ]


# ============================================================
# Hybrid Ranking
# ============================================================

def reciprocal_rank_fusion(
    dense_results: Sequence[tuple[int, float]],
    sparse_results: Sequence[tuple[int, float]],
    *,
    dense_weight: float,
    sparse_weight: float,
    rrf_k: int,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}

    for rank, (index, _) in enumerate(
        dense_results,
        start=1,
    ):
        scores[index] = (
            scores.get(index, 0.0)
            + dense_weight / (rrf_k + rank)
        )

    for rank, (index, _) in enumerate(
        sparse_results,
        start=1,
    ):
        scores[index] = (
            scores.get(index, 0.0)
            + sparse_weight / (rrf_k + rank)
        )

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )


# ============================================================
# Optional Reranker
# ============================================================

class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RAGIndexError(
                "Install sentence-transformers for reranking"
            ) from exc

        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        if not results:
            return []

        pairs = [
            (query, result.text)
            for result in results
        ]

        scores = self.model.predict(
            pairs
        )

        ranked = sorted(
            zip(results, scores),
            key=lambda item: -float(item[1]),
        )

        output = []

        for rank, (result, score) in enumerate(
            ranked,
            start=1,
        ):
            result.score = float(score)
            result.rank = rank
            output.append(result)

        return output


# ============================================================
# Storage Layout
# ============================================================

class IndexPaths:
    def __init__(
        self,
        output_directory: str | Path,
        tenant_id: str,
        collection: str,
    ) -> None:
        validate_identifier(
            tenant_id,
            "tenant_id",
        )

        validate_identifier(
            collection,
            "collection",
        )

        self.root = (
            canonical_path(output_directory)
            / tenant_id
            / collection
        )

        self.generations = (
            self.root / "generations"
        )

        self.current_manifest = (
            self.root / "current.json"
        )

    def create(self) -> None:
        ensure_private_directory(
            self.root
        )

        ensure_private_directory(
            self.generations
        )

    def generation(
        self,
        generation_id: str,
    ) -> Path:
        validate_identifier(
            generation_id,
            "generation_id",
        )

        return self.generations / generation_id


# ============================================================
# Index Builder
# ============================================================

class RAGIndexBuilder:
    def __init__(
        self,
        config: RAGIndexConfig,
    ) -> None:
        config.validate()

        self.config = config

        self.paths = IndexPaths(
            config.output_directory,
            config.tenant_id,
            config.collection,
        )

        self.parser = DocumentParser()

        self.chunker = TokenChunker(
            config.chunk_size,
            config.chunk_overlap,
        )

    def _read_existing_documents(
        self,
    ) -> dict[str, dict[str, Any]]:
        if not self.config.incremental:
            return {}

        if not self.paths.current_manifest.exists():
            return {}

        current = load_json(
            self.paths.current_manifest
        )

        generation_path = self.paths.generation(
            current["generation_id"]
        )

        documents_path = (
            generation_path / "documents.json"
        )

        if not documents_path.exists():
            return {}

        documents = load_json(
            documents_path
        )

        return {
            document["source_path"]: document
            for document in documents
        }

    def build(self) -> dict[str, Any]:
        started = time.perf_counter()

        self.paths.create()

        input_root = canonical_path(
            self.config.input_directory
        )

        files = discover_documents(
            input_root,
            recursive=self.config.recursive,
        )

        if not files:
            raise DocumentParsingError(
                "No supported documents found"
            )

        LOGGER.info(
            "Discovered %d documents",
            len(files),
        )

        previous_documents = (
            self._read_existing_documents()
        )

        documents: list[ParsedDocument] = []

        unchanged_count = 0
        changed_count = 0

        for path in files:
            relative_path = path.relative_to(
                input_root
            ).as_posix()

            current_hash = sha256_file(
                path
            )

            previous = previous_documents.get(
                relative_path
            )

            if (
                previous
                and previous.get("source_sha256")
                == current_hash
            ):
                unchanged_count += 1
            else:
                changed_count += 1

            try:
                document = self.parser.parse(
                    path,
                    root=input_root,
                    tenant_id=self.config.tenant_id,
                    collection=self.config.collection,
                    allowed_roles=self.config.allowed_roles,
                    max_file_bytes=self.config.max_file_bytes,
                )

                documents.append(
                    document
                )

            except DocumentParsingError as exc:
                LOGGER.warning(
                    "Skipping %s: %s",
                    path,
                    exc,
                )

        if not documents:
            raise DocumentParsingError(
                "No documents could be parsed"
            )

        chunks: list[DocumentChunk] = []

        for document in documents:
            chunks.extend(
                self.chunker.chunk(
                    document
                )
            )

        if not chunks:
            raise RAGIndexError(
                "No searchable chunks generated"
            )

        LOGGER.info(
            "Generated %d chunks",
            len(chunks),
        )

        # A new generation is constructed completely before
        # changing the active manifest. This implementation
        # rebuilds embeddings; unchanged_count is diagnostic.
        # It does not yet reuse previously computed vectors.
        embedder = EmbeddingModel(
            self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
        )

        embeddings = embedder.encode(
            [
                chunk.text
                for chunk in chunks
            ]
        )

        dense_index = DenseVectorIndex()
        dense_index.build(
            embeddings
        )

        generation_id = uuid.uuid4().hex

        generation_path = self.paths.generation(
            generation_id
        )

        ensure_private_directory(
            generation_path
        )

        try:
            chunks_path = (
                generation_path / "chunks.json"
            )

            documents_path = (
                generation_path / "documents.json"
            )

            config_path = (
                generation_path / "config.json"
            )

            vectors_path = (
                generation_path / "vectors.faiss"
            )

            metadata_path = (
                generation_path / "metadata.json"
            )

            safe_json_dump(
                chunks_path,
                [
                    asdict(chunk)
                    for chunk in chunks
                ],
            )

            safe_json_dump(
                documents_path,
                [
                    {
                        **asdict(document),
                        "content": "",
                    }
                    for document in documents
                ],
            )

            safe_json_dump(
                config_path,
                asdict(self.config),
            )

            dense_index.save(
                vectors_path
            )

            metadata = {
                "schema_version": SCHEMA_VERSION,
                "generation_id": generation_id,
                "tenant_id": self.config.tenant_id,
                "collection": self.config.collection,
                "embedding_model": (
                    self.config.embedding_model
                ),
                "embedding_dimension": int(
                    embeddings.shape[1]
                ),
                "document_count": len(documents),
                "chunk_count": len(chunks),
                "unchanged_document_count": (
                    unchanged_count
                ),
                "changed_document_count": (
                    changed_count
                ),
                "created_at": utc_now(),
                "simulation_only": True,
            }

            safe_json_dump(
                metadata_path,
                metadata,
            )

            checksums = {}

            for file_path in (
                chunks_path,
                documents_path,
                config_path,
                vectors_path,
                metadata_path,
            ):
                checksums[file_path.name] = (
                    sha256_file(file_path)
                )

            manifest = {
                "schema_version": SCHEMA_VERSION,
                "generation_id": generation_id,
                "created_at": utc_now(),
                "checksums": checksums,
            }

            safe_json_dump(
                generation_path / "manifest.json",
                manifest,
            )

            # Promote only after all artifacts are written.
            safe_json_dump(
                self.paths.current_manifest,
                {
                    "schema_version": SCHEMA_VERSION,
                    "generation_id": generation_id,
                    "promoted_at": utc_now(),
                },
            )

        except Exception:
            shutil.rmtree(
                generation_path,
                ignore_errors=True,
            )
            raise

        duration = (
            time.perf_counter() - started
        )

        report = {
            "status": "completed",
            "tenant_id": self.config.tenant_id,
            "collection": self.config.collection,
            "generation_id": generation_id,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "unchanged_documents": unchanged_count,
            "changed_documents": changed_count,
            "duration_seconds": round(
                duration,
                3,
            ),
            "generation_path": str(
                generation_path
            ),
            "simulation_only": True,
        }

        LOGGER.info(
            "Index build complete: %d documents, %d chunks",
            len(documents),
            len(chunks),
        )

        return report


# ============================================================
# Index Verification
# ============================================================

def verify_index(
    paths: IndexPaths,
) -> dict[str, Any]:
    if not paths.current_manifest.exists():
        raise IndexIntegrityError(
            "Active index manifest not found"
        )

    current = load_json(
        paths.current_manifest
    )

    generation_id = current[
        "generation_id"
    ]

    generation_path = paths.generation(
        generation_id
    )

    manifest_path = (
        generation_path / "manifest.json"
    )

    if not manifest_path.is_file():
        raise IndexIntegrityError(
            "Generation manifest missing"
        )

    manifest = load_json(
        manifest_path
    )

    checksums = manifest.get(
        "checksums",
        {},
    )

    for filename, expected_hash in checksums.items():
        if Path(filename).name != filename:
            raise IndexIntegrityError(
                "Invalid artifact filename"
            )

        artifact_path = (
            generation_path / filename
        )

        if not artifact_path.is_file():
            raise IndexIntegrityError(
                f"Missing artifact: {filename}"
            )

        actual_hash = sha256_file(
            artifact_path
        )

        if actual_hash != expected_hash:
            raise IndexIntegrityError(
                f"Checksum mismatch: {filename}"
            )

    metadata = load_json(
        generation_path / "metadata.json"
    )

    chunks = load_json(
        generation_path / "chunks.json"
    )

    if len(chunks) != metadata["chunk_count"]:
        raise IndexIntegrityError(
            "Chunk count mismatch"
        )

    if metadata["tenant_id"] != paths.root.parent.name:
        raise IndexIntegrityError(
            "Tenant mismatch"
        )

    if metadata["collection"] != paths.root.name:
        raise IndexIntegrityError(
            "Collection mismatch"
        )

    return {
        "valid": True,
        "generation_id": generation_id,
        "document_count": metadata[
            "document_count"
        ],
        "chunk_count": len(chunks),
        "verified_artifacts": len(checksums),
    }


# ============================================================
# Search Engine
# ============================================================

class HybridRAGSearch:
    def __init__(
        self,
        *,
        output_directory: str,
        tenant_id: str,
        collection: str,
        role: str,
        reranker_model: str | None = None,
    ) -> None:
        validate_identifier(
            role,
            "role",
        )

        self.tenant_id = tenant_id
        self.collection = collection
        self.role = role

        self.paths = IndexPaths(
            output_directory,
            tenant_id,
            collection,
        )

        verify_index(
            self.paths
        )

        current = load_json(
            self.paths.current_manifest
        )

        generation_path = self.paths.generation(
            current["generation_id"]
        )

        self.config = RAGIndexConfig(
            **load_json(
                generation_path / "config.json"
            )
        )

        self.chunks = [
            DocumentChunk(**item)
            for item in load_json(
                generation_path / "chunks.json"
            )
        ]

        self.embedder = EmbeddingModel(
            self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
        )

        self.dense_index = DenseVectorIndex()

        self.dense_index.load(
            generation_path / "vectors.faiss"
        )

        if self.dense_index.index.ntotal != len(self.chunks):
            raise IndexIntegrityError(
                "FAISS vector count differs from chunk count"
            )

        self.sparse_index = SparseBM25Index(
            [
                chunk.text
                for chunk in self.chunks
            ]
        )

        self.reranker = (
            CrossEncoderReranker(
                reranker_model
            )
            if reranker_model
            else None
        )

    def _authorized(
        self,
        chunk: DocumentChunk,
    ) -> bool:
        return (
            chunk.tenant_id == self.tenant_id
            and chunk.collection == self.collection
            and self.role in chunk.allowed_roles
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        query = normalize_text(
            query
        )

        if not query:
            raise ValueError(
                "Search query cannot be empty"
            )

        if len(query) > 10_000:
            raise ValueError(
                "Search query is too long"
            )

        top_k = (
            top_k
            if top_k is not None
            else self.config.top_k
        )

        if top_k < 1:
            raise ValueError(
                "top_k must be positive"
            )

        # The index is tenant-specific. ACL filtering is
        # performed before ranking results are exposed.
        # For large multi-role collections, use per-ACL
        # indexes or an ACL-aware vector backend.
        authorized_indices = [
            index
            for index, chunk in enumerate(self.chunks)
            if self._authorized(chunk)
        ]

        if not authorized_indices:
            return []

        authorized_set = set(
            authorized_indices
        )

        query_embedding = self.embedder.encode(
            [query]
        )[0]

        # Search all vectors to avoid underfilling results
        # after role filtering. This simple strategy favors
        # correctness over scalability.
        dense_results = [
            (index, score)
            for index, score in self.dense_index.search(
                query_embedding,
                len(self.chunks),
            )
            if index in authorized_set
        ]

        sparse_results = [
            (index, score)
            for index, score in self.sparse_index.search(
                query,
                len(self.chunks),
            )
            if index in authorized_set
        ]

        fused = reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
            rrf_k=self.config.rrf_k,
        )

        candidate_limit = max(
            top_k * 4,
            top_k,
        )

        results = []

        for rank, (index, score) in enumerate(
            fused[:candidate_limit],
            start=1,
        ):
            chunk = self.chunks[index]

            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    source_path=chunk.source_path,
                    text=chunk.text,
                    citation=chunk.citation,
                    score=float(score),
                    rank=rank,
                    metadata=chunk.metadata,
                )
            )

        if self.reranker:
            results = self.reranker.rerank(
                query,
                results,
            )

        results = results[:top_k]

        for rank, result in enumerate(
            results,
            start=1,
        ):
            result.rank = rank

        return results


# ============================================================
# Retrieval Evaluation
# ============================================================

def evaluate_retrieval(
    search_engine: HybridRAGSearch,
    questions_path: Path,
    *,
    top_k: int,
) -> dict[str, Any]:
    """
    Expected evaluation JSON:

    [
      {
        "question": "What is error E102?",
        "relevant_document_ids": ["..."]
      }
    ]
    """
    questions = load_json(
        questions_path
    )

    if not isinstance(questions, list):
        raise ValueError(
            "Evaluation file must contain a JSON array"
        )

    if not questions:
        raise ValueError(
            "Evaluation dataset is empty"
        )

    recalls = []
    reciprocal_ranks = []

    for item in questions:
        question = str(
            item["question"]
        )

        relevant = set(
            item["relevant_document_ids"]
        )

        if not relevant:
            continue

        results = search_engine.search(
            question,
            top_k=top_k,
        )

        retrieved = [
            result.document_id
            for result in results
        ]

        hits = relevant.intersection(
            retrieved
        )

        recall = (
            len(hits) / len(relevant)
        )

        reciprocal_rank = 0.0

        for rank, document_id in enumerate(
            retrieved,
            start=1,
        ):
            if document_id in relevant:
                reciprocal_rank = 1.0 / rank
                break

        recalls.append(
            recall
        )

        reciprocal_ranks.append(
            reciprocal_rank
        )

    if not recalls:
        raise ValueError(
            "No evaluation questions with relevant documents"
        )

    return {
        "question_count": len(recalls),
        "top_k": top_k,
        "recall_at_k": float(
            np.mean(recalls)
        ),
        "mrr_at_k": float(
            np.mean(reciprocal_ranks)
        ),
    }


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and search a robotics RAG index"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    build_parser = subparsers.add_parser(
        "build"
    )

    build_parser.add_argument(
        "--input",
        required=True,
    )

    build_parser.add_argument(
        "--output",
        default="storage/rag_index",
    )

    build_parser.add_argument(
        "--tenant",
        default="demo",
    )

    build_parser.add_argument(
        "--collection",
        default="robotics",
    )

    build_parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
    )

    build_parser.add_argument(
        "--chunk-size",
        type=int,
        default=350,
    )

    build_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=60,
    )

    build_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    build_parser.add_argument(
        "--allowed-roles",
        nargs="+",
        default=[
            "admin",
            "engineer",
            "researcher",
        ],
    )

    build_parser.add_argument(
        "--full-rebuild",
        action="store_true",
    )

    search_parser = subparsers.add_parser(
        "search"
    )

    search_parser.add_argument(
        "--output",
        default="storage/rag_index",
    )

    search_parser.add_argument(
        "--tenant",
        required=True,
    )

    search_parser.add_argument(
        "--collection",
        required=True,
    )

    search_parser.add_argument(
        "--role",
        default="engineer",
    )

    search_parser.add_argument(
        "--query",
        required=True,
    )

    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    search_parser.add_argument(
        "--reranker-model",
        default=None,
    )

    verify_parser = subparsers.add_parser(
        "verify"
    )

    verify_parser.add_argument(
        "--output",
        default="storage/rag_index",
    )

    verify_parser.add_argument(
        "--tenant",
        required=True,
    )

    verify_parser.add_argument(
        "--collection",
        required=True,
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate"
    )

    evaluate_parser.add_argument(
        "--output",
        default="storage/rag_index",
    )

    evaluate_parser.add_argument(
        "--tenant",
        required=True,
    )

    evaluate_parser.add_argument(
        "--collection",
        required=True,
    )

    evaluate_parser.add_argument(
        "--role",
        default="engineer",
    )

    evaluate_parser.add_argument(
        "--questions",
        required=True,
    )

    evaluate_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "build":
        config = RAGIndexConfig(
            input_directory=args.input,
            output_directory=args.output,
            tenant_id=args.tenant,
            collection=args.collection,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_batch_size=args.batch_size,
            allowed_roles=args.allowed_roles,
            incremental=not args.full_rebuild,
        )

        report = RAGIndexBuilder(
            config
        ).build()

        print(
            json.dumps(
                report,
                indent=2,
            )
        )

        return

    if args.command == "verify":
        paths = IndexPaths(
            args.output,
            args.tenant,
            args.collection,
        )

        report = verify_index(
            paths
        )

        print(
            json.dumps(
                report,
                indent=2,
            )
        )

        return

    if args.command == "search":
        engine = HybridRAGSearch(
            output_directory=args.output,
            tenant_id=args.tenant,
            collection=args.collection,
            role=args.role,
            reranker_model=args.reranker_model,
        )

        results = engine.search(
            args.query,
            top_k=args.top_k,
        )

        print(
            json.dumps(
                [
                    asdict(result)
                    for result in results
                ],
                indent=2,
                ensure_ascii=False,
            )
        )

        return

    if args.command == "evaluate":
        engine = HybridRAGSearch(
            output_directory=args.output,
            tenant_id=args.tenant,
            collection=args.collection,
            role=args.role,
        )

        metrics = evaluate_retrieval(
            engine,
            Path(args.questions),
            top_k=args.top_k,
        )

        print(
            json.dumps(
                metrics,
                indent=2,
            )
        )


if __name__ == "__main__":
    try:
        main()

    except (
        RAGIndexError,
        ValueError,
        FileNotFoundError,
        KeyError,
    ) as exc:
        LOGGER.error(
            "%s",
            exc,
        )
        sys.exit(1)