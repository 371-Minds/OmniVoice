from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional
from urllib import error, parse, request
import uuid

import numpy as np


LOGGER = logging.getLogger(__name__)
DEFAULT_MEMORIA_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3
DEFAULT_BATCH_SIZE = 16


@dataclass
class MemoryRecord:
    user_id: str
    text: str
    metadata: dict[str, Any] | None = None
    mref: str | None = None


@dataclass
class MemoriaConfig:
    mode: str = "off"
    remote_url: str | None = None
    api_key: str | None = None
    user_id: str | None = None
    top_k: int = DEFAULT_TOP_K
    timeout: float = 10.0
    db_path: str | None = None
    onnx_model_path: str | None = None
    tokenizer: str | None = None
    max_length: int = 256
    batch_size: int = DEFAULT_BATCH_SIZE
    model_name: str = DEFAULT_MEMORIA_MODEL

    def normalized(self) -> "MemoriaConfig":
        data = asdict(self)
        data["mode"] = (self.mode or "off").lower()
        data["remote_url"] = self.remote_url.rstrip("/") if self.remote_url else None
        data["db_path"] = self.db_path or str(
            Path.home() / ".cache" / "omnivoice" / "memoria.sqlite3"
        )
        return MemoriaConfig(**data)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self.normalized())


def add_memoria_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("memoria")
    group.add_argument(
        "--memoria-mode",
        type=str,
        default=os.getenv("OMNIVOICE_MEMORIA_MODE", "off"),
        choices=("off", "remote", "local", "auto"),
        help="Memoria backend mode.",
    )
    group.add_argument(
        "--memoria-url",
        type=str,
        default=os.getenv("MEMORIA_URL"),
        help="Base URL of a Memoria API deployment.",
    )
    group.add_argument(
        "--memoria-api-key",
        type=str,
        default=os.getenv("MEMORIA_API_KEY"),
        help="API key for a remote Memoria deployment.",
    )
    group.add_argument(
        "--memoria-user-id",
        type=str,
        default=os.getenv("MEMORIA_USER_ID"),
        help="Default Memoria user id.",
    )
    group.add_argument(
        "--memoria-query",
        type=str,
        default=None,
        help="Query used to retrieve contextual memories before generation.",
    )
    group.add_argument(
        "--memoria-store-text",
        type=str,
        default=None,
        help="Memory text to store after generation.",
    )
    group.add_argument(
        "--memoria-top-k",
        type=int,
        default=int(os.getenv("MEMORIA_TOP_K", str(DEFAULT_TOP_K))),
        help="Maximum number of memories to retrieve.",
    )
    group.add_argument(
        "--memoria-timeout",
        type=float,
        default=float(os.getenv("MEMORIA_TIMEOUT", "10.0")),
        help="Remote Memoria request timeout in seconds.",
    )
    group.add_argument(
        "--memoria-db-path",
        type=str,
        default=os.getenv("MEMORIA_DB_PATH"),
        help="SQLite path for embedded Memoria storage and embedding cache.",
    )
    group.add_argument(
        "--memoria-onnx-model",
        type=str,
        default=os.getenv("MEMORIA_ONNX_MODEL"),
        help="Path to an ONNX embedding model for embedded/offline mode.",
    )
    group.add_argument(
        "--memoria-tokenizer",
        type=str,
        default=os.getenv("MEMORIA_TOKENIZER"),
        help="Tokenizer path or Hugging Face id for the ONNX embedding model.",
    )
    group.add_argument(
        "--memoria-max-length",
        type=int,
        default=int(os.getenv("MEMORIA_MAX_LENGTH", "256")),
        help="Maximum tokenizer sequence length for embedded mode.",
    )
    group.add_argument(
        "--memoria-batch-size",
        type=int,
        default=int(os.getenv("MEMORIA_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
        help="Batch size for embedded/local embedding generation.",
    )
    group.add_argument(
        "--memoria-model-name",
        type=str,
        default=os.getenv("MEMORIA_MODEL_NAME", DEFAULT_MEMORIA_MODEL),
        help="Identifier used for cache keying in embedded mode.",
    )
    return parser


def build_memoria_config(args: argparse.Namespace) -> MemoriaConfig:
    return MemoriaConfig(
        mode=args.memoria_mode,
        remote_url=args.memoria_url,
        api_key=args.memoria_api_key,
        user_id=args.memoria_user_id,
        top_k=args.memoria_top_k,
        timeout=args.memoria_timeout,
        db_path=args.memoria_db_path,
        onnx_model_path=args.memoria_onnx_model,
        tokenizer=args.memoria_tokenizer,
        max_length=args.memoria_max_length,
        batch_size=args.memoria_batch_size,
        model_name=args.memoria_model_name,
    ).normalized()


class LocalOnnxEmbedder:
    def __init__(
        self,
        onnx_model_path: str,
        tokenizer: str | None = None,
        max_length: int = 256,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Embedded Memoria requires onnxruntime. Install omnivoice[memory]."
            ) from exc

        from transformers import AutoTokenizer

        model_path = self._resolve_model_path(onnx_model_path)
        tokenizer_id = tokenizer or str(model_path.parent)
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.max_length = max_length
        self.input_names = {item.name for item in self.session.get_inputs()}

    def _resolve_model_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_file():
            return path
        candidates = [
            path / "model.onnx",
            path / "onnx" / "model.onnx",
            path / "model_fp16.onnx",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        matches = sorted(path.glob("**/*.onnx"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Could not find an ONNX model under {path}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        ort_inputs: dict[str, np.ndarray] = {}
        for name in self.input_names:
            if name in tokens:
                ort_inputs[name] = tokens[name].astype(np.int64)
        outputs = self.session.run(None, ort_inputs)
        hidden = np.asarray(outputs[0], dtype=np.float32)
        attention_mask = np.asarray(tokens["attention_mask"], dtype=np.float32)
        mask = attention_mask[..., None]
        pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1.0, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.maximum(norms, 1e-12)
        return pooled.astype(np.float32).tolist()


class MemoriaManager:
    def __init__(
        self,
        config: MemoriaConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config.normalized()
        self.logger = logger or LOGGER
        self._db_ready = False
        self._embedder: LocalOnnxEmbedder | None = None
        self.backend = self._resolve_backend()

    @property
    def enabled(self) -> bool:
        return self.backend != "off"

    def _resolve_backend(self) -> str:
        if self.config.mode == "off":
            return "off"
        if self.config.mode == "remote":
            return "remote"
        if self.config.mode == "local":
            return "local"
        if self.config.remote_url and self.config.api_key:
            try:
                self._remote_healthcheck()
                return "remote"
            except Exception as exc:
                self.logger.warning("Memoria remote unavailable, falling back: %s", exc)
        if self.config.onnx_model_path:
            return "local"
        self.logger.warning(
            "Memoria auto mode disabled because neither remote credentials nor a local ONNX model were configured."
        )
        return "off"

    def enrich_instruct(
        self,
        instruct: str | None,
        *,
        user_id: str | None = None,
        query: str | None = None,
        top_k: int | None = None,
    ) -> tuple[str | None, list[str]]:
        effective_user = user_id or self.config.user_id
        if not self.enabled or not effective_user or not query:
            return instruct, []
        contexts = self.retrieve_context(
            user_id=effective_user,
            query=query,
            top_k=top_k or self.config.top_k,
        )
        if not contexts:
            return instruct, []
        memory_block = "Retrieved user memory:\n" + "\n".join(
            f"- {item}" for item in contexts
        )
        merged = (
            f"{instruct.strip()}\n\n{memory_block}"
            if instruct and instruct.strip()
            else memory_block
        )
        return merged, contexts

    def retrieve_context(self, user_id: str, query: str, top_k: int) -> list[str]:
        if self.backend == "remote":
            return self._remote_retrieve_context(user_id=user_id, query=query, top_k=top_k)
        if self.backend == "local":
            return self._local_retrieve_context(user_id=user_id, query=query, top_k=top_k)
        return []

    def store_text_async(
        self,
        *,
        user_id: str | None,
        text: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        effective_user = user_id or self.config.user_id
        if not self.enabled or not effective_user or not text or not text.strip():
            return None
        record = MemoryRecord(
            user_id=effective_user,
            text=text.strip(),
            metadata=metadata,
            mref=self._make_mref(),
        )
        return self.store_many_async([record])[0]

    def store_many_async(self, records: Iterable[MemoryRecord]) -> list[str]:
        records_list = [record for record in records if record.text and record.user_id]
        if not self.enabled or not records_list:
            return []
        prepared = []
        for record in records_list:
            mref = record.mref or self._make_mref()
            prepared.append(
                {
                    "user_id": record.user_id,
                    "text": record.text.strip(),
                    "metadata": record.metadata or {},
                    "mref": mref,
                }
            )
            self._record_async_status(mref=mref, status="queued")
        payload = {
            "config": self.config.to_payload(),
            "records": prepared,
        }
        encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
        kwargs: dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "start_new_session": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(
            [sys.executable, "-m", "omnivoice.integrations.memoria_worker", encoded],
            **kwargs,
        )
        return [item["mref"] for item in prepared]

    def store_many_sync(self, records: Iterable[MemoryRecord]) -> list[dict[str, Any]]:
        records_list = [record for record in records if record.text and record.user_id]
        if not records_list:
            return []
        if self.backend == "remote":
            return self._remote_store_many(records_list)
        if self.backend == "local":
            return self._local_store_many(records_list)
        return []

    def _make_mref(self) -> str:
        return f"mref_{uuid.uuid4().hex[:8]}"

    def _remote_healthcheck(self) -> None:
        if not self.config.remote_url:
            raise RuntimeError("Missing Memoria URL")
        self._request_json("GET", f"{self.config.remote_url}/api/health", authenticated=False)

    def _remote_retrieve_context(self, user_id: str, query: str, top_k: int) -> list[str]:
        params = parse.urlencode({"query": query, "topK": top_k})
        response = self._request_json(
            "GET",
            f"{self.config.remote_url}/api/memory/{parse.quote(user_id)}/context?{params}",
        )
        return [item for item in response.get("context", []) if isinstance(item, str)]

    def _remote_store_many(self, records: list[MemoryRecord]) -> list[dict[str, Any]]:
        stored = []
        for record in records:
            payload = {"text": record.text}
            if record.metadata:
                payload["actionPayload"] = record.metadata
            response = self._request_json(
                "POST",
                f"{self.config.remote_url}/api/memory/{parse.quote(record.user_id)}",
                body=payload,
            )
            self._record_async_status(
                mref=record.mref or self._make_mref(),
                status="stored",
                provider_id=response.get("id"),
            )
            stored.append(response)
        return stored

    def _local_retrieve_context(self, user_id: str, query: str, top_k: int) -> list[str]:
        query_embedding = np.asarray(self._embed_texts([query])[0], dtype=np.float32)
        with self._db() as conn:
            rows = conn.execute(
                """
                SELECT text, embedding_json
                FROM memories
                WHERE user_id = ? AND text IS NOT NULL
                """,
                (user_id,),
            ).fetchall()
        if not rows:
            return []
        texts = [row["text"] for row in rows]
        matrix = np.asarray(
            [json.loads(row["embedding_json"]) for row in rows],
            dtype=np.float32,
        )
        denom = np.maximum(
            np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_embedding),
            1e-12,
        )
        scores = np.dot(matrix, query_embedding) / denom
        indices = np.argsort(scores)[::-1][:top_k]
        return [texts[index] for index in indices]

    def _local_store_many(self, records: list[MemoryRecord]) -> list[dict[str, Any]]:
        texts = [record.text for record in records]
        embeddings = self._embed_texts(texts)
        created_at = time.time()
        stored = []
        with self._db() as conn:
            for record, embedding in zip(records, embeddings):
                memory_id = str(uuid.uuid4())
                mref = record.mref or self._make_mref()
                conn.execute(
                    """
                    INSERT INTO memories (
                        id, mref, user_id, text, metadata_json, embedding_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        mref,
                        record.user_id,
                        record.text,
                        json.dumps(record.metadata or {}, ensure_ascii=False),
                        json.dumps(embedding),
                        created_at,
                    ),
                )
                self._record_async_status(
                    mref=mref,
                    status="stored",
                    provider_id=memory_id,
                    conn=conn,
                )
                stored.append({"id": memory_id, "mref": mref})
        return stored

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        cache_hits: dict[int, list[float]] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        with self._db() as conn:
            for index, text in enumerate(texts):
                cached = self._lookup_embedding_cache(conn, text)
                if cached is None:
                    uncached_indices.append(index)
                    uncached_texts.append(text)
                else:
                    cache_hits[index] = cached
        generated: dict[int, list[float]] = {}
        if uncached_texts:
            embedder = self._get_embedder()
            for offset in range(0, len(uncached_texts), self.config.batch_size):
                batch = uncached_texts[offset : offset + self.config.batch_size]
                batch_embeddings = embedder.embed_texts(batch)
                with self._db() as conn:
                    for inner_index, (text, embedding) in enumerate(
                        zip(batch, batch_embeddings)
                    ):
                        absolute_index = uncached_indices[offset + inner_index]
                        generated[absolute_index] = embedding
                        self._store_embedding_cache(conn, text, embedding)
        result = []
        for index in range(len(texts)):
            result.append(cache_hits.get(index) or generated[index])
        return result

    def _get_embedder(self) -> LocalOnnxEmbedder:
        if self._embedder is None:
            if not self.config.onnx_model_path:
                raise RuntimeError(
                    "Embedded Memoria requires --memoria-onnx-model or MEMORIA_ONNX_MODEL."
                )
            self._embedder = LocalOnnxEmbedder(
                self.config.onnx_model_path,
                tokenizer=self.config.tokenizer,
                max_length=self.config.max_length,
            )
        return self._embedder

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256()
        digest.update(self.config.model_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.strip().encode("utf-8"))
        return digest.hexdigest()

    def _lookup_embedding_cache(
        self, conn: sqlite3.Connection, text: str
    ) -> list[float] | None:
        row = conn.execute(
            """
            SELECT embedding_json
            FROM embedding_cache
            WHERE cache_key = ?
            """,
            (self._cache_key(text),),
        ).fetchone()
        return None if row is None else json.loads(row["embedding_json"])

    def _store_embedding_cache(
        self, conn: sqlite3.Connection, text: str, embedding: list[float]
    ) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO embedding_cache (
                cache_key, model_name, text, embedding_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                self._cache_key(text),
                self.config.model_name,
                text,
                json.dumps(embedding),
                time.time(),
            ),
        )

    def _record_async_status(
        self,
        *,
        mref: str,
        status: str,
        provider_id: str | None = None,
        error_message: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        owns_connection = conn is None
        if conn is None:
            conn = self._db()
        try:
            conn.execute(
                """
                INSERT INTO async_receipts (
                    mref, backend, status, provider_id, error_message, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(mref) DO UPDATE SET
                    backend = excluded.backend,
                    status = excluded.status,
                    provider_id = excluded.provider_id,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    mref,
                    self.backend,
                    status,
                    provider_id,
                    error_message,
                    time.time(),
                ),
            )
            if owns_connection:
                conn.commit()
        finally:
            if owns_connection:
                conn.close()

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        body: dict[str, Any] | None = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if authenticated and self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload = None if body is None else json.dumps(body).encode("utf-8")
        req = request.Request(url, method=method, headers=headers, data=payload)
        try:
            with request.urlopen(req, timeout=self.config.timeout) as response:
                data = response.read().decode("utf-8")
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Memoria request failed ({exc.code}): {message or exc.reason}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Memoria request failed: {exc.reason}") from exc
        return json.loads(data) if data else {}

    def _db(self) -> sqlite3.Connection:
        self._ensure_db()
        conn = sqlite3.connect(self.config.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        if self._db_ready:
            return
        db_path = Path(self.config.db_path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path, timeout=30.0) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    mref TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS async_receipts (
                    mref TEXT PRIMARY KEY,
                    backend TEXT NOT NULL,
                    status TEXT NOT NULL,
                    provider_id TEXT,
                    error_message TEXT,
                    updated_at REAL NOT NULL
                )
                """
            )
        self._db_ready = True


def run_async_store_job(payload: dict[str, Any]) -> int:
    config = MemoriaConfig(**payload["config"]).normalized()
    manager = MemoriaManager(config)
    records = [
        MemoryRecord(
            user_id=item["user_id"],
            text=item["text"],
            metadata=item.get("metadata") or None,
            mref=item.get("mref"),
        )
        for item in payload.get("records", [])
    ]
    try:
        manager.store_many_sync(records)
        return 0
    except Exception as exc:
        for record in records:
            if record.mref:
                manager._record_async_status(
                    mref=record.mref,
                    status="failed",
                    error_message=str(exc),
                )
        raise
