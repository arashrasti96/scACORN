from __future__ import annotations

import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CONTRASTIVE_ROOT = SCRIPT_DIR.parent.parent
if str(CONTRASTIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTRASTIVE_ROOT))

from utils import ProjectionHead, build_c2s_prompt, get_hidden_states  # noqa: E402


DEFAULT_2B_MODEL = os.getenv("GEMMA_MODEL_PATH", "vandijklab/C2S-Scale-Gemma-2-2B")
DEFAULT_STAGE1_OUTPUTS = SCRIPT_DIR.parent / "outputs"


def _threadpool_limit_context(limit: int = 1):
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=limit)
    except Exception:
        return nullcontext()


class EmbeddingMethod:
    method_key = "base"
    display_name = "Base"

    def fit(self, train_genes_lists: list[list[str]]) -> None:
        return None

    def embed(self, genes_lists: list[list[str]]) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"method_key": self.method_key, "display_name": self.display_name}

    def close(self) -> None:
        return None


class RankPcaMethod(EmbeddingMethod):
    method_key = "rank_pca"
    display_name = "Rank PCA"

    def __init__(self, n_components: int = 128) -> None:
        from sklearn.decomposition import PCA

        self._pca_cls = PCA
        self._n_components = n_components
        self._vocab: dict[str, int] = {}
        self._pca = None

    def fit(self, train_genes_lists: list[list[str]]) -> None:
        self._vocab = build_vocab(train_genes_lists)
        train_matrix = build_rank_matrix(train_genes_lists, self._vocab)
        component_count = min(self._n_components, train_matrix.shape[0], train_matrix.shape[1])
        if component_count < 1:
            raise ValueError("Rank PCA requires at least one cell and one gene")
        self._pca = self._pca_cls(n_components=component_count, random_state=42, svd_solver="randomized")
        with _threadpool_limit_context(1):
            self._pca.fit(train_matrix)

    def embed(self, genes_lists: list[list[str]]) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("Rank PCA has not been fit on the train split")
        matrix = build_rank_matrix(genes_lists, self._vocab)
        with _threadpool_limit_context(1):
            return self._pca.transform(matrix).astype(np.float32)

    def describe(self) -> dict:
        payload = super().describe()
        payload.update({"n_components": self._n_components})
        return payload


class TfidfSvdMethod(EmbeddingMethod):
    method_key = "tfidf_svd"
    display_name = "TF-IDF + SVD"

    def __init__(self, n_components: int = 128) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[A-Za-z0-9\-]+",
            lowercase=False,
        )
        self._svd_cls = TruncatedSVD
        self._svd = None
        self._n_components = n_components

    def fit(self, train_genes_lists: list[list[str]]) -> None:
        train_sentences = gene_sentences(train_genes_lists)
        tfidf_matrix = self._vectorizer.fit_transform(train_sentences)
        component_count = min(self._n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        component_count = max(1, component_count)
        self._svd = self._svd_cls(n_components=component_count, algorithm="randomized", random_state=42)
        with _threadpool_limit_context(1):
            self._svd.fit(tfidf_matrix)

    def embed(self, genes_lists: list[list[str]]) -> np.ndarray:
        if self._svd is None:
            raise RuntimeError("TF-IDF + SVD has not been fit on the train split")
        sentences = gene_sentences(genes_lists)
        tfidf_matrix = self._vectorizer.transform(sentences)
        with _threadpool_limit_context(1):
            return self._svd.transform(tfidf_matrix).astype(np.float32)

    def describe(self) -> dict:
        payload = super().describe()
        payload.update({"n_components": self._n_components})
        return payload


class SentenceTransformerMethod(EmbeddingMethod):
    method_key = "sentence_transformer"
    display_name = "SentenceTransformer"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 256) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)

    def embed(self, genes_lists: list[list[str]]) -> np.ndarray:
        self._load()
        assert self._model is not None
        embeddings = self._model.encode(
            gene_sentences(genes_lists),
            batch_size=self._batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def describe(self) -> dict:
        payload = super().describe()
        payload.update({"model_name": self._model_name, "batch_size": self._batch_size})
        return payload

    def close(self) -> None:
        self._model = None


class C2SBaseMethod(EmbeddingMethod):
    method_key = "c2s_base"
    display_name = "C2S Base"

    def __init__(
        self,
        model_path: str | None,
        batch_size: int,
        max_seq_len: int,
        use_4bit: bool,
    ) -> None:
        self._requested_model_path = model_path
        self._model_path = model_path or DEFAULT_2B_MODEL
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._use_4bit = use_4bit
        self._tokenizer = None
        self._model = None

    def set_model_path(self, model_path: str | None) -> None:
        if model_path:
            self._model_path = model_path

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            if self._use_4bit:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
        self._model = AutoModelForCausalLM.from_pretrained(self._model_path, **load_kwargs)
        self._model.eval()

    def embed(self, genes_lists: list[list[str]]) -> np.ndarray:
        self._load()
        assert self._model is not None and self._tokenizer is not None
        import torch

        embeddings: list[np.ndarray] = []
        total_batches = max(1, (len(genes_lists) + self._batch_size - 1) // self._batch_size)
        embed_started = time.perf_counter()
        with torch.no_grad():
            for batch_index, start in enumerate(range(0, len(genes_lists), self._batch_size), start=1):
                batch = genes_lists[start : start + self._batch_size]
                prompts = [build_c2s_prompt(genes) for genes in batch]
                hidden = get_hidden_states(
                    self._model,
                    self._tokenizer,
                    prompts,
                    self._max_seq_len,
                    enable_grad=False,
                )
                embeddings.append(hidden.float().cpu().numpy())
                if batch_index == 1 or batch_index == total_batches or batch_index % 25 == 0:
                    elapsed = time.perf_counter() - embed_started
                    print(
                        f"  C2S Base batch {batch_index}/{total_batches} ({len(batch)} cells, {elapsed:.1f}s elapsed)",
                        flush=True,
                    )
        return np.concatenate(embeddings, axis=0).astype(np.float32)

    def describe(self) -> dict:
        payload = super().describe()
        payload.update(
            {
                "model_path": self._model_path,
                "batch_size": self._batch_size,
                "max_seq_len": self._max_seq_len,
                "use_4bit": self._use_4bit,
            }
        )
        return payload

    def close(self) -> None:
        import gc

        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class C2SLoRAMethod(EmbeddingMethod):
    display_name = "C2S + LoRA"

    def __init__(
        self,
        dataset_name: str,
        outputs_root: str | Path,
        batch_size: int,
        max_seq_len: int,
        use_4bit: bool,
        checkpoint_tag: str = "best",
        return_projections: bool = False,
    ) -> None:
        self.method_key = "c2s_lora_projections" if return_projections else "c2s_lora_features"
        self.display_name = "C2S + LoRA Projections" if return_projections else "C2S + LoRA Features"
        self._dataset_name = dataset_name
        self._outputs_root = Path(outputs_root)
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._use_4bit = use_4bit
        self._checkpoint_tag = checkpoint_tag
        self._return_projections = return_projections
        self._checkpoint_dir = resolve_lora_checkpoint_dir(dataset_name, self._outputs_root, checkpoint_tag=checkpoint_tag)
        self._adapter_dir = self._checkpoint_dir / "gemma_lora_stage1"
        self._projection_head_path = self._checkpoint_dir / "projection_head.pt"
        self._metadata_path = self._checkpoint_dir / "metadata.json"
        self._model_path = infer_base_model_path(self._adapter_dir, fallback=DEFAULT_2B_MODEL)
        self._tokenizer = None
        self._model = None
        self._projection_head = None

    def base_model_path(self) -> str:
        return self._model_path

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._projection_head is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            if self._use_4bit:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
        base_model = AutoModelForCausalLM.from_pretrained(self._model_path, **load_kwargs)
        self._model = PeftModel.from_pretrained(base_model, str(self._adapter_dir))
        self._model.eval()

        metadata = {}
        if self._metadata_path.is_file():
            with open(self._metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        projection_payload = torch.load(self._projection_head_path, map_location="cpu") if self._projection_head_path.is_file() else None
        projection_state_dict, projection_config = unpack_projection_head_checkpoint(projection_payload)

        hidden_size = int(projection_config.get("input_dim", metadata.get("hidden_size", 2304)))
        projection_hidden_dim = int(projection_config.get("hidden_dim", hidden_size // 2))
        projection_dim = int(projection_config.get("output_dim", metadata.get("args", {}).get("projection_dim", 128)))
        projection_head = ProjectionHead(
            input_dim=hidden_size,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
        )
        if projection_state_dict is not None:
            projection_head.load_state_dict(projection_state_dict)
        device = next(self._model.parameters()).device
        self._projection_head = projection_head.to(device)
        self._projection_head.eval()

    def embed(self, genes_lists: list[list[str]]) -> np.ndarray:
        self._load()
        assert self._model is not None and self._tokenizer is not None and self._projection_head is not None
        import torch

        features: list[np.ndarray] = []
        projections: list[np.ndarray] = []
        total_batches = max(1, (len(genes_lists) + self._batch_size - 1) // self._batch_size)
        embed_started = time.perf_counter()
        with torch.no_grad():
            for batch_index, start in enumerate(range(0, len(genes_lists), self._batch_size), start=1):
                batch = genes_lists[start : start + self._batch_size]
                prompts = [build_c2s_prompt(genes) for genes in batch]
                hidden = get_hidden_states(
                    self._model,
                    self._tokenizer,
                    prompts,
                    self._max_seq_len,
                    enable_grad=False,
                )
                projected = self._projection_head(hidden)
                features.append(hidden.float().cpu().numpy())
                projections.append(projected.float().cpu().numpy())
                if batch_index == 1 or batch_index == total_batches or batch_index % 25 == 0:
                    elapsed = time.perf_counter() - embed_started
                    label = "C2S + LoRA Proj" if self._return_projections else "C2S + LoRA Feat"
                    print(
                        f"  {label} batch {batch_index}/{total_batches} ({len(batch)} cells, {elapsed:.1f}s elapsed)",
                        flush=True,
                    )

        feature_matrix = np.concatenate(features, axis=0).astype(np.float32)
        projection_matrix = np.concatenate(projections, axis=0).astype(np.float32)
        return projection_matrix if self._return_projections else feature_matrix

    def describe(self) -> dict:
        payload = super().describe()
        payload.update(
            {
                "dataset_name": self._dataset_name,
                "checkpoint_dir": str(self._checkpoint_dir),
                "model_path": self._model_path,
                "checkpoint_tag": self._checkpoint_tag,
                "return_projections": self._return_projections,
                "batch_size": self._batch_size,
                "max_seq_len": self._max_seq_len,
                "use_4bit": self._use_4bit,
            }
        )
        return payload

    def close(self) -> None:
        import gc

        self._model = None
        self._tokenizer = None
        self._projection_head = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class ScGPTMethod(EmbeddingMethod):
    method_key = "scgpt"
    display_name = "scGPT"

    def __init__(self, model_dir: str, batch_size: int) -> None:
        self._model_dir = model_dir
        self._batch_size = batch_size
        self._model = None
        self._vocab = None
        self._pad_token = "<pad>"
        self._pad_value = -2
        self._device = None
        self._gene_ids_in_vocab: list[int] = []

    def _build_pseudo_anndata(self, genes_lists: list[list[str]]):
        import anndata as ad
        import pandas as pd
        import scipy.sparse as sp

        vocab: dict[str, int] = {}
        for genes in genes_lists:
            for gene in genes:
                if gene not in vocab:
                    vocab[gene] = len(vocab)

        gene_names = [""] * len(vocab)
        for gene, index in vocab.items():
            gene_names[index] = gene

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for row_index, genes in enumerate(genes_lists):
            gene_count = len(genes)
            for rank, gene in enumerate(genes):
                rows.append(row_index)
                cols.append(vocab[gene])
                vals.append(float(gene_count - rank))

        matrix = sp.csr_matrix((vals, (rows, cols)), shape=(len(genes_lists), len(vocab)), dtype=np.float32)
        return ad.AnnData(X=matrix, var=pd.DataFrame(index=gene_names))

    def _load(self, genes_lists: list[list[str]]) -> None:
        if self._model is not None:
            return
        import json as _json
        import torch
        from pathlib import Path as _Path
        from scgpt.model import TransformerModel
        from scgpt.tokenizer.gene_tokenizer import GeneVocab
        from scgpt.utils import set_seed

        model_path = _Path(self._model_dir)
        if not model_path.is_dir():
            raise FileNotFoundError(f"scGPT model directory not found: {model_path}")

        adata = self._build_pseudo_anndata(genes_lists)
        vocab = GeneVocab.from_file(model_path / "vocab.json")
        genes_in_vocab = [gene for gene in adata.var_names if gene in vocab]
        if not genes_in_vocab:
            raise ValueError("No genes in the dataset overlap the scGPT vocabulary")
        self._gene_ids_in_vocab = [vocab[gene] for gene in genes_in_vocab]

        with open(model_path / "args.json", "r", encoding="utf-8") as handle:
            model_config = _json.load(handle)

        set_seed(42)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = TransformerModel(
            len(vocab),
            model_config["embsize"],
            model_config["nheads"],
            model_config["d_hid"],
            model_config["nlayers"],
            vocab=vocab,
            pad_value=self._pad_value,
            pad_token=self._pad_token,
        )
        self._model.load_state_dict(torch.load(model_path / "best_model.pt", map_location=self._device))
        self._model.to(self._device)
        self._model.eval()
        self._vocab = vocab

    def embed(self, genes_lists: list[list[str]]) -> np.ndarray:
        self._load(genes_lists)
        assert self._model is not None and self._device is not None and self._vocab is not None
        import scipy.sparse as sp
        import torch

        adata = self._build_pseudo_anndata(genes_lists)
        genes_in_vocab = [gene for gene in adata.var_names if gene in self._vocab]
        adata = adata[:, genes_in_vocab].copy()
        counts = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        gene_ids = np.asarray(self._gene_ids_in_vocab, dtype=np.int64)

        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(counts), self._batch_size):
                batch_counts = counts[start : start + self._batch_size]
                gene_ids_tensor = torch.from_numpy(np.tile(gene_ids, (len(batch_counts), 1))).long().to(self._device)
                values_tensor = torch.from_numpy(batch_counts).float().to(self._device)
                src_key_padding_mask = torch.zeros_like(gene_ids_tensor, dtype=torch.bool)
                encoded = self._model._encode(gene_ids_tensor, values_tensor, src_key_padding_mask)
                cell_embeddings = encoded.mean(dim=1)
                embeddings.append(cell_embeddings.cpu().numpy())
        return np.concatenate(embeddings, axis=0).astype(np.float32)

    def describe(self) -> dict:
        payload = super().describe()
        payload.update({"model_dir": self._model_dir, "batch_size": self._batch_size})
        return payload

    def close(self) -> None:
        import gc

        self._model = None
        self._vocab = None
        self._gene_ids_in_vocab = []
        gc.collect()


class GeneformerMethod(EmbeddingMethod):
    method_key = "geneformer"
    display_name = "Geneformer"

    def __init__(self, model_name: str, batch_size: int) -> None:
        self._model_name = model_name
        self._batch_size = batch_size

    def _load_gene_mappings(self):
        import pickle
        from pathlib import Path as _Path

        import geneformer

        package_dir = _Path(geneformer.__file__).parent
        name_id_path = package_dir / "gene_name_id_dict.pkl"
        if not name_id_path.exists():
            name_id_path = package_dir / "gene_name_id_dict_gc30M.pkl"
        token_path = package_dir / "token_dictionary.pkl"
        if not token_path.exists():
            token_path = package_dir / "token_dictionary_gc30M.pkl"

        with open(name_id_path, "rb") as handle:
            gene_name_to_ensembl = pickle.load(handle)
        with open(token_path, "rb") as handle:
            ensembl_to_token = pickle.load(handle)
        return gene_name_to_ensembl, ensembl_to_token

    def embed(self, genes_lists: list[list[str]]) -> np.ndarray:
        import torch
        from transformers import AutoModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained(self._model_name, trust_remote_code=True).to(device)
        model.eval()

        gene_name_to_ensembl, ensembl_to_token = self._load_gene_mappings()
        pad_token_id = 0
        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(genes_lists), self._batch_size):
                batch = genes_lists[start : start + self._batch_size]
                batch_ids: list[list[int]] = []
                for genes in batch:
                    token_ids: list[int] = []
                    for gene in genes:
                        ensembl_id = gene_name_to_ensembl.get(gene)
                        if ensembl_id is not None and ensembl_id in ensembl_to_token:
                            token_ids.append(ensembl_to_token[ensembl_id])
                    batch_ids.append(token_ids or [pad_token_id])

                max_length = max(len(token_ids) for token_ids in batch_ids)
                padded_ids: list[list[int]] = []
                masks: list[list[int]] = []
                for token_ids in batch_ids:
                    pad_length = max_length - len(token_ids)
                    padded_ids.append(token_ids + [pad_token_id] * pad_length)
                    masks.append([1] * len(token_ids) + [0] * pad_length)

                input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
                attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                embeddings.append(pooled.cpu().numpy())
        return np.concatenate(embeddings, axis=0).astype(np.float32)

    def describe(self) -> dict:
        payload = super().describe()
        payload.update({"model_name": self._model_name, "batch_size": self._batch_size})
        return payload


DEFAULT_METHOD_KEYS = [
    "rank_pca",
    "tfidf_svd",
    "c2s_base",
    "c2s_lora_features",
    "c2s_lora_projections",
]


def available_method_keys() -> list[str]:
    return sorted(
        [
            "c2s_base",
            "c2s_lora_features",
            "c2s_lora_projections",
            "geneformer",
            "rank_pca",
            "scgpt",
            "sentence_transformer",
            "tfidf_svd",
        ]
    )


def build_method(
    method_key: str,
    dataset_name: str,
    outputs_root: str | Path,
    model_path: str | None,
    batch_size: int,
    max_seq_len: int,
    use_4bit: bool,
    sentence_transformer_model: str,
    scgpt_model_dir: str | None,
    geneformer_model: str,
    checkpoint_tag: str,
) -> EmbeddingMethod:
    if method_key == "rank_pca":
        return RankPcaMethod()
    if method_key == "tfidf_svd":
        return TfidfSvdMethod()
    if method_key == "sentence_transformer":
        return SentenceTransformerMethod(model_name=sentence_transformer_model)
    if method_key == "c2s_base":
        resolved_model_path = model_path or infer_base_model_path_from_dataset(dataset_name, outputs_root, fallback=DEFAULT_2B_MODEL)
        return C2SBaseMethod(
            model_path=resolved_model_path,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            use_4bit=use_4bit,
        )
    if method_key == "c2s_lora_features":
        return C2SLoRAMethod(
            dataset_name=dataset_name,
            outputs_root=outputs_root,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            use_4bit=use_4bit,
            checkpoint_tag=checkpoint_tag,
            return_projections=False,
        )
    if method_key == "c2s_lora_projections":
        return C2SLoRAMethod(
            dataset_name=dataset_name,
            outputs_root=outputs_root,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            use_4bit=use_4bit,
            checkpoint_tag=checkpoint_tag,
            return_projections=True,
        )
    if method_key == "scgpt":
        if not scgpt_model_dir:
            raise ValueError("scGPT requires --scgpt-model-dir")
        return ScGPTMethod(model_dir=scgpt_model_dir, batch_size=batch_size)
    if method_key == "geneformer":
        return GeneformerMethod(model_name=geneformer_model, batch_size=batch_size)
    raise ValueError(f"Unknown method: {method_key}")


def build_vocab(genes_lists: list[list[str]]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for genes in genes_lists:
        for gene in genes:
            if gene not in vocab:
                vocab[gene] = len(vocab)
    return vocab


def build_rank_matrix(genes_lists: list[list[str]], vocab: dict[str, int]) -> np.ndarray:
    matrix = np.zeros((len(genes_lists), len(vocab)), dtype=np.float32)
    for row_index, genes in enumerate(genes_lists):
        gene_count = len(genes)
        for rank, gene in enumerate(genes):
            column = vocab.get(gene)
            if column is None:
                continue
            matrix[row_index, column] = gene_count - rank
    return matrix


def gene_sentences(genes_lists: list[list[str]]) -> list[str]:
    return [" ".join(genes) for genes in genes_lists]


def resolve_lora_checkpoint_dir(dataset_name: str, outputs_root: str | Path, checkpoint_tag: str = "best") -> Path:
    outputs_path = Path(outputs_root)
    exact_run = outputs_path / f"gemma_stage1_{dataset_name}"
    candidate_runs: list[Path] = []
    if exact_run.is_dir():
        candidate_runs.append(exact_run)
    candidate_runs.extend(
        run_dir
        for run_dir in sorted(outputs_path.iterdir())
        if run_dir.is_dir() and dataset_name in run_dir.name and run_dir not in candidate_runs
    )
    for run_dir in candidate_runs:
        checkpoint_dir = run_dir / "checkpoints" / checkpoint_tag
        if checkpoint_dir.is_dir():
            return checkpoint_dir
    tried = ", ".join(str(path) for path in candidate_runs) or str(exact_run)
    raise FileNotFoundError(f"No {checkpoint_tag} checkpoint found for dataset '{dataset_name}'. Tried: {tried}")


def infer_base_model_path_from_dataset(dataset_name: str, outputs_root: str | Path, fallback: str) -> str:
    checkpoint_dir = resolve_lora_checkpoint_dir(dataset_name, outputs_root, checkpoint_tag="best")
    return infer_base_model_path(checkpoint_dir / "gemma_lora_stage1", fallback=fallback)


def infer_base_model_path(adapter_dir: str | Path, fallback: str) -> str:
    adapter_config_path = Path(adapter_dir) / "adapter_config.json"
    if adapter_config_path.is_file():
        with open(adapter_config_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        model_path = payload.get("base_model_name_or_path")
        if model_path:
            return str(model_path)
    return fallback


def unpack_projection_head_checkpoint(payload: object) -> tuple[dict | None, dict]:
    if payload is None:
        return None, {}
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload.get("state_dict")
        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        return state_dict, config
    if isinstance(payload, dict):
        return payload, {}
    raise TypeError(f"Unsupported projection head checkpoint payload type: {type(payload)!r}")
