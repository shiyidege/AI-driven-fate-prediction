
# rag/embeddings_transformers.py
from __future__ import annotations
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings



class HFTransformersEmbeddings(Embeddings):
    """
    用 transformers 模型做 embedding（mean pooling + L2 normalize）
    避免 sentence-transformers 目录结构依赖（1_Pooling/config.json 问题）。
    """

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> List[List[float]]:
        # tokenize
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        out = self.model(**batch)
        token_embeddings = out.last_hidden_state  # (B, T, H)
        attention_mask = batch["attention_mask"].unsqueeze(-1)  # (B, T, 1)

        # mean pooling（只平均有效 token）
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        emb = summed / counts

        # normalize
        emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]
