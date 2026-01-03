# rag/build_index.py
from __future__ import annotations

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
          "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(k, None)

os.environ["CHROMA_TELEMETRY"] = "FALSE"
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["CHROMA_DISABLE_TELEMETRY"] = "TRUE"


# rag/build_index.py
from pathlib import Path
import re

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from .embeddings_transformers import HFTransformersEmbeddings



def infer_tradition(filename: str) -> str:
    """
    根据文件名关键词，粗略判断该文档所属的命理/术数体系。
    - 在知识库尚未做精细人工标注的情况下，提供一个“够用”的自动分类方式
    - 用于在 RAG 检索阶段通过 metadata filter 控制检索范围（如只检索八字/紫微/六爻）

    参数：
        filename (str): 文档文件名（不包含路径）

    返回：
        str: 体系标签，如：
            - "bazi"       八字 / 子平
            - "ziwei"      紫微斗数
            - "liuyao"     六爻
            - "tarot"      塔罗
            - "astrology" 占星
            - "yijing"     易经
            - "shushu"     数术 / 梅花易数
            - "common"    未明确分类的通用资料
    """
    name = filename.lower()

    if re.search(r"(六爻|世应|用神|动爻|变卦|卦|爻)", filename):
        return "liuyao"
    if re.search(r"(紫微|斗数|命宫|身宫|四化|化禄|化权|化科|化忌|星曜)", filename):
        return "ziwei"
    if re.search(r"(八字|子平|十神|官杀|印星|食伤|财星|大运|流年|格局|用神|算命)", filename):
        return "bazi"
    if "塔罗" in filename:
        return "tarot"
    if re.search(r"(占星|星座|星盘|十二宫)", filename):
        return "astrology"
    if "易经" in filename:
        return "yijing"
    if  re.search(r"(梅花易数|数术)", filename):
        return "shushu"
    return "common"


def load_all_docs(kb_dir: Path) -> list[Document]:
    """
    递归加载本地知识库目录中的所有可用文档（txt / md / pdf），
    并为每个 Document 自动补充 metadata。
    - 只做“文件 → Document”的转换，不做切分、不做 embedding
    - 跳过 Chroma 自身产生的数据库文件，避免污染知识库

    参数：
        kb_dir (Path): 本地知识库根目录，例如 knowledge_suanming/

    返回：
        list[Document]: LangChain Document 列表，每个 Document 包含：
            - page_content: 文本内容
            - metadata:
                - source_path: 原始文件完整路径
                - source_file: 原始文件名
                - tradition: 自动推断的体系标签
    """
    docs: list[Document] = []
    print(f"[INFO] 扫描知识库目录: {kb_dir.resolve()}")

    for p in kb_dir.rglob("*"):
        if p.is_dir():
            continue
        # 跳过Chroma自身文件
        if p.name in {"chroma.sqlite3"} or p.suffix.lower() in {".bin", ".parquet"}:
            continue

        suf = p.suffix.lower()
        try:
            if suf in {".txt", ".md"}:
                def load_text_with_fallback(p: Path) -> list[Document]:
                    """
                    尝试用多种编码读取 txt/md，解决 Z-Library 常见的 utf-16 / gbk / gb18030 等问题。
                    """
                    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk", "big5", "utf-16"]
                    last_err = None
                    for enc in encodings:
                        try:
                            return TextLoader(str(p), encoding=enc).load()
                        except Exception as e:
                            last_err = e
                            continue
                    raise RuntimeError(f"Error loading {p} with encodings {encodings}. Last error: {last_err}")
                loaded = load_text_with_fallback(p)
            elif suf == ".pdf":
                loaded = PyPDFLoader(str(p)).load()  # 按页Document
            else:
                continue
        except Exception as e:
            print(f"[SKIP] 解析失败: {p} | err={type(e).__name__}: {e}")
            continue


        tradition = infer_tradition(p.name)
        for d in loaded:
            d.metadata = d.metadata or {}
            d.metadata.update({
                "source_path": str(p),
                "source_file": p.name,
                "tradition": tradition,
            })
        docs.extend(loaded)
    return docs


def build_or_update_chroma(
    kb_dir: str = "rag/knowledge_suanming", # 本地知识库目录
    persist_dir: str = "knowledge_suanming/chroma_store", # 向量库存放目录
    collection_name: str = "suanming_kb",  # 向量库名称
):
    """
    从本地知识库构建（或更新）Chroma 向量数据库，并持久化到磁盘。
    - 初次部署时运行一次
    - 本地知识库新增 / 修改文档后手动运行一次

    处理流程：
        1. 加载所有 txt / md / pdf 文档
        2. 使用 RecursiveCharacterTextSplitter 切分为 chunk
        3. 使用本地中文 embedding 模型生成向量
        4. 将向量写入 Chroma 持久化存储

    参数：
        kb_dir (str):
            本地知识库根目录路径
        persist_dir (str):
            Chroma 向量库的持久化目录（会自动创建）
        collection_name (str):
            Chroma collection 名称，用于区分不同知识库

    返回：
        dict:
            - docs:    原始 Document 数量
            - chunks:  切分后的 chunk 数量
            - persist_dir: 实际使用的持久化目录
    """
    kb_path = Path(kb_dir)
    docs = load_all_docs(kb_path)
    if not docs:
        raise RuntimeError("未加载到任何txt/pdf文档。")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = HFTransformersEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        device="cpu",
    )

    # 持久化到 persist_dir（注意：这不是 chroma.sqlite3 所在目录，而是指定的 store 目录）
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    vs.add_documents(chunks)

    return {"docs": len(docs), "chunks": len(chunks), "persist_dir": persist_dir}


if __name__ == "__main__":
    """
    允许直接通过命令行运行本文件，用于手动构建或更新向量索引。

    使用示例：
        python rag/build_index.py
    """
    info = build_or_update_chroma()
    print(info)
