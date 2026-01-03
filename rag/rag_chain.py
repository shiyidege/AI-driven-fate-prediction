# rag/rag_chain.py
from __future__ import annotations
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
          "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(k, None)

os.environ["CHROMA_TELEMETRY"] = "FALSE"
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["CHROMA_DISABLE_TELEMETRY"] = "TRUE"

from typing import Optional
import os

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from .embeddings_transformers import HFTransformersEmbeddings



SUPPORTED_MODELS = [
    "gpt-4o",
    "deepseek-chat",
    "glm-4",
    "gemini-3-pro-preview",
    "doubao-seed-1-6-250615",
]


BASE_SYSTEM = (
    "你是一位严谨的命理术数咨询专家，擅长八字、紫微斗数、六爻、星座塔罗、占星解读。"
    "你需要给出理性客观、安抚、可执行的建议，避免绝对化断言，也避免讨好附和。"
)

RAG_GUARD = (
    "你必须遵守：\n"
    "1) 只能使用【资料】中的内容作为依据；不要编造书名、页码、原文出处。\n"
    "2) 若资料不足，明确说“资料不足”，给出一般性建议。\n"
    "3) 若资料观点冲突，说明不同流派/版本可能有差异，并给出更保守的建议。\n"
)


def get_llm(model_name: str, api_key: str, base_url: str, temperature: float = 0.7):
    """
    根据指定模型名称，创建并返回一个 ChatOpenAI 实例。
    - base_url统一调用多个模型
    - 使用环境变量方式传递 API Key，避免 SecretStr / pydantic 版本冲突
    - 假设 base_url 兼容 OpenAI Chat Completions 接口规范

    参数：
        model_name (str):
            要使用的模型名称，需在 SUPPORTED_MODELS 中
        api_key (str):
            API Key（通过环境变量传递给 LangChain/OpenAI SDK）
        base_url (str):
            OpenAI-compatible 的 API 网关地址
        temperature (float):
            生成温度，控制回答的随机性

    返回：
        ChatOpenAI:
            可直接用于 invoke() 的大语言模型实例
    """
    os.environ["OPENAI_API_KEY"] = api_key  # 让底层 SDK 自行读取

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        timeout=100,
    )


def get_vectorstore(persist_dir: str, collection_name: str):
    """
    加载并返回一个 Chroma 向量数据库实例。
    - embedding 模型需与 build_index.py 中保持一致
    - persist_dir 指向已经通过 build_or_update_chroma() 构建好的向量库目录
    - 本函数只负责“加载”，不负责构建或更新索引

    参数：
        persist_dir (str):
            Chroma 向量库的持久化目录路径
        collection_name (str):
            Chroma collection 名称

    返回：
        Chroma:
            已加载的向量数据库对象
    """
    #embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5",model_kwargs={"device": "cpu"},encode_kwargs={"normalize_embeddings": True},)

    embeddings = HFTransformersEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        device="cpu",
    )

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def format_docs(docs: list[Document], max_chars: int = 4500) -> str:
    """
    将检索到的 Document 列表格式化为可直接注入 Prompt 的上下文文本。
    - 为模型提供“可引用”的资料片段
    - 限制总字符数，避免上下文过长导致截断或费用浪费
    - 在文本中保留来源信息，便于模型进行引用说明

    参数：
        docs (list[Document]):
            检索阶段返回的 Document 列表
        max_chars (int):
            最大上下文字符数限制

    返回：
        str:
            格式化后的上下文字符串，形如：
            [资料1] 来源：xxx.txt
            文本内容...
    """
    if not docs:
        return "(无匹配资料)"

    out = []
    used = 0

    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source_file", d.metadata.get("source_path", ""))
        head = f"[资料{i}] 来源：{src}\n"
        body = d.page_content.strip() + "\n"
        blk = head + body + "\n"

        if used + len(blk) > max_chars:
            break

        out.append(blk)
        used += len(blk)

    return "\n".join(out).strip()


def rag_answer(
    question: str,
    model_name: str,
    api_key: str,
    base_url: str,
    persist_dir: str,
    collection_name: str,
    tradition_filter: Optional[str] = None,
    k: int = 6,
) -> str:
    """
    执行一次完整的 RAG 问答流程，并返回最终回答文本。

    处理流程：
        1. 加载向量数据库
        2. 根据 tradition_filter 构造检索器（可选体系过滤）
        3. 执行相似度检索，获取相关资料
        4. 将资料格式化后注入 Prompt
        5. 调用指定模型生成回答

    参数：
        question (str):
            用户输入的问题
        model_name (str):
            使用的模型名称
        api_key (str):
            API Key
        base_url (str):
            API 网关地址
        persist_dir (str):
            Chroma 向量库持久化目录
        collection_name (str):
            Chroma collection 名称
        tradition_filter (Optional[str]):
            体系过滤条件，例如：
            - "bazi" / "ziwei" / "liuyao"
            - None 表示不限制体系（全库检索）
        k (int):
            检索返回的最大文档数量

    返回：
        str:
            模型生成的最终回答文本
    """
    vs = get_vectorstore(persist_dir, collection_name)

    if tradition_filter:
        retriever = vs.as_retriever(
            search_kwargs={"k": k, "filter": {"tradition": tradition_filter}}
        )
    else:
        retriever = vs.as_retriever(search_kwargs={"k": k})

    docs = retriever.invoke(question)
    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", BASE_SYSTEM),
        ("system", RAG_GUARD),
        ("user",
         "用户问题：{question}\n\n"
         "【资料】（检索所得，可能为空）：\n{context}\n\n"
         "请按结构输出：\n"
         "A. 结论摘要（3-6条）\n"
         "B. 依据解释（引用哪些资料支持哪些点）\n"
         "C. 建议与提醒（可执行建议 + 边界声明）\n"
        )
    ])

    llm = get_llm(model_name, api_key, base_url)
    msg = prompt.format_messages(question=question, context=context)
    resp = llm.invoke(msg)

    # 确保返回字符串类型
    return str(resp.content)
