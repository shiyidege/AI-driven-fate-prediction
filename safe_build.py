import os
import gc
import time
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rag.embeddings_transformers import HFTransformersEmbeddings

# 1. 配置环境
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CHROMA_TELEMETRY"] = "FALSE"
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

# 数据库路径
PERSIST_DIR = "rag/knowledge_suanming/chroma_store"
# 目标大文件路径
TARGET_FILE = "rag/knowledge_suanming/imported_fortune_telling.txt"

def main():
    print("🚀 启动极低内存构建模式...")
    
    # 2. 加载 Embedding 模型 (内存消耗大户，先加载)
    print(" -> 正在加载 Embedding 模型 (BGE-Small)...")
    embeddings = HFTransformersEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        device="cpu"
    )

    # 3. 初始化数据库
    print(f" -> 连接数据库: {PERSIST_DIR}")
    vector_store = Chroma(
        collection_name="suanming_kb",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    if not os.path.exists(TARGET_FILE):
        print(f"❌ 找不到文件: {TARGET_FILE}")
        return

    print(f" -> 开始流式处理文件: {TARGET_FILE}")
    
    # 4. 逐行读取 + 小批次写入
    batch_lines = []
    batch_size = 20  # 每次只处理 20 条问答（非常保守）
    total_processed = 0

    with open(TARGET_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        current_text_block = ""
        
        for line in f:
            line = line.strip()
            # 简单拼接
            current_text_block += line + "\n"
            
            # 遇到分隔符（假设你的数据是用 ----- 分隔的）或者积累了一定长度
            if "----------" in line or len(current_text_block) > 500:
                batch_lines.append(current_text_block)
                current_text_block = "" # 重置 buffer

            # 当积攒够了 batch_size 个小块，就写入一次
            if len(batch_lines) >= batch_size:
                # 转换成 Document 对象
                docs = [Document(page_content=txt, metadata={"source": "fortune_telling_dataset"}) for txt in batch_lines]
                
                try:
                    vector_store.add_documents(docs)
                    total_processed += len(docs)
                    print(f"    v 已存入 {total_processed} 条数据... (内存清理)")
                except Exception as e:
                    print(f"    [WARN] 写入失败: {e}")
                
                # === 关键：彻底释放内存 ===
                del docs
                batch_lines = [] # 清空列表
                gc.collect()     # 强制垃圾回收
                time.sleep(0.1)  # 歇一会，给 CPU 喘息时间

        # 5. 处理最后剩余的
        if batch_lines:
            print(" -> 正在写入最后剩余数据...")
            docs = [Document(page_content=txt, metadata={"source": "fortune_telling_dataset"}) for txt in batch_lines]
            vector_store.add_documents(docs)
            print("    v 完成！")

    print("-" * 30)
    print(f"✅ 构建完成！共存入约 {total_processed} 条记录。")

if __name__ == "__main__":
    main()