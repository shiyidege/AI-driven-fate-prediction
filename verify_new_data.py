# verify_new_data.py
import os
import sys

# 1. 基础配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CHROMA_TELEMETRY"] = "FALSE"
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

# 确保能导入 rag 模块
sys.path.append(os.getcwd())

from langchain_chroma import Chroma
from rag.embeddings_transformers import HFTransformersEmbeddings

def test_retrieval():
    print("🚀 正在连接数据库...")
    
    # 2. 必须和 safe_build.py 里的路径完全一致
    persist_dir = "rag/knowledge_suanming/chroma_store"
    
    if not os.path.exists(persist_dir):
        print(f"❌ 错误：找不到数据库文件夹: {persist_dir}")
        return

    # 3. 加载模型
    embeddings = HFTransformersEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        device="cpu"
    )
    
    # 4. 连接
    db = Chroma(
        collection_name="suanming_kb",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    
    # 5. 关键测试：问一个非常“通俗”的问题
    # 这种问题通常出现在网上的问答数据集中，而不是古籍里
    query = "我最近事业不顺怎么办？" 
    
    print(f"\n🔮 提问: {query}")
    print("   正在检索...")
    
    # 检索前 3 条
    docs = db.similarity_search(query, k=3)
    
    print("-" * 40)
    if docs:
        print(f"✅ 检索成功！找到 {len(docs)} 条结果：\n")
        for i, doc in enumerate(docs):
            # 获取来源元数据
            source = doc.metadata.get('source', '未知')
            content = doc.page_content.replace('\n', '')[:80] # 只显示前80字
            
            print(f"📄 结果 {i+1}")
            print(f"🏷️  来源: {source}")
            print(f"📝 内容: {content}...")
            print("-" * 20)
            
            # 这是一个简单的检查逻辑
            if "fortune" in source or "dataset" in source:
                print("   ✨ 恭喜！这条数据来自你刚才新导入的数据集！")
                print("-" * 20)
    else:
        print("❌ 检索失败，空空如也。")

if __name__ == "__main__":
    test_retrieval()