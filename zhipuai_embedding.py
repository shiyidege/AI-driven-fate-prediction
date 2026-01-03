from typing import List, Optional
from langchain_core.embeddings import Embeddings

class ZhipuAIEmbeddings(Embeddings):
    """`Zhipuai Embeddings` embedding models."""
    
    # 新增api_key参数，支持传参或从环境变量读
    def __init__(self, api_key: Optional[str] = None):
        """
        实例化ZhipuAI客户端
        
        Args:
            api_key (Optional[str]): 智谱AI的API密钥，不传则从环境变量ZHIPUAI_API_KEY读取
        """
        from zhipuai import ZhipuAI
        # 把api_key传给ZhipuAI客户端（核心修正）
        self.client = ZhipuAI(api_key=api_key)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 每个文档的embedding列表
        """
        result = []
        # 批量处理（每次64条，符合智谱接口限制）
        for i in range(0, len(texts), 64):
            batch_texts = texts[i:i+64]
            embeddings_response = self.client.embeddings.create(
                model="embedding-3",  # 智谱embedding模型名
                input=batch_texts
            )
            # 提取embedding值（修正原代码变量名重复问题）
            batch_embeddings = [item.embedding for item in embeddings_response.data]
            result.extend(batch_embeddings)
        return result
    
    def embed_query(self, text: str) -> List[List[float]]:
        """
        生成单条文本的 embedding.
        Args:
            text (str): 要生成 embedding 的文本.

        Returns:
            List[float]: 文本的embedding列表
        """
        return self.embed_documents([text])[0]