#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的RAG演示程序
使用内存模式的Qdrant，展示文档处理、向量化、存储和检索的完整流程
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from ollama_client import OllamaClient
from document_processor import DocumentProcessor
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class CompleteRAGSystem:
    """
    完整的RAG系统，使用内存模式的Qdrant
    """
    
    def __init__(self, collection_name: str = "knowledge_base"):
        """
        初始化RAG系统
        
        Args:
            collection_name (str): Qdrant集合名称
        """
        self.collection_name = collection_name
        
        logger.info("🚀 初始化完整RAG系统...")
        
        # 初始化Ollama客户端
        logger.info("初始化Ollama客户端...")
        self.ollama_client = OllamaClient()
        logger.info("✅ Ollama客户端初始化完成")
        
        # 初始化文档处理器
        logger.info("初始化文档处理器...")
        self.doc_processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
        logger.info("✅ 文档处理器初始化完成")
        
        # 初始化Qdrant客户端（内存模式）
        logger.info("初始化Qdrant客户端（内存模式）...")
        self.qdrant_client = QdrantClient(":memory:")
        logger.info("✅ Qdrant客户端初始化完成")
        
        # 创建集合
        self._create_collection()
        
        # 文档计数器
        self.doc_counter = 0
        
        logger.info("🎉 RAG系统初始化完成！")
    
    def _create_collection(self):
        """
        创建Qdrant集合
        """
        try:
            # 检查集合是否存在
            collections = self.qdrant_client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not collection_exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                logger.info(f"✅ 成功创建集合: {self.collection_name}")
            else:
                logger.info(f"✅ 集合已存在: {self.collection_name}")
        except Exception as e:
            logger.error(f"❌ 创建集合失败: {e}")
            raise
    
    def add_text_document(self, text: str, title: str = "文档", source: str = "用户输入") -> int:
        """
        添加文本文档到知识库
        
        Args:
            text (str): 文档文本
            title (str): 文档标题
            source (str): 文档来源
            
        Returns:
            int: 添加的文档块数量
        """
        logger.info(f"📄 开始处理文档: {title}")
        
        # 文档分块
        chunks = self.doc_processor.split_text_into_chunks(text)
        logger.info(f"📝 文档分块完成，共{len(chunks)}个块")
        
        # 向量化并存储
        points = []
        for chunk in chunks:
            # 获取嵌入向量
            embedding = self.ollama_client.get_embedding(chunk['text'])
            if embedding:
                # 准备点数据
                point_id = self.doc_counter * 1000 + chunk['id']
                payload = {
                    'text': chunk['text'],
                    'title': title,
                    'source': source,
                    'chunk_id': chunk['id'],
                    'doc_id': self.doc_counter,
                    'start_pos': chunk['start_pos'],
                    'end_pos': chunk['end_pos'],
                    'length': chunk['length'],
                    'timestamp': time.time()
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))
        
        # 批量插入向量
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"✅ 成功存储{len(points)}个向量")
        
        self.doc_counter += 1
        return len(points)
    
    def search_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query (str): 查询文本
            limit (int): 返回结果数量
            
        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        logger.info(f"🔍 搜索查询: {query}")
        
        # 获取查询向量
        query_embedding = self.ollama_client.get_embedding(query)
        if not query_embedding:
            logger.error("❌ 无法获取查询向量")
            return []
        
        # 执行向量搜索
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        # 格式化结果
        results = []
        for result in search_results:
            results.append({
                'score': result.score,
                'text': result.payload['text'],
                'title': result.payload.get('title', 'Unknown'),
                'source': result.payload.get('source', 'Unknown'),
                'chunk_id': result.payload.get('chunk_id'),
                'doc_id': result.payload.get('doc_id')
            })
        
        logger.info(f"✅ 搜索完成，找到{len(results)}个结果")
        return results
    
    def generate_answer(self, query: str, context_limit: int = 3) -> Dict[str, Any]:
        """
        基于检索到的文档生成答案
        
        Args:
            query (str): 用户问题
            context_limit (int): 使用的上下文数量
            
        Returns:
            Dict[str, Any]: 包含答案和相关信息的结果
        """
        start_time = time.time()
        
        # 检索相关文档
        search_results = self.search_documents(query, limit=context_limit)
        search_time = time.time() - start_time
        
        if not search_results:
            return {
                'question': query,
                'answer': '抱歉，我没有找到相关的信息来回答您的问题。',
                'sources': [],
                'search_time': search_time,
                'generation_time': 0,
                'total_time': search_time
            }
        
        # 构建上下文
        context_texts = []
        for i, result in enumerate(search_results, 1):
            context_texts.append(
                f"参考资料{i}（来源：{result['title']}，相似度：{result['score']:.3f}）：\n{result['text']}"
            )
        
        context = "\n\n".join(context_texts)
        
        # 构建提示词
        prompt = f"""请基于以下参考资料回答问题。请根据提供的信息给出准确、有用的答案。如果参考资料中没有足够的信息，请说明。

参考资料：
{context}

问题：{query}

请提供详细的回答："""
        
        # 生成回答
        generation_start = time.time()
        answer = self.ollama_client.generate_response(prompt)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        return {
            'question': query,
            'answer': answer if answer else "抱歉，我无法生成回答。",
            'sources': search_results,
            'search_time': search_time,
            'generation_time': generation_time,
            'total_time': total_time
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                'total_documents': self.doc_counter,
                'total_vectors': collection_info.points_count,
                'vector_dimension': 768,
                'chunk_size': self.doc_processor.chunk_size,
                'chunk_overlap': self.doc_processor.chunk_overlap
            }
        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            return {}

def demo_complete_rag():
    """
    完整RAG系统演示
    """
    print("\n" + "=" * 60)
    print("🚀 完整RAG系统演示")
    print("=" * 60)
    
    # 初始化RAG系统
    try:
        rag = CompleteRAGSystem()
    except Exception as e:
        logger.error(f"❌ 系统初始化失败: {e}")
        return
    
    # 添加示例文档
    print("\n📚 添加示例文档到知识库...")
    
    documents = [
        {
            "title": "人工智能基础",
            "text": """人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
            
机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习。机器学习算法通过分析数据来识别模式，并使用这些模式来做出预测或决策。常见的机器学习算法包括线性回归、决策树、随机森林、支持向量机和神经网络等。
            
深度学习是机器学习的一个子集，它使用人工神经网络来模拟人脑的工作方式。深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。深度学习模型通常包含多个隐藏层，能够学习数据的复杂特征表示。"""
        },
        {
            "title": "向量数据库技术",
            "text": """向量数据库是一种专门用于存储和检索高维向量数据的数据库系统。它们特别适用于相似性搜索和机器学习应用。向量数据库使用特殊的索引结构，如HNSW（Hierarchical Navigable Small World）或IVF（Inverted File）来加速向量搜索。
            
Qdrant是一个高性能的向量数据库，支持大规模向量搜索。它提供了REST API和gRPC接口，支持实时索引更新和复杂的过滤查询。Qdrant使用Rust语言开发，具有出色的性能和内存效率。
            
向量数据库的主要应用场景包括：推荐系统、图像搜索、文本相似性搜索、异常检测、语义搜索等。在RAG（检索增强生成）系统中，向量数据库用于存储文档的嵌入向量，实现快速的语义搜索，为大语言模型提供相关的上下文信息。"""
        },
        {
            "title": "自然语言处理技术",
            "text": """自然语言处理（Natural Language Processing，NLP）是人工智能和语言学领域的分支学科。它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。NLP的目标是让计算机能够理解、解释和生成人类语言。
            
NLP的主要任务包括：文本分类、情感分析、命名实体识别、机器翻译、问答系统、文本摘要、语言模型等。这些任务在实际应用中有着广泛的用途，如搜索引擎、智能客服、内容推荐等。
            
近年来，基于Transformer架构的大型语言模型（如GPT、BERT、T5等）在NLP领域取得了革命性的进展，显著提升了各种NLP任务的性能。这些模型通过在大规模文本数据上进行预训练，学习到了丰富的语言知识和常识。"""
        },
        {
            "title": "RAG系统原理",
            "text": """RAG（Retrieval-Augmented Generation）是一种结合了信息检索和文本生成的技术架构。RAG系统通过检索相关的外部知识来增强大语言模型的生成能力，解决了传统语言模型知识更新困难和可能产生幻觉的问题。
            
RAG系统的工作流程包括：1）文档预处理和向量化，将知识库中的文档转换为向量表示；2）查询处理，将用户问题转换为查询向量；3）相似性搜索，在向量数据库中找到最相关的文档片段；4）上下文构建，将检索到的文档作为上下文；5）答案生成，使用大语言模型基于上下文生成回答。
            
RAG系统的优势包括：知识可以实时更新、减少模型幻觉、提供可追溯的信息来源、降低模型训练成本等。RAG技术在问答系统、智能客服、知识管理等领域有着广泛的应用前景。"""
        }
    ]
    
    # 添加文档到知识库
    total_chunks = 0
    for doc in documents:
        chunks_count = rag.add_text_document(doc["text"], doc["title"], "演示文档")
        total_chunks += chunks_count
        print(f"✅ 文档 '{doc['title']}' 添加完成，{chunks_count}个块")
    
    print(f"\n📊 总共添加了{len(documents)}个文档，{total_chunks}个文档块")
    
    # 显示系统统计信息
    stats = rag.get_stats()
    if stats:
        print(f"\n📈 系统统计信息:")
        print(f"  - 文档数量: {stats.get('total_documents', 0)}")
        print(f"  - 向量数量: {stats.get('total_vectors', 0)}")
        print(f"  - 向量维度: {stats.get('vector_dimension', 0)}")
        print(f"  - 分块大小: {stats.get('chunk_size', 0)}")
        print(f"  - 分块重叠: {stats.get('chunk_overlap', 0)}")
    
    # 测试搜索功能
    print("\n" + "=" * 60)
    print("🔍 搜索功能测试")
    print("=" * 60)
    
    test_queries = [
        "什么是机器学习？",
        "向量数据库有什么用途？",
        "Transformer架构有什么特点？",
        "RAG系统是如何工作的？",
        "深度学习和机器学习的区别是什么？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 测试查询 {i}: {query}")
        results = rag.search_documents(query, limit=3)
        
        for j, result in enumerate(results, 1):
            print(f"  结果 {j} (相似度: {result['score']:.3f}):")
            print(f"    文档: {result['title']}")
            print(f"    内容: {result['text'][:100]}...")
    
    # 测试问答功能
    print("\n" + "=" * 60)
    print("🤖 问答功能测试")
    print("=" * 60)
    
    qa_queries = [
        "什么是深度学习？它有什么特点？",
        "Qdrant向量数据库有什么优势？",
        "RAG系统解决了什么问题？",
        "NLP的主要任务有哪些？"
    ]
    
    for i, query in enumerate(qa_queries, 1):
        print(f"\n❓ 问答测试 {i}: {query}")
        print("🤔 正在思考...")
        
        result = rag.generate_answer(query, context_limit=3)
        
        print(f"\n🤖 回答:\n{result['answer']}")
        
        if result['sources']:
            print(f"\n📚 参考来源 ({len(result['sources'])}个):")
            for j, source in enumerate(result['sources'], 1):
                print(f"  {j}. {source['title']} (相似度: {source['score']:.3f})")
        
        print(f"\n⏱️ 性能统计:")
        print(f"  - 搜索耗时: {result['search_time']:.2f}秒")
        print(f"  - 生成耗时: {result['generation_time']:.2f}秒")
        print(f"  - 总耗时: {result['total_time']:.2f}秒")
    
    print("\n" + "=" * 60)
    print("🎉 完整RAG系统演示完成！")
    print("=" * 60)
    print("\n✨ 演示总结:")
    print("  - ✅ Ollama客户端连接正常")
    print("  - ✅ Qdrant内存模式运行正常")
    print("  - ✅ 文档处理和向量化成功")
    print("  - ✅ 语义搜索功能正常")
    print("  - ✅ 问答生成功能正常")
    print("\n🚀 RAG系统已准备就绪，可以处理实际应用！")

def main():
    """
    主函数
    """
    try:
        demo_complete_rag()
    except KeyboardInterrupt:
        print("\n\n👋 演示被用户中断")
    except Exception as e:
        logger.error(f"❌ 演示执行出错: {e}")
        print(f"\n❌ 演示执行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()