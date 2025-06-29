#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG知识库演示程序
整合Qdrant向量数据库和Ollama模型实现检索增强生成
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

# 导入自定义模块
from document_processor import DocumentProcessor
from vector_store import VectorStore
from ollama_client import OllamaClient

# 加载环境变量
load_dotenv()

class RAGSystem:
    """
    RAG（检索增强生成）系统主类
    """
    
    def __init__(self):
        """
        初始化RAG系统
        """
        logger.info("初始化RAG系统...")
        
        # 初始化各个组件
        self.document_processor = DocumentProcessor(
            chunk_size=int(os.getenv('CHUNK_SIZE', '512')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '50'))
        )
        
        self.vector_store = VectorStore()
        self.ollama_client = OllamaClient()
        
        # 配置参数
        self.top_k_results = int(os.getenv('TOP_K_RESULTS', '5'))
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
        self.system_prompt = os.getenv('SYSTEM_PROMPT', 
            '你是一个智能助手，请根据提供的上下文信息回答用户的问题。如果上下文中没有相关信息，请诚实地说明。')
        
        logger.info("RAG系统初始化完成")
    
    def check_system_status(self) -> Dict[str, bool]:
        """
        检查系统各组件状态
        
        Returns:
            Dict[str, bool]: 各组件状态
        """
        status = {
            'ollama': self.ollama_client.check_connection(),
            'qdrant': self.vector_store.check_connection()
        }
        
        logger.info(f"系统状态检查: {status}")
        return status
    
    def add_document(self, file_path: str) -> bool:
        """
        添加文档到知识库
        
        Args:
            file_path (str): 文档文件路径
            
        Returns:
            bool: 添加是否成功
        """
        try:
            logger.info(f"开始添加文档: {file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return False
            
            # 处理文档
            chunks = self.document_processor.process_document(file_path)
            if not chunks:
                logger.error(f"文档处理失败: {file_path}")
                return False
            
            logger.info(f"文档分块完成，共{len(chunks)}个块")
            
            # 生成向量
            vectors = []
            metadata_list = []
            
            logger.info("开始生成向量嵌入...")
            for chunk in tqdm(chunks, desc="生成向量"):
                # 获取文本嵌入
                embedding = self.ollama_client.get_embedding(chunk['text'])
                if embedding:
                    vectors.append(embedding)
                    
                    # 准备元数据
                    metadata = {
                        'text': chunk['text'],
                        'source_file': chunk['source_file'],
                        'source_path': chunk['source_path'],
                        'chunk_id': chunk['id'],
                        'start_pos': chunk['start_pos'],
                        'end_pos': chunk['end_pos'],
                        'length': chunk['length'],
                        'timestamp': time.time()
                    }
                    metadata_list.append(metadata)
                else:
                    logger.warning(f"无法生成嵌入向量: chunk {chunk['id']}")
            
            if not vectors:
                logger.error("没有成功生成任何向量")
                return False
            
            # 存储向量
            logger.info(f"开始存储{len(vectors)}个向量...")
            point_ids = self.vector_store.add_vectors(vectors, metadata_list)
            
            logger.info(f"文档添加成功: {file_path}, 存储了{len(point_ids)}个向量")
            return True
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False
    
    def add_text(self, text: str, source_name: str = "用户输入") -> bool:
        """
        添加文本到知识库
        
        Args:
            text (str): 文本内容
            source_name (str): 来源名称
            
        Returns:
            bool: 添加是否成功
        """
        try:
            logger.info(f"开始添加文本: {source_name}")
            
            # 处理文本
            chunks = self.document_processor.process_text(text, source_name)
            if not chunks:
                logger.error("文本处理失败")
                return False
            
            # 生成向量并存储
            vectors = []
            metadata_list = []
            
            for chunk in tqdm(chunks, desc="生成向量"):
                embedding = self.ollama_client.get_embedding(chunk['text'])
                if embedding:
                    vectors.append(embedding)
                    
                    metadata = {
                        'text': chunk['text'],
                        'source_file': chunk['source_file'],
                        'source_path': chunk['source_path'],
                        'chunk_id': chunk['id'],
                        'start_pos': chunk['start_pos'],
                        'end_pos': chunk['end_pos'],
                        'length': chunk['length'],
                        'timestamp': time.time()
                    }
                    metadata_list.append(metadata)
            
            if vectors:
                point_ids = self.vector_store.add_vectors(vectors, metadata_list)
                logger.info(f"文本添加成功: {source_name}, 存储了{len(point_ids)}个向量")
                return True
            else:
                logger.error("没有成功生成任何向量")
                return False
                
        except Exception as e:
            logger.error(f"添加文本失败: {e}")
            return False
    
    def search_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """
        在知识库中搜索相关信息
        
        Args:
            query (str): 查询文本
            
        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        try:
            logger.info(f"搜索查询: {query}")
            
            # 生成查询向量
            query_embedding = self.ollama_client.get_embedding(query)
            if not query_embedding:
                logger.error("无法生成查询向量")
                return []
            
            # 搜索相似向量
            results = self.vector_store.search_similar(
                query_vector=query_embedding,
                top_k=self.top_k_results,
                score_threshold=self.similarity_threshold
            )
            
            logger.info(f"搜索完成，找到{len(results)}个相关结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def generate_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        基于搜索结果生成答案
        
        Args:
            query (str): 用户查询
            search_results (List[Dict[str, Any]]): 搜索结果
            
        Returns:
            str: 生成的答案
        """
        try:
            # 构建上下文
            context_parts = []
            for i, result in enumerate(search_results):
                score = result['score']
                text = result['payload']['text']
                source = result['payload']['source_file']
                
                context_parts.append(
                    f"[参考{i+1}] (来源: {source}, 相似度: {score:.3f})\n{text}\n"
                )
            
            context = "\n".join(context_parts)
            
            # 生成回答
            logger.info("开始生成答案...")
            answer = self.ollama_client.generate_response(
                prompt=query,
                context=context,
                system_prompt=self.system_prompt
            )
            
            if answer:
                logger.info("答案生成成功")
                return answer
            else:
                logger.error("答案生成失败")
                return "抱歉，我无法生成答案。请检查系统配置或重试。"
                
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return f"生成答案时出现错误: {e}"
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        完整的问答流程
        
        Args:
            question (str): 用户问题
            
        Returns:
            Dict[str, Any]: 包含答案和相关信息的结果
        """
        start_time = time.time()
        
        # 搜索相关知识
        search_results = self.search_knowledge(question)
        
        if not search_results:
            return {
                'question': question,
                'answer': '抱歉，我在知识库中没有找到相关信息来回答您的问题。',
                'sources': [],
                'search_time': time.time() - start_time,
                'total_time': time.time() - start_time
            }
        
        search_time = time.time() - start_time
        
        # 生成答案
        answer = self.generate_answer(question, search_results)
        
        total_time = time.time() - start_time
        
        # 整理来源信息
        sources = []
        for result in search_results:
            sources.append({
                'source_file': result['payload']['source_file'],
                'text_snippet': result['payload']['text'][:100] + '...',
                'similarity_score': result['score']
            })
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'search_time': search_time,
            'total_time': total_time
        }
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            collection_info = self.vector_store.get_collection_info()
            point_count = self.vector_store.count_points()
            
            return {
                'total_documents': point_count,
                'collection_info': collection_info,
                'embedding_dimension': self.vector_store.embedding_dimension,
                'chunk_size': self.document_processor.chunk_size,
                'chunk_overlap': self.document_processor.chunk_overlap
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def clear_knowledge_base(self) -> bool:
        """
        清空知识库
        
        Returns:
            bool: 清空是否成功
        """
        try:
            return self.vector_store.clear_collection()
        except Exception as e:
            logger.error(f"清空知识库失败: {e}")
            return False


def interactive_demo():
    """
    交互式演示程序
    """
    print("🚀 RAG知识库系统演示")
    print("=" * 50)
    
    # 初始化系统
    try:
        rag = RAGSystem()
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return
    
    # 检查系统状态
    status = rag.check_system_status()
    if not all(status.values()):
        print("⚠️  系统组件状态异常:")
        for component, is_ok in status.items():
            print(f"  {component}: {'✅' if is_ok else '❌'}")
        print("\n请检查Ollama和Qdrant服务是否正常运行")
        return
    
    print("✅ 系统初始化成功，所有组件正常")
    
    while True:
        print("\n" + "=" * 50)
        print("请选择操作:")
        print("1. 添加文档到知识库")
        print("2. 添加文本到知识库")
        print("3. 问答查询")
        print("4. 查看知识库统计")
        print("5. 清空知识库")
        print("6. 退出")
        
        choice = input("\n请输入选择 (1-6): ").strip()
        
        if choice == '1':
            file_path = input("请输入文档路径: ").strip()
            if file_path:
                success = rag.add_document(file_path)
                print(f"{'✅ 文档添加成功' if success else '❌ 文档添加失败'}")
        
        elif choice == '2':
            text = input("请输入文本内容: ").strip()
            source_name = input("请输入来源名称 (可选): ").strip() or "用户输入"
            if text:
                success = rag.add_text(text, source_name)
                print(f"{'✅ 文本添加成功' if success else '❌ 文本添加失败'}")
        
        elif choice == '3':
            question = input("请输入您的问题: ").strip()
            if question:
                print("\n🤔 思考中...")
                result = rag.ask(question)
                
                print(f"\n📝 问题: {result['question']}")
                print(f"\n🤖 回答:\n{result['answer']}")
                
                if result['sources']:
                    print(f"\n📚 参考来源 ({len(result['sources'])}个):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"  {i}. {source['source_file']} (相似度: {source['similarity_score']:.3f})")
                        print(f"     {source['text_snippet']}")
                
                print(f"\n⏱️  搜索耗时: {result['search_time']:.2f}秒")
                print(f"⏱️  总耗时: {result['total_time']:.2f}秒")
        
        elif choice == '4':
            stats = rag.get_knowledge_base_stats()
            if stats:
                print("\n📊 知识库统计信息:")
                print(f"  总文档块数: {stats.get('total_documents', 0)}")
                print(f"  向量维度: {stats.get('embedding_dimension', 0)}")
                print(f"  分块大小: {stats.get('chunk_size', 0)}")
                print(f"  分块重叠: {stats.get('chunk_overlap', 0)}")
            else:
                print("❌ 无法获取统计信息")
        
        elif choice == '5':
            confirm = input("确认清空知识库？(y/N): ").strip().lower()
            if confirm == 'y':
                success = rag.clear_knowledge_base()
                print(f"{'✅ 知识库已清空' if success else '❌ 清空失败'}")
        
        elif choice == '6':
            print("👋 再见！")
            break
        
        else:
            print("❌ 无效选择，请重试")


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    # 运行交互式演示
    interactive_demo()