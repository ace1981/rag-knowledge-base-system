#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动内存模式的Qdrant服务进行测试
"""

import os
import sys
from pathlib import Path
from loguru import logger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from ollama_client import OllamaClient
from document_processor import DocumentProcessor
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def test_qdrant_memory():
    """
    测试内存模式的Qdrant
    """
    try:
        # 创建内存模式的Qdrant客户端
        client = QdrantClient(":memory:")
        logger.info("✅ 成功创建内存模式Qdrant客户端")
        
        # 创建一个测试集合
        collection_name = "test_collection"
        
        # 检查集合是否存在
        collections = client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)
        
        if not collection_exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            logger.info(f"✅ 成功创建集合: {collection_name}")
        else:
            logger.info(f"✅ 集合已存在: {collection_name}")
        
        # 测试Ollama客户端
        ollama_client = OllamaClient()
        
        # 测试嵌入功能
        test_text = "这是一个测试文档"
        embedding = ollama_client.get_embedding(test_text)
        
        if embedding:
            logger.info(f"✅ 成功获取嵌入向量，维度: {len(embedding)}")
            
            # 插入测试向量
            from qdrant_client.models import PointStruct
            
            client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=1,
                        vector=embedding,
                        payload={"text": test_text, "source": "test"}
                    )
                ]
            )
            logger.info("✅ 成功插入测试向量")
            
            # 测试搜索
            search_results = client.search(
                collection_name=collection_name,
                query_vector=embedding,
                limit=5
            )
            
            logger.info(f"✅ 搜索测试成功，找到{len(search_results)}个结果")
            if search_results:
                logger.info(f"最佳匹配得分: {search_results[0].score}")
        
        return client
        
    except Exception as e:
        logger.error(f"❌ Qdrant内存模式测试失败: {e}")
        return None

def main():
    """
    主函数
    """
    logger.info("=== 启动Qdrant内存模式测试 ===")
    
    client = test_qdrant_memory()
    
    if client:
        logger.info("\n🎉 Qdrant内存模式运行成功！")
        logger.info("现在可以运行其他测试脚本了")
        logger.info("\n要停止测试，请按 Ctrl+C")
        
        try:
            # 保持程序运行
            input("按回车键退出...")
        except KeyboardInterrupt:
            logger.info("\n👋 程序已退出")
    else:
        logger.error("❌ Qdrant内存模式启动失败")
        sys.exit(1)

if __name__ == "__main__":
    main()