#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant集合重置脚本
用于删除并重新创建具有正确向量维度的集合
"""

import os
from dotenv import load_dotenv
from loguru import logger

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
except ImportError:
    logger.error("qdrant-client未安装，请运行: pip install qdrant-client")
    exit(1)

def reset_collection():
    """重置Qdrant集合"""
    # 加载环境变量
    load_dotenv()
    
    host = os.getenv('QDRANT_HOST', 'localhost')
    port = int(os.getenv('QDRANT_PORT', '6333'))
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'knowledge_base')
    embedding_dimension = 2560  # 强制使用正确的Qwen3-Embedding-4B维度
    
    print(f"连接到Qdrant: {host}:{port}")
    print(f"集合名称: {collection_name}")
    print(f"向量维度: {embedding_dimension}")
    
    try:
        # 初始化客户端
        client = QdrantClient(host=host, port=port, timeout=30)
        
        # 检查集合是否存在
        collections = client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)
        
        if collection_exists:
            print(f"\n发现现有集合: {collection_name}")
            
            # 获取集合信息
            collection_info = client.get_collection(collection_name)
            current_dimension = collection_info.config.params.vectors.size
            print(f"当前集合向量维度: {current_dimension}")
            
            if current_dimension != embedding_dimension:
                print(f"❌ 向量维度不匹配！需要从 {current_dimension} 更新到 {embedding_dimension}")
            else:
                print(f"ℹ️ 当前维度 {current_dimension}，目标维度 {embedding_dimension}")
                
            # 强制删除现有集合以确保使用正确维度
            print("正在删除现有集合...")
            client.delete_collection(collection_name)
            print("✅ 现有集合已删除")
        else:
            print(f"\n集合 {collection_name} 不存在")
        
        # 创建新集合
        print(f"\n正在创建新集合，向量维度: {embedding_dimension}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dimension,
                distance=Distance.COSINE
            )
        )
        
        print("✅ 新集合创建成功")
        
        # 验证集合
        collection_info = client.get_collection(collection_name)
        actual_dimension = collection_info.config.params.vectors.size
        print(f"验证: 新集合向量维度 = {actual_dimension}")
        
        if actual_dimension == embedding_dimension:
            print("✅ 集合重置完成，向量维度正确")
        else:
            print(f"❌ 集合创建异常，期望维度 {embedding_dimension}，实际维度 {actual_dimension}")
            
    except Exception as e:
        print(f"❌ 重置集合时出错: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("Qdrant集合重置工具")
    print("=" * 40)
    
    success = reset_collection()
    
    if success:
        print("\n🎉 集合重置成功！现在可以重启应用程序。")
    else:
        print("\n❌ 集合重置失败，请检查Qdrant服务状态。")

if __name__ == "__main__":
    main()