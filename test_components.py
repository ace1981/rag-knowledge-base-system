#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组件测试脚本
测试各个组件的基本功能
"""

import os
from dotenv import load_dotenv
from ollama_client import OllamaClient
from document_processor import DocumentProcessor

def test_ollama():
    """测试Ollama客户端"""
    print("\n=== 测试Ollama客户端 ===")
    try:
        client = OllamaClient()
        
        # 测试连接
        if client.check_connection():
            print("✅ Ollama连接成功")
            
            # 测试嵌入
            embedding = client.get_embedding("测试文本")
            if embedding:
                print(f"✅ 嵌入功能正常，维度: {len(embedding)}")
            else:
                print("❌ 嵌入功能失败")
                
            # 测试生成
            response = client.generate_response("你好")
            if response:
                print(f"✅ 生成功能正常: {response[:50]}...")
            else:
                print("❌ 生成功能失败")
        else:
            print("❌ Ollama连接失败")
            
    except Exception as e:
        print(f"❌ Ollama测试异常: {e}")

def test_document_processor():
    """测试文档处理器"""
    print("\n=== 测试文档处理器 ===")
    try:
        processor = DocumentProcessor()
        
        # 测试文本分块
        test_text = "这是一个测试文档。" * 100
        chunks = processor.split_text_into_chunks(test_text)
        print(f"✅ 文本分块成功，共{len(chunks)}个块")
        
        # 测试文本清理
        dirty_text = "  这是一个\n\n包含多余空格的\t\t文本  "
        clean_text = processor.clean_text(dirty_text)
        print(f"✅ 文本清理成功: '{clean_text}'")
        
    except Exception as e:
        print(f"❌ 文档处理器测试异常: {e}")

def test_environment():
    """测试环境配置"""
    print("\n=== 测试环境配置 ===")
    
    # 重新加载环境变量
    load_dotenv(dotenv_path='.env', override=True)
    
    required_vars = [
        'OLLAMA_HOST', 'OLLAMA_PORT', 'OLLAMA_EMBEDDING_MODEL', 'OLLAMA_CHAT_MODEL',
        'QDRANT_HOST', 'QDRANT_PORT', 'QDRANT_COLLECTION_NAME',
        'CHUNK_SIZE', 'CHUNK_OVERLAP', 'TOP_K_RESULTS', 'EMBEDDING_DIMENSION'
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: 未设置")

def main():
    """主测试函数"""
    print("开始组件测试...")
    
    test_environment()
    test_ollama()
    test_document_processor()
    
    print("\n=== 测试完成 ===")
    print("\n注意: Qdrant服务需要单独启动")
    print("如果Docker启动失败，可以下载Qdrant二进制文件直接运行")
    print("下载地址: https://github.com/qdrant/qdrant/releases")

if __name__ == "__main__":
    main()