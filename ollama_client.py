#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama客户端模块
用于调用本地Ollama服务进行文本嵌入和生成
"""

import os
import requests
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from loguru import logger

class OllamaClient:
    """
    Ollama客户端类，用于与本地Ollama服务交互
    """
    
    def __init__(self):
        """
        初始化Ollama客户端
        """
        # 清除可能的环境变量缓存并重新加载
        for key in list(os.environ.keys()):
            if key.startswith('OLLAMA_') or key.startswith('QDRANT_'):
                del os.environ[key]
        
        # 重新加载环境变量
        load_dotenv(dotenv_path='.env', override=True)
        
        # 调试环境变量读取
        logger.info(f"当前工作目录: {os.getcwd()}")
        logger.info(f"环境变量OLLAMA_HOST: {os.getenv('OLLAMA_HOST')}")
        
        self.host = os.getenv('OLLAMA_HOST', 'localhost')
        self.port = os.getenv('OLLAMA_PORT', '11434')
        self.base_url = f"http://{self.host}:{self.port}"
        self.embedding_model = os.getenv('OLLAMA_EMBEDDING_MODEL', 'shaw/dmeta-embedding-zh:latest')
        self.chat_model = os.getenv('OLLAMA_CHAT_MODEL', 'qwen2.5:14b')
        
        logger.info(f"初始化Ollama客户端: {self.base_url}")
        logger.info(f"嵌入模型: {self.embedding_model}")
        logger.info(f"对话模型: {self.chat_model}")
    
    def check_connection(self) -> bool:
        """
        检查Ollama服务连接状态
        
        Returns:
            bool: 连接状态
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama服务连接正常")
                return True
            else:
                logger.error(f"Ollama服务连接失败: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"无法连接到Ollama服务: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """
        获取可用的模型列表
        
        Returns:
            List[str]: 模型名称列表
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                logger.info(f"可用模型: {models}")
                return models
            else:
                logger.error(f"获取模型列表失败: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"获取模型列表异常: {e}")
            return []
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        获取文本的向量嵌入
        
        Args:
            text (str): 输入文本
            
        Returns:
            Optional[List[float]]: 向量嵌入，失败时返回None
        """
        try:
            # 使用prompt参数，Ollama API的正确格式
            payload = {
                "model": self.embedding_model,
                "prompt": text  # 使用prompt参数，传入字符串
            }
            
            response = requests.post(
                f"{self.base_url}/api/embeddings",  # 使用正确的端点
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Ollama API响应: {data}")
                # 尝试获取embedding字段，可能是单个向量或向量列表
                embedding = data.get('embedding')
                if embedding:
                    logger.debug(f"成功获取文本嵌入，维度: {len(embedding)}")
                    return embedding
                # 如果没有embedding字段，尝试embeddings字段
                embeddings = data.get('embeddings')
                if embeddings and len(embeddings) > 0:
                    logger.debug(f"成功获取文本嵌入，维度: {len(embeddings[0])}")
                    return embeddings[0]  # 返回第一个嵌入向量
                else:
                    logger.error(f"响应中未找到嵌入向量，完整响应: {data}")
                    return None
            else:
                logger.error(f"获取嵌入失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"获取嵌入异常: {e}")
            return None
    
    def generate_response(self, prompt: str, context: str = "", system_prompt: str = "") -> Optional[str]:
        """
        生成文本响应（非流式）
        
        Args:
            prompt (str): 用户提示
            context (str): 上下文信息
            system_prompt (str): 系统提示
            
        Returns:
            Optional[str]: 生成的响应，失败时返回None
        """
        try:
            # 构建完整的提示
            full_prompt = ""
            if system_prompt:
                full_prompt += f"系统提示: {system_prompt}\n\n"
            if context:
                full_prompt += f"上下文信息:\n{context}\n\n"
            full_prompt += f"用户问题: {prompt}\n\n请基于上述信息回答:"
            
            payload = {
                "model": self.chat_model,
                "prompt": full_prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data.get('response', '').strip()
                if generated_text:
                    logger.debug(f"成功生成响应，长度: {len(generated_text)}")
                    return generated_text
                else:
                    logger.error("响应中未找到生成的文本")
                    return None
            else:
                logger.error(f"生成响应失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"生成响应异常: {e}")
            return None
    
    def generate_stream_response(self, prompt: str, context: str = "", system_prompt: str = ""):
        """
        生成流式文本响应
        
        Args:
            prompt (str): 用户提示
            context (str): 上下文信息
            system_prompt (str): 系统提示
            
        Yields:
            str: 生成的文本片段
        """
        try:
            # 构建完整的提示
            full_prompt = ""
            if system_prompt:
                full_prompt += f"系统提示: {system_prompt}\n\n"
            if context:
                full_prompt += f"上下文信息:\n{context}\n\n"
            full_prompt += f"用户问题: {prompt}\n\n请基于上述信息回答:"
            
            payload = {
                "model": self.chat_model,
                "prompt": full_prompt,
                "stream": True
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                yield data['response']
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                logger.error(f"流式生成失败: {response.status_code}")
                
        except Exception as e:
            logger.error(f"流式生成异常: {e}")


def test_ollama_client():
    """
    测试Ollama客户端功能
    """
    client = OllamaClient()
    
    # 检查连接
    if not client.check_connection():
        print("❌ Ollama服务连接失败")
        return
    
    # 获取模型列表
    models = client.get_available_models()
    print(f"✅ 可用模型: {models}")
    
    # 测试嵌入
    test_text = "这是一个测试文本"
    embedding = client.get_embedding(test_text)
    if embedding:
        print(f"✅ 嵌入测试成功，维度: {len(embedding)}")
    else:
        print("❌ 嵌入测试失败")
    
    # 测试生成
    response = client.generate_response("你好，请介绍一下自己")
    if response:
        print(f"✅ 生成测试成功: {response[:100]}...")
    else:
        print("❌ 生成测试失败")


if __name__ == "__main__":
    test_ollama_client()