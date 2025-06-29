#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量存储模块
使用Qdrant进行向量存储和检索操作
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from loguru import logger

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter, 
        FieldCondition, MatchValue, SearchRequest
    )
except ImportError:
    logger.error("qdrant-client未安装，请运行: pip install qdrant-client")
    QdrantClient = None

# 加载环境变量
load_dotenv()

class VectorStore:
    """
    向量存储类，用于管理Qdrant向量数据库操作
    """
    
    def __init__(self):
        """
        初始化向量存储
        """
        if not QdrantClient:
            raise ImportError("qdrant-client未安装")
        
        self.host = os.getenv('QDRANT_HOST', 'localhost')
        self.port = int(os.getenv('QDRANT_PORT', '6333'))
        self.collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'knowledge_base')
        self.embedding_dimension = int(os.getenv('EMBEDDING_DIMENSION', '1024'))
        
        # 初始化客户端
        try:
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                timeout=30
            )
            logger.info(f"连接到Qdrant: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"无法连接到Qdrant: {e}")
            # 尝试使用内存模式作为备选方案
            try:
                self.client = QdrantClient(":memory:")
                logger.warning("使用内存模式运行Qdrant（数据不会持久化）")
            except Exception as e2:
                logger.error(f"无法启动内存模式Qdrant: {e2}")
                raise
        
        # 确保集合存在
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """
        确保集合存在，如果不存在则创建
        """
        try:
            # 检查集合是否存在
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # 创建集合
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"创建集合: {self.collection_name}")
            else:
                logger.info(f"集合已存在: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"检查/创建集合失败: {e}")
            raise
    
    def check_connection(self) -> bool:
        """
        检查Qdrant连接状态
        
        Returns:
            bool: 连接状态
        """
        try:
            info = self.client.get_collections()
            logger.info("Qdrant连接正常")
            return True
        except Exception as e:
            logger.error(f"Qdrant连接失败: {e}")
            return False
    
    def add_vectors(self, vectors: List[List[float]], 
                   metadata_list: List[Dict[str, Any]]) -> List[str]:
        """
        添加向量到集合
        
        Args:
            vectors (List[List[float]]): 向量列表
            metadata_list (List[Dict[str, Any]]): 元数据列表
            
        Returns:
            List[str]: 插入的点ID列表
        """
        if len(vectors) != len(metadata_list):
            raise ValueError("向量数量与元数据数量不匹配")
        
        points = []
        point_ids = []
        
        for i, (vector, metadata) in enumerate(zip(vectors, metadata_list)):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            # 确保向量维度正确
            if len(vector) != self.embedding_dimension:
                logger.warning(f"向量维度不匹配: 期望{self.embedding_dimension}, 实际{len(vector)}")
                # 如果维度不匹配，可以选择截断或填充
                if len(vector) > self.embedding_dimension:
                    vector = vector[:self.embedding_dimension]
                else:
                    vector.extend([0.0] * (self.embedding_dimension - len(vector)))
            
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=metadata
            )
            points.append(point)
        
        try:
            # 批量插入
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"成功添加{len(points)}个向量到集合")
            return point_ids
            
        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            raise
    
    def search_similar(self, query_vector: List[float], 
                      top_k: int = 1, 
                      score_threshold: float = 0.0,
                      filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        搜索相似向量
        
        Args:
            query_vector (List[float]): 查询向量
            top_k (int): 返回结果数量
            score_threshold (float): 相似度阈值
            filter_conditions (Optional[Dict[str, Any]]): 过滤条件
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
        """
        try:
            # 确保查询向量维度正确
            if len(query_vector) != self.embedding_dimension:
                logger.warning(f"查询向量维度不匹配: 期望{self.embedding_dimension}, 实际{len(query_vector)}")
                if len(query_vector) > self.embedding_dimension:
                    query_vector = query_vector[:self.embedding_dimension]
                else:
                    query_vector.extend([0.0] * (self.embedding_dimension - len(query_vector)))
            
            # 构建过滤器
            search_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # 执行搜索
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=search_filter
            )
            
            # 格式化结果
            results = []
            for scored_point in search_result:
                result = {
                    'id': scored_point.id,
                    'score': scored_point.score,
                    'payload': scored_point.payload
                }
                results.append(result)
            
            logger.debug(f"搜索完成，返回{len(results)}个结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def delete_points(self, point_ids: List[str]) -> bool:
        """
        删除指定的点
        
        Args:
            point_ids (List[str]): 要删除的点ID列表
            
        Returns:
            bool: 删除是否成功
        """
        try:
            operation_info = self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            
            logger.info(f"成功删除{len(point_ids)}个点")
            return True
            
        except Exception as e:
            logger.error(f"删除点失败: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息
        
        Returns:
            Dict[str, Any]: 集合信息
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': info.config.params.vectors.size,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'points_count': info.points_count,
                'segments_count': info.segments_count,
                'status': info.status
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """
        清空集合中的所有数据
        
        Returns:
            bool: 清空是否成功
        """
        try:
            # 删除集合
            self.client.delete_collection(self.collection_name)
            
            # 重新创建集合
            self._ensure_collection_exists()
            
            logger.info(f"成功清空集合: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            return False
    
    def count_points(self) -> int:
        """
        获取集合中的点数量
        
        Returns:
            int: 点数量
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            logger.error(f"获取点数量失败: {e}")
            return 0


def test_vector_store():
    """
    测试向量存储功能
    """
    try:
        store = VectorStore()
        
        # 检查连接
        if not store.check_connection():
            print("❌ Qdrant连接失败")
            return
        
        print("✅ Qdrant连接成功")
        
        # 获取集合信息
        info = store.get_collection_info()
        print(f"✅ 集合信息: {info}")
        
        # 测试添加向量
        test_vectors = [
            [0.1] * 1024,  # 假设1024维向量
            [0.2] * 1024,
            [0.3] * 1024
        ]
        
        test_metadata = [
            {'text': '测试文本1', 'source': 'test'},
            {'text': '测试文本2', 'source': 'test'},
            {'text': '测试文本3', 'source': 'test'}
        ]
        
        point_ids = store.add_vectors(test_vectors, test_metadata)
        print(f"✅ 成功添加{len(point_ids)}个向量")
        
        # 测试搜索
        query_vector = [0.15] * 1024
        results = store.search_similar(query_vector, top_k=2)
        print(f"✅ 搜索结果: {len(results)}个")
        
        for result in results:
            print(f"  - ID: {result['id']}, 分数: {result['score']:.4f}")
        
        # 获取点数量
        count = store.count_points()
        print(f"✅ 集合中共有{count}个点")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    test_vector_store()