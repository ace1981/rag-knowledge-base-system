#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库模型和操作类
用于管理文档信息的SQLite数据库
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class DatabaseManager:
    """
    数据库管理类
    负责SQLite数据库的创建、连接和操作
    """
    
    def __init__(self, db_path: str = "rag_system.db"):
        """
        初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """
        初始化数据库，创建必要的表
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建文档表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    chunks_count INTEGER NOT NULL,
                    vectors_count INTEGER NOT NULL,
                    upload_time TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建对话历史表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    message_type TEXT NOT NULL, -- 'user' or 'bot'
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT -- JSON格式的额外信息
                )
            """)
            
            # 创建系统统计表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stat_date DATE NOT NULL,
                    total_documents INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0,
                    total_vectors INTEGER DEFAULT 0,
                    total_chats INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            print("✅ 数据库初始化完成")
    
    def add_document(self, doc_info: Dict[str, Any]) -> bool:
        """
        添加文档记录
        
        Args:
            doc_info: 文档信息字典
            
        Returns:
            bool: 是否添加成功
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO documents (
                        id, filename, original_filename, file_path, file_size,
                        chunks_count, vectors_count, upload_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_info['id'],
                    doc_info['filename'],
                    doc_info.get('original_filename', doc_info['filename']),
                    doc_info['file_path'],
                    doc_info['file_size'],
                    doc_info['chunks_count'],
                    doc_info['vectors_count'],
                    doc_info['upload_time']
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"❌ 添加文档记录失败: {e}")
            return False
    
    def get_documents(self, page: int = 1, page_size: int = 10) -> Tuple[List[Dict[str, Any]], int]:
        """
        获取文档列表（分页）
        
        Args:
            page: 页码（从1开始）
            page_size: 每页数量
            
        Returns:
            Tuple: (文档列表, 总数量)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # 使结果可以按列名访问
                cursor = conn.cursor()
                
                # 获取总数量
                cursor.execute("SELECT COUNT(*) as total FROM documents")
                total = cursor.fetchone()['total']
                
                # 获取分页数据
                offset = (page - 1) * page_size
                cursor.execute("""
                    SELECT * FROM documents 
                    ORDER BY upload_time DESC 
                    LIMIT ? OFFSET ?
                """, (page_size, offset))
                
                documents = []
                for row in cursor.fetchall():
                    documents.append({
                        'id': row['id'],
                        'filename': row['filename'],
                        'original_filename': row['original_filename'],
                        'file_path': row['file_path'],
                        'file_size': row['file_size'],
                        'chunks_count': row['chunks_count'],
                        'vectors_count': row['vectors_count'],
                        'upload_time': row['upload_time'],
                        'created_at': row['created_at']
                    })
                
                return documents, total
        except Exception as e:
            print(f"❌ 获取文档列表失败: {e}")
            return [], 0
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取文档信息
        
        Args:
            doc_id: 文档ID
            
        Returns:
            Optional[Dict]: 文档信息或None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
                row = cursor.fetchone()
                
                if row:
                    return {
                        'id': row['id'],
                        'filename': row['filename'],
                        'original_filename': row['original_filename'],
                        'file_path': row['file_path'],
                        'file_size': row['file_size'],
                        'chunks_count': row['chunks_count'],
                        'vectors_count': row['vectors_count'],
                        'upload_time': row['upload_time'],
                        'created_at': row['created_at']
                    }
                return None
        except Exception as e:
            print(f"❌ 获取文档信息失败: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档记录
        
        Args:
            doc_id: 文档ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"❌ 删除文档记录失败: {e}")
            return False
    
    def clear_all_documents(self) -> bool:
        """
        清空所有文档记录
        
        Returns:
            bool: 是否清空成功
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents")
                conn.commit()
                return True
        except Exception as e:
            print(f"❌ 清空文档记录失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            Dict: 统计信息
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 获取文档统计
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_documents,
                        COALESCE(SUM(chunks_count), 0) as total_chunks,
                        COALESCE(SUM(vectors_count), 0) as total_vectors
                    FROM documents
                """)
                doc_stats = cursor.fetchone()
                
                # 获取对话统计
                cursor.execute("SELECT COUNT(*) as total_chats FROM chat_history")
                chat_stats = cursor.fetchone()
                
                return {
                    'total_documents': doc_stats[0],
                    'total_chunks': doc_stats[1],
                    'total_vectors': doc_stats[2],
                    'total_chats': chat_stats[0]
                }
        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'total_vectors': 0,
                'total_chats': 0
            }
    
    def add_chat_message(self, session_id: str, message_type: str, content: str, metadata: Dict = None) -> bool:
        """
        添加对话消息
        
        Args:
            session_id: 会话ID
            message_type: 消息类型 ('user' 或 'bot')
            content: 消息内容
            metadata: 额外的元数据
            
        Returns:
            bool: 是否添加成功
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO chat_history (session_id, message_type, content, metadata)
                    VALUES (?, ?, ?, ?)
                """, (
                    session_id,
                    message_type,
                    content,
                    json.dumps(metadata) if metadata else None
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"❌ 添加对话消息失败: {e}")
            return False
    
    def clear_chat_history(self) -> bool:
        """
        清空对话历史
        
        Returns:
            bool: 是否清空成功
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_history")
                conn.commit()
                return True
        except Exception as e:
            print(f"❌ 清空对话历史失败: {e}")
            return False