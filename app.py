#!/usr/bin/env python3
"""
RAG知识库系统 - Flask Web API服务
集成Ollama客户端、文档处理器和Qdrant向量存储
提供文件上传、对话、文档管理等RESTful API接口
"""

import os
import json
import uuid
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# 导入现有的RAG系统组件
from ollama_client import OllamaClient
from document_processor import DocumentProcessor
from vector_store import VectorStore
from database import DatabaseManager


class RAGWebService:
    """
    RAG Web服务类
    管理RAG系统的web接口和业务逻辑
    """
    
    def __init__(self):
        """初始化RAG Web服务"""
        self.ollama_client = None
        self.document_processor = None
        self.vector_store = None
        self.db_manager = None  # 数据库管理器
        self.documents = {}  # 临时存储文档信息（向后兼容）
        self.chat_history = []  # 临时存储对话历史（向后兼容）
        
        # 创建上传目录
        self.upload_folder = Path("uploads")
        self.upload_folder.mkdir(exist_ok=True)
        
        # 允许的文件类型
        self.allowed_extensions = {'txt', 'pdf', 'docx', 'md'}
        
        # 初始化组件
        self._initialize_components()
    
    def _initialize_components(self) -> bool:
        """
        初始化RAG系统组件
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            print("正在初始化RAG系统组件...")
            
            # 初始化数据库管理器
            try:
                self.db_manager = DatabaseManager()
                print("✅ 数据库管理器初始化成功")
            except Exception as e:
                print(f"❌ 数据库管理器初始化失败: {e}")
                # 数据库失败不阻止启动，继续初始化其他组件
            
            # 初始化Ollama客户端
            try:
                self.ollama_client = OllamaClient()
                if not self.ollama_client.check_connection():
                    print("⚠️ Ollama连接失败，但继续启动服务")
                else:
                    print("✅ Ollama客户端连接成功")
            except Exception as e:
                print(f"❌ Ollama客户端初始化失败: {e}")
                # 创建一个模拟客户端以避免None错误
                class MockOllamaClient:
                    def __init__(self):
                        self.host = 'localhost'
                        self.port = '11434'
                        self.chat_model = 'qwen3:14b'
                        self.embedding_model = 'dengcao/Qwen3-Embedding-4B:Q8_0'
                        self.is_mock = True
                    
                    def get_embedding(self, text):
                        # 返回None，让上层逻辑处理
                        return None
                    
                    def generate_response(self, prompt, context=""):
                        return "抱歉，Ollama服务未连接，无法生成回答。请检查Ollama是否正常运行。"
                    
                    def generate_stream_response(self, prompt, context=""):
                        # 返回一个生成器，产生错误消息
                        yield "抱歉，Ollama服务未连接，无法生成回答。请检查Ollama是否正常运行。"
                
                self.ollama_client = MockOllamaClient()
            
            # 初始化文档处理器
            try:
                self.document_processor = DocumentProcessor()
                print("✅ 文档处理器初始化成功")
            except Exception as e:
                print(f"❌ 文档处理器初始化失败: {e}")
            
            # 初始化向量存储
            try:
                self.vector_store = VectorStore()
                print("✅ 向量存储初始化成功")
            except Exception as e:
                print(f"❌ 向量存储初始化失败: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ 组件初始化失败: {e}")
            traceback.print_exc()
            return False
    
    def is_allowed_file(self, filename: str) -> bool:
        """
        检查文件类型是否允许
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否允许的文件类型
        """
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def safe_filename(self, filename: str) -> str:
        """
        安全处理文件名，保留扩展名
        
        Args:
            filename: 原始文件名
            
        Returns:
            str: 安全的文件名
        """
        if '.' in filename:
            name, ext = filename.rsplit('.', 1)
            # 使用secure_filename处理文件名部分
            safe_name = secure_filename(name)
            # 确保扩展名安全且保持原样
            safe_ext = ''.join(c for c in ext if c.isalnum()).lower()
            return f"{safe_name}.{safe_ext}" if safe_name else f"file.{safe_ext}"
        else:
            return secure_filename(filename) or "file"
    
    def upload_document(self, file) -> Dict[str, Any]:
        """
        上传并处理文档
        
        Args:
            file: 上传的文件对象
            
        Returns:
            Dict: 处理结果
        """
        try:
            if not file or file.filename == '':
                return {"success": False, "error": "没有选择文件"}
            
            if not self.is_allowed_file(file.filename):
                return {"success": False, "error": "不支持的文件类型"}
            
            # 保存文件
            original_filename = file.filename  # 保存原始文件名用于显示
            safe_filename = self.safe_filename(file.filename)  # 安全文件名用于处理
            file_id = str(uuid.uuid4())
            # 文件存储时使用UUID作为文件名，保留扩展名
            file_extension = ''
            if '.' in original_filename:
                file_extension = '.' + original_filename.rsplit('.', 1)[1].lower()
            file_path = self.upload_folder / f"{file_id}{file_extension}"
            file.save(str(file_path))
            
            # 处理文档
            print(f"正在处理文档: {original_filename}")
            chunks = self.document_processor.process_document(str(file_path))
            
            if not chunks:
                return {"success": False, "error": "文档处理失败或文档为空"}
            
            # 生成向量并存储
            print(f"正在生成向量，共{len(chunks)}个文档块...")
            vectors = []
            for i, chunk in enumerate(chunks):
                try:
                    vector = self.ollama_client.get_embedding(chunk['text'])
                    if vector:
                        vectors.append({
                            "id": f"{file_id}_{i}",
                            "vector": vector,
                            "payload": {
                                "text": chunk['text'],
                                "file_id": file_id,
                                "filename": original_filename,
                                "chunk_index": i,
                                "upload_time": datetime.now().isoformat()
                            }
                        })
                except Exception as e:
                    print(f"处理第{i}个文档块时出错: {e}")
                    continue
            
            if not vectors:
                return {"success": False, "error": "向量生成失败"}
            
            # 存储到向量数据库
            # 分离向量和元数据
            vector_list = [item["vector"] for item in vectors]
            metadata_list = [item["payload"] for item in vectors]
            success = self.vector_store.add_vectors(vector_list, metadata_list)
            if not success:
                return {"success": False, "error": "向量存储失败"}
            
            # 记录文档信息
            doc_info = {
                "id": file_id,
                "filename": original_filename,  # 保存原始文件名用于显示
                "original_filename": original_filename,
                "file_path": str(file_path),
                "chunks_count": len(chunks),
                "vectors_count": len(vectors),
                "upload_time": datetime.now().isoformat(),
                "file_size": file_path.stat().st_size
            }
            
            # 保存到数据库
            self.db_manager.add_document(doc_info)
            
            # 临时保存到内存（向后兼容）
            self.documents[file_id] = doc_info
            
            print(f"✅ 文档处理完成: {original_filename}, 生成{len(vectors)}个向量")
            
            return {
                "success": True,
                "file_id": file_id,
                "filename": original_filename,  # 返回原始文件名
                "chunks_count": len(chunks),
                "vectors_count": len(vectors)
            }
            
        except Exception as e:
            print(f"❌ 文档上传处理失败: {e}")
            traceback.print_exc()
            return {"success": False, "error": f"处理失败: {str(e)}"}
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        """
        清理知识库和重置Qdrant
        
        Returns:
            Dict: 清理结果
        """
        try:
            print("正在清理知识库...")
            
            # 清理向量存储
            if self.vector_store:
                success = self.vector_store.clear_collection()
                if not success:
                    return {"success": False, "error": "向量存储清理失败"}
            
            # 清理数据库中的文档记录
            if self.db_manager:
                self.db_manager.clear_all_documents()
            
            # 清理内存中的文档记录（向后兼容）
            self.documents.clear()
            
            # 清理对话历史
            self.chat_history.clear()
            
            # 清理上传的文件
            try:
                for file_path in self.upload_folder.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                print("✅ 上传文件清理完成")
            except Exception as e:
                print(f"⚠️ 清理上传文件时出现警告: {e}")
            
            print("✅ 知识库清理完成")
            
            return {
                "success": True,
                "message": "知识库已成功清理",
                "cleared_items": {
                    "documents": True,
                    "vectors": True,
                    "chat_history": True,
                    "uploaded_files": True
                }
            }
            
        except Exception as e:
            print(f"❌ 知识库清理失败: {e}")
            traceback.print_exc()
            return {"success": False, "error": f"清理失败: {str(e)}"}

    def parse_model_response(self, raw_response: str) -> tuple[str, str]:
        """
        解析模型回答，分离思考过程和最终答案
        
        Args:
            raw_response: 模型的原始回答
            
        Returns:
            tuple: (thinking, final_answer)
        """
        if not raw_response or not raw_response.strip():
            return "", ""
        
        # 按行分割回答
        lines = raw_response.strip().split('\n')
        
        # 过滤空行
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if not non_empty_lines:
            return "", ""
        
        # 如果只有一行，直接作为最终答案
        if len(non_empty_lines) == 1:
            return "", non_empty_lines[0]
        
        # 最后一行作为最终答案
        final_answer = non_empty_lines[-1]
        
        # 前面的所有行作为思考过程
        thinking_lines = non_empty_lines[:-1]
        thinking = '\n'.join(thinking_lines)
        
        return thinking, final_answer

    def chat(self, question: str, top_k: int = 1) -> Dict[str, Any]:
        """
        处理对话请求
        
        Args:
            question: 用户问题
            top_k: 返回的相关文档数量
            
        Returns:
            Dict: 对话结果
        """
        try:
            if not question.strip():
                return {"success": False, "error": "问题不能为空"}
            
            # 生成问题的向量
            question_vector = self.ollama_client.get_embedding(question)
            if not question_vector:
                return {"success": False, "error": "问题向量化失败"}
            
            # 搜索相关文档
            search_results = self.vector_store.search_similar(
                query_vector=question_vector,
                top_k=top_k,
                score_threshold=0.3
            )
            
            if not search_results:
                # 没有找到相关文档，使用基础模型直接回答
                system_thinking = f"没有在知识库中找到与问题'{question}'相关的文档，将使用基础模型进行回答。"
                prompt = f"请回答以下问题：{question}"
                raw_response = self.ollama_client.generate_response(prompt)
                if not raw_response:
                    return {"success": False, "error": "回答生成失败"}
                
                # 解析模型回答
                model_thinking, final_answer = self.parse_model_response(raw_response)
                
                # 合并系统思考和模型思考
                combined_thinking = system_thinking
                if model_thinking:
                    combined_thinking += "\n\n模型思考过程：\n" + model_thinking
                
                return {
                    "success": True,
                    "answer": final_answer,
                    "sources": [],
                    "question": question,
                    "mode": "基础模型回答",
                    "thinking": combined_thinking
                }
            
            # 构建上下文，过滤低相似度结果
            context_parts = []
            sources = []
            
            for result in search_results:
                payload = result.get('payload', {})
                text = payload.get('text', '')
                filename = payload.get('filename', '未知文件')
                chunk_index = payload.get('chunk_index', 0)
                score = result.get('score', 0)
                
                # 过滤相似度低于0.5的结果
                if score < 0.5:
                    continue
                    
                context_parts.append(text)
                sources.append({
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "score": round(score, 3),
                    "text_preview": text[:200] + "..." if len(text) > 200 else text
                })
            
            # 检查过滤后是否还有有效结果
            if not context_parts:
                # 没有找到足够相关的文档，使用基础模型直接回答
                system_thinking = f"虽然找到了{len(search_results)}个相关文档，但相似度都低于0.5阈值，将使用基础模型进行回答。"
                prompt = f"请回答以下问题：{question}"
                raw_response = self.ollama_client.generate_response(prompt)
                if not raw_response:
                    return {"success": False, "error": "回答生成失败"}
                
                # 解析模型回答
                model_thinking, final_answer = self.parse_model_response(raw_response)
                
                # 合并系统思考和模型思考
                combined_thinking = system_thinking
                if model_thinking:
                    combined_thinking += "\n\n模型思考过程：\n" + model_thinking
                
                return {
                    "success": True,
                    "answer": final_answer,
                    "sources": [],
                    "question": question,
                    "mode": "基础模型回答",
                    "thinking": combined_thinking
                }
            
            context = "\n\n".join(context_parts)
            
            # 生成系统思考内容
            system_thinking = f"在知识库中找到了{len(sources)}个相关文档片段，相似度范围：{min([s['score'] for s in sources]):.3f}-{max([s['score'] for s in sources]):.3f}。基于这些文档内容来回答问题。"
            
            # 生成回答
            prompt = f"""基于以下上下文信息，回答用户的问题。如果上下文中没有相关信息，请诚实地说明。

上下文信息：
{context}

用户问题：{question}

请提供准确、有用的回答："""
            
            raw_response = self.ollama_client.generate_response(prompt)
            if not raw_response:
                return {"success": False, "error": "回答生成失败"}
            
            # 解析模型回答
            model_thinking, final_answer = self.parse_model_response(raw_response)
            
            # 合并系统思考和模型思考
            combined_thinking = system_thinking
            if model_thinking:
                combined_thinking += "\n\n模型思考过程：\n" + model_thinking
            
            # 记录对话历史
            chat_record = {
                "id": str(uuid.uuid4()),
                "question": question,
                "answer": final_answer,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            }
            self.chat_history.append(chat_record)
            
            return {
                "success": True,
                "answer": final_answer,
                "sources": sources,
                "question": question,
                "mode": "知识库回答",
                "thinking": combined_thinking
            }
            
        except Exception as e:
            print(f"❌ 对话处理失败: {e}")
            traceback.print_exc()
            return {"success": False, "error": f"对话处理失败: {str(e)}"}
    
    def get_documents(self, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """
        获取已上传的文档列表（分页）
        
        Args:
            page: 页码（从1开始）
            page_size: 每页数量
            
        Returns:
            Dict: 包含文档列表和分页信息
        """
        try:
            documents, total = self.db_manager.get_documents(page, page_size)
            total_pages = (total + page_size - 1) // page_size  # 向上取整
            
            return {
                "success": True,
                "documents": documents,
                "pagination": {
                    "current_page": page,
                    "page_size": page_size,
                    "total_items": total,
                    "total_pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1
                }
            }
        except Exception as e:
            print(f"❌ 获取文档列表失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents": [],
                "pagination": {
                    "current_page": 1,
                    "page_size": page_size,
                    "total_items": 0,
                    "total_pages": 0,
                    "has_next": False,
                    "has_prev": False
                }
            }
    
    def delete_document(self, file_id: str) -> Dict[str, Any]:
        """
        删除文档
        
        Args:
            file_id: 文件ID
            
        Returns:
            Dict: 删除结果
        """
        try:
            # 从数据库获取文档信息
            doc_info = self.db_manager.get_document_by_id(file_id)
            if not doc_info:
                return {"success": False, "error": "文档不存在"}
            
            # 删除文件
            file_path = Path(doc_info["file_path"])
            if file_path.exists():
                file_path.unlink()
            
            # 从向量数据库中删除相关向量
            # 注意：这里简化处理，实际应该实现向量删除功能
            
            # 从数据库中删除记录
            if self.db_manager.delete_document(file_id):
                # 从内存中删除（向后兼容）
                if file_id in self.documents:
                    del self.documents[file_id]
                return {"success": True, "message": "文档删除成功"}
            else:
                return {"success": False, "error": "数据库删除失败"}
            
        except Exception as e:
            print(f"❌ 文档删除失败: {e}")
            return {"success": False, "error": f"删除失败: {str(e)}"}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            Dict: 统计信息
        """
        try:
            stats = self.db_manager.get_stats()
            
            return {
                "success": True,
                "stats": {
                    "total_documents": stats['total_documents'],
                    "total_chunks": stats['total_chunks'],
                    "total_vectors": stats['total_vectors'],
                    "total_chats": stats['total_chats'],
                    "system_status": "运行正常"
                }
            }
        except Exception as e:
            return {"success": False, "error": f"获取统计信息失败: {str(e)}"}


# 创建Flask应用
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app)  # 启用跨域支持

# 创建RAG服务实例
rag_service = RAGWebService()


@app.route('/')
def index():
    """主页"""
    return send_from_directory('static', 'index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    """静态文件服务"""
    return send_from_directory('static', filename)


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        # 检查Ollama客户端状态
        is_ollama_available = not hasattr(rag_service.ollama_client, 'is_mock')
        
        # 获取Ollama配置信息
        ollama_info = {
            "url": f"http://{rag_service.ollama_client.host}:{rag_service.ollama_client.port}",
            "chat_model": rag_service.ollama_client.chat_model,
            "embedding_model": rag_service.ollama_client.embedding_model,
            "available": is_ollama_available,
            "status": "connected" if is_ollama_available else "disconnected"
        }
        
        return jsonify({
            "status": "healthy" if is_ollama_available else "degraded",
            "timestamp": datetime.now().isoformat(),
            "service": "RAG Knowledge Base API",
            "ollama": ollama_info
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "service": "RAG Knowledge Base API",
            "error": str(e)
        }), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """文件上传接口"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "没有文件"}), 400
        
        file = request.files['file']
        result = rag_service.upload_document(file)
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except RequestEntityTooLarge:
        return jsonify({"success": False, "error": "文件太大"}), 413
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """对话接口"""
    try:
        # 检查Ollama客户端状态
        if hasattr(rag_service.ollama_client, 'is_mock'):
            return jsonify({
                "success": False, 
                "error": "Ollama服务未连接，请检查Ollama是否正常运行"
            }), 503
        
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"success": False, "error": "缺少问题参数"}), 400
        
        question = data['question']
        top_k = data.get('top_k', 1)
        stream = data.get('stream', False)  # 是否使用流式响应
        
        if stream:
            return stream_chat(question, top_k)
        else:
            result = rag_service.chat(question, top_k)
            
            if result["success"]:
                return jsonify(result), 200
            else:
                return jsonify(result), 400
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def stream_chat(question, top_k=1):
    """流式对话接口"""
    try:
        # 检查Ollama客户端状态
        if hasattr(rag_service.ollama_client, 'is_mock'):
            return jsonify({
                "success": False, 
                "error": "Ollama服务未连接，请检查Ollama是否正常运行"
            }), 503
        
        # 生成问题的向量
        question_vector = rag_service.ollama_client.get_embedding(question)
        if not question_vector:
            return jsonify({"success": False, "error": "问题向量化失败"}), 400
        
        # 搜索相关文档
        search_results = rag_service.vector_store.search_similar(
            query_vector=question_vector,
            top_k=top_k,
            score_threshold=0.3
        )
        
        if not search_results:
            return jsonify({
                "success": True,
                "answer": "抱歉，我在知识库中没有找到相关信息来回答您的问题。请尝试上传相关文档或换个问题。",
                "sources": [],
                "question": question
            }), 200
        
        # 构建上下文，过滤低相似度结果
        context_parts = []
        sources = []
        
        for result in search_results:
            payload = result.get('payload', {})
            text = payload.get('text', '')
            filename = payload.get('filename', '未知文件')
            chunk_index = payload.get('chunk_index', 0)
            score = result.get('score', 0)
            
            # 过滤相似度低于0.5的结果
            if score < 0.5:
                continue
                
            context_parts.append(text)
            sources.append({
                "filename": filename,
                "chunk_index": chunk_index,
                "score": round(score, 3),
                "text_preview": text[:200] + "..." if len(text) > 200 else text
            })
        
        # 检查过滤后是否还有有效结果
        if not context_parts:
            return jsonify({
                "success": True,
                "answer": "抱歉，我在知识库中没有找到足够相关的信息来回答您的问题。请尝试上传相关文档或换个问题。",
                "sources": [],
                "question": question
            }), 200
        
        context = "\n\n".join(context_parts)
        
        # 构建提示
        prompt = f"""基于以下上下文信息，回答用户的问题。如果上下文中没有相关信息，请诚实地说明。

上下文信息：
{context}

用户问题：{question}

请提供准确、有用的回答："""
        
        # 使用流式生成
        def generate():
            # 先返回元数据
            metadata = json.dumps({
                "success": True,
                "sources": sources,
                "question": question,
                "metadata": True
            })
            yield f"data: {metadata}\n\n"
            
            # 然后流式返回生成的内容
            full_answer = ""
            for text_chunk in rag_service.ollama_client.generate_stream_response(prompt):
                full_answer += text_chunk
                chunk_data = json.dumps({"chunk": text_chunk})
                yield f"data: {chunk_data}\n\n"
            
            # 记录对话历史
            chat_record = {
                "id": str(uuid.uuid4()),
                "question": question,
                "answer": full_answer,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            }
            rag_service.chat_history.append(chat_record)
            
            # 发送完成信号
            done_data = json.dumps({"done": True})
            yield f"data: {done_data}\n\n"
        
        return app.response_class(
            generate(),
            mimetype='text/event-stream'
        )
        
    except Exception as e:
        print(f"❌ 流式对话处理失败: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": f"流式对话处理失败: {str(e)}"}), 500


@app.route('/api/documents', methods=['GET'])
def get_documents():
    """
    获取文档列表（支持分页）
    """
    try:
        # 获取分页参数
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 5, type=int)  # 默认每页5条
        
        # 参数验证
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 50:  # 限制每页最多50条
            page_size = 5
        
        result = rag_service.get_documents(page, page_size)
        return jsonify(result)
    except Exception as e:
        print(f"❌ 获取文档列表失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "documents": [],
            "pagination": {
                "current_page": 1,
                "page_size": 5,
                "total_items": 0,
                "total_pages": 0,
                "has_next": False,
                "has_prev": False
            }
        }), 500


@app.route('/api/documents/<file_id>', methods=['DELETE'])
def delete_document(file_id):
    """删除文档接口"""
    try:
        result = rag_service.delete_document(file_id)
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/clear', methods=['POST'])
def clear_knowledge_base():
    """清理知识库接口"""
    try:
        result = rag_service.clear_knowledge_base()
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取统计信息接口"""
    try:
        result = rag_service.get_stats()
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({"success": False, "error": "接口不存在"}), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({"success": False, "error": "服务器内部错误"}), 500


if __name__ == '__main__':
    print("🚀 启动RAG知识库Web服务...")
    
    # 检查组件初始化状态
    if not rag_service.ollama_client:
        print("⚠️ 部分组件初始化失败，但继续启动服务")
    else:
        print("✅ RAG系统初始化成功")
    
    print("📝 API接口:")
    print("   - GET  /api/health     - 健康检查")
    print("   - POST /api/upload     - 文件上传")
    print("   - POST /api/chat       - 对话接口")
    print("   - GET  /api/documents  - 获取文档列表")
    print("   - DELETE /api/documents/<id> - 删除文档")
    print("   - GET  /api/stats      - 获取统计信息")
    print("\n🌐 Web服务地址: http://localhost:5000")
    
    # 根据环境变量决定是否开启调试模式
    debug_mode = os.getenv('FLASK_ENV', 'development') == 'development'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)