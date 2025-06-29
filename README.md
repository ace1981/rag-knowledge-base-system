# Qdrant + Ollama RAG 知识库演示项目

## 项目简介

本项目演示如何使用本地Qdrant向量数据库和Ollama模型实现RAG（检索增强生成）知识库系统。系统支持文档上传、向量化存储、相似性检索和智能问答。

## 环境要求

- Python 3.11+
- Docker Desktop（用于运行Qdrant）
- Ollama（已安装并运行相关模型）
- Windows 10/11

## 安装步骤

### 1. 安装Docker Desktop

如果还未安装Docker Desktop，请：
1. 访问 [Docker Desktop官网](https://www.docker.com/products/docker-desktop/)
2. 下载并安装Docker Desktop for Windows
3. 启动Docker Desktop应用程序
4. 确保Docker服务正在运行

### 2. 启动Qdrant服务

**方法一：使用Docker Compose（推荐）**
```bash
# 在项目根目录执行
docker-compose up -d
```

**方法二：直接使用Docker命令**
```bash
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

**方法三：使用Qdrant Cloud（如果本地Docker有问题）**
- 访问 [Qdrant Cloud](https://cloud.qdrant.io/)
- 创建免费集群
- 更新.env文件中的连接信息

### 3. 创建Python虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 4. 验证Ollama服务

确保Ollama服务正在运行并已安装所需模型：

```bash
# 检查Ollama服务状态
curl http://localhost:11434/api/tags

# 如果需要，拉取模型
ollama pull qwen2.5:4b    # 嵌入模型
ollama pull qwen2.5:14b   # 生成模型
```

### 5. 配置环境变量

检查并根据需要修改 `.env` 文件中的配置：

```env
# Qdrant 配置
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Ollama 配置
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_EMBEDDING_MODEL=qwen2.5:4b
OLLAMA_CHAT_MODEL=qwen2.5:14b
```

## 使用方法

### 基本使用

```bash
# 运行RAG演示
python rag_demo.py
```

### 功能说明

1. **文档上传**: 支持TXT、PDF、DOCX格式
2. **向量化存储**: 自动将文档分块并生成向量
3. **智能检索**: 基于语义相似性检索相关内容
4. **问答生成**: 结合检索结果生成准确答案

## 项目结构

```
qdrant-demo/
├── rag_demo.py           # 主程序入口
├── document_processor.py # 文档处理模块
├── vector_store.py       # 向量存储操作
├── ollama_client.py      # Ollama模型调用
├── requirements.txt      # Python依赖
├── .env                  # 环境配置
├── docker-compose.yml    # Qdrant服务配置
├── README.md            # 项目说明
├── todolist.md          # 任务清单
└── docs/                # 设计文档目录
```

## 故障排除

### Docker相关问题

1. **Docker服务未启动**
   - 启动Docker Desktop应用程序
   - 等待Docker服务完全启动

2. **端口冲突**
   - 检查6333端口是否被占用
   - 修改docker-compose.yml中的端口映射

### Ollama相关问题

1. **模型未安装**
   ```bash
   ollama list  # 查看已安装模型
   ollama pull qwen2.5:4b  # 安装嵌入模型
   ```

2. **服务未启动**
   ```bash
   ollama serve  # 启动Ollama服务
   ```

## 技术架构

- **向量数据库**: Qdrant - 高性能向量搜索引擎
- **嵌入模型**: Qwen2.5 4B - 文本向量化
- **生成模型**: Qwen2.5 14B - 智能问答
- **文档处理**: 支持多种格式的文档解析
- **检索策略**: 基于余弦相似度的语义检索

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 许可证

MIT License