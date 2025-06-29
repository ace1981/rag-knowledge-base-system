# RAG知识库系统 Docker部署指南

## 系统架构

本系统采用Docker容器化部署，包含以下组件：

- **rag-app**: RAG知识库Web应用（Flask + Python）
- **qdrant**: 向量数据库服务
- **ollama**: 大语言模型服务（运行在宿主机）

## 前置要求

### 1. 安装Docker和Docker Compose

```bash
# Windows用户请安装Docker Desktop
# 下载地址：https://www.docker.com/products/docker-desktop/

# 验证安装
docker --version
docker-compose --version
```

### 2. 安装并启动Ollama服务

```bash
# 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 启动Ollama服务
ollama serve

# 下载所需模型
ollama pull shaw/dmeta-embedding-zh:latest
ollama pull qwen2.5:14b
```

## 部署步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd qdrant-demo
```

### 2. 配置环境变量

创建 `.env` 文件（如果不存在）：

```bash
# Ollama配置
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_EMBEDDING_MODEL=shaw/dmeta-embedding-zh:latest
OLLAMA_CHAT_MODEL=qwen2.5:14b

# Qdrant配置
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. 构建和启动服务

```bash
# 构建并启动所有服务
docker-compose up --build -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f rag-app
docker-compose logs -f qdrant
```

### 4. 验证部署

- **RAG应用**: http://localhost:5000
- **Qdrant管理界面**: http://localhost:6333/dashboard
- **健康检查**: http://localhost:5000/api/health

## 服务管理

### 启动服务

```bash
docker-compose up -d
```

### 停止服务

```bash
docker-compose down
```

### 重启服务

```bash
docker-compose restart
```

### 查看日志

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f rag-app
docker-compose logs -f qdrant
```

### 更新应用

```bash
# 重新构建并启动
docker-compose up --build -d
```

## 数据持久化

### 向量数据库数据

- Qdrant数据存储在Docker卷 `qdrant_storage` 中
- 数据会在容器重启后保持

### 应用数据

- 上传的文件存储在 `./uploads` 目录
- SQLite数据库文件 `rag_system.db` 存储在项目根目录
- 这些数据通过Docker卷挂载，会在容器重启后保持

## 故障排除

### 1. 容器启动失败

```bash
# 查看详细错误信息
docker-compose logs rag-app

# 检查容器状态
docker-compose ps
```

### 2. Ollama连接失败

- 确保Ollama服务在宿主机上正常运行
- 检查防火墙设置，确保端口11434可访问
- 验证模型是否已下载

```bash
# 检查Ollama状态
curl http://localhost:11434/api/tags

# 检查模型列表
ollama list
```

### 3. Qdrant连接失败

```bash
# 检查Qdrant健康状态
curl http://localhost:6333/health

# 重启Qdrant服务
docker-compose restart qdrant
```

### 4. 端口冲突

如果端口被占用，可以修改 `docker-compose.yml` 中的端口映射：

```yaml
ports:
  - "5001:5000"  # 将应用端口改为5001
```

## 性能优化

### 1. 资源限制

在 `docker-compose.yml` 中添加资源限制：

```yaml
rag-app:
  # ... 其他配置
  deploy:
    resources:
      limits:
        memory: 2G
        cpus: '1.0'
      reservations:
        memory: 1G
        cpus: '0.5'
```

### 2. 生产环境配置

- 设置 `FLASK_ENV=production`
- 使用专用的生产数据库
- 配置反向代理（如Nginx）
- 启用HTTPS

## 监控和日志

### 健康检查

系统提供了内置的健康检查端点：

- RAG应用: `GET /api/health`
- Qdrant: `GET http://localhost:6333/health`

### 日志管理

```bash
# 实时查看日志
docker-compose logs -f --tail=100

# 导出日志到文件
docker-compose logs > app.log
```

## 备份和恢复

### 备份数据

```bash
# 备份Qdrant数据
docker run --rm -v qdrant_storage:/data -v $(pwd):/backup alpine tar czf /backup/qdrant-backup.tar.gz -C /data .

# 备份应用数据
tar czf app-backup.tar.gz uploads/ rag_system.db
```

### 恢复数据

```bash
# 恢复Qdrant数据
docker run --rm -v qdrant_storage:/data -v $(pwd):/backup alpine tar xzf /backup/qdrant-backup.tar.gz -C /data

# 恢复应用数据
tar xzf app-backup.tar.gz
```

## 安全注意事项

1. **网络安全**: 在生产环境中，确保只暴露必要的端口
2. **数据加密**: 考虑对敏感数据进行加密存储
3. **访问控制**: 实施适当的身份验证和授权机制
4. **定期更新**: 保持Docker镜像和依赖包的最新版本

## 扩展部署

### 多实例部署

```yaml
rag-app:
  # ... 其他配置
  deploy:
    replicas: 3
```

### 负载均衡

可以使用Nginx或Traefik作为负载均衡器，分发请求到多个应用实例。