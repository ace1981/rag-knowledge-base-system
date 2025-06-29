# -*- coding: utf-8 -*-
"""
启动持久化模式的Qdrant服务
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from loguru import logger
import requests

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_docker():
    """
    检查Docker是否安装并运行
    """
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Docker已安装: {result.stdout.strip()}")
            return True
        else:
            logger.error("❌ Docker未安装或无法访问")
            return False
    except FileNotFoundError:
        logger.error("❌ Docker未安装")
        return False

def check_docker_compose():
    """
    检查Docker Compose是否可用
    """
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Docker Compose已安装: {result.stdout.strip()}")
            return True
        else:
            # 尝试新版本的docker compose命令
            result = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Docker Compose已安装: {result.stdout.strip()}")
                return True
            else:
                logger.error("❌ Docker Compose未安装或无法访问")
                return False
    except FileNotFoundError:
        logger.error("❌ Docker Compose未安装")
        return False

def start_qdrant_service():
    """
    启动Qdrant持久化服务
    """
    try:
        logger.info("🚀 启动Qdrant持久化服务...")
        
        # 首先尝试停止现有服务
        logger.info("停止现有Qdrant服务...")
        subprocess.run(['docker-compose', 'down'], cwd=project_root, capture_output=True)
        
        # 启动Qdrant服务
        result = subprocess.run(
            ['docker-compose', 'up', '-d', 'qdrant'],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Qdrant服务启动命令执行成功")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"❌ Qdrant服务启动失败: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 启动Qdrant服务时出错: {e}")
        return False

def wait_for_qdrant(max_wait=60):
    """
    等待Qdrant服务启动完成
    """
    logger.info("⏳ 等待Qdrant服务启动...")
    
    for i in range(max_wait):
        try:
            response = requests.get('http://localhost:6333/collections', timeout=5)
            if response.status_code == 200:
                logger.info("✅ Qdrant服务已就绪")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
        if (i + 1) % 10 == 0:
            logger.info(f"⏳ 等待中... ({i + 1}/{max_wait}秒)")
    
    logger.error("❌ Qdrant服务启动超时")
    return False

def test_qdrant_connection():
    """
    测试Qdrant连接
    """
    try:
        from qdrant_client import QdrantClient
        
        # 连接到持久化的Qdrant服务
        client = QdrantClient(host="localhost", port=6333)
        
        # 获取集合列表
        collections = client.get_collections()
        logger.info(f"✅ 成功连接到Qdrant，当前集合数量: {len(collections.collections)}")
        
        # 显示现有集合
        if collections.collections:
            logger.info("现有集合:")
            for collection in collections.collections:
                logger.info(f"  - {collection.name}")
        else:
            logger.info("暂无集合")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Qdrant连接测试失败: {e}")
        return False

def main():
    """
    主函数
    """
    logger.info("=== 启动Qdrant持久化服务 ===")
    
    # 检查Docker环境
    if not check_docker():
        logger.error("请先安装Docker")
        sys.exit(1)
    
    if not check_docker_compose():
        logger.error("请先安装Docker Compose")
        sys.exit(1)
    
    # 启动Qdrant服务
    if not start_qdrant_service():
        logger.error("Qdrant服务启动失败")
        sys.exit(1)
    
    # 等待服务就绪
    if not wait_for_qdrant():
        logger.error("Qdrant服务未能正常启动")
        sys.exit(1)
    
    # 测试连接
    if test_qdrant_connection():
        logger.info("\n🎉 Qdrant持久化服务启动成功！")
        logger.info("📊 服务信息:")
        logger.info("  - REST API: http://localhost:6333")
        logger.info("  - gRPC: localhost:6334")
        logger.info("  - 数据存储: Docker volume (qdrant_storage)")
        logger.info("  - 模式: 持久化存储")
        logger.info("\n现在可以运行RAG应用了")
    else:
        logger.error("❌ Qdrant服务测试失败")
        sys.exit(1)

if __name__ == "__main__":
    main()