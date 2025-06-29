/**
 * RAG知识库系统 - 前端交互脚本
 * 处理文件上传、对话交互、文档管理等功能
 */

// 全局变量
let isUploading = false;
let currentDocuments = [];
let currentStats = {};

class RAGWebApp {
    constructor() {
        this.apiBase = '/api';
        this.isConnected = false;
        this.isProcessing = false;
        this.documents = [];
        this.chatHistory = [];
        
        // 分页相关属性
        this.currentPage = 1;
        this.pageSize = 5;
        this.totalPages = 1;
        
        this.initializeElements();
        this.bindEvents();
        this.checkConnection();
        this.loadDocuments();
        this.loadStats();
    }
    
    /**
     * 初始化DOM元素引用
     */
    initializeElements() {
        // 状态元素
        this.statusDot = document.getElementById('connection-status');
        this.statusText = document.getElementById('status-text');
        
        // 清理按钮
        this.clearKbBtn = document.getElementById('clear-kb-btn');
        
        // 上传相关元素
        this.uploadArea = document.querySelector('.sidebar'); // 改为整个侧边栏支持拖拽
        this.uploadBtn = document.getElementById('upload-btn');
        this.fileInput = document.getElementById('file-input');
        this.uploadProgress = document.getElementById('upload-progress');
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');
        
        // 文档列表
        this.documentsList = document.getElementById('documents-list');
        
        // 统计元素
        this.docsCount = document.getElementById('docs-count');
        this.chunksCount = document.getElementById('chunks-count');
        this.vectorsCount = document.getElementById('vectors-count');
        this.chatsCount = document.getElementById('chats-count');
        
        // 聊天相关元素
        this.chatMessages = document.getElementById('chat-messages');
        this.chatInput = document.getElementById('chat-input');
        this.sendBtn = document.getElementById('send-btn');
        this.clearChatBtn = document.getElementById('clear-chat-btn');
        this.charCount = document.getElementById('char-count');
        
        // 加载和通知元素
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.loadingText = document.getElementById('loading-text');
        this.notificationContainer = document.getElementById('notification-container');
        
        // 模态框元素
        this.documentModal = document.getElementById('document-modal');
        this.modalTitle = document.getElementById('modal-title');
        this.modalBody = document.getElementById('modal-body');
        this.deleteDocBtn = document.getElementById('delete-doc-btn');
    }
    
    /**
     * 绑定事件监听器
     */
    bindEvents() {
        // 文件上传事件
        this.uploadBtn.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // 聊天事件
        this.chatInput.addEventListener('input', this.handleInputChange.bind(this));
        this.chatInput.addEventListener('keydown', this.handleKeyDown.bind(this));
        this.sendBtn.addEventListener('click', this.sendMessage.bind(this));
        this.clearChatBtn.addEventListener('click', this.clearChat.bind(this));
        
        // 清理知识库事件
        this.clearKbBtn.addEventListener('click', this.clearKnowledgeBase.bind(this));
        
        // 模态框事件
        this.deleteDocBtn.addEventListener('click', this.handleDeleteDocument.bind(this));
        
        // 分页事件
        const prevBtn = document.getElementById('prev-page');
        const nextBtn = document.getElementById('next-page');
        if (prevBtn) {
            prevBtn.addEventListener('click', this.handlePrevPage.bind(this));
        }
        if (nextBtn) {
            nextBtn.addEventListener('click', this.handleNextPage.bind(this));
        }
        
        // 全局事件
        document.addEventListener('click', this.handleGlobalClick.bind(this));
        window.addEventListener('beforeunload', this.handleBeforeUnload.bind(this));
    }
    
    /**
     * 检查服务器连接状态
     */
    async checkConnection() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            if (response.ok) {
                const data = await response.json();
                const ollama = data.ollama;
                
                if (ollama.available) {
                    const statusMessage = `连接正常 | ${ollama.url} | 对话: ${ollama.chat_model} | 嵌入: ${ollama.embedding_model}`;
                    this.setConnectionStatus(true, statusMessage);
                } else {
                    const statusMessage = `Ollama服务未连接 | ${ollama.url} | 状态: ${ollama.status}`;
                    this.setConnectionStatus(false, statusMessage);
                    this.showNotification('Ollama服务未连接，聊天功能不可用', 'warning');
                }
            } else {
                throw new Error('服务器响应异常');
            }
        } catch (error) {
            this.setConnectionStatus(false, '连接失败');
            this.showNotification('连接服务器失败', 'error');
        }
    }
    
    /**
     * 设置连接状态
     */
    setConnectionStatus(connected, message) {
        this.isConnected = connected;
        this.statusDot.className = `status-dot ${connected ? 'online' : 'offline'}`;
        this.statusText.textContent = message;
    }
    


    /**
     * 处理拖拽悬停
     */
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }
    
    /**
     * 处理拖拽离开
     */
    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }
    
    /**
     * 处理文件拖拽放置
     */
    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files);
        this.uploadFiles(files);
    }
    
    /**
     * 处理文件选择
     */
    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        this.uploadFiles(files);
        e.target.value = ''; // 清空input，允许重复选择同一文件
    }
    
    /**
     * 上传文件
     */
    async uploadFiles(files) {
        if (!this.isConnected) {
            this.showNotification('请先连接到服务器', 'error');
            return;
        }
        
        if (files.length === 0) return;
        
        // 检查文件类型
        const allowedTypes = ['pdf', 'docx', 'txt', 'md'];
        const invalidFiles = files.filter(file => {
            const ext = file.name.split('.').pop().toLowerCase();
            return !allowedTypes.includes(ext);
        });
        
        if (invalidFiles.length > 0) {
            this.showNotification(
                `不支持的文件类型: ${invalidFiles.map(f => f.name).join(', ')}`,
                'error'
            );
            return;
        }
        
        // 检查文件大小（16MB限制）
        const oversizedFiles = files.filter(file => file.size > 16 * 1024 * 1024);
        if (oversizedFiles.length > 0) {
            this.showNotification(
                `文件过大: ${oversizedFiles.map(f => f.name).join(', ')}`,
                'error'
            );
            return;
        }
        
        // 逐个上传文件
        for (const file of files) {
            await this.uploadSingleFile(file);
        }
    }
    
    /**
     * 上传单个文件
     */
    async uploadSingleFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            // 阶段1: 上传文件 (0-25%)
            this.showUploadProgress(true, `📤 正在上传文件: ${file.name}`, 25);
            await this.delay(500); // 短暂延迟以显示状态
            
            const response = await fetch(`${this.apiBase}/upload`, {
                method: 'POST',
                body: formData
            });
            
            // 阶段2: 处理响应 (25-50%)
            this.showUploadProgress(true, `📋 正在处理文档: ${file.name}`, 50);
            await this.delay(300);
            
            const result = await response.json();
            
            if (result.success) {
                // 阶段3: 向量化处理 (50-80%)
                this.showUploadProgress(true, `🧠 正在生成向量索引: ${file.name}`, 80);
                await this.delay(800);
                
                // 阶段4: 存储到知识库 (80-95%)
                this.showUploadProgress(true, `💾 正在存储到知识库: ${file.name}`, 95);
                await this.delay(500);
                
                // 完成 (100%)
                this.showUploadProgress(true, `✅ 处理完成: ${file.name}`, 100);
                await this.delay(800);
                
                this.showNotification(
                    `文件处理成功: ${file.name}`,
                    'success'
                );
                this.loadDocuments();
                this.loadStats();
            } else {
                throw new Error(result.error || '上传失败');
            }
        } catch (error) {
            this.showNotification(
                `处理失败: ${file.name} - ${error.message}`,
                'error'
            );
        } finally {
            this.showUploadProgress(false);
        }
    }
    
    /**
     * 延迟函数
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    /**
     * 显示/隐藏上传进度
     */
    showUploadProgress(show, text = '', progress = 100) {
        if (show) {
            this.uploadProgress.style.display = 'block';
            this.progressText.textContent = text;
            // 添加平滑的进度条动画
            setTimeout(() => {
                this.progressFill.style.width = `${progress}%`;
            }, 100);
        } else {
            // 隐藏时先将进度条归零，然后隐藏
            this.progressFill.style.width = '0%';
            setTimeout(() => {
                this.uploadProgress.style.display = 'none';
            }, 300);
        }
    }
    
    /**
     * 加载文档列表
     */
    async loadDocuments(page = 1) {
        try {
            const response = await fetch(`${this.apiBase}/documents?page=${page}&page_size=${this.pageSize}`);
            const result = await response.json();
            
            if (result.success) {
                this.documents = result.documents;
                this.renderDocuments();
                currentDocuments = result.documents;
                
                // 更新分页信息
                if (result.pagination) {
                    this.currentPage = result.pagination.current_page;
                    this.totalPages = result.pagination.total_pages;
                    this.updatePaginationControls(result.pagination);
                }
            } else {
                console.error('加载文档失败:', result.error);
                this.showNotification('加载文档失败', 'error');
            }
        } catch (error) {
            console.error('加载文档列表失败:', error);
            this.showNotification('加载文档失败', 'error');
        }
    }
    
    /**
     * 更新分页控件
     */
    updatePaginationControls(pagination) {
        const paginationControls = document.getElementById('pagination-controls');
        const prevBtn = document.getElementById('prev-page');
        const nextBtn = document.getElementById('next-page');
        const pageInfo = document.getElementById('pagination-info');
        
        // 显示或隐藏分页控件
        if (pagination.total_pages > 1) {
            paginationControls.style.display = 'flex';
        } else {
            paginationControls.style.display = 'none';
        }
        
        // 更新按钮状态
        if (prevBtn) {
            prevBtn.disabled = !pagination.has_prev;
        }
        
        if (nextBtn) {
            nextBtn.disabled = !pagination.has_next;
        }
        
        // 更新页码信息
        if (pageInfo) {
            pageInfo.textContent = `${pagination.current_page} / ${pagination.total_pages}`;
        }
    }
    
    /**
     * 渲染文档列表
     */
    renderDocuments() {
        const documentsList = document.getElementById('documents-list');
        
        if (!this.documents || this.documents.length === 0) {
            documentsList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-folder-open"></i>
                    <p>暂无文档</p>
                </div>
            `;
            return;
        }
        
        documentsList.innerHTML = this.documents.map(doc => `
            <div class="document-item" data-doc-id="${doc.id}">
                <div class="document-info">
                    <div class="document-name" title="${doc.filename}">
                        <i class="fas fa-file-alt"></i>
                        ${doc.filename}
                    </div>
                    <div class="document-meta">
                        <span class="document-size">${this.formatFileSize(doc.file_size)}</span>
                        <span class="document-chunks">${doc.chunks_count} 块</span>
                        <span class="document-date">${this.formatDate(doc.upload_time)}</span>
                    </div>
                </div>
                <div class="document-actions">
                    <button class="action-btn view-btn" title="查看详情">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="action-btn delete-btn" title="删除文档">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `).join('');
        
        // 添加事件监听器
        this.documents.forEach(doc => {
            const docElement = document.querySelector(`[data-doc-id="${doc.id}"]`);
            
            // 查看详情
            const viewBtn = docElement.querySelector('.view-btn');
            viewBtn.addEventListener('click', () => {
                this.showDocumentDetails(doc.id);
            });
            
            // 删除文档
            const deleteBtn = docElement.querySelector('.delete-btn');
            deleteBtn.addEventListener('click', () => {
                this.deleteDocument(doc.id);
            });
        });
    }
    
    /**
     * 删除文档
     */
    async deleteDocument(docId) {
        if (!confirm('确定要删除这个文档吗？')) {
            return;
        }
        
        try {
            const response = await fetch(`${this.apiBase}/documents/${docId}`, {
                method: 'DELETE'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification('文档删除成功', 'success');
                // 重新加载当前页，如果当前页没有数据则加载上一页
                this.loadDocuments(currentPage);
                this.loadStats();
            } else {
                this.showNotification(result.error || '删除失败', 'error');
            }
        } catch (error) {
            console.error('删除文档失败:', error);
            this.showNotification('删除失败', 'error');
        }
    }
    
    /**
     * 获取文件图标
     */
    getFileIcon(ext) {
        const iconMap = {
            'pdf': 'pdf',
            'docx': 'word',
            'txt': 'alt',
            'md': 'markdown'
        };
        return iconMap[ext] || 'alt';
    }
    
    /**
     * 格式化文件大小
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }
    
    /**
     * 格式化日期
     */
    formatDate(dateString) {
        if (!dateString) return '未知时间';
        try {
            const date = new Date(dateString);
            return date.toLocaleString('zh-CN', {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch (error) {
            return '时间格式错误';
        }
    }
    
    /**
     * 显示文档详情
     */
    showDocumentDetails(docId) {
        const doc = this.documents.find(d => d.id === docId);
        if (!doc) return;
        
        const uploadTime = new Date(doc.upload_time).toLocaleString('zh-CN');
        const fileSize = this.formatFileSize(doc.file_size);
        
        this.modalTitle.textContent = doc.filename;
        this.modalBody.innerHTML = `
            <div class="document-details">
                <div class="detail-row">
                    <strong>文件名:</strong> ${doc.filename}
                </div>
                <div class="detail-row">
                    <strong>文件大小:</strong> ${fileSize}
                </div>
                <div class="detail-row">
                    <strong>文档块数:</strong> ${doc.chunks_count}
                </div>
                <div class="detail-row">
                    <strong>向量数:</strong> ${doc.vectors_count}
                </div>
                <div class="detail-row">
                    <strong>上传时间:</strong> ${uploadTime}
                </div>
            </div>
        `;
        
        this.deleteDocBtn.dataset.docId = docId;
        this.documentModal.style.display = 'flex';
    }
    
    /**
     * 处理删除文档
     */
    async handleDeleteDocument() {
        const docId = this.deleteDocBtn.dataset.docId;
        if (!docId) return;
        
        if (!confirm('确定要删除这个文档吗？此操作不可撤销。')) {
            return;
        }
        
        this.showLoading(true, '正在删除文档...');
        
        try {
            const response = await fetch(`${this.apiBase}/documents/${docId}`, {
                method: 'DELETE'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification('文档删除成功', 'success');
                this.closeModal();
                this.loadDocuments();
                this.loadStats();
            } else {
                throw new Error(result.error || '删除失败');
            }
        } catch (error) {
            this.showNotification(`删除失败: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    /**
     * 关闭模态框
     */
    closeModal() {
        this.documentModal.style.display = 'none';
    }
    
    /**
     * 加载统计信息
     */
    async loadStats() {
        try {
            const response = await fetch(`${this.apiBase}/stats`);
            const result = await response.json();
            
            if (result.success) {
                const stats = result.stats;
                this.docsCount.textContent = stats.total_documents;
                this.vectorsCount.textContent = stats.total_vectors;
            }
        } catch (error) {
            console.error('加载统计信息失败:', error);
        }
    }
    
    /**
     * 处理输入变化
     */
    handleInputChange() {
        const text = this.chatInput.value;
        const length = text.length;
        
        this.charCount.textContent = length;
        this.sendBtn.disabled = length === 0 || this.isProcessing;
        
        // 自动调整高度
        this.chatInput.style.height = 'auto';
        this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 120) + 'px';
    }
    
    /**
     * 处理键盘事件
     */
    handleKeyDown(e) {
        if (e.key === 'Enter') {
            if (e.ctrlKey || e.metaKey) {
                // Ctrl+Enter 换行
                return;
            } else {
                // Enter 发送消息
                e.preventDefault();
                this.sendMessage();
            }
        }
    }
    
    /**
     * 发送消息
     */
    async sendMessage() {
        const question = this.chatInput.value.trim();
        if (!question || this.isProcessing) return;
        
        if (!this.isConnected) {
            this.showNotification('服务未连接，请检查连接状态', 'error');
            return;
        }
        
        // 添加用户消息到界面
        this.addMessage(question, 'user');
        this.chatInput.value = '';
        this.handleInputChange();
        
        // 显示打字指示器
        const typingId = this.showTypingIndicator();
        
        this.isProcessing = true;
        this.sendBtn.disabled = true;
        
        try {
            // 使用流式响应
            await this.sendStreamMessage(question, typingId);
        } catch (error) {
            this.removeTypingIndicator(typingId);
            this.addMessage(
                `抱歉，处理您的问题时出现错误: ${error.message}`,
                'bot'
            );
            this.showNotification(`对话失败: ${error.message}`, 'error');
        } finally {
            this.isProcessing = false;
            this.sendBtn.disabled = false;
        }
    }
    
    /**
     * 发送消息（非流式）
     */
    async sendStreamMessage(question, typingId) {
        try {
            const response = await fetch(`${this.apiBase}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    question: question,
                    stream: false 
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            // 解析JSON响应
            const data = await response.json();
            
            // 移除打字指示器
            this.removeTypingIndicator(typingId);
            
            // 检查响应是否成功
            if (!data.success) {
                throw new Error(data.error || '服务器返回错误');
            }
            
            // 创建机器人消息容器
            const messageId = 'msg-' + Date.now();
            const messageHtml = `
                <div class="message bot-message" id="${messageId}">
                    <div class="message-avatar"><i class="fas fa-robot"></i></div>
                    <div class="message-content">
                        <p class="message-text"></p>
                        <div class="sources-container" style="display: none;"></div>
                    </div>
                </div>
            `;
            
            // 移除欢迎消息（如果存在）
            const welcomeMessage = this.chatMessages.querySelector('.welcome-message');
            if (welcomeMessage) {
                welcomeMessage.remove();
            }
            
            this.chatMessages.insertAdjacentHTML('beforeend', messageHtml);
            this.scrollToBottom();
            
            const messageElement = document.getElementById(messageId);
            const contentElement = messageElement.querySelector('.message-text');
            const sourcesContainer = messageElement.querySelector('.sources-container');
            
            // 显示回答
            const fullAnswer = data.answer || '抱歉，没有获取到回答';
            const sources = data.sources || [];
            const mode = data.mode || '未知模式';
            const thinking = data.thinking || '';
            
            // 添加模式标识
            let modeHtml = '';
            if (mode === '基础模型回答') {
                modeHtml = '<div class="answer-mode base-model"><i class="fas fa-brain"></i> 基础模型回答</div>';
            } else if (mode === '知识库回答') {
                modeHtml = '<div class="answer-mode knowledge-base"><i class="fas fa-database"></i> 知识库回答</div>';
            }
            
            // 添加思考内容（使用引用格式）
            let thinkingHtml = '';
            if (thinking) {
                thinkingHtml = '<div class="thinking-content"><blockquote><strong>💭 思考过程：</strong><br>' + thinking + '</blockquote></div>';
            }
            
            contentElement.innerHTML = modeHtml + thinkingHtml + this.formatMessageContent(fullAnswer);
             this.scrollToBottom();
             
             // 添加sources信息
             if (sources && sources.length > 0) {
                 let sourcesHtml = '<div class="message-sources">';
                 sourcesHtml += '<div class="sources-header"><i class="fas fa-book"></i> 参考来源:</div>';
                 sources.forEach((source, index) => {
                     sourcesHtml += `
                         <div class="source-item" data-filename="${source.filename}" data-chunk="${source.chunk_index}">
                             <span class="source-name">${source.filename}</span>
                             <span class="source-score">相似度: ${source.score}</span>
                         </div>
                     `;
                 });
                 sourcesHtml += '</div>';
                 
                 sourcesContainer.innerHTML = sourcesHtml;
                 sourcesContainer.style.display = 'block';
             }
             
             // 添加到历史记录
             this.chatHistory.push({ 
                 type: 'bot', 
                 content: fullAnswer, 
                 sources: sources, 
                 timestamp: new Date() 
             });
             
             this.loadStats(); // 更新对话统计
            
        } catch (error) {
            console.error('流式消息发送失败:', error);
            throw error;
        }
    }
    
    /**
     * 添加消息到聊天界面
     */
    addMessage(content, type, sources = null) {
        const messageId = 'msg-' + Date.now();
        const avatar = type === 'user' 
            ? '<i class="fas fa-user"></i>' 
            : '<i class="fas fa-robot"></i>';
        
        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            const sourcesItems = sources.map(source => `
                <div class="source-item">
                    <div class="source-filename">
                        <i class="fas fa-file-alt"></i> ${source.filename}
                    </div>
                    <div class="source-meta">
                        第${source.chunk_index + 1}块 • 相似度: ${(source.score * 100).toFixed(1)}%
                    </div>
                    <div class="source-preview">${source.text_preview}</div>
                </div>
            `).join('');
            
            sourcesHtml = `
                <div class="sources-container">
                    <div class="sources-title">
                        <i class="fas fa-link"></i> 相关知识来源
                    </div>
                    ${sourcesItems}
                </div>
            `;
        }
        
        // 为用户消息添加重新发送按钮
        const resendButton = type === 'user' ? `
            <button class="resend-btn" onclick="chatApp.resendMessage('${messageId}', '${content.replace(/'/g, "\\'")}')"
                    title="重新发送">
                <i class="fas fa-redo"></i>
            </button>
        ` : '';
        
        const messageHtml = `
            <div class="message ${type}-message" id="${messageId}">
                ${resendButton}
                <div class="message-avatar">${avatar}</div>
                <div class="message-content">
                    <p>${this.formatMessageContent(content)}</p>
                    ${sourcesHtml}
                </div>
            </div>
        `;
        
        // 移除欢迎消息（如果存在）
        const welcomeMessage = this.chatMessages.querySelector('.welcome-message');
        if (welcomeMessage && type === 'user') {
            welcomeMessage.remove();
        }
        
        this.chatMessages.insertAdjacentHTML('beforeend', messageHtml);
        this.scrollToBottom();
        
        // 添加到历史记录
        this.chatHistory.push({ type, content, sources, timestamp: new Date() });
    }
    
    /**
     * 格式化消息内容
     */
    formatMessageContent(content) {
        try {
            // 使用marked.js渲染markdown
            if (typeof marked !== 'undefined') {
                const html = marked.parse(content);
                // 渲染完成后触发MathJax重新渲染
                setTimeout(() => {
                    if (window.MathJax && window.MathJax.typesetPromise) {
                        window.MathJax.typesetPromise();
                    }
                }, 100);
                return html;
            } else {
                // 降级处理：简单的换行处理
                return content.replace(/\n/g, '<br>');
            }
        } catch (error) {
            console.error('Markdown渲染错误:', error);
            return content.replace(/\n/g, '<br>');
        }
    }
    
    /**
     * 显示打字指示器
     */
    showTypingIndicator() {
        const typingId = 'typing-' + Date.now();
        const typingHtml = `
            <div class="message bot-message" id="${typingId}">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="typing-indicator">
                        <span>正在思考</span>
                        <div class="typing-dots">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        this.chatMessages.insertAdjacentHTML('beforeend', typingHtml);
        this.scrollToBottom();
        
        return typingId;
    }
    
    /**
     * 移除打字指示器
     */
    removeTypingIndicator(typingId) {
        const typingElement = document.getElementById(typingId);
        if (typingElement) {
            typingElement.remove();
        }
    }
    
    /**
     * 滚动到底部
     */
    scrollToBottom() {
        requestAnimationFrame(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        });
    }
    
    /**
     * 重新发送消息
     */
    resendMessage(messageId, content) {
        if (this.isProcessing) return;
        
        // 设置输入框内容并发送
        this.chatInput.value = content;
        this.handleInputChange();
        this.sendMessage();
    }
    
    /**
     * 清空聊天
     */
    clearChat() {
        if (this.chatHistory.length === 0) return;
        
        if (!confirm('确定要清空所有对话吗？')) return;
        
        this.chatMessages.innerHTML = `
            <div class="welcome-message">
                <div class="message bot-message">
                    <div class="message-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="message-content">
                        <p>👋 欢迎使用RAG知识库系统！</p>
                        <p>请先上传文档到知识库，然后就可以开始提问了。我会基于您上传的文档内容来回答问题。</p>
                    </div>
                </div>
            </div>
        `;
        
        this.chatHistory = [];
        this.showNotification('对话已清空', 'success');
    }
    
    /**
     * 显示加载遮罩
     */
    showLoading(show, text = '处理中...') {
        if (show) {
            this.loadingText.textContent = text;
            this.loadingOverlay.style.display = 'flex';
        } else {
            this.loadingOverlay.style.display = 'none';
        }
    }
    
    /**
     * 显示通知
     */
    showNotification(message, type = 'info', duration = 5000) {
        const notificationId = 'notification-' + Date.now();
        const iconMap = {
            'success': 'fas fa-check-circle',
            'error': 'fas fa-exclamation-circle',
            'warning': 'fas fa-exclamation-triangle',
            'info': 'fas fa-info-circle'
        };
        
        const notificationHtml = `
            <div class="notification ${type}" id="${notificationId}">
                <i class="${iconMap[type]}"></i>
                <div class="notification-content">
                    <div class="notification-message">${message}</div>
                </div>
            </div>
        `;
        
        this.notificationContainer.insertAdjacentHTML('beforeend', notificationHtml);
        
        // 自动移除通知
        setTimeout(() => {
            const notification = document.getElementById(notificationId);
            if (notification) {
                notification.style.animation = 'notificationSlideOut 0.3s ease-out';
                setTimeout(() => notification.remove(), 300);
            }
        }, duration);
    }
    
    /**
     * 处理全局点击事件
     */
    handleGlobalClick(e) {
        // 点击模态框外部关闭模态框
        if (e.target === this.documentModal) {
            this.closeModal();
        }
    }
    
    /**
     * 处理页面卸载前事件
     */
    handleBeforeUnload(e) {
        if (this.isProcessing) {
            e.preventDefault();
            e.returnValue = '正在处理请求，确定要离开吗？';
        }
    }
    
    /**
     * 清理知识库
     */
    async clearKnowledgeBase() {
        // 显示确认对话框
        const confirmed = confirm('确定要清理整个知识库吗？\n\n此操作将：\n• 删除所有上传的文档\n• 清空向量数据库\n• 清除聊天历史\n\n此操作不可撤销！');
        
        if (!confirmed) {
            return;
        }
        
        try {
            this.isProcessing = true;
            this.clearKbBtn.disabled = true;
            this.clearKbBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>清理中...</span>';
            
            const response = await fetch(`${this.apiBase}/clear`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification('知识库已成功清理', 'success');
                
                // 重新加载页面数据
                this.documents = [];
                this.chatHistory = [];
                this.renderDocuments();
                this.clearChatMessages();
                this.loadStats();
            } else {
                throw new Error(result.error || '清理失败');
            }
        } catch (error) {
            console.error('清理知识库失败:', error);
            this.showNotification(`清理失败: ${error.message}`, 'error');
        } finally {
            this.isProcessing = false;
            this.clearKbBtn.disabled = false;
            this.clearKbBtn.innerHTML = '<i class="fas fa-trash-alt"></i><span>清理知识库</span>';
        }
    }
    
    /**
     * 处理上一页
     */
    handlePrevPage() {
        if (this.currentPage > 1) {
            this.loadDocuments(this.currentPage - 1);
        }
    }
    
    /**
     * 处理下一页
     */
    handleNextPage() {
        if (this.currentPage < this.totalPages) {
            this.loadDocuments(this.currentPage + 1);
        }
    }
    
    /**
     * 清空聊天消息
     */
    clearChatMessages() {
        this.chatMessages.innerHTML = '';
    }
}

// 全局函数
function closeModal() {
    if (window.ragApp) {
        window.ragApp.closeModal();
    }
}

// 添加通知滑出动画
const style = document.createElement('style');
style.textContent = `
    @keyframes notificationSlideOut {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100%);
        }
    }
`;
document.head.appendChild(style);

// 页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.ragApp = new RAGWebApp();
    console.log('RAG Web App 初始化完成');
});