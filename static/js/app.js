/**
 * LLM Manager — Frontend Application
 * Handles model selection, server lifecycle, and streaming chat.
 */

(() => {
    'use strict';

    // --- DOM References ---
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const modelSelect = $('#model-select');
    const selectTrigger = $('#select-trigger');
    const selectOptions = $('#select-options');
    const modelActionsBar = $('#model-actions-bar');
    const selectedModelInfo = $('#selected-model-info');
    const startModelBtn = $('#start-model-btn');
    const stopModelBtn = $('#stop-model-btn');

    const statusCard = $('#status-card');
    const statusDot = $('#status-dot');
    const statusText = $('#status-text');
    const statusDetails = $('#status-details');
    const activeModelName = $('#active-model-name');
    const chatHeaderBadge = $('#chat-header-badge');
    const chatMessages = $('#chat-messages');
    const welcomeScreen = $('#welcome-screen');
    const chatInput = $('#chat-input');
    const sendBtn = $('#send-btn');
    const clearChatBtn = $('#clear-chat-btn');
    const sidebarToggle = $('#sidebar-toggle');
    const sidebar = $('#sidebar');
    const sidebarOverlay = $('#sidebar-overlay');
    const ctxSizeSlider = $('#ctx-size');
    const ctxSizeValue = $('#ctx-size-value');
    const gpuLayersSlider = $('#gpu-layers');
    const gpuLayersValue = $('#gpu-layers-value');
    const enableToolsCheckbox = $('#enable-tools');
    const systemPromptInput = $('#system-prompt');
    const inputHint = $('#input-hint');

    // Vision UI
    const attachBtn = $('#attach-btn');
    const fileInput = $('#file-input');
    const imagePreview = $('#image-preview');
    const previewImg = $('#preview-img');
    const clearImageBtn = $('#clear-image-btn');
    const chatInputArea = $('#chat-input-area');

    const agentCards = $$('.agent-card');

    // --- State ---
    let currentAgent = 'researcher';
    let currentImageBase64 = null;
    let models = [];
    let selectedModelPath = null;
    let serverState = 'idle';
    let conversationHistory = [];
    let isStreaming = false;
    let statusPollInterval = null;

    // --- Init ---
    async function init() {
        await loadModels();
        await pollStatus();
        startStatusPolling();
        setupEventListeners();

        // Initialize system prompt based on default agent
        if (currentAgent === 'researcher') {
            systemPromptInput.value = "You are the Researcher Agent. Expert at web searches and factual verification.";
        }
    }

    // --- Event Listeners ---
    function setupEventListeners() {
        // Agent selection
        agentCards.forEach(card => {
            card.addEventListener('click', () => {
                agentCards.forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                currentAgent = card.dataset.agent;

                // Update system prompt placeholder or value based on agent
                if (currentAgent === 'researcher') {
                    systemPromptInput.value = "You are the Researcher Agent. Expert at web searches and factual verification.";
                } else {
                    systemPromptInput.value = "You are the Analyst Agent. Professional Technical Analyst specialized in price patterns.";
                }

                // Close sidebar on mobile
                if (window.innerWidth <= 768) {
                    toggleSidebar(false);
                }
            });
        });

        // Dropdown toggle
        selectTrigger.addEventListener('click', (e) => {
            e.stopPropagation();
            modelSelect.classList.toggle('open');
            selectTrigger.classList.toggle('active');
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!modelSelect.contains(e.target)) {
                modelSelect.classList.remove('open');
                selectTrigger.classList.remove('active');
            }
        });

        startModelBtn.addEventListener('click', () => {
            if (selectedModelPath) window.__startModel(selectedModelPath);
        });

        stopModelBtn.addEventListener('click', () => {
            window.__stopModel();
        });

        sendBtn.addEventListener('click', sendMessage);
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        chatInput.addEventListener('input', autoResizeTextarea);
        clearChatBtn.addEventListener('click', clearChat);

        // Sidebar Toggle
        sidebarToggle.addEventListener('click', () => toggleSidebar());
        sidebarOverlay.addEventListener('click', () => toggleSidebar(false));

        ctxSizeSlider.addEventListener('input', () => {
            ctxSizeValue.textContent = ctxSizeSlider.value;
        });
        gpuLayersSlider.addEventListener('input', () => {
            gpuLayersValue.textContent = gpuLayersSlider.value;
        });

        // Vision Events
        if (attachBtn) {
            attachBtn.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', handleFileSelect);
            clearImageBtn.addEventListener('click', clearImage);

            chatInput.addEventListener('keydown', (e) => {
                if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
                    // Let default paste happen, handled by 'paste' event
                }
            });
            chatInput.addEventListener('paste', handlePaste);

            chatInputArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                chatInputArea.style.borderColor = 'var(--accent-start)';
            });
            chatInputArea.addEventListener('dragleave', (e) => {
                e.preventDefault();
                chatInputArea.style.borderColor = '';
            });
            chatInputArea.addEventListener('drop', handleDrop);
        }
    }

    function autoResizeTextarea() {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
    }

    // --- Models ---
    async function loadModels() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            models = data.models || [];
            renderModels();
        } catch (e) {
            selectTrigger.innerHTML = '<span class="placeholder" style="color:var(--error)">Failed to load models</span>';
        }
    }

    function renderModels() {
        if (models.length === 0) {
            selectOptions.innerHTML = '<div style="padding:10px; color:var(--text-muted)">No models found</div>';
            return;
        }

        // Generate dropdown options
        selectOptions.innerHTML = models.map(m => `
            <div class="select-option ${m.path === selectedModelPath ? 'selected' : ''}"
                 data-path="${m.path}">
                <div class="select-option-name">${escapeHtml(m.name)}</div>
                <div class="select-option-meta">
                    <span>${m.size_gb} GB</span>
                    ${m.quantization ? `<span>• ${escapeHtml(m.quantization)}</span>` : ''}
                </div>
            </div>
        `).join('');

        // Add click listeners to options
        selectOptions.querySelectorAll('.select-option').forEach(opt => {
            opt.addEventListener('click', (e) => {
                e.stopPropagation();
                const path = opt.dataset.path;
                selectModel(path);
                modelSelect.classList.remove('open');
                selectTrigger.classList.remove('active');

                // Close sidebar on mobile
                if (window.innerWidth <= 768) {
                    toggleSidebar(false);
                }
            });
        });

        // Update trigger text if model selected
        if (selectedModelPath) {
            const model = models.find(m => m.path === selectedModelPath);
            if (model) {
                selectTrigger.querySelector('.placeholder').textContent = model.name;
                selectTrigger.querySelector('.placeholder').style.color = 'var(--text-primary)';

                // Show actions bar
                modelActionsBar.style.display = 'flex';
                selectedModelInfo.innerHTML = `
                    <span class="selected-model-tag">${model.size_gb} GB</span>
                    ${model.quantization ? `<span class="selected-model-tag">${escapeHtml(model.quantization)}</span>` : ''}
                `;
            }
        } else {
            selectTrigger.querySelector('.placeholder').textContent = 'Select a model...';
            selectTrigger.querySelector('.placeholder').style.color = 'var(--text-muted)';
            modelActionsBar.style.display = 'none';
        }

        // Update button states
        if (serverState === 'running') {
            startModelBtn.style.display = 'none';
            stopModelBtn.style.display = 'flex';
        } else if (serverState === 'starting') {
            startModelBtn.style.display = 'flex';
            startModelBtn.disabled = true;
            startModelBtn.innerHTML = '<span class="btn-loading-text">Loading...</span>';
            stopModelBtn.style.display = 'none';
        } else {
            startModelBtn.style.display = 'flex';
            startModelBtn.disabled = false;
            startModelBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                Start Model
            `;
            stopModelBtn.style.display = 'none';
        }
    }

    function selectModel(path) {
        selectedModelPath = path;
        renderModels();

        // Check for vision support
        const model = models.find(m => m.path === path);
        if (model) {
            const name = model.name.toLowerCase();
            const isVision = name.includes('vision') ||
                name.includes('vl') ||
                name.includes('llava') ||
                name.includes('moondream') ||
                name.includes('bakllava') ||
                name.includes('yi-vl');

            attachBtn.disabled = !isVision;
            if (isVision) {
                attachBtn.removeAttribute('disabled');
                attachBtn.title = "Upload chart/image";
            } else {
                attachBtn.setAttribute('disabled', 'true');
                attachBtn.title = "Select a vision model to upload images";
            }
        }
    }

    window.__startModel = async (path) => {
        selectedModelPath = path;
        serverState = 'starting';
        renderModels();
        updateStatusUI('starting', 'Loading model...', '');

        try {
            const res = await fetch('/api/server/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_path: path,
                    ctx_size: parseInt(ctxSizeSlider.value),
                    n_gpu_layers: parseInt(gpuLayersSlider.value),
                }),
            });
            const data = await res.json();
            handleStatusUpdate(data);
        } catch (e) {
            updateStatusUI('error', 'Failed to start', e.message);
        }
    };

    window.__stopModel = async () => {
        try {
            const res = await fetch('/api/server/stop', { method: 'POST' });
            const data = await res.json();
            handleStatusUpdate(data);
        } catch (e) {
            console.error('Stop failed:', e);
        }
    };

    // --- Status Polling ---
    function startStatusPolling() {
        if (statusPollInterval) clearInterval(statusPollInterval);
        statusPollInterval = setInterval(pollStatus, 3000);
    }

    async function pollStatus() {
        try {
            const res = await fetch('/api/server/status');
            const data = await res.json();
            handleStatusUpdate(data);
        } catch (e) {
            // Server might be down
        }
    }

    function handleStatusUpdate(data) {
        const prevState = serverState;
        serverState = data.state;

        // Only sync selectedModelPath from server when transitioning to
        // running/starting, NOT on every poll. This prevents overwriting
        // the user's dropdown selection.
        if (data.model_path && (data.state === 'running' || data.state === 'starting')) {
            // If user hasn't manually selected a different model, sync from server
            if (!selectedModelPath || prevState === 'idle' || prevState === 'starting') {
                selectedModelPath = data.model_path;
            }
        }

        // When server goes idle, don't clear the user's dropdown selection
        // so they can easily restart the same model.

        let details = '';
        if (data.model_name && data.state !== 'idle') {
            details += data.model_name;
        }
        if (data.uptime_seconds && data.state === 'running') {
            details += ` • ${formatUptime(data.uptime_seconds)}`;
        }
        if (data.ctx_size && data.state === 'running') {
            details += ` • ctx:${data.ctx_size}`;
        }

        updateStatusUI(data.state, data.state.charAt(0).toUpperCase() + data.state.slice(1), details);
        renderModels();
        updateChatState();
    }

    function updateStatusUI(state, text, details) {
        statusCard.className = 'status-card ' + state;
        statusDot.className = 'status-dot ' + state;
        statusText.textContent = text;
        statusDetails.textContent = details || '';

        chatHeaderBadge.className = 'chat-header-badge ' + state;
        chatHeaderBadge.textContent = state;

        if (state === 'running') {
            const model = models.find(m => m.path === selectedModelPath);
            activeModelName.textContent = model ? model.name : 'Model Running';
        } else if (state === 'starting') {
            activeModelName.textContent = 'Loading model...';
        } else {
            activeModelName.textContent = 'No model loaded';
        }
    }

    function updateChatState() {
        const canChat = serverState === 'running';
        chatInput.disabled = !canChat;
        sendBtn.disabled = !canChat;
        inputHint.textContent = canChat
            ? 'Press Enter to send, Shift+Enter for new line'
            : 'Start a model to begin chatting';
    }

    function toggleSidebar(force) {
        const isOpen = sidebar.classList.toggle('open', force);
        sidebarOverlay.classList.toggle('open', isOpen);
    }

    // --- Chat ---
    function clearChat() {
        conversationHistory = [];
        chatMessages.innerHTML = '';
        chatMessages.appendChild(welcomeScreen);
        welcomeScreen.style.display = 'flex';
    }

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text || isStreaming || serverState !== 'running') return;

        // Hide welcome screen
        welcomeScreen.style.display = 'none';

        // Add user message
        let userMsg;
        if (currentImageBase64) {
            userMsg = {
                role: 'user',
                content: [
                    { type: 'text', text: text },
                    { type: 'image_url', image_url: { url: currentImageBase64 } }
                ]
            };

            const msgEl = appendMessage('user', text);
            const img = document.createElement('img');
            img.src = currentImageBase64;
            img.style.maxWidth = '200px';
            img.style.borderRadius = 'var(--radius-sm)';
            img.style.marginTop = '8px';
            img.style.display = 'block';
            msgEl.querySelector('.message-content').appendChild(img);
        } else {
            userMsg = { role: 'user', content: text };
            appendMessage('user', text);
        }

        conversationHistory.push(userMsg);
        clearImage();

        chatInput.value = '';
        chatInput.style.height = 'auto';
        isStreaming = true;
        sendBtn.disabled = true;

        // Create assistant message placeholder
        const assistantEl = appendMessage('assistant', '');
        const contentEl = assistantEl.querySelector('.message-content');
        showTypingIndicator(contentEl);

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: conversationHistory,
                    enable_tools: enableToolsCheckbox.checked,
                    system_prompt: systemPromptInput.value.trim(),
                    agent: currentAgent,
                }),
            });

            if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let fullContent = '';
            let buffer = '';
            let typingRemoved = false;

            const textEl = contentEl.querySelector('.message-text');

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const dataStr = line.slice(6).trim();
                    if (!dataStr || dataStr === '[DONE]') continue;

                    let event;
                    try {
                        event = JSON.parse(dataStr);
                    } catch {
                        continue;
                    }

                    if (!typingRemoved) {
                        removeTypingIndicator(contentEl);
                        typingRemoved = true;
                    }

                    switch (event.type) {
                        case 'chart_analysis': {
                            console.log('[CHART] Analysis event:', event);
                            const chartCard = document.createElement('div');
                            chartCard.className = 'chart-analysis-card';

                            let html = '';
                            if (event.patterns && event.patterns.length > 0) {
                                html += '<strong>📊 Patterns Detected</strong>';
                                for (const p of event.patterns) {
                                    const barWidth = Math.min(p.probability, 100);
                                    html += `<div class="chart-pattern-row">
                                        <span class="pattern-label">${escapeHtml(p.label)}</span>
                                        <div class="pattern-bar-bg"><div class="pattern-bar" style="width:${barWidth}%"></div></div>
                                        <span class="pattern-pct">${p.probability}%</span>
                                    </div>`;
                                }
                            } else {
                                html += `<strong>📊 Analyzing chart...</strong>`;
                            }

                            // Render annotated image at the bottom
                            if (event.annotated_image) {
                                html += `<div class="chart-annotated-wrapper" style="margin-top: 12px;">
                                    <img src="data:image/jpeg;base64,${event.annotated_image}" class="chart-annotated-img" style="max-width: 800px; width: 100%;" alt="Annotated Chart" title="Click to expand" onclick="window.open(this.src, '_blank')">
                                </div>`;
                            }

                            chartCard.innerHTML = html;
                            contentEl.appendChild(chartCard);
                            scrollToBottom();
                            break;
                        }

                        case 'content':
                            fullContent += event.content;
                            textEl.innerHTML = renderMarkdown(fullContent);
                            scrollToBottom();
                            break;

                        case 'tool_start':
                            const toolCard = createToolCard(event.tool, event.args, 'running');
                            contentEl.appendChild(toolCard);
                            scrollToBottom();
                            break;

                        case 'tool_result':
                            updateToolCard(contentEl, event.tool, event.result);
                            scrollToBottom();
                            break;

                        case 'tool_update':
                            updateToolStatus(contentEl, event.tool, event.status, event.model, event.t_s);
                            break;

                        case 'error':
                            contentEl.innerHTML += `<div style="color: var(--error); margin-top: 8px;">⚠ ${escapeHtml(event.content)}</div>`;
                            break;

                        case 'done':
                            break;
                    }
                }
            }

            if (fullContent) {
                conversationHistory.push({ role: 'assistant', content: fullContent });
            }

        } catch (e) {
            if (!contentEl.querySelector('.typing-indicator')) {
                contentEl.innerHTML = `<div style="color: var(--error);">⚠ Error: ${escapeHtml(e.message)}</div>`;
            } else {
                removeTypingIndicator(contentEl);
                contentEl.innerHTML = `<div style="color: var(--error);">⚠ Error: ${escapeHtml(e.message)}</div>`;
            }
        } finally {
            isStreaming = false;
            sendBtn.disabled = serverState !== 'running';
        }
    }

    // --- Message Rendering ---
    function appendMessage(role, content) {
        const el = document.createElement('div');
        el.className = `message ${role}`;
        el.innerHTML = `
            <div class="message-avatar">${role === 'user' ? 'U' : 'AI'}</div>
            <div class="message-content">
                <div class="message-text">${content ? renderMarkdown(content) : ''}</div>
            </div>
        `;
        chatMessages.appendChild(el);
        scrollToBottom();
        return el;
    }

    function showTypingIndicator(container) {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        container.appendChild(indicator);
    }

    function removeTypingIndicator(container) {
        const indicator = container.querySelector('.typing-indicator');
        if (indicator) indicator.remove();
    }

    function createToolCard(toolName, args, status) {
        const card = document.createElement('div');
        card.className = 'tool-call-card';
        card.dataset.tool = toolName;

        const argsStr = args ? JSON.stringify(args, null, 2) : '';
        card.innerHTML = `
            <div class="tool-call-header" onclick="this.parentElement.classList.toggle('expanded')">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/>
                    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
                </svg>
                <span class="tool-name">🔧 ${escapeHtml(toolName)}</span>
                <span class="tool-toggle">▼ details</span>
            </div>
            <div class="tool-call-body">${escapeHtml(argsStr)}</div>
            <div class="tool-call-status ${status}">
                <span class="status-label">${status === 'running' ? 'Executing...' : 'Done'}</span>
                <span class="tool-perf-info"></span>
            </div>
        `;
        return card;
    }

    function updateToolCard(container, toolName, result) {
        const cards = container.querySelectorAll('.tool-call-card');
        for (const card of cards) {
            if (card.dataset.tool === toolName) {
                const body = card.querySelector('.tool-call-body');
                body.textContent += '\n\n--- Result ---\n' + result;
                const status = card.querySelector('.tool-call-status');
                status.className = 'tool-call-status';
                status.querySelector('.status-label').textContent = '✓ Complete';
                break;
            }
        }
    }

    function updateToolStatus(container, toolName, statusMsg, modelName, t_s) {
        const cards = container.querySelectorAll('.tool-call-card');
        for (const card of cards) {
            if (card.dataset.tool === toolName) {
                if (statusMsg) {
                    card.querySelector('.status-label').textContent = statusMsg;
                }
                const perfEl = card.querySelector('.tool-perf-info');
                if (modelName) {
                    let text = `[${modelName}]`;
                    if (t_s) text += ` ${t_s.toFixed(1)} t/s`;
                    perfEl.textContent = text;
                }
                break;
            }
        }
    }

    // --- Markdown Rendering (lightweight) ---
    function renderMarkdown(text) {
        // Handle literal \n strings if they appear (some models escape newlines incorrectly)
        let processed = text.replace(/\\n/g, '\n');

        // Hide tool call tags (both complete and partial) from the live stream view
        processed = processed.replace(/<tool_call>[\s\S]*?(<\/tool_call>|$)/gi, '');
        processed = processed.replace(/<TOOLCALL>[\s\S]*?(<\/TOOLCALL>|$)/gi, '');

        let html = escapeHtml(processed);

        // Code blocks
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            return `<pre><code>${code.trim()}</code></pre>`;
        });

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Italic
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Headers
        html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>');
        html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^# (.+)$/gm, '<h2>$1</h2>');

        // Links
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" style="color: var(--text-accent)">$1</a>');

        // Lists
        html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

        // Paragraphs
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');
        html = '<p>' + html + '</p>';
        html = html.replace(/<p><\/p>/g, '');
        html = html.replace(/<p>(<h[234]>)/g, '$1');
        html = html.replace(/(<\/h[234]>)<\/p>/g, '$1');
        html = html.replace(/<p>(<pre>)/g, '$1');
        html = html.replace(/(<\/pre>)<\/p>/g, '$1');
        html = html.replace(/<p>(<ul>)/g, '$1');
        html = html.replace(/(<\/ul>)<\/p>/g, '$1');

        return html;
    }

    // --- Utilities ---
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function formatUptime(seconds) {
        if (seconds < 60) return `${Math.floor(seconds)}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        return `${h}h ${m}m`;
    }

    function scrollToBottom() {
        requestAnimationFrame(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        });
    }

    // --- Vision Handlers ---
    function handleFileCheck(file) {
        if (!file.type.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            currentImageBase64 = e.target.result;
            previewImg.src = currentImageBase64;
            imagePreview.style.display = 'flex';
            attachBtn.style.color = 'var(--accent-start)';
        };
        reader.readAsDataURL(file);
    }

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) handleFileCheck(file);
    }

    function handlePaste(e) {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (const item of items) {
            if (item.type.indexOf('image') === 0) {
                const file = item.getAsFile();
                handleFileCheck(file);
                e.preventDefault();
            }
        }
    }

    function handleDrop(e) {
        e.preventDefault();
        chatInputArea.style.borderColor = '';
        const file = e.dataTransfer.files[0];
        if (file) handleFileCheck(file);
    }

    function clearImage() {
        currentImageBase64 = null;
        fileInput.value = '';
        imagePreview.style.display = 'none';
        previewImg.src = '';
        attachBtn.style.color = '';
    }

    // --- Boot ---
    init();
})();
