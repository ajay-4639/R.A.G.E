// RAG OS Demo - Frontend JavaScript

const API = {
    upload: '/api/upload',
    documents: '/api/documents',
    query: '/api/query',
    stats: '/api/stats',
    health: '/api/health',
};

// DOM elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const docList = document.getElementById('docList');
const messages = document.getElementById('messages');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const refreshDocs = document.getElementById('refreshDocs');
const statDocs = document.getElementById('statDocs');
const statChunks = document.getElementById('statChunks');

let isQuerying = false;

// ---- File Upload ----

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) uploadFile(files[0]);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        uploadFile(fileInput.files[0]);
        fileInput.value = '';
    }
});

async function uploadFile(file) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!['.txt', '.md', '.pdf'].includes(ext)) {
        showUploadStatus(`Unsupported file type: ${ext}`, 'error');
        return;
    }

    showUploadStatus(`Uploading ${file.name}...`, 'loading');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(API.upload, { method: 'POST', body: formData });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Upload failed');
        }
        const data = await res.json();
        showUploadStatus(`${file.name}: ${data.chunk_count} chunks indexed`, 'success');
        loadDocuments();
        loadStats();
    } catch (e) {
        showUploadStatus(e.message, 'error');
    }
}

function showUploadStatus(msg, type) {
    uploadStatus.textContent = msg;
    uploadStatus.className = `upload-status ${type}`;
    uploadStatus.hidden = false;

    if (type !== 'loading') {
        setTimeout(() => { uploadStatus.hidden = true; }, 4000);
    }
}

// ---- Documents ----

async function loadDocuments() {
    try {
        const res = await fetch(API.documents);
        const docs = await res.json();

        if (docs.length === 0) {
            docList.innerHTML = '<p class="empty-state">No documents yet</p>';
            return;
        }

        docList.innerHTML = docs.map(doc => `
            <div class="doc-item" data-id="${doc.doc_id}">
                <div class="doc-info">
                    <div class="doc-name" title="${doc.filename}">${doc.filename}</div>
                    <div class="doc-meta">${doc.chunk_count} chunks</div>
                </div>
                <span class="doc-type">${doc.file_type}</span>
                <button class="delete-btn" onclick="deleteDocument('${doc.doc_id}')" title="Delete">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"/>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                </button>
            </div>
        `).join('');
    } catch (e) {
        docList.innerHTML = '<p class="empty-state">Failed to load documents</p>';
    }
}

async function deleteDocument(docId) {
    try {
        const res = await fetch(`${API.documents}/${docId}`, { method: 'DELETE' });
        if (res.ok) {
            loadDocuments();
            loadStats();
        }
    } catch (e) {
        console.error('Delete failed:', e);
    }
}

// ---- Stats ----

async function loadStats() {
    try {
        const res = await fetch(API.stats);
        const data = await res.json();
        statDocs.textContent = data.total_documents || 0;
        statChunks.textContent = data.total_chunks || 0;
    } catch (e) {
        // Silently fail
    }
}

// ---- Chat ----

queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
    }
});

sendBtn.addEventListener('click', sendQuery);

async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query || isQuerying) return;

    isQuerying = true;
    sendBtn.disabled = true;
    queryInput.value = '';

    // Remove welcome message if present
    const welcome = messages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    // Add user message
    addMessage(query, 'user');

    // Add loading indicator
    const loadingId = addLoadingMessage();

    try {
        const res = await fetch(API.query, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });

        removeMessage(loadingId);

        if (!res.ok) {
            const err = await res.json();
            addMessage(`Error: ${err.detail || 'Query failed'}`, 'assistant');
            return;
        }

        const data = await res.json();
        addAssistantMessage(data);

    } catch (e) {
        removeMessage(loadingId);
        addMessage(`Error: ${e.message}`, 'assistant');
    } finally {
        isQuerying = false;
        sendBtn.disabled = false;
        queryInput.focus();
    }
}

function addMessage(text, role) {
    const div = document.createElement('div');
    div.className = `message message-${role}`;
    div.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    messages.appendChild(div);
    scrollToBottom();
    return div;
}

function addAssistantMessage(data) {
    const div = document.createElement('div');
    div.className = 'message message-assistant';

    // Format the answer text (basic markdown-like formatting)
    const formattedAnswer = formatAnswer(data.answer);

    let sourcesHtml = '';
    if (data.sources && data.sources.length > 0) {
        const sourceItems = data.sources.map(s => `
            <div class="source-item">
                <div class="source-header">
                    <span class="source-label">[${s.index}] ${escapeHtml(s.filename)}</span>
                    <span class="source-score">Score: ${s.score}</span>
                </div>
                <div class="source-content">${escapeHtml(s.content)}</div>
            </div>
        `).join('');

        sourcesHtml = `
            <button class="sources-toggle" onclick="toggleSources(this)">
                <span class="arrow">&#9654;</span>
                ${data.sources.length} source${data.sources.length > 1 ? 's' : ''}
            </button>
            <div class="sources-list">${sourceItems}</div>
        `;
    }

    const metaHtml = `
        <div class="message-meta">
            <span>${data.duration_ms}ms</span>
            ${data.tokens ? `<span>${data.tokens.total} tokens</span>` : ''}
        </div>
    `;

    div.innerHTML = `
        <div class="message-content">
            ${formattedAnswer}
            ${sourcesHtml}
            ${metaHtml}
        </div>
    `;

    messages.appendChild(div);
    scrollToBottom();
}

function addLoadingMessage() {
    const id = 'loading-' + Date.now();
    const div = document.createElement('div');
    div.className = 'message message-assistant message-loading';
    div.id = id;
    div.innerHTML = `
        <div class="message-content">
            <div class="dot-loader">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    messages.appendChild(div);
    scrollToBottom();
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function toggleSources(btn) {
    btn.classList.toggle('open');
    const list = btn.nextElementSibling;
    list.classList.toggle('visible');
}

function formatAnswer(text) {
    // Basic formatting: paragraphs and code blocks
    return text
        .split('\n\n')
        .map(p => `<p>${escapeHtml(p).replace(/\n/g, '<br>')}</p>`)
        .join('');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    messages.scrollTop = messages.scrollHeight;
}

// ---- Refresh ----

refreshDocs.addEventListener('click', () => {
    loadDocuments();
    loadStats();
});

// ---- Init ----

loadDocuments();
loadStats();
