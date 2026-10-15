import { DbService } from '../../services/db.service.js';
import { RealtimeService } from '../../services/realtime.service.js';
import { state } from '../../config/app-config.js';
import { $, $$, esc, escAttr, toast, avatar } from '../../lib/dom.js';
import { formatTime, timeAgo } from '../../lib/time.js';

let _activeConvoId = null;
let _chatMsgUnsub = null;
let _cachedMessages = [];

export const ChatFeature = {
  async mount() {
    await this.loadConversations();
    this.bindEvents();
  },

  unmount() {
    this.closeActiveChat();
    RealtimeService.unsubscribeTag('chat');
  },

  async loadConversations() {
    const listEl = $('#convo-list');
    if (!listEl) return;
    if (!state.user) return;

    listEl.innerHTML = '<div class="spinner-center"><div class="spinner"></div></div>';
    try {
      const convos = await DbService.getConversations(state.user.$id);
      this.renderConversations(convos);
    } catch (e) {
      listEl.innerHTML = `<div class="empty-state"><p>No conversations found</p></div>`;
    }
  },

  renderConversations(convos) {
    const listEl = $('#convo-list');
    if (!listEl) return;

    if (!convos || convos.length === 0) {
      listEl.innerHTML = `<div class="empty-state"><p>No chats yet. Connect with classmates in Explore!</p></div>`;
      return;
    }

    listEl.innerHTML = convos.map(c => `
      <div class="convo-item" data-action="open-chat" data-convo-id="${escAttr(c.$id)}">
        ${avatar(c.name || 'Chat', c.photo, 'avatar-md')}
        <div class="convo-meta">
          <div class="convo-top">
            <strong class="convo-name">${esc(c.name || 'Chat')}</strong>
            <span class="convo-time">${esc(timeAgo(c.updatedAt))}</span>
          </div>
          <p class="convo-preview">${esc(c.lastMessage || 'Start the conversation')}</p>
        </div>
      </div>
    `).join('');
  },

  async openChat(convoId, displayName = 'Chat') {
    _activeConvoId = convoId;
    const chatScreen = $('#chat-view');
    const headerTitle = $('#chat-header-name');
    const msgsContainer = $('#chat-messages');

    if (headerTitle) headerTitle.textContent = displayName;
    if (chatScreen) chatScreen.classList.add('active');

    if (msgsContainer) {
      msgsContainer.innerHTML = '<div class="spinner-center"><div class="spinner"></div></div>';
    }

    try {
      _cachedMessages = await DbService.getMessages(convoId);
      this.renderMessages(_cachedMessages);

      // Realtime subscription for incoming messages
      if (_chatMsgUnsub) _chatMsgUnsub();
      _chatMsgUnsub = RealtimeService.subscribeToCollection('messages', response => {
        const payload = response.payload;
        if (payload.convoId === convoId && response.events[0]?.includes('.create')) {
          _cachedMessages.push(payload);
          this.renderMessages(_cachedMessages);
        }
      }, 'chat');
    } catch (e) {
      if (msgsContainer) msgsContainer.innerHTML = '<p class="error-text">Failed to load messages</p>';
    }
  },

  closeActiveChat() {
    if (_chatMsgUnsub) {
      _chatMsgUnsub();
      _chatMsgUnsub = null;
    }
    _activeConvoId = null;
    const chatScreen = $('#chat-view');
    if (chatScreen) chatScreen.classList.remove('active');
  },

  renderMessages(messages) {
    const container = $('#chat-messages');
    if (!container) return;

    container.innerHTML = messages.map(m => {
      const isMine = state.user && (m.senderId === state.user.$id);
      return `
        <div class="msg-bubble ${isMine ? 'msg-outgoing' : 'msg-incoming'}">
          <p class="msg-text">${esc(m.text)}</p>
          <span class="msg-time">${esc(formatTime(m.createdAt))}</span>
        </div>
      `;
    }).join('');

    container.scrollTop = container.scrollHeight;
  },

  async handleSend() {
    const input = $('#chat-input');
    if (!input || !input.value.trim() || !_activeConvoId) return;

    const text = input.value.trim();
    input.value = '';

    try {
      await DbService.sendMessage(_activeConvoId, { text });
    } catch (e) {
      toast('Failed to send message');
    }
  },

  bindEvents() {
    const listEl = $('#convo-list');
    if (listEl && !listEl._bound) {
      listEl._bound = true;
      listEl.addEventListener('click', e => {
        const item = e.target.closest('[data-action="open-chat"]');
        if (item) {
          const convoId = item.dataset.convoId;
          const name = item.querySelector('.convo-name')?.textContent || 'Chat';
          this.openChat(convoId, name);
        }
      });
    }

    const sendBtn = $('#chat-send-btn');
    if (sendBtn && !sendBtn._bound) {
      sendBtn._bound = true;
      sendBtn.addEventListener('click', () => this.handleSend());
    }

    const backBtn = $('#chat-back-btn');
    if (backBtn && !backBtn._bound) {
      backBtn._bound = true;
      backBtn.addEventListener('click', () => this.closeActiveChat());
    }
  }
};
