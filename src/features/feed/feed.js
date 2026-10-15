import { DbService } from '../../services/db.service.js';
import { StorageService } from '../../services/storage.service.js';
import { RealtimeService } from '../../services/realtime.service.js';
import { state } from '../../config/app-config.js';
import { $, $$, esc, escAttr, toast, avatar, openModal, closeModal } from '../../lib/dom.js';
import { timeAgo } from '../../lib/time.js';

let _feedUnsub = null;
let _cachedPosts = [];

export const FeedFeature = {
  async mount() {
    const container = $('#page-feed');
    if (!container) return;

    await this.loadPosts();

    // Subscribe to realtime post creations & reactions
    _feedUnsub = RealtimeService.subscribeToCollection('posts', response => {
      const event = response.events[0] || '';
      if (event.includes('.create')) {
        _cachedPosts.unshift(response.payload);
        this.renderPosts(_cachedPosts);
      } else if (event.includes('.delete')) {
        _cachedPosts = _cachedPosts.filter(p => p.$id !== response.payload.$id);
        this.renderPosts(_cachedPosts);
      }
    }, 'feed');

    this.bindEvents();
  },

  unmount() {
    if (_feedUnsub) {
      _feedUnsub();
      _feedUnsub = null;
    }
    RealtimeService.unsubscribeTag('feed');
  },

  async loadPosts() {
    const listEl = $('#feed-posts');
    if (!listEl) return;
    listEl.innerHTML = '<div class="spinner-center"><div class="spinner"></div></div>';

    try {
      _cachedPosts = await DbService.getPosts({ limit: 30 });
      this.renderPosts(_cachedPosts);
    } catch (e) {
      listEl.innerHTML = `<div class="empty-state"><p>Could not load posts</p></div>`;
    }
  },

  renderPosts(posts) {
    const listEl = $('#feed-posts');
    if (!listEl) return;

    if (!posts || posts.length === 0) {
      listEl.innerHTML = `
        <div class="empty-state">
          <p>No campus moments yet. Be the first to share!</p>
        </div>
      `;
      return;
    }

    listEl.innerHTML = posts.map(post => this.renderPostCard(post)).join('');
  },

  renderPostCard(post) {
    const isOwner = state.user && (state.user.$id === post.authorId);
    const mediaHtml = post.mediaURL
      ? `<div class="post-media-wrap"><img src="${escAttr(post.mediaURL)}" alt="Post media" class="post-media" loading="lazy"></div>`
      : '';

    return `
      <article class="post-card" data-post-id="${escAttr(post.$id)}">
        <header class="post-header">
          <div class="post-author-row">
            ${avatar(post.authorName, post.authorPhoto, 'avatar-md')}
            <div class="post-author-meta">
              <strong>${esc(post.authorName || 'Student')}</strong>
              <span class="post-timestamp">${esc(timeAgo(post.createdAt))}</span>
            </div>
          </div>
          ${isOwner ? `<button class="icon-btn post-action-btn" data-action="delete-post" data-id="${escAttr(post.$id)}">&times;</button>` : ''}
        </header>

        <div class="post-content">
          <p>${esc(post.content).replace(/\n/g, '<br>')}</p>
        </div>

        ${mediaHtml}

        <footer class="post-footer">
          <button class="post-stat-btn" data-action="like-post" data-id="${escAttr(post.$id)}">
            ❤️ <span>${post.likesCount || 0}</span>
          </button>
          <button class="post-stat-btn" data-action="comment-post" data-id="${escAttr(post.$id)}">
            💬 <span>${post.commentsCount || 0}</span>
          </button>
        </footer>
      </article>
    `;
  },

  bindEvents() {
    const container = $('#page-feed');
    if (!container || container._feedEventsBound) return;
    container._feedEventsBound = true;

    container.addEventListener('click', async e => {
      const btn = e.target.closest('[data-action]');
      if (!btn) return;

      const action = btn.dataset.action;
      const postId = btn.dataset.id;

      if (action === 'delete-post') {
        if (!confirm('Delete this post?')) return;
        try {
          await DbService.deletePost(postId);
          toast('Post deleted');
        } catch (err) {
          toast('Failed to delete post');
        }
      }
    });
  }
};
