import { DbService } from '../../services/db.service.js';
import { $, esc, escAttr, openModal, closeModal } from '../../lib/dom.js';
import { timeAgo } from '../../lib/time.js';

let _cachedListings = [];

export const MarketplaceFeature = {
  async mount() {
    await this.loadListings();
    this.bindEvents();
  },

  unmount() {
    _cachedListings = [];
  },

  async loadListings() {
    const grid = $('#market-grid');
    if (!grid) return;
    grid.innerHTML = '<div class="spinner-center"><div class="spinner"></div></div>';

    try {
      _cachedListings = await DbService.getListings(40);
      this.renderListings(_cachedListings);
    } catch (e) {
      grid.innerHTML = '<div class="empty-state"><p>Could not load listings</p></div>';
    }
  },

  renderListings(items) {
    const grid = $('#market-grid');
    if (!grid) return;

    if (!items || items.length === 0) {
      grid.innerHTML = '<div class="empty-state"><p>No items for sale right now. Sell your textbooks or gear!</p></div>';
      return;
    }

    grid.innerHTML = items.map(item => `
      <div class="market-card" data-action="view-item" data-item-id="${escAttr(item.$id)}">
        <div class="market-img-wrap">
          <img src="${escAttr(item.imageUrl || 'unino-icon.png')}" alt="${escAttr(item.title)}" loading="lazy">
          <span class="market-price-tag">R${esc(item.price)}</span>
        </div>
        <div class="market-info">
          <strong class="market-title">${esc(item.title)}</strong>
          <span class="market-campus">${esc(item.campus || 'Campus')} · ${esc(timeAgo(item.createdAt))}</span>
        </div>
      </div>
    `).join('');
  },

  bindEvents() {
    const grid = $('#market-grid');
    if (!grid || grid._bound) return;
    grid._bound = true;

    grid.addEventListener('click', e => {
      const card = e.target.closest('[data-action="view-item"]');
      if (card) {
        const itemId = card.dataset.itemId;
        const item = _cachedListings.find(i => i.$id === itemId);
        if (item) this.showItemModal(item);
      }
    });
  },

  showItemModal(item) {
    openModal(`
      <div class="modal-header">
        <h2>${esc(item.title)}</h2>
        <button class="icon-btn" onclick="closeModal()">&times;</button>
      </div>
      <div class="modal-body" style="padding:16px">
        <img src="${escAttr(item.imageUrl || 'unino-icon.png')}" style="width:100%;max-height:260px;object-fit:cover;border-radius:12px;margin-bottom:12px">
        <div style="font-size:20px;font-weight:800;color:var(--accent);margin-bottom:8px">R${esc(item.price)}</div>
        <p style="font-size:14px;color:var(--text-secondary);line-height:1.5;margin-bottom:16px">${esc(item.description || 'No description provided.')}</p>
        <button class="btn-primary btn-full" onclick="alert('Contacting seller...')">Message Seller</button>
      </div>
    `);
  }
};
