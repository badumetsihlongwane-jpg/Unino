import { COLORS } from '../config/app-config.js';

export const $ = (s, parent = document) => parent.querySelector(s);
export const $$ = (s, parent = document) => Array.from(parent.querySelectorAll(s));

export function esc(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

export function escAttr(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

export function escJs(s) {
  if (s == null) return '';
  return String(s)
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/"/g, '\\"')
    .replace(/\r/g, '\\r')
    .replace(/\n/g, '\\n')
    .replace(/</g, '\\x3c')
    .replace(/>/g, '\\x3e');
}

export function toast(msg, duration = 3000) {
  let el = $('#toast');
  if (!el) {
    el = document.createElement('div');
    el.id = 'toast';
    el.className = 'toast';
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), duration);
}

export function openModal(html) {
  const bg = $('#modal-bg');
  const inner = $('#modal-inner');
  if (!bg || !inner) return;
  inner.innerHTML = html;
  bg.style.display = 'flex';
  document.body.style.overflow = 'hidden';
}

export function closeModal() {
  const bg = $('#modal-bg');
  if (!bg) return;
  bg.style.display = 'none';
  const inner = $('#modal-inner');
  if (inner) inner.innerHTML = '';
  document.body.style.overflow = '';
}

export function avatar(name = 'User', photo = null, cls = 'avatar-md') {
  if (photo) {
    return `<img src="${escAttr(photo)}" alt="${escAttr(name)}" class="${cls}" style="object-fit:cover;border-radius:50%" crossorigin="anonymous" referrerpolicy="no-referrer">`;
  }
  const initials = (name || 'U')
    .split(' ')
    .map(p => p[0])
    .filter(Boolean)
    .slice(0, 2)
    .join('')
    .toUpperCase() || 'U';
  const color = COLORS[Math.abs(hashString(name || 'U')) % COLORS.length];
  return `<div class="${cls}" style="background:${color};color:#fff;display:inline-flex;align-items:center;justify-content:center;font-weight:700;border-radius:50%">${initials}</div>`;
}

export function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return hash;
}

export function delegate(el, eventType, selector, handler) {
  el.addEventListener(eventType, e => {
    const target = e.target.closest(selector);
    if (target && el.contains(target)) {
      handler(e, target);
    }
  });
}
