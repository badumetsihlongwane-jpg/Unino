/* ══════════════════════════════════════════════════════
 *  UNIBO — Campus Social Engine v5
 *  Firebase Auth + Firestore | base64 images
 *  Feed (Discover tabs), Explore (Radar/List + Modules),
 *  Marketplace, Messaging (fixed), Profiles (fixed)
 * ══════════════════════════════════════════════════════ */

// ─── State ───────────────────────────────────────
const state = { user: null, profile: null, page: 'feed', status: 'online', manualStatus: 'online', unsubs: [], lastMsgTab: 'dm' };

// ─── Shortcuts ───────────────────────────────────
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
const FieldVal = firebase.firestore.FieldValue;
const COLORS = ['#6C5CE7','#8B5CF6','#A855F7','#7C3AED','#6366F1','#818CF8','#C084FC','#D946EF','#E879F9','#A78BFA'];

// ─── Admin / Official Account ────────────────────
const ADMIN_EMAIL = 'admin@mynwu.ac.za';
const VERIFIED_UIDS = new Set(); // populated on login
let _isAdmin = false;
let verifiedUsersUnsub = null;
let _groupAlertUnsub = null;
let _asgPendingAlerts = [];
let _dmUnreadCount = 0;
let _dmReplyTo = null;
let _gReplyTo = null;
let _dmMsgLookup = new Map();
let _gMsgLookup = new Map();
let _chatViewportCleanup = null;
let _gchatViewportCleanup = null;
let _inAppBackInit = false;
let _inAppBackListenerBound = false;
let _exploreSearchQuery = '';
let _pendingCommentImageFile = null;
let _pendingReelCommentImageFile = null;
let _reelCommentReplyTo = null;
let _sendingReelComment = false;
let _lastFeedCommentSubmit = { key: '', at: 0 };
let _lastReelCommentSubmit = { key: '', at: 0 };
const _authorPhotoCache = {};
function isVerifiedUser(uid) { return VERIFIED_UIDS.has(uid) || uid === state.profile?.id && _isAdmin; }
function verifiedBadge(uid) { return isVerifiedUser(uid) ? '<span class="verified-badge" title="Official">✔</span>' : ''; }

function clampText(v = '', max = 80) {
  const t = (v || '').replace(/\s+/g, ' ').trim();
  return t.length > max ? `${t.slice(0, max - 1)}…` : t;
}

function resolvePostAuthorPhoto(post = {}) {
  if (post.authorPhoto) return post.authorPhoto;
  if (!post.authorId) return null;
  if (post.authorId === state.user?.uid) return state.profile?.photoURL || null;
  if (Object.prototype.hasOwnProperty.call(_authorPhotoCache, post.authorId)) return _authorPhotoCache[post.authorId];
  _authorPhotoCache[post.authorId] = null;
  db.collection('users').doc(post.authorId).get().then(doc => {
    _authorPhotoCache[post.authorId] = doc.exists ? (doc.data().photoURL || null) : null;
    updateFeedAuthorAvatars(post.authorId);
  }).catch(() => {});
  return null;
}

function updateFeedAuthorAvatars(authorId) {
  if (!authorId) return;
  const photo = _authorPhotoCache[authorId] || null;
  document.querySelectorAll(`.feed-author-avatar[data-author-id="${authorId}"]`).forEach(el => {
    const name = el.getAttribute('data-author-name') || 'User';
    el.innerHTML = avatar(name, photo, 'avatar-md');
  });
}

function jumpToMessage(messageId, containerId) {
  if (!messageId) return;
  const container = document.getElementById(containerId);
  const row = document.getElementById(`msg-${messageId}`);
  if (!container || !row) return;
  row.scrollIntoView({ behavior: 'smooth', block: 'center' });
  row.classList.add('msg-jump-highlight');
  setTimeout(() => row.classList.remove('msg-jump-highlight'), 1200);
}

function patchPostEngagement(post) {
  if (!post?.id) return;
  const postEl = document.getElementById(`post-${post.id}`);
  if (!postEl) return;
  const liked = (post.likes || []).includes(state.user.uid);
  const likeCount = (post.likes || []).length;
  const commentCount = post.commentsCount || 0;

  const likeBtn = postEl.querySelector(`.post-action[onclick*="toggleLike('${post.id}')"]`);
  if (likeBtn) {
    likeBtn.classList.toggle('liked', liked);
    likeBtn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="${liked ? 'var(--red)' : 'none'}" stroke="${liked ? 'var(--red)' : 'currentColor'}" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
      ${likeCount || 'Like'}
    `;
  }

  const stats = postEl.querySelector('.post-stats');
  if (stats) {
    stats.innerHTML = `
      ${likeCount ? `<span class="stat-item"><svg width="14" height="14" viewBox="0 0 24 24" fill="var(--red)" stroke="none"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg> ${likeCount}</span>` : ''}
      ${commentCount ? `<span class="stat-item">${commentCount} comment${commentCount > 1 ? 's' : ''}</span>` : ''}
    `;
  }
}

function scrollToLatest(msgsEl) {
  if (!msgsEl) return;
  requestAnimationFrame(() => { msgsEl.scrollTop = msgsEl.scrollHeight; });
}

function setupViewportFollow(msgsEl) {
  if (!msgsEl || !window.visualViewport) return () => {};
  const onVv = () => setTimeout(() => scrollToLatest(msgsEl), 20);
  window.visualViewport.addEventListener('resize', onVv);
  window.visualViewport.addEventListener('scroll', onVv);
  return () => {
    window.visualViewport.removeEventListener('resize', onVv);
    window.visualViewport.removeEventListener('scroll', onVv);
  };
}

// ─── Helpers ─────────────────────────────────────
function colorFor(n) {
  let h = 0;
  for (let i = 0; i < (n || '').length; i++) h = n.charCodeAt(i) + ((h << 5) - h);
  return COLORS[Math.abs(h) % COLORS.length];
}

function normalizeModules(modules = []) {
  return (modules || []).map(m => (m || '').trim().toUpperCase()).filter(Boolean);
}

function normalizeAddress(address = '') {
  return (address || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\b(res|residence|residency|res\.|street|st|road|rd|avenue|ave|block|unit|room|house|flat|hostel)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function addressTokens(address = '') {
  return normalizeAddress(address).split(' ').filter(token => token.length > 2);
}

function addressMatchScore(a = '', b = '') {
  const left = addressTokens(a);
  const right = addressTokens(b);
  if (!left.length || !right.length) return 0;
  const rightSet = new Set(right);
  return left.filter(token => rightSet.has(token)).length;
}

function sanitizeFriendRequests(requests = []) {
  const seen = new Set();
  return (requests || []).filter(req => {
    const uid = req && req.uid;
    if (!uid || seen.has(uid)) return false;
    seen.add(uid);
    return true;
  });
}

function anonNicknameKey(viewerUid, otherUid) {
  return `${viewerUid}_${otherUid}`;
}

function defaultAnonLabel(convoId = '') {
  const suffix = (convoId || '').slice(-4).toUpperCase() || 'CHAT';
  return `Anonymous #${suffix}`;
}

function getAnonDisplayName(convo = {}, viewerUid, otherUid) {
  const custom = (convo.anonNicknames || {})[anonNicknameKey(viewerUid, otherUid)] || '';
  return custom || defaultAnonLabel(convo.id || convo._id || otherUid || '');
}

let _commentAnonChoice = null;

function initials(n) {
  if (!n) return '?';
  const p = n.trim().split(/\s+/);
  return (p[0][0] + (p[1] ? p[1][0] : '')).toUpperCase();
}

function avatar(name, photo, cls = 'avatar-sm') {
  const bg = colorFor(name);
  if (photo) return `<div class="${cls}" style="background:${bg}"><img src="${photo}" alt="" onerror="this.remove();this.parentElement.textContent='${initials(name)}'"></div>`;
  return `<div class="${cls}" style="background:${bg}">${initials(name)}</div>`;
}

function timeAgo(ts) {
  if (!ts) return '';
  const d = ts.toDate ? ts.toDate() : new Date(ts);
  const m = Math.floor((Date.now() - d) / 60000);
  if (m < 1) return 'Just now';
  if (m < 60) return m + 'm';
  const h = Math.floor(m / 60);
  if (h < 24) return h + 'h';
  const days = Math.floor(h / 24);
  if (days < 7) return days + 'd';
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function chatTime(ts) {
  if (!ts) return '';
  const d = ts.toDate ? ts.toDate() : new Date(ts);
  return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
}

function dateSeparatorLabel(ts) {
  if (!ts) return null;
  const d = ts.toDate ? ts.toDate() : new Date(ts);
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const msgDay = new Date(d.getFullYear(), d.getMonth(), d.getDate());
  const diff = Math.floor((today - msgDay) / 86400000);
  if (diff === 0) return 'Today';
  if (diff === 1) return 'Yesterday';
  return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
}

function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

function isStudentEmail(email = '') {
  const e = (email || '').trim().toLowerCase();
  if (e === ADMIN_EMAIL.toLowerCase()) return true;
  return /@mynwu\.ac\.za$/i.test(e);
}

function scoreSeed(str = '') {
  let hash = 0;
  for (let i = 0; i < str.length; i++) hash = ((hash << 5) - hash) + str.charCodeAt(i);
  return Math.abs(hash % 1000) / 1000;
}

function extractHashTags(text = '') {
  const found = new Set();
  const regex = /#([A-Za-z][A-Za-z0-9_]{1,23})\b/g;
  const source = `${text || ''}`;
  let match;
  while ((match = regex.exec(source))) found.add(match[1].toLowerCase());
  return [...found].slice(0, 8);
}

function extractModuleTags(text = '', manualTags = '') {
  const found = new Set();
  extractHashTags(`${text} ${manualTags}`)
    .map(tag => tag.toUpperCase())
    .filter(tag => /^[A-Z]{3,5}\d{3}$/.test(tag))
    .forEach(tag => found.add(tag));
  manualTags.split(',').map(tag => tag.trim().toUpperCase()).filter(Boolean).forEach(tag => found.add(tag));
  return [...found].slice(0, 5);
}

function getPostHashTags(post = {}) {
  if (Array.isArray(post.hashTags) && post.hashTags.length) return [...new Set(post.hashTags.map(tag => (tag || '').toLowerCase()).filter(Boolean))];
  return extractHashTags(post.content || '');
}

function renderPostModuleTags(moduleTags = []) {
  if (!moduleTags.length) return '';
  return `<div class="post-module-tags">${moduleTags.map(tag => `<button class="module-chip clickable" onclick="openModuleFeed('${tag}')">${esc(tag)}</button>`).join('')}</div>`;
}

function renderPostHashTags(hashTags = []) {
  if (!hashTags.length) return '';
  return `<div class="post-module-tags">${hashTags.slice(0, 4).map(tag => `<button class="module-chip clickable hashtag-chip" onclick="openTagFeed('${esc(tag)}')">#${esc(tag)}</button>`).join('')}</div>`;
}

const _expandedPostKeys = {};
const _postTextStore = {};
const _postTextLimit = {};
let _trendingRailTimer = null;

function buildExpandablePostHTML(text, key, maxChars = 320) {
  const raw = text || '';
  _postTextStore[key] = raw;
  _postTextLimit[key] = maxChars;
  const expanded = !!_expandedPostKeys[key];
  const shouldTrim = raw.length > maxChars;
  const shown = shouldTrim && !expanded ? `${raw.slice(0, maxChars).trimEnd()}...` : raw;
  return `
    <div class="expandable-post-body">
      <div class="post-content">${formatContent(shown)}</div>
      ${shouldTrim ? `<button class="post-expand-btn" onclick="event.stopPropagation();togglePostExpand('${key}')">${expanded ? 'Show less' : 'Show more'}</button>` : ''}
    </div>
  `;
}

function renderExpandablePostContent(text, key, maxChars = 320) {
  return `<div id="expandable-${key}">${buildExpandablePostHTML(text, key, maxChars)}</div>`;
}

function togglePostExpand(key) {
  _expandedPostKeys[key] = !_expandedPostKeys[key];
  const host = document.getElementById(`expandable-${key}`);
  if (!host) return;
  host.innerHTML = buildExpandablePostHTML(_postTextStore[key] || '', key, _postTextLimit[key] || 320);
}

function shiftTrendingRail(dir = 1) {
  const rail = document.getElementById('trending-post-scroll');
  if (!rail) return;
  const amount = rail.clientWidth * 0.82;
  const atEnd = rail.scrollLeft + rail.clientWidth >= rail.scrollWidth - 12;
  if (dir > 0 && atEnd) rail.scrollTo({ left: 0, behavior: 'smooth' });
  else rail.scrollBy({ left: amount * dir, behavior: 'smooth' });
}

function setupTrendingRail() {
  const rail = document.getElementById('trending-post-scroll');
  if (!rail) return;
  if (_trendingRailTimer) clearInterval(_trendingRailTimer);
  const pause = () => { if (_trendingRailTimer) { clearInterval(_trendingRailTimer); _trendingRailTimer = null; } };
  const resume = () => {
    if (_trendingRailTimer) clearInterval(_trendingRailTimer);
    _trendingRailTimer = setInterval(() => shiftTrendingRail(1), 4500);
  };
  rail.onmouseenter = pause;
  rail.onmouseleave = resume;
  rail.ontouchstart = pause;
  rail.ontouchend = () => setTimeout(resume, 1200);
  resume();
}

function renderTrendingPostsRail(posts = []) {
  const railHost = document.getElementById('feed-trending-posts');
  if (!railHost) {
    if (_trendingRailTimer) clearInterval(_trendingRailTimer);
    _trendingRailTimer = null;
    return;
  }
  const trending = [...posts]
    .filter(post => post.content || post.imageURL || post.videoURL)
    .sort((a, b) => (((b.likes || []).length + (b.commentsCount || 0) * 2) - (((a.likes || []).length + (a.commentsCount || 0) * 2))))
    .slice(0, 10);
  if (trending.length === 0) {
    if (_trendingRailTimer) clearInterval(_trendingRailTimer);
    _trendingRailTimer = null;
    railHost.innerHTML = '';
    return;
  }
  railHost.innerHTML = `
    <div class="trending-posts-card">
      <div class="trending-posts-head">
        <div>
          <h3>Trending Now</h3>
          <span>Auto-scrolling highlights</span>
        </div>
        <div class="trend-nav-actions">
          <button class="trend-nav-btn" onclick="shiftTrendingRail(-1)">‹</button>
          <button class="trend-nav-btn" onclick="shiftTrendingRail(1)">›</button>
        </div>
      </div>
      <div class="trending-post-scroll" id="trending-post-scroll">
        ${trending.map(post => `
          <div class="trending-post-card" onclick="viewPost('${post.id}')">
            ${post.videoURL || post.mediaType === 'video' ? `<div class="trending-post-media has-video"><video src="${post.videoURL || post.imageURL}" muted playsinline preload="metadata"></video><div class="trending-post-video-badge">▶</div></div>` : post.imageURL ? `<div class="trending-post-media"><img src="${post.imageURL}" alt=""></div>` : ''}
            <div class="trending-post-meta-top">
              <span>${post.isAnonymous ? '👻 Anonymous' : esc(post.authorName || 'User')}</span>
              <span>${((post.likes || []).length + (post.commentsCount || 0))} reacts</span>
            </div>
            ${post.content ? `<div class="trending-post-copy">${formatContent((post.content || '').slice(0, 150) + ((post.content || '').length > 150 ? '...' : ''))}</div>` : ''}
            ${renderPostModuleTags((post.moduleTags || []).slice(0, 2))}
          </div>
        `).join('')}
      </div>
    </div>
  `;
  setupTrendingRail();
}

async function openModuleFeed(tag) {
  const moduleTag = (tag || '').toUpperCase();
  if (!moduleTag) return;
  let posts = (state.posts || []).filter(post => (post.moduleTags || []).includes(moduleTag));
  if (!posts.length) {
    try {
      const snap = await db.collection('posts').orderBy('createdAt', 'desc').limit(50).get();
      posts = snap.docs.map(d => ({ id: d.id, ...d.data() })).filter(post => (post.moduleTags || []).includes(moduleTag));
    } catch (e) { console.error(e); }
  }
  openModal(`
    <div class="modal-header"><h2>${esc(moduleTag)} Posts</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body module-feed-modal">
      ${posts.length ? posts.map(post => `
        <div class="module-post-item" onclick="closeModal();viewPost('${post.id}')">
          <div class="module-post-top">
            ${post.isAnonymous ? `<div class="avatar-sm anon-avatar">👻</div>` : avatar(post.authorName, post.authorPhoto, 'avatar-sm')}
            <div>
              <div class="module-post-author">${post.isAnonymous ? 'Anonymous' : esc(post.authorName || 'User')}</div>
              <div class="module-post-time">${timeAgo(post.createdAt)}</div>
            </div>
          </div>
          ${post.content ? `<div class="module-post-text">${renderExpandablePostContent(post.content, `module-${post.id}`, 180)}</div>` : ''}
          ${renderPostModuleTags(post.moduleTags || [])}
        </div>`).join('') : '<div class="empty-state"><h3>No posts yet</h3><p>Be the first to post for this module.</p></div>'}
    </div>
  `);
}

async function openTagFeed(tag) {
  const hashTag = (tag || '').replace(/^#/, '').toLowerCase();
  if (!hashTag) return;
  let posts = (state.posts || []).filter(post => getPostHashTags(post).includes(hashTag));
  if (!posts.length) {
    try {
      const snap = await db.collection('posts').orderBy('createdAt', 'desc').limit(80).get();
      posts = snap.docs.map(d => ({ id: d.id, ...d.data() })).filter(post => getPostHashTags(post).includes(hashTag));
    } catch (e) { console.error(e); }
  }
  openModal(`
    <div class="modal-header"><h2>#${esc(hashTag)}</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body module-feed-modal">
      ${posts.length ? posts.map(post => `
        <div class="module-post-item" onclick="closeModal();viewPost('${post.id}')">
          <div class="module-post-top">
            ${post.isAnonymous ? `<div class="avatar-sm anon-avatar">👻</div>` : avatar(post.authorName, post.authorPhoto, 'avatar-sm')}
            <div>
              <div class="module-post-author">${post.isAnonymous ? 'Anonymous' : esc(post.authorName || 'User')}</div>
              <div class="module-post-time">${timeAgo(post.createdAt)}</div>
            </div>
          </div>
          ${post.content ? `<div class="module-post-text">${renderExpandablePostContent(post.content, `tag-${post.id}`, 140)}</div>` : ''}
          ${renderPostHashTags(getPostHashTags(post).filter(item => item !== hashTag))}
        </div>`).join('') : '<div class="empty-state"><h3>No posts yet</h3><p>Use this tag in a post to start the thread.</p></div>'}
    </div>
  `);
}

function openAnonPostActions(uid, postId = null) {
  if (!uid || uid === state.user.uid) return toast("That's you!");
  openModal(`
    <div class="modal-body" style="padding:20px 18px">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
        <div class="avatar-md anon-avatar">👻</div>
        <div>
          <h3 style="margin-bottom:4px">Anonymous ${postId ? 'Reply' : 'Message'}</h3>
          <p style="color:var(--text-secondary);font-size:13px">${postId ? 'Reply anonymously to this post. Both identities stay hidden until reveal.' : 'Start a private anonymous chat. Both sides stay hidden until reveal.'}</p>
        </div>
      </div>
      <button class="btn-primary btn-full" onclick="closeModal();startAnonChat('${uid}', null, null, false, ${postId ? `'${postId}'` : 'null'})">Send Anonymous ${postId ? 'Reply' : 'Message'}</button>
      <button class="btn-secondary btn-full" style="margin-top:10px" onclick="closeModal()">Cancel</button>
    </div>
  `);
}

async function notifyRelevantModuleUsers(moduleTags = [], text = '', postId) {
  const uniqueTags = [...new Set(moduleTags)].slice(0, 3);
  if (!uniqueTags.length || !postId) return;
  const notified = new Set();
  const notifTextFor = tag => /notes|summary|slides|past\s*paper|resource/i.test(text)
    ? `shared notes in ${tag}`
    : `posted in ${tag}`;
  try {
    const snap = await db.collection('users').limit(120).get();
    for (const doc of snap.docs) {
      if (doc.id === state.user.uid || notified.has(doc.id)) continue;
      const userData = doc.data() || {};
      const matchedTag = uniqueTags.find(tag => normalizeModules(userData.modules || []).includes(tag));
      if (!matchedTag) continue;
      notified.add(doc.id);
      await addNotification(doc.id, 'module', notifTextFor(matchedTag), { postId, moduleTag: matchedTag });
    }
  } catch (e) { console.error(e); }
}

function renderModuleTrends(posts = []) {
  const trendEl = document.getElementById('module-trends');
  if (!trendEl) return;
  const moduleCounts = new Map();
  const hashCounts = new Map();
  posts.forEach(post => {
    (post.moduleTags || []).forEach(tag => moduleCounts.set(tag, (moduleCounts.get(tag) || 0) + 1));
    getPostHashTags(post)
      .filter(tag => !/^[A-Z]{3,5}\d{3}$/.test(tag.toUpperCase()))
      .forEach(tag => hashCounts.set(tag, (hashCounts.get(tag) || 0) + 1));
  });
  const moduleTrends = [...moduleCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8);
  const hashTrends = [...hashCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8);
  if (!moduleTrends.length && !hashTrends.length) {
    trendEl.innerHTML = '';
    return;
  }
  trendEl.innerHTML = `
    ${moduleTrends.length ? `<div class="module-trends-card">
      <div class="module-trends-head">
        <h3>Module Trends</h3>
        <span>Tap to explore</span>
      </div>
      <div class="module-trends-row">
        ${moduleTrends.map(([tag, count]) => `<button class="trend-chip" onclick="openModuleFeed('${tag}')">${esc(tag)} <span>${count}</span></button>`).join('')}
      </div>
    </div>` : ''}
    ${hashTrends.length ? `<div class="module-trends-card hashtag-trends-card">
      <div class="module-trends-head">
        <h3>Campus Tags</h3>
        <span>Open live tag threads</span>
      </div>
      <div class="module-trends-row">
        ${hashTrends.map(([tag, count]) => `<button class="trend-chip" onclick="openTagFeed('${tag}')">#${esc(tag)} <span>${count}</span></button>`).join('')}
      </div>
    </div>` : ''}
  `;
}

// ─── Custom Voice Note Player ────────────────────
let _vnCounter = 0;
const _vnAudios = {};

function renderVoiceMsg(audioURL) {
  const id = `vn-${++_vnCounter}`;
  return `<div class="vn-player" id="${id}" data-src="${audioURL}">
    <button class="vn-play-btn" onclick="toggleVN('${id}')">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
    </button>
    <div class="vn-track" onclick="seekVN(event,'${id}')">
      <div class="vn-bar-bg"></div>
      <div class="vn-progress"></div>
      <div class="vn-dot"></div>
    </div>
    <span class="vn-time">0:00</span>
  </div>`;
}

function toggleVN(id) {
  const el = document.getElementById(id);
  if (!el) return;
  const src = el.dataset.src;
  // Pause any other playing VN
  Object.keys(_vnAudios).forEach(k => {
    if (k !== id && !_vnAudios[k].paused) {
      _vnAudios[k].pause();
      const o = document.getElementById(k);
      if (o) { o.querySelector('.vn-play-btn').innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>'; o.classList.remove('playing'); }
    }
  });
  if (!_vnAudios[id]) {
    const audio = new Audio(src);
    _vnAudios[id] = audio;
    audio.addEventListener('loadedmetadata', () => {
      if (audio.duration && isFinite(audio.duration)) {
        el.querySelector('.vn-time').textContent = fmtDur(audio.duration);
      }
    });
    audio.addEventListener('timeupdate', () => {
      if (!audio.duration) return;
      const pct = (audio.currentTime / audio.duration) * 100;
      el.querySelector('.vn-progress').style.width = pct + '%';
      el.querySelector('.vn-dot').style.left = pct + '%';
      el.querySelector('.vn-time').textContent = fmtDur(audio.duration - audio.currentTime);
    });
    audio.addEventListener('ended', () => {
      el.classList.remove('playing');
      el.querySelector('.vn-play-btn').innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
      el.querySelector('.vn-progress').style.width = '0%';
      el.querySelector('.vn-dot').style.left = '0%';
      if (audio.duration && isFinite(audio.duration)) el.querySelector('.vn-time').textContent = fmtDur(audio.duration);
    });
  }
  const audio = _vnAudios[id];
  const btn = el.querySelector('.vn-play-btn');
  if (audio.paused) {
    audio.play();
    el.classList.add('playing');
    btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="3" width="4" height="18" rx="1"/><rect x="15" y="3" width="4" height="18" rx="1"/></svg>';
  } else {
    audio.pause();
    el.classList.remove('playing');
    btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
  }
}

function seekVN(e, id) {
  const el = document.getElementById(id);
  if (!el || !_vnAudios[id]) return;
  const track = el.querySelector('.vn-track');
  const rect = track.getBoundingClientRect();
  const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  _vnAudios[id].currentTime = pct * _vnAudios[id].duration;
}

function fmtDur(s) {
  if (!s || !isFinite(s)) return '0:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

function compress(file, max = 800, q = 0.7) {
  return new Promise(ok => {
    const r = new FileReader();
    r.onload = e => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        let w = img.width, h = img.height;
        if (w > max) { h = h * (max / w); w = max; }
        c.width = w; c.height = h;
        c.getContext('2d').drawImage(img, 0, 0, w, h);
        ok(c.toDataURL('image/jpeg', q));
      };
      img.src = e.target.result;
    };
    r.readAsDataURL(file);
  });
}

// ─── R2 Cloud Media Storage ──────────────────────
const R2_BASE = 'https://app-media.badumetsihlongwane.workers.dev/Images/';

/**
 * Upload a file to R2. Returns the public URL.
 * Path: /Images/{userId}/{timestamp}_{random}_{filename}
 * This scopes files per user to prevent data leaks.
 */
async function uploadToR2(file, folder = '') {
  const uid = state.user?.uid || 'anon';
  const ts = Date.now();
  const rand = Math.random().toString(36).slice(2, 8);
  const safeName = file.name.replace(/[^a-zA-Z0-9._-]/g, '_');
  const path = folder ? `${uid}/${folder}/${ts}_${rand}_${safeName}` : `${uid}/${ts}_${rand}_${safeName}`;
  const url = R2_BASE + path;
  try {
    const resp = await fetch(url, {
      method: 'PUT',
      body: file,
      headers: { 'Content-Type': file.type || 'application/octet-stream' },
      mode: 'cors'
    });
    if (!resp.ok) {
      console.error('R2 upload status:', resp.status, resp.statusText);
      throw new Error('R2 status ' + resp.status);
    }
    return url;
  } catch (e) {
    console.error('R2 upload failed:', e.message || e);
    // Fallback to base64 for images only
    if (file.type.startsWith('image/')) {
      console.log('R2 blocked (CORS?), falling back to base64');
      return await compress(file);
    }
    // For video/audio — no base64 fallback possible
    toast('Media upload failed — update your R2 worker (see worker/r2-worker.js)');
    return null;
  }
}

/** Quick local preview (for showing before upload finishes) */
function localPreview(file) {
  return URL.createObjectURL(file);
}

/** Check if a file is a video */
function isVideo(file) {
  return file && file.type && file.type.startsWith('video/');
}

/** Check if a URL points to a video */
function isVideoURL(url) {
  if (!url) return false;
  return /\.(mp4|webm|mov|avi|mkv)([?#]|$)/i.test(url) || url.includes('/videos/');
}

function formatContent(text) {
  if (!text) return '';
  return esc(text).replace(/#(\w+)/g, (_, rawTag) => {
    const tag = (rawTag || '').toUpperCase();
    if (/^[A-Z]{3,5}\d{3}$/.test(tag)) {
      return `<span class="hashtag module-hashtag" onclick="openModuleFeed('${tag}')">#${tag}</span>`;
    }
    return `<span class="hashtag" onclick="openTagFeed('${rawTag.toLowerCase()}')">#${rawTag}</span>`;
  });
}

// ─── Custom Video Player Engine ──────────────────
let _playerIdCounter = 0;

function buildPlayerHTML(src, id) {
  return `
  <div class="unino-player show-controls" id="up-${id}" data-player-id="${id}">
    <video preload="metadata" playsinline>
      <source src="${src}" type="video/mp4">
      <source src="${src}" type="video/webm">
    </video>

    <div class="up-loader"><div class="up-loader-ring"></div></div>

    <div class="up-big-play" data-act="toggle">
      <svg viewBox="0 0 24 24"><polygon points="5,3 19,12 5,21"/></svg>
    </div>

    <div class="up-play-anim"><svg viewBox="0 0 24 24"><polygon points="5,3 19,12 5,21"/></svg></div>

    <div class="up-skip-indicator left">-10s</div>
    <div class="up-skip-indicator right">+10s</div>

    <div class="up-tap-zone left"></div>
    <div class="up-tap-zone center" data-act="toggle"></div>
    <div class="up-tap-zone right"></div>

    <div class="up-top-bar">
      <span class="up-top-title"></span>
    </div>

    <div class="up-controls">
      <div class="up-progress-wrap">
        <div class="up-time-preview">0:00</div>
        <div class="up-progress-track">
          <div class="up-progress-buffer" style="width:0"></div>
          <div class="up-progress-fill" style="width:0"></div>
        </div>
        <div class="up-progress-thumb" style="left:0"></div>
      </div>

      <div class="up-controls-row">
        <button class="up-btn play-btn" data-act="toggle" aria-label="Play">
          <svg class="up-icon-play" viewBox="0 0 24 24" fill="#fff" stroke="none"><polygon points="5,3 19,12 5,21"/></svg>
          <svg class="up-icon-pause" viewBox="0 0 24 24" fill="#fff" stroke="none" style="display:none"><rect x="5" y="3" width="4" height="18" rx="1"/><rect x="15" y="3" width="4" height="18" rx="1"/></svg>
        </button>

        <span class="up-time"><span class="up-cur">0:00</span><span class="up-time-sep"> / </span><span class="up-dur">0:00</span></span>

        <span class="up-spacer"></span>

        <div class="up-vol-wrap">
          <button class="up-btn vol-btn" aria-label="Volume">
            <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" fill="#fff" stroke="none"/><path d="M15.54 8.46a5 5 0 010 7.07"/><path d="M19.07 4.93a10 10 0 010 14.14"/></svg>
          </button>
          <div class="up-vol-slider-wrap"><input type="range" class="up-vol-slider" min="0" max="1" step="0.05" value="1"></div>
        </div>

        <span class="up-speed-badge" data-act="speed">1x</span>

        <button class="up-btn" data-act="pip" aria-label="Picture in picture">
          <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><rect x="12" y="9" width="8" height="6" rx="1" fill="rgba(255,255,255,0.3)"/></svg>
        </button>

        <button class="up-btn" data-act="fullscreen" aria-label="Fullscreen">
          <svg class="up-icon-fs" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>
          <svg class="up-icon-fs-exit" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" style="display:none"><polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/></svg>
        </button>
      </div>
    </div>
  </div>`;
}

function initPlayer(id) {
  const root = document.getElementById('up-' + id);
  if (!root) return;
  const vid = root.querySelector('video');
  if (!vid) return;

  // Elements
  const bigPlay = root.querySelector('.up-big-play');
  const playAnim = root.querySelector('.up-play-anim');
  const loader = root.querySelector('.up-loader');
  const progressWrap = root.querySelector('.up-progress-wrap');
  const progressFill = root.querySelector('.up-progress-fill');
  const progressBuffer = root.querySelector('.up-progress-buffer');
  const progressThumb = root.querySelector('.up-progress-thumb');
  const timePreview = root.querySelector('.up-time-preview');
  const curTime = root.querySelector('.up-cur');
  const durTime = root.querySelector('.up-dur');
  const playBtn = root.querySelector('.play-btn');
  const iconPlay = root.querySelector('.up-icon-play');
  const iconPause = root.querySelector('.up-icon-pause');
  const volSlider = root.querySelector('.up-vol-slider');
  const speedBadge = root.querySelector('.up-speed-badge');
  const skipLeft = root.querySelector('.up-skip-indicator.left');
  const skipRight = root.querySelector('.up-skip-indicator.right');
  const iconFs = root.querySelector('.up-icon-fs');
  const iconFsExit = root.querySelector('.up-icon-fs-exit');
  const tapLeft = root.querySelector('.up-tap-zone.left');
  const tapRight = root.querySelector('.up-tap-zone.right');

  let controlsTimer = null;
  let isSeeking = false;
  const speeds = [0.5, 0.75, 1, 1.25, 1.5, 2];
  let speedIdx = 2;

  // Format time
  const fmt = (s) => {
    if (isNaN(s) || !isFinite(s)) return '0:00';
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return m + ':' + sec.toString().padStart(2, '0');
  };

  // Show/hide controls
  const showControls = () => {
    root.classList.add('show-controls');
    clearTimeout(controlsTimer);
    if (!vid.paused) {
      controlsTimer = setTimeout(() => root.classList.remove('show-controls'), 3000);
    }
  };

  const hideControlsSoon = () => {
    clearTimeout(controlsTimer);
    if (!vid.paused) {
      controlsTimer = setTimeout(() => root.classList.remove('show-controls'), 3000);
    }
  };

  // Toggle play/pause
  const togglePlay = () => {
    if (vid.paused) {
      vid.play().catch(() => {});
    } else {
      vid.pause();
    }
  };

  // Update play/pause icons
  const syncPlayState = () => {
    const playing = !vid.paused;
    root.classList.toggle('playing', playing);
    iconPlay.style.display = playing ? 'none' : 'block';
    iconPause.style.display = playing ? 'block' : 'none';
    // Update big play anim
    playAnim.querySelector('svg').innerHTML = playing
      ? '<polygon points="5,3 19,12 5,21"/>'
      : '<rect x="5" y="3" width="4" height="18" rx="1"/><rect x="15" y="3" width="4" height="18" rx="1"/>';
    showControls();
  };

  // Flash play/pause animation
  const flashAnim = () => {
    playAnim.classList.remove('pop');
    void playAnim.offsetWidth;
    playAnim.classList.add('pop');
  };

  // Update progress
  const updateProgress = () => {
    if (isSeeking || !vid.duration) return;
    const pct = (vid.currentTime / vid.duration) * 100;
    progressFill.style.width = pct + '%';
    progressThumb.style.left = pct + '%';
    curTime.textContent = fmt(vid.currentTime);
  };

  // Update buffer
  const updateBuffer = () => {
    if (vid.buffered.length > 0) {
      const end = vid.buffered.end(vid.buffered.length - 1);
      progressBuffer.style.width = (end / vid.duration) * 100 + '%';
    }
  };

  // Seek via progress bar
  const seekFromEvent = (e) => {
    const rect = progressWrap.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    vid.currentTime = pct * vid.duration;
    progressFill.style.width = (pct * 100) + '%';
    progressThumb.style.left = (pct * 100) + '%';
    curTime.textContent = fmt(vid.currentTime);
  };

  // Double-tap skip
  let tapTimers = {};
  const doubleTapSkip = (direction, zone) => {
    const key = direction;
    if (tapTimers[key]) {
      clearTimeout(tapTimers[key]);
      tapTimers[key] = null;
      const skipSec = direction === 'left' ? -10 : 10;
      vid.currentTime = Math.max(0, Math.min(vid.duration, vid.currentTime + skipSec));
      const indicator = direction === 'left' ? skipLeft : skipRight;
      indicator.classList.remove('show');
      void indicator.offsetWidth;
      indicator.classList.add('show');
      showControls();
    } else {
      tapTimers[key] = setTimeout(() => {
        tapTimers[key] = null;
        togglePlay();
        flashAnim();
      }, 250);
    }
  };

  // === EVENT LISTENERS ===

  vid.addEventListener('play', syncPlayState);
  vid.addEventListener('pause', () => { syncPlayState(); root.classList.add('show-controls'); clearTimeout(controlsTimer); });
  vid.addEventListener('timeupdate', updateProgress);
  vid.addEventListener('progress', updateBuffer);
  vid.addEventListener('loadedmetadata', () => { durTime.textContent = fmt(vid.duration); });
  vid.addEventListener('ended', () => { root.classList.remove('playing'); root.classList.add('show-controls'); clearTimeout(controlsTimer); });
  vid.addEventListener('waiting', () => loader.classList.add('active'));
  vid.addEventListener('canplay', () => loader.classList.remove('active'));

  // Controls hover/touch
  root.addEventListener('mousemove', showControls);
  root.addEventListener('mouseleave', hideControlsSoon);
  root.addEventListener('touchstart', () => {
    if (root.classList.contains('show-controls')) {
      root.classList.remove('show-controls');
    } else {
      showControls();
    }
  }, { passive: true });

  // Big play
  bigPlay.addEventListener('click', (e) => { e.stopPropagation(); togglePlay(); flashAnim(); });

  // Tap zones
  tapLeft.addEventListener('click', (e) => { e.stopPropagation(); doubleTapSkip('left', tapLeft); });
  tapRight.addEventListener('click', (e) => { e.stopPropagation(); doubleTapSkip('right', tapRight); });
  root.querySelector('.up-tap-zone.center').addEventListener('click', (e) => {
    e.stopPropagation(); togglePlay(); flashAnim();
  });

  // Play button
  playBtn.addEventListener('click', (e) => { e.stopPropagation(); togglePlay(); flashAnim(); });

  // Progress scrubbing
  let scrubbing = false;
  progressWrap.addEventListener('mousedown', (e) => {
    e.stopPropagation();
    scrubbing = true;
    isSeeking = true;
    seekFromEvent(e);
  });
  document.addEventListener('mousemove', (e) => {
    if (!scrubbing) return;
    seekFromEvent(e);
  });
  document.addEventListener('mouseup', () => {
    if (scrubbing) { scrubbing = false; isSeeking = false; }
  });
  // Touch scrubbing
  progressWrap.addEventListener('touchstart', (e) => {
    e.stopPropagation();
    scrubbing = true;
    isSeeking = true;
    const touch = e.touches[0];
    seekFromEvent(touch);
  }, { passive: false });
  progressWrap.addEventListener('touchmove', (e) => {
    if (!scrubbing) return;
    e.preventDefault();
    seekFromEvent(e.touches[0]);
  }, { passive: false });
  progressWrap.addEventListener('touchend', () => {
    scrubbing = false; isSeeking = false;
  });

  // Hover time preview
  progressWrap.addEventListener('mousemove', (e) => {
    const rect = progressWrap.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    timePreview.textContent = fmt(pct * (vid.duration || 0));
    timePreview.style.left = (pct * 100) + '%';
  });

  // Volume
  if (volSlider) {
    volSlider.addEventListener('input', (e) => {
      e.stopPropagation();
      vid.volume = parseFloat(volSlider.value);
      vid.muted = vid.volume === 0;
    });
    const volBtn = root.querySelector('.vol-btn');
    if (volBtn) {
      volBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        vid.muted = !vid.muted;
        volSlider.value = vid.muted ? 0 : vid.volume || 1;
      });
    }
  }

  // Speed
  speedBadge.addEventListener('click', (e) => {
    e.stopPropagation();
    speedIdx = (speedIdx + 1) % speeds.length;
    vid.playbackRate = speeds[speedIdx];
    speedBadge.textContent = speeds[speedIdx] + 'x';
  });

  // PiP
  root.querySelector('[data-act="pip"]')?.addEventListener('click', async (e) => {
    e.stopPropagation();
    try {
      if (document.pictureInPictureElement) await document.exitPictureInPicture();
      else await vid.requestPictureInPicture();
    } catch (err) { console.warn('PiP not supported'); }
  });

  // Fullscreen
  root.querySelector('[data-act="fullscreen"]')?.addEventListener('click', (e) => {
    e.stopPropagation();
    if (document.fullscreenElement === root) {
      document.exitFullscreen();
    } else {
      (root.requestFullscreen || root.webkitRequestFullscreen || root.msRequestFullscreen).call(root);
    }
  });

  // Fullscreen change icon
  const onFsChange = () => {
    const fs = document.fullscreenElement === root;
    iconFs.style.display = fs ? 'none' : 'block';
    iconFsExit.style.display = fs ? 'block' : 'none';
  };
  document.addEventListener('fullscreenchange', onFsChange);
  document.addEventListener('webkitfullscreenchange', onFsChange);

  // Keyboard shortcuts when player focused
  root.addEventListener('keydown', (e) => {
    switch (e.key) {
      case ' ': case 'k': e.preventDefault(); togglePlay(); flashAnim(); break;
      case 'ArrowRight': e.preventDefault(); vid.currentTime = Math.min(vid.duration, vid.currentTime + 10); break;
      case 'ArrowLeft': e.preventDefault(); vid.currentTime = Math.max(0, vid.currentTime - 10); break;
      case 'ArrowUp': e.preventDefault(); vid.volume = Math.min(1, vid.volume + 0.1); if (volSlider) volSlider.value = vid.volume; break;
      case 'ArrowDown': e.preventDefault(); vid.volume = Math.max(0, vid.volume - 0.1); if (volSlider) volSlider.value = vid.volume; break;
      case 'f': e.preventDefault(); root.querySelector('[data-act="fullscreen"]')?.click(); break;
      case 'm': e.preventDefault(); vid.muted = !vid.muted; break;
    }
  });

  // Make focusable
  root.setAttribute('tabindex', '0');

  // Initial controls auto-hide
  controlsTimer = setTimeout(() => root.classList.remove('show-controls'), 4000);

  return { root, vid, togglePlay };
}

function createVideoPlayer(src) {
  const id = ++_playerIdCounter;
  return { html: buildPlayerHTML(src, id), id };
}

// ─── Screen Manager ──────────────────────────────
function stopAllVideos() {
  document.querySelectorAll('video').forEach(v => { v.pause(); v.currentTime = 0; });
  document.querySelectorAll('audio').forEach(a => { a.pause(); a.currentTime = 0; });
}

function showScreen(id) {
  stopAllVideos();
  $$('.screen').forEach(s => s.classList.remove('active'));
  const el = document.getElementById(id);
  if (el) el.classList.add('active');
  // Push state for back navigation
  if (_inAppBackInit && state.user) {
    history.pushState({ app: true, screen: id, page: state.page, msgTab: state.lastMsgTab }, '');
  }
}

function unsub() { state.unsubs.forEach(fn => fn()); state.unsubs = []; }

// ══════════════════════════════════════════════════
//  THEME
// ══════════════════════════════════════════════════
function initTheme() {
  const saved = localStorage.getItem('unino-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  $('#theme-btn')?.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('unino-theme', next);
  });
}

// ══════════════════════════════════════════════════
//  AUTH
// ══════════════════════════════════════════════════
function initAuth() {
  $('#to-signup')?.addEventListener('click', e => {
    e.preventDefault();
    $('#login-form').classList.remove('active');
    $('#signup-form').classList.add('active');
  });
  $('#to-login')?.addEventListener('click', e => {
    e.preventDefault();
    $('#signup-form').classList.remove('active');
    $('#login-form').classList.add('active');
  });

  // LOGIN
  $('#login-form')?.addEventListener('submit', async e => {
    e.preventDefault();
    const btn = $('#l-btn'), email = $('#l-email').value.trim(), pass = $('#l-pass').value;
    if (!email || !pass) return toast('Enter email and password');
    if (!isStudentEmail(email)) return toast('Use your @mynwu.ac.za student email');
    btn.disabled = true; btn.innerHTML = '<span class="inline-spinner"></span>';
    try { await auth.signInWithEmailAndPassword(email, pass); }
    catch (err) { toast(friendlyErr(err.code)); btn.disabled = false; btn.textContent = 'Log In'; }
  });

  // SIGNUP
  $('#signup-form')?.addEventListener('submit', async e => {
    e.preventDefault();
    const btn = $('#s-btn');
    const fname = $('#s-fname').value.trim(), lname = $('#s-lname').value.trim();
    const email = $('#s-email').value.trim(), pass = $('#s-pass').value;
    const uni = $('#s-uni').value, major = $('#s-major').value, year = $('#s-year')?.value || '';
    const modulesRaw = $('#s-modules')?.value || '';
    const modules = modulesRaw.split(',').map(m => m.trim().toUpperCase()).filter(Boolean);
    const address = $('#s-address')?.value.trim() || '';
    if (!fname || !lname || !email || !pass || !uni || !major || !address) return toast('All fields required');
    if (pass.length < 6) return toast('Password must be 6+ characters');
    if (!isStudentEmail(email)) return toast('Only @mynwu.ac.za emails can sign up');
    btn.disabled = true; btn.innerHTML = '<span class="inline-spinner"></span>';
    try {
      const cred = await auth.createUserWithEmailAndPassword(email, pass);
      const uid = cred.user.uid;
      const displayName = `${fname} ${lname}`;
      await db.collection('users').doc(uid).set({
        displayName, firstName: fname, lastName: lname,
        email, university: uni, major, year, modules, address,
        bio: `${major} student at ${uni}`,
        photoURL: '', status: 'online', allowAutoFill: true,
        joinedAt: FieldVal.serverTimestamp(), friends: []
      });
      await cred.user.updateProfile({ displayName });
      db.collection('stats').doc('global').set({ totalUsers: FieldVal.increment(1) }, { merge: true }).catch(() => {});
    } catch (err) { toast(friendlyErr(err.code)); btn.disabled = false; btn.textContent = 'Create Account'; }
  });

  // Suggest common addresses during signup.
  const signupAddressInput = $('#s-address');
  const signupAddressSuggestions = $('#s-address-suggestions');
  if (signupAddressInput && signupAddressSuggestions) {
    const staticAddressHints = [
      ...CAMPUS_LOCATIONS.map(l => l.name),
      'Potch Main Campus', 'Cachet Park', 'Bult', 'Die Bult', 'Mohadin',
      'Student Village', 'Res Halls', 'Library Side', 'Engineering Block',
      'Main Gate', 'North Gate', 'South Gate'
    ];
    const addressHints = Array.from(new Set(staticAddressHints));

    const renderAddressHints = (query = '') => {
      const q = query.trim().toLowerCase();
      if (!q || q.length < 2) {
        signupAddressSuggestions.style.display = 'none';
        signupAddressSuggestions.innerHTML = '';
        return;
      }
      const hits = addressHints.filter(a => a.toLowerCase().includes(q)).slice(0, 7);
      if (!hits.length) {
        signupAddressSuggestions.style.display = 'none';
        signupAddressSuggestions.innerHTML = '';
        return;
      }
      signupAddressSuggestions.innerHTML = hits.map(h => `<button type="button" class="address-suggestion-item" data-v="${esc(h)}">${esc(h)}</button>`).join('');
      signupAddressSuggestions.style.display = 'block';
      signupAddressSuggestions.querySelectorAll('.address-suggestion-item').forEach(btn => {
        btn.onclick = () => {
          signupAddressInput.value = btn.getAttribute('data-v') || '';
          signupAddressSuggestions.style.display = 'none';
        };
      });
    };

    signupAddressInput.addEventListener('input', e => renderAddressHints(e.target.value));
    signupAddressInput.addEventListener('focus', e => renderAddressHints(e.target.value));
    signupAddressInput.addEventListener('blur', () => setTimeout(() => {
      signupAddressSuggestions.style.display = 'none';
    }, 120));
  }

  // AUTH STATE
  auth.onAuthStateChanged(async user => {
    if (user) {
      if (!isStudentEmail(user.email || '')) {
        toast('Only @mynwu.ac.za accounts are allowed');
        await auth.signOut().catch(() => {});
        return;
      }
      state.user = user;
      try {
        const doc = await db.collection('users').doc(user.uid).get();
        state.profile = doc.exists
          ? { id: doc.id, ...doc.data() }
          : { id: user.uid, displayName: user.displayName, email: user.email, status: 'online', modules: [] };
      } catch {
        state.profile = { id: user.uid, displayName: user.displayName, email: user.email, status: 'online', modules: [] };
      }
      state.manualStatus = state.profile.manualStatus || state.profile.status || 'online';
      state.status = state.profile.status || state.manualStatus;
      // Admin detection
      _isAdmin = (user.email || '').toLowerCase() === ADMIN_EMAIL.toLowerCase();
      if (_isAdmin) VERIFIED_UIDS.add(user.uid);
      if (state.profile.isVerified) VERIFIED_UIDS.add(user.uid);
      enterApp();
    } else {
      if (verifiedUsersUnsub) { verifiedUsersUnsub(); verifiedUsersUnsub = null; }
      if (_groupAlertUnsub) { _groupAlertUnsub(); _groupAlertUnsub = null; }
      VERIFIED_UIDS.clear();
      _asgPendingAlerts = [];
      _dmUnreadCount = 0;
      state.user = null; state.profile = null; unsub(); showScreen('auth-screen');
    }
  });

  db.collection('stats').doc('global').onSnapshot(doc => {
    const el = $('#auth-count'); if (el && !state.user) el.textContent = '0';
  });
}

function friendlyErr(code) {
  return { 'auth/user-not-found':'Account not found','auth/wrong-password':'Incorrect password',
    'auth/email-already-in-use':'Email already registered','auth/weak-password':'Password too weak',
    'auth/invalid-email':'Invalid email' }[code] || 'Something went wrong';
}

// ══════════════════════════════════════════════════
//  ENTER APP
// ══════════════════════════════════════════════════
function enterApp() {
  showScreen('app'); setupHeader(); setupNav(); setupStatusPill(); 
  if (!_inAppBackInit) {
    // Fence: replace current history with app state, then push initial state
    history.replaceState({ app: true, screen: 'app', page: 'feed' }, '');
    _inAppBackInit = true;
  }
  navigate('feed');
  listenForVerifiedUsers();
  listenForNotifications();
  listenForUnreadDMs();
  listenForGroupAlerts();
  setupPresenceTracking();
  listenForOnlineCount();
}

let _onlineCountSub = null;
function listenForOnlineCount() {
  if (_onlineCountSub) _onlineCountSub();
  _onlineCountSub = db.collection('users').where('status', '==', 'online').onSnapshot(snap => {
    const count = snap.size || 0;
    const authEl = $('#auth-count'); if (authEl && state.user) authEl.textContent = count;
    const headerEl = $('#hdr-count'); if (headerEl) headerEl.textContent = count;
    const feedEl = $('#feed-online'); if (feedEl) feedEl.textContent = count;
  }, () => {});
}

let _unreadDMSub = null;
function refreshChatBadge() {
  const asgPendingTotal = _asgPendingAlerts.reduce((sum, g) => sum + (g.pendingRequests || []).length, 0);
  const total = (_dmUnreadCount || 0) + asgPendingTotal;
  const badge = document.getElementById('chat-badge');
  if (badge) {
    badge.textContent = total || '';
    badge.style.display = total ? 'flex' : 'none';
  }
  const asgBadge = document.getElementById('asg-tab-badge');
  if (asgBadge) {
    asgBadge.textContent = asgPendingTotal || '';
    asgBadge.style.display = asgPendingTotal ? 'inline-flex' : 'none';
  }
}

function listenForVerifiedUsers() {
  if (verifiedUsersUnsub) verifiedUsersUnsub();
  verifiedUsersUnsub = db.collection('users').where('isVerified', '==', true).onSnapshot(snap => {
    VERIFIED_UIDS.clear();
    snap.docs.forEach(d => VERIFIED_UIDS.add(d.id));
    if (_isAdmin && state.user?.uid) VERIFIED_UIDS.add(state.user.uid);
    // Refresh visible areas so badges appear as soon as verified list updates.
    if (state.page === 'feed' && Array.isArray(state.posts) && state.posts.length) {
      renderPosts(state.posts);
    }
    if (document.getElementById('profile-view')?.classList.contains('active')) {
      const pid = document.getElementById('prof-back')?.dataset?.uid;
      if (pid) openProfile(pid);
    }
  }, () => {
    if (_isAdmin && state.user?.uid) VERIFIED_UIDS.add(state.user.uid);
    if (state.page === 'feed' && Array.isArray(state.posts) && state.posts.length) {
      renderPosts(state.posts);
    }
  });
}

function listenForUnreadDMs() {
  if (_unreadDMSub) _unreadDMSub();
  _unreadDMSub = db.collection('conversations')
    .where('participants', 'array-contains', state.user.uid)
    .onSnapshot(snap => {
      const uid = state.user.uid;
      let total = 0;
      snap.docs.forEach(d => { total += (d.data().unread || {})[uid] || 0; });
      _dmUnreadCount = total;
      refreshChatBadge();
    }, () => {});
}

function listenForGroupAlerts() {
  if (_groupAlertUnsub) _groupAlertUnsub();
  _groupAlertUnsub = db.collection('assignmentGroups')
    .where('createdBy', '==', state.user.uid)
    .onSnapshot(snap => {
      _asgPendingAlerts = snap.docs
        .map(d => ({ id: d.id, ...d.data() }))
        .filter(g => (g.status || 'open') === 'open' && (g.pendingRequests || []).length > 0);
      refreshChatBadge();
      updateNotifBadge();
      const dd = $('#notif-dropdown');
      if (dd && dd.style.display === 'block') loadNotifications();
    }, () => {});
}

function _updateDMTabBadge() {
  db.collection('conversations').where('participants', 'array-contains', state.user.uid).get().then(snap => {
    const uid = state.user.uid;
    let total = 0;
    snap.docs.forEach(d => { total += (d.data().unread || {})[uid] || 0; });
    const b = document.getElementById('dm-tab-badge');
    if (b) { b.textContent = total || ''; b.style.display = total ? 'inline-flex' : 'none'; }
  }).catch(() => {});
}

function setupHeader() {
  const el = $('#hdr-avatar');
  if (!el || !state.profile) return;
  const p = state.profile;
  if (p.photoURL) { el.innerHTML = `<img src="${p.photoURL}" alt="">`; el.style.background = 'transparent'; }
  else { el.textContent = initials(p.displayName); el.style.background = colorFor(p.displayName); }
  el.onclick = () => openProfile(state.user.uid);
  // Admin panel button
  const existingAdmin = document.getElementById('admin-btn');
  if (_isAdmin && !existingAdmin) {
    const btn = document.createElement('button');
    btn.className = 'icon-btn'; btn.id = 'admin-btn'; btn.title = 'Admin Panel';
    btn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>';
    btn.onclick = () => openAdminPanel();
    document.querySelector('.hdr-r')?.insertBefore(btn, el);
  }
}

function setupNav() {
  $$('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const p = btn.dataset.p;
      if (p === 'create') return openCreateModal();
      navigate(p);
    });
  });
}

function setupStatusPill() {
  const pill = $('#status-pill'); if (!pill) return;
  updateStatusUI();
  pill.onclick = async () => {
    const modes = ['online','study','offline'];
    state.manualStatus = modes[(modes.indexOf(state.manualStatus) + 1) % 3];
    refreshPresence(true).catch(() => {});
    toast('Status: ' + state.manualStatus.charAt(0).toUpperCase() + state.manualStatus.slice(1));
  };
}

function updateStatusUI() {
  const dot = $('#status-dot'), txt = $('#status-text'), pill = $('#status-pill');
  if (!dot || !txt) return;
  pill.className = 'status-pill'; dot.className = 'dot';
  if (state.status === 'online') { pill.classList.add('online'); dot.classList.add('green'); txt.textContent = 'Online'; }
  else if (state.status === 'study') { pill.classList.add('away'); dot.classList.add('orange'); txt.textContent = 'Studying'; }
  else { pill.classList.add('offline'); dot.classList.add('gray'); txt.textContent = 'Offline'; }
}

let _lastActivityAt = Date.now();
let _presenceTimer = null;

async function syncConversationPresence(status) {
  if (!state.user?.uid) return;
  try {
    const snap = await db.collection('conversations').where('participants', 'array-contains', state.user.uid).get();
    await Promise.all(snap.docs.map(d => d.ref.set({ participantStatuses: { [state.user.uid]: status } }, { merge: true })));
  } catch (e) { console.error(e); }
}

async function refreshPresence(force = false) {
  if (!state.user?.uid) return;
  const inactive = (Date.now() - _lastActivityAt) > 3 * 60 * 1000;
  const effective = state.manualStatus === 'online' ? (inactive ? 'offline' : 'online') : state.manualStatus;
  if (!force && state.status === effective) return;
  state.status = effective;
  updateStatusUI();
  try {
    await db.collection('users').doc(state.user.uid).update({ status: effective, manualStatus: state.manualStatus, lastActiveAt: FieldVal.serverTimestamp() });
    syncConversationPresence(effective);
  } catch (e) { console.error(e); }
}

function markActivity() {
  _lastActivityAt = Date.now();
  if (state.manualStatus === 'online' && state.status !== 'online') refreshPresence().catch(() => {});
}

function setupPresenceTracking() {
  ['mousemove','keydown','click','touchstart','scroll'].forEach(evt => {
    document.addEventListener(evt, markActivity, { passive: true });
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      if (state.manualStatus === 'online') {
        _lastActivityAt = 0;
        refreshPresence(true).catch(() => {});
      }
    } else {
      markActivity();
      refreshPresence(true).catch(() => {});
    }
  });
  // Keep phone back navigation inside the app instead of leaving the site.
  if (!_inAppBackListenerBound) {
    window.addEventListener('popstate', (e) => {
      if (!state.user) {
        // Prevent going back before login
        history.pushState({ app: true }, '');
        return;
      }

      // Handle back within app
      if ($('#modal-bg')?.style.display === 'block') {
        closeModal();
        return;
      }

      if ($('#chat-view')?.classList.contains('active')) {
        $('#chat-back')?.click();
        return;
      }

      if ($('#group-chat-view')?.classList.contains('active')) {
        $('#gchat-back')?.click();
        return;
      }

      if ($('#profile-view')?.classList.contains('active') || $('#settings-view')?.classList.contains('active')) {
        showScreen('app');
        return;
      }

      if (state.page !== 'feed') {
        navigate('feed');
        return;
      }

      // At feed - prevent going further back
      history.pushState({ app: true, screen: 'app', page: 'feed' }, '');
    });
    _inAppBackListenerBound = true;
  }
  clearInterval(_presenceTimer);
  _presenceTimer = setInterval(() => refreshPresence().catch(() => {}), 30000);
  refreshPresence(true).catch(() => {});
}

// ─── Navigation ──────────────────────────────────
function navigate(page) {
  state.page = page; unsub();
  stopAllVideos();
  if (_leafletMap) { _leafletMap.remove(); _leafletMap = null; }
  $$('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.p === page));
  switch (page) {
    case 'feed': renderFeed(); break;
    case 'explore': renderExplore(); break;
    case 'hustle': renderHustle(); break;
    case 'chat': renderMessages(); break;
  }
  // Push state for navigation
  if (_inAppBackInit && state.user) {
    history.pushState({ app: true, screen: 'app', page, msgTab: state.lastMsgTab }, '');
  }
}

// ══════════════════════════════════════════════════
//  FEED — Clean with unified Discover tabs
// ══════════════════════════════════════════════════
function renderFeed() {
  const c = $('#content'), p = state.profile;
  c.innerHTML = `
    <div class="feed-page">
      ${!window._greetingShown ? `<div class="welcome-banner" id="welcome-banner">
        <div class="welcome-text">
          <h2>Hey, ${esc(p.firstName || p.displayName?.split(' ')[0])} 👋</h2>
          <p>${esc(p.university || 'NWU Campus')}</p>
        </div>
        <div class="welcome-stat">
          <span class="dot green"></span> <span id="feed-online">0</span> online
        </div>
      </div>` : `<div class="welcome-stat" style="padding:12px 16px;display:flex;align-items:center;gap:6px;font-size:13px;color:var(--text-secondary)">
        <span class="dot green"></span> <span id="feed-online">0</span> online
      </div>`}

      <div class="stories-row" id="stories-row">
        <div class="story-item add-story" onclick="openStoryCreator()">
          <div class="story-avatar"><div class="story-avatar-inner">+</div></div>
          <div class="story-name">Story</div>
        </div>
      </div>

      <div class="discover-section">
        <div class="discover-tabs">
          <button class="discover-tab active" data-dt="people">👥 People</button>
          <button class="discover-tab" data-dt="events">📅 Events</button>
        </div>
        <div class="discover-content" id="discover-content">
          <div style="padding:20px;text-align:center"><span class="inline-spinner"></span></div>
        </div>
      </div>

      <div class="create-post-prompt" onclick="openCreateModal()">
        ${avatar(p.displayName, p.photoURL, 'avatar-md')}
        <div class="placeholder-text">What's on your mind?</div>
        <div class="prompt-actions"><span class="prompt-action">+</span></div>
      </div>

      <div id="module-trends"></div>

      <div id="feed-trending-posts"></div>

      <div id="feed-posts">
        <div style="padding:40px;text-align:center"><span class="inline-spinner" style="width:28px;height:28px;color:var(--accent)"></span></div>
      </div>
      <button class="reels-fab" onclick="openReelsViewer()" title="Watch Reels">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="#fff" stroke="#fff" stroke-width="1.5"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2" fill="none" stroke="#fff" stroke-width="2"/></svg>
      </button>
    </div>
  `;

  // Mark greeting as shown for this session
  window._greetingShown = true;

  // Wire discover tabs
  $$('.discover-tab').forEach(tab => {
    tab.onclick = () => {
      $$('.discover-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      if (tab.dataset.dt === 'people') loadDiscoverPeople();
      else loadDiscoverEvents();
    };
  });

  loadDiscoverPeople();
  loadStories();

  // Real-time posts
  const u = db.collection('posts').orderBy('createdAt', 'desc').limit(50)
    .onSnapshot(snap => {
      const myFriends = state.profile.friends || [];
      const uid = state.user.uid;
      const allPosts = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      // Filter: show public posts, own posts, and friends-only posts from friends
      const visible = allPosts.filter(post => {
        if (post.authorId === uid) return true;
        if (post.visibility === 'friends') return myFriends.includes(post.authorId);
        return true; // public or no visibility set
      });

      const previousOrder = new Map((state.posts || []).map((post, index) => [post.id, index]));
      const likedTagPrefs = new Set();
      const likedModulePrefs = new Set(normalizeModules(state.profile.modules || []));
      (state.posts || []).forEach(existingPost => {
        if ((existingPost.likes || []).includes(uid)) {
          normalizeModules(existingPost.moduleTags || []).forEach(tag => likedModulePrefs.add(tag));
          getPostHashTags(existingPost).forEach(tag => likedTagPrefs.add(tag));
        }
      });

      // Discovery ranking for new posts, while preserving the current on-screen order for existing ones.
      const scored = visible.map(p => {
        const likes = (p.likes || []).length;
        const comments = p.commentsCount || 0;
        const isFriend = myFriends.includes(p.authorId);
        const sharedModules = normalizeModules(p.moduleTags || []).filter(tag => likedModulePrefs.has(tag)).length;
        const sharedTags = getPostHashTags(p).filter(tag => likedTagPrefs.has(tag)).length;
        const ageHrs = p.createdAt ? (Date.now() - (p.createdAt.toDate ? p.createdAt.toDate() : new Date(p.createdAt)).getTime()) / 3600000 : 999;
        const freshness = Math.max(0, 1 - ageHrs / 48); // decay over 48h
        const engagement = (likes * 2 + comments * 3) * 0.3;
        const friendBoost = isFriend ? 8 : 0;
        const interestBoost = sharedModules * 10 + sharedTags * 5;
        const randomFactor = scoreSeed(p.id) * 10;
        return { ...p, _score: engagement + freshness * 15 + friendBoost + interestBoost + randomFactor };
      });
      scored.sort((a, b) => {
        const ai = previousOrder.has(a.id) ? previousOrder.get(a.id) : -1;
        const bi = previousOrder.has(b.id) ? previousOrder.get(b.id) : -1;
        if (ai !== -1 && bi !== -1) return ai - bi;
        if (ai !== -1) return 1;
        if (bi !== -1) return -1;
        return b._score - a._score || (b.createdAt?.seconds || 0) - (a.createdAt?.seconds || 0);
      });

      // Keep feed position stable for like updates by patching only one card.
      if (window._lastLikedPost && state.posts?.length) {
        const prevIds = state.posts.map(p => p.id).join(',');
        const nextIds = scored.map(p => p.id).join(',');
        if (prevIds === nextIds) {
          state.posts = scored;
          const likedPost = scored.find(p => p.id === window._lastLikedPost);
          if (likedPost) patchPostEngagement(likedPost);
          window._lastLikedPost = null;
          return;
        }
      }

      state.posts = scored;
      // Save scroll position before re-render to prevent "jump to top"
      const contentEl = document.getElementById('content');
      const savedScroll = contentEl ? contentEl.scrollTop : 0;
      renderPosts(scored);
      renderModuleTrends(scored);
      renderTrendingPostsRail(scored);
      // Restore scroll position after re-render
      if (contentEl && savedScroll > 0) {
        requestAnimationFrame(() => { contentEl.scrollTop = savedScroll; });
      }
    });
  state.unsubs.push(u);
}

// ─── Discover: People tab ────────────────────────
function loadDiscoverPeople() {
  const el = $('#discover-content'); if (!el) return;
  const myMajor = state.profile.major || '';
  const myModules = normalizeModules(state.profile.modules || []);
  const myAddress = state.profile.address || '';

  db.collection('users').limit(30).get().then(snap => {
    let users = snap.docs.map(d => ({ id: d.id, ...d.data() })).filter(u => u.id !== state.user.uid);

    // Score & sort by relevance
    users = users.map(u => {
      let score = 0;
      const theirModules = normalizeModules(u.modules || []);
      const shared = myModules.filter(m => theirModules.includes(m));
      const nearbyScore = addressMatchScore(myAddress, u.address || '');
      if (shared.length) score += 30 + shared.length * 10;
      if (nearbyScore > 0) score += 20 + nearbyScore * 6;
      if (u.major === myMajor) score += 10;
      if (u.status === 'online') score += 5;
      return { ...u, score, sharedModules: shared, nearbyScore };
    }).sort((a, b) => b.score - a.score).slice(0, 10);

    if (!users.length) {
      el.innerHTML = `<div class="discover-empty"><span>👥</span><p>No students found yet. Invite friends!</p></div>`;
      return;
    }

    el.innerHTML = `<div class="discover-scroll">${users.map(u => {
      const tag = u.sharedModules?.length
        ? `🔗 ${u.sharedModules.length} shared module${u.sharedModules.length > 1 ? 's' : ''}`
        : u.nearbyScore > 0 ? '📍 Nearby area'
        : u.major ? `📚 ${esc(u.major)}` : '';
      const online = u.status === 'online' ? '<span class="online-dot"></span>' : '';
      const isFriend = (state.profile.friends || []).includes(u.id);
      const isPending = (state.profile.sentRequests || []).includes(u.id);
      const actionBtn = isFriend
        ? `<button class="discover-card-btn" onclick="event.stopPropagation();startChat('${u.id}','${esc(u.displayName)}','${u.photoURL || ''}')">Message</button>`
        : isPending
          ? `<button class="discover-card-btn" disabled style="opacity:0.6;cursor:not-allowed">Pending…</button>`
          : `<button class="discover-card-btn" onclick="event.stopPropagation();sendFriendRequest('${u.id}','${esc(u.displayName)}','${u.photoURL || ''}', this)">Add Friend</button>`;
      return `
        <div class="discover-card" onclick="openProfile('${u.id}')">
          <div class="discover-card-avatar">
            ${avatar(u.displayName, u.photoURL, 'avatar-lg')}
            ${online}
          </div>
          <div class="discover-card-name">${esc(u.displayName)}</div>
          <div class="discover-card-meta">${esc(u.major || 'Student')}</div>
          ${tag ? `<div class="discover-card-tag">${tag}</div>` : ''}
          ${actionBtn}
        </div>`;
    }).join('')}</div>`;
  }).catch(() => { el.innerHTML = '<div class="discover-empty"><p>Could not load</p></div>'; });
}

// ─── Discover: Events tab ────────────────────────
function loadDiscoverEvents() {
  const el = $('#discover-content'); if (!el) return;
  // Use allCampusEvents if loaded, otherwise fetch
  const renderEvts = (events) => {
    if (!events.length) {
      el.innerHTML = `<div class="discover-empty"><span>📅</span><p>No events yet. Check the Campus map!</p></div>`;
      return;
    }
    el.innerHTML = `<div class="discover-scroll">${events.slice(0, 8).map(ev => {
      const loc = CAMPUS_LOCATIONS.find(l => l.id === ev.location);
      const locName = loc ? loc.name : esc(ev.location || '?');
      const grad = ev.gradient || 'linear-gradient(135deg,#6C5CE7,#A855F7)';
      const goingCount = (ev.going || []).length;
      const thumb = (ev.imageURLs && ev.imageURLs.length) ? ev.imageURLs[0] : null;
      return `
        <div class="discover-card event-card" onclick="${ev.id ? `openEventDetail('${ev.id}')` : `toast('View on Campus map!')`}">
          ${thumb ? `<img src="${thumb}" style="width:100%;height:120px;object-fit:cover;border-radius:var(--radius);margin-bottom:8px">` : `<div style="background:${grad};width:100%;height:120px;border-radius:var(--radius);display:flex;align-items:center;justify-content:center;font-size:36px;margin-bottom:8px">${ev.emoji || '📅'}</div>`}
          <div class="discover-card-name">${esc(ev.title)}</div>
          <div class="discover-card-meta">${esc(ev.date || '')} ${esc(ev.time || '')}</div>
          <div class="discover-card-tag">📍 ${locName}</div>
          ${goingCount ? `<div style="font-size:11px;color:var(--text-tertiary);margin-top:4px">👥 ${goingCount} going</div>` : ''}
        </div>`;
    }).join('')}</div>`;
  };
  if (allCampusEvents.length) { renderEvts(allCampusEvents); }
  else {
    loadCampusEvents().then(() => renderEvts(allCampusEvents));
  }
}

// ─── Stories System ──────────────────────────────
function loadStories() {
  const row = $('#stories-row'); if (!row) return;
  // Clear all but the add-story button
  row.querySelectorAll('.story-item:not(.add-story)').forEach(el => el.remove());

  const cutoff = new Date(Date.now() - 24 * 60 * 60 * 1000);
  db.collection('stories').where('expiresAt', '>', cutoff).orderBy('expiresAt','desc').limit(30)
    .get().then(snap => {
      // Group stories by author
      const byUser = {};
      const myFriends = state.profile.friends || [];
      snap.docs.forEach(d => {
        const s = { id: d.id, ...d.data() };
        // Only show own stories and friends' stories
        if (s.authorId !== state.user.uid && !myFriends.includes(s.authorId)) return;
        if (!byUser[s.authorId]) byUser[s.authorId] = [];
        byUser[s.authorId].push(s);
      });
      // Put current user first if they have stories
      const uid = state.user.uid;
      const ordered = [];
      if (byUser[uid]) { ordered.push({ uid, stories: byUser[uid] }); delete byUser[uid]; }
      Object.keys(byUser).forEach(k => ordered.push({ uid: k, stories: byUser[k] }));

      ordered.forEach(group => {
        const s = group.stories[0];
        const isMe = group.uid === uid;
        const name = isMe ? 'You' : esc(s.authorFirstName || s.authorName?.split(' ')[0] || '?');
        const hasNew = group.stories.some(st => !(st.viewedBy || []).includes(uid));
        row.insertAdjacentHTML('beforeend', `
          <div class="story-item ${hasNew ? 'has-unseen' : 'seen'}" onclick="event.stopPropagation();viewStory('${group.uid}')">
            <div class="story-avatar"><div class="story-avatar-inner">
              ${s.authorPhoto ? `<img src="${s.authorPhoto}" alt="" draggable="false">` : initials(s.authorName)}
            </div></div>
            <div class="story-name">${name}</div>
          </div>
        `);
      });

    }).catch(() => {});
}

function openStoryCreator() {
  let bgColor = '#6C5CE7';
  const bgOptions = ['#6C5CE7','#A855F7','#7C3AED','#D946EF','#FF6B6B','#00BA88','#3B82F6','#FF9F43'];
  window._storyFile = null;
  window._storyVideoFile = null;
  openModal(`
    <div class="modal-header"><h2>Create Story</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      <div class="story-creator">
        <div class="story-text-preview" id="story-text-bg" style="background:${bgColor}">
          <textarea id="story-text-input" placeholder="Type your story..." maxlength="200"></textarea>
        </div>
        <div class="story-bg-picker">${bgOptions.map(c => `<button class="bg-dot" style="background:${c}" onclick="document.getElementById('story-text-bg').style.background='${c}';window._storyBg='${c}'"></button>`).join('')}</div>
        <div id="story-media-preview" style="display:none;margin-top:12px;position:relative;border-radius:var(--radius);overflow:hidden">
          <div id="story-media-content"></div>
          <button onclick="document.getElementById('story-media-preview').style.display='none';window._storyFile=null;window._storyVideoFile=null;document.getElementById('story-text-bg').style.display='flex'" style="position:absolute;top:6px;right:6px;width:28px;height:28px;border-radius:50%;background:rgba(0,0,0,0.6);color:#fff;border:none;font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center">&times;</button>
        </div>
        <div style="display:flex;gap:8px;margin-top:12px;align-items:center">
          <label class="story-upload-btn">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
            <span>Photo / Video</span>
            <input type="file" hidden accept="image/*,video/*" id="story-media-file">
          </label>
        </div>
        <button class="btn-primary btn-full" id="story-submit" style="margin-top:16px">Share Story</button>
      </div>
    </div>
  `);
  window._storyBg = bgColor;
  $('#story-media-file').onchange = e => {
    const file = e.target.files[0];
    if (!file) return;
    const isVideo = file.type.startsWith('video/');
    const preview = $('#story-media-preview');
    const content = $('#story-media-content');
    if (isVideo) {
      window._storyVideoFile = file;
      window._storyFile = null;
      content.innerHTML = `<video src="${localPreview(file)}" style="width:100%;max-height:220px;object-fit:cover;border-radius:var(--radius)" controls></video>`;
    } else {
      window._storyFile = file;
      window._storyVideoFile = null;
      content.innerHTML = `<img src="${localPreview(file)}" style="width:100%;max-height:220px;object-fit:cover;border-radius:var(--radius)">`;
    }
    preview.style.display = 'block';
    // Hide text bg when media is selected
    $('#story-text-bg').style.display = 'none';
  };
  $('#story-submit').onclick = async () => {
    const text = $('#story-text-input')?.value.trim();
    const hasMedia = window._storyFile || window._storyVideoFile;
    if (!text && !hasMedia) return toast('Add text or media!');
    const p = state.profile;
    let storyData = {
      authorId: state.user.uid,
      authorName: p.displayName,
      authorFirstName: p.firstName || p.displayName?.split(' ')[0],
      authorPhoto: p.photoURL || null,
      createdAt: FieldVal.serverTimestamp(),
      expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
      viewedBy: []
    };
    if (window._storyVideoFile) {
      storyData.type = 'video';
      storyData.caption = text || '';
      closeModal(); toast('Uploading video story...');
      storyData.videoURL = await uploadToR2(window._storyVideoFile, 'stories');
    } else if (window._storyFile) {
      storyData.type = 'photo';
      storyData.caption = text || '';
      closeModal(); toast('Uploading story...');
      storyData.imageURL = await uploadToR2(window._storyFile, 'stories');
    } else {
      storyData.type = 'text';
      storyData.text = text;
      storyData.bgColor = window._storyBg || '#6C5CE7';
    }
    if (document.querySelector('.modal-overlay')) closeModal();
    toast('Posting story...');
    try {
      await db.collection('stories').add(storyData);
      toast('Story shared!');
      loadStories();
    } catch (e) { toast('Failed'); console.error(e); }
  };
}

let storyViewerData = { groups: [], currentGroup: 0, currentStory: 0, timer: null };

async function viewStory(userId) {
  const cutoff = new Date(Date.now() - 24 * 60 * 60 * 1000);
  try {
    const snap = await db.collection('stories').where('expiresAt', '>', cutoff).orderBy('expiresAt','desc').get();
    const byUser = {};
    snap.docs.forEach(d => {
      const s = { id: d.id, ...d.data() };
      if (!byUser[s.authorId]) byUser[s.authorId] = [];
      byUser[s.authorId].push(s);
    });
    const groups = Object.keys(byUser).map(uid => ({ uid, stories: byUser[uid] }));
    const startIdx = groups.findIndex(g => g.uid === userId);
    if (startIdx === -1) return toast('No stories');

    storyViewerData = { groups, currentGroup: startIdx, currentStory: 0, timer: null };
    showStoryFrame();
  } catch (e) { console.error(e); toast('Could not load stories'); }
}

function showStoryFrame() {
  const { groups, currentGroup, currentStory } = storyViewerData;
  if (currentGroup >= groups.length) return closeStoryViewer();
  const group = groups[currentGroup];
  if (currentStory >= group.stories.length) {
    storyViewerData.currentGroup++;
    storyViewerData.currentStory = 0;
    return showStoryFrame();
  }
  const story = group.stories[currentStory];
  const viewer = $('#story-viewer');
  viewer.style.display = 'flex';

  // Progress bar
  const bar = $('#story-progress-bar');
  bar.innerHTML = group.stories.map((_, i) =>
    `<div class="story-progress-seg ${i < currentStory ? 'done' : i === currentStory ? 'active' : ''}"><div class="story-progress-fill"></div></div>`
  ).join('');

  // Header
  const hdr = $('#story-viewer-user');
  const isMyStory = story.authorId === state.user.uid;
  const viewCount = Math.max(0, (story.viewedBy || []).filter(uid => uid !== state.user.uid).length);
  hdr.innerHTML = `
    ${avatar(story.authorName, story.authorPhoto, 'avatar-sm')}
    <div><b>${esc(story.authorName)}</b><br><small>${timeAgo(story.createdAt)}${isMyStory ? ` · ${viewCount} view${viewCount === 1 ? '' : 's'}` : ''}</small></div>
    ${isMyStory ? `<button class="story-delete-btn" onclick="deleteStory('${story.id}')">Delete</button>` : ''}
  `;

  // Content
  const content = $('#story-viewer-content');
  let autoAdvanceMs = 5000;
  if (story.type === 'video' && story.videoURL) {
    content.innerHTML = `
      <video src="${story.videoURL}" class="story-full-video" autoplay playsinline loop style="width:100%;height:100%;object-fit:cover"></video>
      ${story.caption ? `<div class="story-caption">${esc(story.caption)}</div>` : ''}
    `;
    content.style.background = '#000';
    autoAdvanceMs = 15000;
    const vid = content.querySelector('video');
    if (vid) {
      vid.onended = () => advanceStory(1);
    }
  } else if (story.type === 'photo') {
    content.innerHTML = `
      <img src="${story.imageURL}" class="story-full-img">
      ${story.caption ? `<div class="story-caption">${esc(story.caption)}</div>` : ''}
    `;
    content.style.background = '#000';
  } else {
    content.innerHTML = `<div class="story-text-display">${esc(story.text)}</div>`;
    content.style.background = story.bgColor || '#6C5CE7';
  }

  // Mark as viewed
  if (!(story.viewedBy || []).includes(state.user.uid)) {
    db.collection('stories').doc(story.id).update({ viewedBy: FieldVal.arrayUnion(state.user.uid) }).catch(() => {});
  }

  // Auto-advance timer
  clearTimeout(storyViewerData.timer);
  storyViewerData.timer = setTimeout(() => advanceStory(1), autoAdvanceMs);

  // Navigation
  $('#story-prev').onclick = () => advanceStory(-1);
  $('#story-next').onclick = () => advanceStory(1);
  $('#story-close').onclick = closeStoryViewer;
}

function advanceStory(dir) {
  clearTimeout(storyViewerData.timer);
  if (dir > 0) {
    storyViewerData.currentStory++;
  } else {
    storyViewerData.currentStory--;
    if (storyViewerData.currentStory < 0) {
      storyViewerData.currentGroup--;
      if (storyViewerData.currentGroup < 0) return closeStoryViewer();
      storyViewerData.currentStory = storyViewerData.groups[storyViewerData.currentGroup].stories.length - 1;
    }
  }
  showStoryFrame();
}

function closeStoryViewer() {
  clearTimeout(storyViewerData.timer);
  $('#story-viewer').style.display = 'none';
}

async function deleteStory(storyId) {
  if (!storyId) return;
  if (!window.confirm('Delete this story?')) return;
  try {
    const doc = await db.collection('stories').doc(storyId).get();
    if (!doc.exists) return toast('Story not found');
    if (doc.data().authorId !== state.user.uid) return toast('Only your own story can be deleted');
    await db.collection('stories').doc(storyId).delete();
    closeStoryViewer();
    loadStories();
    toast('Story deleted');
  } catch (e) { toast('Failed to delete story'); console.error(e); }
}

// ─── Multi-image Collage Renderer ────────────────
function renderCollage(urls) {
  if (!urls || urls.length <= 1) return '';
  const count = urls.length;
  const cls = `collage-grid collage-${Math.min(count, 4)}`;
  return `<div class="${cls}">${urls.slice(0, 4).map((url, i) =>
    `<div class="collage-item${count > 4 && i === 3 ? ' collage-more-item' : ''}" onclick="viewImage('${url}')">
      <img src="${url}" loading="lazy">
      ${count > 4 && i === 3 ? `<div class="collage-more-overlay">+${count - 4}</div>` : ''}
    </div>`
  ).join('')}</div>`;
}

// ─── Quote Embed Renderer ────────────────────────
const _pendingQuotePlayers = [];
function renderQuoteEmbed(rp) {
  if (!rp) return '';
  const hasImg = rp.imageURL && rp.mediaType !== 'video';
  const hasVid = rp.videoURL || (rp.mediaType === 'video');
  const vidUrl = hasVid ? (rp.videoURL || rp.imageURL) : null;
  let vidHtml = '';
  if (hasVid && vidUrl) {
    const vpd = createVideoPlayer(vidUrl);
    _pendingQuotePlayers.push(vpd);
    vidHtml = `<div onclick="event.stopPropagation()" style="border-radius:8px;overflow:hidden;max-height:200px">${vpd.html}</div>`;
  }
  return `
    <div class="quote-embed" onclick="${rp.id ? `viewPost('${rp.id}')` : ''}" style="cursor:pointer;border:1px solid var(--border);border-radius:var(--radius);padding:12px;margin:8px 0;background:var(--bg-secondary)">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
        ${avatar(rp.authorName || 'User', rp.authorPhoto, 'avatar-sm')}
        <span style="font-weight:600;font-size:13px">${esc(rp.authorName || 'User')}</span>
      </div>
      ${rp.content ? `<div style="font-size:13px;color:var(--text-secondary);margin-bottom:8px;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden">${esc(rp.content)}</div>` : ''}
      ${hasImg && rp.imageURL ? `<img src="${rp.imageURL}" style="width:100%;max-height:160px;object-fit:cover;border-radius:8px" onclick="event.stopPropagation();viewImage('${rp.imageURL}')">` : ''}
      ${vidHtml}
    </div>`;
}

// ─── Render Posts ────────────────────────────────
function renderPosts(posts) {
  const el = $('#feed-posts'); if (!el) return;
  if (!posts.length) {
    el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">📝</div><h3>No posts yet</h3><p>Be the first to share something!</p></div>`;
    return;
  }
  const _videoPlayers = [];
  el.innerHTML = posts.map(post => {
    const liked = (post.likes || []).includes(state.user.uid);
    const lc = (post.likes || []).length, cc = post.commentsCount || 0;
    const canAnonMessage = post.isAnonymous && post.authorId !== state.user.uid;
    const hasCollage = post.imageURLs && post.imageURLs.length > 1 && !post.repostOf;
    const hasVideo = post.videoURL || (post.mediaType === 'video');
    const hasImage = post.imageURL && !hasVideo && !hasCollage;
    const mediaURL = hasVideo ? (post.videoURL || post.imageURL) : post.imageURL;
    const displayAuthorPhoto = post.isAnonymous ? null : (post.authorPhoto || resolvePostAuthorPhoto(post));
    let videoPlayerData = null;
    if (hasVideo && mediaURL) {
      videoPlayerData = createVideoPlayer(mediaURL);
      _videoPlayers.push(videoPlayerData);
    }
    return `
      <div class="post-card" id="post-${post.id}">
        ${post.repostOf ? `<div class="repost-badge">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
          <span>${esc(post.authorName)} reposted</span>
        </div>` : ''}
        <div class="post-header">
          ${post.isAnonymous
            ? `<div class="avatar-md anon-avatar" onclick="openAnonPostActions('${post.authorId}', '${post.id}')" style="cursor:pointer">👻</div>`
            : `<div class="feed-author-avatar" data-author-id="${post.authorId}" data-author-name="${esc(post.authorName)}" onclick="openProfile('${post.authorId}')" style="cursor:pointer">${avatar(post.authorName, displayAuthorPhoto, 'avatar-md')}</div>`}
          <div class="post-header-info">
            <div class="post-author-name" ${post.isAnonymous ? `onclick="openAnonPostActions('${post.authorId}', '${post.id}')" style="cursor:pointer"` : `onclick="openProfile('${post.authorId}')"`}>${post.isAnonymous ? '👻 Anonymous' : esc(post.authorName) + verifiedBadge(post.authorId)}</div>
            <div class="post-meta">${post.visibility === 'friends' ? '👫 ' : post.isAnonymous ? '👻 ' : '🌍 '}${post.isAnonymous ? '' : esc(post.authorUni || '')}${post.isAnonymous ? '' : ' · '}${timeAgo(post.createdAt)}</div>
          </div>
          ${!post.isAnonymous && post.authorId === state.user.uid ? `<button class="icon-btn post-more-btn" onclick="showPostOptions('${post.id}')" title="Options" style="margin-left:auto;font-size:18px;color:var(--text-tertiary)">⋯</button>` : ''}
        </div>
        ${post.content ? renderExpandablePostContent(post.content, `feed-${post.id}`, 180) : ''}
        ${renderPostModuleTags(post.moduleTags || [])}
        ${renderPostHashTags(getPostHashTags(post).filter(tag => !(post.moduleTags || []).includes(tag.toUpperCase())))}
        ${!post.repostOf && hasImage ? `<div class="post-media-wrap"><img src="${mediaURL}" class="post-image" loading="lazy" onclick="viewImage('${mediaURL}')"></div>` : ''}
        ${hasCollage ? renderCollage(post.imageURLs) : ''}
        ${!post.repostOf && hasVideo && videoPlayerData ? videoPlayerData.html : ''}
        ${post.repostOf ? renderQuoteEmbed(post.repostOf) : ''}
        <div class="post-engagement">
          <div class="post-stats">
            ${lc ? `<span class="stat-item"><svg width="14" height="14" viewBox="0 0 24 24" fill="var(--red)" stroke="none"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg> ${lc}</span>` : ''}
            ${cc ? `<span class="stat-item">${cc} comment${cc > 1 ? 's' : ''}</span>` : ''}
          </div>
          <div class="post-actions">
            ${canAnonMessage ? `<button class="post-action anon-inline-action" onclick="openAnonPostActions('${post.authorId}', '${post.id}')">👻 Reply</button>` : ''}
            <button class="post-action ${liked ? 'liked' : ''}" onclick="toggleLike('${post.id}')">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="${liked ? 'var(--red)' : 'none'}" stroke="${liked ? 'var(--red)' : 'currentColor'}" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
              ${lc || 'Like'}
            </button>
            <button class="post-action" onclick="openComments('${post.id}')">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
              ${cc || 'Comment'}
            </button>
            <button class="post-action" onclick="openShareModal('${post.id}')">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
              Share
            </button>
          </div>
        </div>
      </div>`;
  }).join('');

  // Initialize all custom video players after DOM update
  requestAnimationFrame(() => {
    _videoPlayers.forEach(p => initPlayer(p.id));
    _pendingQuotePlayers.forEach(p => initPlayer(p.id));
    _pendingQuotePlayers.length = 0;
    setupFeedVideoAutoplay();
  });
}

// ─── Reels Viewer (TikTok-style fullscreen vertical scroll) ─────────
let _reelsActive = false;
let _reelVideos = [];

function openReelsViewer() {
  // Fetch ALL video posts from Firestore, not just what's in state
  db.collection('posts').orderBy('createdAt', 'desc').limit(100).get().then(snap => {
    const allPosts = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    _reelVideos = allPosts.filter(p => p.videoURL || p.mediaType === 'video');
    // Shuffle with engagement-weighted randomization for discovery
    _reelVideos = _reelVideos.map(p => {
      const likes = (p.likes || []).length;
      const comments = p.commentsCount || 0;
      return { ...p, _score: (likes + comments) * 0.3 + Math.random() * 10 };
    }).sort((a, b) => b._score - a._score);
    if (!_reelVideos.length) return toast('No reels yet — post a video!');
    _reelsActive = true;
    renderReelsUI();
  }).catch(e => { console.error(e); toast('Could not load reels'); });
}

function renderReelsUI() {
  const existing = document.getElementById('reels-viewer');
  if (existing) existing.remove();

  const container = document.createElement('div');
  container.id = 'reels-viewer';
  container.className = 'reels-container';
  container.innerHTML = `
    <div class="reels-header">
      <h3>Clips</h3>
      <button class="reels-close-btn" onclick="closeReelsViewer()">&times;</button>
    </div>
    <div class="reels-scroll" id="reels-scroll">
      ${_reelVideos.map((p, i) => {
        const url = p.videoURL || p.imageURL;
        const liked = (p.likes || []).includes(state.user.uid);
        const lc = (p.likes || []).length;
        const cc = p.commentsCount || 0;
        return `
        <div class="reel-slide" data-idx="${i}">
          <video class="reel-video" src="${url}" loop playsinline preload="metadata" muted></video>
          <div class="reel-overlay-bottom">
            <div class="reel-author" onclick="closeReelsViewer();openProfile('${p.authorId}')">
              ${avatar(p.authorName, p.authorPhoto, 'avatar-sm')}
              <span class="reel-author-name">${esc(p.authorName || 'User')}</span>
            </div>
            ${p.content ? `<p class="reel-caption">${esc(p.content)}</p>` : ''}
          </div>
          <div class="reel-actions">
            <button class="reel-act-btn ${liked ? 'liked' : ''}" onclick="reelLike('${p.id}', this)">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="${liked ? '#ff4757' : 'none'}" stroke="${liked ? '#ff4757' : '#fff'}" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
              <span>${lc || ''}</span>
            </button>
            <button class="reel-act-btn" onclick="event.stopPropagation();openReelComments('${p.id}')">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
              <span>${cc || ''}</span>
            </button>
            <button class="reel-act-btn" onclick="openShareModal('${p.id}')">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
              <span>Share</span>
            </button>
          </div>
          <div class="reel-play-toggle" onclick="toggleReelPlay(this)"></div>
        </div>`;
      }).join('')}
    </div>
  `;
  document.body.appendChild(container);

  const scrollEl = document.getElementById('reels-scroll');
  // Intersection observer for auto-play
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const video = entry.target.querySelector('.reel-video');
      if (!video) return;
      if (entry.isIntersecting && entry.intersectionRatio >= 0.7) {
        video.muted = false;
        video.play().catch(() => {});
      } else {
        video.pause();
      }
    });
  }, { root: scrollEl, threshold: [0.7] });

  scrollEl.querySelectorAll('.reel-slide').forEach(slide => observer.observe(slide));

  // Play first reel
  requestAnimationFrame(() => {
    const firstVid = scrollEl.querySelector('.reel-slide:first-child .reel-video');
    if (firstVid) { firstVid.muted = false; firstVid.play().catch(() => {}); }
  });
}

function toggleReelPlay(overlay) {
  const vid = overlay.parentElement.querySelector('.reel-video');
  if (!vid) return;
  if (vid.paused) { vid.play().catch(() => {}); overlay.classList.remove('paused'); }
  else { vid.pause(); overlay.classList.add('paused'); }
}

function closeReelsViewer() {
  _reelsActive = false;
  const el = document.getElementById('reels-viewer');
  if (el) {
    el.querySelectorAll('video').forEach(v => v.pause());
    el.remove();
  }
}

// ─── Inline Reel Comments ────────────────────────
async function openReelComments(postId) {
  const reelsScroll = document.querySelector('.reels-scroll');
  const scrollPos = reelsScroll ? reelsScroll.scrollTop : 0;
  
  _pendingReelCommentImageFile = null;
  _reelCommentReplyTo = null;

  const existing = document.getElementById('reel-comments-panel');
  if (existing) existing.remove();
  
  if (reelsScroll) reelsScroll.scrollTop = scrollPos;

  let comments = [];
  try {
    const snap = await db.collection('posts').doc(postId).collection('comments').limit(100).get();
    comments = snap.docs.map(d => ({ id: d.id, ...d.data() }));
  } catch (e) { console.error(e); }

  comments.forEach(c => { c.likeCount = (c.likes || []).length; });
  const topLevel = comments.filter(c => !c.replyTo);
  const replies = comments.filter(c => c.replyTo);
  topLevel.sort((a, b) => b.likeCount - a.likeCount || (b.createdAt?.seconds || 0) - (a.createdAt?.seconds || 0));
  replies.sort((a, b) => (a.createdAt?.seconds || 0) - (b.createdAt?.seconds || 0));
  const replyMap = {};
  replies.forEach(r => {
    if (!replyMap[r.replyTo]) replyMap[r.replyTo] = [];
    replyMap[r.replyTo].push(r);
  });
  const commentById = {};
  comments.forEach(c => { commentById[c.id] = c; });

  const renderComment = (c, isReply = false) => {
    const liked = (c.likes || []).includes(state.user.uid);
    const cReplies = replyMap[c.id] || [];
    const target = c.replyTo ? commentById[c.replyTo] : null;
    const fromLabel = c.authorId === state.user.uid ? 'me' : (c.authorName || 'User');
    const toLabel = target ? (target.authorId === state.user.uid ? 'me' : (target.authorName || 'User')) : '';
    return `
      <div class="comment-item ${isReply ? 'reply-item' : ''}" id="rc-${c.id}">
        <div class="comment-avatar-col">${avatar(c.authorName || 'User', c.authorPhoto, 'avatar-sm')}</div>
        <div class="comment-content-col">
          <div class="comment-bubble enhanced">
            <div class="comment-header"><span class="comment-author">${esc(c.authorName || 'User')}</span></div>
            ${target ? `<div class="comment-reply-to">${esc(fromLabel)} &gt; ${esc(toLabel)}</div>` : ''}
            ${c.text ? `<div class="comment-text">${esc(c.text)}</div>` : ''}
            ${c.imageURL ? `<img src="${c.imageURL}" class="comment-inline-image" onclick="viewImage('${c.imageURL}')">` : ''}
          </div>
          <div class="comment-actions-row">
            <span class="comment-time">${timeAgo(c.createdAt)}</span>
            <button class="c-act ${liked ? 'liked' : ''}" onclick="toggleReelCommentLike('${c.id}','${postId}')">Like ${c.likeCount > 0 ? c.likeCount : ''}</button>
            <button class="c-act" onclick="setReelCommentReply('${c.id}','${esc(c.authorName || 'User')}')">Reply</button>
          </div>
          ${cReplies.length ? `<div class="comment-replies">${cReplies.map(r => renderComment(r, true)).join('')}</div>` : ''}
        </div>
      </div>`;
  };

  const panel = document.createElement('div');
  panel.id = 'reel-comments-panel';
  panel.className = 'reel-comments-panel';
  panel.onclick = e => e.stopPropagation();
  panel.innerHTML = `
    <div class="reel-comments-header">
      <h3>Comments</h3>
      <button class="icon-btn" onclick="closeReelComments()">✕</button>
    </div>
    <div class="reel-comments-list" id="reel-comments-list">
      ${topLevel.length ? topLevel.map(c => renderComment(c)).join('') : '<div class="empty-msg" style="text-align:center;padding:20px;color:var(--text-tertiary)">No comments yet</div>'}
    </div>
    <div id="reel-comment-reply-indicator" class="reply-indicator" style="display:none">
      <span id="reel-comment-reply-label"></span>
      <button onclick="clearReelCommentReply()">&times;</button>
    </div>
    <div class="comment-input-wrap modern reel-input-wrap">
      <div id="reel-comment-img-preview" class="comment-img-preview" style="display:none"></div>
      <div class="comment-compose-row compact">
        <label class="add-photo-btn comment-attach-btn" title="Add sticker/image">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
          <input type="file" hidden accept="image/*" id="reel-comment-image-input">
        </label>
        <textarea id="reel-comment-input" placeholder="Add a comment..." autocomplete="off"></textarea>
        <button class="send-btn" onclick="postReelComment('${postId}')"><svg class="send-icon" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M2 21l20-9L2 3v7l14 2-14 2z"/></svg></button>
      </div>
    </div>
  `;
  document.getElementById('reels-viewer')?.appendChild(panel);

  const list = document.getElementById('reel-comments-list');
  if (list) list.scrollTop = list.scrollHeight;

  document.getElementById('reel-comment-input')?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      postReelComment(postId);
    }
  });

  const reelImgInput = document.getElementById('reel-comment-image-input');
  if (reelImgInput) {
    reelImgInput.onchange = e => {
      const f = e.target.files?.[0];
      if (!f) return;
      _pendingReelCommentImageFile = f;
      const prev = document.getElementById('reel-comment-img-preview');
      if (prev) {
        prev.innerHTML = `<div class="comment-img-preview-item"><img src="${URL.createObjectURL(f)}"><button type="button" class="image-preview-remove" onclick="clearReelCommentImage()">&times;</button></div>`;
        prev.style.display = 'block';
      }
    };
  }
}

function closeReelComments() {
  const panel = document.getElementById('reel-comments-panel');
  if (panel) panel.remove();
}

async function postReelComment(postId) {
  if (_sendingReelComment) return; // Prevent double-sends
  _sendingReelComment = true;
  const input = document.getElementById('reel-comment-input');
  const sendBtn = input?.parentElement?.querySelector('.send-btn');
  if (sendBtn) sendBtn.disabled = true;
  
  const text = input?.value.trim();
  const imgFile = _pendingReelCommentImageFile;
  if (!text && !imgFile) { 
    _sendingReelComment = false;
    if (sendBtn) sendBtn.disabled = false;
    return; 
  }
  input.value = '';
  const replyTo = _reelCommentReplyTo ? _reelCommentReplyTo.id : null;
  const fileSig = imgFile ? `${imgFile.name}:${imgFile.size}:${imgFile.lastModified}` : 'noimg';
  const dedupeKey = `${postId}|${replyTo || 'root'}|${text || ''}|${fileSig}`;
  if (_lastReelCommentSubmit.key === dedupeKey && (Date.now() - _lastReelCommentSubmit.at) < 8000) {
    _sendingReelComment = false;
    if (sendBtn) sendBtn.disabled = false;
    return;
  }
  _lastReelCommentSubmit = { key: dedupeKey, at: Date.now() };
  try {
    let imageURL = null;
    if (imgFile) imageURL = await uploadToR2(imgFile, 'comments');
    await db.collection('posts').doc(postId).collection('comments').add({
      text: text || '', imageURL,
      authorId: state.user.uid,
      authorName: state.profile.displayName,
      authorPhoto: state.profile.photoURL || null,
      likes: [],
      replyTo: replyTo || null,
      createdAt: FieldVal.serverTimestamp()
    });
    await db.collection('posts').doc(postId).update({ commentsCount: FieldVal.increment(1) });
    _pendingReelCommentImageFile = null;
    _reelCommentReplyTo = null;
    clearReelCommentImage();
    clearReelCommentReply();
    openReelComments(postId);
  } catch (e) { console.error(e); toast('Failed'); }
  finally { _sendingReelComment = false; if (sendBtn) sendBtn.disabled = false; }
}

function clearReelCommentImage() {
  _pendingReelCommentImageFile = null;
  const prev = document.getElementById('reel-comment-img-preview');
  if (prev) { prev.style.display = 'none'; prev.innerHTML = ''; }
  const inp = document.getElementById('reel-comment-image-input');
  if (inp) inp.value = '';
}

function setReelCommentReply(commentId, authorName) {
  _reelCommentReplyTo = { id: commentId, authorName };
  const ind = document.getElementById('reel-comment-reply-indicator');
  const label = document.getElementById('reel-comment-reply-label');
  if (ind) ind.style.display = 'flex';
  if (label) label.innerHTML = `<span style="font-weight:600">↩ Replying to:</span> ${esc(authorName)}`;
  document.getElementById('reel-comment-input')?.focus();
}

function clearReelCommentReply() {
  _reelCommentReplyTo = null;
  const ind = document.getElementById('reel-comment-reply-indicator');
  if (ind) ind.style.display = 'none';
}

async function toggleReelCommentLike(commentId, postId) {
  try {
    const ref = db.collection('posts').doc(postId).collection('comments').doc(commentId);
    const doc = await ref.get();
    if (!doc.exists) return;
    const likes = doc.data().likes || [];
    if (likes.includes(state.user.uid)) {
      await ref.update({ likes: FieldVal.arrayRemove(state.user.uid) });
    } else {
      await ref.update({ likes: FieldVal.arrayUnion(state.user.uid) });
    }
    openReelComments(postId);
  } catch (e) { console.error(e); }
}

async function reelLike(pid, btn) {
  try {
    const ref = db.collection('posts').doc(pid);
    const doc = await ref.get(); if (!doc.exists) return;
    const data = doc.data();
    const likes = data.likes || [];
    const svgEl = btn.querySelector('svg');
    const spanEl = btn.querySelector('span');
    if (likes.includes(state.user.uid)) {
      await ref.update({ likes: FieldVal.arrayRemove(state.user.uid) });
      btn.classList.remove('liked');
      svgEl.setAttribute('fill', 'none');
      svgEl.setAttribute('stroke', '#fff');
      spanEl.textContent = likes.length - 1 || '';
    } else {
      await ref.update({ likes: FieldVal.arrayUnion(state.user.uid) });
      btn.classList.add('liked');
      svgEl.setAttribute('fill', '#ff4757');
      svgEl.setAttribute('stroke', '#ff4757');
      spanEl.textContent = likes.length + 1;
      addNotification(data.authorId, 'like', 'liked your reel', { postId: pid });
    }
  } catch (e) { console.error(e); }
}

// ─── Auto-play videos on scroll in feed ─────────
function setupFeedVideoAutoplay() {
  const feedEl = document.getElementById('feed-posts');
  if (!feedEl) return;
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const vid = entry.target.querySelector('video');
      if (!vid) return;
      if (entry.isIntersecting && entry.intersectionRatio >= 0.6) {
        vid.play().catch(() => {});
      } else {
        vid.pause();
      }
    });
  }, { threshold: [0.6] });
  feedEl.querySelectorAll('.unino-player').forEach(p => observer.observe(p));
}

// ─── Like ────────────────────────────────────────
async function toggleLike(pid) {
  const ref = db.collection('posts').doc(pid);
  window._lastLikedPost = pid;
  // Optimistic UI: update DOM instantly before Firestore round-trip
  const postEl = document.getElementById('post-' + pid);
  if (postEl) {
    const likeBtn = postEl.querySelector('.post-action.liked, .post-action:first-child');
    if (likeBtn) {
      const isLiked = likeBtn.classList.contains('liked');
      likeBtn.classList.toggle('liked', !isLiked);
      const svg = likeBtn.querySelector('svg');
      if (svg) { svg.setAttribute('fill', !isLiked ? 'var(--red)' : 'none'); svg.setAttribute('stroke', !isLiked ? 'var(--red)' : 'currentColor'); }
    }
  }
  try {
    const doc = await ref.get(); if (!doc.exists) return;
    const data = doc.data();
    const likes = data.likes || [];
    if (likes.includes(state.user.uid)) {
      await ref.update({ likes: FieldVal.arrayRemove(state.user.uid) });
    } else {
      await ref.update({ likes: FieldVal.arrayUnion(state.user.uid) });
      addNotification(data.authorId, 'like', 'liked your post', { postId: pid });
    }
  } catch (e) { console.error(e); }
}

// ─── Comments with Replies ────────────────────────────────────
let _commentReplyTo = null; // { id, authorName } or null
let _sendingComment = false;

async function toggleCommentLike(cid, pid) {
  try {
    const ref = db.collection('posts').doc(pid).collection('comments').doc(cid);
    const doc = await ref.get();
    if (!doc.exists) return;
    const d = doc.data();
    const likes = d.likes || [];
    if (likes.includes(state.user.uid)) {
      await ref.update({ likes: FieldVal.arrayRemove(state.user.uid) });
    } else {
      await ref.update({ likes: FieldVal.arrayUnion(state.user.uid) });
    }
    openComments(pid); // Refresh
  } catch (e) { console.error(e); }
}

async function openComments(postId) {
  let postData = null;
  let comments = [];
  try {
    const postDoc = await db.collection('posts').doc(postId).get();
    postData = postDoc.exists ? postDoc.data() : null;
    const snap = await db.collection('posts').doc(postId).collection('comments').limit(100).get();
    comments = snap.docs.map(d => ({ id: d.id, ...d.data() }));
  } catch (e) { console.error(e); }

  // Process likes
  comments.forEach(c => { c.likeCount = (c.likes || []).length; });

  const topLevel = comments.filter(c => !c.replyTo);
  const replies = comments.filter(c => c.replyTo);
  
  // Sort: Top (likes) -> Newest
  topLevel.sort((a,b) => b.likeCount - a.likeCount || (b.createdAt?.seconds||0) - (a.createdAt?.seconds||0));
  // Replies: Chronological
  replies.sort((a,b) => (a.createdAt?.seconds||0) - (b.createdAt?.seconds||0));

  const replyMap = {};
  replies.forEach(r => {
    if (!replyMap[r.replyTo]) replyMap[r.replyTo] = [];
    replyMap[r.replyTo].push(r);
  });
  const commentById = {};
  comments.forEach(c => { commentById[c.id] = c; });

  _commentReplyTo = null;
  _pendingCommentImageFile = null;
  const forceAnon = !!postData?.isAnonymous && postData?.authorId === state.user.uid;
  const supportsAnonChoice = !!postData?.isAnonymous && postData?.authorId !== state.user.uid;
  _commentAnonChoice = forceAnon ? true : (supportsAnonChoice ? true : false);

  function renderComment(c, isReply = false) {
     const liked = (c.likes || []).includes(state.user.uid);
     const cReplies = replyMap[c.id] || [];
      const hiddenIdentity = !!c.isAnonymous || (!!postData?.isAnonymous && c.authorId === postData.authorId);
      const displayName = hiddenIdentity ? 'Anonymous' : c.authorName;
      const displayPhoto = hiddenIdentity ? null : c.authorPhoto;
      const target = c.replyTo ? commentById[c.replyTo] : null;
      const targetHidden = !!target && (!!target.isAnonymous || (!!postData?.isAnonymous && target.authorId === postData.authorId));
      const targetDisplayName = target ? (targetHidden ? 'Anonymous' : (target.authorName || 'User')) : '';
      const fromLabel = c.authorId === state.user.uid ? 'me' : displayName;
      const toLabel = target ? (target.authorId === state.user.uid ? 'me' : targetDisplayName) : '';
     
     return `
      <div class="comment-item ${isReply ? 'reply-item' : ''}" id="c-${c.id}">
        <div class="comment-avatar-col">
          ${hiddenIdentity ? `<div class="avatar-sm anon-avatar">👻</div>` : avatar(displayName, displayPhoto, 'avatar-sm')}
        </div>
        <div class="comment-content-col">
           <div class="comment-bubble enhanced">
              <div class="comment-header">
                  <span class="comment-author" ${hiddenIdentity && c.authorId !== state.user.uid ? `onclick="openAnonPostActions('${c.authorId}')" style="cursor:pointer"` : hiddenIdentity ? '' : `onclick="openProfile('${c.authorId}')"`}>${esc(displayName)}</span>
              </div>
                ${target ? `<div class="comment-reply-to">${esc(fromLabel)} &gt; ${esc(toLabel)}</div>` : ''}
              <div class="comment-text">${esc(c.text)}</div>
              ${c.imageURL ? `<img src="${c.imageURL}" class="comment-inline-image" onclick="viewImage('${c.imageURL}')">` : ''}
           </div>
           <div class="comment-actions-row">
               <span class="comment-time">${timeAgo(c.createdAt)}</span>
               <button class="c-act ${liked?'liked':''}" onclick="toggleCommentLike('${c.id}','${postId}')">
                  ${liked ? 'Like' : 'Like'} ${c.likeCount > 0 ? c.likeCount : ''}
               </button>
            <button class="c-act" onclick="setCommentReply('${c.id}','${esc(displayName)}')">Reply</button>
           </div>
           ${cReplies.length ? `<div class="comment-replies">
               ${cReplies.map(r => renderComment(r, true)).join('')}
           </div>` : ''}
        </div>
      </div>`;
  }

  function renderCommentTree() {
    if (!comments.length) return '<p class="empty-msg" style="text-align:center;padding:20px;color:var(--text-tertiary)">No comments. be the first.</p>';
    return topLevel.map(c => renderComment(c)).join('');
  }

  openModal(`
    <div class="modal-header"><h2>Comments</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body comment-modal-body" style="display:flex;flex-direction:column;height:72vh;padding:0">
      <div id="comments-container" class="comments-scroll" style="flex:1;overflow-y:auto;padding:16px 16px 8px">
        ${renderCommentTree()}
      </div>
      <div id="comment-reply-indicator" class="reply-indicator" style="display:none">
        <span id="comment-reply-label"></span>
        <button onclick="clearCommentReply()">&times;</button>
      </div>
      <div class="comment-input-wrap modern" style="position:sticky;bottom:0;background:var(--bg-secondary);padding:10px 14px;border-top:1px solid var(--border);flex-shrink:0">
        ${postData?.isAnonymous ? `<div style="display:flex;align-items:flex-start;gap:8px;font-size:12px;color:var(--text-secondary);margin-bottom:8px">
          <input type="checkbox" id="comment-anon-toggle" ${_commentAnonChoice ? 'checked' : ''} ${forceAnon ? 'disabled' : ''} onchange="setCommentAnonChoice(this.checked)" style="margin-top:2px;flex-shrink:0">
          <span>${forceAnon ? 'Your comments stay anonymous on your anonymous post' : 'Comment anonymously on this anonymous post'}</span>
        </div>` : ''}
        <div id="comment-img-preview" class="comment-img-preview" style="display:none"></div>
        <div class="comment-compose-row compact">
          <label class="add-photo-btn comment-attach-btn" title="Add sticker/image">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
            <input type="file" hidden accept="image/*" id="comment-image-input">
          </label>
          <textarea id="comment-input" placeholder="Write a comment..." autocomplete="off"></textarea>
          <button class="send-btn" onclick="postComment('${postId}')"><svg class="send-icon" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M2 21l20-9L2 3v7l14 2-14 2z"/></svg></button>
        </div>
      </div>
    </div>
  `);

  const cInput = $('#comment-input');
  if (cInput) {
    cInput.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        postComment(postId);
      }
    });
  }
  const cImgInput = $('#comment-image-input');
  if (cImgInput) {
    cImgInput.onchange = e => {
      const f = e.target.files?.[0];
      if (!f) return;
      _pendingCommentImageFile = f;
      const prev = $('#comment-img-preview');
      if (prev) {
        prev.innerHTML = `<div class="comment-img-preview-item"><img src="${URL.createObjectURL(f)}"><button type="button" class="image-preview-remove" onclick="clearCommentImage()">&times;</button></div>`;
        prev.style.display = 'block';
      }
    };
  }
}

function setCommentAnonChoice(next) {
  _commentAnonChoice = !!next;
}

function setCommentReply(commentId, authorName) {
  _commentReplyTo = { id: commentId, authorName };
  const ind = $('#comment-reply-indicator');
  const label = $('#comment-reply-label');
  if (ind) { ind.style.display = 'flex'; }
  if (label) { label.innerHTML = `<span style="font-weight:600">↩ Replying to:</span> ${esc(authorName)}`; }
  $('#comment-input')?.focus();
}

function clearCommentReply() {
  _commentReplyTo = null;
  const ind = $('#comment-reply-indicator');
  if (ind) ind.style.display = 'none';
}

async function postComment(postId) {
  if (_sendingComment) return; // Prevent double-sends
  _sendingComment = true;
  const input = $('#comment-input');
  const sendBtn = input?.parentElement?.querySelector('.send-btn');
  if (sendBtn) sendBtn.disabled = true;
  
  const text = input?.value.trim();
  const imgFile = _pendingCommentImageFile;
  if (!text && !imgFile) { 
    _sendingComment = false;
    if (sendBtn) sendBtn.disabled = false;
    return; 
  }
  input.value = '';
  const replyTo = _commentReplyTo ? _commentReplyTo.id : null;
  const fileSig = imgFile ? `${imgFile.name}:${imgFile.size}:${imgFile.lastModified}` : 'noimg';
  const dedupeKey = `${postId}|${replyTo || 'root'}|${text || ''}|${fileSig}`;
  if (_lastFeedCommentSubmit.key === dedupeKey && (Date.now() - _lastFeedCommentSubmit.at) < 8000) {
    _sendingComment = false;
    if (sendBtn) sendBtn.disabled = false;
    return;
  }
  _lastFeedCommentSubmit = { key: dedupeKey, at: Date.now() };
  try {
    const pDoc = await db.collection('posts').doc(postId).get();
    const postData = pDoc.exists ? pDoc.data() : null;
    const isAnonThread = !!postData?.isAnonymous;
    const forceAnon = isAnonThread && postData?.authorId === state.user.uid;
    const commentAnon = forceAnon ? true : (isAnonThread ? !!_commentAnonChoice : false);
    let imageURL = null;
    if (imgFile) imageURL = await uploadToR2(imgFile, 'comments');
    await db.collection('posts').doc(postId).collection('comments').add({
      text: text || '', imageURL,
      authorId: state.user.uid, authorName: commentAnon ? 'Anonymous' : state.profile.displayName,
      authorPhoto: commentAnon ? null : (state.profile.photoURL || null), isAnonymous: commentAnon,
      likes: [],
      replyTo: replyTo || null,
      createdAt: FieldVal.serverTimestamp()
    });
    await db.collection('posts').doc(postId).update({ commentsCount: FieldVal.increment(1) });
    
    if (postData) addNotification(postData.authorId, 'comment', 'commented on your post', { postId });

    // Reopen to show the new comment
    _pendingCommentImageFile = null;
    _commentReplyTo = null;
    clearCommentImage();
    clearCommentReply();
    openComments(postId);
  } catch (e) { console.error(e); toast('Failed'); }
  finally { _sendingComment = false; if (sendBtn) sendBtn.disabled = false; }
}

function clearCommentImage() {
  _pendingCommentImageFile = null;
  const prev = $('#comment-img-preview');
  if (prev) { prev.style.display = 'none'; prev.innerHTML = ''; }
  const inp = $('#comment-image-input');
  if (inp) inp.value = '';
}

// ─── Image Viewer ────────────────────────────────
function viewImage(url) { const v = $('#img-view'); if (!v) return; $('#img-full').src = url; v.style.display = 'flex'; }

// ─── Create Post ─────────────────────────────────
function openCreateModal() {
  let pendingFiles = [];
  let pendingIsVideo = false;
  let createTab = 'post'; // 'post' or 'event'
  window._eventFiles = [];

  const renderCreateInner = () => {
    const mi = $('#modal-inner');
    if (!mi) return;
    mi.innerHTML = `
    <div class="modal-header"><h2>Create</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="create-tabs">
      <button class="create-tab ${createTab === 'post' ? 'active' : ''}" data-ct="post">📝 Post</button>
      <button class="create-tab ${createTab === 'event' ? 'active' : ''}" data-ct="event">📅 Event</button>
    </div>
    <div class="modal-body">
      ${createTab === 'post' ? `
      <div style="display:flex;gap:12px;margin-bottom:16px">
        ${avatar(state.profile.displayName, state.profile.photoURL, 'avatar-md')}
        <div>
          <div style="font-weight:600">${esc(state.profile.displayName)}</div>
          <div style="font-size:12px;color:var(--text-secondary)">Posting to ${esc(state.profile.university || 'NWU')}</div>
        </div>
      </div>
      <textarea id="create-text" placeholder="What's on your mind?" style="width:100%;min-height:100px;border:none;background:transparent;color:var(--text-primary);font-size:16px;resize:none;outline:none"></textarea>
      <div class="create-post-hint">Use #MATH301-style tags inside your post so students in that module can discover it. Choose Anonymous if you want the post hidden from your identity.</div>
      <div id="create-preview" class="media-preview" style="display:none">
        <div id="create-preview-content" class="collage-preview-grid"></div>
        <button class="image-preview-remove" onclick="document.getElementById('create-preview').style.display='none';window._createPendingFiles=[]">&times;</button>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;border-top:1px solid var(--border);padding-top:12px;margin-top:12px">
        <div style="display:flex;align-items:center;gap:8px">
          <label class="add-photo-btn" title="Photos"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg><input type="file" hidden accept="image/*" id="create-file" multiple></label>
          <label class="add-photo-btn" title="Video"><svg width="22" height="22" viewBox="0 0 24 24" stroke="var(--accent)" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7" fill="var(--accent)"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2" fill="none"/></svg><input type="file" hidden accept="video/*" id="create-video-file"></label>
          <select id="create-visibility" style="padding:6px 10px;border-radius:100px;border:1px solid var(--border);background:var(--bg-tertiary);color:var(--text-primary);font-size:12px;font-weight:600">
            <option value="public">🌍 Public</option>
            <option value="friends">👫 Friends</option>
            <option value="anonymous">👻 Anonymous</option>
          </select>
        </div>
        <button class="btn-primary" id="create-submit" style="padding:10px 28px">Post</button>
      </div>
      ` : `
      <div class="form-group"><label>Event Title</label><input type="text" id="ev-title" placeholder="e.g. Study Session, Party, Workshop"></div>
      <div class="form-group"><label>Location</label><input type="text" id="ev-location-text" placeholder="e.g. Library 2nd Floor, Res Common Room, Mooi River Mall"></div>
      <div style="display:flex;gap:8px">
        <div class="form-group" style="flex:1"><label>Date</label><input type="date" id="ev-date"></div>
        <div class="form-group" style="flex:1"><label>Time</label><input type="time" id="ev-time"></div>
      </div>
      <div class="form-group"><label>Description (optional)</label><textarea id="ev-desc" placeholder="What's happening?" style="resize:none;height:60px"></textarea></div>
      <div class="form-group"><label>Event Photos (up to 4)</label><input type="file" accept="image/*" id="ev-file" multiple></div>
      <div id="ev-img-preview" class="ev-img-preview-grid"></div>
      <button class="btn-primary btn-full" id="ev-create-btn">Create Event</button>
      `}
    </div>`;

    // Wire tab switching
    $$('.create-tab').forEach(tab => {
      tab.onclick = () => {
        createTab = tab.dataset.ct;
        renderCreateInner();
      };
    });

    if (createTab === 'post') wirePostTab();
    else wireEventTab();
  };

  const wirePostTab = () => {
    const showPreviews = () => {
      const pc = $('#create-preview-content');
      if (!pendingFiles.length) { $('#create-preview').style.display = 'none'; return; }
      if (pendingIsVideo) {
        pc.innerHTML = `<video src="${localPreview(pendingFiles[0])}" style="width:100%;max-height:200px;border-radius:var(--radius)" controls></video>`;
      } else {
        const count = pendingFiles.length;
        pc.className = `collage-preview-grid collage-${Math.min(count, 4)}`;
        pc.innerHTML = pendingFiles.slice(0, 4).map((f, i) =>
          `<div class="collage-preview-item${count > 4 && i === 3 ? ' collage-more' : ''}">
            <img src="${localPreview(f)}" style="width:100%;height:100%;object-fit:cover;border-radius:4px">
            ${count > 4 && i === 3 ? `<div class="collage-more-overlay">+${count - 4}</div>` : ''}
          </div>`
        ).join('');
      }
      $('#create-preview').style.display = 'block';
    };
    if ($('#create-file')) $('#create-file').onchange = e => {
      if (e.target.files.length) {
        pendingFiles = [...pendingFiles, ...Array.from(e.target.files)];
        pendingIsVideo = false;
        showPreviews();
      }
    };
    if ($('#create-video-file')) $('#create-video-file').onchange = e => {
      if (e.target.files[0]) {
        pendingFiles = [e.target.files[0]];
        pendingIsVideo = true;
        showPreviews();
      }
    };
    if ($('#create-submit')) $('#create-submit').onclick = async () => {
      const text = $('#create-text').value.trim();
      const moduleTags = extractModuleTags(text);
      const hashTags = extractHashTags(text);
      if (!text && !pendingFiles.length) return toast('Post cannot be empty');
      const visibility = $('#create-visibility')?.value || 'public';
      const isAnon = visibility === 'anonymous';
      closeModal(); toast('Uploading...');
      try {
        let mediaURL = null, mediaType = 'text', imageURLs = null;
        if (pendingFiles.length && pendingIsVideo) {
          mediaURL = await uploadToR2(pendingFiles[0], 'videos');
          mediaType = 'video';
        } else if (pendingFiles.length === 1) {
          mediaURL = await uploadToR2(pendingFiles[0], 'images');
          mediaType = 'image';
        } else if (pendingFiles.length > 1) {
          imageURLs = [];
          for (const f of pendingFiles) { imageURLs.push(await uploadToR2(f, 'images')); }
          mediaURL = imageURLs[0];
          mediaType = 'collage';
        }
        const postRef = await db.collection('posts').add({
          content: text,
          imageURL: mediaType === 'image' || mediaType === 'collage' ? mediaURL : null,
          imageURLs: imageURLs || null,
          videoURL: mediaType === 'video' ? mediaURL : null,
          mediaType,
          authorId: state.user.uid,
          authorName: isAnon ? 'Anonymous' : state.profile.displayName,
          authorPhoto: isAnon ? null : (state.profile.photoURL || null),
          authorUni: state.profile.university || '',
          moduleTags,
          hashTags,
          isAnonymous: isAnon || false,
          visibility: isAnon ? 'public' : visibility,
          createdAt: FieldVal.serverTimestamp(), likes: [], commentsCount: 0
        });
        if (moduleTags.length) notifyRelevantModuleUsers(moduleTags, text, postRef.id);
        toast('Posted!');
      } catch (e) { toast('Failed'); console.error(e); }
    };
  };

  const wireEventTab = () => {
    if ($('#ev-file')) $('#ev-file').onchange = e => {
      const files = Array.from(e.target.files).slice(0, 4);
      window._eventFiles = files;
      const previewEl = $('#ev-img-preview');
      if (previewEl && files.length) {
        previewEl.innerHTML = files.map((f, i) => `<div class="ev-img-thumb"><img src="${URL.createObjectURL(f)}"><button type="button" class="ev-img-remove" onclick="removeEventImage(${i})">&times;</button></div>`).join('');
      }
    };
    if ($('#ev-create-btn')) $('#ev-create-btn').onclick = async () => {
      const title = $('#ev-title')?.value.trim();
      const locationText = $('#ev-location-text')?.value.trim() || '';
      const date = $('#ev-date')?.value;
      const time = $('#ev-time')?.value || '';
      const desc = $('#ev-desc')?.value.trim() || '';
      if (!title || !date) return toast('Title and date required');
      const filesToUpload = [...(window._eventFiles || [])];
      window._eventFiles = [];
      closeModal(); toast('Creating event...');
      const gradients = ['linear-gradient(135deg,#6C5CE7,#A855F7)','linear-gradient(135deg,#7C3AED,#C084FC)','linear-gradient(135deg,#8B5CF6,#D946EF)','linear-gradient(135deg,#6366F1,#818CF8)','linear-gradient(135deg,#D946EF,#E879F9)'];
      try {
        let imageURLs = [];
        for (const f of filesToUpload) {
          const url = await uploadToR2(f, 'events');
          if (url) imageURLs.push(url);
        }
        await db.collection('events').add({
          title, location: locationText, date, time, description: desc,
          imageURLs,
          gradient: gradients[Math.floor(Math.random() * gradients.length)],
          createdBy: state.user.uid,
          creatorName: state.profile.displayName,
          going: [state.user.uid],
          createdAt: FieldVal.serverTimestamp()
        });
        toast('Event created!');
        await loadCampusEvents();
        if (exploreView === 'radar') renderRadarView();
      } catch (e) { toast('Failed'); console.error(e); }
    };
  };

  openModal('<div></div>');
  renderCreateInner();
}

// ══════════════════════════════════════════════════
//  EXPLORE — Radar + List with Module Matching
// ══════════════════════════════════════════════════
let exploreView = 'radar';
let allExploreUsers = [];

function renderExplore() {
  const c = $('#content');
  c.innerHTML = `
    <div class="explore-page">
      <div class="explore-toggle">
        <button class="explore-toggle-btn active" data-v="radar">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
          Radar
        </button>
        <button class="explore-toggle-btn" data-v="list">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>
          List
        </button>
      </div>
      <div id="explore-body">
        <div style="padding:40px;text-align:center"><span class="inline-spinner" style="width:28px;height:28px;color:var(--accent)"></span></div>
      </div>
    </div>
  `;

  $$('.explore-toggle-btn').forEach(btn => {
    btn.onclick = () => {
      $$('.explore-toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      exploreView = btn.dataset.v;
      renderExploreView();
    };
  });
  loadExploreUsers();
}

async function loadExploreUsers() {
  try {
    // Load events in parallel with users
    const [snap] = await Promise.all([
      db.collection('users').limit(50).get(),
      loadCampusEvents()
    ]);
    const myMajor = state.profile.major || '';
    const myModules = normalizeModules(state.profile.modules || []);
    const myAddress = state.profile.address || '';

    allExploreUsers = snap.docs
      .map(d => ({ id: d.id, ...d.data() }))
      .filter(u => u.id !== state.user.uid)
      .map(u => {
        const uModules = normalizeModules(u.modules || []);
        const shared = myModules.filter(m => uModules.includes(m));
        const nearbyScore = addressMatchScore(myAddress, u.address || '');
        let proximity = 'far';
        if (shared.length > 0) proximity = 'module';
        else if (nearbyScore > 0) proximity = 'nearby';
        else if (u.major === myMajor) proximity = 'course';
        return { ...u, sharedModules: shared, proximity, nearbyScore };
      });
    renderExploreView();
  } catch (e) {
    console.error(e);
    const body = $('#explore-body');
    if (body) body.innerHTML = '<div class="empty-state"><h3>Could not load students</h3></div>';
  }
}

function renderExploreView() {
  if (exploreView === 'radar') renderRadarView();
  else renderListView();
}

function renderRadarView() {
  const body = $('#explore-body'); if (!body) return;
  body.innerHTML = `
    <div style="padding:0 16px 12px">
      <div class="search-bar radar-search-wrap">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <input type="text" id="radar-search" placeholder="Search people or location..." value="${esc(window._radarSearchQuery || '')}">
      </div>
      <div id="radar-suggestions" class="radar-suggestions" style="display:none"></div>
    </div>
    <div class="radar-map-wrap">
      <div id="radar-map" style="width:100%;height:320px;z-index:0"></div>
      <div class="radar-overlay">
        <div class="radar-ring r3"></div>
        <div class="radar-ring r2"></div>
        <div class="radar-ring r1"></div>
        <div class="radar-sweep-anim"></div>
      </div>
    </div>
    <div id="radar-dynamic"></div>
  `;

  const applyRadarFilter = (queryRaw = '') => {
    const searchQuery = (queryRaw || '').trim().toLowerCase();
    let filteredUsers = allExploreUsers;
    if (searchQuery) {
      filteredUsers = allExploreUsers.filter(u =>
        (u.displayName || '').toLowerCase().includes(searchQuery) ||
        (u.address || '').toLowerCase().includes(searchQuery) ||
        (u.major || '').toLowerCase().includes(searchQuery) ||
        (u.modules || []).some(m => m.toLowerCase().includes(searchQuery))
      );
    }

    const moduleUsers = filteredUsers.filter(u => u.proximity === 'module');
    const nearbyUsers = filteredUsers.filter(u => u.proximity === 'nearby');
    const courseUsers = filteredUsers.filter(u => u.proximity === 'course');
    const otherUsers = filteredUsers.filter(u => u.proximity === 'far');

    const dynamic = $('#radar-dynamic');
    if (dynamic) {
      dynamic.innerHTML = `
        <div class="radar-legend" style="padding:0 16px">
          <span><span class="legend-dot module"></span> Shared modules (${moduleUsers.length})</span>
          <span><span class="legend-dot campus"></span> Nearby area (${nearbyUsers.length})</span>
          <span><span class="legend-dot campus"></span> Same course (${courseUsers.length})</span>
          <span><span class="legend-dot far"></span> Other (${otherUsers.length})</span>
        </div>

        ${moduleUsers.length ? `
        <div class="proximity-section">
          <div class="proximity-header"><h3>🔗 Shared Modules</h3><span class="proximity-count">${moduleUsers.length}</span></div>
          <div class="proximity-scroll">${moduleUsers.map(u => proximityCard(u)).join('')}</div>
        </div>` : ''}

        <div class="proximity-section">
          <div class="proximity-header"><h3>📍 Nearby Area</h3><span class="proximity-count">${nearbyUsers.length}</span></div>
          <div class="proximity-scroll">
            ${nearbyUsers.length ? nearbyUsers.map(u => proximityCard(u)).join('')
              : '<p style="padding:12px;color:var(--text-tertiary);font-size:13px">No one found yet</p>'}
          </div>
        </div>

        ${courseUsers.length ? `
        <div class="proximity-section">
          <div class="proximity-header"><h3>📚 Same Course</h3><span class="proximity-count">${courseUsers.length}</span></div>
          <div class="proximity-scroll">${courseUsers.map(u => proximityCard(u)).join('')}</div>
        </div>` : ''}

        ${otherUsers.length ? `
        <div class="proximity-section">
          <div class="proximity-header"><h3>🎓 Other Students</h3><span class="proximity-count">${otherUsers.length}</span></div>
          <div class="proximity-scroll">${otherUsers.slice(0, 12).map(u => proximityCard(u)).join('')}</div>
        </div>` : ''}

        <div class="proximity-section">
          <div class="proximity-header"><h3>📅 Events</h3><button class="btn-primary btn-sm" onclick="openCreateEvent()">+ Event</button></div>
          ${allCampusEvents.length ? `<div class="proximity-scroll">${allCampusEvents.slice(0, 10).map(ev => {
            const loc = CAMPUS_LOCATIONS.find(l => l.id === ev.location);
            const thumb = (ev.imageURLs && ev.imageURLs.length) ? ev.imageURLs[0] : null;
            const grad = ev.gradient || 'linear-gradient(135deg,#6C5CE7,#A855F7)';
            return `<div class="event-scroll-card" onclick="openEventDetail('${ev.id || ''}')">
              ${thumb ? `<img class="event-scroll-thumb" src="${thumb}">` : `<div class="event-scroll-icon" style="background:${grad}">${ev.emoji || '📅'}</div>`}
              <div class="event-scroll-title">${esc(ev.title)}</div>
              <div class="event-scroll-meta">${loc ? loc.emoji + ' ' + loc.name : esc(ev.location || '')}</div>
            </div>`;
          }).join('')}</div>` : '<p style="padding:12px 16px;color:var(--text-tertiary);font-size:13px">No events yet — create one!</p>'}
        </div>
      `;
    }

    const eventsByLoc = {};
    allCampusEvents.forEach(ev => {
      if (!eventsByLoc[ev.location]) eventsByLoc[ev.location] = [];
      eventsByLoc[ev.location].push(ev);
    });
    renderRadarMap(moduleUsers, nearbyUsers, courseUsers, otherUsers, eventsByLoc);
  };

  const renderRadarSuggestions = (queryRaw = '') => {
    const box = $('#radar-suggestions');
    if (!box) return;
    const q = (queryRaw || '').trim().toLowerCase();
    if (!q) {
      box.style.display = 'none';
      box.innerHTML = '';
      return;
    }

    const peopleHits = allExploreUsers
      .filter(u => (u.displayName || '').toLowerCase().includes(q))
      .slice(0, 4)
      .map(u => ({ type: 'person', uid: u.id, label: u.displayName || 'User', sub: u.major || u.address || '' }));

    const addressPool = [
      ...CAMPUS_LOCATIONS.map(l => l.name),
      ...allExploreUsers.map(u => u.address || '').filter(Boolean)
    ];
    const addressHits = Array.from(new Set(addressPool))
      .filter(a => a.toLowerCase().includes(q))
      .slice(0, 4)
      .map(a => ({ type: 'address', label: a, sub: 'Address' }));

    const hits = [...peopleHits, ...addressHits].slice(0, 7);
    if (!hits.length) {
      box.style.display = 'none';
      box.innerHTML = '';
      return;
    }

    box.innerHTML = hits.map(h => {
      if (h.type === 'person') {
        const user = allExploreUsers.find(u => u.id === h.uid);
        const photo = user?.photoURL || null;
        const avatarHTML = photo ? `<img src="${photo}" alt="" class="radar-suggestion-avatar">` : `<div class="radar-suggestion-avatar">${initials(h.label)}</div>`;
        return `
          <button type="button" class="radar-suggestion-item" data-v="${esc(h.label)}" data-type="${h.type}" data-uid="${h.uid || ''}">
            ${avatarHTML}
            <div class="radar-suggestion-text">
              <span class="radar-suggestion-main">${esc(h.label)}</span>
              ${h.sub ? `<span class="radar-suggestion-sub">${esc(h.sub)}</span>` : ''}
            </div>
          </button>
        `;
      }
      return `
        <button type="button" class="radar-suggestion-item" data-v="${esc(h.label)}" data-type="${h.type}" data-uid="${h.uid || ''}">
          <span class="radar-suggestion-icon">📍</span>
          <div class="radar-suggestion-text">
            <span class="radar-suggestion-main">${esc(h.label)}</span>
            ${h.sub ? `<span class="radar-suggestion-sub">${esc(h.sub)}</span>` : ''}
          </div>
        </button>
      `;
    }).join('');
    box.style.display = 'block';
    box.querySelectorAll('.radar-suggestion-item').forEach(btn => {
      btn.onclick = () => {
        const selected = btn.getAttribute('data-v') || '';
        const kind = btn.getAttribute('data-type') || '';
        const uid = btn.getAttribute('data-uid') || '';
        if (kind === 'person' && uid) {
          box.style.display = 'none';
          openProfile(uid);
          return;
        }
        const inp = $('#radar-search');
        if (inp) inp.value = selected;
        window._radarSearchQuery = selected;
        _exploreSearchQuery = selected;
        renderRadarSuggestions(selected);
        applyRadarFilter(selected);
      };
    });
  };

  applyRadarFilter(window._radarSearchQuery || '');
  renderRadarSuggestions(window._radarSearchQuery || '');

  // Wire up live search; update cards/map without recreating the input element.
  const radarSearchInput = $('#radar-search');
  const switchToListWithQuery = (queryRaw = '') => {
    const q = (queryRaw || '').trim();
    window._radarSearchQuery = q;
    _exploreSearchQuery = q;
    exploreView = 'list';
    $$('.explore-toggle-btn').forEach(b => b.classList.toggle('active', b.dataset.v === 'list'));
    renderListView(q);
  };

  if (radarSearchInput) {
    let searchTimer = null;
    radarSearchInput.addEventListener('input', e => {
      const nextQuery = e.target.value;
      window._radarSearchQuery = nextQuery;
      _exploreSearchQuery = nextQuery;
      renderRadarSuggestions(nextQuery);
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => applyRadarFilter(nextQuery), 120);
    });
    radarSearchInput.addEventListener('keydown', e => {
      if (e.key !== 'Enter') return;
      e.preventDefault();
      switchToListWithQuery(radarSearchInput.value);
    });
    radarSearchInput.addEventListener('focus', () => renderRadarSuggestions(radarSearchInput.value));
    radarSearchInput.addEventListener('blur', () => setTimeout(() => {
      const box = $('#radar-suggestions');
      if (box) box.style.display = 'none';
    }, 120));
  }

}


function renderRadarMap(moduleUsers, nearbyUsers, courseUsers, otherUsers, eventsByLoc) {
  requestAnimationFrame(() => {
    const el = document.getElementById('radar-map');
    if (!el || typeof L === 'undefined') return;
    if (_leafletMap) { _leafletMap.remove(); _leafletMap = null; }

    _leafletMap = L.map('radar-map', { zoomControl: false }).setView([-26.6840, 27.0945], 16);
    L.control.zoom({ position: 'topright' }).addTo(_leafletMap);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; CARTO', maxZoom: 20, subdomains: 'abcd'
    }).addTo(_leafletMap);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png', {
      maxZoom: 20, subdomains: 'abcd', pane: 'overlayPane'
    }).addTo(_leafletMap);

    const myIcon = L.divIcon({
      className: 'leaflet-user-pin',
      html: `<div class="map-user-pin map-me-pin">${state.profile.photoURL ? `<img src="${state.profile.photoURL}">` : `<span>${initials(state.profile.displayName)}</span>`}</div>`,
      iconSize: [36, 36], iconAnchor: [18, 18]
    });
    L.marker([-26.6840, 27.0945], { icon: myIcon }).addTo(_leafletMap).bindPopup('<b>You</b>');

    CAMPUS_LOCATIONS.forEach(loc => {
      const evts = eventsByLoc[loc.id] || [];
      const icon = L.divIcon({
        className: 'leaflet-emoji-pin',
        html: `<div class="map-pin-wrap ${evts.length ? 'has-events' : ''}"><span class="map-pin-emoji">${loc.emoji}</span>${evts.length ? `<span class="map-pin-count">${evts.length}</span>` : ''}</div>`,
        iconSize: [36, 36], iconAnchor: [18, 36]
      });
      L.marker([loc.lat, loc.lng], { icon }).addTo(_leafletMap)
        .bindPopup(`<b>${loc.emoji} ${loc.name}</b>${evts.length ? `<br><small>${evts.length} event${evts.length > 1 ? 's' : ''}</small>` : ''}`)
        .on('click', () => openLocationDetail(loc.id));
    });

    const usersToPlot = [...moduleUsers, ...nearbyUsers, ...courseUsers, ...otherUsers.slice(0, 8)];
    usersToPlot.forEach((u, i) => {
      const baseR = u.proximity === 'module' ? 0.001 : u.proximity === 'nearby' ? 0.0018 : u.proximity === 'course' ? 0.0025 : 0.004;
      const angle = (i / usersToPlot.length) * Math.PI * 2 + (i * 0.7);
      const lat = -26.6840 + Math.cos(angle) * baseR * (0.6 + Math.random() * 0.8);
      const lng = 27.0945 + Math.sin(angle) * baseR * (0.6 + Math.random() * 0.8);
      const cls = u.proximity === 'module' ? 'pin-module' : (u.proximity === 'nearby' || u.proximity === 'course') ? 'pin-campus' : 'pin-far';
      const uIcon = L.divIcon({
        className: 'leaflet-user-pin',
        html: `<div class="map-user-pin ${cls}">${u.photoURL ? `<img src="${u.photoURL}">` : `<span>${initials(u.displayName)}</span>`}</div>`,
        iconSize: [28, 28], iconAnchor: [14, 14]
      });
      L.marker([lat, lng], { icon: uIcon }).addTo(_leafletMap)
        .bindPopup(`<b>${esc(u.displayName)}</b><br><small>${esc(u.major || '')}</small>${u.sharedModules?.length ? `<br><small>🔗 ${u.sharedModules.join(', ')}</small>` : ''}`)
        .on('click', () => openProfile(u.id));
    });

    setTimeout(() => _leafletMap?.invalidateSize(), 300);
  });
}

function renderRadarDots(users, radius, type) {
  if (!users.length) return '';
  return users.slice(0, 8).map((u, i) => {
    const angle = (i / Math.min(users.length, 8)) * Math.PI * 2 - Math.PI / 2;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    const bg = colorFor(u.displayName);
    return `<div class="radar-dot ${type}" style="transform:translate(${x}px,${y}px);background:${bg}" onclick="openProfile('${u.id}')" title="${esc(u.displayName)}">
      ${u.photoURL ? `<img src="${u.photoURL}" alt="">` : initials(u.displayName)}
    </div>`;
  }).join('');
}

function proximityCard(u) {
  const online = u.status === 'online' ? '<span class="online-dot"></span>' : '';
  const tag = u.sharedModules?.length ? `🔗 ${u.sharedModules.join(', ')}`
    : u.proximity === 'course' ? `📚 ${esc(u.major)}`
    : u.proximity === 'nearby' ? `📍 ${esc(u.address || 'Nearby')}`
    : `🎓 ${esc(u.university || '')}`;
  return `
    <div class="proximity-card" onclick="showUserPreview('${u.id}')">
      <div class="proximity-card-avatar">${avatar(u.displayName, u.photoURL, 'avatar-md')}${online}</div>
      <div class="proximity-card-name">${esc(u.displayName)}</div>
      <div class="proximity-card-meta">${tag}</div>
    </div>`;
}

async function showUserPreview(uid) {
  try {
    let user;
    if (uid === state.user.uid) { user = state.profile; }
    else {
      const doc = await db.collection('users').doc(uid).get();
      if (!doc.exists) return toast('User not found');
      user = { id: doc.id, ...doc.data() };
    }
    const isMe = uid === state.user.uid;
    const isFriend = (state.profile.friends || []).includes(uid);
    const isPending = (state.profile.sentRequests || []).includes(uid);
    const modules = (user.modules || []).slice(0, 3);
    openModal(`
      <div class="modal-body" style="text-align:center;padding:24px">
        <div style="margin-bottom:12px">${avatar(user.displayName, user.photoURL, 'avatar-xl')}</div>
        <div style="font-size:18px;font-weight:700;margin-bottom:4px">${esc(user.displayName)}</div>
        <div style="font-size:13px;color:var(--text-secondary);margin-bottom:4px">${esc(user.major || 'Student')}${user.university ? ' · ' + esc(user.university) : ''}</div>
        ${user.bio ? `<p style="font-size:13px;color:var(--text-secondary);margin-bottom:12px;line-height:1.4">${esc(user.bio)}</p>` : ''}
        ${modules.length ? `<div style="display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin-bottom:16px">${modules.map(m => `<span class="module-chip">${esc(m)}</span>`).join('')}</div>` : ''}
        <div style="display:flex;gap:8px;justify-content:center">
          ${isMe ? '' : isFriend
            ? `<button class="btn-primary" onclick="closeModal();startChat('${uid}','${esc(user.displayName)}','${user.photoURL || ''}')">Message</button>`
            : `<button class="btn-outline anon-msg-btn" onclick="closeModal();startAnonChat('${uid}','${esc(user.displayName)}','${user.photoURL || ''}', true)">👻 Anonymous</button>
               ${isPending
                 ? `<button class="btn-outline" disabled style="opacity:0.6">Pending…</button>`
                 : `<button class="btn-outline" onclick="closeModal();sendFriendRequest('${uid}','${esc(user.displayName)}','${user.photoURL || ''}')">Add Friend</button>`}`}
          <button class="btn-secondary" onclick="closeModal();openProfile('${uid}')">View Profile</button>
        </div>
      </div>
    `);
  } catch (e) { toast('Could not load user'); console.error(e); }
}

function renderListView(initialQuery = _exploreSearchQuery || window._radarSearchQuery || '') {
  const body = $('#explore-body'); if (!body) return;
  body.innerHTML = `
    <div style="padding:0 16px 16px">
      <div class="search-bar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <input type="text" id="explore-search" placeholder="Search by name, module, course..." value="${esc(initialQuery)}">
      </div>
      <div class="filter-chips">
        <span class="chip active" data-f="all">All</span>
        <span class="chip" data-f="nearby">Nearby</span>
        <span class="chip" data-f="module">Shared Modules</span>
        <span class="chip" data-f="course">Same Course</span>
      </div>
      <div class="users-grid" id="explore-grid"></div>
    </div>
  `;
  _exploreSearchQuery = initialQuery || '';
  renderExploreGrid(_exploreSearchQuery);
  let timer;
  $('#explore-search')?.addEventListener('input', e => {
    clearTimeout(timer); timer = setTimeout(() => {
      _exploreSearchQuery = e.target.value;
      renderExploreGrid(_exploreSearchQuery);
    }, 160);
  });
  $$('#explore-body .filter-chips .chip').forEach(ch => {
    ch.onclick = () => {
      $$('#explore-body .filter-chips .chip').forEach(c2 => c2.classList.remove('active'));
      ch.classList.add('active');
      renderExploreGrid($('#explore-search')?.value, ch.dataset.f);
    };
  });
}

function renderExploreGrid(query = '', filter = 'all') {
  const grid = $('#explore-grid'); if (!grid) return;
  let users = [...allExploreUsers];

  if (query) {
    const q = query.toLowerCase();
    users = users.filter(u =>
      (u.displayName || '').toLowerCase().includes(q) ||
      (u.address || '').toLowerCase().includes(q) ||
      (u.major || '').toLowerCase().includes(q) ||
      (u.university || '').toLowerCase().includes(q) ||
      (u.modules || []).some(m => m.toLowerCase().includes(q))
    );
  }
  if (filter === 'nearby') users = users.filter(u => u.proximity === 'nearby');
  else if (filter === 'module') users = users.filter(u => u.sharedModules?.length > 0);
  else if (filter === 'course') users = users.filter(u => u.major === state.profile.major);

  if (!users.length) {
    grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><h3>No matches</h3><p>Try different filters</p></div>';
    return;
  }

  grid.innerHTML = users.map(u => {
    const tag = u.sharedModules?.length ? `🔗 ${u.sharedModules.length} shared`
      : u.proximity === 'nearby' ? '📍 Nearby area'
      : u.proximity === 'course' ? '📚 Same course'
      : u.university ? `🎓 ${esc(u.university)}` : '';
    return `
      <div class="user-card" onclick="openProfile('${u.id}')">
        ${avatar(u.displayName, u.photoURL, 'avatar-lg')}
        <div class="user-card-name">${esc(u.displayName)}</div>
        <div class="user-card-uni">${esc(u.major || '')}</div>
        ${(u.modules || []).length ? `<div class="user-card-modules">${(u.modules || []).slice(0, 3).map(m => `<span class="module-chip">${esc(m)}</span>`).join('')}</div>` : ''}
        ${tag ? `<div class="user-card-distance">${tag}</div>` : ''}
      </div>`;
  }).join('');
}

// ══════════════════════════════════════════════════
//  CAMPUS MAP — Events & Locations on visual map
// ══════════════════════════════════════════════════
const CAMPUS_LOCATIONS = [
  { id: 'library', name: 'Library', emoji: '📚', x: 30, y: 20, lat: -26.6820, lng: 27.0929 },
  { id: 'main-hall', name: 'Main Hall', emoji: '🏛', x: 50, y: 15, lat: -26.6825, lng: 27.0945 },
  { id: 'student-center', name: 'Student Center', emoji: '☕', x: 70, y: 25, lat: -26.6830, lng: 27.0960 },
  { id: 'cs-building', name: 'CS Building', emoji: '💻', x: 20, y: 50, lat: -26.6840, lng: 27.0920 },
  { id: 'sports-complex', name: 'Sports Complex', emoji: '⚽', x: 80, y: 50, lat: -26.6850, lng: 27.0975 },
  { id: 'amphitheatre', name: 'Amphitheatre', emoji: '🎭', x: 45, y: 45, lat: -26.6838, lng: 27.0940 },
  { id: 'quad', name: 'The Quad', emoji: '🌳', x: 55, y: 60, lat: -26.6845, lng: 27.0950 },
  { id: 'cafeteria', name: 'Cafeteria', emoji: '🍕', x: 35, y: 70, lat: -26.6855, lng: 27.0935 },
  { id: 'res-halls', name: 'Res Halls', emoji: '🏠', x: 75, y: 75, lat: -26.6860, lng: 27.0965 },
  { id: 'lab-block', name: 'Lab Block', emoji: '🔬', x: 15, y: 35, lat: -26.6833, lng: 27.0915 },
  { id: 'admin', name: 'Admin Block', emoji: '🏢', x: 60, y: 35, lat: -26.6828, lng: 27.0955 },
  { id: 'parking', name: 'Parking', emoji: '🅿️', x: 90, y: 85, lat: -26.6868, lng: 27.0985 },
];

let allCampusEvents = [];

async function loadCampusEvents() {
  try {
    const snap = await db.collection('events').orderBy('date','asc').limit(50).get();
    allCampusEvents = snap.docs.map(d => ({ id: d.id, ...d.data() }));
  } catch (e) {
    console.error(e);
    allCampusEvents = [];
  }
}

let _leafletMap = null;

function renderCampusMapView() {
  const body = $('#explore-body'); if (!body) return;

  const eventsByLoc = {};
  allCampusEvents.forEach(ev => {
    if (!eventsByLoc[ev.location]) eventsByLoc[ev.location] = [];
    eventsByLoc[ev.location].push(ev);
  });

  body.innerHTML = `
    <div class="campus-map-container">
      <div class="campus-map-header">
        <h3>NWU Campus Map</h3>
        <button class="btn-primary btn-sm" onclick="openCreateEvent()">+ Event</button>
      </div>
      <div id="leaflet-map" style="width:100%;height:300px;border-radius:var(--radius);overflow:hidden;margin-bottom:16px;z-index:0"></div>

      <div class="campus-events-section">
        <div class="campus-events-header">
          <h3>📅 Upcoming Events</h3>
          <span style="font-size:12px;color:var(--text-tertiary)">${allCampusEvents.length} events</span>
        </div>
        ${allCampusEvents.length ? allCampusEvents.map(ev => {
          const loc = CAMPUS_LOCATIONS.find(l => l.id === ev.location);
          const goingCount = (ev.going || []).length;
          const amGoing = (ev.going || []).includes(state.user.uid);
          const grad = ev.gradient || 'linear-gradient(135deg,#6C5CE7,#A855F7)';
          return `
            <div class="campus-event-card" onclick="openEventDetail('${ev.id || ''}')">
              <div class="campus-event-icon" style="background:${grad}">${ev.emoji || '📅'}</div>
              <div class="campus-event-info">
                <div class="campus-event-title">${esc(ev.title)}</div>
                <div class="campus-event-meta">
                  📍 ${loc ? loc.name : esc(ev.location || '?')} · 🕐 ${esc(ev.date || '')} ${esc(ev.time || '')}
                </div>
                <div class="campus-event-going">
                  ${amGoing ? '<span style="color:var(--green);font-weight:700">✓ Going</span>' : ''}
                  ${goingCount ? `<span>${goingCount} going</span>` : ''}
                </div>
              </div>
            </div>`;
        }).join('') : '<div class="empty-state"><h3>No events yet</h3><p>Be the first to create one!</p></div>'}
      </div>
    </div>
  `;

  // Init Leaflet map
  requestAnimationFrame(() => initLeafletMap(eventsByLoc));
}

function initLeafletMap(eventsByLoc) {
  const el = document.getElementById('leaflet-map');
  if (!el || typeof L === 'undefined') return;

  if (_leafletMap) { _leafletMap.remove(); _leafletMap = null; }

  // NWU Potchefstroom campus center
  _leafletMap = L.map('leaflet-map', { zoomControl: false }).setView([-26.6840, 27.0945], 16);
  L.control.zoom({ position: 'topright' }).addTo(_leafletMap);

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
    maxZoom: 20,
    subdomains: 'abcd'
  }).addTo(_leafletMap);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png', {
    maxZoom: 20, subdomains: 'abcd', pane: 'overlayPane'
  }).addTo(_leafletMap);

  // Campus location pins
  CAMPUS_LOCATIONS.forEach(loc => {
    const evts = eventsByLoc[loc.id] || [];
    const hasEvents = evts.length > 0;
    const icon = L.divIcon({
      className: 'leaflet-emoji-pin',
      html: `<div class="map-pin-wrap ${hasEvents ? 'has-events' : ''}">
        <span class="map-pin-emoji">${loc.emoji}</span>
        ${hasEvents ? `<span class="map-pin-count">${evts.length}</span>` : ''}
      </div>`,
      iconSize: [36, 36],
      iconAnchor: [18, 36]
    });
    const marker = L.marker([loc.lat, loc.lng], { icon }).addTo(_leafletMap);
    marker.bindPopup(`<b>${loc.emoji} ${loc.name}</b>${hasEvents ? `<br><small>${evts.length} event${evts.length > 1 ? 's' : ''}</small>` : ''}`);
    marker.on('click', () => openLocationDetail(loc.id));
  });

  // User pins - show friends on map
  allExploreUsers.forEach(u => {
    if (!u.lat || !u.lng) return;
    const userIcon = L.divIcon({
      className: 'leaflet-user-pin',
      html: `<div class="map-user-pin">${u.photoURL ? `<img src="${u.photoURL}">` : `<span>${initials(u.displayName)}</span>`}</div>`,
      iconSize: [28, 28],
      iconAnchor: [14, 14]
    });
    const m = L.marker([u.lat, u.lng], { icon: userIcon }).addTo(_leafletMap);
    m.bindPopup(`<b>${esc(u.displayName)}</b><br><small>${esc(u.major || '')}</small>`);
    m.on('click', () => openProfile(u.id));
  });

  // Invalidate size after animation
  setTimeout(() => _leafletMap?.invalidateSize(), 300);
}

function openLocationDetail(locationId) {
  const loc = CAMPUS_LOCATIONS.find(l => l.id === locationId);
  if (!loc) return;
  const evts = allCampusEvents.filter(ev => ev.location === locationId);
  openModal(`
    <div class="modal-header"><h2>${loc.emoji} ${loc.name}</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      ${evts.length ? `
        <p style="font-size:13px;color:var(--text-secondary);margin-bottom:12px">${evts.length} event${evts.length > 1 ? 's' : ''} at this location</p>
        ${evts.map(ev => {
          const amGoing = (ev.going || []).includes(state.user.uid);
          return `
          <div style="background:var(--bg-tertiary);border:1px solid var(--border);border-radius:var(--radius);padding:12px;margin-bottom:8px">
            <div style="font-weight:700;font-size:15px">${ev.emoji || '📅'} ${esc(ev.title)}</div>
            <div style="font-size:12px;color:var(--text-secondary);margin-top:4px">🕐 ${esc(ev.date || '')} at ${esc(ev.time || '')}</div>
            <div style="font-size:12px;margin-top:4px">${(ev.going||[]).length} going ${amGoing ? '(including you ✓)' : ''}</div>
            ${ev.id ? `<button class="btn-sm ${amGoing ? 'btn-secondary' : 'btn-primary'}" style="margin-top:8px" onclick="toggleEventGoing('${ev.id}');closeModal()">${amGoing ? 'Cancel RSVP' : 'I\'m Going!'}</button>` : ''}
          </div>`;
        }).join('')}
      ` : '<div class="empty-state"><h3>No events here</h3><p>Nothing happening at ${esc(loc.name)} yet</p></div>'}
      <button class="btn-primary btn-full" style="margin-top:12px" onclick="closeModal();openCreateEvent('${locationId}')">+ Create Event Here</button>
    </div>
  `);
}

function openCreateEvent(presetLoc) {
  // Open the create modal and switch to event tab
  openCreateModal();
  // Auto-switch to event tab after a tick (modal needs to render first)
  setTimeout(() => {
    const evTab = document.querySelector('.create-tab[data-ct="event"]');
    if (evTab) evTab.click();
  }, 50);
}

function removeEventImage(idx) {
  if (window._eventFiles) {
    window._eventFiles.splice(idx, 1);
    const previewEl = $('#ev-img-preview');
    if (previewEl) {
      previewEl.innerHTML = window._eventFiles.map((f, i) => `<div class="ev-img-thumb"><img src="${URL.createObjectURL(f)}"><button type="button" class="ev-img-remove" onclick="removeEventImage(${i})">&times;</button></div>`).join('');
    }
  }
}

async function openEventDetail(eventId) {
  if (!eventId) return;
  try {
    const doc = await db.collection('events').doc(eventId).get();
    if (!doc.exists) return toast('Event not found');
    const ev = { id: doc.id, ...doc.data() };
    const loc = CAMPUS_LOCATIONS.find(l => l.id === ev.location);
    const locName = loc ? loc.name : esc(ev.location || 'TBA');
    const amGoing = (ev.going || []).includes(state.user.uid);
    const goingCount = (ev.going || []).length;
    const isCreator = ev.createdBy === state.user.uid;
    openModal(`
      <div class="modal-header"><h2>${ev.emoji || '📅'} Event</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
      <div class="modal-body">
        ${(ev.imageURLs && ev.imageURLs.length) ? `<div class="ev-detail-images ${ev.imageURLs.length === 1 ? 'single' : 'grid'}">${ev.imageURLs.map(url => `<img src="${url}" onclick="viewImage('${url}')" style="cursor:pointer">`).join('')}</div>` : ''}
        <div style="font-size:22px;font-weight:800;margin-bottom:8px">${esc(ev.title)}</div>
        <div style="display:flex;flex-wrap:wrap;gap:12px;font-size:13px;color:var(--text-secondary);margin-bottom:16px">
          <span>📍 ${locName}</span>
          <span>📅 ${esc(ev.date)}</span>
          ${ev.time ? `<span>🕐 ${esc(ev.time)}</span>` : ''}
          <span>👥 ${goingCount} going</span>
        </div>
        ${ev.description ? `<p style="font-size:14px;line-height:1.5;margin-bottom:16px">${esc(ev.description)}</p>` : ''}
        <div style="margin-bottom:16px">
          <div style="font-weight:600;font-size:13px;margin-bottom:8px">Created by</div>
          <div style="display:flex;align-items:center;gap:8px">
            ${avatar(ev.creatorName || 'System', null, 'avatar-sm')}
            <span>${esc(ev.creatorName || 'Unino')}</span>
          </div>
        </div>
        <button class="btn-primary btn-full" onclick="toggleEventGoing('${ev.id}');closeModal()">
          ${amGoing ? '✓ Going — Tap to Cancel' : "I'm Going!"}
        </button>
        ${isCreator ? `<button class="btn-outline btn-full" style="margin-top:10px;color:#ef4444;border-color:rgba(239,68,68,0.25)" onclick="deleteEvent('${ev.id}')">Delete Event</button>` : ''}
      </div>
    `);
  } catch (e) { toast('Could not load event'); console.error(e); }
}

async function deleteEvent(eventId) {
  if (!eventId) return;
  if (!window.confirm('Delete this event? This cannot be undone.')) return;
  try {
    const doc = await db.collection('events').doc(eventId).get();
    if (!doc.exists) return toast('Event not found');
    if (doc.data().createdBy !== state.user.uid) return toast('Only the creator can delete this event');
    await db.collection('events').doc(eventId).delete();
    closeModal();
    await loadCampusEvents();
    if (state.page === 'explore') renderExploreView();
    if (state.page === 'feed') loadDiscoverEvents();
    toast('Event deleted');
  } catch (e) { toast('Failed to delete event'); console.error(e); }
}

async function toggleEventGoing(eventId) {
  if (!eventId) return;
  const uid = state.user.uid;
  try {
    const doc = await db.collection('events').doc(eventId).get();
    if (!doc.exists) return;
    const going = doc.data().going || [];
    if (going.includes(uid)) {
      await db.collection('events').doc(eventId).update({ going: FieldVal.arrayRemove(uid) });
      toast('RSVP cancelled');
    } else {
      await db.collection('events').doc(eventId).update({ going: FieldVal.arrayUnion(uid) });
      toast("You're going! 🎉");
    }
    await loadCampusEvents();
    if (exploreView === 'map') renderCampusMapView();
  } catch (e) { toast('Failed'); console.error(e); }
}

// ══════════════════════════════════════════════════
//  HUSTLE (Marketplace)
// ══════════════════════════════════════════════════
function renderHustle() {
  const c = $('#content');
  c.innerHTML = `
    <div class="hustle-page">
      <div class="hustle-header"><h2>Marketplace</h2><button class="btn-primary btn-sm" onclick="openSellModal()">+ Sell</button></div>
      <div style="padding:0 16px 12px">
        <div class="search-bar">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          <input type="text" id="hustle-search" placeholder="Search items...">
        </div>
      </div>
      <div class="category-tabs">
        <span class="chip active" data-cat="all">All</span>
        <span class="chip" data-cat="books">Books</span>
        <span class="chip" data-cat="tech">Tech</span>
        <span class="chip" data-cat="notes">Notes</span>
        <span class="chip" data-cat="services">Services</span>
        <span class="chip" data-cat="other">Other</span>
      </div>
      <div class="listings-grid" id="listings-grid">
        <div style="grid-column:1/-1;text-align:center;padding:32px"><span class="inline-spinner"></span></div>
      </div>
    </div>
  `;
  loadListings();
  let searchTimer;
  $('#hustle-search')?.addEventListener('input', e => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      const activeCat = document.querySelector('.category-tabs .chip.active')?.dataset.cat || 'all';
      loadListings(activeCat, e.target.value.trim());
    }, 300);
  });
  $$('.category-tabs .chip').forEach(ch => {
    ch.onclick = () => {
      $$('.category-tabs .chip').forEach(c2 => c2.classList.remove('active'));
      ch.classList.add('active');
      loadListings(ch.dataset.cat, $('#hustle-search')?.value.trim());
    };
  });
}

async function loadListings(cat = 'all', query = '') {
  const grid = $('#listings-grid'); if (!grid) return;
  try {
    const snap = await db.collection('listings').where('status', '==', 'active').limit(50).get();
    let items = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    
    // Calculate top categories for dynamic filters
    if (!window._categoryTabsBuilt && items.length) {
      const categoryCounts = {};
      items.forEach(i => {
        const cat = (i.category || 'Other').trim();
        categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
      });
      const topCategories = Object.entries(categoryCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(e => e[0]);
      
      // Rebuild category tabs
      const catTabs = document.querySelector('.category-tabs');
      if (catTabs && topCategories.length) {
        catTabs.innerHTML = `
          <span class="chip active" data-cat="all">All</span>
          ${topCategories.map(c => `<span class="chip" data-cat="${esc(c.toLowerCase())}">${esc(c)}</span>`).join('')}
        `;
        // Re-wire click events
        $$('.category-tabs .chip').forEach(ch => {
          ch.onclick = () => {
            $$('.category-tabs .chip').forEach(c2 => c2.classList.remove('active'));
            ch.classList.add('active');
            loadListings(ch.dataset.cat, $('#hustle-search')?.value.trim());
          };
        });
        window._categoryTabsBuilt = true;
      }
    }
    
    if (cat !== 'all') items = items.filter(i => (i.category || '').toLowerCase() === cat);
    if (query) {
      const q = query.toLowerCase();
      items = items.filter(i => (i.title || '').toLowerCase().includes(q) || (i.category || '').toLowerCase().includes(q) || (i.sellerName || '').toLowerCase().includes(q));
    }
    items.sort((a, b) => (b.createdAt?.seconds || 0) - (a.createdAt?.seconds || 0));

    if (!items.length) {
      grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1"><div class="empty-state-icon">🛒</div><h3>No listings yet</h3><p>Be the first to sell something!</p></div>`;
      return;
    }
    grid.innerHTML = items.map(item => `
      <div class="listing-card" onclick="openProductDetail('${item.id}')">
        ${item.imageURL ? `<img class="listing-image" src="${item.imageURL}" loading="lazy">` : '<div class="listing-placeholder">📦</div>'}
        <div class="listing-info">
          <div class="listing-price">R${esc(String(item.price))}</div>
          <div class="listing-title">${esc(item.title)}</div>
          <div class="listing-seller">${avatar(item.sellerName, null, 'avatar-sm')}<span>${esc(item.sellerName)}</span></div>
        </div>
      </div>
    `).join('');
    // store items for detail view
    window._hustleItems = {};
    items.forEach(i => window._hustleItems[i.id] = i);
  } catch (e) { grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><h3>Error loading</h3></div>'; }
}

function openSellModal() {
  let pendingImg = null;
  openModal(`
    <div class="modal-header"><h2>Sell Item</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      <div class="form-group"><label>What are you selling?</label><input type="text" id="sell-title" placeholder="e.g. Calculus Textbook"></div>
      <div class="form-group"><label>Price (R)</label><input type="number" id="sell-price" placeholder="150"></div>
      <div class="form-group"><label>Description</label><textarea id="sell-desc" placeholder="Details about your item..." rows="3"></textarea></div>
      <div class="form-group"><label>Category</label><input type="text" id="sell-cat" placeholder="e.g. Books, Tech, Notes, Services"></div>
      <div class="form-group"><label>Photo</label><input type="file" accept="image/*" id="sell-file"></div>
      <div id="sell-preview" class="image-preview" style="display:none"><img src=""><button class="image-preview-remove" onclick="document.getElementById('sell-preview').style.display='none'">&times;</button></div>
      <button class="btn-primary btn-full" id="sell-submit">List Item</button>
    </div>
  `);
  $('#sell-file').onchange = async e => {
    if (e.target.files[0]) { window._sellFile = e.target.files[0]; $('#sell-preview img').src = localPreview(e.target.files[0]); $('#sell-preview').style.display = 'block'; }
  };
  $('#sell-submit').onclick = async () => {
    const title = $('#sell-title').value.trim(), price = $('#sell-price').value.trim();
    const category = $('#sell-cat').value.trim() || 'Other';
    const description = $('#sell-desc').value.trim() || '';
    if (!title || !price) return toast('Title and price required');
    const fileToUpload = window._sellFile || null;
    window._sellFile = null;
    closeModal(); toast('Uploading...');
    try {
      let sellImgURL = null;
      if (fileToUpload) { sellImgURL = await uploadToR2(fileToUpload, 'listings'); }
      await db.collection('listings').add({
        title, price, category, description, imageURL: sellImgURL,
        sellerId: state.user.uid, sellerName: state.profile.displayName,
        status: 'active', createdAt: FieldVal.serverTimestamp()
      });
      toast('Listed!'); navigate('hustle');
    } catch (e) { toast('Failed'); console.error(e); }
  };
}

// ══════════════════════════════════════════════════
//  MESSAGES — Fixed: no orderBy = no composite index
// ══════════════════════════════════════════════════

// ─── Product Detail Popup ──────────────────
function openProductDetail(itemId) {
  const item = (window._hustleItems || {})[itemId];
  if (!item) return toast('Product not found');
  const isSelf = item.sellerId === state.user.uid;
  const isFriend = (state.profile.friends || []).includes(item.sellerId);

  openModal(`
    <div class="modal-header"><h2>Product Details</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      ${item.imageURL ? `<div style="border-radius:var(--radius);overflow:hidden;margin-bottom:16px"><img src="${item.imageURL}" style="width:100%;max-height:280px;object-fit:cover;cursor:pointer" onclick="viewImage('${item.imageURL}')"></div>` : ''}
      <div style="font-size:24px;font-weight:800;color:var(--accent);margin-bottom:4px">R${esc(String(item.price))}</div>
      <div style="font-size:18px;font-weight:700;margin-bottom:12px">${esc(item.title)}</div>
      ${item.description ? `<p style="font-size:14px;line-height:1.5;color:var(--text-secondary);margin-bottom:12px">${esc(item.description)}</p>` : ''}
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
        ${avatar(item.sellerName, null, 'avatar-sm')}
        <div>
          <div style="font-weight:600;font-size:14px">${esc(item.sellerName)}</div>
          <div style="font-size:12px;color:var(--text-secondary)">Seller</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;padding:10px 0;border-top:1px solid var(--border);font-size:13px;color:var(--text-secondary)">
        <span>📁 ${esc(item.category || 'Other')}</span>
        <span>·</span>
        <span>📅 ${timeAgo(item.createdAt)}</span>
      </div>
      <div style="display:flex;gap:12px;margin-top:16px">
        ${isSelf ? `<button class="btn-secondary" style="flex:1" disabled>Your Listing</button>` :
          `<button class="btn-primary" style="flex:1" id="hustle-buy-btn" onclick="hustleBuyInterest('${itemId}')">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            I'm Interested
          </button>`}
        <button class="btn-outline" style="flex:1" onclick="closeModal();openProfile('${item.sellerId}')">
          View Profile
        </button>
      </div>
    </div>
  `);
}

async function hustleBuyInterest(itemId) {
  const item = (window._hustleItems || {})[itemId];
  if (!item) return;
  const btn = document.getElementById('hustle-buy-btn');
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="inline-spinner" style="width:14px;height:14px"></span> Sending…'; }

  const uid = state.user.uid;
  const sellerId = item.sellerId;
  const sellerName = item.sellerName;
  const isFriend = (state.profile.friends || []).includes(sellerId);
  const interestMsg = `Hi, I'm interested in "${item.title}" (R${item.price}). Is it still available?`;

  try {
    // 1. Send friend request if not friends yet
    if (!isFriend) {
      await sendFriendRequest(sellerId, sellerName, '');
    }

    // 2. Find or create conversation
    const snap = await db.collection('conversations').where('participants', 'array-contains', uid).get();
    let convoId = null;
    const existing = snap.docs.find(d => d.data().participants.includes(sellerId));
    if (existing) {
      convoId = existing.id;
    } else {
      const doc = await db.collection('conversations').add({
        participants: [uid, sellerId],
        participantNames: [state.profile.displayName, sellerName],
        participantPhotos: [state.profile.photoURL || null, ''],
        lastMessage: '', updatedAt: FieldVal.serverTimestamp(),
        unread: { [sellerId]: 0, [uid]: 0 }
      });
      convoId = doc.id;
    }

    // 3. Send the interest message automatically
    await db.collection('conversations').doc(convoId).collection('messages').add({
      text: interestMsg,
      senderId: uid,
      createdAt: FieldVal.serverTimestamp(),
      status: 'sent',
      type: 'hustle_interest',
      payload: { itemId, title: item.title, price: item.price, imageURL: item.imageURL || null }
    });
    await db.collection('conversations').doc(convoId).set({
      lastMessage: interestMsg,
      updatedAt: FieldVal.serverTimestamp(),
      unread: { [sellerId]: FieldVal.increment(1), [uid]: 0 }
    }, { merge: true });

    // 4. Send notification to seller
    await db.collection('users').doc(sellerId).collection('notifications').add({
      type: 'hustle_interest',
      fromUid: uid,
      fromName: state.profile.displayName,
      fromPhoto: state.profile.photoURL || null,
      itemTitle: item.title,
      itemPrice: item.price,
      message: `${state.profile.displayName} is interested in your listing "${item.title}"`,
      read: false,
      createdAt: FieldVal.serverTimestamp()
    });

    closeModal();
    toast('Interest sent to seller!');

    // Open the chat if friends, otherwise inform
    if (isFriend) {
      openChat(convoId);
    } else {
      toast('Friend request sent — chat will unlock when accepted');
    }
  } catch (e) {
    console.error(e);
    toast('Failed to send interest');
    if (btn) { btn.disabled = false; btn.textContent = "I'm Interested"; }
  }
}

// ══════════════════════════════════════════════════
//  GROUP CHAT
// ══════════════════════════════════════════════════
function openCreateGroup() {
  openModal(`
    <div class="modal-header"><h2>New Group</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      <div class="form-group"><label>Group Name</label><input type="text" id="grp-name" placeholder="e.g. MAT101 Study Group"></div>
      <div class="form-group"><label>Description</label><input type="text" id="grp-desc" placeholder="What's this group for?"></div>
      <div class="form-group"><label>Type</label>
        <select id="grp-type"><option value="study">📚 Study Group</option><option value="social">🎉 Social</option><option value="project">💻 Project</option><option value="module">🧩 Module</option></select>
      </div>
      <div class="form-group"><label>Cover Photo (optional)</label><input type="file" accept="image/*" id="grp-cover-file"></div>
      <div class="form-group" style="display:flex;align-items:center;gap:8px">
        <input type="checkbox" id="grp-anon-toggle" style="width:auto">
        <label for="grp-anon-toggle" style="margin:0;font-size:14px">Allow anonymous posting</label>
      </div>
      <button class="btn-primary btn-full" id="grp-create-btn">Create Group</button>
    </div>
  `);
  $('#grp-create-btn').onclick = async () => {
    const name = $('#grp-name')?.value.trim();
    const desc = $('#grp-desc')?.value.trim() || '';
    const type = $('#grp-type')?.value || 'study';
    const allowAnon = $('#grp-anon-toggle')?.checked || false;
    if (!name) return toast('Name required');
    closeModal(); toast('Creating group...');
    try {
      let coverURL = null;
      const coverFile = $('#grp-cover-file')?.files?.[0];
      if (coverFile) coverURL = await uploadToR2(coverFile, 'group-covers');
      await db.collection('groups').add({
        name, description: desc, type, coverURL,
        createdBy: state.user.uid,
        admins: [state.user.uid],
        members: [state.user.uid],
        memberNames: { [state.user.uid]: state.profile.displayName },
        memberPhotos: { [state.user.uid]: state.profile.photoURL || '' },
        allowAnonymous: allowAnon,
        lastMessage: '', updatedAt: FieldVal.serverTimestamp(),
        createdAt: FieldVal.serverTimestamp()
      });
      toast('Group created!');
      navigate('chat');
    } catch (e) { toast('Failed'); console.error(e); }
  };
}

let gchatUnsub = null;

async function openGroupChat(groupId, collection = 'groups') {
  try {
    const gDoc = await db.collection(collection).doc(groupId).get();
    if (!gDoc.exists) return toast('Group not found');
    const group = { id: groupId, ...gDoc.data() };
    const uid = state.user.uid;
    const gName = group.name || group.groupTitle || 'Group';
    const gType = group.type || 'study';
    const gEmoji = collection === 'assignmentGroups' ? '📋' : (gType === 'study' ? '📚' : gType === 'project' ? '💻' : gType === 'module' ? '🧩' : '🎉');

    showScreen('group-chat-view');
    $('#gchat-hdr-info').innerHTML = `
      <div class="group-header-info">
        <div class="group-icon">${gEmoji}</div>
        <div><h3 style="font-size:15px;font-weight:700">${esc(gName)}</h3>
        <small style="color:var(--text-secondary)">${(group.members || []).length} members${group.moduleCode ? ' · ' + esc(group.moduleCode) : ''}</small></div>
      </div>
    `;
    if (gchatUnsub) gchatUnsub();
    if (_gchatViewportCleanup) { _gchatViewportCleanup(); _gchatViewportCleanup = null; }
    _gReplyTo = null;
    const gReply = $('#gchat-reply-indicator');
    if (gReply) gReply.style.display = 'none';
    const msgs = $('#gchat-msgs');
    if (msgs) msgs.innerHTML = '<div style="text-align:center;padding:32px"><span class="inline-spinner"></span></div>';
    _gchatViewportCleanup = setupViewportFollow(msgs);
    gchatUnsub = db.collection(collection).doc(groupId)
      .collection('messages').orderBy('createdAt','asc').limit(100)
      .onSnapshot(snap => {
        const messages = snap.docs.map(d => ({ id: d.id, ...d.data() }));
        _gMsgLookup = new Map(messages.map(m => [m.id, m]));
        if (!messages.length) {
          msgs.innerHTML = '<div style="text-align:center;padding:32px;opacity:0.5">Start the conversation! 💬</div>';
        } else {
          msgs.innerHTML = messages.map((m, idx) => {
            const isMe = m.senderId === uid;
            let content = '';
            if (m.audioURL) content += renderVoiceMsg(m.audioURL);
            if (m.imageURL) content += `<img src="${m.imageURL}" class="msg-image" onclick="viewImage('${m.imageURL}')">`;
            if (m.text) content += esc(m.text);
            // Support both new and legacy replies: infer original sender from replyToId when needed.
            const replyToSenderId = m.replyToSenderId || _gMsgLookup.get(m.replyToId || '')?.senderId;
            const replyDisplayName = replyToSenderId === uid ? 'me' : (m.replyToName || 'Message');
            const replyMeta = m.replyToText
              ? `<div class="msg-reply-snippet">↩ ${esc(replyDisplayName)}: ${esc(clampText(m.replyToText, 50))}</div>`
              : '';
            const newCls = (idx === messages.length - 1 && isMe) ? 'msg-new' : '';
            return `<div class="msg-row ${isMe ? 'msg-row-sent' : 'msg-row-received'}" id="msg-${m.id}">
              ${!isMe ? `<div class="msg-avatar-wrap">${avatar(m.senderName || '?', m.senderPhoto, 'avatar-xs')}</div>` : ''}
              <div class="msg-bubble ${isMe ? 'msg-sent' : 'msg-received'} ${newCls}">
              ${!isMe ? `<div class="gchat-sender">${esc(m.senderName?.split(' ')[0] || '?')}</div>` : ''}
              ${m.replyToId && m.replyToText ? `<div class="msg-reply-snippet" onclick="jumpToMessage('${m.replyToId}','gchat-msgs')">↩ ${esc(replyDisplayName)}: ${esc(clampText(m.replyToText, 50))}</div>` : ''}
              ${content}
              <button class="msg-reply-btn" title="Reply" aria-label="Reply" onclick="setGroupReply('${m.id}')"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 17 4 12 9 7"></polyline><path d="M20 18v-2a4 4 0 0 0-4-4H4"></path></svg></button>
              <div class="msg-time">${m.createdAt ? timeAgo(m.createdAt) : ''}</div>
            </div></div>`;
          }).join('');
          scrollToLatest(msgs);
        }
      });

    const gInput = $('#gchat-input');
    const resizeGInput = () => {
      if (!gInput) return;
      gInput.style.height = '40px';
      gInput.style.height = `${Math.min(gInput.scrollHeight, 84)}px`;
      scrollToLatest(msgs);
    };
    if (gInput) {
      gInput.style.height = '40px';
      gInput.oninput = resizeGInput;
    }

    const sendGMsg = async () => {
      const input = gInput || $('#gchat-input');
      const text = input?.value.trim();
      let audioURL = null;
      if (window._gchatVoiceBlob) {
        const af = new File([window._gchatVoiceBlob], `voice_${Date.now()}.webm`, { type: 'audio/webm' });
        audioURL = await uploadToR2(af, 'voice');
        window._gchatVoiceBlob = null;
      }
      if (!text && !audioURL) return;
      const replyPayload = _gReplyTo ? {
        replyToId: _gReplyTo.id,
        replyToText: _gReplyTo.text,
        replyToName: _gReplyTo.name,
        replyToSenderId: _gReplyTo.senderId
      } : {};
      input.value = '';
      resizeGInput();
      input.focus();
      _gReplyTo = null;
      if (gReply) gReply.style.display = 'none';
      try {
        await db.collection(collection).doc(groupId).collection('messages').add({
          text: text || '', audioURL: audioURL || null,
          senderId: uid, senderName: state.profile.displayName,
          senderPhoto: state.profile.photoURL || null,
          ...replyPayload,
          createdAt: FieldVal.serverTimestamp()
        });
        await db.collection(collection).doc(groupId).update({
          lastMessage: audioURL ? '🎤 Voice' : text, updatedAt: FieldVal.serverTimestamp()
        });
      } catch (e) { console.error(e); }
    };
    $('#gchat-send').onclick = sendGMsg;
    $('#gchat-input').onkeydown = e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendGMsg();
      }
    };
    $('#gchat-input').onfocus = () => scrollToLatest(msgs);
    $('#gchat-input').onblur = () => setTimeout(() => scrollToLatest(msgs), 80);
    $('#gchat-back').onclick = () => {
      if (gchatUnsub) { gchatUnsub(); gchatUnsub = null; }
      if (_gchatViewportCleanup) { _gchatViewportCleanup(); _gchatViewportCleanup = null; }
      showScreen('app');
      if (state.page !== 'chat') navigate('chat');
    };
  } catch (e) { console.error(e); toast('Could not open group'); }
}

function setGroupReply(messageId) {
  const m = _gMsgLookup.get(messageId);
  if (!m) return;
  const replyText = m.text || (m.audioURL ? '[voice message]' : (m.imageURL ? '[image]' : '[message]'));
  _gReplyTo = {
    id: m.id,
    text: replyText,
    name: m.senderName || 'User',
    senderId: m.senderId
  };
  const ind = $('#gchat-reply-indicator');
  if (!ind) return;
  ind.innerHTML = `<span>↩ Replying to <strong>${esc(_gReplyTo.name)}</strong>: ${esc(clampText(replyText, 42))}</span><button class="chat-reply-close" onclick="clearGroupReply()">&times;</button>`;
  ind.style.display = 'flex';
  $('#gchat-input')?.focus();
}

function clearGroupReply() {
  _gReplyTo = null;
  const ind = $('#gchat-reply-indicator');
  if (ind) ind.style.display = 'none';
}

function setDmReply(messageId) {
  const m = _dmMsgLookup.get(messageId);
  if (!m) return;
  const otherName = m.senderId === state.user?.uid ? 'You' : 'Them';
  const replyText = m.text || (m.audioURL ? '[voice message]' : (m.imageURL ? '[image]' : '[message]'));
  _dmReplyTo = { id: m.id, text: replyText, name: otherName, senderId: m.senderId };
  const ind = $('#dm-reply-indicator');
  if (!ind) return;
  ind.innerHTML = `<span>↩ Replying to <strong>${esc(otherName)}</strong>: ${esc(clampText(replyText, 42))}</span><button class="chat-reply-close" onclick="clearDmReply()">&times;</button>`;
  ind.style.display = 'flex';
  $('#chat-input')?.focus();
}

function clearDmReply() {
  _dmReplyTo = null;
  const ind = $('#dm-reply-indicator');
  if (ind) ind.style.display = 'none';
}

async function joinGroup(groupId) {
  try {
    const uid = state.user.uid;
    await db.collection('groups').doc(groupId).update({
      members: FieldVal.arrayUnion(uid),
      [`memberNames.${uid}`]: state.profile.displayName,
      [`memberPhotos.${uid}`]: state.profile.photoURL || ''
    });
    toast('Joined group!');
    openGroupChat(groupId);
  } catch (e) { toast('Failed to join'); console.error(e); }
}

function renderMessages() {
  const c = $('#content');
  c.innerHTML = `
    <div class="messages-page">
      <div class="messages-header"><h2>Messages</h2></div>
      <div class="msg-tabs">
        <button class="msg-tab active" data-mt="dm">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          DMs <span class="tab-badge" id="dm-tab-badge"></span>
        </button>
        <button class="msg-tab" data-mt="groups">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
          Groups <span class="tab-badge" id="asg-tab-badge"></span>
        </button>
      </div>
      <div id="msg-tab-content">
        <div class="convo-list" id="convo-list">
          <div style="padding:40px;text-align:center"><span class="inline-spinner"></span></div>
        </div>
      </div>
      <button class="archive-fab" id="archive-fab" onclick="toggleArchiveDmView()" aria-label="Archived chats" title="Show archived chats">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="21 8 21 21 3 21 3 8"/><rect x="1" y="3" width="22" height="5"/><line x1="10" y1="12" x2="14" y2="12"/></svg>
      </button>
    </div>
  `;
  // Compute DM unread count for the tab badge
  _updateDMTabBadge();
  refreshChatBadge();
  $$('.msg-tab').forEach(tab => {
    tab.onclick = () => {
      $$('.msg-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.lastMsgTab = tab.dataset.mt;
      if (tab.dataset.mt === 'dm') loadDMList();
      else loadGroups();
      updateArchiveFabState();
    };
  });
  // Restore last active tab
  const restoreTab = state.lastMsgTab || 'dm';
  if (restoreTab === 'groups') state.lastMsgTab = 'dm';
  if (restoreTab === 'archived') {
    $$('.msg-tab').forEach(t => t.classList.remove('active'));
    loadArchivedDMList();
  } else {
    const tabBtn = document.querySelector(`.msg-tab[data-mt="${state.lastMsgTab || 'dm'}"]`);
    if (tabBtn) { tabBtn.click(); } else { loadDMList(); }
  }
  updateArchiveFabState();
}

function toggleArchiveDmView() {
  if (state.lastMsgTab === 'archived') {
    state.lastMsgTab = 'dm';
    const dmTab = document.querySelector('.msg-tab[data-mt="dm"]');
    if (dmTab) dmTab.click();
    else loadDMList();
  } else {
    state.lastMsgTab = 'archived';
    $$('.msg-tab').forEach(t => t.classList.remove('active'));
    loadArchivedDMList();
  }
  updateArchiveFabState();
}

function updateArchiveFabState() {
  const fab = $('#archive-fab');
  if (!fab) return;
  const active = state.lastMsgTab === 'archived';
  fab.classList.toggle('active', active);
  fab.title = active ? 'Back to DMs' : 'Show archived chats';
  fab.setAttribute('aria-pressed', active ? 'true' : 'false');
}

function refreshCurrentMessageList() {
  if (state.page !== 'chat') return;
  if (state.lastMsgTab === 'archived') loadArchivedDMList();
  else if (state.lastMsgTab === 'groups') loadGroups();
  else loadDMList();
  updateArchiveFabState();
}

function loadGroupList() {
  const container = $('#msg-tab-content'); if (!container) return;
  container.innerHTML = `<div style="padding:12px 16px"><button class="btn-primary btn-full" onclick="openCreateGroup()">+ New Group</button></div><div class="convo-list" id="group-list"><div style="padding:40px;text-align:center"><span class="inline-spinner"></span></div></div>`;
  db.collection('groups').orderBy('updatedAt','desc').limit(30).get().then(snap => {
    const groups = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    const el = $('#group-list');
    if (!groups.length) {
      el.innerHTML = '<div class="empty-state"><div class="empty-state-icon">👥</div><h3>No groups yet</h3><p>Create one to get started!</p></div>';
      return;
    }
    const uid = state.user.uid;
    el.innerHTML = groups.map(g => {
      const isMember = (g.members || []).includes(uid);
      const emoji = g.type === 'study' ? '📚' : g.type === 'project' ? '💻' : g.type === 'module' ? '🧩' : '🎉';
      return `
        <div class="convo-item" onclick="${isMember ? `openGroupChat('${g.id}')` : `joinGroup('${g.id}')`}">
          <div class="convo-avatar"><div class="avatar-md group-avatar-icon">${emoji}</div></div>
          <div class="convo-info">
            <div class="convo-name">${esc(g.name)}</div>
            <div class="convo-last-msg">${isMember ? esc(g.lastMessage || 'No messages yet') : '<em>Tap to join</em>'}</div>
          </div>
          <div class="convo-right">
            <div class="convo-time">${timeAgo(g.updatedAt)}</div>
            <div style="font-size:11px;color:var(--text-tertiary)">${(g.members||[]).length} members</div>
          </div>
        </div>`;
    }).join('');
  }).catch(e => { console.error(e); });
}

// ══════════════════════════════════════════════════
//  GROUPS — Intent-Based, Temporary
// ══════════════════════════════════════════════════
function loadGroups() {
  const container = $('#msg-tab-content'); if (!container) return;
  const myModules = state.profile.modules || [];
  container.innerHTML = `
    <div class="asg-page">
      <div style="padding:12px 16px;display:flex;gap:8px">
        <button class="btn-primary" style="flex:1" onclick="openCreateModuleGroup()">+ New Group</button>
      </div>
      <div class="asg-filter-row">
        <span class="chip active" data-af="my">My Modules</span>
        <span class="chip" data-af="all">All Open</span>
        <span class="chip" data-af="mine">My Groups</span>
      </div>
      <div id="asg-list"><div style="padding:40px;text-align:center"><span class="inline-spinner"></span></div></div>
    </div>
  `;
  $$('.asg-filter-row .chip').forEach(ch => {
    ch.onclick = () => {
      $$('.asg-filter-row .chip').forEach(c => c.classList.remove('active'));
      ch.classList.add('active');
      loadAsgList(ch.dataset.af);
    };
  });
  loadAsgList('my');
}

async function loadAsgList(filter = 'my') {
  const el = $('#asg-list'); if (!el) return;
  const uid = state.user.uid;
  const myModules = state.profile.modules || [];
  try {
    let snap;
    if (filter === 'mine') {
      snap = await db.collection('assignmentGroups').where('members', 'array-contains', uid).limit(30).get();
    } else {
      snap = await db.collection('assignmentGroups').where('status', '==', 'open').limit(50).get();
    }
    let groups = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    if (filter === 'my') {
      groups = groups.filter(g => myModules.includes(g.moduleCode));
    }
    groups.sort((a, b) => (b.createdAt?.seconds || 0) - (a.createdAt?.seconds || 0));

    if (!groups.length) {
      el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">📋</div><h3>No groups</h3><p>${filter === 'my' ? 'None for your modules yet' : 'Create one to get started!'}</p></div>`;
      return;
    }
    el.innerHTML = groups.map(g => {
      const isMember = (g.members || []).includes(uid);
      const spotsLeft = (g.maxSize || 10) - (g.members || []).length;
      const isHost = g.createdBy === uid;
      const isLocked = g.locked || false;
      const isFull = spotsLeft <= 0;
      // Check preference conflicts
      const myPrefs = (g.preferences || {})[uid];
      const dontWant = myPrefs?.dontWant || [];
      const conflicts = (g.members || []).filter(m => dontWant.includes(m));
      const hasConflict = conflicts.length > 0;

      let statusBadge = '';
      if (g.status === 'archived') statusBadge = '<span class="asg-badge archived">Archived</span>';
      else if (isLocked) statusBadge = '<span class="asg-badge locked">Locked</span>';
      else if (isFull) statusBadge = '<span class="asg-badge full">Full</span>';
      else statusBadge = `<span class="asg-badge open">${spotsLeft} spot${spotsLeft !== 1 ? 's' : ''} left</span>`;

      // Only show member avatars to group members
      const canSeeMembers = isMember || _isAdmin;

      return `
        <div class="asg-card ${isMember ? 'is-member' : ''} ${g.status === 'archived' ? 'is-archived' : ''}" onclick="openGroupDetail('${g.id}')">
          <div class="asg-card-top">
            <div class="asg-card-module">${esc(g.moduleCode || '???')}</div>
            ${statusBadge}
          </div>
          <div class="asg-card-title">${esc(g.groupTitle || g.assignmentTitle)}</div>
          <div class="asg-card-meta">
            <span>\ud83d\udc65 ${(g.members||[]).length}/${g.maxSize||10}</span>
            <span>\u00b7</span>
            <span>${g.joinMode === 'open' ? '\ud83d\udd13 Open' : g.joinMode === 'invite' ? '\ud83d\udd12 Invite' : '\ud83e\udd16 Auto-fill'}</span>
            ${g.dueDate ? `<span>\u00b7 \ud83d\udcc5 ${esc(g.dueDate)}</span>` : ''}
          </div>
          ${canSeeMembers ? `<div class="asg-card-members">
            ${(g.members||[]).slice(0,5).map(mid => {
              const mName = (g.memberNames||{})[mid] || '?';
              return avatar(mName, (g.memberPhotos||{})[mid] || null, 'avatar-sm');
            }).join('')}
            ${(g.members||[]).length > 5 ? `<span class="asg-more">+${(g.members||[]).length - 5}</span>` : ''}
          </div>` : ''}
          ${hasConflict && isMember ? '<div class="asg-conflict">\u26a0\ufe0f 1 person you preferred not to work with is in this group</div>' : ''}
          <div class="asg-card-host">Created by ${esc((g.memberNames||{})[g.createdBy] || 'Someone')} \u00b7 ${timeAgo(g.createdAt)}</div>
        </div>`;
    }).join('');
  } catch (e) { console.error(e); el.innerHTML = '<div class="empty-state"><h3>Could not load</h3></div>'; }
}

function openCreateModuleGroup() {
  const myModules = state.profile.modules || [];
  const moduleOptions = myModules.length
    ? myModules.map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join('')
    : '<option value="">Add modules in your profile first</option>';

  openModal(`
    <div class="modal-header"><h2>New Group</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      <div class="form-group"><label>Module</label>
        <select id="asg-module">${moduleOptions}<option value="_custom">Other (type below)</option></select>
      </div>
      <div class="form-group" id="asg-custom-wrap" style="display:none"><label>Module Code</label><input type="text" id="asg-custom-module" placeholder="e.g. BIO214"></div>
      <div class="form-group"><label>Group Title</label><input type="text" id="asg-title" placeholder="e.g. Genetics Project"></div>
      <div class="form-group"><label>Max Group Size</label>
        <select id="asg-size"><option value="3">3</option><option value="4">4</option><option value="5" selected>5</option><option value="6">6</option><option value="8">8</option><option value="10">10</option></select>
      </div>
      <div class="form-group"><label>Due Date (optional)</label><input type="date" id="asg-due"></div>
      <div class="form-group"><label>Join Mode</label>
        <select id="asg-join">
          <option value="open">🔓 Open — anyone can join</option>
          <option value="invite">🔒 Invite — you approve requests</option>
          <option value="auto">🤖 Auto-fill — system fills remaining spots</option>
        </select>
      </div>
      <div class="form-group" style="display:flex;align-items:center;gap:8px">
        <input type="checkbox" id="asg-autofill-toggle" style="width:auto" checked>
        <label for="asg-autofill-toggle" style="margin:0;font-size:14px">Enable auto-fill for empty spots</label>
      </div>
      <p style="color:var(--text-tertiary);font-size:11px;margin:-8px 0 12px">Auto-fill adds consenting students from the same module when spots remain.</p>
      <div class="form-group"><label>Visibility</label>
        <select id="asg-vis">
          <option value="public">🌍 Public — all NWU students</option>
          <option value="friends">👫 Friends only</option>
        </select>
      </div>
      <button class="btn-primary btn-full" id="asg-create-btn">Create Group</button>
    </div>
  `);
  $('#asg-module').onchange = () => {
    const wrap = $('#asg-custom-wrap');
    wrap.style.display = $('#asg-module').value === '_custom' ? 'block' : 'none';
  };
  $('#asg-create-btn').onclick = async () => {
    let moduleCode = $('#asg-module').value;
    if (moduleCode === '_custom') moduleCode = ($('#asg-custom-module')?.value || '').trim().toUpperCase();
    const title = $('#asg-title')?.value.trim();
    const maxSize = parseInt($('#asg-size')?.value) || 5;
    const dueDate = $('#asg-due')?.value || '';
    const joinMode = $('#asg-join')?.value || 'open';
    const visibility = $('#asg-vis')?.value || 'public';
    const autoFillEnabled = $('#asg-autofill-toggle')?.checked || false;
    if (!moduleCode || !title) return toast('Module and title required');
    closeModal(); toast('Creating group...');
    const uid = state.user.uid;
    try {
      const doc = await db.collection('assignmentGroups').add({
        moduleCode, groupTitle: title, assignmentTitle: title, maxSize, dueDate, joinMode, visibility,
        autoFillEnabled,
        createdBy: uid, status: 'open', locked: false,
        members: [uid],
        memberNames: { [uid]: state.profile.displayName },
        memberPhotos: { [uid]: state.profile.photoURL || '' },
        pendingRequests: [],
        preferences: {},
        lastMessage: '', updatedAt: FieldVal.serverTimestamp(),
        createdAt: FieldVal.serverTimestamp()
      });
      toast('Group created!');
      openGroupDetail(doc.id);
    } catch (e) { toast('Failed'); console.error(e); }
  };
}

async function openGroupDetail(groupId) {
  try {
    const gDoc = await db.collection('assignmentGroups').doc(groupId).get();
    if (!gDoc.exists) return toast('Not found');
    const g = { id: gDoc.id, ...gDoc.data() };
    const uid = state.user.uid;
    const isMember = (g.members || []).includes(uid);
    const isHost = g.createdBy === uid;
    const spotsLeft = (g.maxSize || 10) - (g.members || []).length;
    const isLocked = g.locked || false;
    const isFull = spotsLeft <= 0;
    const isArchived = g.status === 'archived';
    const myPrefs = (g.preferences || {})[uid] || {};
    const canSeeMembers = isMember || _isAdmin;

    let membersHtml = '';
    if (canSeeMembers) {
      membersHtml = (g.members || []).map(mid => {
        const mName = (g.memberNames||{})[mid] || 'Unknown';
        const mPhoto = (g.memberPhotos||{})[mid] || null;
        const isCreator = mid === g.createdBy;
        const warnMe = (myPrefs.dontWantUids || myPrefs.dontWant || []).includes(mid) && mid !== uid;
        return `
          <div class="asg-member ${warnMe ? 'conflict' : ''}">
            ${avatar(mName, mPhoto, 'avatar-md')}
            <div class="asg-member-info">
              <div class="asg-member-name">${esc(mName)}${verifiedBadge(mid)} ${isCreator ? '<span class="asg-host-tag">Host</span>' : ''}</div>
              ${warnMe ? '<div class="asg-member-warn">\u26a0\ufe0f Preference conflict</div>' : ''}
            </div>
            ${isHost && mid !== uid && !isLocked ? `<button class="btn-sm btn-ghost" onclick="event.stopPropagation();removeFromAsg('${groupId}','${mid}')">Remove</button>` : ''}
          </div>`;
      }).join('');
    } else {
      membersHtml = `<p style="color:var(--text-tertiary);font-size:13px;padding:8px 0">\ud83d\udd12 Member details visible only to group members (${(g.members||[]).length} members)</p>`;
    }

    // Pending requests (host can see)
    let pendingHtml = '';
    if (isHost && (g.pendingRequests || []).length > 0) {
      pendingHtml = `<div class="asg-section"><h4>Pending Requests</h4>${(g.pendingRequests||[]).map(r => `
        <div class="asg-member pending">
          <div class="asg-member-info"><div class="asg-member-name">${esc(r.name)}</div></div>
          <div style="display:flex;gap:6px">
            <button class="btn-sm btn-primary" onclick="event.stopPropagation();approveAsgRequest('${groupId}','${r.uid}','${esc(r.name)}','${r.photo || ''}')">Accept</button>
            <button class="btn-sm btn-ghost" onclick="event.stopPropagation();rejectAsgRequest('${groupId}','${r.uid}')">Decline</button>
          </div>
        </div>
      `).join('')}</div>`;
    }

    // Actions
    let actionsHtml = '';
    if (isArchived) {
      actionsHtml = '<p style="text-align:center;color:var(--text-tertiary);padding:8px">This group has been archived.</p>';
    } else if (isMember) {
      actionsHtml = `
        <button class="btn-primary btn-full" onclick="openAsgChat('${groupId}')">\ud83d\udcac Open Group Chat</button>
        <div style="display:flex;gap:8px;margin-top:8px">
          <button class="btn-outline" style="flex:1" onclick="openAsgPreferences('${groupId}')">\u2699\ufe0f Preferences</button>
          ${isHost ? `<button class="btn-outline" style="flex:1" onclick="toggleAsgLock('${groupId}', ${!isLocked})">${isLocked ? '\ud83d\udd13 Unlock' : '\ud83d\udd12 Lock Group'}</button>` : ''}
        </div>
        ${isHost ? `<div style="display:flex;gap:8px;margin-top:8px">
          ${!isLocked && spotsLeft > 0 ? `<button class="btn-secondary" style="flex:1" onclick="autoFillAsg('${groupId}')">\ud83e\udd16 Auto-fill Spots</button>` : ''}
          <button class="btn-danger" style="flex:1;border-radius:var(--radius)" onclick="archiveAsg('${groupId}')">\ud83d\udce6 Archive</button>
        </div>` : ''}
        ${!isHost ? `<button class="btn-ghost" style="width:100%;margin-top:8px;color:var(--red)" onclick="leaveAsg('${groupId}')">Leave Group</button>` : ''}
      `;
    } else if (!isFull && !isLocked) {
      if (g.joinMode === 'open') {
        actionsHtml = `<button class="btn-primary btn-full" onclick="joinAsg('${groupId}')">Join Group</button>`;
      } else {
        actionsHtml = `<button class="btn-primary btn-full" onclick="requestJoinAsg('${groupId}')">Request to Join</button>`;
      }
    } else {
      actionsHtml = '<p style="text-align:center;color:var(--text-tertiary);padding:8px">This group is full or locked.</p>';
    }

    openModal(`
      <div class="modal-header"><h2>${esc(g.moduleCode)}</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
      <div class="modal-body asg-detail">
        <div class="asg-detail-title">${esc(g.groupTitle || g.assignmentTitle)}</div>
        <div class="asg-detail-meta">
          <span>\ud83d\udc65 ${(g.members||[]).length}/${g.maxSize||10}</span>
          <span>${g.joinMode === 'open' ? '\ud83d\udd13 Open' : g.joinMode === 'invite' ? '\ud83d\udd12 Invite' : '\ud83e\udd16 Auto-fill'}</span>
          ${g.dueDate ? `<span>\ud83d\udcc5 Due: ${esc(g.dueDate)}</span>` : ''}
          <span>${g.visibility === 'friends' ? '\ud83d\udc6b Friends only' : '\ud83c\udf0d Public'}</span>
        </div>
        <div class="asg-section"><h4>Members (${(g.members||[]).length})</h4>${membersHtml}</div>
        ${pendingHtml}
        <div class="asg-actions">${actionsHtml}</div>
      </div>
    `);
  } catch (e) { console.error(e); toast('Could not load group'); }
}

async function joinAsg(groupId) {
  const uid = state.user.uid;
  try {
    const gDoc = await db.collection('assignmentGroups').doc(groupId).get();
    const g = gDoc.data();
    if ((g.members||[]).length >= (g.maxSize||10)) return toast('Group is full');
    await db.collection('assignmentGroups').doc(groupId).update({
      members: FieldVal.arrayUnion(uid),
      [`memberNames.${uid}`]: state.profile.displayName,
      [`memberPhotos.${uid}`]: state.profile.photoURL || ''
    });
    closeModal(); toast('Joined!');
    openGroupDetail(groupId);
  } catch (e) { toast('Failed'); console.error(e); }
}

async function requestJoinAsg(groupId) {
  const uid = state.user.uid;
  try {
    const gDoc = await db.collection('assignmentGroups').doc(groupId).get();
    const g = gDoc.data() || {};
    await db.collection('assignmentGroups').doc(groupId).update({
      pendingRequests: FieldVal.arrayUnion({ uid, name: state.profile.displayName, photo: state.profile.photoURL || '' })
    });
    if (g.createdBy && g.createdBy !== uid) {
      addNotification(g.createdBy, 'group', `requested to join ${g.groupTitle || g.assignmentTitle || g.moduleCode || 'your group'}`, { groupId, kind: 'asg_join_request' });
    }
    closeModal(); toast('Request sent! The host will review it.');
  } catch (e) { toast('Failed'); console.error(e); }
}

async function approveAsgRequestByUid(groupId, reqUid) {
  try {
    const gDoc = await db.collection('assignmentGroups').doc(groupId).get();
    if (!gDoc.exists) return;
    const req = (gDoc.data().pendingRequests || []).find(r => r.uid === reqUid);
    if (!req) return toast('Request no longer available');
    await approveAsgRequest(groupId, reqUid, req.name || 'Student', req.photo || '');
  } catch (e) { console.error(e); toast('Failed'); }
}

async function approveAsgRequest(groupId, reqUid, reqName, reqPhoto) {
  try {
    const gDoc = await db.collection('assignmentGroups').doc(groupId).get();
    const g = gDoc.data();
    if ((g.members||[]).length >= (g.maxSize||10)) return toast('Group is full');
    const newPending = (g.pendingRequests||[]).filter(r => r.uid !== reqUid);
    await db.collection('assignmentGroups').doc(groupId).update({
      members: FieldVal.arrayUnion(reqUid),
      [`memberNames.${reqUid}`]: reqName,
      [`memberPhotos.${reqUid}`]: reqPhoto || '',
      pendingRequests: newPending
    });
    addNotification(reqUid, 'group', `approved your request to join ${g.groupTitle || g.assignmentTitle || g.moduleCode || 'a group'}`, { groupId, kind: 'asg_approved' });
    closeModal(); toast(`${reqName} approved!`);
    openGroupDetail(groupId);
  } catch (e) { toast('Failed'); console.error(e); }
}

async function rejectAsgRequest(groupId, reqUid) {
  try {
    const gDoc = await db.collection('assignmentGroups').doc(groupId).get();
    const g = gDoc.data();
    const req = (g.pendingRequests || []).find(r => r.uid === reqUid);
    const newPending = (g.pendingRequests||[]).filter(r => r.uid !== reqUid);
    await db.collection('assignmentGroups').doc(groupId).update({ pendingRequests: newPending });
    if (req?.uid) {
      addNotification(req.uid, 'group', `declined your request for ${g.groupTitle || g.assignmentTitle || g.moduleCode || 'a group'}`, { groupId, kind: 'asg_declined' });
    }
    closeModal(); toast('Request declined');
    openGroupDetail(groupId);
  } catch (e) { toast('Failed'); console.error(e); }
}

async function removeFromAsg(groupId, memberUid) {
  try {
    await db.collection('assignmentGroups').doc(groupId).update({
      members: FieldVal.arrayRemove(memberUid)
    });
    closeModal(); toast('Removed');
    openGroupDetail(groupId);
  } catch (e) { toast('Failed'); console.error(e); }
}

async function leaveAsg(groupId) {
  const uid = state.user.uid;
  try {
    await db.collection('assignmentGroups').doc(groupId).update({
      members: FieldVal.arrayRemove(uid)
    });
    closeModal(); toast('Left group');
    loadGroups();
  } catch (e) { toast('Failed'); console.error(e); }
}

async function toggleAsgLock(groupId, lock) {
  try {
    await db.collection('assignmentGroups').doc(groupId).update({ locked: lock });
    closeModal(); toast(lock ? 'Group locked' : 'Group unlocked');
    openGroupDetail(groupId);
  } catch (e) { toast('Failed'); console.error(e); }
}

async function archiveAsg(groupId) {
  openModal(`
    <div class="modal-body" style="text-align:center;padding:24px">
      <h3 style="margin-bottom:8px">Archive this group?</h3>
      <p style="color:var(--text-secondary);font-size:14px;margin-bottom:20px">The chat will be archived and the group closed. This can't be undone.</p>
      <div style="display:flex;gap:12px;justify-content:center">
        <button class="btn-secondary" onclick="closeModal()" style="flex:1">Cancel</button>
        <button class="btn-danger" onclick="doArchiveAsg('${groupId}')" style="flex:1;border-radius:var(--radius)">Archive</button>
      </div>
    </div>
  `);
}

async function doArchiveAsg(groupId) {
  try {
    await db.collection('assignmentGroups').doc(groupId).update({ status: 'archived', locked: true });
    closeModal(); toast('Group archived');
    loadGroups();
  } catch (e) { toast('Failed'); console.error(e); }
}

async function autoFillAsg(groupId) {
  try {
    const gDoc = await db.collection('assignmentGroups').doc(groupId).get();
    const g = gDoc.data();
    const spotsLeft = (g.maxSize || 10) - (g.members || []).length;
    if (spotsLeft <= 0) return toast('Group is already full');
    // Find students in same module who aren't in any group for this
    const allSnap = await db.collection('users').where('modules', 'array-contains', g.moduleCode).limit(50).get();
    const candidates = allSnap.docs
      .map(d => ({ id: d.id, ...d.data() }))
      .filter(u => !g.members.includes(u.id) && u.id !== state.user.uid);
    // Only include users who consented to auto-fill
    const consented = candidates.filter(c => c.allowAutoFill !== false);
    // Respect preferences: exclude people the host marked as "don't want"
    const hostPrefs = (g.preferences || {})[g.createdBy] || {};
    const dontWant = hostPrefs.dontWant || [];
    const filtered = consented.filter(c => !dontWant.includes(c.id));
    // Check if they're already in another group with the same module
    const existingSnap = await db.collection('assignmentGroups')
      .where('moduleCode', '==', g.moduleCode)
      .where('status', '==', 'open').limit(50).get();
    const takenUids = new Set();
    existingSnap.docs.forEach(d => {
      if (d.id === groupId) return;
      (d.data().members || []).forEach(m => takenUids.add(m));
    });
    const available = filtered.filter(c => !takenUids.has(c.id));
    const toAdd = available.slice(0, spotsLeft);
    if (!toAdd.length) return toast('No consenting students available to auto-fill');
    const updates = { members: g.members };
    toAdd.forEach(u => {
      updates.members.push(u.id);
      updates[`memberNames.${u.id}`] = u.displayName;
      updates[`memberPhotos.${u.id}`] = u.photoURL || '';
    });
    await db.collection('assignmentGroups').doc(groupId).update(updates);
    closeModal(); toast(`Added ${toAdd.length} student${toAdd.length > 1 ? 's' : ''}!`);
    openGroupDetail(groupId);
  } catch (e) { toast('Auto-fill failed'); console.error(e); }
}

async function openAsgPreferences(groupId) {
  const gDoc = await db.collection('assignmentGroups').doc(groupId).get();
  if (!gDoc.exists) return toast('Group not found');
  const g = { id: gDoc.id, ...gDoc.data() };
  const uid = state.user.uid;
  const myPrefs = (g.preferences || {})[uid] || {};

  openModal(`
    <div class="modal-header"><h2>Group Preferences</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body pref-modal">
      <div class="pref-section">
        <h4>People doing ${esc(g.moduleCode)}</h4>
        <input type="text" id="pref-search" class="pref-search-input" placeholder="Search by name..." autocomplete="off">
        <div id="pref-module-people" class="pref-people-list"><div class="inline-spinner" style="margin:16px auto"></div></div>
      </div>
      <div class="pref-section">
        <h4>Already in groups for this module</h4>
        <div id="pref-grouped" class="pref-grouped-list"><div class="inline-spinner" style="margin:16px auto"></div></div>
      </div>
      <div class="pref-section">
        <h4>Your Friends</h4>
        <div id="pref-friends-list" class="pref-people-list"><div class="inline-spinner" style="margin:16px auto"></div></div>
      </div>
      <div class="pref-actions">
        <button class="btn-primary btn-full" id="pref-save">Save Preferences</button>
        <p class="pref-hint">Select who you'd prefer (green) or prefer not (red) to work with. Used for auto-fill matching.</p>
      </div>
    </div>
  `);

  const _prefChoices = {}; // uid -> 'want' | 'dontwant' | null
  // Pre-fill existing prefs
  (myPrefs.wantUids || []).forEach(u => _prefChoices[u] = 'want');
  (myPrefs.dontWantUids || []).forEach(u => _prefChoices[u] = 'dontwant');

  function renderPersonItem(u, isGrouped = false) {
    const choice = _prefChoices[u.id] || '';
    const isMe = u.id === uid;
    const inThisGroup = (g.members || []).includes(u.id);
    if (isMe) return '';
    return `
      <div class="pref-person ${choice}" data-uid="${u.id}" ${isGrouped ? '' : `onclick="window._togglePrefChoice('${u.id}',this)"`}>
        ${avatar(u.displayName, u.photoURL, 'avatar-sm')}
        <div class="pref-person-info">
          <div class="pref-person-name">${esc(u.displayName)}${verifiedBadge(u.id)}</div>
          <div class="pref-person-meta">${esc(u.major || '')}${u.year ? ' · ' + esc(u.year) : ''}${inThisGroup ? ' · <span style="color:var(--green)">In this group</span>' : ''}</div>
        </div>
        ${isGrouped ? '<span class="pref-grouped-tag">In group</span>' : `<div class="pref-choice-indicator">${choice === 'want' ? '✓' : choice === 'dontwant' ? '✗' : '○'}</div>`}
      </div>`;
  }

  window._togglePrefChoice = (uid, el) => {
    const current = _prefChoices[uid] || '';
    if (!current) _prefChoices[uid] = 'want';
    else if (current === 'want') _prefChoices[uid] = 'dontwant';
    else delete _prefChoices[uid];
    const choice = _prefChoices[uid] || '';
    el.className = `pref-person ${choice}`;
    const ind = el.querySelector('.pref-choice-indicator');
    if (ind) ind.textContent = choice === 'want' ? '✓' : choice === 'dontwant' ? '✗' : '○';
  };

  // Load people from same module
  let modulePeople = [];
  try {
    const snap = await db.collection('users').where('modules', 'array-contains', g.moduleCode).limit(60).get();
    modulePeople = snap.docs.map(d => ({ id: d.id, ...d.data() })).filter(u => u.id !== uid);
  } catch (e) { console.error(e); }

  // Load existing groups for this module
  let groupedPeople = [];
  try {
    const gSnap = await db.collection('assignmentGroups').where('moduleCode', '==', g.moduleCode).where('status', '==', 'open').limit(30).get();
    const groups = gSnap.docs.map(d => ({ id: d.id, ...d.data() }));
    const takenMap = {};
    groups.forEach(grp => {
      if (grp.id === groupId) return;
      (grp.members || []).forEach(mid => {
        takenMap[mid] = { groupName: grp.groupTitle || grp.assignmentTitle || grp.moduleCode, groupId: grp.id };
      });
    });
    groupedPeople = Object.entries(takenMap);
  } catch (e) { console.error(e); }

  // Render module people
  const peopleEl = $('#pref-module-people');
  if (modulePeople.length) {
    peopleEl.innerHTML = modulePeople.map(u => renderPersonItem(u)).join('');
  } else {
    peopleEl.innerHTML = '<p class="pref-empty">No other students found for this module.</p>';
  }

  // Render grouped people
  const groupedEl = $('#pref-grouped');
  if (groupedPeople.length) {
    const groupedUsers = modulePeople.filter(u => groupedPeople.some(([mid]) => mid === u.id));
    const unknownGrouped = groupedPeople.filter(([mid]) => !modulePeople.some(u => u.id === mid));
    let html = groupedUsers.map(u => {
      const gInfo = groupedPeople.find(([mid]) => mid === u.id);
      return `<div class="pref-grouped-item">${avatar(u.displayName, u.photoURL, 'avatar-sm')}<div class="pref-person-info"><div class="pref-person-name">${esc(u.displayName)}</div><div class="pref-person-meta">In: ${esc(gInfo?.[1]?.groupName || 'a group')}</div></div></div>`;
    }).join('');
    if (!html) html = '<p class="pref-empty">No one grouped yet for this module.</p>';
    groupedEl.innerHTML = html;
  } else {
    groupedEl.innerHTML = '<p class="pref-empty">No existing groups for this module yet.</p>';
  }

  // Render friends
  const friendsEl = $('#pref-friends-list');
  const friends = state.profile.friends || [];
  const friendsInModule = modulePeople.filter(u => friends.includes(u.id));
  if (friendsInModule.length) {
    friendsEl.innerHTML = friendsInModule.map(u => renderPersonItem(u)).join('');
  } else {
    friendsEl.innerHTML = '<p class="pref-empty">None of your friends are doing this module.</p>';
  }

  // Search filter
  $('#pref-search').oninput = (e) => {
    const q = (e.target.value || '').toLowerCase();
    const filtered = q ? modulePeople.filter(u => (u.displayName || '').toLowerCase().includes(q)) : modulePeople;
    peopleEl.innerHTML = filtered.length ? filtered.map(u => renderPersonItem(u)).join('') : '<p class="pref-empty">No matches found.</p>';
  };

  // Save
  $('#pref-save').onclick = async () => {
    const wantUids = Object.entries(_prefChoices).filter(([, v]) => v === 'want').map(([k]) => k);
    const dontWantUids = Object.entries(_prefChoices).filter(([, v]) => v === 'dontwant').map(([k]) => k);
    const wantNames = wantUids.map(u => { const p = modulePeople.find(p => p.id === u); return p?.displayName || u; });
    const dontWantNames = dontWantUids.map(u => { const p = modulePeople.find(p => p.id === u); return p?.displayName || u; });
    try {
      await db.collection('assignmentGroups').doc(groupId).update({
        [`preferences.${uid}`]: { want: wantNames, dontWant: dontWantNames, wantUids, dontWantUids }
      });
      closeModal(); toast('Preferences saved');
    } catch (e) { toast('Failed'); console.error(e); }
  };
}

function openAsgChat(groupId) {
  closeModal();
  openGroupChat(groupId, 'assignmentGroups');
}

// ══════════════════════════════════════════════════
//  FRIEND REQUEST SYSTEM
// ══════════════════════════════════════════════════
async function sendFriendRequest(toUid, toName, toPhoto, btnEl = null) {
  if (toUid === state.user.uid) return toast("That's you!");
  const myFriends = state.profile.friends || [];
  if (myFriends.includes(toUid)) return toast('Already friends!');
  try {
    // Check if already sent
    const theirDoc = await db.collection('users').doc(toUid).get();
    const theirData = theirDoc.data();
    const theirRequests = theirData.friendRequests || [];
    if (theirRequests.some(r => r.uid === state.user.uid)) return toast('Request already sent');
    // Check if they sent US a request (auto-accept)
    const myRequests = state.profile.friendRequests || [];
    if (myRequests.some(r => r.uid === toUid)) {
      return acceptFriendRequest(toUid, toName, toPhoto);
    }
    // Add request to their profile
    await db.collection('users').doc(toUid).update({
      friendRequests: FieldVal.arrayUnion({
        uid: state.user.uid,
        name: state.profile.displayName,
        photo: state.profile.photoURL || '',
        timestamp: Date.now()
      })
    });
    // Track on our side
    await db.collection('users').doc(state.user.uid).update({
      sentRequests: FieldVal.arrayUnion(toUid)
    });
    state.profile.sentRequests = [...(state.profile.sentRequests || []), toUid];
    if (btnEl) {
      btnEl.textContent = 'Pending…';
      btnEl.disabled = true;
      btnEl.style.opacity = '0.6';
    }
    toast('Friend request sent!');
  } catch (e) { toast('Failed to send request'); console.error(e); }
}

async function acceptFriendRequest(fromUid, fromName, fromPhoto) {
  const uid = state.user.uid;
  try {
    // Add to both friends arrays
    await db.collection('users').doc(uid).update({
      friends: FieldVal.arrayUnion(fromUid),
      friendRequests: (state.profile.friendRequests || []).filter(r => r.uid !== fromUid)
    });
    await db.collection('users').doc(fromUid).update({
      friends: FieldVal.arrayUnion(uid),
      sentRequests: FieldVal.arrayRemove(uid)
    });
    await ensureFriendDMConversation(fromUid, fromName, fromPhoto);
    // Update local state
    state.profile.friends = [...(state.profile.friends || []), fromUid];
    state.profile.friendRequests = (state.profile.friendRequests || []).filter(r => r.uid !== fromUid);
    toast(`You and ${fromName} are now friends!`);
    // Auto-close dropdown if no more requests
    loadNotifications();
    if (!(state.profile.friendRequests || []).length) {
      const dd = $('#notif-dropdown'); if (dd) dd.style.display = 'none';
    }
  } catch (e) { toast('Failed'); console.error(e); }
}

async function ensureFriendDMConversation(otherUid, otherName, otherPhoto) {
  const myUid = state.user.uid;
  const snap = await db.collection('conversations').where('participants', 'array-contains', myUid).get();
  const existing = snap.docs.find(d => (d.data().participants || []).includes(otherUid));
  if (existing) return existing.id;
  const ref = await db.collection('conversations').add({
    participants: [myUid, otherUid],
    participantNames: [state.profile.displayName, otherName || 'Friend'],
    participantPhotos: [state.profile.photoURL || null, otherPhoto || null],
    lastMessage: '',
    updatedAt: FieldVal.serverTimestamp(),
    unread: { [otherUid]: 0, [myUid]: 0 },
    participantStatuses: { [myUid]: state.status || 'online', [otherUid]: 'offline' }
  });
  return ref.id;
}

async function rejectFriendRequest(fromUid) {
  const uid = state.user.uid;
  try {
    await db.collection('users').doc(uid).update({
      friendRequests: (state.profile.friendRequests || []).filter(r => r.uid !== fromUid)
    });
    await db.collection('users').doc(fromUid).update({
      sentRequests: FieldVal.arrayRemove(uid)
    });
    state.profile.friendRequests = (state.profile.friendRequests || []).filter(r => r.uid !== fromUid);
    toast('Request declined');
    loadNotifications();
    if (!(state.profile.friendRequests || []).length) {
      const dd = $('#notif-dropdown'); if (dd) dd.style.display = 'none';
    }
  } catch (e) { toast('Failed'); console.error(e); }
}

async function unfriend(targetUid) {
  const uid = state.user.uid;
  try {
    await db.collection('users').doc(uid).update({ friends: FieldVal.arrayRemove(targetUid) });
    await db.collection('users').doc(targetUid).update({ friends: FieldVal.arrayRemove(uid) });
    state.profile.friends = (state.profile.friends || []).filter(f => f !== targetUid);
    toast('Unfriended');
  } catch (e) { toast('Failed'); console.error(e); }
}

// ══════════════════════════════════════════════════
//  NOTIFICATIONS — Friend request accept/reject
// ══════════════════════════════════════════════════
let notifUnsub = null;
let generalNotifUnsub = null;
let _notifications = [];

function listenForNotifications() {
  if (notifUnsub) notifUnsub();
  if (generalNotifUnsub) generalNotifUnsub();

  notifUnsub = db.collection('users').doc(state.user.uid).onSnapshot(doc => {
    if (!doc.exists) return;
    const data = doc.data();
    state.profile.friends = data.friends || [];
    state.profile.friendRequests = sanitizeFriendRequests(data.friendRequests || []);
    state.profile.sentRequests = data.sentRequests || [];
    state.profile.blockedUsers = data.blockedUsers || [];
    state.profile.blockedBy = data.blockedBy || [];
    updateNotifBadge();
    const dd = $('#notif-dropdown');
    if (dd && dd.style.display === 'block') loadNotifications();
  });

  generalNotifUnsub = db.collection('users').doc(state.user.uid).collection('notifications')
    .limit(30)
    .onSnapshot(snap => {
      _notifications = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      _notifications.sort((a,b) => (b.createdAt?.seconds||0) - (a.createdAt?.seconds||0));
      _notifications = _notifications.slice(0, 20);
      updateNotifBadge();
      const dd = $('#notif-dropdown');
      if (dd && dd.style.display === 'block') loadNotifications();
    }, err => { console.warn('Notif listener error:', err); });
}

function updateNotifBadge() {
  const requests = sanitizeFriendRequests(state.profile.friendRequests || []);
  const unreadCount = _notifications.filter(n => !n.read).length;
  const pendingAsg = _asgPendingAlerts.reduce((sum, g) => sum + (g.pendingRequests || []).length, 0);
  const revealCount = _notifications.filter(n => n.type === 'reveal_request' && !n.read).length;
  const dot = $('#notif-dot');
  if (dot) dot.style.display = (requests.length > 0 || unreadCount > 0 || pendingAsg > 0 || revealCount > 0) ? 'block' : 'none';
}

function loadNotifications() {
  const dd = $('#notif-dropdown');
  const requests = sanitizeFriendRequests(state.profile.friendRequests || []);
  const asgAlerts = _asgPendingAlerts;
  const notifs = _notifications;
  
  // Separate reveal requests (top priority)
  const revealRequests = notifs.filter(n => n.type === 'reveal_request');
  const otherNotifs = notifs.filter(n => n.type !== 'reveal_request');

  if (!requests.length && !asgAlerts.length && !revealRequests.length && !otherNotifs.length) {
    dd.innerHTML = `
      <div class="notif-header"><h3>Notifications</h3></div>
      <div style="padding:32px;text-align:center;color:var(--text-tertiary)">
        <div style="font-size:32px;margin-bottom:8px">🔔</div>
        <p>No new notifications</p>
      </div>`;
    return;
  }

  let html = '<div class="notif-header"><h3>Notifications</h3></div><div class="notif-scroll" style="max-height:400px;overflow-y:auto">';
  
  // Show reveal requests FIRST (top priority)
  if (revealRequests.length) {
    html += `<div style="padding:8px 16px;font-weight:600;font-size:13px;color:var(--text-secondary)">🎭 Identity Reveal Requests</div>`;
    html += revealRequests.map(n => {
      const from = n.from || { name: 'Someone', photo: null };
      const convoId = n.payload?.convoId || '';
      return `
       <div class="notif-item ${n.read ? '' : 'unread'}" onclick="openChat('${convoId}');markNotifRead('${n.id}')">
         <div style="position:relative">
           <div class="avatar-md anon-avatar">👻</div>
           <div style="position:absolute;bottom:-2px;right:-2px;font-size:12px;background:var(--bg-secondary);border-radius:50%;padding:2px">🎭</div>
         </div>
         <div class="notif-content">
           <div class="notif-text"><strong>Anonymous contact</strong> wants to reveal their identity</div>
           <div class="notif-time">${timeAgo(n.createdAt)}</div>
         </div>
       </div>`;
    }).join('');
  }

  if (requests.length) {
    html += `<div style="padding:8px 16px;font-weight:600;font-size:13px;color:var(--text-secondary)">Friend Requests</div>`;
    html += requests.map(r => `
      <div class="notif-item unread" onclick="openProfile('${r.uid}')">
        ${avatar(r.name, r.photo, 'avatar-md')}
        <div class="notif-content">
          <div class="notif-text"><strong>${esc(r.name)}</strong> sent you a friend request</div>
          <div class="notif-actions">
            <button class="btn-primary btn-sm" onclick="event.stopPropagation();acceptFriendRequest('${r.uid}','${esc(r.name)}','${r.photo || ''}')">Accept</button>
            <button class="btn-outline btn-sm" onclick="event.stopPropagation();rejectFriendRequest('${r.uid}')">Decline</button>
          </div>
        </div>
      </div>`).join('');
  }

  if (asgAlerts.length) {
    if (requests.length) html += `<div style="height:1px;background:var(--border);margin:8px 0"></div>`;
    html += `<div style="padding:8px 16px;font-weight:600;font-size:13px;color:var(--text-secondary)">Group Requests</div>`;
    html += asgAlerts.map(g => `
      <div class="notif-item unread" onclick="openGroupDetail('${g.id}')">
        <div style="position:relative">
          <div class="avatar-md group-avatar-icon">📋</div>
          <div style="position:absolute;bottom:-2px;right:-2px;font-size:12px;background:var(--bg-secondary);border-radius:50%;padding:2px">⏳</div>
        </div>
        <div class="notif-content">
          <div class="notif-text"><strong>${esc(g.groupTitle || g.assignmentTitle || g.moduleCode || 'Group')}</strong> has ${(g.pendingRequests || []).length} pending request${(g.pendingRequests || []).length === 1 ? '' : 's'}</div>
          <div class="notif-time">Tap to review</div>
          <div class="notif-actions" style="margin-top:6px;display:flex;flex-direction:column;gap:6px">
            ${(g.pendingRequests || []).slice(0, 3).map(r => `
              <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;background:var(--bg-tertiary);border-radius:10px;padding:6px 8px">
                <span style="font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:120px">${esc(r.name || 'Student')}</span>
                <div style="display:flex;gap:6px">
                  <button class="btn-primary btn-sm" onclick="event.stopPropagation();approveAsgRequestByUid('${g.id}','${r.uid}')">Accept</button>
                  <button class="btn-outline btn-sm" onclick="event.stopPropagation();rejectAsgRequest('${g.id}','${r.uid}')">Decline</button>
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `).join('');
  }

  if (otherNotifs.length) {
    if (requests.length || asgAlerts.length || revealRequests.length) html += `<div style="height:1px;background:var(--border);margin:8px 0"></div>`;
    html += otherNotifs.map(n => {
      const icon = n.type === 'like' ? '❤️' : n.type === 'comment' ? '💬' : n.type === 'module' ? '📚' : n.type === 'group' ? '📋' : '🔔';
      const clickAction = n.payload?.postId
        ? `viewPost('${n.payload.postId}');markNotifRead('${n.id}')`
        : n.payload?.groupId
          ? `openGroupDetail('${n.payload.groupId}');markNotifRead('${n.id}')`
          : `markNotifRead('${n.id}')`;
      const from = n.from || { name: 'Unino', photo: null };
      return `
       <div class="notif-item ${n.read ? '' : 'unread'}" onclick="${clickAction}">
         <div style="position:relative">
           ${avatar(from.name, from.photo, 'avatar-md')}
           <div style="position:absolute;bottom:-2px;right:-2px;font-size:12px;background:var(--bg-secondary);border-radius:50%;padding:2px">${icon}</div>
         </div>
         <div class="notif-content">
           <div class="notif-text"><strong>${esc(from.name)}</strong> ${esc(n.text)}</div>
           <div class="notif-time">${timeAgo(n.createdAt)}</div>
         </div>
       </div>`;
    }).join('');
  }

  html += '</div>';
  dd.innerHTML = html;
}

async function markNotifRead(nid) {
  try { await db.collection('users').doc(state.user.uid).collection('notifications').doc(nid).update({ read: true }); } catch (e) { }
}

async function addNotification(targetId, type, text, payload) {
  if (targetId === state.user.uid) return;
  try {
    await db.collection('users').doc(targetId).collection('notifications').add({
      type, text, payload, read: false, createdAt: FieldVal.serverTimestamp(),
      from: { uid: state.user.uid, name: state.profile.displayName, photo: state.profile.photoURL || null }
    });
  } catch (e) { console.error(e); }
}

async function viewPost(pid) {
  // Show the post in a modal so tapping a like notification opens the actual post
  try {
    const doc = await db.collection('posts').doc(pid).get();
    if (!doc.exists) return toast('Post not found');
    const p = { id: doc.id, ...doc.data() };
    const liked = (p.likes || []).includes(state.user.uid);
    const lc = (p.likes || []).length, cc = p.commentsCount || 0;
    const hasVideo = p.videoURL || (p.mediaType === 'video');
    const hasImage = p.imageURL && !hasVideo;
    const mediaURL = hasVideo ? (p.videoURL || p.imageURL) : p.imageURL;

    let videoPlayerData = null;
    if (hasVideo && mediaURL) { videoPlayerData = createVideoPlayer(mediaURL); }

    openModal(`
      <div class="modal-header"><h2>Post</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
      <div class="modal-body" style="padding:16px">
        <div class="post-card" style="box-shadow:none;border:none;margin:0;padding:0">
          ${p.repostOf ? `<div class="repost-badge" style="margin:-0 -0 10px">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
            <span>Reposted</span>
          </div>` : ''}
          <div class="post-header">
            ${p.isAnonymous ? `<div class="avatar-md anon-avatar" onclick="closeModal();openAnonPostActions('${p.authorId}')" style="cursor:pointer">👻</div>` : `<div onclick="closeModal();openProfile('${p.authorId}')" style="cursor:pointer">${avatar(p.authorName, p.authorPhoto, 'avatar-md')}</div>`}
            <div class="post-header-info">
              <div class="post-author-name" ${p.isAnonymous ? `onclick="closeModal();openAnonPostActions('${p.authorId}')" style="cursor:pointer"` : `onclick="closeModal();openProfile('${p.authorId}')"`}>${p.isAnonymous ? '👻 Anonymous' : esc(p.authorName || 'User') + verifiedBadge(p.authorId)}</div>
              <div class="post-meta">${timeAgo(p.createdAt)}</div>
            </div>
          </div>
          ${p.content ? renderExpandablePostContent(p.content, `modal-${p.id}`, 180) : ''}
          ${renderPostModuleTags(p.moduleTags || [])}
          ${renderPostHashTags(getPostHashTags(p).filter(tag => !(p.moduleTags || []).includes(tag.toUpperCase())))}
          ${hasImage && mediaURL ? `<div class="post-media-wrap"><img src="${mediaURL}" class="post-image" onclick="viewImage('${mediaURL}')" style="max-height:300px"></div>` : ''}
          ${!p.repostOf && hasVideo && videoPlayerData ? videoPlayerData.html : ''}
          ${p.repostOf ? renderQuoteEmbed(p.repostOf) : ''}
          <div class="post-actions" style="border-top:1px solid var(--border);padding-top:12px;margin-top:12px">
            ${p.isAnonymous && p.authorId !== state.user.uid ? `<button class="post-action anon-inline-action" onclick="closeModal();openAnonPostActions('${p.authorId}')">👻 Message</button>` : ''}
            <button class="post-action ${liked ? 'liked' : ''}" onclick="toggleLike('${p.id}');closeModal()">❤ ${lc || 'Like'}</button>
            <button class="post-action" onclick="closeModal();openComments('${p.id}')">💬 ${cc || 'Comment'}</button>
            <button class="post-action" onclick="closeModal();openShareModal('${p.id}')">↗ Share</button>
          </div>
        </div>
      </div>
    `);

    // Init video players inside modal
    requestAnimationFrame(() => {
      if (videoPlayerData) initPlayer(videoPlayerData.id);
      _pendingQuotePlayers.forEach(p => initPlayer(p.id));
      _pendingQuotePlayers.length = 0;
    });
  } catch (e) { console.error(e); toast('Could not load post'); }
}

// ─── DM List ─────────────────────────────────────
function loadDMList() {
  const container = $('#msg-tab-content'); if (!container) return;
  updateArchiveFabState();
  container.innerHTML = `<div class="convo-list" id="convo-list"><div style="padding:40px;text-align:center"><span class="inline-spinner"></span></div></div>`;

  // KEY FIX: No .orderBy() — sort client-side to avoid Firestore index requirement
  unsub(); // clear old listeners before adding new
  const u = db.collection('conversations')
    .where('participants', 'array-contains', state.user.uid)
    .onSnapshot(snap => {
      const convos = snap.docs
        .map(d => ({ id: d.id, ...d.data() }))
        .sort((a, b) => (b.updatedAt?.seconds || 0) - (a.updatedAt?.seconds || 0));

      const el = $('#convo-list'); if (!el) return;

      if (!convos.length) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">💬</div><h3>No chats yet</h3><p>Visit a profile to start a conversation</p></div>`;
        return;
      }

      const uid = state.user.uid;
      el.innerHTML = convos.map(c => {
        const idx = c.participants.indexOf(uid) === 0 ? 1 : 0;
        const otherUid = c.participants[idx];
        const rawName = (c.participantNames || [])[idx] || 'User';
        const rawPhoto = (c.participantPhotos || [])[idx] || null;
        const otherStatus = (c.participantStatuses || {})[otherUid] || 'offline';
        const theirAnon = !!((c.anonymous || {})[otherUid]);
        const displayName = theirAnon ? getAnonDisplayName(c, uid, otherUid) : rawName;
        const displayPhoto = theirAnon ? null : rawPhoto;
        const avatarHtml = theirAnon
          ? '<div class="avatar-md anon-avatar">👻</div>'
          : avatar(displayName, displayPhoto, 'avatar-md');
        const unread = (c.unread || {})[uid] || 0;
        const isArchived = (c.archived || []).includes(uid);
        if (isArchived) return '';
        return `
          <div class="convo-item ${unread ? 'unread' : ''}" onclick="openChat('${c.id}')" oncontextmenu="event.preventDefault();showConvoActions('${c.id}','${esc(displayName)}','${otherUid}')">
            <div class="convo-avatar">${avatarHtml}${otherStatus === 'online' ? '<span class="online-indicator"></span>' : ''}</div>
            <div class="convo-info">
              <div class="convo-name">${esc(displayName)}</div>
              <div class="convo-last-msg">${esc(c.lastMessage || 'Start chatting...')}</div>
            </div>
            <div class="convo-right">
              <div class="convo-time">${timeAgo(c.updatedAt)}</div>
              ${unread ? `<div class="convo-unread-badge">${unread}</div>` : ''}
              <button class="convo-menu-btn" onclick="event.stopPropagation();showConvoActions('${c.id}','${esc(displayName)}','${otherUid}')" style="margin-top:4px">⋯</button>
            </div>
          </div>`;
      }).join('');
    }, err => {
      console.error('Messages query error:', err);
      const el = $('#convo-list');
      if (el) el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">💬</div><h3>No chats yet</h3><p>Visit a profile to start a conversation</p></div>`;
    });
  state.unsubs.push(u);
}

// ─── Archived DM List ─────────────────────────────────────
function loadArchivedDMList() {
  const container = $('#msg-tab-content'); if (!container) return;
  updateArchiveFabState();
  container.innerHTML = `<div class="convo-list" id="convo-list"><div style="padding:40px;text-align:center"><span class="inline-spinner"></span></div></div>`;

  unsub();
  const u = db.collection('conversations')
    .where('participants', 'array-contains', state.user.uid)
    .onSnapshot(snap => {
      const convos = snap.docs
        .map(d => ({ id: d.id, ...d.data() }))
        .filter(c => (c.archived || []).includes(state.user.uid))
        .sort((a, b) => (b.updatedAt?.seconds || 0) - (a.updatedAt?.seconds || 0));

      const el = $('#convo-list'); if (!el) return;

      if (!convos.length) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">📦</div><h3>No archived chats</h3><p>Archived conversations will appear here</p></div>`;
        return;
      }

      const uid = state.user.uid;
      el.innerHTML = convos.map(c => {
        const idx = c.participants.indexOf(uid) === 0 ? 1 : 0;
        const otherUid = c.participants[idx];
        const rawName = (c.participantNames || [])[idx] || 'User';
        const rawPhoto = (c.participantPhotos || [])[idx] || null;
        const otherStatus = (c.participantStatuses || {})[otherUid] || 'offline';
        const theirAnon = !!((c.anonymous || {})[otherUid]);
        const displayName = theirAnon ? getAnonDisplayName(c, uid, otherUid) : rawName;
        const displayPhoto = theirAnon ? null : rawPhoto;
        const avatarHtml = theirAnon
          ? '<div class="avatar-md anon-avatar">👻</div>'
          : avatar(displayName, displayPhoto, 'avatar-md');
        const unread = (c.unread || {})[uid] || 0;
        return `
          <div class="convo-item ${unread ? 'unread' : ''}" onclick="openChat('${c.id}')">
            <div class="convo-avatar">${avatarHtml}${otherStatus === 'online' ? '<span class="online-indicator"></span>' : ''}</div>
            <div class="convo-info">
              <div class="convo-name">${esc(displayName)}</div>
              <div class="convo-last-msg">${esc(c.lastMessage || 'Start chatting...')}</div>
            </div>
            <div class="convo-right">
              <div class="convo-time">${timeAgo(c.updatedAt)}</div>
              ${unread ? `<div class="convo-unread-badge">${unread}</div>` : ''}
              <button class="btn-sm btn-outline" onclick="event.stopPropagation();unarchiveConvo('${c.id}')" style="margin-top:4px;padding:4px 8px;font-size:11px">Unarchive</button>
            </div>
          </div>`;
      }).join('');
    }, err => {
      console.error('Archived messages query error:', err);
      const el = $('#convo-list');
      if (el) el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">📦</div><h3>No archived chats</h3></div>`;
    });
  state.unsubs.push(u);
}

async function unarchiveConvo(convoId) {
  try {
    await db.collection('conversations').doc(convoId).update({
      archived: FieldVal.arrayRemove(state.user.uid)
    });
    toast('Chat unarchived');
    refreshCurrentMessageList();
  } catch (e) { toast('Failed to unarchive'); console.error(e); }
}

// ─── Chat View ───────────────────────────────────
let chatUnsub = null;
let _anonUnsub = null;

function getAnonState(convo, uid) {
  const anonMap = convo.anonymous || {};
  return {
    meAnon: !!anonMap[uid],
    themAnon: !!anonMap[Object.keys(anonMap).find(k => k !== uid)],
    revealRequests: convo.revealRequests || {}
  };
}

function updateAnonUI(convo, uid, realName, realPhoto) {
  const { meAnon, themAnon, revealRequests } = getAnonState(convo, uid);
  const otherUid = convo.participants.find(p => p !== uid);
  const idx = convo.participants.indexOf(uid) === 0 ? 1 : 0;
  const otherName = (convo.participantNames || [])[idx] || 'User';
  const otherPhoto = (convo.participantPhotos || [])[idx] || null;
  const otherStatus = (convo.participantStatuses || {})[otherUid] || 'offline';
  const anonName = getAnonDisplayName(convo, uid, otherUid);

  // Update header display
  const hdrInfo = $('#chat-hdr-info');
  if (themAnon) {
    hdrInfo.innerHTML = `
      <div class="avatar-sm anon-avatar">👻</div>
      <div><div style="display:flex;align-items:center;gap:8px"><h3 style="font-size:15px;font-weight:700">${esc(anonName)}</h3><button class="anon-name-edit" onclick="editAnonNickname('${convo._id}')">✎</button></div>
      <span style="font-size:11px;color:var(--text-tertiary)">${otherStatus === 'online' ? 'Online' : otherStatus === 'study' ? 'Studying' : 'Offline'} · Identity hidden</span></div>
    `;
  } else {
    hdrInfo.innerHTML = `
      ${avatar(otherName, otherPhoto, 'avatar-sm')}
      <div><h3 style="font-size:15px;font-weight:700">${esc(otherName)}</h3><span style="font-size:11px;color:var(--text-tertiary)">${otherStatus === 'online' ? 'Online' : otherStatus === 'study' ? 'Studying' : 'Offline'}</span></div>
    `;
  }

  // Update toggle button visual
  const toggleBtn = $('#chat-anon-toggle');
  if (toggleBtn) {
    toggleBtn.querySelector('.anon-icon-off').style.display = meAnon ? 'none' : 'block';
    toggleBtn.querySelector('.anon-icon-on').style.display = meAnon ? 'block' : 'none';
    toggleBtn.classList.toggle('anon-active', meAnon);
  }

  // Update reveal banner
  const banner = $('#anon-reveal-banner');
  if (banner) {
    if (meAnon && themAnon) {
      const iRequested = !!revealRequests[uid];
      const theyRequested = !!revealRequests[otherUid];
      if (iRequested && theyRequested) {
        // Mutual reveal! Both requested
        banner.innerHTML = `<span>🎉 Both revealed! You can now see each other.</span>`;
        banner.className = 'anon-reveal-banner reveal-success';
        banner.style.display = 'flex';
        // Auto-remove anonymous for both
        db.collection('conversations').doc(convo._id).update({
          [`anonymous.${uid}`]: false,
          [`anonymous.${otherUid}`]: false,
          revealRequests: {}
        }).catch(() => {});
      } else if (iRequested) {
        banner.innerHTML = `<span>⏳ Waiting for them to reveal too...</span>`;
        banner.className = 'anon-reveal-banner reveal-waiting';
        banner.style.display = 'flex';
      } else if (theyRequested) {
        banner.innerHTML = `<span>👀 They want to reveal!</span><button class="btn-primary btn-sm" onclick="mutualReveal('${convo._id}')">Reveal Together</button>`;
        banner.className = 'anon-reveal-banner reveal-request';
        banner.style.display = 'flex';
      } else {
        banner.innerHTML = `<span>🎭 Both anonymous</span><button class="btn-outline btn-sm" onclick="requestReveal('${convo._id}')">Request Mutual Reveal</button>`;
        banner.className = 'anon-reveal-banner reveal-available';
        banner.style.display = 'flex';
      }
    } else if (meAnon || themAnon) {
      banner.innerHTML = `<span>🎭 ${meAnon ? 'You are' : 'They are'} anonymous</span>`;
      banner.className = 'anon-reveal-banner';
      banner.style.display = 'flex';
    } else {
      banner.style.display = 'none';
    }
  }
}

async function toggleAnonymous(convoId) {
  const uid = state.user.uid;
  try {
    const doc = await db.collection('conversations').doc(convoId).get();
    if (!doc.exists) return;
    const convo = doc.data();
    const currentAnon = (convo.anonymous || {})[uid] || false;
    await db.collection('conversations').doc(convoId).update({
      [`anonymous.${uid}`]: !currentAnon,
      // Clear reveal requests when toggling
      revealRequests: {}
    });
    toast(!currentAnon ? 'You are now Anonymous 👻' : 'Identity revealed');
  } catch (e) { toast('Failed'); console.error(e); }
}

async function requestReveal(convoId) {
  const uid = state.user.uid;
  try {
    const snap = await db.collection('conversations').doc(convoId).get();
    if (!snap.exists) return;
    const convo = snap.data();
    const otherUid = convo.participants.find(p => p !== uid);
    if (!otherUid) return;
    
    // Mark my reveal request
    await db.collection('conversations').doc(convoId).update({
      [`revealRequests.${uid}`]: true
    });
    
    // Create notification for the other user
    await db.collection('users').doc(otherUid).collection('notifications').add({
      type: 'reveal_request',
      text: 'Someone wants to reveal their identity in an anonymous chat',
      payload: { convoId },
      read: false,
      createdAt: FieldVal.serverTimestamp(),
      from: { uid: 'anonymous', name: 'Anonymous', photo: null }
    });
    
    toast('Reveal request sent!');
  } catch (e) { toast('Failed'); console.error(e); }
}

async function mutualReveal(convoId) {
  const uid = state.user.uid;
  try {
    await db.collection('conversations').doc(convoId).update({
      [`revealRequests.${uid}`]: true
    });
  } catch (e) { toast('Failed'); console.error(e); }
}

async function openChat(convoId) {
  try {
    const convoDoc = await db.collection('conversations').doc(convoId).get();
    if (!convoDoc.exists) return toast('Chat not found');
    const convo = convoDoc.data();
    convo._id = convoId;
    const uid = state.user.uid;
    const idx = convo.participants.indexOf(uid) === 0 ? 1 : 0;
    const name = (convo.participantNames || [])[idx] || 'User';
    const photo = (convo.participantPhotos || [])[idx] || null;

    showScreen('chat-view');
    
    // IMMEDIATELY clear old messages to prevent flash of previous chat
    const msgs = $('#chat-msgs');
    if (msgs) msgs.innerHTML = '<div style="text-align:center;padding:32px"><span class="inline-spinner"></span></div>';

    // Only show anon toggle for anonymous conversations (non-friend chats)
    const anonBtn = $('#chat-anon-toggle');
    if (anonBtn) {
      if (convo.isAnonymous) {
        anonBtn.style.display = '';
        anonBtn.onclick = () => toggleAnonymous(convoId);
      } else {
        anonBtn.style.display = 'none';
      }
    }

    // Set initial header (will be updated by anon listener)
    updateAnonUI(convo, uid, name, photo);

    // Listen for anonymous state changes in real-time
    if (_anonUnsub) _anonUnsub();
    _anonUnsub = db.collection('conversations').doc(convoId).onSnapshot(snap => {
      if (!snap.exists) return;
      const liveConvo = snap.data();
      liveConvo._id = convoId;
      updateAnonUI(liveConvo, uid, name, photo);
    });

    // Mark as read
    db.collection('conversations').doc(convoId).set({ unread: { [uid]: 0 } }, { merge: true }).catch(() => {});

    // Messages listener
    if (chatUnsub) chatUnsub();
    if (_chatViewportCleanup) { _chatViewportCleanup(); _chatViewportCleanup = null; }
    _dmReplyTo = null;
    const dmReply = $('#dm-reply-indicator');
    if (dmReply) dmReply.style.display = 'none';
    _chatViewportCleanup = setupViewportFollow(msgs);
    chatUnsub = db.collection('conversations').doc(convoId)
      .collection('messages').orderBy('createdAt', 'asc').limit(100)
      .onSnapshot(snap => {
        const messages = snap.docs.map(d => ({ id: d.id, ...d.data() }));
        _dmMsgLookup = new Map(messages.map(m => [m.id, m]));
        if (!messages.length) {
          msgs.innerHTML = '<div style="text-align:center;padding:32px;opacity:0.5">Say hi! 👋</div>';
        } else {
          let lastDateLabel = '';
          msgs.innerHTML = messages.map((m, idx) => {
            const isMe = m.senderId === uid;
            let content = '';
            if (m.audioURL) content += renderVoiceMsg(m.audioURL);
            if (m.imageURL) content += `<img src="${m.imageURL}" class="msg-image" onclick="viewImage('${m.imageURL}')">`;
            // Handle shared post messages
            if (m.type === 'share_post' && m.payload?.postId) {
              const pl = m.payload;
              let mediaPreview = '';
              if (pl.mediaURL && pl.mediaType === 'video') {
                mediaPreview = `<div style="position:relative;border-radius:8px;overflow:hidden;margin-bottom:6px;max-height:140px">
                  <video src="${pl.mediaURL}" style="width:100%;max-height:140px;object-fit:cover;display:block" preload="metadata" muted playsinline></video>
                  <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.25)">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="white"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                  </div>
                </div>`;
              } else if (pl.mediaURL && (pl.mediaType === 'image' || !pl.mediaType)) {
                mediaPreview = `<div style="border-radius:8px;overflow:hidden;margin-bottom:6px;max-height:140px"><img src="${pl.mediaURL}" style="width:100%;max-height:140px;object-fit:cover;display:block"></div>`;
              }
              let snippetHTML = pl.content ? `<div style="font-size:12px;color:var(--text-secondary);margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(pl.content)}</div>` : '';
              content = `<div class="shared-post-card" onclick="viewPost('${pl.postId}')">
                ${mediaPreview}
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/></svg>
                  <span style="font-size:12px;font-weight:600;color:var(--accent)">Shared Post</span>
                  ${pl.authorName ? `<span style="font-size:11px;color:var(--text-tertiary)">by ${esc(pl.authorName)}</span>` : ''}
                </div>
                ${snippetHTML}
                <div style="font-size:11px;color:var(--text-tertiary)">Tap to view</div>
              </div>`;
            } else if (m.text && !m.text.startsWith('shared post::')) {
              content += esc(m.text);
            } else if (m.text && m.text.startsWith('shared post::')) {
              const spId = m.text.replace('shared post::','');
              content = `<div class="shared-post-card" onclick="viewPost('${spId}')">
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/></svg>
                  <span style="font-size:12px;font-weight:600;color:var(--accent)">Shared Post</span>
                </div>
                <div style="font-size:13px;color:var(--text-secondary)">Tap to view post</div>
              </div>`;
            }
            const ts = m.createdAt || m.timestamp;

            // Date separator
            let dateSep = '';
            const curLabel = dateSeparatorLabel(ts);
            if (curLabel && curLabel !== lastDateLabel) {
              lastDateLabel = curLabel;
              dateSep = `<div class="chat-date-sep"><span>${curLabel}</span></div>`;
            }

            // Delivery status for sent messages
            let statusIcon = '';
            if (isMe) {
              const st = m.status || 'sent';
              if (st === 'read') statusIcon = '<span class="msg-status read" title="Read">✓✓</span>';
              else if (st === 'delivered') statusIcon = '<span class="msg-status delivered" title="Delivered">✓✓</span>';
              else statusIcon = '<span class="msg-status sent" title="Sent">✓</span>';
            }

            // Determine anonymous display
            const senderAnon = m.senderAnon || false;
            const displayName = (!isMe && senderAnon) ? 'Anonymous' : name;
            const displayPhoto = (!isMe && senderAnon) ? null : photo;
            const avatarHTML = senderAnon && !isMe
              ? '<div class="avatar-xs anon-avatar">👻</div>'
              : avatar(displayName, displayPhoto, 'avatar-xs');

            // Support both new and legacy replies: infer original sender from replyToId when needed.
            const replyToSenderId = m.replyToSenderId || _dmMsgLookup.get(m.replyToId || '')?.senderId;
            const replyDisplayName = replyToSenderId === uid ? 'me' : (m.replyToName || 'Message');
            const replyMeta = m.replyToText
              ? `<div class="msg-reply-snippet">↩ ${esc(replyDisplayName)}: ${esc(clampText(m.replyToText, 50))}</div>`
              : '';
            const newCls = (idx === messages.length - 1 && isMe) ? 'msg-new' : '';
            return `${dateSep}<div class="msg-row ${isMe ? 'msg-row-sent' : 'msg-row-received'}" id="msg-${m.id}">
              ${!isMe ? `<div class="msg-avatar-wrap">${avatarHTML}</div>` : ''}
              <div class="msg-bubble ${isMe ? 'msg-sent' : 'msg-received'} ${newCls}">${m.replyToId && m.replyToText ? `<div class="msg-reply-snippet" onclick="jumpToMessage('${m.replyToId}','chat-msgs')">↩ ${esc(replyDisplayName)}: ${esc(clampText(m.replyToText, 50))}</div>` : ''}${content}<button class="msg-reply-btn" title="Reply" aria-label="Reply" onclick="setDmReply('${m.id}')"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 17 4 12 9 7"></polyline><path d="M20 18v-2a4 4 0 0 0-4-4H4"></path></svg></button><div class="msg-time">${ts ? chatTime(ts) : ''}${statusIcon}</div></div>
            </div>`;
          }).join('');
          scrollToLatest(msgs);
        }

        // Mark incoming as read
        const incoming = messages.filter(m => m.senderId !== uid && m.status !== 'read');
        incoming.forEach(m => {
          db.collection('conversations').doc(convoId).collection('messages').doc(m.id).update({ status: 'read' }).catch(() => {});
        });
      });

    // Send message + image
    const input = $('#chat-input');
    let chatPendingImg = null;
    const resizeDmInput = () => {
      if (!input) return;
      input.style.height = '40px';
      input.style.height = `${Math.min(input.scrollHeight, 84)}px`;
      scrollToLatest(msgs);
    };
    if (input) {
      input.style.height = '40px';
      input.oninput = resizeDmInput;
    }

    const sendMsg = async () => {
      // Check if blocked before sending
      const otherUid = convo.participants.find(p => p !== uid);
      const myBlocked = state.profile.blockedUsers || [];
      const blockedBy = state.profile.blockedBy || [];
      if (myBlocked.includes(otherUid)) {
        toast('You blocked this user');
        return;
      }
      if (blockedBy.includes(otherUid)) {
        toast('This user has blocked you');
        return;
      }
      
      const text = input.value.trim();
      let img = chatPendingImg;
      const chatFile = window._chatFile || null;
      if (!text && !img && !window._chatVoiceBlob) return;
      const replyPayload = _dmReplyTo ? {
        replyToId: _dmReplyTo.id,
        replyToText: _dmReplyTo.text,
        replyToName: _dmReplyTo.name,
        replyToSenderId: _dmReplyTo.senderId
      } : {};
      input.value = ''; chatPendingImg = null; window._chatFile = null;
      resizeDmInput();
      input.focus();
      _dmReplyTo = null;
      if (dmReply) dmReply.style.display = 'none';
      const preview = $('#chat-img-preview'); if (preview) preview.style.display = 'none';
      try {
        // Upload image to R2 if file exists
        let imageURL = null;
        if (chatFile) { imageURL = await uploadToR2(chatFile, 'chat-images'); }
        // Upload voice to R2
        let audioURL = null;
        if (window._chatVoiceBlob) {
          const af = new File([window._chatVoiceBlob], `voice_${Date.now()}.webm`, { type: 'audio/webm' });
          audioURL = await uploadToR2(af, 'voice');
          window._chatVoiceBlob = null;
          const vrec = $('#voice-recorder'); if (vrec) vrec.style.display = 'none';
        }
        // Check if sender is currently anonymous
        let senderAnon = false;
        try {
          const freshConvo = await db.collection('conversations').doc(convoId).get();
          if (freshConvo.exists) senderAnon = !!(freshConvo.data().anonymous || {})[uid];
        } catch (_) {}
        await db.collection('conversations').doc(convoId).collection('messages').add({
          text: text || '', imageURL: imageURL || null, audioURL: audioURL || null,
          senderId: uid, senderAnon, ...replyPayload,
          createdAt: FieldVal.serverTimestamp(), status: 'sent'
        });
        const lastMsg = audioURL ? '🎤 Voice' : imageURL ? (text || '📷 Photo') : text;
        await db.collection('conversations').doc(convoId).set({
          lastMessage: lastMsg, updatedAt: FieldVal.serverTimestamp(),
          unread: { [otherUid]: FieldVal.increment(1), [uid]: 0 }
        }, { merge: true });
      } catch (e) { console.error(e); }
    };
    $('#chat-send').onclick = sendMsg;
    input.onkeydown = e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMsg();
      }
    };
    input.onfocus = () => scrollToLatest(msgs);
    input.onblur = () => setTimeout(() => scrollToLatest(msgs), 80);

    // Wire image upload button in chat
    const chatFileInput = $('#chat-file-input');
    if (chatFileInput) {
      chatFileInput.onchange = async e => {
        if (e.target.files[0]) {
          window._chatFile = e.target.files[0];
          chatPendingImg = localPreview(e.target.files[0]);
          const preview = $('#chat-img-preview');
          if (preview) { preview.querySelector('img').src = chatPendingImg; preview.style.display = 'block'; }
        }
      };
    }

    // Back button
    $('#chat-back').onclick = () => {
      if (chatUnsub) { chatUnsub(); chatUnsub = null; }
      if (_anonUnsub) { _anonUnsub(); _anonUnsub = null; }
      if (_chatViewportCleanup) { _chatViewportCleanup(); _chatViewportCleanup = null; }
      const banner = $('#anon-reveal-banner'); if (banner) banner.style.display = 'none';
      showScreen('app');
      // Don't navigate, just ensure current state is right
      if (state.page !== 'chat') navigate('chat');
    };
  } catch (e) { console.error(e); toast('Could not open chat'); }
}

async function startChat(uid, name, photo) {
  if (uid === state.user.uid) return toast("That's you!");
  const myFriends = state.profile.friends || [];
  if (!myFriends.includes(uid)) return toast('Add as friend first to message');
  
  // Check if blocked
  const myBlocked = state.profile.blockedUsers || [];
  const blockedBy = state.profile.blockedBy || [];
  if (myBlocked.includes(uid)) return toast('You blocked this user');
  if (blockedBy.includes(uid)) return toast('Cannot message this user');
  
  try {
    const snap = await db.collection('conversations').where('participants', 'array-contains', state.user.uid).get();
    const existing = snap.docs.find(d => d.data().participants.includes(uid));
    if (existing) { openChat(existing.id); }
    else {
      const doc = await db.collection('conversations').add({
        participants: [state.user.uid, uid],
        participantNames: [state.profile.displayName, name],
        participantPhotos: [state.profile.photoURL || null, photo || null],
        lastMessage: '', updatedAt: FieldVal.serverTimestamp(),
        unread: { [uid]: 0, [state.user.uid]: 0 },
        participantStatuses: { [state.user.uid]: state.status, [uid]: 'offline' }
      });
      openChat(doc.id);
    }
  } catch (e) { toast('Could not start chat'); console.error(e); }
}

// Anonymous messaging for non-friends (shy users)
const ANON_DAILY_LIMIT = 5;

async function startAnonChat(uid, name, photo, forceNew = false, replyToPostId = null) {
  if (uid === state.user.uid) return toast("That's you!");
  
  // Check if blocked (anon chats also blocked)
  const myBlocked = state.profile.blockedUsers || [];
  const blockedBy = state.profile.blockedBy || [];
  if (myBlocked.includes(uid)) return toast('You blocked this user');
  if (blockedBy.includes(uid)) return toast('This user has blocked anonymous messages');

  // Check daily anonymous usage limit
  const anonCount = state.profile.anonUsageToday || 0;
  const lastAnonDate = state.profile.anonUsageDate || '';
  const today = new Date().toISOString().split('T')[0];
  const todayCount = (lastAnonDate === today) ? anonCount : 0;

  if (todayCount >= ANON_DAILY_LIMIT) {
    return toast(`Anonymous limit reached (${ANON_DAILY_LIMIT}/day). Try again tomorrow!`);
  }

  try {
    let targetName = name;
    let targetPhoto = photo;
    if (!targetName || targetName === 'Anonymous' || targetName.includes('Anonymous')) {
      const userDoc = await db.collection('users').doc(uid).get();
      if (userDoc.exists) {
        const userData = userDoc.data() || {};
        targetName = userData.displayName || userData.firstName || 'User';
        targetPhoto = userData.photoURL || '';
      }
    }

    // Check for existing anon conversation
    const snap = await db.collection('conversations').where('participants', 'array-contains', state.user.uid).get();
    const existing = forceNew ? null : snap.docs.find(d => {
      const data = d.data();
      const stillAnonymous = !!((data.anonymous || {})[state.user.uid]) || !!((data.anonymous || {})[uid]);
      return data.participants.includes(uid) && data.isAnonymous && stillAnonymous;
    });
    if (existing) { openChat(existing.id); return; }

    // Increment anon usage
    await db.collection('users').doc(state.user.uid).update({
      anonUsageToday: todayCount + 1,
      anonUsageDate: today
    });
    state.profile.anonUsageToday = todayCount + 1;
    state.profile.anonUsageDate = today;

    // Create anonymous conversation
    const doc = await db.collection('conversations').add({
      participants: [state.user.uid, uid],
      participantNames: [state.profile.displayName, targetName || 'User'],
      participantPhotos: [state.profile.photoURL || null, targetPhoto || null],
      lastMessage: '', updatedAt: FieldVal.serverTimestamp(),
      unread: { [uid]: 0, [state.user.uid]: 0 },
      participantStatuses: { [state.user.uid]: state.status, [uid]: 'offline' },
      isAnonymous: true,
      anonymous: { [state.user.uid]: true, [uid]: true },
      anonStartedBy: state.user.uid,
      replyToPost: replyToPostId || null,
      anonContext: forceNew ? 'profile' : 'discovery'
    });
    toast(`Anonymous chat started (${todayCount + 1}/${ANON_DAILY_LIMIT} today)`);
    openChat(doc.id);
  } catch (e) { toast('Could not start chat'); console.error(e); }
}

async function editAnonNickname(convoId) {
  if (!convoId) return;
  try {
    const snap = await db.collection('conversations').doc(convoId).get();
    if (!snap.exists) return;
    const convo = { _id: convoId, ...snap.data() };
    const otherUid = (convo.participants || []).find(uid => uid !== state.user.uid);
    if (!otherUid) return;
    const current = (convo.anonNicknames || {})[anonNicknameKey(state.user.uid, otherUid)] || '';
    const next = window.prompt('Set a private nickname for this anonymous chat', current || defaultAnonLabel(convoId));
    if (next === null) return;
    const cleaned = next.trim();
    const updates = {};
    updates[`anonNicknames.${anonNicknameKey(state.user.uid, otherUid)}`] = cleaned || FieldVal.delete();
    await db.collection('conversations').doc(convoId).update(updates);
  } catch (e) { console.error(e); toast('Could not save nickname'); }
}

function showConvoActions(convoId, displayName, otherUid) {
  openModal(`
    <div class="modal-header"><h2>Chat Actions</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" style="padding:16px">
      <p style="margin-bottom:16px;color:var(--text-secondary);font-size:13px">Manage conversation with <strong>${displayName}</strong></p>
      <button class="btn-outline btn-full" style="margin-bottom:8px" onclick="archiveConvo('${convoId}')">📦 Archive Chat</button>
      <button class="btn-outline btn-full" style="margin-bottom:8px" onclick="deleteConvo('${convoId}')">🗑️ Delete Chat</button>
      <button class="btn-danger btn-full" onclick="blockUserFromChat('${otherUid}','${displayName}','${convoId}')">🚫 Block User</button>
    </div>
  `);
}

async function archiveConvo(convoId) {
  closeModal();
  try {
    await db.collection('conversations').doc(convoId).update({
      archived: FieldVal.arrayUnion(state.user.uid)
    });
    toast('Chat archived');
    refreshCurrentMessageList();
  } catch (e) { toast('Failed to archive'); console.error(e); }
}

async function deleteConvo(convoId) {
  if (!window.confirm('Delete this conversation? This cannot be undone.')) return;
  closeModal();
  try {
    const snap = await db.collection('conversations').doc(convoId).get();
    if (!snap.exists) return;
    const convo = snap.data();
    const otherUid = convo.participants.find(p => p !== state.user.uid);
    
    // If both users delete, remove from Firestore
    const deleted = convo.deletedBy || [];
    if (deleted.includes(otherUid)) {
      await db.collection('conversations').doc(convoId).delete();
    } else {
      await db.collection('conversations').doc(convoId).update({
        deletedBy: FieldVal.arrayUnion(state.user.uid),
        archived: FieldVal.arrayUnion(state.user.uid)
      });
    }
    toast('Chat deleted');
    refreshCurrentMessageList();
    showScreen('app');
  } catch (e) { toast('Failed to delete'); console.error(e); }
}

async function blockUserFromChat(uid, name, convoId) {
  if (!window.confirm(`Block ${name}? They won't be able to message you (including anonymously).`)) return;
  closeModal();
  try {
    // Add to my blocked list
    await db.collection('users').doc(state.user.uid).update({
      blockedUsers: FieldVal.arrayUnion(uid)
    });
    // Add me to their blockedBy list
    await db.collection('users').doc(uid).update({
      blockedBy: FieldVal.arrayUnion(state.user.uid)
    });
    // Archive the conversation
    await db.collection('conversations').doc(convoId).update({
      archived: FieldVal.arrayUnion(state.user.uid),
      blocked: true
    });
    state.profile.blockedUsers = [...(state.profile.blockedUsers || []), uid];
    toast(`${name} blocked`);
    refreshCurrentMessageList();
    showScreen('app');
  } catch (e) { toast('Failed to block'); console.error(e); }
}

async function unblockUser(uid, name = '') {
  if (!name) {
    // Get name from blocked list
    try {
      const userDoc = await db.collection('users').doc(uid).get();
      if (userDoc.exists) name = userDoc.data().displayName || 'this user';
    } catch (e) {
      name = 'this user';
    }
  }
  if (!window.confirm(`Unblock ${name}?`)) return;
  closeModal();
  try {
    await db.collection('users').doc(state.user.uid).update({
      blockedUsers: FieldVal.arrayRemove(uid)
    });
    await db.collection('users').doc(uid).update({
      blockedBy: FieldVal.arrayRemove(state.user.uid)
    });
    state.profile.blockedUsers = (state.profile.blockedUsers || []).filter(b => b !== uid);
    toast(`${name} unblocked`);
    // Refresh the blocked list if it's visible
    if ($('#blocked-users-list')) {
      setTimeout(() => loadBlockedUsersList(), 100);
    }
  } catch (e) { toast('Failed to unblock'); console.error(e); }
}

// ══════════════════════════════════════════════════
//  PROFILE — Fixed avatar position (inside cover)
// ══════════════════════════════════════════════════
async function openProfile(uid) {
  showScreen('profile-view');
  const body = $('#prof-body');
  const backBtn = $('#prof-back');
  if (backBtn) backBtn.dataset.uid = uid;
  body.innerHTML = '<div style="padding:60px;text-align:center"><span class="inline-spinner" style="width:28px;height:28px;color:var(--accent)"></span></div>';
  $('#prof-top-name').textContent = '';

  try {
    let user;
    if (uid === state.user.uid) { user = state.profile; }
    else {
      const doc = await db.collection('users').doc(uid).get();
      if (!doc.exists) throw new Error('Not found');
      user = { id: doc.id, ...doc.data() };
    }

    $('#prof-top-name').textContent = user.displayName;

    let posts = [];
    try {
      const pSnap = await db.collection('posts').where('authorId', '==', uid).limit(20).get();
      posts = pSnap.docs.map(d => ({ id: d.id, ...d.data() }));
      posts.sort((a, b) => (b.createdAt?.seconds || 0) - (a.createdAt?.seconds || 0));

      // Filter logic: If not me and not friend, hide friends-only posts
      if (uid !== state.user.uid) {
        const isFriend = (state.profile.friends || []).includes(uid);
        if (!isFriend) {
          posts = posts.filter(p => p.visibility !== 'friends');
        }
        posts = posts.filter(p => !p.isAnonymous);
      }
    } catch (e) { console.error('Posts', e); }

    const isMe = uid === state.user.uid;
    const modules = user.modules || [];
    const showFriendCount = user.showFriendsCount !== false && isMe; // only show to self unless they opted in

    // KEY FIX: avatar-wrap is INSIDE profile-cover so position:absolute works relative to cover
    body.innerHTML = `
      <div class="profile-cover">
        <div class="profile-avatar-wrap">
          <div class="profile-avatar-large${user.photoURL ? ' clickable' : ''}" ${user.photoURL ? `onclick="viewImage('${user.photoURL}')"` : ''}>
            ${user.photoURL ? `<img src="${user.photoURL}" alt="">` : initials(user.displayName)}
          </div>
          ${user.status === 'online' ? '<div class="avatar-online-dot"></div>' : ''}
        </div>
      </div>

      <div class="profile-info">
        <div class="profile-name">${esc(user.displayName)}${verifiedBadge(uid)}</div>
        <div class="profile-handle">${esc(user.major || '')}${user.major && user.university ? ' · ' : ''}${esc(user.university || '')}</div>
        <div class="profile-badges">
          ${user.year ? `<span class="profile-badge">🎓 ${esc(user.year)}</span>` : ''}
          ${isMe && user.address ? `<span class="profile-badge">📍 ${esc(user.address)}</span>` : ''}
        </div>
        ${user.bio ? `<p class="profile-bio">${esc(user.bio)}</p>` : ''}
        ${modules.length ? `<div class="profile-modules">${modules.map(m => `<span class="module-chip clickable" onclick="openModuleFeed('${esc(m)}')">${esc(m)}</span>`).join('')}</div>` : ''}

        <div class="profile-stats">
          <div class="profile-stat"><div class="stat-num">${posts.length}</div><div class="stat-label">Posts</div></div>
          ${showFriendCount ? `<div class="profile-stat"><div class="stat-num">${(user.friends || []).length}</div><div class="stat-label">Friends</div></div>` : ''}
          ${modules.length ? `<div class="profile-stat"><div class="stat-num">${modules.length}</div><div class="stat-label">Modules</div></div>` : ''}
        </div>

        <div class="profile-actions">
          ${isMe
            ? `<button class="btn-primary" onclick="editProfile()">Edit Profile</button>
               <button class="btn-secondary" onclick="doLogout()">Log Out</button>`
            : (() => {
                const isFriend = (state.profile.friends || []).includes(uid);
                const isPending = (state.profile.sentRequests || []).includes(uid);
                const theyRequested = (state.profile.friendRequests || []).some(r => r.uid === uid);
                let friendBtn = '';
                if (isFriend) {
                  friendBtn = `<button class="btn-secondary" onclick="unfriend('${uid}');this.textContent='Add Friend';this.className='btn-outline'">✓ Friends</button>`;
                } else if (theyRequested) {
                  friendBtn = `<button class="btn-primary" onclick="acceptFriendRequest('${uid}','${esc(user.displayName)}','${user.photoURL || ''}');setTimeout(()=>openProfile('${uid}'),500)">Accept Request</button>`;
                } else if (isPending) {
                  friendBtn = `<button class="btn-outline" disabled style="opacity:0.6">Pending…</button>`;
                } else {
                  friendBtn = `<button class="btn-outline" onclick="sendFriendRequest('${uid}','${esc(user.displayName)}','${user.photoURL || ''}', this)">Add Friend</button>`;
                }
                const isFriendForChat = isFriend;
                const msgBtn = isFriendForChat
                  ? `<button class="btn-primary" onclick="startChat('${uid}','${esc(user.displayName)}','${user.photoURL || ''}')">Message</button>`
                  : `<button class="btn-outline anon-msg-btn" onclick="startAnonChat('${uid}','${esc(user.displayName)}','${user.photoURL || ''}', true)">👻 Anonymous Message</button>`;
                return `${msgBtn}\n               ${friendBtn}`;
              })()}
        </div>
      </div>

      <div class="profile-tabs">
        <button class="profile-tab active" data-pt="posts">Posts</button>
        <button class="profile-tab" data-pt="photos">Photos</button>
        <button class="profile-tab" data-pt="about">About</button>
      </div>
      <div id="profile-tab-content">${renderProfilePosts(posts, user)}</div>
    `;

    // Wire tabs
    $$('.profile-tab').forEach(tab => {
      tab.onclick = () => {
        $$('.profile-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const tc = $('#profile-tab-content');
        if (tab.dataset.pt === 'posts') tc.innerHTML = renderProfilePosts(posts, user);
        else if (tab.dataset.pt === 'photos') tc.innerHTML = renderProfilePhotos(posts);
        else {
          tc.innerHTML = renderProfileAbout(user);
          if (uid === state.user.uid && (state.profile.blockedUsers || []).length) {
            setTimeout(() => loadBlockedUsersList(), 100);
          }
        }
      };
    });
  } catch (e) {
    console.error(e);
    body.innerHTML = '<div class="empty-state"><h3>Could not load profile</h3></div>';
  }

  $('#prof-back').onclick = () => showScreen('app');
}

function renderProfilePosts(posts, user) {
  const isMe = user.id === state.user.uid;
  const visiblePosts = isMe ? posts : posts.filter(p => !p.isAnonymous);
  if (!visiblePosts.length) return '<div class="empty-state"><h3>No posts yet</h3></div>';
  const _profPlayers = [];
  const html = `<div class="profile-posts">${visiblePosts.map(p => {
    const hasVideo = p.videoURL || (p.mediaType === 'video');
    const hasImage = p.imageURL && !hasVideo;
    const mediaURL = hasVideo ? (p.videoURL || p.imageURL) : p.imageURL;
    let videoPlayerData = null;
    if (hasVideo && mediaURL) {
      videoPlayerData = createVideoPlayer(mediaURL);
      _profPlayers.push(videoPlayerData);
    }
    return `
    <div class="post-card">
      ${p.repostOf ? `<div style="padding-bottom:6px;margin-bottom:6px;font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:6px">
         <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
         ${esc(user.displayName)} reposted
       </div>` : ''}
      <div class="post-header">
        ${p.isAnonymous ? `<div class="avatar-md anon-avatar">👻</div>` : avatar(user.displayName, user.photoURL, 'avatar-md')}
        <div class="post-header-info">
          <div class="post-author-name">${p.isAnonymous ? '👻 Anonymous' : esc(user.displayName)}</div>
          <div class="post-meta">${timeAgo(p.createdAt)}</div>
        </div>
        ${isMe ? `<button class="icon-btn" onclick="showPostOptions('${p.id}')" style="margin-left:auto">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/></svg>
        </button>` : ''}
      </div>
      ${p.content ? renderExpandablePostContent(p.content, `profile-${p.id}`, 170) : ''}
      ${renderPostModuleTags(p.moduleTags || [])}
      ${renderPostHashTags(getPostHashTags(p).filter(tag => !(p.moduleTags || []).includes(tag.toUpperCase())))}
      ${!p.repostOf && hasImage && mediaURL ? `<div class="post-image-wrap"><img src="${mediaURL}" class="post-image" onclick="viewImage('${mediaURL}')"></div>` : ''}
      ${!p.repostOf && hasVideo && videoPlayerData ? videoPlayerData.html : ''}
      ${p.repostOf ? renderQuoteEmbed(p.repostOf) : ''}
      <div class="post-actions">
        <button class="post-action ${(p.likes||[]).includes(state.user.uid)?'liked':''}" onclick="toggleLike('${p.id}')">❤ ${(p.likes||[]).length||'Like'}</button>
        <button class="post-action" onclick="openComments('${p.id}')">💬 ${p.commentsCount||'Comment'}</button>
        <button class="post-action" onclick="openShareModal('${p.id}')">↗ Share</button>
      </div>
    </div>`;
  }).join('')}</div>`;
  // Init players after render
  requestAnimationFrame(() => {
    _profPlayers.forEach(p => initPlayer(p.id));
    _pendingQuotePlayers.forEach(p => initPlayer(p.id));
    _pendingQuotePlayers.length = 0;
  });
  return html;
}

// ─── iOS-style Post Options + Reporting ─────────
async function reportPost(postId) {
  closeModal();
  openModal(`
    <div class="modal-header"><h2>Report Post</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" style="padding:16px">
      <div style="display:flex;flex-direction:column;gap:10px;margin-bottom:14px">
        <label><input type="radio" name="report-reason" value="spam"> Spam</label>
        <label><input type="radio" name="report-reason" value="inappropriate"> Inappropriate content</label>
        <label><input type="radio" name="report-reason" value="harassment"> Harassment</label>
        <label><input type="radio" name="report-reason" value="misinformation"> Misinformation</label>
        <label><input type="radio" name="report-reason" value="other"> Other</label>
      </div>
      <button class="btn-primary btn-full" onclick="submitPostReport('${postId}')">Submit Report</button>
    </div>
  `);
}

async function submitPostReport(postId) {
  const reason = document.querySelector('input[name="report-reason"]:checked')?.value;
  if (!reason) return toast('Select a reason');
  try {
    const postRef = db.collection('posts').doc(postId);
    await postRef.update({
      reportsCount: FieldVal.increment(1),
      reportedBy: FieldVal.arrayUnion(state.user.uid),
      lastReportReason: reason,
      lastReportedAt: FieldVal.serverTimestamp()
    });
    closeModal();
    toast('Post reported');
  } catch (e) {
    console.error(e);
    toast('Report failed');
  }
}

function showPostOptions(postId) {
  db.collection('posts').doc(postId).get().then(doc => {
    if (!doc.exists) return toast('Post not found');
    const post = doc.data();
    const isOwner = post.authorId === state.user.uid;
    const reportsCount = post.reportsCount || 0;
    const reportBtn = !isOwner
      ? `<button class="ios-action-btn" onclick="reportPost('${postId}')"><span style="color:var(--orange)">Report${reportsCount ? ` (${reportsCount})` : ''}</span></button>`
      : '';
    const deleteBtn = isOwner
      ? `<button class="ios-action-btn" onclick="confirmDeletePost('${postId}')"><span style="color:var(--red)">Delete Post</span></button>`
      : '';
    const adminBtn = _isAdmin
      ? `<button class="ios-action-btn" onclick="adminDeletePost('${postId}')"><span style="color:var(--red)">Admin Remove</span></button>`
      : '';
    openModal(`
      <div class="modal-body" style="padding:8px 0">
        ${reportBtn}
        ${deleteBtn}
        ${adminBtn}
        <div style="height:1px;background:var(--border);margin:4px 16px"></div>
        <button class="ios-action-btn" onclick="closeModal()"><span>Cancel</span></button>
      </div>
    `);
  }).catch(() => toast('Could not open options'));
}

async function confirmDeletePost(postId) {
  closeModal();
  openModal(`
    <div class="modal-body" style="text-align:center;padding:24px">
      <h3 style="margin-bottom:8px">Delete this post?</h3>
      <p style="color:var(--text-secondary);font-size:14px;margin-bottom:20px">This cannot be undone.</p>
      <div style="display:flex;gap:12px;justify-content:center">
        <button class="btn-secondary" onclick="closeModal()" style="flex:1">Cancel</button>
        <button class="btn-danger" onclick="deletePost('${postId}')" style="flex:1;border-radius:var(--radius)">Delete</button>
      </div>
    </div>
  `);
}

async function deletePost(postId) {
  closeModal();
  try {
    await db.collection('posts').doc(postId).delete();
    toast('Post deleted');
    const el = document.getElementById(`post-${postId}`);
    if (el) el.remove();
    if (document.getElementById('profile-view')?.classList.contains('active')) openProfile(state.user.uid);
  } catch (e) {
    console.error(e);
    toast('Failed to delete');
  }
}

async function adminDeletePost(postId) {
  if (!_isAdmin) return toast('Admin only');
  closeModal();
  try {
    await db.collection('posts').doc(postId).delete();
    toast('Post removed');
    const el = document.getElementById(`post-${postId}`);
    if (el) el.remove();
  } catch (e) {
    console.error(e);
    toast('Failed to remove post');
  }
}

function showAdminDataClear() {
  if (!_isAdmin) return toast('Admin only');
  openModal(`
    <div class="modal-header"><h2>Kill Switch (Step 1/2)</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" style="padding:16px">
      <p style="margin-bottom:10px;color:var(--text-secondary)">Type admin email to continue.</p>
      <input id="admin-kill-email" type="email" placeholder="admin@mynwu.ac.za" style="width:100%;margin-bottom:12px">
      <button class="btn-danger btn-full" onclick="adminDataClearStepTwo()">Continue</button>
    </div>
  `);
}

function adminDataClearStepTwo() {
  const entered = ($('#admin-kill-email')?.value || '').trim().toLowerCase();
  const me = (state.user?.email || '').toLowerCase();
  if (!entered) return toast('Enter admin email');
  if (entered !== ADMIN_EMAIL.toLowerCase() || me !== ADMIN_EMAIL.toLowerCase()) {
    return toast('Admin email verification failed');
  }
  openModal(`
    <div class="modal-header"><h2>Kill Switch (Step 2/2)</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" style="padding:16px">
      <p style="margin-bottom:10px;color:var(--text-secondary)">Final check: type <b>DELETE ALL DATA</b> to wipe Firestore content for a clean launch.</p>
      <input id="admin-kill-phrase" type="text" placeholder="DELETE ALL DATA" style="width:100%;margin-bottom:12px">
      <button class="btn-danger btn-full" onclick="doAdminDataClear()">Confirm Wipe</button>
    </div>
  `);
}

async function _wipeCollection(path, limit = 200) {
  while (true) {
    const snap = await db.collection(path).limit(limit).get();
    if (snap.empty) break;
    const batch = db.batch();
    snap.docs.forEach(d => batch.delete(d.ref));
    await batch.commit();
  }
}

async function doAdminDataClear() {
  if (!_isAdmin) return toast('Admin only');
  const phrase = ($('#admin-kill-phrase')?.value || '').trim();
  if (phrase !== 'DELETE ALL DATA') return toast('Final confirmation phrase incorrect');
  closeModal();
  toast('Wiping data...');
  try {
    // Delete nested message/comment/notification subcollections first
    const [postSnap, convoSnap, groupSnap, asgSnap, userSnap] = await Promise.all([
      db.collection('posts').get(),
      db.collection('conversations').get(),
      db.collection('groups').get(),
      db.collection('assignmentGroups').get(),
      db.collection('users').get()
    ]);

    for (const d of postSnap.docs) {
      await _wipeCollection(`posts/${d.id}/comments`);
    }
    for (const d of convoSnap.docs) {
      await _wipeCollection(`conversations/${d.id}/messages`);
    }
    for (const d of groupSnap.docs) {
      await _wipeCollection(`groups/${d.id}/messages`);
    }
    for (const d of asgSnap.docs) {
      await _wipeCollection(`assignmentGroups/${d.id}/messages`);
    }
    for (const d of userSnap.docs) {
      await _wipeCollection(`users/${d.id}/notifications`);
    }

    // Then wipe main collections
    const collections = ['posts', 'groups', 'conversations', 'events', 'assignmentGroups', 'stories', 'products', 'stats'];
    for (const col of collections) await _wipeCollection(col);

    toast('Kill switch complete. Data wiped.');
    navigate('feed');
  } catch (e) {
    console.error(e);
    toast('Wipe failed');
  }
}
function renderProfileAbout(user) {
  const modules = user.modules || [];
  const isMe = user.id === state.user?.uid;
  const blockedUsers = isMe ? (user.blockedUsers || []) : [];
  return `
    <div class="profile-about">
      <div class="about-item"><span class="about-icon">🎓</span><div><div class="about-label">University</div><div class="about-value">${esc(user.university || 'Not set')}</div></div></div>
      <div class="about-item"><span class="about-icon">📚</span><div><div class="about-label">Major</div><div class="about-value">${esc(user.major || 'Not set')}</div></div></div>
      <div class="about-item"><span class="about-icon">📅</span><div><div class="about-label">Year</div><div class="about-value">${esc(user.year || 'Not set')}</div></div></div>
      ${isMe && user.address ? `<div class="about-item"><span class="about-icon">📍</span><div><div class="about-label">Location</div><div class="about-value">${esc(user.address)}</div></div></div>` : ''}
      ${modules.length ? `<div class="about-item"><span class="about-icon">🧩</span><div><div class="about-label">Modules</div><div class="about-modules">${modules.map(m => `<span class="module-chip">${esc(m)}</span>`).join('')}</div></div></div>` : ''}
      ${user.joinedAt ? `<div class="about-item"><span class="about-icon">🗓</span><div><div class="about-label">Joined</div><div class="about-value">${timeAgo(user.joinedAt)}</div></div></div>` : ''}
      ${isMe ? `<div class="about-item"><span class="about-icon">🚫</span><div><div class="about-label">Blocked Users</div><div id="blocked-users-list">${blockedUsers.length ? '<span class="inline-spinner"></span>' : 'None'}</div></div></div>` : ''}
    </div>${isMe && blockedUsers.length ? '<script>loadBlockedUsersList()</script>' : ''}`;
}

function renderProfilePhotos(posts) {
  const photos = posts
    .filter(p => p.imageURL && p.mediaType !== 'video')
    .map(p => p.imageURL);
  const videos = posts
    .filter(p => p.videoURL || p.mediaType === 'video')
    .map(p => p.videoURL || p.imageURL);
  const all = [...photos.map(u => ({ type:'img', url:u })), ...videos.map(u => ({ type:'vid', url:u }))];
  if (!all.length) return '<div class="empty-state"><h3>No photos yet</h3></div>';
  return `<div class="profile-photo-grid">${all.map(m => {
    if (m.type === 'img') {
      return `<div class="photo-grid-item" onclick="viewImage('${m.url}')"><img src="${m.url}" loading="lazy"></div>`;
    }
    return `<div class="photo-grid-item" onclick="viewImage('${m.url}')"><video src="${m.url}" preload="metadata"></video><div class="photo-grid-play">▶</div></div>`;
  }).join('')}</div>`;
}

async function loadBlockedUsersList() {
  const blockedUsers = state.profile.blockedUsers || [];
  if (!blockedUsers.length) {
    $('#blocked-users-list').innerHTML = 'None';
    return;
  }
  try {
    const userDocs = await Promise.all(blockedUsers.map(uid => db.collection('users').doc(uid).get()));
    const users = userDocs.filter(d => d.exists).map(d => ({ id: d.id, ...d.data() }));
    $('#blocked-users-list').innerHTML = users.map(u => `
      <div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--border)">
        ${avatar(u.displayName, u.photoURL, 'avatar-sm')}
        <div style="flex:1">
          <div style="font-weight:500;font-size:14px">${esc(u.displayName)}</div>
        </div>
        <button class="btn-outline" style="padding:4px 12px;font-size:12px" onclick="unblockUser('${u.id}')">Unblock</button>
      </div>
    `).join('');
  } catch (e) {
    console.error('Failed to load blocked users:', e);
    $('#blocked-users-list').innerHTML = 'Error loading list';
  }
}

// ─── Edit Profile (with modules) ─────────────────
function editProfile() {
  const p = state.profile;
  const mods = (p.modules || []).join(', ');
  openModal(`
    <div class="modal-header"><h2>Edit Profile</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      <div class="form-group"><label>Display Name</label><input type="text" id="edit-name" value="${esc(p.displayName)}"></div>
      <div class="form-group"><label>Bio</label><textarea id="edit-bio">${esc(p.bio || '')}</textarea></div>
      <div class="form-group"><label>Location / Res</label><input type="text" id="edit-address" value="${esc(p.address || '')}" placeholder="e.g. Potch Main Campus"></div>
      <div class="form-group"><label>Modules (comma-separated)</label><input type="text" id="edit-modules" value="${esc(mods)}" placeholder="MAT101, COS132, PHY121"></div>
      <div class="form-group"><label>Profile Photo</label><input type="file" accept="image/*" id="edit-photo"></div>
      <div class="form-group" style="display:flex;align-items:center;gap:8px">
        <input type="checkbox" id="edit-autofill" style="width:auto" ${p.allowAutoFill !== false ? 'checked' : ''}>
        <label for="edit-autofill" style="margin:0;font-size:14px">Allow auto-fill into groups</label>
      </div>
      <p style="color:var(--text-tertiary);font-size:11px;margin:-8px 0 12px">When enabled, group hosts can auto-fill you into their groups for your modules.</p>
      <button class="btn-primary btn-full" id="edit-save">Save</button>
    </div>
  `);
  let newPhoto = null; let newPhotoFile = null;
  $('#edit-photo').onchange = async e => {
    if (e.target.files[0]) { newPhotoFile = e.target.files[0]; newPhoto = 'pending'; toast('Photo selected'); }
  };
  $('#edit-save').onclick = async () => {
    const name = $('#edit-name').value.trim();
    const bio = $('#edit-bio').value.trim();
    const address = $('#edit-address')?.value.trim() || '';
    const modulesRaw = $('#edit-modules').value || '';
    const modules = modulesRaw.split(',').map(m => m.trim().toUpperCase()).filter(Boolean);
    if (!name) return toast('Name required');
    closeModal(); toast('Saving...');
    const allowAutoFill = $('#edit-autofill')?.checked !== false;
    const updates = { displayName: name, bio, modules, address, allowAutoFill };
    if (newPhotoFile) { updates.photoURL = await uploadToR2(newPhotoFile, 'profile'); }
    try {
      await db.collection('users').doc(state.user.uid).update(updates);
      Object.assign(state.profile, updates);
      if (name !== state.user.displayName) await state.user.updateProfile({ displayName: name });
      setupHeader(); toast('Profile updated!'); openProfile(state.user.uid);
    } catch (e) { toast('Failed'); console.error(e); }
  };
}

// ─── Voice Recording ─────────────────────────────
let _voiceRecorder = null;
let _voiceChunks = [];
let _voiceInterval = null;
let _voiceStartTime = 0;
let _voiceContext = ''; // '' for DM, 'gchat' for group

function startVoiceRecord(ctx = '') {
  _voiceContext = ctx;
  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    _voiceRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    _voiceChunks = [];
    _voiceRecorder.ondataavailable = e => { if (e.data.size > 0) _voiceChunks.push(e.data); };
    _voiceRecorder.start();
    _voiceStartTime = Date.now();
    const timerId = ctx ? 'gchat-voice-timer' : 'voice-timer';
    const recId = ctx ? 'gchat-voice-recorder' : 'voice-recorder';
    const el = document.getElementById(recId);
    if (el) el.style.display = 'flex';
    _voiceInterval = setInterval(() => {
      const secs = Math.floor((Date.now() - _voiceStartTime) / 1000);
      const m = Math.floor(secs / 60), s = secs % 60;
      const te = document.getElementById(timerId);
      if (te) te.textContent = `${m}:${s.toString().padStart(2, '0')}`;
    }, 500);
  }).catch(() => toast('Microphone access denied'));
}

function cancelVoiceRecord(ctx = '') {
  if (_voiceRecorder && _voiceRecorder.state !== 'inactive') {
    _voiceRecorder.stop();
    _voiceRecorder.stream.getTracks().forEach(t => t.stop());
  }
  _voiceRecorder = null; _voiceChunks = [];
  clearInterval(_voiceInterval);
  const recId = ctx ? 'gchat-voice-recorder' : 'voice-recorder';
  const el = document.getElementById(recId);
  if (el) el.style.display = 'none';
}

function stopVoiceAndSend(ctx = '') {
  if (!_voiceRecorder) return;
  _voiceRecorder.onstop = () => {
    const blob = new Blob(_voiceChunks, { type: 'audio/webm' });
    _voiceRecorder.stream.getTracks().forEach(t => t.stop());
    _voiceRecorder = null; _voiceChunks = [];
    clearInterval(_voiceInterval);
    const recId = ctx ? 'gchat-voice-recorder' : 'voice-recorder';
    const el = document.getElementById(recId);
    if (el) el.style.display = 'none';
    if (ctx === 'gchat') {
      window._gchatVoiceBlob = blob;
      document.getElementById('gchat-send')?.click();
    } else {
      window._chatVoiceBlob = blob;
      document.getElementById('chat-send')?.click();
    }
  };
  _voiceRecorder.stop();
}

function doLogout() { auth.signOut().then(() => window.location.reload()); }

// ─── Modal System ────────────────────────────────
function openModal(innerHtml) {
  const bg = $('#modal-bg');
  $('#modal-inner').innerHTML = innerHtml;
  bg.style.display = 'flex';
  bg.onclick = e => { if (e.target === bg) closeModal(); };
}

function closeModal() {
  $('#modal-bg').style.display = 'none';
  $('#modal-inner').innerHTML = '';
}

// ─── Share System ────────────────────────────────
async function openShareModal(postId) {
  openModal(`
    <div class="modal-header"><h2>Share Post</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" style="padding:16px">
       <button class="btn-primary btn-full" style="margin-bottom:12px;background:var(--accent);color:white;border:none;padding:12px;border-radius:12px;font-weight:600;width:100%" onclick="openQuoteRepost('${postId}')">🔄 Quote Repost</button>
       <div style="height:1px;background:var(--border);margin:16px 0"></div>
       <h3 style="margin-bottom:12px;font-size:16px">Send to Friend</h3>
       <div id="share-friends-list" style="max-height:300px;overflow-y:auto;display:flex;flex-direction:column;gap:8px">
          <div class="inline-spinner" style="margin:20px auto"></div>
       </div>
    </div>
  `);
  
  const friends = state.profile.friends || [];
  const list = $('#share-friends-list');
  if(!friends.length) {
     list.innerHTML = '<p style="color:var(--text-tertiary);text-align:center">Add friends to share directly.</p>';
     return;
  }
  
  try {
     const chunks = [];
     for(let i=0; i<friends.length; i+=10) {
        const batch = friends.slice(i, i+10);
        if(batch.length) {
           const s = await db.collection('users').where(firebase.firestore.FieldPath.documentId(), 'in', batch).get();
           chunks.push(...s.docs.map(d=>({id:d.id, ...d.data()})));
        }
     }
     list.innerHTML = chunks.map(f => `
       <div class="share-friend-item" onclick="shareToFriend('${f.id}','${esc(f.displayName)}','${postId}')" style="display:flex;align-items:center;gap:12px;padding:8px;border-radius:12px;cursor:pointer;transition:background 0.2s">
          ${avatar(f.displayName, f.photoURL, 'avatar-sm')}
          <div style="flex:1;display:flex;flex-direction:column">
             <span style="font-weight:600;font-size:14px">${esc(f.displayName)}</span>
             <span style="font-size:12px;color:var(--text-secondary)">${esc(f.university||'Student')}</span>
          </div>
          <button class="btn-sm btn-secondary" style="pointer-events:none">Send</button>
       </div>
     `).join('');
     $$('.share-friend-item').forEach(el => {
        el.onmouseenter = () => el.style.background = 'var(--bg-secondary)';
        el.onmouseleave = () => el.style.background = 'transparent';
     });
  } catch(e) { console.error(e); list.innerHTML='Error loading friends'; }
}

async function shareToFriend(uid, name, postId) {
   try {
     const myId = state.user.uid;

     // Fetch post preview data for rich chat card
     let preview = {};
     try {
       const pDoc = await db.collection('posts').doc(postId).get();
       if (pDoc.exists) {
         const pd = pDoc.data();
         preview = {
           content: (pd.content || '').slice(0, 80),
           mediaURL: pd.mediaURL || '',
           mediaType: pd.mediaType || '',
           authorName: pd.authorName || '',
           authorPhoto: pd.authorPhotoURL || ''
         };
       }
     } catch(_) {}

     const snap = await db.collection('conversations').where('participants','array-contains',myId).get();
     let convo = snap.docs.find(d => d.data().participants.includes(uid));
     
     if(!convo) {
         const ref = await db.collection('conversations').add({
             participants: [myId, uid],
             participantNames: [state.profile.displayName, name],
             participantPhotos: [state.profile.photoURL||null, null],
             updatedAt: FieldVal.serverTimestamp(),
             lastMessage: 'Shared a post',
             unread: { [uid]: 1, [myId]: 0 }
         });
         convo = { id: ref.id };
     } else {
         await db.collection('conversations').doc(convo.id).update({
             lastMessage: 'Shared a post',
             updatedAt: FieldVal.serverTimestamp(),
             [`unread.${uid}`]: FieldVal.increment(1)
         });
     }
     
     await db.collection('conversations').doc(convo.id).collection('messages').add({
         text: `shared post::${postId}`, 
         type: 'share_post',
         payload: { postId, ...preview },
         senderId: myId,
         createdAt: FieldVal.serverTimestamp()
     });
     
     toast(`Sent to ${name}`);
     closeModal();
   } catch(e) { console.error(e); toast('Failed to send'); }
}

async function openQuoteRepost(postId) {
  closeModal();
  try {
    const origRef = await db.collection('posts').doc(postId).get();
    if (!origRef.exists) return toast('Post not found');
    const orig = origRef.data();
    const hasImg = orig.imageURL && orig.mediaType !== 'video';
    const hasVid = orig.videoURL || (orig.mediaType === 'video');
    const vidUrl = hasVid ? (orig.videoURL || orig.imageURL) : null;
    openModal(`
      <div class="modal-header"><h2>Quote Repost</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
      <div class="modal-body" style="padding:16px">
        <div style="display:flex;gap:10px;margin-bottom:12px">
          ${avatar(state.profile.displayName, state.profile.photoURL, 'avatar-md')}
          <div><div style="font-weight:600">${esc(state.profile.displayName)}</div><div style="font-size:12px;color:var(--text-secondary)">Quoting post</div></div>
        </div>
        <textarea id="quote-text" placeholder="Add your thoughts…" style="width:100%;min-height:80px;border:none;background:transparent;color:var(--text-primary);font-size:15px;resize:none;outline:none;margin-bottom:12px"></textarea>
        <div class="quote-embed-preview" style="border:1px solid var(--border);border-radius:var(--radius);padding:12px;background:var(--bg-secondary)">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            ${avatar(orig.authorName, orig.authorPhoto, 'avatar-sm')}
            <span style="font-weight:600;font-size:13px">${esc(orig.authorName || 'User')}</span>
          </div>
          ${orig.content ? `<div style="font-size:13px;color:var(--text-secondary);margin-bottom:8px;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden">${esc(orig.content)}</div>` : ''}
          ${hasImg ? `<img src="${orig.imageURL}" style="width:100%;max-height:120px;object-fit:cover;border-radius:8px">` : ''}
          ${hasVid && vidUrl ? `<video src="${vidUrl}" style="width:100%;max-height:120px;object-fit:cover;border-radius:8px" controls preload="metadata"></video>` : ''}
        </div>
        <div style="display:flex;justify-content:flex-end;margin-top:12px">
          <button class="btn-primary" id="quote-submit" style="padding:10px 28px">Post</button>
        </div>
      </div>
    `);
    $('#quote-submit').onclick = async () => {
      const quoteText = ($('#quote-text')?.value || '').trim();
      closeModal(); toast('Reposting…');
      try {
        await db.collection('posts').add({
          authorId: state.user.uid,
          authorName: state.profile.displayName,
          authorPhoto: state.profile.photoURL || null,
          authorUni: state.profile.university || '',
          content: quoteText,
          mediaType: 'text',
          repostOf: {
            id: postId,
            authorId: orig.authorId,
            authorName: orig.authorName || '',
            authorPhoto: orig.authorPhoto || null,
            content: orig.content || '',
            imageURL: orig.imageURL || null,
            videoURL: orig.videoURL || null,
            mediaType: orig.mediaType || 'text'
          },
          createdAt: FieldVal.serverTimestamp(),
          likes: [], commentsCount: 0, visibility: 'public'
        });
        toast('Reposted!');
        navigate('feed');
      } catch (e) { toast('Failed'); console.error(e); }
    };
  } catch (e) { console.error(e); toast('Error loading post'); }
}

async function repost(postId) {
   try {
      const origRef = await db.collection('posts').doc(postId).get();
      if(!origRef.exists) return;
      const orig = origRef.data();
      openQuoteRepost(postId);
   } catch(e) { console.error(e); }
}

// ══════════════════════════════════════════════════
//  ADMIN PANEL — Official oversight account
// ══════════════════════════════════════════════════
async function openAdminPanel() {
  if (!_isAdmin) return toast('Admin only');
  openModal(`
    <div class="modal-header"><h2>\ud83d\udee1\ufe0f Admin Panel</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body admin-panel">
      <div class="admin-stats" id="admin-stats"><div class="inline-spinner" style="margin:16px auto"></div></div>
      <div class="admin-section">
        <h4>Quick Actions</h4>
        <button class="btn-outline btn-full" style="margin-bottom:8px" onclick="adminViewAllGroups()">\ud83d\udccb View All Groups</button>
        <button class="btn-outline btn-full" style="margin-bottom:8px" onclick="adminViewAllUsers()">\ud83d\udc65 View All Users</button>
        <button class="btn-outline btn-full" style="margin-bottom:8px" onclick="adminVerifyUser()">\u2714\ufe0f Verify a User</button>
        <button class="btn-outline btn-full" style="margin-bottom:8px" onclick="adminModeratePosts()">\ud83e\uddf9 Moderate Posts</button>
        <button class="btn-outline btn-full" style="margin-bottom:8px" onclick="adminBroadcastPrompt()">📢 Broadcast Notice</button>
        <button class="btn-outline btn-full" style="margin-bottom:8px;color:var(--red);border-color:rgba(239,68,68,0.35)" onclick="showAdminDataClear()">Kill Switch: Wipe Data</button>
      </div>
    </div>
  `);
  try {
    const [usersSnap, postsSnap, groupsSnap, asgSnap] = await Promise.all([
      db.collection('users').get(),
      db.collection('posts').get(),
      db.collection('groups').get(),
      db.collection('assignmentGroups').get()
    ]);
    const statsEl = $('#admin-stats');
    if (statsEl) {
      const onlineCount = usersSnap.docs.filter(d => d.data().status === 'online').length;
      statsEl.innerHTML = `
        <div class="admin-stat-grid">
          <div class="admin-stat-card"><div class="admin-stat-num">${usersSnap.size}</div><div class="admin-stat-label">Total Users</div></div>
          <div class="admin-stat-card"><div class="admin-stat-num">${onlineCount}</div><div class="admin-stat-label">Online Now</div></div>
          <div class="admin-stat-card"><div class="admin-stat-num">${postsSnap.size}</div><div class="admin-stat-label">Total Posts</div></div>
          <div class="admin-stat-card"><div class="admin-stat-num">${groupsSnap.size}</div><div class="admin-stat-label">Groups</div></div>
          <div class="admin-stat-card"><div class="admin-stat-num">${asgSnap.size}</div><div class="admin-stat-label">Groups</div></div>
        </div>`;
    }
  } catch (e) { console.error(e); }
}

async function adminViewAllGroups() {
  if (!_isAdmin) return;
  closeModal();
  openModal(`
    <div class="modal-header"><h2>All Groups</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" id="admin-groups-body"><div class="inline-spinner" style="margin:24px auto"></div></div>
  `);
  try {
    const snap = await db.collection('assignmentGroups').orderBy('createdAt', 'desc').limit(50).get();
    const groups = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    const body = $('#admin-groups-body');
    body.innerHTML = groups.length ? groups.map(g => `
      <div class="asg-card" style="margin:8px 0;cursor:pointer" onclick="closeModal();openGroupDetail('${g.id}')">
        <div class="asg-card-top"><div class="asg-card-module">${esc(g.moduleCode || '?')}</div><span class="asg-badge ${g.status === 'archived' ? 'archived' : 'open'}">${g.status || 'open'}</span></div>
        <div class="asg-card-title">${esc(g.groupTitle || g.assignmentTitle)}</div>
        <div class="asg-card-meta"><span>\ud83d\udc65 ${(g.members||[]).length}/${g.maxSize||10}</span><span>\u00b7 ${timeAgo(g.createdAt)}</span></div>
      </div>
    `).join('') : '<p style="padding:16px;color:var(--text-secondary)">No groups yet.</p>';
  } catch (e) { console.error(e); }
}

async function adminViewAllUsers() {
  if (!_isAdmin) return;
  closeModal();
  openModal(`
    <div class="modal-header"><h2>All Users</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      <input type="text" id="admin-user-search" placeholder="Search users..." style="width:100%;margin-bottom:12px">
      <div id="admin-users-list"><div class="inline-spinner" style="margin:24px auto"></div></div>
    </div>
  `);
  try {
    const snap = await db.collection('users').limit(100).get();
    const users = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    function renderAdminUsers(list) {
      return list.map(u => `
        <div class="pref-person" onclick="closeModal();openProfile('${u.id}')" style="cursor:pointer">
          ${avatar(u.displayName, u.photoURL, 'avatar-sm')}
          <div class="pref-person-info">
            <div class="pref-person-name">${esc(u.displayName)}${u.isVerified || VERIFIED_UIDS.has(u.id) ? '<span class="verified-badge">\u2714</span>' : ''}</div>
            <div class="pref-person-meta">${esc(u.email || '')} \u00b7 ${esc(u.major || '')} \u00b7 ${u.status || 'offline'}</div>
          </div>
        </div>
      `).join('');
    }
    $('#admin-users-list').innerHTML = renderAdminUsers(users);
    $('#admin-user-search').oninput = (e) => {
      const q = (e.target.value || '').toLowerCase();
      const filtered = q ? users.filter(u => (u.displayName||'').toLowerCase().includes(q) || (u.email||'').toLowerCase().includes(q)) : users;
      $('#admin-users-list').innerHTML = renderAdminUsers(filtered);
    };
  } catch (e) { console.error(e); }
}

function adminVerifyUser() {
  if (!_isAdmin) return;
  closeModal();
  openModal(`
    <div class="modal-header"><h2>\u2714\ufe0f Verify User</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      <p style="color:var(--text-secondary);font-size:13px;margin-bottom:12px">Search for a user to give them the official verified badge.</p>
      <input type="text" id="verify-search" placeholder="Search by name or email..." style="width:100%;margin-bottom:12px">
      <div id="verify-results"></div>
    </div>
  `);
  let _searchTimer = null;
  $('#verify-search').oninput = (e) => {
    clearTimeout(_searchTimer);
    _searchTimer = setTimeout(async () => {
      const q = (e.target.value || '').trim();
      if (q.length < 2) { $('#verify-results').innerHTML = ''; return; }
      try {
        const snap = await db.collection('users').limit(50).get();
        const users = snap.docs.map(d => ({ id: d.id, ...d.data() }))
          .filter(u => (u.displayName||'').toLowerCase().includes(q.toLowerCase()) || (u.email||'').toLowerCase().includes(q.toLowerCase()));
        $('#verify-results').innerHTML = users.map(u => `
          <div class="pref-person" style="cursor:pointer">
            ${avatar(u.displayName, u.photoURL, 'avatar-sm')}
            <div class="pref-person-info">
              <div class="pref-person-name">${esc(u.displayName)}${u.isVerified || VERIFIED_UIDS.has(u.id) ? '<span class="verified-badge">\u2714</span>' : ''}</div>
              <div class="pref-person-meta">${esc(u.email || '')}</div>
            </div>
            <button class="btn-sm ${u.isVerified ? 'btn-ghost' : 'btn-primary'}" onclick="event.stopPropagation();doVerifyUser('${u.id}', ${!u.isVerified})">${u.isVerified ? 'Unverify' : 'Verify'}</button>
          </div>
        `).join('') || '<p style="color:var(--text-tertiary)">No matches.</p>';
      } catch (e) { console.error(e); }
    }, 300);
  };
}

async function doVerifyUser(uid, verify) {
  if (!_isAdmin) return;
  try {
    await db.collection('users').doc(uid).update({ isVerified: verify });
    if (verify) VERIFIED_UIDS.add(uid); else VERIFIED_UIDS.delete(uid);
    toast(verify ? 'User verified!' : 'Verification removed');
    adminVerifyUser(); // refresh
  } catch (e) { toast('Failed'); console.error(e); }
}

async function adminModeratePosts() {
  if (!_isAdmin) return;
  closeModal();
  openModal(`
    <div class="modal-header"><h2>Moderate Posts</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" id="admin-posts-body"><div class="inline-spinner" style="margin:24px auto"></div></div>
  `);
  try {
    const snap = await db.collection('posts').orderBy('createdAt', 'desc').limit(50).get();
    const posts = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    const body = $('#admin-posts-body');
    body.innerHTML = posts.length ? posts.map(p => `
      <div class="asg-card" style="margin:8px 0">
        <div class="asg-card-title">${esc((p.content || '').slice(0, 100) || '[media post]')}</div>
        <div class="asg-card-meta"><span>By ${esc(p.authorName || 'User')}${verifiedBadge(p.authorId)}</span><span>\u00b7 ${timeAgo(p.createdAt)}</span></div>
        <div style="display:flex;gap:8px;margin-top:10px">
          <button class="btn-outline" style="flex:1" onclick="viewPost('${p.id}')">View</button>
          <button class="btn-danger" style="flex:1;border-radius:var(--radius)" onclick="adminDeletePost('${p.id}')">Delete</button>
        </div>
      </div>
    `).join('') : '<p style="padding:16px;color:var(--text-secondary)">No posts found.</p>';
  } catch (e) { console.error(e); }
}

async function adminDeletePost(postId) {
  if (!_isAdmin) return;
  try {
    await db.collection('posts').doc(postId).delete();
    toast('Post removed');
    adminModeratePosts();
  } catch (e) { toast('Delete failed'); console.error(e); }
}

function adminBroadcastPrompt() {
  if (!_isAdmin) return;
  closeModal();
  openModal(`
    <div class="modal-header"><h2>Broadcast Notice</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      <div class="form-group"><label>Message</label><textarea id="admin-broadcast-text" placeholder="Important campus-wide update..." style="resize:none;height:90px"></textarea></div>
      <button class="btn-primary btn-full" onclick="adminSendBroadcast()">Send to All Users</button>
    </div>
  `);
}

async function adminSendBroadcast() {
  if (!_isAdmin) return;
  const text = ($('#admin-broadcast-text')?.value || '').trim();
  if (!text) return toast('Message required');
  try {
    const usersSnap = await db.collection('users').get();
    await Promise.all(usersSnap.docs.map(d => db.collection('users').doc(d.id).collection('notifications').add({
      type: 'admin', text, payload: { admin: true }, read: false, createdAt: FieldVal.serverTimestamp(),
      from: { uid: state.user.uid, name: 'Unibo Admin', photo: state.profile.photoURL || null }
    })));
    closeModal();
    toast(`Broadcast sent to ${usersSnap.size} users`);
  } catch (e) { toast('Broadcast failed'); console.error(e); }
}

// ══════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initAuth();

  // Dismiss splash
  setTimeout(() => { const s = $('#splash'); if (s) s.classList.remove('active'); }, 1500);

  // Image viewer close
  $('#img-close')?.addEventListener('click', () => { $('#img-view').style.display = 'none'; });

  // Notifications dropdown toggle
  $('#notif-btn')?.addEventListener('click', (e) => {
    e.stopPropagation();
    const dd = $('#notif-dropdown');
    if (dd.style.display === 'block') { dd.style.display = 'none'; return; }
    loadNotifications();
    dd.style.display = 'block';
    // Close on outside click
    const closeDD = (ev) => {
      if (!dd.contains(ev.target) && ev.target !== $('#notif-btn') && !$('#notif-btn').contains(ev.target)) {
        dd.style.display = 'none';
        document.removeEventListener('click', closeDD);
      }
    };
    setTimeout(() => document.addEventListener('click', closeDD), 10);
  });

  // Expose globals for inline onclick
  Object.assign(window, {
    navigate, openProfile, openCreateModal, openSellModal,
    toggleLike, openComments, postComment, viewImage,
    startChat, openChat, closeModal, editProfile, doLogout, toast,
    showPostOptions, confirmDeletePost, deletePost, openProductDetail,
    openStoryCreator, viewStory, closeStoryViewer, advanceStory, deleteStory,
    openCreateGroup, openGroupChat, joinGroup, loadStories,
    openCreateModuleGroup, openGroupDetail, joinAsg, requestJoinAsg,
    approveAsgRequest, approveAsgRequestByUid, rejectAsgRequest, removeFromAsg, leaveAsg,
    toggleAsgLock, archiveAsg, doArchiveAsg, autoFillAsg,
    openAsgPreferences, openAsgChat, loadGroups,
    sendFriendRequest, acceptFriendRequest, rejectFriendRequest, unfriend,
    loadNotifications, setCommentReply, clearCommentReply,
    setDmReply, clearDmReply, setGroupReply, clearGroupReply,
    setCommentAnonChoice, editAnonNickname,
    openCreateEvent, openEventDetail, openLocationDetail, toggleEventGoing, deleteEvent,
    startAnonChat, removeEventImage, showUserPreview, openModuleFeed, openTagFeed, openAnonPostActions,
    startVoiceRecord, cancelVoiceRecord, stopVoiceAndSend, openReelsViewer,
    toggleCommentLike, openShareModal, repost, openQuoteRepost, shareToFriend, viewPost, markNotifRead,
    clearCommentImage, clearReelCommentImage, toggleReelCommentLike,
    setReelCommentReply, clearReelCommentReply,
    closeReelsViewer, toggleReelPlay, reelLike, togglePostExpand, shiftTrendingRail,
    toggleVN, seekVN,
    openAdminPanel, adminViewAllGroups, adminViewAllUsers, adminVerifyUser, doVerifyUser,
    adminModeratePosts, adminDeletePost, adminBroadcastPrompt, adminSendBroadcast,
    reportPost, submitPostReport, showAdminDataClear, adminDataClearStepTwo, doAdminDataClear,
    showConvoActions, archiveConvo, deleteConvo, blockUserFromChat, unblockUser, requestReveal,
    unarchiveConvo, loadArchivedDMList, toggleArchiveDmView, loadBlockedUsersList
  });
});
