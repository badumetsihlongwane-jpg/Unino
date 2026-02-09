/* ══════════════════════════════════════════════════════
 *  UNINO — Complete Application Engine v2
 *  Firebase Auth + Firestore | No Storage (base64)
 *  Features: Feed, Friends, Hashtags, Status, Marketplace,
 *            Messaging, Profiles, Explore, Suggestions
 * ══════════════════════════════════════════════════════ */

// ─── State ────────────────────────────────────────────
const state = {
  user: null,
  profile: null,
  currentPage: 'feed',
  posts: [],
  listings: [],
  conversations: [],
  users: [],
  totalUsers: 0,
  friends: [],        // array of friend UIDs
  friendRequests: [],  // incoming requests
  unsubscribers: [],
};

// ─── Shortcuts ────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const FieldVal = firebase.firestore.FieldValue;

const COLORS = [
  '#6C5CE7','#3B82F6','#10B981','#F59E0B','#EF4444',
  '#EC4899','#8B5CF6','#06B6D4','#F97316','#14B8A6'
];

function colorFor(name) {
  let h = 0;
  for (let i = 0; i < (name||'').length; i++) h = name.charCodeAt(i) + ((h << 5) - h);
  return COLORS[Math.abs(h) % COLORS.length];
}

function initials(name) {
  if (!name) return '?';
  const p = name.trim().split(/\s+/);
  return (p[0][0] + (p[1] ? p[1][0] : '')).toUpperCase();
}

function avatar(name, photo, cls = 'avatar-sm') {
  const bg = colorFor(name);
  if (photo) return `<div class="${cls}" style="background:${bg}"><img src="${photo}" alt="" onerror="this.parentElement.textContent='${initials(name)}'"></div>`;
  return `<div class="${cls}" style="background:${bg}">${initials(name)}</div>`;
}

function ago(ts) {
  if (!ts) return '';
  const d = ts.toDate ? ts.toDate() : new Date(ts);
  const m = Math.floor((Date.now() - d.getTime()) / 60000);
  if (m < 1) return 'Just now';
  if (m < 60) return m + 'm';
  const h = Math.floor(m / 60);
  if (h < 24) return h + 'h';
  const days = Math.floor(h / 24);
  if (days < 7) return days + 'd';
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function fmtTime(ts) {
  if (!ts) return '';
  const d = ts.toDate ? ts.toDate() : new Date(ts);
  return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

// Parse #hashtags in text, return HTML with clickable links
function parseHashtags(text) {
  return esc(text).replace(/#(\w+)/g, '<a class="hashtag" href="#" data-tag="$1">#$1</a>');
}

function compress(file, maxW = 800, q = 0.7) {
  return new Promise(res => {
    const r = new FileReader();
    r.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        const ratio = Math.min(maxW / img.width, 1);
        c.width = img.width * ratio;
        c.height = img.height * ratio;
        c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
        res(c.toDataURL('image/jpeg', q));
      };
      img.src = e.target.result;
    };
    r.readAsDataURL(file);
  });
}

// ─── Screens ──────────────────────────────────────────
function showScreen(id) {
  $$('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id)?.classList.add('active');
}

// ─── Theme ────────────────────────────────────────────
function initTheme() {
  const saved = localStorage.getItem('unino-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  $('#theme-toggle')?.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    const nxt = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', nxt);
    localStorage.setItem('unino-theme', nxt);
  });
}

// ─── User Count ───────────────────────────────────────
function listenUserCount() {
  const ref = db.collection('stats').doc('global');
  ref.get().then(s => { if (!s.exists) ref.set({ totalUsers: 0 }); });
  const unsub = ref.onSnapshot(s => {
    if (s.exists) { state.totalUsers = s.data().totalUsers || 0; updateCountUI(); }
  });
  state.unsubscribers.push(unsub);
}
function updateCountUI() {
  const c = state.totalUsers;
  const e1 = $('#auth-count-num'), e2 = $('#header-count');
  if (e1) e1.textContent = c;
  if (e2) e2.textContent = c;
}

// ══════════════════════════════════════════════════════
//  AUTH
// ══════════════════════════════════════════════════════
function initAuth() {
  $('#show-signup-link')?.addEventListener('click', e => {
    e.preventDefault();
    $('#login-form').classList.remove('active');
    $('#signup-form').classList.add('active');
  });
  $('#show-login-link')?.addEventListener('click', e => {
    e.preventDefault();
    $('#signup-form').classList.remove('active');
    $('#login-form').classList.add('active');
  });

  // Login
  $('#login-form')?.addEventListener('submit', async e => {
    e.preventDefault();
    const btn = $('#login-btn');
    const email = $('#login-email').value.trim();
    const pass = $('#login-password').value;
    if (!email || !pass) return toast('Fill in all fields');
    btn.disabled = true;
    btn.innerHTML = '<span class="inline-spinner"></span>';
    try { await auth.signInWithEmailAndPassword(email, pass); }
    catch (err) { toast(authErr(err.code)); btn.disabled = false; btn.textContent = 'Log In'; }
  });

  // Signup
  $('#signup-form')?.addEventListener('submit', async e => {
    e.preventDefault();
    const btn = $('#signup-btn');
    const fname = $('#signup-fname').value.trim();
    const lname = $('#signup-lname').value.trim();
    const email = $('#signup-email').value.trim();
    const pass = $('#signup-password').value;
    const uni = $('#signup-university').value;
    const major = $('#signup-major').value;
    const year = $('#signup-year').value;
    if (!fname || !lname || !email || !pass || !uni || !major) return toast('Please fill required fields');
    if (pass.length < 6) return toast('Password must be 6+ characters');
    btn.disabled = true;
    btn.innerHTML = '<span class="inline-spinner"></span>';
    try {
      const cred = await auth.createUserWithEmailAndPassword(email, pass);
      const uid = cred.user.uid;
      const displayName = fname + ' ' + lname;
      await db.collection('users').doc(uid).set({
        displayName, firstName: fname, lastName: lname, email,
        university: uni, major, year: year || '', bio: '',
        photoURL: '', status: 'online',
        joinedAt: FieldVal.serverTimestamp(),
        postsCount: 0, friendsCount: 0, friends: [],
      });
      await db.collection('stats').doc('global').set(
        { totalUsers: FieldVal.increment(1) }, { merge: true }
      );
      await cred.user.updateProfile({ displayName });
    } catch (err) {
      toast(authErr(err.code));
      btn.disabled = false; btn.textContent = 'Create Account';
    }
  });

  // Auth state
  auth.onAuthStateChanged(async user => {
    if (user) {
      state.user = user;
      const doc = await db.collection('users').doc(user.uid).get();
      state.profile = doc.exists
        ? { id: doc.id, ...doc.data() }
        : { id: user.uid, displayName: user.displayName || user.email, email: user.email, photoURL: '', university: '', major: '', bio: '', status: 'online', friends: [] };
      // Set online
      db.collection('users').doc(user.uid).update({ status: 'online' }).catch(() => {});
      enterApp();
    } else {
      state.user = null; state.profile = null;
      cleanup(); showScreen('auth-screen');
    }
  });
}

function authErr(code) {
  return ({
    'auth/user-not-found': 'No account with that email',
    'auth/wrong-password': 'Incorrect password',
    'auth/email-already-in-use': 'Email already registered',
    'auth/weak-password': 'Password too weak (6+ chars)',
    'auth/invalid-email': 'Invalid email address',
    'auth/too-many-requests': 'Too many attempts, try later',
    'auth/invalid-credential': 'Invalid email or password',
  })[code] || 'Something went wrong';
}

// ══════════════════════════════════════════════════════
//  ENTER APP
// ══════════════════════════════════════════════════════
function enterApp() {
  showScreen('app-shell');
  updateHeader();
  updateStatusUI();
  listenUserCount();
  loadFriendsList();
  navigateTo('feed');
  initNav();
  initHeaderActions();
}

function updateHeader() {
  const el = $('#header-avatar');
  if (!el || !state.profile) return;
  const n = state.profile.displayName || '';
  const p = state.profile.photoURL || '';
  el.innerHTML = p ? `<img src="${p}" alt="">` : initials(n);
  el.style.background = p ? 'none' : colorFor(n);
  el.onclick = () => showProfile(state.user.uid);
}

function updateStatusUI() {
  const s = state.profile?.status || 'online';
  const dot = $('#status-dot');
  const label = $('#status-label');
  if (dot) {
    dot.className = 'status-dot ' + s;
  }
  if (label) label.textContent = s === 'online' ? 'Online' : s === 'study' ? 'Studying' : 'Offline';
}

// ─── Friends List (Local Cache) ───────────────────────
async function loadFriendsList() {
  if (!state.user) return;
  try {
    const doc = await db.collection('users').doc(state.user.uid).get();
    state.friends = doc.data()?.friends || [];
  } catch (e) { state.friends = []; }
}

// ─── Nav ──────────────────────────────────────────────
function initNav() {
  $$('#bottom-nav .nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
      const pg = btn.dataset.page;
      if (pg === 'create') { openCreatePostModal(); return; }
      navigateTo(pg);
    });
  });
}

function navigateTo(page) {
  state.currentPage = page;
  $$('#bottom-nav .nav-item').forEach(b => b.classList.toggle('active', b.dataset.page === page));
  cleanup();
  const c = $('#app-content');
  c.scrollTop = 0;
  switch (page) {
    case 'feed': renderFeed(); break;
    case 'explore': renderExplore(); break;
    case 'hustle': renderHustle(); break;
    case 'messages': renderMessages(); break;
    default: renderFeed();
  }
}

function cleanup() {
  state.unsubscribers.forEach(fn => fn());
  state.unsubscribers = [];
}

function initHeaderActions() {
  // Status toggle
  $('#status-toggle-btn')?.addEventListener('click', cycleStatus);

  // Notifications
  $('#notif-btn')?.addEventListener('click', () => {
    const existing = document.querySelector('.notif-dropdown');
    if (existing) { existing.remove(); return; }
    showNotifDropdown();
  });
}

async function cycleStatus() {
  const order = ['online', 'study', 'offline'];
  const cur = state.profile?.status || 'online';
  const next = order[(order.indexOf(cur) + 1) % order.length];
  state.profile.status = next;
  updateStatusUI();
  try {
    await db.collection('users').doc(state.user.uid).update({ status: next });
    toast(next === 'online' ? 'You\'re visible' : next === 'study' ? 'Study mode on' : 'Gone invisible');
  } catch (e) { /* silent */ }
}

async function showNotifDropdown() {
  const dropdown = document.createElement('div');
  dropdown.className = 'notif-dropdown';

  // Load friend requests — simple query, filter client-side to avoid index
  let requestsHTML = '';
  try {
    const snap = await db.collection('friendRequests')
      .where('toId', '==', state.user.uid)
      .limit(20).get();
    const requests = snap.docs
      .map(d => ({ id: d.id, ...d.data() }))
      .filter(r => r.status === 'pending');
    if (requests.length > 0) {
      requestsHTML = requests.map(r => `
        <div class="notif-item" style="display:flex;align-items:center;gap:10px;padding:12px 16px;border-bottom:1px solid var(--border)">
          ${avatar(r.fromName, r.fromPhoto, 'avatar-sm')}
          <div style="flex:1;min-width:0">
            <div style="font-weight:600;font-size:13px">${esc(r.fromName)}</div>
            <div style="font-size:12px;color:var(--text-secondary)">Friend request</div>
          </div>
          <button class="btn-primary" style="padding:6px 12px;font-size:12px" onclick="acceptFriend('${r.id}','${r.fromId}')">Accept</button>
          <button class="btn-ghost" style="padding:6px 8px;font-size:12px" onclick="declineFriend('${r.id}')">✕</button>
        </div>
      `).join('');
    }
  } catch (e) { /* no requests */ }

  dropdown.innerHTML = `
    <div class="notif-dropdown-header"><h3>Notifications</h3></div>
    ${requestsHTML || '<div style="padding:32px 16px;text-align:center;color:var(--text-tertiary);font-size:14px">No new notifications</div>'}
  `;
  $('#app-header').appendChild(dropdown);
  setTimeout(() => {
    document.addEventListener('click', function close(e) {
      if (!dropdown.contains(e.target) && !e.target.closest('#notif-btn')) {
        dropdown.remove();
        document.removeEventListener('click', close);
      }
    });
  }, 10);
}

// ─── Friend Actions ───────────────────────────────────
async function sendFriendRequest(toId, toName, toPhoto) {
  const uid = state.user.uid;
  if (toId === uid) return;
  try {
    // Check if already friends
    if (state.friends.includes(toId)) return toast('Already friends!');
    // Check if already sent — simple query, filter client-side
    const existing = await db.collection('friendRequests')
      .where('fromId', '==', uid).limit(20).get();
    const alreadySent = existing.docs.some(d => d.data().toId === toId);
    if (alreadySent) return toast('Request already sent');
    await db.collection('friendRequests').add({
      fromId: uid,
      fromName: state.profile.displayName,
      fromPhoto: state.profile.photoURL || '',
      toId,
      toName: toName,
      toPhoto: toPhoto || '',
      status: 'pending',
      createdAt: FieldVal.serverTimestamp()
    });
    toast('Friend request sent!');
  } catch (e) { toast('Could not send request'); }
}

async function acceptFriend(requestId, fromId) {
  const uid = state.user.uid;
  try {
    await db.collection('friendRequests').doc(requestId).update({ status: 'accepted' });
    // Add to both users' friends arrays
    await db.collection('users').doc(uid).update({
      friends: FieldVal.arrayUnion(fromId),
      friendsCount: FieldVal.increment(1)
    });
    await db.collection('users').doc(fromId).update({
      friends: FieldVal.arrayUnion(uid),
      friendsCount: FieldVal.increment(1)
    });
    state.friends.push(fromId);
    state.profile.friendsCount = (state.profile.friendsCount || 0) + 1;
    toast('Friend added!');
    document.querySelector('.notif-dropdown')?.remove();
  } catch (e) { toast('Could not accept'); }
}

async function declineFriend(requestId) {
  try {
    await db.collection('friendRequests').doc(requestId).delete();
    toast('Request declined');
    document.querySelector('.notif-dropdown')?.remove();
  } catch (e) { /* silent */ }
}

async function removeFriend(otherId) {
  const uid = state.user.uid;
  try {
    await db.collection('users').doc(uid).update({
      friends: FieldVal.arrayRemove(otherId),
      friendsCount: FieldVal.increment(-1)
    });
    await db.collection('users').doc(otherId).update({
      friends: FieldVal.arrayRemove(uid),
      friendsCount: FieldVal.increment(-1)
    });
    state.friends = state.friends.filter(f => f !== otherId);
    state.profile.friendsCount = Math.max(0, (state.profile.friendsCount || 1) - 1);
    toast('Unfriended');
  } catch (e) { toast('Could not unfriend'); }
}

// ══════════════════════════════════════════════════════
//  FEED PAGE
// ══════════════════════════════════════════════════════
function renderFeed() {
  const c = $('#app-content');
  const greeting = getGreeting();
  c.innerHTML = `
    <div class="feed-page">
      <!-- Welcome Banner -->
      <div class="feed-welcome">
        <div class="feed-welcome-text">
          <span class="feed-greeting">${greeting}</span>
          <span class="feed-name">${esc(state.profile?.firstName || state.profile?.displayName || '')}</span>
        </div>
        <div class="status-pill" id="status-toggle-btn">
          <span class="status-dot ${state.profile?.status || 'online'}" id="status-dot"></span>
          <span id="status-label">${(state.profile?.status || 'online') === 'online' ? 'Online' : state.profile?.status === 'study' ? 'Studying' : 'Offline'}</span>
        </div>
      </div>

      <!-- Suggested Friends -->
      <div id="suggestions-section"></div>

      <!-- Create Post -->
      <div class="create-post-prompt" id="feed-create-prompt">
        ${avatar(state.profile?.displayName, state.profile?.photoURL, 'avatar-md')}
        <span class="placeholder-text">What\'s on your mind?</span>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
      </div>

      <!-- Trending Tags -->
      <div id="trending-tags"></div>

      <!-- Posts -->
      <div id="feed-posts">
        <div class="feed-loader"><span class="inline-spinner" style="width:32px;height:32px;border-width:3px;color:var(--accent)"></span></div>
      </div>
    </div>
  `;

  $('#feed-create-prompt')?.addEventListener('click', openCreatePostModal);
  $('#status-toggle-btn')?.addEventListener('click', cycleStatus);
  loadSuggestions();
  listenPosts();
}

function getGreeting() {
  const h = new Date().getHours();
  if (h < 12) return 'Good morning,';
  if (h < 17) return 'Good afternoon,';
  return 'Good evening,';
}

async function loadSuggestions() {
  const sec = $('#suggestions-section');
  if (!sec) return;
  try {
    const snap = await db.collection('users').limit(20).get();
    const users = snap.docs
      .map(d => ({ id: d.id, ...d.data() }))
      .filter(u => u.id !== state.user?.uid && !state.friends.includes(u.id));

    // Sort: same uni first, then same major
    const myUni = state.profile?.university || '';
    const myMajor = state.profile?.major || '';
    users.sort((a, b) => {
      const aScore = (a.university === myUni ? 2 : 0) + (a.major === myMajor ? 1 : 0);
      const bScore = (b.university === myUni ? 2 : 0) + (b.major === myMajor ? 1 : 0);
      return bScore - aScore;
    });

    const top = users.slice(0, 8);
    if (top.length === 0) { sec.innerHTML = ''; return; }

    sec.innerHTML = `
      <div class="suggestions-row">
        <div class="section-header"><h3>Suggested Friends</h3></div>
        <div class="suggestions-scroll">
          ${top.map(u => `
            <div class="suggestion-card" data-uid="${u.id}">
              ${avatar(u.displayName, u.photoURL, 'avatar-lg')}
              <div class="suggestion-name">${esc(u.displayName)}</div>
              <div class="suggestion-detail">${esc(u.university || u.major || '')}</div>
              ${u.status && u.status !== 'offline' ? `<span class="status-indicator ${u.status}"></span>` : ''}
              <button class="btn-primary suggestion-add-btn" data-uid="${u.id}" data-name="${esc(u.displayName)}" data-photo="${u.photoURL || ''}" style="padding:6px 14px;font-size:12px;margin-top:8px">Add Friend</button>
            </div>
          `).join('')}
        </div>
      </div>
    `;

    sec.querySelectorAll('.suggestion-card').forEach(card => {
      card.querySelector('.suggestion-add-btn')?.addEventListener('click', (e) => {
        e.stopPropagation();
        const uid = e.target.dataset.uid;
        sendFriendRequest(uid, e.target.dataset.name, e.target.dataset.photo);
        e.target.textContent = 'Sent!';
        e.target.disabled = true;
      });
      card.addEventListener('click', () => showProfile(card.dataset.uid));
    });
  } catch (e) { sec.innerHTML = ''; }
}

function listenPosts() {
  const unsub = db.collection('posts')
    .orderBy('createdAt', 'desc')
    .limit(50)
    .onSnapshot(snap => {
      state.posts = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      renderPosts();
      buildTrendingTags();
    }, err => {
      console.error('Posts error:', err);
      $('#feed-posts').innerHTML = emptyState('📝', 'No posts yet', 'Be the first to post!');
    });
  state.unsubscribers.push(unsub);
}

function buildTrendingTags() {
  const sec = $('#trending-tags');
  if (!sec) return;
  const tagCount = {};
  state.posts.forEach(p => {
    const tags = (p.content || '').match(/#(\w+)/g) || [];
    tags.forEach(t => { tagCount[t.toLowerCase()] = (tagCount[t.toLowerCase()] || 0) + 1; });
  });
  const sorted = Object.entries(tagCount).sort((a, b) => b[1] - a[1]).slice(0, 6);
  if (sorted.length === 0) { sec.innerHTML = ''; return; }
  sec.innerHTML = `
    <div class="trending-section">
      <div class="trending-chips">
        <span class="trending-label">🔥 Trending</span>
        ${sorted.map(([tag, count]) => `<span class="chip hashtag-chip" data-tag="${tag.replace('#', '')}">${tag} <small>${count}</small></span>`).join('')}
      </div>
    </div>
  `;
  sec.querySelectorAll('.hashtag-chip').forEach(chip => {
    chip.addEventListener('click', () => filterFeedByTag(chip.dataset.tag));
  });
}

function filterFeedByTag(tag) {
  const filtered = state.posts.filter(p => (p.content || '').toLowerCase().includes('#' + tag.toLowerCase()));
  const container = $('#feed-posts');
  if (!container) return;
  if (filtered.length === 0) {
    container.innerHTML = emptyState('🏷️', 'No posts with #' + tag, '');
    return;
  }
  container.innerHTML = `
    <div style="padding:12px 16px;display:flex;align-items:center;gap:8px">
      <button class="btn-ghost" onclick="renderPosts()" style="font-size:13px">← All posts</button>
      <span style="font-size:14px;font-weight:600;color:var(--accent)">#${esc(tag)}</span>
    </div>
    ${filtered.map(postHTML).join('')}
  `;
  attachPostListeners();
}

function renderPosts() {
  const container = $('#feed-posts');
  if (!container) return;
  if (state.posts.length === 0) {
    container.innerHTML = emptyState('📝', 'No posts yet', 'Be the first to share something!');
    return;
  }
  container.innerHTML = state.posts.map(postHTML).join('');
  attachPostListeners();
}

function postHTML(post) {
  const own = post.authorId === state.user?.uid;
  const liked = (post.likedBy || []).includes(state.user?.uid);
  const lc = post.likesCount || 0;
  const cc = post.commentsCount || 0;

  return `
    <div class="post-card" data-post-id="${post.id}">
      <div class="post-header">
        ${avatar(post.authorName, post.authorPhoto, 'avatar-md')}
        <div class="post-header-info">
          <div class="post-author-name" data-uid="${post.authorId}">${esc(post.authorName)}</div>
          <div class="post-meta">${esc(post.authorUni || '')} · ${ago(post.createdAt)}</div>
        </div>
        ${own ? `<button class="icon-btn post-more-btn" data-post-id="${post.id}"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/></svg></button>` : ''}
      </div>
      ${post.content ? `<div class="post-content">${parseHashtags(post.content)}</div>` : ''}
      ${post.imageURL ? `<div class="post-image-wrap"><img class="post-image" src="${post.imageURL}" alt="Post image" loading="lazy" data-full="${post.imageURL}"></div>` : ''}
      <div class="post-stats">
        <span class="like-count-${post.id}">${lc ? lc + (lc > 1 ? ' likes' : ' like') : ''}</span>
        <span class="comment-count-btn" data-post-id="${post.id}">${cc ? cc + (cc > 1 ? ' comments' : ' comment') : ''}</span>
      </div>
      <div class="post-actions">
        <button class="post-action like-btn ${liked ? 'liked' : ''}" data-post-id="${post.id}">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="${liked ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
          ${liked ? 'Liked' : 'Like'}
        </button>
        <button class="post-action comment-btn" data-post-id="${post.id}">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          Comment
        </button>
      </div>
      <div class="comments-section" id="comments-${post.id}" style="display:none"></div>
    </div>
  `;
}

function attachPostListeners() {
  $$('.like-btn').forEach(b => b.onclick = () => toggleLike(b.dataset.postId));
  $$('.comment-btn, .comment-count-btn').forEach(b => b.onclick = () => toggleComments(b.dataset.postId));
  $$('.post-image').forEach(i => i.onclick = () => openImageViewer(i.dataset.full));
  $$('.post-author-name').forEach(e => e.onclick = () => showProfile(e.dataset.uid));
  $$('.post-more-btn').forEach(b => b.onclick = (e) => { e.stopPropagation(); showPostOptions(b.dataset.postId); });
  // Hashtag clicks
  $$('.hashtag').forEach(a => {
    a.onclick = (e) => { e.preventDefault(); filterFeedByTag(a.dataset.tag); };
  });
}

// ─── Like ─────────────────────────────────────────────
async function toggleLike(pid) {
  const uid = state.user.uid;
  const ref = db.collection('posts').doc(pid);
  try {
    const doc = await ref.get();
    if (!doc.exists) return;
    const liked = (doc.data().likedBy || []).includes(uid);
    await ref.update({
      likedBy: liked ? FieldVal.arrayRemove(uid) : FieldVal.arrayUnion(uid),
      likesCount: FieldVal.increment(liked ? -1 : 1)
    });
  } catch (e) { console.error('Like err:', e); }
}

// ─── Comments ─────────────────────────────────────────
function toggleComments(pid) {
  const s = $(`#comments-${pid}`);
  if (!s) return;
  if (s.style.display === 'none') {
    s.style.display = 'block';
    s.innerHTML = '<div style="padding:12px;text-align:center"><span class="inline-spinner"></span></div>';
    loadComments(pid);
  } else s.style.display = 'none';
}

async function loadComments(pid) {
  const s = $(`#comments-${pid}`);
  if (!s) return;
  try {
    const snap = await db.collection('posts').doc(pid).collection('comments')
      .orderBy('createdAt', 'asc').limit(20).get();
    const comments = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    s.innerHTML = `
      ${comments.map(c => `
        <div class="comment-item">
          ${avatar(c.authorName, c.authorPhoto, 'avatar-sm')}
          <div class="comment-body">
            <span class="comment-author" data-uid="${c.authorId}" style="cursor:pointer">${esc(c.authorName)}</span>
            <div class="comment-text">${parseHashtags(c.content)}</div>
            <div class="comment-time">${ago(c.createdAt)}</div>
          </div>
        </div>
      `).join('')}
      <div class="comment-input-row">
        <input type="text" placeholder="Write a comment..." id="ci-${pid}">
        <button onclick="submitComment('${pid}')">Post</button>
      </div>
    `;
    $(`#ci-${pid}`)?.addEventListener('keypress', e => { if (e.key === 'Enter') submitComment(pid); });
    s.querySelectorAll('.comment-author').forEach(el => el.onclick = () => showProfile(el.dataset.uid));
  } catch (e) {
    s.innerHTML = '<p style="padding:12px;color:var(--text-tertiary);font-size:13px">Could not load comments</p>';
  }
}

async function submitComment(pid) {
  const input = $(`#ci-${pid}`);
  if (!input) return;
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  try {
    await db.collection('posts').doc(pid).collection('comments').add({
      content: text, authorId: state.user.uid,
      authorName: state.profile.displayName,
      authorPhoto: state.profile.photoURL || '',
      createdAt: FieldVal.serverTimestamp()
    });
    await db.collection('posts').doc(pid).update({ commentsCount: FieldVal.increment(1) });
    loadComments(pid);
  } catch (e) { toast('Could not post comment'); }
}

// ─── Post Options ─────────────────────────────────────
function showPostOptions(pid) {
  openModal(`
    <div style="padding:20px">
      <h3 style="margin-bottom:16px">Post Options</h3>
      <button class="btn-danger btn-full" id="delete-post-btn" style="margin-bottom:8px">Delete Post</button>
      <button class="btn-ghost btn-full" onclick="closeModal()">Cancel</button>
    </div>
  `);
  $('#delete-post-btn')?.addEventListener('click', async () => {
    try {
      await db.collection('posts').doc(pid).delete();
      await db.collection('users').doc(state.user.uid).update({ postsCount: FieldVal.increment(-1) });
      toast('Post deleted'); closeModal();
    } catch (e) { toast('Could not delete'); }
  });
}

// ─── Create Post ──────────────────────────────────────
function openCreatePostModal() {
  let pendingImg = '';
  openModal(`
    <div class="create-post-form">
      <h2>Create Post</h2>
      <div class="create-post-top">
        ${avatar(state.profile?.displayName, state.profile?.photoURL, 'avatar-md')}
        <div>
          <strong>${esc(state.profile?.displayName)}</strong>
          <div style="font-size:12px;color:var(--text-secondary)">${state.profile?.university || ''}</div>
        </div>
      </div>
      <textarea id="new-post-text" placeholder="What's happening? Use #hashtags for courses!" maxlength="1000"></textarea>
      <div class="hashtag-hint">💡 Tip: Use #CS101 or #StudyGroup to tag courses & topics</div>
      <div class="image-preview-container" id="post-img-preview">
        <img src="" alt="" id="post-preview-img">
        <button class="image-preview-remove" id="post-rm-img">&times;</button>
      </div>
      <div class="create-post-bottom">
        <div class="attach-row">
          <label class="attach-btn" for="post-img-upload">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
            Photo
          </label>
          <input type="file" id="post-img-upload" accept="image/*" hidden>
        </div>
        <button class="btn-primary" id="submit-post-btn">Post</button>
      </div>
    </div>
  `);

  $('#post-img-upload')?.addEventListener('change', async e => {
    const f = e.target.files[0];
    if (!f) return;
    if (f.size > 5*1024*1024) return toast('Image too large (max 5MB)');
    pendingImg = await compress(f, 800, 0.7);
    $('#post-preview-img').src = pendingImg;
    $('#post-img-preview').style.display = 'block';
  });
  $('#post-rm-img')?.addEventListener('click', () => {
    pendingImg = '';
    $('#post-img-preview').style.display = 'none';
    $('#post-img-upload').value = '';
  });
  $('#submit-post-btn')?.addEventListener('click', async () => {
    const text = $('#new-post-text').value.trim();
    if (!text && !pendingImg) return toast('Write something or add a photo');
    const btn = $('#submit-post-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="inline-spinner"></span>';
    try {
      await db.collection('posts').add({
        content: text, imageURL: pendingImg || '',
        authorId: state.user.uid, authorName: state.profile.displayName,
        authorPhoto: state.profile.photoURL || '',
        authorUni: state.profile.university || '',
        likesCount: 0, commentsCount: 0, likedBy: [],
        createdAt: FieldVal.serverTimestamp()
      });
      await db.collection('users').doc(state.user.uid).update({ postsCount: FieldVal.increment(1) });
      toast('Posted!'); closeModal();
    } catch (e) {
      toast('Could not create post');
      btn.disabled = false; btn.textContent = 'Post';
    }
  });
}

// ══════════════════════════════════════════════════════
//  EXPLORE PAGE
// ══════════════════════════════════════════════════════
function renderExplore() {
  const c = $('#app-content');
  c.innerHTML = `
    <div class="explore-page">
      <div class="search-bar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <input type="text" placeholder="Search students..." id="explore-search">
      </div>
      <div class="filter-chips" id="explore-filters">
        <span class="chip active" data-filter="all">All</span>
        <span class="chip" data-filter="Computer Science">CS</span>
        <span class="chip" data-filter="Engineering">Eng</span>
        <span class="chip" data-filter="Business / Commerce">Biz</span>
        <span class="chip" data-filter="Law">Law</span>
        <span class="chip" data-filter="Medicine / Health Sciences">Med</span>
        <span class="chip" data-filter="Arts & Design">Arts</span>
      </div>
      <div class="users-grid" id="explore-users">
        <div style="grid-column:1/-1" class="feed-loader"><span class="inline-spinner" style="width:32px;height:32px;border-width:3px;color:var(--accent)"></span></div>
      </div>
    </div>
  `;
  loadExploreUsers();
  let tmr;
  $('#explore-search')?.addEventListener('input', e => {
    clearTimeout(tmr);
    tmr = setTimeout(() => filterUsers(e.target.value.trim()), 300);
  });
  $$('#explore-filters .chip').forEach(c => c.addEventListener('click', () => {
    $$('#explore-filters .chip').forEach(x => x.classList.remove('active'));
    c.classList.add('active');
    filterUsers($('#explore-search')?.value || '', c.dataset.filter);
  }));
}

async function loadExploreUsers() {
  try {
    const snap = await db.collection('users').limit(50).get();
    state.users = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    displayUsers(state.users.filter(u => u.id !== state.user?.uid));
  } catch (e) {
    $('#explore-users').innerHTML = emptyState('🔍', 'No students found', 'Invite friends to join!');
  }
}

function filterUsers(q, major = 'all') {
  let f = state.users.filter(u => u.id !== state.user?.uid);
  if (q) {
    const ql = q.toLowerCase();
    f = f.filter(u => (u.displayName||'').toLowerCase().includes(ql) || (u.university||'').toLowerCase().includes(ql) || (u.major||'').toLowerCase().includes(ql));
  }
  if (major && major !== 'all') f = f.filter(u => u.major === major);
  displayUsers(f);
}

function displayUsers(users) {
  const c = $('#explore-users');
  if (!c) return;
  if (!users.length) { c.innerHTML = `<div style="grid-column:1/-1">${emptyState('🔍', 'No students found', '')}</div>`; return; }
  c.innerHTML = users.map(u => {
    const isFriend = state.friends.includes(u.id);
    const statusClass = u.status && u.status !== 'offline' ? u.status : '';
    return `
      <div class="user-card" data-uid="${u.id}">
        <div class="user-card-avatar-wrap">
          ${avatar(u.displayName, u.photoURL, 'avatar-lg')}
          ${statusClass ? `<span class="status-indicator ${statusClass}"></span>` : ''}
        </div>
        <div class="user-card-name">${esc(u.displayName)}</div>
        <div class="user-card-uni">${esc(u.university || '')}</div>
        ${u.major ? `<div class="user-card-major">${esc(u.major)}</div>` : ''}
        ${isFriend ? '<div style="font-size:11px;color:var(--green);font-weight:600;margin-top:4px">✓ Friends</div>' : ''}
      </div>
    `;
  }).join('');
  c.querySelectorAll('.user-card').forEach(card => card.onclick = () => showProfile(card.dataset.uid));
}

// ══════════════════════════════════════════════════════
//  HUSTLE (MARKETPLACE)
// ══════════════════════════════════════════════════════
function renderHustle() {
  const c = $('#app-content');
  c.innerHTML = `
    <div class="hustle-page">
      <div class="hustle-header">
        <h2>💰 The Hustle</h2>
        <button class="btn-primary" id="create-listing-btn" style="padding:10px 16px;font-size:13px">+ Sell</button>
      </div>
      <div class="search-bar" style="margin-bottom:12px">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <input type="text" placeholder="Search marketplace..." id="hustle-search">
      </div>
      <div class="category-tabs" id="hustle-cats">
        <span class="chip active" data-cat="all">All</span>
        <span class="chip" data-cat="textbook">📚 Textbooks</span>
        <span class="chip" data-cat="electronics">💻 Electronics</span>
        <span class="chip" data-cat="furniture">🪑 Furniture</span>
        <span class="chip" data-cat="service">🛠️ Services</span>
        <span class="chip" data-cat="tutoring">📖 Tutoring</span>
        <span class="chip" data-cat="other">🏷️ Other</span>
      </div>
      <div class="listings-grid" id="listings-grid">
        <div style="grid-column:1/-1" class="feed-loader"><span class="inline-spinner" style="width:32px;height:32px;border-width:3px;color:var(--accent)"></span></div>
      </div>
    </div>
  `;
  loadListings();
  $('#create-listing-btn')?.addEventListener('click', openCreateListing);
  let tmr;
  $('#hustle-search')?.addEventListener('input', e => { clearTimeout(tmr); tmr = setTimeout(() => filterListings(e.target.value.trim()), 300); });
  $$('#hustle-cats .chip').forEach(ch => ch.addEventListener('click', () => {
    $$('#hustle-cats .chip').forEach(x => x.classList.remove('active'));
    ch.classList.add('active');
    filterListings($('#hustle-search')?.value || '', ch.dataset.cat);
  }));
}

async function loadListings() {
  try {
    const snap = await db.collection('listings').where('status', '==', 'active').orderBy('createdAt', 'desc').limit(50).get();
    state.listings = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    displayListings(state.listings);
  } catch (e) {
    console.error('Listings err:', e);
    $('#listings-grid').innerHTML = `<div style="grid-column:1/-1">${emptyState('🏪', 'Marketplace empty', 'List something!')}</div>`;
  }
}

function filterListings(q, cat = 'all') {
  let f = [...state.listings];
  if (q) { const ql = q.toLowerCase(); f = f.filter(l => (l.title||'').toLowerCase().includes(ql) || (l.description||'').toLowerCase().includes(ql)); }
  if (cat && cat !== 'all') f = f.filter(l => l.category === cat);
  displayListings(f);
}

function displayListings(listings) {
  const c = $('#listings-grid');
  if (!c) return;
  if (!listings.length) { c.innerHTML = `<div style="grid-column:1/-1">${emptyState('🏪', 'Nothing here', 'Try another search')}</div>`; return; }
  const emoji = { textbook:'📚', electronics:'💻', furniture:'🪑', service:'🛠️', tutoring:'📖', other:'🏷️' };
  c.innerHTML = listings.map(l => `
    <div class="listing-card" data-lid="${l.id}">
      ${l.imageURL
        ? `<img class="listing-image" src="${l.imageURL}" alt="">`
        : `<div class="listing-image" style="display:flex;align-items:center;justify-content:center;font-size:48px">${emoji[l.category] || '🏷️'}</div>`}
      <div class="listing-info">
        <div class="listing-title">${esc(l.title)}</div>
        <div class="listing-price">R${Number(l.price||0).toLocaleString()}</div>
        <div class="listing-seller">${avatar(l.sellerName, l.sellerPhoto, 'avatar-sm')} ${esc(l.sellerName||'')}</div>
      </div>
    </div>
  `).join('');
  c.querySelectorAll('.listing-card').forEach(card => card.onclick = () => showListingDetail(card.dataset.lid));
}

function showListingDetail(lid) {
  const l = state.listings.find(x => x.id === lid);
  if (!l) return;
  const own = l.sellerId === state.user?.uid;
  const emoji = { textbook:'📚', electronics:'💻', furniture:'🪑', service:'🛠️', tutoring:'📖', other:'🏷️' };
  openModal(`
    <div class="listing-detail">
      ${l.imageURL ? `<img class="listing-detail-image" src="${l.imageURL}" alt="">` : `<div class="listing-detail-image" style="display:flex;align-items:center;justify-content:center;font-size:80px">${emoji[l.category]||'🏷️'}</div>`}
      <div class="listing-detail-title">${esc(l.title)}</div>
      <div class="listing-detail-price">R${Number(l.price||0).toLocaleString()}</div>
      <div class="listing-detail-desc">${esc(l.description || 'No description')}</div>
      <div class="listing-detail-seller" data-uid="${l.sellerId}">
        ${avatar(l.sellerName, l.sellerPhoto, 'avatar-md')}
        <div class="listing-detail-seller-info">
          <div class="listing-detail-seller-name">${esc(l.sellerName)}</div>
          <div class="listing-detail-seller-uni">${esc(l.sellerUni || '')}</div>
        </div>
      </div>
      ${own ? `<button class="btn-danger btn-full" id="del-listing">Remove</button>` : `<button class="btn-primary btn-full" id="msg-seller">Message Seller</button>`}
    </div>
  `);
  $('.listing-detail-seller')?.addEventListener('click', () => { closeModal(); showProfile(l.sellerId); });
  $('#msg-seller')?.addEventListener('click', () => { closeModal(); startConvo(l.sellerId, l.sellerName, l.sellerPhoto); });
  $('#del-listing')?.addEventListener('click', async () => {
    try { await db.collection('listings').doc(lid).delete(); toast('Removed'); closeModal(); loadListings(); }
    catch (e) { toast('Could not delete'); }
  });
}

function openCreateListing() {
  let pendingImg = '';
  openModal(`
    <div class="create-listing-form">
      <h2>Sell Something</h2>
      <div class="form-group"><label>Title *</label><input type="text" id="l-title" placeholder="What are you selling?" maxlength="100"></div>
      <div class="form-group"><label>Price (ZAR) *</label><input type="number" id="l-price" placeholder="0" min="0"></div>
      <div class="form-group"><label>Category *</label>
        <select id="l-cat"><option value="">Select</option><option value="textbook">📚 Textbook</option><option value="electronics">💻 Electronics</option><option value="furniture">🪑 Furniture</option><option value="service">🛠️ Service</option><option value="tutoring">📖 Tutoring</option><option value="other">🏷️ Other</option></select>
      </div>
      <div class="form-group"><label>Description</label><textarea id="l-desc" rows="3" style="resize:vertical" placeholder="Describe..."></textarea></div>
      <div class="form-group">
        <label>Photo</label>
        <div class="image-preview-container" id="l-img-prev"><img src="" id="l-prev-img" alt=""><button class="image-preview-remove" id="l-rm-img">&times;</button></div>
        <label class="attach-btn" for="l-img-upload" style="display:inline-flex;margin-top:8px"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg> Add Photo</label>
        <input type="file" id="l-img-upload" accept="image/*" hidden>
      </div>
      <button class="btn-primary btn-full" id="submit-listing" style="margin-top:8px">List for Sale</button>
    </div>
  `);
  $('#l-img-upload')?.addEventListener('change', async e => {
    const f = e.target.files[0]; if (!f) return;
    if (f.size > 5*1024*1024) return toast('Too large');
    pendingImg = await compress(f, 600, 0.7);
    $('#l-prev-img').src = pendingImg; $('#l-img-prev').style.display = 'block';
  });
  $('#l-rm-img')?.addEventListener('click', () => { pendingImg = ''; $('#l-img-prev').style.display = 'none'; });
  $('#submit-listing')?.addEventListener('click', async () => {
    const title = $('#l-title').value.trim(), price = parseFloat($('#l-price').value)||0, cat = $('#l-cat').value, desc = $('#l-desc').value.trim();
    if (!title || !cat) return toast('Title and category required');
    const btn = $('#submit-listing'); btn.disabled = true; btn.innerHTML = '<span class="inline-spinner"></span>';
    try {
      await db.collection('listings').add({
        title, price, category: cat, description: desc, imageURL: pendingImg || '',
        sellerId: state.user.uid, sellerName: state.profile.displayName,
        sellerPhoto: state.profile.photoURL || '', sellerUni: state.profile.university || '',
        status: 'active', createdAt: FieldVal.serverTimestamp()
      });
      toast('Listed!'); closeModal(); loadListings();
    } catch (e) { toast('Could not list'); btn.disabled = false; btn.textContent = 'List for Sale'; }
  });
}

// ══════════════════════════════════════════════════════
//  MESSAGES
// ══════════════════════════════════════════════════════
function renderMessages() {
  const c = $('#app-content');
  c.innerHTML = `<div class="messages-page"><div class="messages-header"><h2>Messages</h2></div><div id="convo-list"><div class="feed-loader"><span class="inline-spinner" style="width:32px;height:32px;border-width:3px;color:var(--accent)"></span></div></div></div>`;
  listenConvos();
}

function listenConvos() {
  const uid = state.user?.uid; if (!uid) return;
  const unsub = db.collection('conversations')
    .where('participants', 'array-contains', uid)
    .orderBy('lastMessageAt', 'desc').limit(30)
    .onSnapshot(snap => {
      state.conversations = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      displayConvos();
    }, () => displayConvos());
  state.unsubscribers.push(unsub);
}

function displayConvos() {
  const c = $('#convo-list'); if (!c) return;
  if (!state.conversations.length) { c.innerHTML = emptyState('💬', 'No messages', 'Start from a profile or listing'); return; }
  const uid = state.user.uid;
  c.innerHTML = `<div class="convo-list">${state.conversations.map(cv => {
    const idx = cv.participants?.[0] === uid ? 1 : 0;
    const name = cv.participantNames?.[idx] || 'User';
    const photo = cv.participantPhotos?.[idx] || '';
    const unread = cv.unreadCount?.[uid] || 0;
    return `<div class="convo-item" data-cid="${cv.id}">${avatar(name, photo, 'avatar-md')}<div class="convo-info"><div class="convo-name">${esc(name)}</div><div class="convo-last-msg">${esc(cv.lastMessage || 'No messages')}</div></div><div class="convo-right"><div class="convo-time">${ago(cv.lastMessageAt)}</div>${unread > 0 ? `<div class="convo-unread">${unread}</div>` : ''}</div></div>`;
  }).join('')}</div>`;
  c.querySelectorAll('.convo-item').forEach(i => i.onclick = () => openChat(i.dataset.cid));
  // Badge
  const total = state.conversations.reduce((s, c) => s + (c.unreadCount?.[state.user.uid] || 0), 0);
  const badge = $('#chat-badge');
  if (badge) badge.textContent = total > 0 ? total : '';
}

async function startConvo(otherId, otherName, otherPhoto) {
  const uid = state.user.uid;
  if (otherId === uid) return toast('That\'s you!');
  try {
    const snap = await db.collection('conversations').where('participants', 'array-contains', uid).get();
    const ex = snap.docs.find(d => d.data().participants?.includes(otherId));
    if (ex) { openChat(ex.id); return; }
    const ref = await db.collection('conversations').add({
      participants: [uid, otherId],
      participantNames: [state.profile.displayName, otherName],
      participantPhotos: [state.profile.photoURL || '', otherPhoto || ''],
      lastMessage: '', lastMessageAt: FieldVal.serverTimestamp(),
      unreadCount: { [uid]: 0, [otherId]: 0 },
      createdAt: FieldVal.serverTimestamp()
    });
    openChat(ref.id);
  } catch (e) { toast('Could not start conversation'); }
}

let chatUnsub = null;
function openChat(cid) {
  const cv = state.conversations.find(c => c.id === cid);
  const uid = state.user.uid;
  const idx = cv?.participants?.[0] === uid ? 1 : 0;
  const name = cv?.participantNames?.[idx] || 'User';
  const photo = cv?.participantPhotos?.[idx] || '';

  showScreen('chat-view');
  $('#chat-name').textContent = name;
  $('#chat-status').textContent = 'Online';
  const avatarEl = $('#chat-avatar');
  if (avatarEl) avatarEl.outerHTML = avatar(name, photo, 'avatar-sm');

  const msgC = $('#chat-messages');
  msgC.innerHTML = '<div class="feed-loader"><span class="inline-spinner" style="width:28px;height:28px;border-width:2px;color:var(--accent)"></span></div>';

  db.collection('conversations').doc(cid).update({ [`unreadCount.${uid}`]: 0 }).catch(() => {});

  if (chatUnsub) chatUnsub();
  chatUnsub = db.collection('conversations').doc(cid).collection('messages')
    .orderBy('createdAt', 'asc').limit(100)
    .onSnapshot(snap => {
      renderChat(snap.docs.map(d => ({ id: d.id, ...d.data() })));
    });

  $('#chat-back-btn').onclick = () => { if (chatUnsub) { chatUnsub(); chatUnsub = null; } showScreen('app-shell'); };

  const input = $('#chat-input');
  const send = async () => {
    const text = input.value.trim(); if (!text) return; input.value = '';
    const otherUid = cv?.participants?.find(p => p !== uid) || '';
    try {
      await db.collection('conversations').doc(cid).collection('messages').add({ text, senderId: uid, senderName: state.profile.displayName, createdAt: FieldVal.serverTimestamp() });
      await db.collection('conversations').doc(cid).update({ lastMessage: text, lastMessageAt: FieldVal.serverTimestamp(), [`unreadCount.${otherUid}`]: FieldVal.increment(1) });
    } catch (e) { toast('Send failed'); }
  };
  $('#chat-send-btn').onclick = send;
  input.onkeypress = e => { if (e.key === 'Enter') send(); };

  $('#chat-img-input').onchange = async e => {
    const f = e.target.files[0]; if (!f) return;
    const data = await compress(f, 500, 0.6);
    const otherUid = cv?.participants?.find(p => p !== uid) || '';
    try {
      await db.collection('conversations').doc(cid).collection('messages').add({ text: '', imageURL: data, senderId: uid, senderName: state.profile.displayName, createdAt: FieldVal.serverTimestamp() });
      await db.collection('conversations').doc(cid).update({ lastMessage: '📷 Photo', lastMessageAt: FieldVal.serverTimestamp(), [`unreadCount.${otherUid}`]: FieldVal.increment(1) });
    } catch (e) { toast('Image failed'); }
  };
}

function renderChat(msgs) {
  const c = $('#chat-messages'); if (!c) return;
  const uid = state.user.uid;
  if (!msgs.length) { c.innerHTML = '<div style="text-align:center;padding:32px;color:var(--text-tertiary)">Say hello! 👋</div>'; return; }
  c.innerHTML = msgs.map(m => {
    const sent = m.senderId === uid;
    return `<div class="msg-bubble ${sent ? 'msg-sent' : 'msg-received'}">
      ${m.imageURL ? `<img class="msg-image" src="${m.imageURL}" alt="" onclick="openImageViewer(this.src)">` : ''}
      ${m.text ? esc(m.text) : ''}
      <div class="msg-time">${fmtTime(m.createdAt)}</div>
    </div>`;
  }).join('');
  c.scrollTop = c.scrollHeight;
}

// ══════════════════════════════════════════════════════
//  PROFILE VIEW — FIXED (no compound index needed)
// ══════════════════════════════════════════════════════
async function showProfile(uid) {
  showScreen('profile-view');
  const pc = $('#profile-content');
  pc.innerHTML = '<div class="feed-loader" style="padding:48px"><span class="inline-spinner" style="width:32px;height:32px;border-width:3px;color:var(--accent)"></span></div>';
  const isOwn = uid === state.user?.uid;

  try {
    let profile;
    if (isOwn && state.profile) {
      profile = state.profile;
    } else {
      const doc = await db.collection('users').doc(uid).get();
      if (!doc.exists) { pc.innerHTML = emptyState('😢', 'User not found', ''); return; }
      profile = { id: doc.id, ...doc.data() };
    }

    // Fetch posts WITHOUT compound index — just filter by authorId, sort client-side
    let userPosts = [];
    try {
      const postsSnap = await db.collection('posts').where('authorId', '==', uid).limit(20).get();
      userPosts = postsSnap.docs.map(d => ({ id: d.id, ...d.data() }));
      // Sort client-side instead of Firestore orderBy (avoids index requirement)
      userPosts.sort((a, b) => {
        const ta = a.createdAt?.toDate?.() || new Date(0);
        const tb = b.createdAt?.toDate?.() || new Date(0);
        return tb - ta;
      });
    } catch (e) { console.warn('Posts fetch failed:', e); }

    // Friend status
    const isFriend = state.friends.includes(uid);
    let pendingRequest = false;
    if (!isOwn && !isFriend) {
      try {
        // Simple query — just check fromId, filter client-side to avoid compound index
        const reqSnap = await db.collection('friendRequests')
          .where('fromId', '==', state.user.uid).limit(20).get();
        pendingRequest = reqSnap.docs.some(d => {
          const data = d.data();
          return data.toId === uid && data.status === 'pending';
        });
      } catch (e) { console.warn('Friend request check failed:', e); }
    }

    $('#profile-top-name').textContent = profile.displayName || '';

    const statusClass = profile.status && profile.status !== 'offline' ? profile.status : '';
    pc.innerHTML = `
      <div class="profile-banner"></div>
      <div class="profile-avatar-wrapper">
        <div class="profile-avatar-large" style="background:${colorFor(profile.displayName)}">
          ${profile.photoURL ? `<img src="${profile.photoURL}" alt="" onerror="this.remove()">` : initials(profile.displayName)}
        </div>
        ${statusClass ? `<span class="profile-status-dot ${statusClass}"></span>` : ''}
      </div>
      <div class="profile-info">
        <div class="profile-name">${esc(profile.displayName)}</div>
        <div class="profile-uni">${esc(profile.university || '')} ${profile.year ? '· ' + esc(profile.year) : ''}</div>
        ${profile.major ? `<div class="profile-major-pill">${esc(profile.major)}</div>` : ''}
        ${profile.bio ? `<div class="profile-bio">${esc(profile.bio)}</div>` : ''}
        <div class="profile-stats-row">
          <div class="profile-stat"><div class="profile-stat-num">${userPosts.length}</div><div class="profile-stat-label">Posts</div></div>
          <div class="profile-stat"><div class="profile-stat-num">${profile.friendsCount || 0}</div><div class="profile-stat-label">Friends</div></div>
        </div>
        <div class="profile-actions">
          ${isOwn ? `
            <button class="btn-primary" id="edit-profile-btn">Edit Profile</button>
            <button class="btn-outline" id="logout-btn" style="color:var(--red);border-color:var(--red)">Log Out</button>
          ` : `
            ${isFriend
              ? `<button class="btn-outline" id="unfriend-btn">✓ Friends</button>`
              : pendingRequest
                ? `<button class="btn-outline" disabled>Request Sent</button>`
                : `<button class="btn-primary" id="add-friend-btn">+ Add Friend</button>`
            }
            <button class="btn-outline" id="dm-profile-btn">Message</button>
          `}
        </div>
      </div>
      <div class="profile-posts-header">${isOwn ? 'Your' : esc(profile.firstName || 'Their')} Posts</div>
      <div id="profile-posts-list">
        ${!userPosts.length
          ? '<div style="padding:24px;text-align:center;color:var(--text-tertiary);font-size:14px">No posts yet</div>'
          : userPosts.map(postHTML).join('')
        }
      </div>
    `;

    // Attach listeners
    if (isOwn) {
      $('#edit-profile-btn')?.addEventListener('click', openEditProfile);
      $('#logout-btn')?.addEventListener('click', () => {
        db.collection('users').doc(state.user.uid).update({ status: 'offline' }).catch(() => {});
        auth.signOut();
        showScreen('auth-screen');
      });
    } else {
      $('#dm-profile-btn')?.addEventListener('click', () => {
        showScreen('app-shell');
        startConvo(uid, profile.displayName, profile.photoURL || '');
      });
      $('#add-friend-btn')?.addEventListener('click', (e) => {
        sendFriendRequest(uid, profile.displayName, profile.photoURL || '');
        e.target.textContent = 'Sent!';
        e.target.disabled = true;
      });
      $('#unfriend-btn')?.addEventListener('click', async () => {
        await removeFriend(uid);
        showProfile(uid); // refresh
      });
    }
    attachPostListeners();

  } catch (err) {
    console.error('Profile error:', err);
    pc.innerHTML = emptyState('😢', 'Could not load profile', 'Please try again');
  }

  $('#profile-back-btn').onclick = () => showScreen('app-shell');
}

// ─── Edit Profile ─────────────────────────────────────
function openEditProfile() {
  const p = state.profile;
  let newPhoto = '';
  openModal(`
    <div class="edit-profile-form">
      <h2>Edit Profile</h2>
      <div class="avatar-upload">
        <div class="avatar-upload-preview" id="edit-av-prev" style="background:${colorFor(p.displayName)}">
          ${p.photoURL ? `<img src="${p.photoURL}" alt="">` : initials(p.displayName)}
        </div>
        <div>
          <label class="btn-outline" for="edit-av-upload" style="cursor:pointer;display:inline-block">Change Photo</label>
          <input type="file" id="edit-av-upload" accept="image/*" hidden>
          <div style="font-size:12px;color:var(--text-tertiary);margin-top:4px">Max 2MB</div>
        </div>
      </div>
      <div class="form-group"><label>Display Name</label><input type="text" id="edit-name" value="${esc(p.displayName||'')}" maxlength="50"></div>
      <div class="form-group"><label>Bio</label><textarea id="edit-bio" rows="3" maxlength="200" style="resize:vertical">${esc(p.bio||'')}</textarea></div>
      <div class="form-group"><label>University</label><input type="text" id="edit-uni" value="${esc(p.university||'')}"></div>
      <div class="form-group"><label>Major</label><input type="text" id="edit-major" value="${esc(p.major||'')}"></div>
      <button class="btn-primary btn-full" id="save-profile" style="margin-top:8px">Save Changes</button>
    </div>
  `);
  $('#edit-av-upload')?.addEventListener('change', async e => {
    const f = e.target.files[0]; if (!f) return;
    if (f.size > 3*1024*1024) return toast('Too large');
    newPhoto = await compress(f, 200, 0.6);
    $('#edit-av-prev').innerHTML = `<img src="${newPhoto}" alt="">`;
  });
  $('#save-profile')?.addEventListener('click', async () => {
    const btn = $('#save-profile'); btn.disabled = true; btn.innerHTML = '<span class="inline-spinner"></span>';
    const updates = {
      displayName: $('#edit-name').value.trim() || p.displayName,
      bio: $('#edit-bio').value.trim(),
      university: $('#edit-uni').value.trim(),
      major: $('#edit-major').value.trim(),
    };
    if (newPhoto) updates.photoURL = newPhoto;
    try {
      await db.collection('users').doc(state.user.uid).update(updates);
      Object.assign(state.profile, updates);
      updateHeader();
      toast('Saved!'); closeModal(); showProfile(state.user.uid);
    } catch (e) { toast('Could not save'); btn.disabled = false; btn.textContent = 'Save Changes'; }
  });
}

// ══════════════════════════════════════════════════════
//  MODAL, IMAGE VIEWER, HELPERS
// ══════════════════════════════════════════════════════
function openModal(html) {
  const o = $('#modal-overlay'), b = $('#modal-body');
  b.innerHTML = html; o.style.display = 'flex';
  o.onclick = e => { if (e.target === o) closeModal(); };
}

function closeModal() {
  $('#modal-overlay').style.display = 'none';
  $('#modal-body').innerHTML = '';
}

function openImageViewer(src) {
  const v = $('#image-viewer');
  $('#image-viewer-img').src = src;
  v.style.display = 'flex';
}

function initImageViewer() {
  $('#image-viewer-close')?.addEventListener('click', () => { $('#image-viewer').style.display = 'none'; });
  $('#image-viewer')?.addEventListener('click', e => { if (e.target === $('#image-viewer')) $('#image-viewer').style.display = 'none'; });
}

function emptyState(icon, title, desc) {
  return `<div class="empty-state"><div class="empty-state-icon">${icon}</div><h3>${title}</h3>${desc ? `<p>${desc}</p>` : ''}</div>`;
}

// ══════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initImageViewer();
  initAuth();
  listenUserCount();
  setTimeout(() => {
    if ($('#loading-screen').classList.contains('active')) showScreen('auth-screen');
  }, 4000);
});

// Global handlers for inline onclick
window.submitComment = submitComment;
window.closeModal = closeModal;
window.openImageViewer = openImageViewer;
window.renderPosts = renderPosts;
window.acceptFriend = acceptFriend;
window.declineFriend = declineFriend;
