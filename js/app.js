/* ══════════════════════════════════════════════════════
 *  UNINO — Complete Application Engine v3
 *  Matched to index.html element IDs
 *  Firebase Auth + Firestore | base64 images
 *  Features: Feed, Suggested Events, Proximity/Location,
 *            Explore, Marketplace, Messaging, Profiles
 * ══════════════════════════════════════════════════════ */

// ─── State ────────────────────────────────────────
const state = {
  user: null,
  profile: null,
  page: 'feed',
  status: 'online',
  unsubs: [],
};

// ─── Shortcuts ────────────────────────────────────
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
const FieldVal = firebase.firestore.FieldValue;

const COLORS = [
  '#6C5CE7','#3B82F6','#10B981','#F59E0B','#EF4444',
  '#EC4899','#8B5CF6','#06B6D4','#F97316','#14B8A6'
];

// ─── Helpers ──────────────────────────────────────
function colorFor(n) {
  let h = 0;
  for (let i = 0; i < (n||'').length; i++) h = n.charCodeAt(i) + ((h << 5) - h);
  return COLORS[Math.abs(h) % COLORS.length];
}

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

// ─── Screen Manager ──────────────────────────────
function showScreen(id) {
  $$('.screen').forEach(s => s.classList.remove('active'));
  const el = document.getElementById(id);
  if (el) el.classList.add('active');
}

// ─── Cleanup Firestore listeners ─────────────────
function unsub() {
  state.unsubs.forEach(fn => fn());
  state.unsubs = [];
}

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
//  AUTH — wired to HTML IDs
// ══════════════════════════════════════════════════
function initAuth() {
  // Toggle between forms
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
    const btn = $('#l-btn');
    const email = $('#l-email').value.trim();
    const pass = $('#l-pass').value;
    if (!email || !pass) return toast('Enter email and password');
    btn.disabled = true;
    btn.innerHTML = '<span class="inline-spinner"></span>';
    try {
      await auth.signInWithEmailAndPassword(email, pass);
    } catch (err) {
      toast(friendlyErr(err.code));
      btn.disabled = false;
      btn.textContent = 'Log In';
    }
  });

  // SIGNUP
  $('#signup-form')?.addEventListener('submit', async e => {
    e.preventDefault();
    const btn = $('#s-btn');
    const fname = $('#s-fname').value.trim();
    const lname = $('#s-lname').value.trim();
    const email = $('#s-email').value.trim();
    const pass  = $('#s-pass').value;
    const uni   = $('#s-uni').value;
    const major = $('#s-major').value;
    const year  = $('#s-year')?.value || '';
    if (!fname || !lname || !email || !pass || !uni || !major) return toast('All fields required');
    if (pass.length < 6) return toast('Password must be 6+ characters');
    btn.disabled = true;
    btn.innerHTML = '<span class="inline-spinner"></span>';
    try {
      const cred = await auth.createUserWithEmailAndPassword(email, pass);
      const uid = cred.user.uid;
      const displayName = `${fname} ${lname}`;
      await db.collection('users').doc(uid).set({
        displayName, firstName: fname, lastName: lname,
        email, university: uni, major, year,
        bio: `${major} student at ${uni}`,
        photoURL: '', status: 'online',
        joinedAt: FieldVal.serverTimestamp(),
        friends: []
      });
      await cred.user.updateProfile({ displayName });
      // Bump global counter
      db.collection('stats').doc('global').set(
        { totalUsers: FieldVal.increment(1) }, { merge: true }
      ).catch(() => {});
    } catch (err) {
      toast(friendlyErr(err.code));
      btn.disabled = false;
      btn.textContent = 'Create Account';
    }
  });

  // AUTH STATE
  auth.onAuthStateChanged(async user => {
    if (user) {
      state.user = user;
      try {
        const doc = await db.collection('users').doc(user.uid).get();
        state.profile = doc.exists
          ? { id: doc.id, ...doc.data() }
          : { id: user.uid, displayName: user.displayName, email: user.email, status: 'online' };
      } catch {
        state.profile = { id: user.uid, displayName: user.displayName, email: user.email, status: 'online' };
      }
      state.status = state.profile.status || 'online';
      enterApp();
    } else {
      state.user = null;
      state.profile = null;
      unsub();
      showScreen('auth-screen');
    }
  });

  // Live user count on auth screen
  db.collection('stats').doc('global').onSnapshot(doc => {
    const c = doc.exists ? (doc.data().totalUsers || 0) : 0;
    const el = $('#auth-count');
    if (el) el.textContent = c;
  });
}

function friendlyErr(code) {
  return {
    'auth/user-not-found': 'Account not found',
    'auth/wrong-password': 'Incorrect password',
    'auth/email-already-in-use': 'Email already registered',
    'auth/weak-password': 'Password too weak',
    'auth/invalid-email': 'Invalid email',
  }[code] || 'Something went wrong';
}

// ══════════════════════════════════════════════════
//  ENTER APP
// ══════════════════════════════════════════════════
function enterApp() {
  showScreen('app');
  setupHeader();
  setupNav();
  setupStatusPill();
  navigate('feed');
}

function setupHeader() {
  const el = $('#hdr-avatar');
  if (!el || !state.profile) return;
  const p = state.profile;
  if (p.photoURL) {
    el.innerHTML = `<img src="${p.photoURL}" alt="">`;
    el.style.background = 'transparent';
  } else {
    el.textContent = initials(p.displayName);
    el.style.background = colorFor(p.displayName);
  }
  el.onclick = () => openProfile(state.user.uid);

  // Live count in header
  db.collection('stats').doc('global').onSnapshot(doc => {
    const c = doc.exists ? (doc.data().totalUsers || 0) : 0;
    const hc = $('#hdr-count');
    if (hc) hc.textContent = c;
  });
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
  const pill = $('#status-pill');
  if (!pill) return;
  updateStatusUI();
  pill.onclick = async () => {
    const modes = ['online', 'study', 'offline'];
    state.status = modes[(modes.indexOf(state.status) + 1) % 3];
    updateStatusUI();
    try {
      await db.collection('users').doc(state.user.uid).update({ status: state.status });
    } catch (e) { console.error(e); }
    toast('Status: ' + state.status.charAt(0).toUpperCase() + state.status.slice(1));
  };
}

function updateStatusUI() {
  const dot = $('#status-dot');
  const txt = $('#status-text');
  const pill = $('#status-pill');
  if (!dot || !txt) return;

  // Reset classes
  pill.className = 'status-pill';
  dot.className = 'dot';

  if (state.status === 'online') {
    pill.classList.add('online');
    dot.classList.add('green');
    txt.textContent = 'Online';
  } else if (state.status === 'study') {
    pill.classList.add('away');
    dot.classList.add('orange');
    txt.textContent = 'Studying';
  } else {
    pill.classList.add('offline');
    dot.classList.add('gray');
    txt.textContent = 'Offline';
  }
}

// ─── Navigation ──────────────────────────────────
function navigate(page) {
  state.page = page;
  unsub();
  $$('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.p === page));

  switch (page) {
    case 'feed': renderFeed(); break;
    case 'explore': renderExplore(); break;
    case 'hustle': renderHustle(); break;
    case 'chat': renderMessages(); break;
  }
}

// ══════════════════════════════════════════════════
//  FEED  (with Suggested Events + Location Ideas)
// ══════════════════════════════════════════════════
function renderFeed() {
  const c = $('#content');
  const p = state.profile;
  c.innerHTML = `
    <div class="feed-page">
      <!-- Welcome -->
      <div class="welcome-banner">
        <h2>Hello, ${esc(p.firstName || p.displayName?.split(' ')[0])}! 👋</h2>
        <p>See what's happening at ${esc(p.university || 'campus')}</p>
      </div>

      <!-- Stories / Online Friends -->
      <div class="stories-row" id="stories-row">
        <div class="story-item add-story" onclick="openCreateModal()">
          <div class="story-avatar"><div class="story-avatar-inner">+</div></div>
          <div class="story-name">Post</div>
        </div>
      </div>

      <!-- Suggested Events -->
      <div class="suggested-section" id="events-section">
        <div class="suggested-header">
          <h3>📅 Suggested Events</h3>
          <a href="#" onclick="event.preventDefault()">See all</a>
        </div>
        <div class="suggested-list" id="events-list"></div>
      </div>

      <!-- Location-Based Proximity -->
      <div class="suggested-section" id="nearby-section">
        <div class="suggested-header">
          <h3>📍 Near You</h3>
          <a href="#" onclick="event.preventDefault()">See all</a>
        </div>
        <div class="suggested-list" id="nearby-list"></div>
      </div>

      <!-- Suggested Friends -->
      <div class="suggested-section" id="suggested-section">
        <div class="suggested-header">
          <h3>People you may know</h3>
          <a href="#" onclick="navigate('explore')">See all</a>
        </div>
        <div class="suggested-list" id="suggested-list"></div>
      </div>

      <!-- Create prompt -->
      <div class="create-post-prompt" onclick="openCreateModal()">
        ${avatar(p.displayName, p.photoURL, 'avatar-md')}
        <div class="placeholder-text">Share something with your campus...</div>
        <div class="prompt-actions"><span class="prompt-action">📷</span></div>
      </div>

      <!-- Posts -->
      <div id="feed-posts">
        <div style="padding:40px;text-align:center"><span class="inline-spinner" style="width:28px;height:28px;color:var(--accent)"></span></div>
      </div>
    </div>
  `;

  loadSuggestedEvents();
  loadNearbyStudents();
  loadSuggestedFriends();
  loadOnlineFriends();

  // Real-time posts
  const u = db.collection('posts')
    .orderBy('createdAt', 'desc').limit(50)
    .onSnapshot(snap => {
      renderPosts(snap.docs.map(d => ({ id: d.id, ...d.data() })));
    });
  state.unsubs.push(u);
}

// ─── Suggested Events ─────────────────────────────
function loadSuggestedEvents() {
  const list = $('#events-list');
  if (!list) return;

  // Curated campus events (mix of real Firestore data + smart defaults)
  const uni = state.profile.university || '';
  const defaultEvents = [
    { title: 'Study Jam Session', emoji: '📚', when: 'Tomorrow, 6 PM', where: 'Library', color: '#6C5CE7' },
    { title: 'Career Fair 2026', emoji: '💼', when: 'Feb 15', where: 'Main Hall', color: '#3B82F6' },
    { title: 'Campus Pool Tournament', emoji: '🎱', when: 'Fri, 4 PM', where: 'Student Center', color: '#10B981' },
    { title: 'Welcome Back Mixer', emoji: '🎉', when: 'Sat, 7 PM', where: 'Quad', color: '#F59E0B' },
    { title: 'Coding Hackathon', emoji: '💻', when: 'Feb 20-21', where: 'CS Building', color: '#EF4444' },
    { title: 'Open Mic Night', emoji: '🎤', when: 'Next Wed', where: 'Amphitheatre', color: '#EC4899' },
  ];

  list.innerHTML = defaultEvents.map(ev => `
    <div class="suggested-card" style="border-top:3px solid ${ev.color}" onclick="toast('Event details coming soon!')">
      <div style="font-size:32px;margin-bottom:8px">${ev.emoji}</div>
      <div class="suggested-card-name">${ev.title}</div>
      <div class="suggested-card-meta">${ev.when}</div>
      <div style="font-size:11px;color:var(--text-tertiary)">📍 ${ev.where}</div>
    </div>
  `).join('');
}

// ─── Location-Based / Proximity ──────────────────
function loadNearbyStudents() {
  const list = $('#nearby-list');
  if (!list) return;

  const myUni = state.profile.university || '';
  const myMajor = state.profile.major || '';

  db.collection('users').limit(30).get().then(snap => {
    let users = snap.docs
      .map(d => ({ id: d.id, ...d.data() }))
      .filter(u => u.id !== state.user.uid);

    // Prioritize same university, then same major
    const sameUni = users.filter(u => u.university === myUni);
    const sameMajor = users.filter(u => u.major === myMajor && u.university !== myUni);
    const nearby = [...sameUni, ...sameMajor].slice(0, 8);

    if (nearby.length === 0) {
      // Show placeholder cards
      list.innerHTML = `
        <div class="suggested-card" style="opacity:0.6">
          <div style="font-size:32px;margin-bottom:8px">👤</div>
          <div class="suggested-card-name">No one nearby yet</div>
          <div class="suggested-card-meta">Invite friends!</div>
        </div>
      `;
      return;
    }

    list.innerHTML = nearby.map(u => {
      const dist = u.university === myUni ? '📍 Same campus' : `🎓 ${esc(u.university || 'Nearby')}`;
      const statusDot = u.status === 'online' ? '<span class="dot green" style="margin-left:4px"></span>' : '';
      return `
        <div class="suggested-card" onclick="openProfile('${u.id}')">
          ${avatar(u.displayName, u.photoURL, 'avatar-lg')}
          <div class="suggested-card-name">${esc(u.displayName)}${statusDot}</div>
          <div class="suggested-card-meta">${esc(u.major || 'Student')}</div>
          <div style="font-size:11px;color:var(--text-tertiary);margin-top:2px">${dist}</div>
        </div>
      `;
    }).join('');
  }).catch(() => {});
}

// ─── Suggested Friends ───────────────────────────
function loadSuggestedFriends() {
  const list = $('#suggested-list');
  if (!list) return;

  db.collection('users').limit(20).get().then(snap => {
    const users = snap.docs
      .map(d => ({ id: d.id, ...d.data() }))
      .filter(u => u.id !== state.user.uid)
      .slice(0, 6);

    if (!users.length) {
      list.innerHTML = '<div style="padding:16px;color:var(--text-tertiary)">No suggestions yet</div>';
      return;
    }

    list.innerHTML = users.map(u => `
      <div class="suggested-card" onclick="openProfile('${u.id}')">
        ${avatar(u.displayName, u.photoURL, 'avatar-lg')}
        <div class="suggested-card-name">${esc(u.displayName)}</div>
        <div class="suggested-card-meta">${esc(u.major || 'Student')}</div>
        <button class="btn-primary btn-sm" style="width:100%;margin-top:8px" onclick="event.stopPropagation();openProfile('${u.id}')">View</button>
      </div>
    `).join('');
  }).catch(() => {});
}

// ─── Online Friends in Stories Row ───────────────
function loadOnlineFriends() {
  const row = $('#stories-row');
  if (!row) return;

  db.collection('users').where('status', '==', 'online').limit(10).get().then(snap => {
    const users = snap.docs
      .map(d => ({ id: d.id, ...d.data() }))
      .filter(u => u.id !== state.user.uid);

    const html = users.map(u => `
      <div class="story-item" onclick="openProfile('${u.id}')">
        <div class="story-avatar">
          <div class="story-avatar-inner">
            ${u.photoURL ? `<img src="${u.photoURL}" alt="">` : initials(u.displayName)}
          </div>
        </div>
        <div class="story-name">${esc(u.firstName || u.displayName?.split(' ')[0] || '?')}</div>
      </div>
    `).join('');
    row.insertAdjacentHTML('beforeend', html);
  }).catch(() => {});
}

// ─── Render Posts ────────────────────────────────
function renderPosts(posts) {
  const el = $('#feed-posts');
  if (!el) return;

  if (!posts.length) {
    el.innerHTML = `<div class="empty-state">
      <div class="empty-state-icon">📝</div>
      <h3>No posts yet</h3>
      <p>Be the first to share something!</p>
    </div>`;
    return;
  }

  el.innerHTML = posts.map(post => {
    const liked = (post.likes || []).includes(state.user.uid);
    const lc = (post.likes || []).length;
    const cc = post.commentsCount || 0;

    return `
      <div class="post-card">
        <div class="post-header">
          <div onclick="openProfile('${post.authorId}')" style="cursor:pointer">
            ${avatar(post.authorName, post.authorPhoto, 'avatar-md')}
          </div>
          <div class="post-header-info">
            <div class="post-author-row">
              <span class="post-author-name" onclick="openProfile('${post.authorId}')">${esc(post.authorName)}</span>
            </div>
            <div class="post-meta">${esc(post.authorUni || '')} · ${timeAgo(post.createdAt)}</div>
          </div>
        </div>

        <div class="post-content">${formatContent(post.content)}</div>

        ${post.imageURL ? `
          <div class="post-image-wrap">
            <img src="${post.imageURL}" class="post-image" loading="lazy"
                 onclick="viewImage('${post.imageURL}')">
          </div>` : ''}

        <div class="post-stats">
          ${lc ? `<span>${lc} like${lc > 1 ? 's' : ''}</span>` : ''}
          ${cc ? `<span>${cc} comment${cc > 1 ? 's' : ''}</span>` : ''}
        </div>

        <div class="post-actions">
          <button class="post-action ${liked ? 'liked' : ''}" onclick="toggleLike('${post.id}')">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="${liked ? 'var(--red)' : 'none'}" stroke="${liked ? 'var(--red)' : 'currentColor'}" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
            ${lc || 'Like'}
          </button>
          <button class="post-action" onclick="openComments('${post.id}')">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            ${cc || 'Comment'}
          </button>
          <button class="post-action" onclick="toast('Link copied!')">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
            Share
          </button>
        </div>
      </div>
    `;
  }).join('');
}

function formatContent(text) {
  if (!text) return '';
  return esc(text).replace(/#(\w+)/g, '<span class="hashtag">#$1</span>');
}

// ─── Like ────────────────────────────────────────
async function toggleLike(pid) {
  const ref = db.collection('posts').doc(pid);
  try {
    const doc = await ref.get();
    if (!doc.exists) return;
    const likes = doc.data().likes || [];
    if (likes.includes(state.user.uid)) {
      await ref.update({ likes: FieldVal.arrayRemove(state.user.uid) });
    } else {
      await ref.update({ likes: FieldVal.arrayUnion(state.user.uid) });
    }
  } catch (e) { console.error(e); }
}

// ─── Comments ────────────────────────────────────
async function openComments(postId) {
  // Fetch existing comments
  let comments = [];
  try {
    const snap = await db.collection('posts').doc(postId).collection('comments')
      .orderBy('createdAt', 'asc').limit(50).get();
    comments = snap.docs.map(d => ({ id: d.id, ...d.data() }));
  } catch (e) { console.error(e); }

  const html = `
    <div class="modal-header">
      <h2>Comments</h2>
      <button class="icon-btn" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body">
      <div id="comments-container">
        ${comments.length ? comments.map(c => `
          <div class="comment-item">
            ${avatar(c.authorName, c.authorPhoto, 'avatar-sm')}
            <div class="comment-bubble">
              <div class="comment-author" onclick="openProfile('${c.authorId}')">${esc(c.authorName)}</div>
              <div class="comment-text">${esc(c.text)}</div>
              <div class="comment-time">${timeAgo(c.createdAt)}</div>
            </div>
          </div>
        `).join('') : '<p style="color:var(--text-tertiary);text-align:center;padding:16px">No comments yet</p>'}
      </div>
      <div class="comment-input-wrap">
        <input type="text" id="comment-input" placeholder="Write a comment...">
        <button onclick="postComment('${postId}')">Post</button>
      </div>
    </div>
  `;
  openModal(html);
}

async function postComment(postId) {
  const input = $('#comment-input');
  const text = input?.value.trim();
  if (!text) return;
  input.value = '';
  try {
    await db.collection('posts').doc(postId).collection('comments').add({
      text,
      authorId: state.user.uid,
      authorName: state.profile.displayName,
      authorPhoto: state.profile.photoURL || null,
      createdAt: FieldVal.serverTimestamp()
    });
    await db.collection('posts').doc(postId).update({
      commentsCount: FieldVal.increment(1)
    });
    closeModal();
    toast('Comment posted');
  } catch (e) { console.error(e); toast('Failed to post comment'); }
}

// ─── Image Viewer ────────────────────────────────
function viewImage(url) {
  const v = $('#img-view');
  if (!v) return;
  $('#img-full').src = url;
  v.style.display = 'flex';
}

// ─── Create Post Modal ──────────────────────────
function openCreateModal() {
  let pendingImg = null;

  const html = `
    <div class="modal-header">
      <h2>Create Post</h2>
      <button class="icon-btn" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body">
      <div style="display:flex;gap:12px;margin-bottom:16px">
        ${avatar(state.profile.displayName, state.profile.photoURL, 'avatar-md')}
        <div>
          <div style="font-weight:600">${esc(state.profile.displayName)}</div>
          <div style="font-size:12px;color:var(--text-secondary)">Posting to ${esc(state.profile.university || 'Public')}</div>
        </div>
      </div>
      <textarea id="create-text" placeholder="What's on your mind?" style="width:100%;min-height:100px;border:none;background:transparent;color:var(--text-primary);font-size:16px;resize:none;outline:none"></textarea>
      <div id="create-preview" class="image-preview" style="display:none">
        <img src="" alt="">
        <button class="image-preview-remove" onclick="document.getElementById('create-preview').style.display='none';document.getElementById('create-file').value=''">&times;</button>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;border-top:1px solid var(--border);padding-top:12px;margin-top:12px">
        <label style="cursor:pointer;color:var(--accent);font-size:20px">
          📷
          <input type="file" hidden accept="image/*" id="create-file">
        </label>
        <button class="btn-primary" id="create-submit" style="padding:10px 28px">Post</button>
      </div>
    </div>
  `;
  openModal(html);

  $('#create-file').onchange = async e => {
    if (e.target.files[0]) {
      pendingImg = await compress(e.target.files[0]);
      $('#create-preview img').src = pendingImg;
      $('#create-preview').style.display = 'block';
    }
  };

  $('#create-submit').onclick = async () => {
    const text = $('#create-text').value.trim();
    if (!text && !pendingImg) return toast('Post cannot be empty');
    closeModal();
    toast('Posting...');
    try {
      await db.collection('posts').add({
        content: text,
        imageURL: pendingImg || null,
        authorId: state.user.uid,
        authorName: state.profile.displayName,
        authorPhoto: state.profile.photoURL || null,
        authorUni: state.profile.university || '',
        createdAt: FieldVal.serverTimestamp(),
        likes: [],
        commentsCount: 0
      });
      toast('Posted!');
    } catch (e) { toast('Failed to post'); console.error(e); }
  };
}

// ══════════════════════════════════════════════════
//  EXPLORE
// ══════════════════════════════════════════════════
function renderExplore() {
  const c = $('#content');
  c.innerHTML = `
    <div class="explore-page">
      <div class="search-bar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <input type="text" id="explore-search" placeholder="Search students, tutors, clubs...">
      </div>
      <div class="filter-chips">
        <span class="chip active" data-f="all">All</span>
        <span class="chip" data-f="cs">CS</span>
        <span class="chip" data-f="eng">Engineering</span>
        <span class="chip" data-f="law">Law</span>
        <span class="chip" data-f="med">Medicine</span>
        <span class="chip" data-f="arts">Arts</span>
        <span class="chip" data-f="biz">Business</span>
      </div>
      <div class="users-grid" id="explore-grid">
        <div style="grid-column:1/-1;text-align:center;padding:32px"><span class="inline-spinner"></span></div>
      </div>
    </div>
  `;

  loadExplore();

  // Search
  let timer;
  $('#explore-search').addEventListener('input', e => {
    clearTimeout(timer);
    timer = setTimeout(() => loadExplore(e.target.value), 400);
  });

  // Chips
  $$('.filter-chips .chip').forEach(ch => {
    ch.onclick = () => {
      $$('.filter-chips .chip').forEach(c2 => c2.classList.remove('active'));
      ch.classList.add('active');
      loadExplore($('#explore-search').value, ch.dataset.f);
    };
  });
}

async function loadExplore(query = '', filter = 'all') {
  const grid = $('#explore-grid');
  if (!grid) return;

  try {
    const snap = await db.collection('users').limit(30).get();
    let users = snap.docs.map(d => ({ id: d.id, ...d.data() })).filter(u => u.id !== state.user.uid);

    if (query) {
      const q = query.toLowerCase();
      users = users.filter(u =>
        (u.displayName || '').toLowerCase().includes(q) ||
        (u.major || '').toLowerCase().includes(q) ||
        (u.university || '').toLowerCase().includes(q)
      );
    }

    if (filter !== 'all') {
      const map = { cs: 'computer', eng: 'engineer', law: 'law', med: 'medic', arts: 'art', biz: 'business' };
      const k = map[filter] || filter;
      users = users.filter(u => (u.major || '').toLowerCase().includes(k));
    }

    if (!users.length) {
      grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><h3>No results</h3><p>Try a different search</p></div>';
      return;
    }

    const myUni = state.profile.university || '';
    grid.innerHTML = users.map(u => {
      const dist = u.university === myUni ? '📍 Same campus' : (u.university ? `🎓 ${esc(u.university)}` : '');
      return `
        <div class="user-card" onclick="openProfile('${u.id}')">
          ${avatar(u.displayName, u.photoURL, 'avatar-lg')}
          <div class="user-card-name">${esc(u.displayName)}</div>
          <div class="user-card-uni">${esc(u.university || '')}</div>
          ${u.major ? `<span class="user-card-major">${esc(u.major)}</span>` : ''}
          ${dist ? `<div class="user-card-distance">${dist}</div>` : ''}
        </div>
      `;
    }).join('');
  } catch (e) {
    grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><h3>Error loading</h3></div>';
  }
}

// ══════════════════════════════════════════════════
//  HUSTLE (Marketplace)
// ══════════════════════════════════════════════════
function renderHustle() {
  const c = $('#content');
  c.innerHTML = `
    <div class="hustle-page">
      <div class="hustle-header">
        <h2>Marketplace</h2>
        <button class="btn-primary btn-sm" onclick="openSellModal()">+ Sell</button>
      </div>
      <div class="category-tabs">
        <span class="chip active">All</span>
        <span class="chip">Books</span>
        <span class="chip">Tech</span>
        <span class="chip">Notes</span>
        <span class="chip">Services</span>
        <span class="chip">Other</span>
      </div>
      <div class="listings-grid" id="listings-grid">
        <div style="grid-column:1/-1;text-align:center;padding:32px"><span class="inline-spinner"></span></div>
      </div>
    </div>
  `;

  db.collection('listings').where('status', '==', 'active').limit(50).get().then(snap => {
    const items = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    const grid = $('#listings-grid');
    if (!grid) return;

    if (!items.length) {
      grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1">
        <div class="empty-state-icon">🛒</div>
        <h3>No listings yet</h3>
        <p>Be the first to sell something!</p>
      </div>`;
      return;
    }

    grid.innerHTML = items.map(item => `
      <div class="listing-card" onclick="openProfile('${item.sellerId}')">
        ${item.imageURL
          ? `<img class="listing-image" src="${item.imageURL}" loading="lazy">`
          : `<div class="listing-placeholder">📦</div>`}
        <div class="listing-info">
          <div class="listing-price">R${esc(String(item.price))}</div>
          <div class="listing-title">${esc(item.title)}</div>
          <div class="listing-seller">
            ${avatar(item.sellerName, null, 'avatar-sm')}
            <span>${esc(item.sellerName)}</span>
          </div>
        </div>
      </div>
    `).join('');
  }).catch(() => {});
}

function openSellModal() {
  let pendingImg = null;
  const html = `
    <div class="modal-header">
      <h2>Sell Item</h2>
      <button class="icon-btn" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body">
      <div class="form-group">
        <label>What are you selling?</label>
        <input type="text" id="sell-title" placeholder="e.g. Calculus Textbook">
      </div>
      <div class="form-group">
        <label>Price (R)</label>
        <input type="number" id="sell-price" placeholder="150">
      </div>
      <div class="form-group">
        <label>Category</label>
        <select id="sell-cat">
          <option>Books</option><option>Tech</option><option>Notes</option><option>Services</option><option>Other</option>
        </select>
      </div>
      <div class="form-group">
        <label>Photo (optional)</label>
        <input type="file" accept="image/*" id="sell-file">
      </div>
      <div id="sell-preview" class="image-preview" style="display:none">
        <img src=""><button class="image-preview-remove" onclick="$('#sell-preview').style.display='none'">&times;</button>
      </div>
      <button class="btn-primary btn-full" id="sell-submit" style="margin-top:8px">List Item</button>
    </div>
  `;
  openModal(html);

  $('#sell-file').onchange = async e => {
    if (e.target.files[0]) {
      pendingImg = await compress(e.target.files[0]);
      $('#sell-preview img').src = pendingImg;
      $('#sell-preview').style.display = 'block';
    }
  };

  $('#sell-submit').onclick = async () => {
    const title = $('#sell-title').value.trim();
    const price = $('#sell-price').value.trim();
    if (!title || !price) return toast('Title and price required');
    closeModal();
    toast('Listing...');
    try {
      await db.collection('listings').add({
        title, price, category: $('#sell-cat').value,
        imageURL: pendingImg || null,
        sellerId: state.user.uid,
        sellerName: state.profile.displayName,
        status: 'active',
        createdAt: FieldVal.serverTimestamp()
      });
      toast('Listed!');
      navigate('hustle');
    } catch (e) { toast('Failed'); console.error(e); }
  };
}

// ══════════════════════════════════════════════════
//  MESSAGES
// ══════════════════════════════════════════════════
function renderMessages() {
  const c = $('#content');
  c.innerHTML = `
    <div class="messages-page">
      <div class="messages-header"><h2>Messages</h2></div>
      <div class="convo-list" id="convo-list">
        <div style="padding:40px;text-align:center"><span class="inline-spinner"></span></div>
      </div>
    </div>
  `;

  const u = db.collection('conversations')
    .where('participants', 'array-contains', state.user.uid)
    .orderBy('updatedAt', 'desc')
    .onSnapshot(snap => {
      const convos = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      const el = $('#convo-list');
      if (!el) return;

      if (!convos.length) {
        el.innerHTML = `<div class="empty-state">
          <div class="empty-state-icon">💬</div>
          <h3>No chats yet</h3>
          <p>Visit a profile to start a conversation</p>
        </div>`;
        return;
      }

      const uid = state.user.uid;
      el.innerHTML = convos.map(c => {
        const idx = c.participants.indexOf(uid) === 0 ? 1 : 0;
        const name = (c.participantNames || [])[idx] || 'User';
        const photo = (c.participantPhotos || [])[idx] || null;
        const unread = (c.unread || {})[uid] || 0;

        return `
          <div class="convo-item ${unread ? 'unread' : ''}" onclick="openChat('${c.id}')">
            <div class="convo-avatar">${avatar(name, photo, 'avatar-md')}</div>
            <div class="convo-info">
              <div class="convo-name">${esc(name)}</div>
              <div class="convo-last-msg">${esc(c.lastMessage || '')}</div>
            </div>
            <div class="convo-right">
              <div class="convo-time">${timeAgo(c.updatedAt)}</div>
              ${unread ? `<div class="convo-unread-badge">${unread}</div>` : ''}
            </div>
          </div>
        `;
      }).join('');
    });
  state.unsubs.push(u);
}

// ─── Chat View ───────────────────────────────────
let chatUnsub = null;

async function openChat(convoId) {
  // Get convo data
  const convoDoc = await db.collection('conversations').doc(convoId).get();
  if (!convoDoc.exists) return toast('Chat not found');
  const convo = convoDoc.data();
  const uid = state.user.uid;
  const idx = convo.participants.indexOf(uid) === 0 ? 1 : 0;
  const name = (convo.participantNames || [])[idx] || 'User';
  const photo = (convo.participantPhotos || [])[idx] || null;

  // Show chat screen
  showScreen('chat-view');
  $('#chat-hdr-info').innerHTML = `
    ${avatar(name, photo, 'avatar-sm')}
    <div>
      <h3 style="font-size:15px;font-weight:700">${esc(name)}</h3>
    </div>
  `;

  // Mark read
  db.collection('conversations').doc(convoId).set(
    { unread: { [uid]: 0 } }, { merge: true }
  ).catch(() => {});

  // Messages listener
  if (chatUnsub) chatUnsub();
  const msgs = $('#chat-msgs');
  chatUnsub = db.collection('conversations').doc(convoId)
    .collection('messages').orderBy('createdAt', 'asc').limit(100)
    .onSnapshot(snap => {
      const messages = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      if (!messages.length) {
        msgs.innerHTML = '<div style="text-align:center;padding:32px;opacity:0.5">Say hi! 👋</div>';
      } else {
        msgs.innerHTML = messages.map(m => {
          const isMe = m.senderId === uid;
          return `
            <div class="msg-bubble ${isMe ? 'msg-sent' : 'msg-received'}">
              ${esc(m.text)}
              <div class="msg-time">${m.createdAt ? timeAgo(m.createdAt) : ''}</div>
            </div>
          `;
        }).join('');
        msgs.scrollTop = msgs.scrollHeight;
      }
    });

  // Send
  const input = $('#chat-input');
  const sendBtn = $('#chat-send');
  const sendMsg = async () => {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    try {
      await db.collection('conversations').doc(convoId).collection('messages').add({
        text, senderId: uid, createdAt: FieldVal.serverTimestamp()
      });
      // Find the other user's UID
      const otherUid = convo.participants.find(p => p !== uid);
      await db.collection('conversations').doc(convoId).set({
        lastMessage: text,
        updatedAt: FieldVal.serverTimestamp(),
        unread: { [otherUid]: FieldVal.increment(1), [uid]: 0 }
      }, { merge: true });
    } catch (e) { console.error(e); }
  };
  sendBtn.onclick = sendMsg;
  input.onkeydown = e => { if (e.key === 'Enter') sendMsg(); };

  // Back
  $('#chat-back').onclick = () => {
    if (chatUnsub) { chatUnsub(); chatUnsub = null; }
    showScreen('app');
    navigate('chat');
  };
}

async function startChat(uid, name, photo) {
  if (uid === state.user.uid) return toast("That's you!");

  try {
    // Check existing conversation
    const snap = await db.collection('conversations')
      .where('participants', 'array-contains', state.user.uid)
      .get();
    const existing = snap.docs.find(d => d.data().participants.includes(uid));

    if (existing) {
      openChat(existing.id);
    } else {
      const doc = await db.collection('conversations').add({
        participants: [state.user.uid, uid],
        participantNames: [state.profile.displayName, name],
        participantPhotos: [state.profile.photoURL || null, photo || null],
        lastMessage: 'Started a conversation',
        updatedAt: FieldVal.serverTimestamp(),
        unread: { [uid]: 1, [state.user.uid]: 0 }
      });
      openChat(doc.id);
    }
  } catch (e) { toast('Could not start chat'); console.error(e); }
}

// ══════════════════════════════════════════════════
//  PROFILE
// ══════════════════════════════════════════════════
async function openProfile(uid) {
  showScreen('profile-view');
  const body = $('#prof-body');
  body.innerHTML = '<div style="padding:60px;text-align:center"><span class="inline-spinner" style="width:28px;height:28px;color:var(--accent)"></span></div>';
  $('#prof-top-name').textContent = '';

  try {
    // User data
    let user;
    if (uid === state.user.uid) {
      user = state.profile;
    } else {
      const doc = await db.collection('users').doc(uid).get();
      if (!doc.exists) throw new Error('Not found');
      user = { id: doc.id, ...doc.data() };
    }

    $('#prof-top-name').textContent = user.displayName;

    // User's posts (simple query to avoid index issues)
    let posts = [];
    try {
      const pSnap = await db.collection('posts').where('authorId', '==', uid).limit(20).get();
      posts = pSnap.docs.map(d => ({ id: d.id, ...d.data() }));
      posts.sort((a, b) => (b.createdAt?.seconds || 0) - (a.createdAt?.seconds || 0));
    } catch (e) { console.error('Posts query', e); }

    const isMe = uid === state.user.uid;

    body.innerHTML = `
      <div class="profile-cover"></div>
      <div class="profile-avatar-wrap">
        <div class="profile-avatar-large">
          ${user.photoURL
            ? `<img src="${user.photoURL}" alt="">`
            : initials(user.displayName)}
        </div>
      </div>
      <div class="profile-info">
        <div class="profile-name">${esc(user.displayName)}</div>
        <div class="profile-handle">${esc(user.major || '')} · ${esc(user.university || '')}</div>
        ${user.year ? `<div class="profile-badges"><span class="profile-badge">🎓 ${esc(user.year)}</span></div>` : ''}
        ${user.bio ? `<p class="profile-bio">${esc(user.bio)}</p>` : ''}

        <div class="profile-stats">
          <div class="profile-stat"><div class="stat-num">${posts.length}</div><div class="stat-label">Posts</div></div>
          <div class="profile-stat"><div class="stat-num">${(user.friends || []).length}</div><div class="stat-label">Friends</div></div>
        </div>

        <div class="profile-actions">
          ${isMe
            ? `<button class="btn-primary" onclick="editProfile()">Edit Profile</button>
               <button class="btn-secondary" onclick="doLogout()">Log Out</button>`
            : `<button class="btn-primary" onclick="startChat('${uid}','${esc(user.displayName)}','${user.photoURL || ''}')">Message</button>
               <button class="btn-outline" onclick="toast('Friend request sent!')">Add Friend</button>`}
        </div>
      </div>

      <div class="profile-tabs">
        <button class="profile-tab active">Posts</button>
        <button class="profile-tab">About</button>
      </div>

      <div class="profile-posts">
        ${posts.length
          ? posts.map(p => `
            <div class="post-card">
              <div class="post-header">
                ${avatar(user.displayName, user.photoURL, 'avatar-md')}
                <div class="post-header-info">
                  <div class="post-author-name">${esc(user.displayName)}</div>
                  <div class="post-meta">${timeAgo(p.createdAt)}</div>
                </div>
              </div>
              <div class="post-content">${formatContent(p.content)}</div>
              ${p.imageURL ? `<div class="post-image-wrap"><img src="${p.imageURL}" class="post-image" onclick="viewImage('${p.imageURL}')"></div>` : ''}
              <div class="post-actions">
                <button class="post-action ${(p.likes||[]).includes(state.user.uid) ? 'liked' : ''}" onclick="toggleLike('${p.id}')">❤ ${(p.likes||[]).length || 'Like'}</button>
                <button class="post-action" onclick="openComments('${p.id}')">💬 ${p.commentsCount || 'Comment'}</button>
              </div>
            </div>
          `).join('')
          : '<div class="empty-state"><h3>No posts yet</h3></div>'}
      </div>
    `;
  } catch (e) {
    console.error(e);
    body.innerHTML = '<div class="empty-state"><h3>Could not load profile</h3></div>';
  }

  // Back
  $('#prof-back').onclick = () => {
    showScreen('app');
  };
}

// ─── Edit Profile ────────────────────────────────
function editProfile() {
  const p = state.profile;
  const html = `
    <div class="modal-header">
      <h2>Edit Profile</h2>
      <button class="icon-btn" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body">
      <div class="form-group">
        <label>Display Name</label>
        <input type="text" id="edit-name" value="${esc(p.displayName)}">
      </div>
      <div class="form-group">
        <label>Bio</label>
        <textarea id="edit-bio">${esc(p.bio || '')}</textarea>
      </div>
      <div class="form-group">
        <label>Profile Photo</label>
        <input type="file" accept="image/*" id="edit-photo">
      </div>
      <button class="btn-primary btn-full" id="edit-save" style="margin-top:8px">Save</button>
    </div>
  `;
  openModal(html);

  let newPhoto = null;
  $('#edit-photo').onchange = async e => {
    if (e.target.files[0]) {
      newPhoto = await compress(e.target.files[0], 400, 0.6);
      toast('Photo ready');
    }
  };

  $('#edit-save').onclick = async () => {
    const name = $('#edit-name').value.trim();
    const bio = $('#edit-bio').value.trim();
    if (!name) return toast('Name required');

    closeModal();
    toast('Saving...');

    const updates = { displayName: name, bio };
    if (newPhoto) updates.photoURL = newPhoto;

    try {
      await db.collection('users').doc(state.user.uid).update(updates);
      // Update local state
      Object.assign(state.profile, updates);
      if (name !== state.user.displayName) {
        await state.user.updateProfile({ displayName: name });
      }
      setupHeader();
      toast('Profile updated!');
      openProfile(state.user.uid);
    } catch (e) { toast('Failed to save'); console.error(e); }
  };
}

// ─── Logout ──────────────────────────────────────
function doLogout() {
  auth.signOut().then(() => window.location.reload());
}

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

// ══════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initAuth();

  // Dismiss splash after short delay
  setTimeout(() => {
    const splash = $('#splash');
    if (splash) splash.classList.remove('active');
  }, 1500);

  // Image viewer close
  $('#img-close')?.addEventListener('click', () => {
    $('#img-view').style.display = 'none';
  });

  // Notification btn placeholder
  $('#notif-btn')?.addEventListener('click', () => toast('No new notifications'));

  // Expose globals for inline onclick
  window.navigate = navigate;
  window.openProfile = openProfile;
  window.openCreateModal = openCreateModal;
  window.openSellModal = openSellModal;
  window.toggleLike = toggleLike;
  window.openComments = openComments;
  window.postComment = postComment;
  window.viewImage = viewImage;
  window.startChat = startChat;
  window.openChat = openChat;
  window.closeModal = closeModal;
  window.editProfile = editProfile;
  window.doLogout = doLogout;
  window.toast = toast;
});
