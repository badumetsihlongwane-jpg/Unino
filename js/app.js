/* ══════════════════════════════════════════════════════
 *  UNINO — Campus Social Engine v4
 *  Firebase Auth + Firestore | base64 images
 *  Feed (Discover tabs), Explore (Radar/List + Modules),
 *  Marketplace, Messaging (fixed), Profiles (fixed)
 * ══════════════════════════════════════════════════════ */

// ─── State ───────────────────────────────────────
const state = { user: null, profile: null, page: 'feed', status: 'online', unsubs: [] };

// ─── Shortcuts ───────────────────────────────────
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
const FieldVal = firebase.firestore.FieldValue;
const COLORS = ['#6C5CE7','#8B5CF6','#A855F7','#7C3AED','#6366F1','#818CF8','#C084FC','#D946EF','#E879F9','#A78BFA'];

// ─── Helpers ─────────────────────────────────────
function colorFor(n) {
  let h = 0;
  for (let i = 0; i < (n || '').length; i++) h = n.charCodeAt(i) + ((h << 5) - h);
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

function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

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

function formatContent(text) {
  if (!text) return '';
  return esc(text).replace(/#(\w+)/g, '<span class="hashtag">#$1</span>');
}

// ─── Screen Manager ──────────────────────────────
function showScreen(id) {
  $$('.screen').forEach(s => s.classList.remove('active'));
  const el = document.getElementById(id);
  if (el) el.classList.add('active');
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
    if (!fname || !lname || !email || !pass || !uni || !major) return toast('All fields required');
    if (pass.length < 6) return toast('Password must be 6+ characters');
    btn.disabled = true; btn.innerHTML = '<span class="inline-spinner"></span>';
    try {
      const cred = await auth.createUserWithEmailAndPassword(email, pass);
      const uid = cred.user.uid;
      const displayName = `${fname} ${lname}`;
      await db.collection('users').doc(uid).set({
        displayName, firstName: fname, lastName: lname,
        email, university: uni, major, year, modules,
        bio: `${major} student at ${uni}`,
        photoURL: '', status: 'online',
        joinedAt: FieldVal.serverTimestamp(), friends: []
      });
      await cred.user.updateProfile({ displayName });
      db.collection('stats').doc('global').set({ totalUsers: FieldVal.increment(1) }, { merge: true }).catch(() => {});
    } catch (err) { toast(friendlyErr(err.code)); btn.disabled = false; btn.textContent = 'Create Account'; }
  });

  // AUTH STATE
  auth.onAuthStateChanged(async user => {
    if (user) {
      state.user = user;
      try {
        const doc = await db.collection('users').doc(user.uid).get();
        state.profile = doc.exists
          ? { id: doc.id, ...doc.data() }
          : { id: user.uid, displayName: user.displayName, email: user.email, status: 'online', modules: [] };
      } catch {
        state.profile = { id: user.uid, displayName: user.displayName, email: user.email, status: 'online', modules: [] };
      }
      state.status = state.profile.status || 'online';
      enterApp();
    } else {
      state.user = null; state.profile = null; unsub(); showScreen('auth-screen');
    }
  });

  db.collection('stats').doc('global').onSnapshot(doc => {
    const c = doc.exists ? (doc.data().totalUsers || 0) : 0;
    const el = $('#auth-count'); if (el) el.textContent = c;
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
  showScreen('app'); setupHeader(); setupNav(); setupStatusPill(); navigate('feed');
}

function setupHeader() {
  const el = $('#hdr-avatar');
  if (!el || !state.profile) return;
  const p = state.profile;
  if (p.photoURL) { el.innerHTML = `<img src="${p.photoURL}" alt="">`; el.style.background = 'transparent'; }
  else { el.textContent = initials(p.displayName); el.style.background = colorFor(p.displayName); }
  el.onclick = () => openProfile(state.user.uid);
  db.collection('stats').doc('global').onSnapshot(doc => {
    const hc = $('#hdr-count'); if (hc) hc.textContent = doc.exists ? (doc.data().totalUsers || 0) : 0;
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
  const pill = $('#status-pill'); if (!pill) return;
  updateStatusUI();
  pill.onclick = async () => {
    const modes = ['online','study','offline'];
    state.status = modes[(modes.indexOf(state.status) + 1) % 3];
    updateStatusUI();
    try { await db.collection('users').doc(state.user.uid).update({ status: state.status }); } catch (e) { console.error(e); }
    toast('Status: ' + state.status.charAt(0).toUpperCase() + state.status.slice(1));
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

// ─── Navigation ──────────────────────────────────
function navigate(page) {
  state.page = page; unsub();
  $$('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.p === page));
  switch (page) {
    case 'feed': renderFeed(); break;
    case 'explore': renderExplore(); break;
    case 'hustle': renderHustle(); break;
    case 'chat': renderMessages(); break;
  }
}

// ══════════════════════════════════════════════════
//  FEED — Clean with unified Discover tabs
// ══════════════════════════════════════════════════
function renderFeed() {
  const c = $('#content'), p = state.profile;
  c.innerHTML = `
    <div class="feed-page">
      <div class="welcome-banner">
        <div class="welcome-text">
          <h2>Hey, ${esc(p.firstName || p.displayName?.split(' ')[0])} 👋</h2>
          <p>${esc(p.university || 'Your Campus')}</p>
        </div>
        <div class="welcome-stat">
          <span class="dot green"></span> <span id="feed-online">0</span> online
        </div>
      </div>

      <div class="stories-row" id="stories-row">
        <div class="story-item add-story" onclick="openCreateModal()">
          <div class="story-avatar"><div class="story-avatar-inner">+</div></div>
          <div class="story-name">Post</div>
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
        <div class="prompt-actions"><span class="prompt-action">📷</span></div>
      </div>

      <div id="feed-posts">
        <div style="padding:40px;text-align:center"><span class="inline-spinner" style="width:28px;height:28px;color:var(--accent)"></span></div>
      </div>
    </div>
  `;

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
  loadOnlineFriends();

  // Live count
  db.collection('stats').doc('global').onSnapshot(doc => {
    const el = $('#feed-online');
    if (el) el.textContent = doc.exists ? (doc.data().totalUsers || 0) : 0;
  });

  // Real-time posts
  const u = db.collection('posts').orderBy('createdAt', 'desc').limit(50)
    .onSnapshot(snap => { renderPosts(snap.docs.map(d => ({ id: d.id, ...d.data() }))); });
  state.unsubs.push(u);
}

// ─── Discover: People tab ────────────────────────
function loadDiscoverPeople() {
  const el = $('#discover-content'); if (!el) return;
  const myUni = state.profile.university || '';
  const myMajor = state.profile.major || '';
  const myModules = state.profile.modules || [];

  db.collection('users').limit(30).get().then(snap => {
    let users = snap.docs.map(d => ({ id: d.id, ...d.data() })).filter(u => u.id !== state.user.uid);

    // Score & sort by relevance
    users = users.map(u => {
      let score = 0;
      const shared = (myModules).filter(m => (u.modules || []).includes(m));
      if (shared.length) score += 30 + shared.length * 10;
      if (u.university === myUni) score += 20;
      if (u.major === myMajor) score += 10;
      if (u.status === 'online') score += 5;
      return { ...u, score, sharedModules: shared };
    }).sort((a, b) => b.score - a.score).slice(0, 10);

    if (!users.length) {
      el.innerHTML = `<div class="discover-empty"><span>👥</span><p>No students found yet. Invite friends!</p></div>`;
      return;
    }

    el.innerHTML = `<div class="discover-scroll">${users.map(u => {
      const tag = u.sharedModules?.length
        ? `🔗 ${u.sharedModules.length} shared module${u.sharedModules.length > 1 ? 's' : ''}`
        : u.university === myUni ? '📍 Same campus'
        : u.university ? `🎓 ${esc(u.university)}` : '';
      const online = u.status === 'online' ? '<span class="online-dot"></span>' : '';
      return `
        <div class="discover-card" onclick="openProfile('${u.id}')">
          <div class="discover-card-avatar">
            ${avatar(u.displayName, u.photoURL, 'avatar-lg')}
            ${online}
          </div>
          <div class="discover-card-name">${esc(u.displayName)}</div>
          <div class="discover-card-meta">${esc(u.major || 'Student')}</div>
          ${tag ? `<div class="discover-card-tag">${tag}</div>` : ''}
          <button class="discover-card-btn" onclick="event.stopPropagation();startChat('${u.id}','${esc(u.displayName)}','${u.photoURL || ''}')">Message</button>
        </div>`;
    }).join('')}</div>`;
  }).catch(() => { el.innerHTML = '<div class="discover-empty"><p>Could not load</p></div>'; });
}

// ─── Discover: Events tab ────────────────────────
function loadDiscoverEvents() {
  const el = $('#discover-content'); if (!el) return;
  const events = [
    { title: 'Study Jam Session', emoji: '📚', when: 'Tomorrow, 6 PM', where: 'Library', gradient: 'linear-gradient(135deg,#6C5CE7,#A855F7)' },
    { title: 'Career Fair 2026', emoji: '💼', when: 'Feb 15', where: 'Main Hall', gradient: 'linear-gradient(135deg,#7C3AED,#C084FC)' },
    { title: 'Pool Tournament', emoji: '🎱', when: 'Fri, 4 PM', where: 'Student Center', gradient: 'linear-gradient(135deg,#8B5CF6,#D946EF)' },
    { title: 'Welcome Mixer', emoji: '🎉', when: 'Sat, 7 PM', where: 'Amphitheatre', gradient: 'linear-gradient(135deg,#6366F1,#818CF8)' },
    { title: 'Hackathon', emoji: '💻', when: 'Feb 20-21', where: 'CS Building', gradient: 'linear-gradient(135deg,#7C3AED,#A855F7)' },
    { title: 'Open Mic Night', emoji: '🎤', when: 'Next Wed', where: 'Quad', gradient: 'linear-gradient(135deg,#D946EF,#E879F9)' },
  ];
  el.innerHTML = `<div class="discover-scroll">${events.map(ev => `
    <div class="discover-card event-card" style="background:${ev.gradient}" onclick="toast('Event details coming soon!')">
      <div style="font-size:36px;margin-bottom:8px">${ev.emoji}</div>
      <div class="discover-card-name" style="color:#fff">${ev.title}</div>
      <div class="discover-card-meta" style="color:rgba(255,255,255,0.8)">${ev.when}</div>
      <div class="discover-card-tag" style="background:rgba(255,255,255,0.2);color:#fff">📍 ${ev.where}</div>
    </div>
  `).join('')}</div>`;
}

// ─── Online Friends → Stories Row ────────────────
function loadOnlineFriends() {
  const row = $('#stories-row'); if (!row) return;
  db.collection('users').where('status', '==', 'online').limit(15).get().then(snap => {
    const users = snap.docs.map(d => ({ id: d.id, ...d.data() })).filter(u => u.id !== state.user.uid);
    row.insertAdjacentHTML('beforeend', users.map(u => `
      <div class="story-item" onclick="openProfile('${u.id}')">
        <div class="story-avatar"><div class="story-avatar-inner">
          ${u.photoURL ? `<img src="${u.photoURL}" alt="">` : initials(u.displayName)}
        </div></div>
        <div class="story-name">${esc(u.firstName || u.displayName?.split(' ')[0] || '?')}</div>
      </div>
    `).join(''));
  }).catch(() => {});
}

// ─── Render Posts ────────────────────────────────
function renderPosts(posts) {
  const el = $('#feed-posts'); if (!el) return;
  if (!posts.length) {
    el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">📝</div><h3>No posts yet</h3><p>Be the first to share something!</p></div>`;
    return;
  }
  el.innerHTML = posts.map(post => {
    const liked = (post.likes || []).includes(state.user.uid);
    const lc = (post.likes || []).length, cc = post.commentsCount || 0;
    return `
      <div class="post-card">
        <div class="post-header">
          <div onclick="openProfile('${post.authorId}')" style="cursor:pointer">${avatar(post.authorName, post.authorPhoto, 'avatar-md')}</div>
          <div class="post-header-info">
            <div class="post-author-name" onclick="openProfile('${post.authorId}')">${esc(post.authorName)}</div>
            <div class="post-meta">${esc(post.authorUni || '')} · ${timeAgo(post.createdAt)}</div>
          </div>
        </div>
        <div class="post-content">${formatContent(post.content)}</div>
        ${post.imageURL ? `<div class="post-image-wrap"><img src="${post.imageURL}" class="post-image" loading="lazy" onclick="viewImage('${post.imageURL}')"></div>` : ''}
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
      </div>`;
  }).join('');
}

// ─── Like ────────────────────────────────────────
async function toggleLike(pid) {
  const ref = db.collection('posts').doc(pid);
  try {
    const doc = await ref.get(); if (!doc.exists) return;
    const likes = doc.data().likes || [];
    if (likes.includes(state.user.uid)) await ref.update({ likes: FieldVal.arrayRemove(state.user.uid) });
    else await ref.update({ likes: FieldVal.arrayUnion(state.user.uid) });
  } catch (e) { console.error(e); }
}

// ─── Comments ────────────────────────────────────
async function openComments(postId) {
  let comments = [];
  try {
    const snap = await db.collection('posts').doc(postId).collection('comments').orderBy('createdAt','asc').limit(50).get();
    comments = snap.docs.map(d => ({ id: d.id, ...d.data() }));
  } catch (e) { console.error(e); }
  openModal(`
    <div class="modal-header"><h2>Comments</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" style="display:flex;flex-direction:column;max-height:60vh;padding:0">
      <div id="comments-container" style="flex:1;overflow-y:auto;padding:16px">
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
      <div class="comment-input-wrap" style="position:sticky;bottom:0;background:var(--bg-secondary);padding:12px 16px;border-top:1px solid var(--border);flex-shrink:0">
        <input type="text" id="comment-input" placeholder="Write a comment...">
        <button onclick="postComment('${postId}')">Post</button>
      </div>
    </div>
  `);
  // scroll to bottom of comments
  const cc = $('#comments-container'); if (cc) cc.scrollTop = cc.scrollHeight;
}

async function postComment(postId) {
  const input = $('#comment-input'); const text = input?.value.trim(); if (!text) return;
  input.value = '';
  try {
    await db.collection('posts').doc(postId).collection('comments').add({
      text, authorId: state.user.uid, authorName: state.profile.displayName,
      authorPhoto: state.profile.photoURL || null, createdAt: FieldVal.serverTimestamp()
    });
    await db.collection('posts').doc(postId).update({ commentsCount: FieldVal.increment(1) });
    closeModal(); toast('Comment posted');
  } catch (e) { console.error(e); toast('Failed'); }
}

// ─── Image Viewer ────────────────────────────────
function viewImage(url) { const v = $('#img-view'); if (!v) return; $('#img-full').src = url; v.style.display = 'flex'; }

// ─── Create Post ─────────────────────────────────
function openCreateModal() {
  let pendingImg = null;
  openModal(`
    <div class="modal-header"><h2>Create Post</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
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
        <img src="" alt=""><button class="image-preview-remove" onclick="document.getElementById('create-preview').style.display='none'">&times;</button>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;border-top:1px solid var(--border);padding-top:12px;margin-top:12px">
        <label class="add-photo-btn"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg><input type="file" hidden accept="image/*" id="create-file"></label>
        <button class="btn-primary" id="create-submit" style="padding:10px 28px">Post</button>
      </div>
    </div>
  `);
  $('#create-file').onchange = async e => {
    if (e.target.files[0]) { pendingImg = await compress(e.target.files[0]); $('#create-preview img').src = pendingImg; $('#create-preview').style.display = 'block'; }
  };
  $('#create-submit').onclick = async () => {
    const text = $('#create-text').value.trim();
    if (!text && !pendingImg) return toast('Post cannot be empty');
    closeModal(); toast('Posting...');
    try {
      await db.collection('posts').add({
        content: text, imageURL: pendingImg || null,
        authorId: state.user.uid, authorName: state.profile.displayName,
        authorPhoto: state.profile.photoURL || null, authorUni: state.profile.university || '',
        createdAt: FieldVal.serverTimestamp(), likes: [], commentsCount: 0
      });
      toast('Posted!');
    } catch (e) { toast('Failed'); console.error(e); }
  };
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
    const snap = await db.collection('users').limit(50).get();
    const myUni = state.profile.university || '';
    const myMajor = state.profile.major || '';
    const myModules = state.profile.modules || [];

    allExploreUsers = snap.docs
      .map(d => ({ id: d.id, ...d.data() }))
      .filter(u => u.id !== state.user.uid)
      .map(u => {
        const uModules = u.modules || [];
        const shared = myModules.filter(m => uModules.includes(m));
        let proximity = 'far';
        if (shared.length > 0) proximity = 'module';
        else if (u.university === myUni && u.major === myMajor) proximity = 'course';
        else if (u.university === myUni) proximity = 'campus';
        return { ...u, sharedModules: shared, proximity };
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
  const moduleUsers = allExploreUsers.filter(u => u.proximity === 'module');
  const campusUsers = allExploreUsers.filter(u => u.proximity === 'campus' || u.proximity === 'course');
  const otherUsers = allExploreUsers.filter(u => u.proximity === 'far');

  body.innerHTML = `
    <div class="radar-container">
      <div class="radar-visual">
        <div class="radar-ring r3"></div>
        <div class="radar-ring r2"></div>
        <div class="radar-ring r1"></div>
        <div class="radar-center-dot">
          ${state.profile.photoURL ? `<img src="${state.profile.photoURL}" alt="">` : initials(state.profile.displayName)}
        </div>
        <div class="radar-sweep"></div>
        ${renderRadarDots(moduleUsers, 55, 'module')}
        ${renderRadarDots(campusUsers, 90, 'campus')}
        ${renderRadarDots(otherUsers.slice(0, 6), 120, 'far')}
      </div>
      <div class="radar-legend">
        <span><span class="legend-dot module"></span> Shared modules (${moduleUsers.length})</span>
        <span><span class="legend-dot campus"></span> Same campus (${campusUsers.length})</span>
        <span><span class="legend-dot far"></span> Other (${otherUsers.length})</span>
      </div>
    </div>

    ${moduleUsers.length ? `
    <div class="proximity-section">
      <div class="proximity-header"><h3>🔗 Shared Modules</h3><span class="proximity-count">${moduleUsers.length}</span></div>
      <div class="proximity-scroll">${moduleUsers.map(u => proximityCard(u)).join('')}</div>
    </div>` : ''}

    <div class="proximity-section">
      <div class="proximity-header"><h3>📍 Same Campus</h3><span class="proximity-count">${campusUsers.length}</span></div>
      <div class="proximity-scroll">
        ${campusUsers.length ? campusUsers.map(u => proximityCard(u)).join('')
          : '<p style="padding:12px;color:var(--text-tertiary);font-size:13px">No one found yet</p>'}
      </div>
    </div>

    ${otherUsers.length ? `
    <div class="proximity-section">
      <div class="proximity-header"><h3>🎓 Other Students</h3><span class="proximity-count">${otherUsers.length}</span></div>
      <div class="proximity-scroll">${otherUsers.slice(0, 12).map(u => proximityCard(u)).join('')}</div>
    </div>` : ''}
  `;
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
    : `🎓 ${esc(u.university || '')}`;
  return `
    <div class="proximity-card" onclick="openProfile('${u.id}')">
      <div class="proximity-card-avatar">${avatar(u.displayName, u.photoURL, 'avatar-md')}${online}</div>
      <div class="proximity-card-name">${esc(u.displayName)}</div>
      <div class="proximity-card-meta">${tag}</div>
    </div>`;
}

function renderListView() {
  const body = $('#explore-body'); if (!body) return;
  body.innerHTML = `
    <div style="padding:0 16px 16px">
      <div class="search-bar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <input type="text" id="explore-search" placeholder="Search by name, module, course...">
      </div>
      <div class="filter-chips">
        <span class="chip active" data-f="all">All</span>
        <span class="chip" data-f="campus">Same Campus</span>
        <span class="chip" data-f="module">Shared Modules</span>
        <span class="chip" data-f="course">Same Course</span>
      </div>
      <div class="users-grid" id="explore-grid"></div>
    </div>
  `;
  renderExploreGrid();
  let timer;
  $('#explore-search')?.addEventListener('input', e => {
    clearTimeout(timer); timer = setTimeout(() => renderExploreGrid(e.target.value), 300);
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
      (u.major || '').toLowerCase().includes(q) ||
      (u.university || '').toLowerCase().includes(q) ||
      (u.modules || []).some(m => m.toLowerCase().includes(q))
    );
  }
  if (filter === 'campus') users = users.filter(u => u.university === state.profile.university);
  else if (filter === 'module') users = users.filter(u => u.sharedModules?.length > 0);
  else if (filter === 'course') users = users.filter(u => u.major === state.profile.major);

  if (!users.length) {
    grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><h3>No matches</h3><p>Try different filters</p></div>';
    return;
  }

  grid.innerHTML = users.map(u => {
    const tag = u.sharedModules?.length ? `🔗 ${u.sharedModules.length} shared`
      : u.proximity === 'campus' || u.proximity === 'course' ? '📍 Same campus'
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
//  HUSTLE (Marketplace)
// ══════════════════════════════════════════════════
function renderHustle() {
  const c = $('#content');
  c.innerHTML = `
    <div class="hustle-page">
      <div class="hustle-header"><h2>Marketplace</h2><button class="btn-primary btn-sm" onclick="openSellModal()">+ Sell</button></div>
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
  $$('.category-tabs .chip').forEach(ch => {
    ch.onclick = () => {
      $$('.category-tabs .chip').forEach(c2 => c2.classList.remove('active'));
      ch.classList.add('active'); loadListings(ch.dataset.cat);
    };
  });
}

async function loadListings(cat = 'all') {
  const grid = $('#listings-grid'); if (!grid) return;
  try {
    const snap = await db.collection('listings').where('status', '==', 'active').limit(50).get();
    let items = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    if (cat !== 'all') items = items.filter(i => (i.category || '').toLowerCase() === cat);
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
      <div class="form-group"><label>Category</label><select id="sell-cat"><option>Books</option><option>Tech</option><option>Notes</option><option>Services</option><option>Other</option></select></div>
      <div class="form-group"><label>Photo</label><input type="file" accept="image/*" id="sell-file"></div>
      <div id="sell-preview" class="image-preview" style="display:none"><img src=""><button class="image-preview-remove" onclick="document.getElementById('sell-preview').style.display='none'">&times;</button></div>
      <button class="btn-primary btn-full" id="sell-submit">List Item</button>
    </div>
  `);
  $('#sell-file').onchange = async e => {
    if (e.target.files[0]) { pendingImg = await compress(e.target.files[0]); $('#sell-preview img').src = pendingImg; $('#sell-preview').style.display = 'block'; }
  };
  $('#sell-submit').onclick = async () => {
    const title = $('#sell-title').value.trim(), price = $('#sell-price').value.trim();
    if (!title || !price) return toast('Title and price required');
    closeModal(); toast('Listing...');
    try {
      await db.collection('listings').add({
        title, price, category: $('#sell-cat').value, imageURL: pendingImg || null,
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
  openModal(`
    <div class="modal-header"><h2>Product Details</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      ${item.imageURL ? `<div style="border-radius:var(--radius);overflow:hidden;margin-bottom:16px"><img src="${item.imageURL}" style="width:100%;max-height:280px;object-fit:cover;cursor:pointer" onclick="viewImage('${item.imageURL}')"></div>` : ''}
      <div style="font-size:24px;font-weight:800;color:var(--accent);margin-bottom:4px">R${esc(String(item.price))}</div>
      <div style="font-size:18px;font-weight:700;margin-bottom:12px">${esc(item.title)}</div>
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
        <button class="btn-primary" style="flex:1" onclick="closeModal();startChat('${item.sellerId}','${esc(item.sellerName)}','')">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          Contact Seller
        </button>
        <button class="btn-outline" style="flex:1" onclick="closeModal();openProfile('${item.sellerId}')">
          View Profile
        </button>
      </div>
    </div>
  `);
}

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

  // KEY FIX: No .orderBy() — sort client-side to avoid Firestore index requirement
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
        const name = (c.participantNames || [])[idx] || 'User';
        const photo = (c.participantPhotos || [])[idx] || null;
        const unread = (c.unread || {})[uid] || 0;
        return `
          <div class="convo-item ${unread ? 'unread' : ''}" onclick="openChat('${c.id}')">
            <div class="convo-avatar">${avatar(name, photo, 'avatar-md')}</div>
            <div class="convo-info">
              <div class="convo-name">${esc(name)}</div>
              <div class="convo-last-msg">${esc(c.lastMessage || 'Start chatting...')}</div>
            </div>
            <div class="convo-right">
              <div class="convo-time">${timeAgo(c.updatedAt)}</div>
              ${unread ? `<div class="convo-unread-badge">${unread}</div>` : ''}
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

// ─── Chat View ───────────────────────────────────
let chatUnsub = null;

async function openChat(convoId) {
  try {
    const convoDoc = await db.collection('conversations').doc(convoId).get();
    if (!convoDoc.exists) return toast('Chat not found');
    const convo = convoDoc.data();
    const uid = state.user.uid;
    const idx = convo.participants.indexOf(uid) === 0 ? 1 : 0;
    const name = (convo.participantNames || [])[idx] || 'User';
    const photo = (convo.participantPhotos || [])[idx] || null;

    showScreen('chat-view');
    $('#chat-hdr-info').innerHTML = `
      ${avatar(name, photo, 'avatar-sm')}
      <div><h3 style="font-size:15px;font-weight:700">${esc(name)}</h3></div>
    `;

    // Mark as read
    db.collection('conversations').doc(convoId).set({ unread: { [uid]: 0 } }, { merge: true }).catch(() => {});

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
            let content = '';
            if (m.imageURL) content += `<img src="${m.imageURL}" class="msg-image" onclick="viewImage('${m.imageURL}')">`;
            if (m.text) content += esc(m.text);
            return `<div class="msg-bubble ${isMe ? 'msg-sent' : 'msg-received'}">${content}<div class="msg-time">${m.createdAt ? timeAgo(m.createdAt) : ''}</div></div>`;
          }).join('');
          msgs.scrollTop = msgs.scrollHeight;
        }
      });

    // Send message + image
    const input = $('#chat-input');
    let chatPendingImg = null;

    const sendMsg = async () => {
      const text = input.value.trim();
      const img = chatPendingImg;
      if (!text && !img) return;
      input.value = ''; chatPendingImg = null;
      const preview = $('#chat-img-preview'); if (preview) preview.style.display = 'none';
      try {
        await db.collection('conversations').doc(convoId).collection('messages').add({
          text: text || '', imageURL: img || null, senderId: uid, createdAt: FieldVal.serverTimestamp()
        });
        const otherUid = convo.participants.find(p => p !== uid);
        await db.collection('conversations').doc(convoId).set({
          lastMessage: img ? (text || '📷 Photo') : text, updatedAt: FieldVal.serverTimestamp(),
          unread: { [otherUid]: FieldVal.increment(1), [uid]: 0 }
        }, { merge: true });
      } catch (e) { console.error(e); }
    };
    $('#chat-send').onclick = sendMsg;
    input.onkeydown = e => { if (e.key === 'Enter') sendMsg(); };

    // Wire image upload button in chat
    const chatFileInput = $('#chat-file-input');
    if (chatFileInput) {
      chatFileInput.onchange = async e => {
        if (e.target.files[0]) {
          chatPendingImg = await compress(e.target.files[0], 600, 0.6);
          const preview = $('#chat-img-preview');
          if (preview) { preview.querySelector('img').src = chatPendingImg; preview.style.display = 'block'; }
        }
      };
    }

    // Back button
    $('#chat-back').onclick = () => {
      if (chatUnsub) { chatUnsub(); chatUnsub = null; }
      showScreen('app'); navigate('chat');
    };
  } catch (e) { console.error(e); toast('Could not open chat'); }
}

async function startChat(uid, name, photo) {
  if (uid === state.user.uid) return toast("That's you!");
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
        unread: { [uid]: 0, [state.user.uid]: 0 }
      });
      openChat(doc.id);
    }
  } catch (e) { toast('Could not start chat'); console.error(e); }
}

// ══════════════════════════════════════════════════
//  PROFILE — Fixed avatar position (inside cover)
// ══════════════════════════════════════════════════
async function openProfile(uid) {
  showScreen('profile-view');
  const body = $('#prof-body');
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
    } catch (e) { console.error('Posts', e); }

    const isMe = uid === state.user.uid;
    const modules = user.modules || [];

    // KEY FIX: avatar-wrap is INSIDE profile-cover so position:absolute works relative to cover
    body.innerHTML = `
      <div class="profile-cover">
        <div class="profile-avatar-wrap">
          <div class="profile-avatar-large">
            ${user.photoURL ? `<img src="${user.photoURL}" alt="">` : initials(user.displayName)}
          </div>
        </div>
      </div>

      <div class="profile-info">
        <div class="profile-name">${esc(user.displayName)}</div>
        <div class="profile-handle">${esc(user.major || '')} · ${esc(user.university || '')}</div>
        ${user.year ? `<div class="profile-badges"><span class="profile-badge">🎓 ${esc(user.year)}</span></div>` : ''}
        ${user.bio ? `<p class="profile-bio">${esc(user.bio)}</p>` : ''}
        ${modules.length ? `<div class="profile-modules">${modules.map(m => `<span class="module-chip">${esc(m)}</span>`).join('')}</div>` : ''}

        <div class="profile-stats">
          <div class="profile-stat"><div class="stat-num">${posts.length}</div><div class="stat-label">Posts</div></div>
          <div class="profile-stat"><div class="stat-num">${(user.friends || []).length}</div><div class="stat-label">Friends</div></div>
          ${modules.length ? `<div class="profile-stat"><div class="stat-num">${modules.length}</div><div class="stat-label">Modules</div></div>` : ''}
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
        <button class="profile-tab active" data-pt="posts">Posts</button>
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
        tc.innerHTML = tab.dataset.pt === 'posts' ? renderProfilePosts(posts, user) : renderProfileAbout(user);
      };
    });
  } catch (e) {
    console.error(e);
    body.innerHTML = '<div class="empty-state"><h3>Could not load profile</h3></div>';
  }

  $('#prof-back').onclick = () => showScreen('app');
}

function renderProfilePosts(posts, user) {
  if (!posts.length) return '<div class="empty-state"><h3>No posts yet</h3></div>';
  const isMe = user.id === state.user.uid;
  return `<div class="profile-posts">${posts.map(p => `
    <div class="post-card">
      <div class="post-header">
        ${avatar(user.displayName, user.photoURL, 'avatar-md')}
        <div class="post-header-info">
          <div class="post-author-name">${esc(user.displayName)}</div>
          <div class="post-meta">${timeAgo(p.createdAt)}</div>
        </div>
        ${isMe ? `<button class="icon-btn" onclick="showPostOptions('${p.id}')" style="margin-left:auto">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/></svg>
        </button>` : ''}
      </div>
      <div class="post-content">${formatContent(p.content)}</div>
      ${p.imageURL ? `<div class="post-image-wrap"><img src="${p.imageURL}" class="post-image" onclick="viewImage('${p.imageURL}')"></div>` : ''}
      <div class="post-actions">
        <button class="post-action ${(p.likes||[]).includes(state.user.uid)?'liked':''}" onclick="toggleLike('${p.id}')">❤ ${(p.likes||[]).length||'Like'}</button>
        <button class="post-action" onclick="openComments('${p.id}')">💬 ${p.commentsCount||'Comment'}</button>
      </div>
    </div>
  `).join('')}</div>`;
}

// ─── iOS-style Post Options (Delete) ─────────────
function showPostOptions(postId) {
  openModal(`
    <div class="modal-body" style="padding:8px 0">
      <button class="ios-action-btn" onclick="confirmDeletePost('${postId}')">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>
        <span style="color:var(--red)">Delete Post</span>
      </button>
      <div style="height:1px;background:var(--border);margin:4px 16px"></div>
      <button class="ios-action-btn" onclick="closeModal()">
        <span>Cancel</span>
      </button>
    </div>
  `);
}

async function confirmDeletePost(postId) {
  closeModal();
  openModal(`
    <div class="modal-body" style="text-align:center;padding:24px">
      <h3 style="margin-bottom:8px">Delete this post?</h3>
      <p style="color:var(--text-secondary);font-size:14px;margin-bottom:20px">This can't be undone.</p>
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
    openProfile(state.user.uid);
  } catch (e) { toast('Failed to delete'); console.error(e); }
}

function renderProfileAbout(user) {
  const modules = user.modules || [];
  return `
    <div class="profile-about">
      <div class="about-item"><span class="about-icon">🎓</span><div><div class="about-label">University</div><div class="about-value">${esc(user.university || 'Not set')}</div></div></div>
      <div class="about-item"><span class="about-icon">📚</span><div><div class="about-label">Major</div><div class="about-value">${esc(user.major || 'Not set')}</div></div></div>
      <div class="about-item"><span class="about-icon">📅</span><div><div class="about-label">Year</div><div class="about-value">${esc(user.year || 'Not set')}</div></div></div>
      ${modules.length ? `<div class="about-item"><span class="about-icon">🧩</span><div><div class="about-label">Modules</div><div class="about-modules">${modules.map(m => `<span class="module-chip">${esc(m)}</span>`).join('')}</div></div></div>` : ''}
      <div class="about-item"><span class="about-icon">📧</span><div><div class="about-label">Email</div><div class="about-value">${esc(user.email || 'Private')}</div></div></div>
      ${user.joinedAt ? `<div class="about-item"><span class="about-icon">🗓</span><div><div class="about-label">Joined</div><div class="about-value">${timeAgo(user.joinedAt)}</div></div></div>` : ''}
    </div>`;
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
      <div class="form-group"><label>Modules (comma-separated)</label><input type="text" id="edit-modules" value="${esc(mods)}" placeholder="MAT101, COS132, PHY121"></div>
      <div class="form-group"><label>Profile Photo</label><input type="file" accept="image/*" id="edit-photo"></div>
      <button class="btn-primary btn-full" id="edit-save">Save</button>
    </div>
  `);
  let newPhoto = null;
  $('#edit-photo').onchange = async e => {
    if (e.target.files[0]) { newPhoto = await compress(e.target.files[0], 400, 0.6); toast('Photo ready'); }
  };
  $('#edit-save').onclick = async () => {
    const name = $('#edit-name').value.trim();
    const bio = $('#edit-bio').value.trim();
    const modulesRaw = $('#edit-modules').value || '';
    const modules = modulesRaw.split(',').map(m => m.trim().toUpperCase()).filter(Boolean);
    if (!name) return toast('Name required');
    closeModal(); toast('Saving...');
    const updates = { displayName: name, bio, modules };
    if (newPhoto) updates.photoURL = newPhoto;
    try {
      await db.collection('users').doc(state.user.uid).update(updates);
      Object.assign(state.profile, updates);
      if (name !== state.user.displayName) await state.user.updateProfile({ displayName: name });
      setupHeader(); toast('Profile updated!'); openProfile(state.user.uid);
    } catch (e) { toast('Failed'); console.error(e); }
  };
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

  // Notifications placeholder
  $('#notif-btn')?.addEventListener('click', () => toast('No new notifications'));

  // Expose globals for inline onclick
  Object.assign(window, {
    navigate, openProfile, openCreateModal, openSellModal,
    toggleLike, openComments, postComment, viewImage,
    startChat, openChat, closeModal, editProfile, doLogout, toast,
    showPostOptions, confirmDeletePost, deletePost, openProductDetail
  });
});
