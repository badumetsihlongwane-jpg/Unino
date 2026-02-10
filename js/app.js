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
          <p>${esc(p.university || 'NWU Campus')}</p>
        </div>
        <div class="welcome-stat">
          <span class="dot green"></span> <span id="feed-online">0</span> online
        </div>
      </div>

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
  loadStories();

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
      snap.docs.forEach(d => {
        const s = { id: d.id, ...d.data() };
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
          <div class="story-item ${hasNew ? 'has-unseen' : 'seen'}" onclick="viewStory('${group.uid}')">
            <div class="story-avatar"><div class="story-avatar-inner">
              ${s.authorPhoto ? `<img src="${s.authorPhoto}" alt="">` : initials(s.authorName)}
            </div></div>
            <div class="story-name">${name}</div>
          </div>
        `);
      });

      // Also load online users who don't have stories
      loadOnlineFriends(Object.keys(byUser).concat(ordered.map(o => o.uid)));
    }).catch(() => loadOnlineFriends([]));
}

function loadOnlineFriends(excludeIds = []) {
  const row = $('#stories-row'); if (!row) return;
  db.collection('users').where('status', '==', 'online').limit(15).get().then(snap => {
    const users = snap.docs.map(d => ({ id: d.id, ...d.data() }))
      .filter(u => u.id !== state.user.uid && !excludeIds.includes(u.id));
    row.insertAdjacentHTML('beforeend', users.map(u => `
      <div class="story-item no-story" onclick="openProfile('${u.id}')">
        <div class="story-avatar"><div class="story-avatar-inner">
          ${u.photoURL ? `<img src="${u.photoURL}" alt="">` : initials(u.displayName)}
        </div></div>
        <div class="story-name">${esc(u.firstName || u.displayName?.split(' ')[0] || '?')}</div>
      </div>
    `).join(''));
  }).catch(() => {});
}

function openStoryCreator() {
  let pendingImg = null;
  let bgColor = '#6C5CE7';
  const bgOptions = ['#6C5CE7','#A855F7','#7C3AED','#D946EF','#FF6B6B','#00BA88','#3B82F6','#FF9F43'];
  openModal(`
    <div class="modal-header"><h2>Create Story</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      <div class="story-creator">
        <div class="story-type-tabs">
          <button class="story-type-tab active" data-st="text">Text</button>
          <button class="story-type-tab" data-st="photo">Photo</button>
        </div>
        <div id="story-text-creator" class="story-type-content active">
          <div class="story-text-preview" id="story-text-bg" style="background:${bgColor}">
            <textarea id="story-text-input" placeholder="Type your story..." maxlength="200"></textarea>
          </div>
          <div class="story-bg-picker">${bgOptions.map(c => `<button class="bg-dot" style="background:${c}" onclick="document.getElementById('story-text-bg').style.background='${c}';window._storyBg='${c}'"></button>`).join('')}</div>
        </div>
        <div id="story-photo-creator" class="story-type-content">
          <div id="story-photo-preview" class="story-photo-drop">
            <label style="cursor:pointer;text-align:center">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
              <p style="color:var(--text-secondary);font-size:13px;margin-top:8px">Tap to add photo</p>
              <input type="file" hidden accept="image/*" id="story-photo-file">
            </label>
          </div>
          <input type="text" id="story-photo-caption" placeholder="Add a caption..." style="margin-top:12px">
        </div>
        <button class="btn-primary btn-full" id="story-submit" style="margin-top:16px">Share Story</button>
      </div>
    </div>
  `);
  window._storyBg = bgColor;
  $$('.story-type-tab').forEach(tab => {
    tab.onclick = () => {
      $$('.story-type-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      $$('.story-type-content').forEach(c => c.classList.remove('active'));
      const target = tab.dataset.st === 'text' ? '#story-text-creator' : '#story-photo-creator';
      $(target)?.classList.add('active');
    };
  });
  $('#story-photo-file').onchange = async e => {
    if (e.target.files[0]) {
      pendingImg = await compress(e.target.files[0], 600, 0.65);
      const prev = $('#story-photo-preview');
      prev.innerHTML = `<img src="${pendingImg}" style="width:100%;height:100%;object-fit:cover;border-radius:var(--radius)">`;
    }
  };
  $('#story-submit').onclick = async () => {
    const activeTab = document.querySelector('.story-type-tab.active')?.dataset.st;
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
    if (activeTab === 'text') {
      const text = $('#story-text-input')?.value.trim();
      if (!text) return toast('Type something!');
      storyData.type = 'text';
      storyData.text = text;
      storyData.bgColor = window._storyBg || '#6C5CE7';
    } else {
      if (!pendingImg) return toast('Add a photo!');
      storyData.type = 'photo';
      storyData.imageURL = pendingImg;
      storyData.caption = $('#story-photo-caption')?.value.trim() || '';
    }
    closeModal(); toast('Posting story...');
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
  hdr.innerHTML = `
    ${avatar(story.authorName, story.authorPhoto, 'avatar-sm')}
    <div><b>${esc(story.authorName)}</b><br><small>${timeAgo(story.createdAt)}</small></div>
  `;

  // Content
  const content = $('#story-viewer-content');
  if (story.type === 'photo') {
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
  storyViewerData.timer = setTimeout(() => advanceStory(1), 5000);

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
          <div style="font-size:12px;color:var(--text-secondary)">Posting to ${esc(state.profile.university || 'NWU')}</div>
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
      <button class="btn-primary btn-full" id="grp-create-btn">Create Group</button>
    </div>
  `);
  $('#grp-create-btn').onclick = async () => {
    const name = $('#grp-name')?.value.trim();
    const desc = $('#grp-desc')?.value.trim() || '';
    const type = $('#grp-type')?.value || 'study';
    if (!name) return toast('Name required');
    closeModal(); toast('Creating group...');
    try {
      await db.collection('groups').add({
        name, description: desc, type,
        createdBy: state.user.uid,
        members: [state.user.uid],
        memberNames: { [state.user.uid]: state.profile.displayName },
        memberPhotos: { [state.user.uid]: state.profile.photoURL || '' },
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
    const group = gDoc.data();
    const uid = state.user.uid;

    const gName = group.name || group.assignmentTitle || 'Group';
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
    const msgs = $('#gchat-msgs');
    gchatUnsub = db.collection(collection).doc(groupId)
      .collection('messages').orderBy('createdAt','asc').limit(100)
      .onSnapshot(snap => {
        const messages = snap.docs.map(d => ({ id: d.id, ...d.data() }));
        if (!messages.length) {
          msgs.innerHTML = '<div style="text-align:center;padding:32px;opacity:0.5">Start the conversation! 💬</div>';
        } else {
          msgs.innerHTML = messages.map(m => {
            const isMe = m.senderId === uid;
            return `<div class="msg-bubble ${isMe ? 'msg-sent' : 'msg-received'}">
              ${!isMe ? `<div class="gchat-sender">${esc(m.senderName?.split(' ')[0] || '?')}</div>` : ''}
              ${m.imageURL ? `<img src="${m.imageURL}" class="msg-image" onclick="viewImage('${m.imageURL}')">` : ''}
              ${m.text ? esc(m.text) : ''}
              <div class="msg-time">${m.createdAt ? timeAgo(m.createdAt) : ''}</div>
            </div>`;
          }).join('');
          msgs.scrollTop = msgs.scrollHeight;
        }
      });

    const sendGMsg = async () => {
      const input = $('#gchat-input');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      try {
        await db.collection(collection).doc(groupId).collection('messages').add({
          text, senderId: uid, senderName: state.profile.displayName,
          senderPhoto: state.profile.photoURL || null,
          createdAt: FieldVal.serverTimestamp()
        });
        await db.collection(collection).doc(groupId).update({
          lastMessage: text, updatedAt: FieldVal.serverTimestamp()
        });
      } catch (e) { console.error(e); }
    };
    $('#gchat-send').onclick = sendGMsg;
    $('#gchat-input').onkeydown = e => { if (e.key === 'Enter') sendGMsg(); };
    $('#gchat-back').onclick = () => {
      if (gchatUnsub) { gchatUnsub(); gchatUnsub = null; }
      showScreen('app'); navigate('chat');
    };
  } catch (e) { console.error(e); toast('Could not open group'); }
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
        <button class="msg-tab active" data-mt="dm">Direct</button>
        <button class="msg-tab" data-mt="groups">Groups</button>
        <button class="msg-tab" data-mt="assignments">Assignments</button>
      </div>
      <div id="msg-tab-content">
        <div class="convo-list" id="convo-list">
          <div style="padding:40px;text-align:center"><span class="inline-spinner"></span></div>
        </div>
      </div>
    </div>
  `;
  $$('.msg-tab').forEach(tab => {
    tab.onclick = () => {
      $$('.msg-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      if (tab.dataset.mt === 'dm') loadDMList();
      else if (tab.dataset.mt === 'groups') loadGroupList();
      else loadAssignmentGroups();
    };
  });
  loadDMList();
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
//  ASSIGNMENT GROUPS — Intent-Based, Temporary
// ══════════════════════════════════════════════════
function loadAssignmentGroups() {
  const container = $('#msg-tab-content'); if (!container) return;
  const myModules = state.profile.modules || [];
  container.innerHTML = `
    <div class="asg-page">
      <div style="padding:12px 16px;display:flex;gap:8px">
        <button class="btn-primary" style="flex:1" onclick="openCreateAssignmentGroup()">+ New Assignment Group</button>
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
      el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">📋</div><h3>No assignment groups</h3><p>${filter === 'my' ? 'None for your modules yet' : 'Create one to get started!'}</p></div>`;
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

      return `
        <div class="asg-card ${isMember ? 'is-member' : ''} ${g.status === 'archived' ? 'is-archived' : ''}" onclick="openAssignmentDetail('${g.id}')">
          <div class="asg-card-top">
            <div class="asg-card-module">${esc(g.moduleCode || '???')}</div>
            ${statusBadge}
          </div>
          <div class="asg-card-title">${esc(g.assignmentTitle)}</div>
          <div class="asg-card-meta">
            <span>👥 ${(g.members||[]).length}/${g.maxSize||10}</span>
            <span>·</span>
            <span>${g.joinMode === 'open' ? '🔓 Open' : g.joinMode === 'invite' ? '🔒 Invite' : '🤖 Auto-fill'}</span>
            ${g.dueDate ? `<span>· 📅 ${esc(g.dueDate)}</span>` : ''}
          </div>
          <div class="asg-card-members">
            ${(g.members||[]).slice(0,5).map(mid => {
              const mName = (g.memberNames||{})[mid] || '?';
              return avatar(mName, (g.memberPhotos||{})[mid] || null, 'avatar-sm');
            }).join('')}
            ${(g.members||[]).length > 5 ? `<span class="asg-more">+${(g.members||[]).length - 5}</span>` : ''}
          </div>
          ${hasConflict && isMember ? '<div class="asg-conflict">⚠️ 1 person you preferred not to work with is in this group</div>' : ''}
          <div class="asg-card-host">Created by ${esc((g.memberNames||{})[g.createdBy] || 'Someone')} · ${timeAgo(g.createdAt)}</div>
        </div>`;
    }).join('');
  } catch (e) { console.error(e); el.innerHTML = '<div class="empty-state"><h3>Could not load</h3></div>'; }
}

function openCreateAssignmentGroup() {
  const myModules = state.profile.modules || [];
  const moduleOptions = myModules.length
    ? myModules.map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join('')
    : '<option value="">Add modules in your profile first</option>';

  openModal(`
    <div class="modal-header"><h2>New Assignment Group</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      <div class="form-group"><label>Module</label>
        <select id="asg-module">${moduleOptions}<option value="_custom">Other (type below)</option></select>
      </div>
      <div class="form-group" id="asg-custom-wrap" style="display:none"><label>Module Code</label><input type="text" id="asg-custom-module" placeholder="e.g. BIO214"></div>
      <div class="form-group"><label>Assignment Title</label><input type="text" id="asg-title" placeholder="e.g. Genetics Project"></div>
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
    if (!moduleCode || !title) return toast('Module and title required');
    closeModal(); toast('Creating assignment group...');
    const uid = state.user.uid;
    try {
      const doc = await db.collection('assignmentGroups').add({
        moduleCode, assignmentTitle: title, maxSize, dueDate, joinMode, visibility,
        createdBy: uid, status: 'open', locked: false,
        members: [uid],
        memberNames: { [uid]: state.profile.displayName },
        memberPhotos: { [uid]: state.profile.photoURL || '' },
        pendingRequests: [],
        preferences: {},
        lastMessage: '', updatedAt: FieldVal.serverTimestamp(),
        createdAt: FieldVal.serverTimestamp()
      });
      toast('Assignment group created!');
      openAssignmentDetail(doc.id);
    } catch (e) { toast('Failed'); console.error(e); }
  };
}

async function openAssignmentDetail(groupId) {
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

    let membersHtml = (g.members || []).map(mid => {
      const mName = (g.memberNames||{})[mid] || 'Unknown';
      const mPhoto = (g.memberPhotos||{})[mid] || null;
      const isCreator = mid === g.createdBy;
      const warnMe = (myPrefs.dontWant || []).includes(mid) && mid !== uid;
      return `
        <div class="asg-member ${warnMe ? 'conflict' : ''}">
          ${avatar(mName, mPhoto, 'avatar-md')}
          <div class="asg-member-info">
            <div class="asg-member-name">${esc(mName)} ${isCreator ? '<span class="asg-host-tag">Host</span>' : ''}</div>
            ${warnMe ? '<div class="asg-member-warn">⚠️ Preference conflict</div>' : ''}
          </div>
          ${isHost && mid !== uid && !isLocked ? `<button class="btn-sm btn-ghost" onclick="event.stopPropagation();removeFromAsg('${groupId}','${mid}')">Remove</button>` : ''}
        </div>`;
    }).join('');

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
        <button class="btn-primary btn-full" onclick="openAsgChat('${groupId}')">💬 Open Group Chat</button>
        <div style="display:flex;gap:8px;margin-top:8px">
          <button class="btn-outline" style="flex:1" onclick="openAsgPreferences('${groupId}')">⚙️ Preferences</button>
          ${isHost ? `<button class="btn-outline" style="flex:1" onclick="toggleAsgLock('${groupId}', ${!isLocked})">${isLocked ? '🔓 Unlock' : '🔒 Lock Group'}</button>` : ''}
        </div>
        ${isHost ? `<div style="display:flex;gap:8px;margin-top:8px">
          ${!isLocked && spotsLeft > 0 && g.joinMode === 'auto' ? `<button class="btn-secondary" style="flex:1" onclick="autoFillAsg('${groupId}')">🤖 Auto-fill Spots</button>` : ''}
          <button class="btn-danger" style="flex:1;border-radius:var(--radius)" onclick="archiveAsg('${groupId}')">📦 Archive</button>
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
        <div class="asg-detail-title">${esc(g.assignmentTitle)}</div>
        <div class="asg-detail-meta">
          <span>👥 ${(g.members||[]).length}/${g.maxSize||10}</span>
          <span>${g.joinMode === 'open' ? '🔓 Open' : g.joinMode === 'invite' ? '🔒 Invite' : '🤖 Auto-fill'}</span>
          ${g.dueDate ? `<span>📅 Due: ${esc(g.dueDate)}</span>` : ''}
          <span>${g.visibility === 'friends' ? '👫 Friends only' : '🌍 Public'}</span>
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
    openAssignmentDetail(groupId);
  } catch (e) { toast('Failed'); console.error(e); }
}

async function requestJoinAsg(groupId) {
  const uid = state.user.uid;
  try {
    await db.collection('assignmentGroups').doc(groupId).update({
      pendingRequests: FieldVal.arrayUnion({ uid, name: state.profile.displayName, photo: state.profile.photoURL || '' })
    });
    closeModal(); toast('Request sent! The host will review it.');
  } catch (e) { toast('Failed'); console.error(e); }
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
    closeModal(); toast(`${reqName} approved!`);
    openAssignmentDetail(groupId);
  } catch (e) { toast('Failed'); console.error(e); }
}

async function rejectAsgRequest(groupId, reqUid) {
  try {
    const gDoc = await db.collection('assignmentGroups').doc(groupId).get();
    const g = gDoc.data();
    const newPending = (g.pendingRequests||[]).filter(r => r.uid !== reqUid);
    await db.collection('assignmentGroups').doc(groupId).update({ pendingRequests: newPending });
    closeModal(); toast('Request declined');
    openAssignmentDetail(groupId);
  } catch (e) { toast('Failed'); console.error(e); }
}

async function removeFromAsg(groupId, memberUid) {
  try {
    await db.collection('assignmentGroups').doc(groupId).update({
      members: FieldVal.arrayRemove(memberUid)
    });
    closeModal(); toast('Removed');
    openAssignmentDetail(groupId);
  } catch (e) { toast('Failed'); console.error(e); }
}

async function leaveAsg(groupId) {
  const uid = state.user.uid;
  try {
    await db.collection('assignmentGroups').doc(groupId).update({
      members: FieldVal.arrayRemove(uid)
    });
    closeModal(); toast('Left group');
    loadAssignmentGroups();
  } catch (e) { toast('Failed'); console.error(e); }
}

async function toggleAsgLock(groupId, lock) {
  try {
    await db.collection('assignmentGroups').doc(groupId).update({ locked: lock });
    closeModal(); toast(lock ? 'Group locked' : 'Group unlocked');
    openAssignmentDetail(groupId);
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
    loadAssignmentGroups();
  } catch (e) { toast('Failed'); console.error(e); }
}

async function autoFillAsg(groupId) {
  try {
    const gDoc = await db.collection('assignmentGroups').doc(groupId).get();
    const g = gDoc.data();
    const spotsLeft = (g.maxSize || 10) - (g.members || []).length;
    if (spotsLeft <= 0) return toast('Group is already full');
    // Find students in same module who aren't in any group for this assignment
    const allSnap = await db.collection('users').where('modules', 'array-contains', g.moduleCode).limit(50).get();
    const candidates = allSnap.docs
      .map(d => ({ id: d.id, ...d.data() }))
      .filter(u => !g.members.includes(u.id) && u.id !== state.user.uid);
    // Respect preferences: exclude people the host marked as "don't want"
    const hostPrefs = (g.preferences || {})[g.createdBy] || {};
    const dontWant = hostPrefs.dontWant || [];
    const filtered = candidates.filter(c => !dontWant.includes(c.id));
    const toAdd = filtered.slice(0, spotsLeft);
    if (!toAdd.length) return toast('No matching students found to auto-fill');
    const updates = { members: g.members };
    toAdd.forEach(u => {
      updates.members.push(u.id);
      updates[`memberNames.${u.id}`] = u.displayName;
      updates[`memberPhotos.${u.id}`] = u.photoURL || '';
    });
    await db.collection('assignmentGroups').doc(groupId).update(updates);
    closeModal(); toast(`Added ${toAdd.length} student${toAdd.length > 1 ? 's' : ''}!`);
    openAssignmentDetail(groupId);
  } catch (e) { toast('Auto-fill failed'); console.error(e); }
}

function openAsgPreferences(groupId) {
  openModal(`
    <div class="modal-header"><h2>Preferences</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      <p style="color:var(--text-secondary);font-size:13px;margin-bottom:16px">Set soft preferences for group matching. These are private and only used by the auto-fill system.</p>
      <div class="form-group">
        <label>Prefer to work with (names or student IDs, comma-separated)</label>
        <input type="text" id="pref-want" placeholder="e.g. John, Sarah">
      </div>
      <div class="form-group">
        <label>Prefer NOT to work with</label>
        <input type="text" id="pref-dontwant" placeholder="e.g. someone you had a conflict with">
      </div>
      <button class="btn-primary btn-full" id="pref-save">Save Preferences</button>
      <p style="color:var(--text-tertiary);font-size:11px;margin-top:8px;text-align:center">Preferences are soft — used for auto-fill matching, not guaranteed.</p>
    </div>
  `);
  $('#pref-save').onclick = async () => {
    const want = ($('#pref-want')?.value || '').split(',').map(s => s.trim()).filter(Boolean);
    const dontWant = ($('#pref-dontwant')?.value || '').split(',').map(s => s.trim()).filter(Boolean);
    try {
      await db.collection('assignmentGroups').doc(groupId).update({
        [`preferences.${state.user.uid}`]: { want, dontWant }
      });
      closeModal(); toast('Preferences saved');
    } catch (e) { toast('Failed'); console.error(e); }
  };
}

function openAsgChat(groupId) {
  // Reuses the group chat system
  openGroupChat(groupId, 'assignmentGroups');
}

// ─── DM List ─────────────────────────────────────
function loadDMList() {
  const container = $('#msg-tab-content'); if (!container) return;
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
    showPostOptions, confirmDeletePost, deletePost, openProductDetail,
    openStoryCreator, viewStory, closeStoryViewer, advanceStory,
    openCreateGroup, openGroupChat, joinGroup, loadStories,
    openCreateAssignmentGroup, openAssignmentDetail, joinAsg, requestJoinAsg,
    approveAsgRequest, rejectAsgRequest, removeFromAsg, leaveAsg,
    toggleAsgLock, archiveAsg, doArchiveAsg, autoFillAsg,
    openAsgPreferences, openAsgChat, loadAssignmentGroups
  });
});
