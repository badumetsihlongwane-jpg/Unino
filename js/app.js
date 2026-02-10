/* ══════════════════════════════════════════════════════
 *  UNINO — Complete Application Engine
 *  Firebase Auth + Firestore (No Storage needed)
 * ══════════════════════════════════════════════════════ */

// ─── State ────────────────────────────────────────────
const state = {
  user: null,           // Firebase User
  profile: null,        // Firestore User Doc
  currentPage: 'feed',
  posts: [],
  listings: [],
  conversations: [],
  status: 'online',     // online | study | offline
  unsubscribers: [],    // Listeners to clean up
  usersCache: []
};

// ─── Constants ────────────────────────────────────────
const AVATAR_COLORS = [
  '#6C5CE7', '#3B82F6', '#10B981', '#F59E0B', '#EF4444',
  '#EC4899', '#8B5CF6', '#06B6D4', '#F97316', '#14B8A6'
];

// ─── Helpers ──────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function getAvatarColor(name) {
  let hash = 0;
  for (let i = 0; i < (name || '').length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}

function getInitials(name) {
  if (!name) return '?';
  const parts = name.trim().split(/\s+/);
  return (parts[0][0] + (parts[1] ? parts[1][0] : '')).toUpperCase().substring(0, 2);
}

function avatarHTML(name, photoURL, sizeClass = 'avatar-sm', status = null) {
  const color = getAvatarColor(name);
  let html = '';
  
  if (photoURL) {
    html = `<div class="${sizeClass} avatar-img-wrap"><img src="${photoURL}" alt="${name}" onerror="this.onerror=null;this.parentElement.innerHTML='${getInitials(name)}';this.parentElement.style.background='${color}'"></div>`;
  } else {
    html = `<div class="${sizeClass}" style="background:${color}">${getInitials(name)}</div>`;
  }

  // Wrap in container for status dot if needed
  if (status && status !== 'offline') {
    const colorClass = status === 'study' ? 'orange' : 'green';
    return `<div class="avatar-with-status ${sizeClass}-wrapper">
      ${html}
      <span class="status-dot ${colorClass} status-${sizeClass}"></span>
    </div>`;
  }
  return html;
}

function timeAgo(timestamp) {
  if (!timestamp) return '';
  const date = timestamp.toDate ? timestamp.toDate() : new Date(timestamp);
  const diff = Date.now() - date.getTime();
  const mins = Math.floor(diff / 60000);
  
  if (mins < 1) return 'Just now';
  if (mins < 60) return mins + 'm';
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return hrs + 'h';
  const days = Math.floor(hrs / 24);
  if (days < 7) return days + 'd';
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function formatTime(timestamp) {
  if (!timestamp) return '';
  const date = timestamp.toDate ? timestamp.toDate() : new Date(timestamp);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function escapeHTML(str) {
  const div = document.createElement('div');
  div.textContent = str || '';
  return div.innerHTML;
}

function showToast(msg) {
  const toast = $('#toast');
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2500);
}

function compressImage(file, maxWidth = 800, quality = 0.7) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        let width = img.width;
        let height = img.height;
        
        if (width > maxWidth) {
          height = height * (maxWidth / width);
          width = maxWidth;
        }
        
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);
        resolve(canvas.toDataURL('image/jpeg', quality));
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

// ─── Screen Manager ───────────────────────────────────
function showScreen(id) {
  $$('.screen').forEach(s => s.classList.remove('active'));
  const el = document.getElementById(id);
  if (el) {
    el.classList.add('active');
    window.scrollTo(0, 0);
  }
}

// ─── Theme Toggle ─────────────────────────────────────
function initTheme() {
  const saved = localStorage.getItem('unino-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  
  $('#theme-toggle')?.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('unino-theme', next);
  });
}

// ─── Auth System ──────────────────────────────────────
function initAuth() {
  // Toggle Forms
  $('#show-signup-link')?.addEventListener('click', (e) => {
    e.preventDefault();
    $('#login-form').classList.remove('active');
    $('#signup-form').classList.add('active');
  });
  
  $('#show-login-link')?.addEventListener('click', (e) => {
    e.preventDefault();
    $('#signup-form').classList.remove('active');
    $('#login-form').classList.add('active');
  });

  // Login
  $('#login-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = $('#login-btn');
    const email = $('#login-email').value.trim();
    const pass = $('#login-password').value;
    
    if (!email || !pass) return showToast('Please enter both email and password');
    
    btn.disabled = true;
    btn.innerHTML = '<span class="inline-spinner"></span>';
    
    try {
      await auth.signInWithEmailAndPassword(email, pass);
    } catch (err) {
      showToast(friendlyError(err.code));
      btn.disabled = false;
      btn.textContent = 'Log In';
    }
  });

  // Signup
  $('#signup-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = $('#signup-btn');
    const fname = $('#signup-fname').value.trim();
    const lname = $('#signup-lname').value.trim();
    const email = $('#signup-email').value.trim();
    const pass = $('#signup-password').value;
    const uni = $('#signup-university').value;
    const major = $('#signup-major').value;

    if (!fname || !lname || !email || !pass || !uni || !major) {
      return showToast('Full details required to join campus');
    }
    if (pass.length < 6) return showToast('Password must be 6+ characters');

    btn.disabled = true;
    btn.innerHTML = '<span class="inline-spinner"></span>';
    
    try {
      const cred = await auth.createUserWithEmailAndPassword(email, pass);
      const uid = cred.user.uid;
      const displayName = `${fname} ${lname}`;

      // Create Profile
      await db.collection('users').doc(uid).set({
        displayName,
        firstName: fname,
        lastName: lname,
        email,
        university: uni,
        major,
        bio: `Student at ${uni}`,
        photoURL: '',
        status: 'online',
        joinedAt: firebase.firestore.FieldValue.serverTimestamp(),
        friends: []
      });
      
      // Update global count
      db.collection('stats').doc('global').update({
        totalUsers: firebase.firestore.FieldValue.increment(1)
      }).catch(() => db.collection('stats').doc('global').set({ totalUsers: 1 }));

      await cred.user.updateProfile({ displayName });
      
    } catch (err) {
      showToast(friendlyError(err.code));
      btn.disabled = false;
      btn.textContent = 'Create Account';
    }
  });

  // Auth Listener
  auth.onAuthStateChanged(async (user) => {
    if (user) {
      state.user = user;
      const doc = await db.collection('users').doc(user.uid).get();
      if (doc.exists) {
        state.profile = { id: doc.id, ...doc.data() };
        state.status = state.profile.status || 'online';
      } else {
        // Fallback for race condition
        state.profile = {
          id: user.uid,
          displayName: user.displayName,
          email: user.email,
          status: 'online'
        };
      }
      enterApp();
    } else {
      state.user = null;
      state.profile = null;
      cleanupListeners();
      showScreen('auth-screen');
    }
  });
}

function friendlyError(code) {
  const map = {
    'auth/user-not-found': 'Account not found',
    'auth/wrong-password': 'Incorrect password',
    'auth/email-already-in-use': 'Email already registered',
    'auth/weak-password': 'Password too weak',
    'auth/invalid-email': 'Invalid email format'
  };
  return map[code] || 'Something went wrong. Try again.';
}

// ─── App Logic ────────────────────────────────────────
function enterApp() {
  showScreen('app-shell');
  updateHeaderProfile();
  updateStatusUI();
  
  // Default to Feed
  navigateTo('feed');
  
  // Navigation
  $$('.nav-item').forEach(btn => {
    btn.onclick = () => {
      const page = btn.dataset.page;
      if (page === 'create') openCreatePostModal();
      else navigateTo(page);
    };
  });

  // Status Toggle
  $('#status-pill').onclick = toggleStatus;
}

function updateHeaderProfile() {
  const el = $('#header-avatar');
  if (state.profile) {
    el.innerHTML = state.profile.photoURL 
      ? `<img src="${state.profile.photoURL}" alt="">` 
      : getInitials(state.profile.displayName);
    el.style.background = state.profile.photoURL ? 'transparent' : getAvatarColor(state.profile.displayName);
    el.onclick = () => showProfile(state.user.uid);
  }
}

async function toggleStatus() {
  const modes = ['online', 'study', 'offline'];
  const currentIdx = modes.indexOf(state.status);
  const nextStatus = modes[(currentIdx + 1) % modes.length];
  
  state.status = nextStatus;
  updateStatusUI();
  
  try {
    await db.collection('users').doc(state.user.uid).update({ status: nextStatus });
    showToast(`Status: ${nextStatus.charAt(0).toUpperCase() + nextStatus.slice(1)}`);
  } catch (err) {
    console.error('Status update failed', err);
  }
}

function updateStatusUI() {
  const dot = $('#status-dot');
  const text = $('#status-text');
  
  dot.className = 'status-dot'; // Reset
  
  if (state.status === 'online') {
    dot.classList.add('green');
    text.textContent = 'Online';
  } else if (state.status === 'study') {
    dot.classList.add('orange');
    text.textContent = 'Studying';
  } else {
    dot.classList.add('gray');
    text.textContent = 'Offline';
  }
}

// ─── Navigation ───────────────────────────────────────
function navigateTo(page) {
  state.currentPage = page;
  
  // Update Nav UI
  $$('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.page === page));
  
  // Clean Listeners
  cleanupListeners();
  
  // Render Page
  switch(page) {
    case 'feed': renderFeed(); break;
    case 'explore': renderExplore(); break;
    case 'hustle': renderHustle(); break;
    case 'messages': renderMessages(); break;
  }
}

function cleanupListeners() {
  state.unsubscribers.forEach(fn => fn());
  state.unsubscribers = [];
}

// ══════════════════════════════════════════════════════
//  FEED
// ══════════════════════════════════════════════════════
function renderFeed() {
  const content = $('#app-content');
  content.innerHTML = `
    <div class="feed-page">
      <div class="welcome-header">
        <h1>Hello, ${state.profile.firstName}! 👋</h1>
        <p>See what's happening at ${state.profile.university || 'campus'}</p>
      </div>

      <!-- Quick Actions / Groups -->
      <div style="padding: 0 16px 16px;">
        <div style="display:flex; gap:12px; overflow-x:auto; padding-bottom:4px;">
          <div class="group-pill active">All Posts</div>
          <div class="group-pill">${state.profile.major || 'Major'}</div>
          <div class="group-pill">Following</div>
          <div class="group-pill">Exam Prep</div>
          <div class="group-pill">Events</div>
        </div>
      </div>

      <div class="create-input-card" onclick="openCreatePostModal()">
        ${avatarHTML(state.profile.displayName, state.profile.photoURL, 'avatar-md')}
        <div class="fake-input">Share something with your campus...</div>
        <button class="icon-btn-small">📷</button>
      </div>

      <div id="feed-posts">
        <div style="padding:40px; text-align:center;">
          <span class="inline-spinner" style="width:32px;height:32px;color:var(--accent)"></span>
        </div>
      </div>
    </div>
  `;

  const unsub = db.collection('posts')
    .orderBy('createdAt', 'desc')
    .limit(50)
    .onSnapshot(snap => {
      const posts = snap.docs.map(d => ({id: d.id, ...d.data()}));
      renderPosts(posts);
    });
    
  state.unsubscribers.push(unsub);
}

function renderPosts(posts) {
  const container = $('#feed-posts');
  if (!container) return; // Page changed

  if (posts.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📝</div>
        <h3>No posts yet</h3>
        <p>Be the first to post something!</p>
      </div>`;
    return;
  }

  container.innerHTML = posts.map(post => {
    const isLiked = post.likes?.includes(state.user.uid);
    const likeCount = post.likes?.length || 0;
    const commentCount = post.commentsCount || 0;
    
    return `
      <div class="post-card" onclick="openPostDetail('${post.id}')">
        <div class="post-header">
          <div onclick="event.stopPropagation(); showProfile('${post.authorId}')" style="cursor:pointer">
            ${avatarHTML(post.authorName, post.authorPhoto, 'avatar-md')}
          </div>
          <div class="post-info">
            <div class="post-author" onclick="event.stopPropagation(); showProfile('${post.authorId}')">
              ${escapeHTML(post.authorName)}
            </div>
            <div class="post-meta">${escapeHTML(post.authorUni || '')} • ${timeAgo(post.createdAt)}</div>
          </div>
        </div>
        
        <div class="post-content">${escapeHTML(post.content)}</div>
        
        ${post.imageURL ? `
          <div class="post-image-container">
            <img src="${post.imageURL}" class="post-image" loading="lazy">
          </div>
        ` : ''}

        <div class="post-actions">
          <button class="action-btn ${isLiked ? 'active' : ''}" onclick="event.stopPropagation(); toggleLike('${post.id}')">
            ${isLiked ? '❤️' : '🤍'} ${likeCount || 'Like'}
          </button>
          <button class="action-btn">
            💬 ${commentCount || 'Comment'}
          </button>
          <button class="action-btn" onclick="event.stopPropagation(); sharePost('${post.id}')">
            Example ↗
          </button>
        </div>
      </div>
    `;
  }).join('');
}

// ─── Post Actions ─────────────────────────────────────
async function toggleLike(postId) {
  const ref = db.collection('posts').doc(postId);
  const uid = state.user.uid;
  
  try {
    const doc = await ref.get();
    if (!doc.exists) return;
    
    const likes = doc.data().likes || [];
    if (likes.includes(uid)) {
      await ref.update({ likes: firebase.firestore.FieldValue.arrayRemove(uid) });
    } else {
      await ref.update({ likes: firebase.firestore.FieldValue.arrayUnion(uid) });
    }
  } catch (err) {
    console.error('Like failed', err);
  }
}

function openCreatePostModal() {
  let pendingImage = null;
  
  const html = `
    <div style="padding:20px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
        <h2 style="font-size:18px; margin:0;">Create Post</h2>
        <button onclick="closeModal()" class="icon-btn">✕</button>
      </div>
      
      <div style="display:flex; gap:12px; margin-bottom:16px;">
        ${avatarHTML(state.profile.displayName, state.profile.photoURL, 'avatar-md')}
        <div>
          <div style="font-weight:600;">${escapeHTML(state.profile.displayName)}</div>
          <div style="font-size:12px; color:var(--text-secondary);">Posting to Public Feed</div>
        </div>
      </div>
      
      <textarea id="post-input-text" placeholder="What's on your mind?" style="width:100%; height:120px; border:none; background:transparent; font-size:16px; resize:none; outline:none; color:var(--text-primary);"></textarea>
      
      <div id="post-img-preview" style="display:none; margin:12px 0; position:relative;">
        <img src="" style="width:100%; border-radius:12px; max-height:200px; object-fit:cover;">
        <button onclick="this.parentElement.style.display='none'; document.getElementById('post-file-input').value='';" 
           style="position:absolute; top:8px; right:8px; background:rgba(0,0,0,0.6); color:#fff; border:none; border-radius:50%; width:24px; height:24px;">✕</button>
      </div>

      <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid var(--border); padding-top:16px;">
        <label class="icon-btn" style="color:var(--accent);">
          📷
          <input type="file" hidden accept="image/*" id="post-file-input">
        </label>
        <button id="submit-post-btn" class="btn-primary" style="padding:8px 24px;">Post</button>
      </div>
    </div>
  `;
  
  openModal(html);
  
  $('#post-file-input').onchange = async (e) => {
    if (e.target.files[0]) {
      pendingImage = await compressImage(e.target.files[0], 800);
      $('#post-img-preview img').src = pendingImage;
      $('#post-img-preview').style.display = 'block';
    }
  };
  
  $('#submit-post-btn').onclick = async () => {
    const text = $('#post-input-text').value.trim();
    if (!text && !pendingImage) return showToast('Post cannot be empty');
    
    closeModal(); // Optimistic close
    showToast('Posting...');
    
    try {
      await db.collection('posts').add({
        content: text,
        imageURL: pendingImage || null,
        authorId: state.user.uid,
        authorName: state.profile.displayName,
        authorPhoto: state.profile.photoURL || null,
        authorUni: state.profile.university,
        createdAt: firebase.firestore.FieldValue.serverTimestamp(),
        likes: [],
        commentsCount: 0
      });
      showToast('Posted successfully!');
    } catch (err) {
      showToast('Failed to post');
      console.error(err);
    }
  };
}

// ══════════════════════════════════════════════════════
//  EXPLORE
// ══════════════════════════════════════════════════════
function renderExplore() {
  const content = $('#app-content');
  content.innerHTML = `
    <div class="explore-page">
      <div class="search-bar-container">
        <input type="text" id="explore-search" class="search-input" placeholder="Find students, tutors...">
      </div>
      
      <div class="filter-row">
        <span class="chip active">All</span>
        <span class="chip">CS</span>
        <span class="chip">Eng</span>
        <span class="chip">Law</span>
        <span class="chip">Med</span>
        <span class="chip">Arts</span>
      </div>
      
      <h3 style="margin: 16px 16px 8px; font-size:15px;">Suggested for you</h3>
      <div id="users-grid" class="users-grid">
        <div style="grid-column:1/-1; text-align:center; padding:32px;">
          <span class="inline-spinner"></span>
        </div>
      </div>
    </div>
  `;
  
  // Initial Load
  loadUsers();
  
  // Search
  let typingTimer;
  $('#explore-search').addEventListener('input', (e) => {
    clearTimeout(typingTimer);
    typingTimer = setTimeout(() => loadUsers(e.target.value), 500);
  });
}

async function loadUsers(query = '') {
  const container = $('#users-grid');
  if (!container) return;
  
  try {
    let ref = db.collection('users').limit(20);
    // Simple query, client-side filtering for simplicity/reliability
    const snap = await ref.get();
    
    let users = snap.docs
                .map(d => ({id: d.id, ...d.data()}))
                .filter(u => u.id !== state.user.uid);
                
    if (query) {
      const q = query.toLowerCase();
      users = users.filter(u => 
        (u.displayName || '').toLowerCase().includes(q) ||
        (u.major || '').toLowerCase().includes(q)
      );
    }
    
    if (users.length === 0) {
      container.innerHTML = `<div class="empty-state" style="grid-column:1/-1"><h3>No results found</h3></div>`;
      return;
    }
    
    container.innerHTML = users.map(u => `
      <div class="user-card" onclick="showProfile('${u.id}')">
        ${avatarHTML(u.displayName, u.photoURL, 'avatar-lg', u.status)}
        <div class="user-name">${escapeHTML(u.displayName)}</div>
        <div class="user-desc">${escapeHTML(u.major || 'Student')}</div>
        <button class="btn-outline-small" style="margin-top:8px;" onclick="event.stopPropagation(); showProfile('${u.id}')">View</button>
      </div>
    `).join('');
    
  } catch (err) {
    console.error('Explore error', err);
    container.innerHTML = `<div class="empty-state" style="grid-column:1/-1"><h3>Error loading students</h3></div>`;
  }
}


// ══════════════════════════════════════════════════════
//  MESSAGING
// ══════════════════════════════════════════════════════
function renderMessages() {
  $('#app-content').innerHTML = `
    <div class="messages-page">
      <div class="page-header">
        <h2>Messages</h2>
      </div>
      <div id="conversations-list">
        <div style="padding:40px; text-align:center;"><span class="inline-spinner"></span></div>
      </div>
    </div>
  `;
  
  const unsub = db.collection('conversations')
    .where('participants', 'array-contains', state.user.uid)
    .orderBy('updatedAt', 'desc')
    .onSnapshot(snap => {
      const convos = snap.docs.map(d => ({id: d.id, ...d.data()}));
      renderConversations(convos);
    });
    
  state.unsubscribers.push(unsub);
}

function renderConversations(convos) {
  const container = $('#conversations-list');
  if (!container) return;
  
  if (convos.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">💬</div>
        <h3>No chats yet</h3>
        <p>Visit a profile to start chatting!</p>
      </div>`;
    return;
  }
  
  const myUid = state.user.uid;
  
  container.innerHTML = convos.map(c => {
    // Find "other" user data
    const idx = c.participants.indexOf(myUid) === 0 ? 1 : 0;
    const name = c.participantNames[idx] || 'User';
    const photo = c.participantPhotos[idx] || null;
    const unread = (c.unread || {})[myUid] || 0;
    
    return `
      <div class="convo-item" onclick="openChat('${c.id}', '${name.replace(/'/g, "\\'")}', '${photo || ''}')">
        ${avatarHTML(name, photo, 'avatar-md')}
        <div class="convo-info">
          <div class="convo-header">
            <span class="convo-name">${escapeHTML(name)}</span>
            <span class="convo-time">${timeAgo(c.updatedAt)}</span>
          </div>
          <div class="convo-last ${unread ? 'unread-text' : ''}">
            ${escapeHTML(c.lastMessage)}
          </div>
        </div>
        ${unread ? `<div class="unread-badge">${unread}</div>` : ''}
      </div>
    `;
  }).join('');
}

// ─── Chat Detail ──────────────────────────────────────
let currentChatUnsub = null;

function openChat(convoId, name, photo) {
  const html = `
    <div class="chat-screen screen active" id="chat-screen">
      <div class="chat-header">
        <button onclick="document.getElementById('chat-screen').remove()" class="icon-btn">⬅</button>
        <div style="display:flex; align-items:center; gap:10px; flex:1;">
          ${avatarHTML(name, photo, 'avatar-sm')}
          <div style="font-weight:600; font-size:15px;">${escapeHTML(name)}</div>
        </div>
      </div>
      <div class="chat-body" id="chat-messages">
        <div style="padding:20px; text-align:center;"><span class="inline-spinner"></span></div>
      </div>
      <div class="chat-footer">
        <input type="text" id="chat-input" placeholder="Message..." autocomplete="off">
        <button id="chat-send-btn" class="icon-btn" style="background:var(--accent); color:white;">➤</button>
      </div>
    </div>
  `;
  
  // Append to body (modal-like)
  document.body.insertAdjacentHTML('beforeend', html);
  
  const msgContainer = $('#chat-messages');
  const input = $('#chat-input');
  
  // Mark read
  db.collection('conversations').doc(convoId).set({
    unread: { [state.user.uid]: 0 }
  }, { merge: true });
  
  // Listen
  currentChatUnsub = db.collection('conversations').doc(convoId)
    .collection('messages')
    .orderBy('createdAt', 'asc')
    .limit(50)
    .onSnapshot(snap => {
      const msgs = snap.docs.map(d => ({id: d.id, ...d.data()}));
      
      if (msgs.length === 0) {
        msgContainer.innerHTML = '<div style="text-align:center; padding:20px; opacity:0.6;">Say hi! 👋</div>';
      } else {
        msgContainer.innerHTML = msgs.map(m => {
          const isMe = m.senderId === state.user.uid;
          return `
            <div class="msg-row ${isMe ? 'me' : 'them'}">
              <div class="msg-bubble">${escapeHTML(m.text)}</div>
              <div class="msg-time">${formatTime(m.createdAt)}</div>
            </div>
          `;
        }).join('');
        // Scroll to bottom
        msgContainer.scrollTop = msgContainer.scrollHeight;
      }
    });
    
  // Send
  $('#chat-send-btn').onclick = async () => {
    const text = input.value.trim();
    if (!text) return;
    input.value = ''; // Optimistic clear
    
    try {
      // 1. Add message
      await db.collection('conversations').doc(convoId).collection('messages').add({
        text,
        senderId: state.user.uid,
        createdAt: firebase.firestore.FieldValue.serverTimestamp()
      });
      
      // 2. Update conversation meta
      // Get 'other' user ID by inspecting convo participants or meta
      // Simplified: we just increment counters for *others* (but we need ID)
      // For MVP, simply setting updatedAt and lastMessage
      await db.collection('conversations').doc(convoId).set({
        lastMessage: text,
        updatedAt: firebase.firestore.FieldValue.serverTimestamp(),
        // Note: Real unread count requires knowing the OTHER user ID here
        // We'll skip complex unread logic for this specific snippet to keep it robust
      }, { merge: true });
      
    } catch (err) {
      console.error('Send failed', err);
    }
  };
}

async function startChatWith(uid, name, photo) {
  if (uid === state.user.uid) return showToast("That's you!");
  
  try {
    // Check existing
    const snap = await db.collection('conversations')
       .where('participants', 'array-contains', state.user.uid)
       .get();
       
    const existing = snap.docs.find(d => d.data().participants.includes(uid));
    
    if (existing) {
      openChat(existing.id, name, photo);
    } else {
      // Create new
      const doc = await db.collection('conversations').add({
        participants: [state.user.uid, uid],
        participantNames: [state.profile.displayName, name],
        participantPhotos: [state.profile.photoURL || null, photo || null],
        lastMessage: 'Started a conversation',
        updatedAt: firebase.firestore.FieldValue.serverTimestamp(),
        unread: { [uid]: 1, [state.user.uid]: 0 }
      });
      openChat(doc.id, name, photo);
    }
  } catch (err) {
    showToast('Could not start chat');
    console.error(err);
  }
}

// ══════════════════════════════════════════════════════
//  PROFILE (FIXED)
// ══════════════════════════════════════════════════════
async function showProfile(uid) {
  // Use a modal-like full screen
  const html = `
    <div class="profile-screen screen active" id="profile-overlay">
      <div class="profile-header-nav">
        <button onclick="document.getElementById('profile-overlay').remove()" class="icon-btn-bg">⬅</button>
        ${uid === state.user.uid ? '<button class="icon-btn-bg" onclick="logout()">LOGOUT</button>' : ''}
      </div>
      
      <div id="profile-body-content">
        <div style="padding:100px; text-align:center;"><span class="inline-spinner"></span></div>
      </div>
    </div>
  `;
  document.body.insertAdjacentHTML('beforeend', html);
  
  const content = $('#profile-body-content');
  
  try {
    // 1. Get User Data
    let user;
    if (uid === state.user.uid) user = state.profile;
    else {
      const doc = await db.collection('users').doc(uid).get();
      if (!doc.exists) throw new Error('User not found');
      user = doc.data();
    }
    
    // 2. Get Posts (Simple Query)
    const postsSnap = await db.collection('posts')
      .where('authorId', '==', uid)
      // .orderBy('createdAt', 'desc') // Removed to avoid index errors on new accounts
      .limit(20)
      .get();
      
    const posts = postsSnap.docs.map(d => ({id: d.id, ...d.data()}));
    // Sort manually since we removed orderBy
    posts.sort((a,b) => (b.createdAt?.seconds||0) - (a.createdAt?.seconds||0));

    // 3. Render
    content.innerHTML = `
      <div class="profile-cover"></div>
      <div class="profile-info-card">
        <div class="profile-avatar-row">
          <div class="profile-avatar-lg">
             ${user.photoURL ? `<img src="${user.photoURL}">` : `<div style="width:100%;height:100%;background:${getAvatarColor(user.displayName)};display:flex;align-items:center;justify-content:center;font-size:32px;color:white;">${getInitials(user.displayName)}</div>`}
          </div>
        </div>
        
        <h2 style="text-align:center; margin:8px 0 4px;">${escapeHTML(user.displayName)}</h2>
        <div style="text-align:center; color:var(--text-secondary); margin-bottom:16px;">
          ${escapeHTML(user.major || '')} @ ${escapeHTML(user.university || 'Unino')}
        </div>
        
        ${uid !== state.user.uid 
          ? `<button class="btn-primary" style="width:100%;" onclick="startChatWith('${uid}', '${user.displayName.replace(/'/g, "\\'")}', '${user.photoURL||''}')">Message</button>`
          : `<button class="btn-outline" style="width:100%;" onclick="editProfile()">Edit Profile</button>`
        }
        
        <div class="stats-row">
          <div class="stat-item">
            <div class="stat-num">${posts.length}</div>
            <div class="stat-label">Posts</div>
          </div>
          <div class="stat-item">
            <div class="stat-num">0</div>
            <div class="stat-label">Friends</div>
          </div>
        </div>
      </div>
      
      <div class="profile-posts-grid">
        ${posts.length ? posts.map(p => `
           <div class="mini-post" onclick="document.getElementById('profile-overlay').remove()">
             ${p.imageURL ? `<img src="${p.imageURL}">` : `<div style="padding:16px; font-size:12px;">${escapeHTML(p.content.substring(0,50))}...</div>`}
           </div>
        `).join('') : '<div style="padding:32px; text-align:center; color:var(--text-tertiary);">No posts yet</div>'}
      </div>
    `;
    
  } catch (err) {
    console.error(err);
    content.innerHTML = `<div class="empty-state"><h3>Could not load profile</h3></div>`;
  }
}

// ─── Marketplace ──────────────────────────────────────
function renderHustle() {
  $('#app-content').innerHTML = `
    <div class="hustle-page">
      <div class="page-header">
        <h2>Marketplace</h2>
        <button class="btn-primary-small" onclick="openSellModal()">+ Sell</button>
      </div>
      
      <div class="scroll-tabs">
        <div class="chip active">All</div>
        <div class="chip">Books</div>
        <div class="chip">Tech</div>
        <div class="chip">Services</div>
      </div>
      
      <div id="hustle-grid" class="hustle-grid">
        <div style="grid-column:1/-1; text-align:center; padding:32px;"><span class="inline-spinner"></span></div>
      </div>
    </div>
  `;
  
  db.collection('listings').where('status', '==', 'active').limit(50).get()
    .then(snap => {
      const items = snap.docs.map(d => ({id: d.id, ...d.data()}));
      const grid = $('#hustle-grid');
      
      if (!items.length) {
        grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1"><h3>Empty Marketplace</h3></div>`;
        return;
      }
      
      grid.innerHTML = items.map(item => `
        <div class="listing-card" onclick="startChatWith('${item.sellerId}', '${escapeHTML(item.sellerName)}', null)">
          <div class="listing-img-box">
            ${item.imageURL ? `<img src="${item.imageURL}">` : `<div style="font-size:32px;">📦</div>`}
          </div>
          <div class="listing-details">
            <div class="listing-price">R${item.price}</div>
            <div class="listing-title">${escapeHTML(item.title)}</div>
          </div>
        </div>
      `).join('');
    });
}

function openSellModal() {
  // Simplified sell modal
  const html = `
    <div style="padding:20px;">
      <h2>Sell Item</h2>
      <input type="text" id="sell-title" placeholder="What are you selling?" class="modal-input">
      <input type="number" id="sell-price" placeholder="Price (R)" class="modal-input">
      <button class="btn-primary" style="width:100%; margin-top:16px;" id="post-listing-btn">List Item</button>
    </div>
  `;
  openModal(html);
  
  $('#post-listing-btn').onclick = async () => {
    const title = $('#sell-title').value;
    const price = $('#sell-price').value;
    if(!title || !price) return showToast('Details required');
    
    closeModal();
    
    await db.collection('listings').add({
      title, price,
      sellerId: state.user.uid,
      sellerName: state.profile.displayName,
      status: 'active',
      createdAt: firebase.firestore.FieldValue.serverTimestamp()
    });
    showToast('Listed!');
    navigateTo('hustle'); // Refresh
  };
}


// ─── Modal System ─────────────────────────────────────
function openModal(innerHTML) {
  const modal = document.createElement('div');
  modal.className = 'modal-overlay';
  modal.id = 'dynamic-modal';
  modal.innerHTML = `
    <div class="modal-card">
      ${innerHTML}
    </div>
  `;
  modal.onclick = (e) => { if (e.target === modal) closeModal(); };
  document.body.appendChild(modal);
}

function closeModal() {
  const m = $('#dynamic-modal');
  if (m) m.remove();
}

// ─── Logout ───────────────────────────────────────────
function logout() {
  auth.signOut().then(() => {
    document.querySelectorAll('.screen').forEach(s => s.remove()); // Hard reset DOM overlay
    window.location.reload();
  });
}

// ─── Init ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initAuth();
  
  // Expose global for inline onclick
  window.openCreatePostModal = openCreatePostModal;
  window.showProfile = showProfile;
  window.openPostDetail = (id) => {}; // Placeholder
  window.toggleLike = toggleLike;
  window.sharePost = (id) => { showToast('Link copied!'); };
  window.editProfile = () => { showToast('Edit coming soon'); };
  window.logout = logout;
  window.startChatWith = startChatWith;
  window.openChat = openChat;
  window.closeModal = closeModal;
  window.openSellModal = openSellModal;
  window.editProfile = () => showToast('Feature in next update');
});
