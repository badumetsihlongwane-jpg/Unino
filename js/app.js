/* ══════════════════════════════════════════════════════
 *  UNINO — Complete Application Engine
 *  Firebase Auth + Firestore (No Storage — base64 images)
 * ══════════════════════════════════════════════════════ */

// ─── State ────────────────────────────────────────────
const state = {
  user: null,           // Firebase User object
  profile: null,        // Firestore user doc
  currentPage: 'feed',
  posts: [],
  listings: [],
  conversations: [],
  users: [],
  totalUsers: 0,
  unsubscribers: [],    // Firestore listeners to clean up
};

// ─── Helpers ──────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const AVATAR_COLORS = [
  '#6C5CE7', '#3B82F6', '#10B981', '#F59E0B', '#EF4444',
  '#EC4899', '#8B5CF6', '#06B6D4', '#F97316', '#14B8A6'
];

function getAvatarColor(name) {
  let hash = 0;
  for (let i = 0; i < (name || '').length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}

function getInitials(name) {
  if (!name) return '?';
  const parts = name.trim().split(/\s+/);
  return (parts[0][0] + (parts[1] ? parts[1][0] : '')).toUpperCase();
}

function avatarHTML(name, photoURL, sizeClass = 'avatar-sm') {
  const color = getAvatarColor(name);
  if (photoURL) {
    return `<div class="${sizeClass}" style="background:${color}"><img src="${photoURL}" alt="${name}" onerror="this.remove()"></div>`;
  }
  return `<div class="${sizeClass}" style="background:${color}">${getInitials(name)}</div>`;
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
  return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
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
        const ratio = Math.min(maxWidth / img.width, 1);
        canvas.width = img.width * ratio;
        canvas.height = img.height * ratio;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL('image/jpeg', quality));
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

function compressAvatar(file) {
  return compressImage(file, 200, 0.6);
}

// ─── Screen Manager ───────────────────────────────────
function showScreen(id) {
  $$('.screen').forEach(s => s.classList.remove('active'));
  const el = document.getElementById(id);
  if (el) el.classList.add('active');
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

// ─── User Count (Realtime) ────────────────────────────
function listenUserCount() {
  const ref = db.collection('stats').doc('global');
  // Initialize if doesn't exist
  ref.get().then(snap => {
    if (!snap.exists) ref.set({ totalUsers: 0 });
  });
  const unsub = ref.onSnapshot(snap => {
    if (snap.exists) {
      state.totalUsers = snap.data().totalUsers || 0;
      updateUserCountUI();
    }
  });
  state.unsubscribers.push(unsub);
}

function updateUserCountUI() {
  const count = state.totalUsers;
  const el1 = $('#auth-count-num');
  const el2 = $('#header-count');
  if (el1) el1.textContent = count;
  if (el2) el2.textContent = count;
}

// ─── Auth ─────────────────────────────────────────────
function initAuth() {
  // Toggle login/signup forms
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
    if (!email || !pass) return showToast('Fill in all fields');
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
    const year = $('#signup-year').value;

    if (!fname || !lname || !email || !pass || !uni || !major) {
      return showToast('Please fill in all required fields');
    }
    if (pass.length < 6) return showToast('Password must be 6+ characters');

    btn.disabled = true;
    btn.innerHTML = '<span class="inline-spinner"></span>';
    try {
      const cred = await auth.createUserWithEmailAndPassword(email, pass);
      const uid = cred.user.uid;
      const displayName = fname + ' ' + lname;

      // Create user profile in Firestore
      await db.collection('users').doc(uid).set({
        displayName,
        firstName: fname,
        lastName: lname,
        email,
        university: uni,
        major,
        year: year || '',
        bio: '',
        photoURL: '',
        joinedAt: firebase.firestore.FieldValue.serverTimestamp(),
        postsCount: 0,
        friendsCount: 0,
      });

      // Increment global user count
      await db.collection('stats').doc('global').set({
        totalUsers: firebase.firestore.FieldValue.increment(1)
      }, { merge: true });

      await cred.user.updateProfile({ displayName });
    } catch (err) {
      showToast(friendlyError(err.code));
      btn.disabled = false;
      btn.textContent = 'Create Account';
    }
  });

  // Auth state listener
  auth.onAuthStateChanged(async (user) => {
    if (user) {
      state.user = user;
      // Fetch profile
      const doc = await db.collection('users').doc(user.uid).get();
      if (doc.exists) {
        state.profile = { id: doc.id, ...doc.data() };
      } else {
        // Profile might not be created yet (race condition)
        state.profile = {
          id: user.uid,
          displayName: user.displayName || user.email,
          email: user.email,
          photoURL: '',
          university: '',
          major: '',
          bio: '',
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
    'auth/user-not-found': 'No account with that email',
    'auth/wrong-password': 'Incorrect password',
    'auth/email-already-in-use': 'Email already registered',
    'auth/weak-password': 'Password too weak (6+ chars)',
    'auth/invalid-email': 'Invalid email address',
    'auth/too-many-requests': 'Too many attempts. Try later',
    'auth/invalid-credential': 'Invalid email or password',
  };
  return map[code] || 'Something went wrong. Try again.';
}

// ─── Enter App ────────────────────────────────────────
function enterApp() {
  showScreen('app-shell');
  updateHeaderAvatar();
  listenUserCount();
  navigateTo('feed');
  initNavigation();
  initHeaderActions();
}

function updateHeaderAvatar() {
  const el = $('#header-avatar');
  if (!el || !state.profile) return;
  const name = state.profile.displayName || '';
  const photo = state.profile.photoURL || '';
  if (photo) {
    el.innerHTML = `<img src="${photo}" alt="">`;
  } else {
    el.textContent = getInitials(name);
  }
  el.style.background = photo ? 'none' : getAvatarColor(name);
  el.onclick = () => showProfile(state.user.uid);
}

// ─── Navigation ───────────────────────────────────────
function initNavigation() {
  $$('#bottom-nav .nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
      const page = btn.dataset.page;
      if (page === 'create') {
        openCreatePostModal();
        return;
      }
      navigateTo(page);
    });
  });
}

function navigateTo(page) {
  state.currentPage = page;
  $$('#bottom-nav .nav-item').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.page === page);
  });

  // Cleanup old page listeners
  cleanupListeners();

  const content = $('#app-content');
  content.scrollTop = 0;

  switch (page) {
    case 'feed': renderFeed(); break;
    case 'explore': renderExplore(); break;
    case 'hustle': renderHustle(); break;
    case 'messages': renderMessages(); break;
    default: renderFeed();
  }
}

function cleanupListeners() {
  state.unsubscribers.forEach(fn => fn());
  state.unsubscribers = [];
}

function initHeaderActions() {
  // Notifications
  let notifOpen = false;
  $('#notif-btn')?.addEventListener('click', () => {
    const existing = document.querySelector('.notif-dropdown');
    if (existing) { existing.remove(); notifOpen = false; return; }
    notifOpen = true;
    const dropdown = document.createElement('div');
    dropdown.className = 'notif-dropdown';
    dropdown.innerHTML = `
      <div class="notif-dropdown-header"><h3>Notifications</h3></div>
      <div style="padding: 32px 16px; text-align: center; color: var(--text-tertiary); font-size: 14px;">
        No new notifications
      </div>
    `;
    $('#app-header').appendChild(dropdown);
    setTimeout(() => {
      document.addEventListener('click', function closeNotif(e) {
        if (!dropdown.contains(e.target) && e.target !== $('#notif-btn')) {
          dropdown.remove();
          notifOpen = false;
          document.removeEventListener('click', closeNotif);
        }
      });
    }, 10);
  });
}

// ══════════════════════════════════════════════════════
//  FEED PAGE
// ══════════════════════════════════════════════════════
function renderFeed() {
  const content = $('#app-content');
  content.innerHTML = `
    <div class="feed-page">
      <div class="create-post-prompt" id="feed-create-prompt">
        ${avatarHTML(state.profile?.displayName, state.profile?.photoURL, 'avatar-md')}
        <span class="placeholder-text">What\'s on your mind?</span>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
      </div>
      <div id="feed-posts">
        <div class="empty-state" id="feed-loading">
          <div class="inline-spinner" style="width:32px;height:32px;border-width:3px;color:var(--accent)"></div>
        </div>
      </div>
    </div>
  `;

  $('#feed-create-prompt')?.addEventListener('click', openCreatePostModal);
  listenPosts();
}

function listenPosts() {
  const unsub = db.collection('posts')
    .orderBy('createdAt', 'desc')
    .limit(50)
    .onSnapshot(snap => {
      state.posts = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      renderPostsList();
    }, err => {
      console.error('Posts listener error:', err);
      $('#feed-posts').innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">📝</div>
          <h3>No posts yet</h3>
          <p>Be the first to share something with your campus!</p>
        </div>
      `;
    });
  state.unsubscribers.push(unsub);
}

function renderPostsList() {
  const container = $('#feed-posts');
  if (!container) return;

  if (state.posts.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">📝</div>
        <h3>No posts yet</h3>
        <p>Be the first to share something with your campus!</p>
      </div>
    `;
    return;
  }

  container.innerHTML = state.posts.map(post => postCardHTML(post)).join('');
  attachPostListeners();
}

function postCardHTML(post) {
  const isOwn = post.authorId === state.user?.uid;
  const liked = post.likedBy && post.likedBy.includes(state.user?.uid);
  const likeCount = post.likesCount || 0;
  const commentCount = post.commentsCount || 0;

  return `
    <div class="post-card" data-post-id="${post.id}">
      <div class="post-header">
        ${avatarHTML(post.authorName, post.authorPhoto, 'avatar-md')}
        <div class="post-header-info">
          <div class="post-author-name" data-uid="${post.authorId}">${escapeHTML(post.authorName)}</div>
          <div class="post-meta">${post.authorUni || ''} · ${timeAgo(post.createdAt)}</div>
        </div>
        ${isOwn ? `<button class="icon-btn post-more-btn" data-post-id="${post.id}">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/></svg>
        </button>` : ''}
      </div>
      ${post.content ? `<div class="post-content">${escapeHTML(post.content)}</div>` : ''}
      ${post.imageURL ? `<img class="post-image" src="${post.imageURL}" alt="Post image" data-full="${post.imageURL}">` : ''}
      <div class="post-stats">
        <span class="like-count-${post.id}">${likeCount ? likeCount + ' like' + (likeCount > 1 ? 's' : '') : ''}</span>
        <span class="comment-count-btn" data-post-id="${post.id}">${commentCount ? commentCount + ' comment' + (commentCount > 1 ? 's' : '') : ''}</span>
      </div>
      <div class="post-actions">
        <button class="post-action like-btn ${liked ? 'liked' : ''}" data-post-id="${post.id}">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
          <span>${liked ? 'Liked' : 'Like'}</span>
        </button>
        <button class="post-action comment-btn" data-post-id="${post.id}">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          <span>Comment</span>
        </button>
      </div>
      <div class="comments-section" id="comments-${post.id}" style="display:none"></div>
    </div>
  `;
}

function attachPostListeners() {
  // Like buttons
  $$('.like-btn').forEach(btn => {
    btn.addEventListener('click', () => toggleLike(btn.dataset.postId));
  });

  // Comment buttons
  $$('.comment-btn, .comment-count-btn').forEach(btn => {
    btn.addEventListener('click', () => toggleComments(btn.dataset.postId));
  });

  // Post images — open viewer
  $$('.post-image').forEach(img => {
    img.addEventListener('click', () => openImageViewer(img.dataset.full));
  });

  // Author name → profile
  $$('.post-author-name').forEach(el => {
    el.addEventListener('click', () => showProfile(el.dataset.uid));
  });

  // Post more options (delete)
  $$('.post-more-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      showPostOptions(btn.dataset.postId);
    });
  });
}

// ─── Like ─────────────────────────────────────────────
async function toggleLike(postId) {
  const uid = state.user.uid;
  const postRef = db.collection('posts').doc(postId);

  try {
    const doc = await postRef.get();
    if (!doc.exists) return;
    const data = doc.data();
    const likedBy = data.likedBy || [];
    const isLiked = likedBy.includes(uid);

    await postRef.update({
      likedBy: isLiked
        ? firebase.firestore.FieldValue.arrayRemove(uid)
        : firebase.firestore.FieldValue.arrayUnion(uid),
      likesCount: firebase.firestore.FieldValue.increment(isLiked ? -1 : 1)
    });
  } catch (err) {
    console.error('Like error:', err);
  }
}

// ─── Comments ─────────────────────────────────────────
async function toggleComments(postId) {
  const section = $(`#comments-${postId}`);
  if (!section) return;

  if (section.style.display === 'none') {
    section.style.display = 'block';
    section.innerHTML = '<div style="padding:12px;text-align:center"><span class="inline-spinner"></span></div>';
    loadComments(postId);
  } else {
    section.style.display = 'none';
  }
}

async function loadComments(postId) {
  const section = $(`#comments-${postId}`);
  if (!section) return;

  try {
    const snap = await db.collection('posts').doc(postId)
      .collection('comments')
      .orderBy('createdAt', 'asc')
      .limit(20)
      .get();

    const comments = snap.docs.map(d => ({ id: d.id, ...d.data() }));

    section.innerHTML = `
      ${comments.map(c => `
        <div class="comment-item">
          ${avatarHTML(c.authorName, c.authorPhoto, 'avatar-sm')}
          <div class="comment-body">
            <span class="comment-author" data-uid="${c.authorId}">${escapeHTML(c.authorName)}</span>
            <div class="comment-text">${escapeHTML(c.content)}</div>
            <div class="comment-time">${timeAgo(c.createdAt)}</div>
          </div>
        </div>
      `).join('')}
      <div class="comment-input-row">
        <input type="text" placeholder="Write a comment..." id="comment-input-${postId}">
        <button onclick="submitComment('${postId}')">Post</button>
      </div>
    `;

    // Enter key to submit
    $(`#comment-input-${postId}`)?.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') submitComment(postId);
    });

    // Click author name to show profile
    section.querySelectorAll('.comment-author').forEach(el => {
      el.style.cursor = 'pointer';
      el.addEventListener('click', () => showProfile(el.dataset.uid));
    });
  } catch (err) {
    section.innerHTML = '<p style="padding:12px;color:var(--text-tertiary);font-size:13px">Could not load comments</p>';
  }
}

async function submitComment(postId) {
  const input = $(`#comment-input-${postId}`);
  if (!input) return;
  const text = input.value.trim();
  if (!text) return;
  input.value = '';

  try {
    await db.collection('posts').doc(postId).collection('comments').add({
      content: text,
      authorId: state.user.uid,
      authorName: state.profile.displayName,
      authorPhoto: state.profile.photoURL || '',
      createdAt: firebase.firestore.FieldValue.serverTimestamp()
    });
    await db.collection('posts').doc(postId).update({
      commentsCount: firebase.firestore.FieldValue.increment(1)
    });
    loadComments(postId);
  } catch (err) {
    showToast('Could not post comment');
  }
}

// ─── Post Options (Delete) ────────────────────────────
function showPostOptions(postId) {
  openModal(`
    <div style="padding:20px">
      <h3 style="margin-bottom:16px">Post Options</h3>
      <button class="btn-danger btn-full" id="delete-post-btn" style="margin-bottom:8px">Delete Post</button>
      <button class="btn-ghost btn-full" onclick="closeModal()">Cancel</button>
    </div>
  `);
  $('#delete-post-btn')?.addEventListener('click', async () => {
    try {
      await db.collection('posts').doc(postId).delete();
      await db.collection('users').doc(state.user.uid).update({
        postsCount: firebase.firestore.FieldValue.increment(-1)
      });
      showToast('Post deleted');
      closeModal();
    } catch (err) {
      showToast('Could not delete post');
    }
  });
}

// ─── Create Post Modal ────────────────────────────────
function openCreatePostModal() {
  let pendingImage = '';

  openModal(`
    <div class="create-post-form">
      <h2>Create Post</h2>
      <div class="create-post-top">
        ${avatarHTML(state.profile?.displayName, state.profile?.photoURL, 'avatar-md')}
        <div>
          <strong>${escapeHTML(state.profile?.displayName)}</strong>
          <div style="font-size:12px;color:var(--text-secondary)">${state.profile?.university || ''}</div>
        </div>
      </div>
      <textarea id="new-post-text" placeholder="What's happening on campus?" maxlength="1000"></textarea>
      <div class="image-preview-container" id="post-image-preview">
        <img src="" alt="Preview" id="post-preview-img">
        <button class="image-preview-remove" id="post-remove-img">&times;</button>
      </div>
      <div class="create-post-bottom">
        <div class="attach-row">
          <label class="attach-btn" for="post-image-upload">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
            Photo
          </label>
          <input type="file" id="post-image-upload" accept="image/*" hidden>
        </div>
        <button class="btn-primary" id="submit-post-btn">Post</button>
      </div>
    </div>
  `);

  $('#post-image-upload')?.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (file.size > 5 * 1024 * 1024) { showToast('Image too large (max 5MB)'); return; }
    pendingImage = await compressImage(file, 800, 0.7);
    $('#post-preview-img').src = pendingImage;
    $('#post-image-preview').style.display = 'block';
  });

  $('#post-remove-img')?.addEventListener('click', () => {
    pendingImage = '';
    $('#post-image-preview').style.display = 'none';
    $('#post-image-upload').value = '';
  });

  $('#submit-post-btn')?.addEventListener('click', async () => {
    const text = $('#new-post-text').value.trim();
    if (!text && !pendingImage) return showToast('Write something or add a photo');

    const btn = $('#submit-post-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="inline-spinner"></span>';

    try {
      await db.collection('posts').add({
        content: text,
        imageURL: pendingImage || '',
        authorId: state.user.uid,
        authorName: state.profile.displayName,
        authorPhoto: state.profile.photoURL || '',
        authorUni: state.profile.university || '',
        likesCount: 0,
        commentsCount: 0,
        likedBy: [],
        createdAt: firebase.firestore.FieldValue.serverTimestamp()
      });

      await db.collection('users').doc(state.user.uid).update({
        postsCount: firebase.firestore.FieldValue.increment(1)
      });

      showToast('Posted!');
      closeModal();
    } catch (err) {
      console.error('Post error:', err);
      showToast('Could not create post');
      btn.disabled = false;
      btn.textContent = 'Post';
    }
  });
}

// ══════════════════════════════════════════════════════
//  EXPLORE PAGE
// ══════════════════════════════════════════════════════
function renderExplore() {
  const content = $('#app-content');
  content.innerHTML = `
    <div class="explore-page">
      <div class="search-bar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <input type="text" placeholder="Search students..." id="explore-search">
      </div>
      <div class="filter-chips" id="explore-filters">
        <span class="chip active" data-filter="all">All Students</span>
        <span class="chip" data-filter="Computer Science">CS</span>
        <span class="chip" data-filter="Engineering">Engineering</span>
        <span class="chip" data-filter="Business / Commerce">Business</span>
        <span class="chip" data-filter="Law">Law</span>
        <span class="chip" data-filter="Medicine / Health Sciences">Medicine</span>
        <span class="chip" data-filter="Arts & Design">Arts</span>
      </div>
      <div class="users-grid" id="explore-users">
        <div style="grid-column:1/-1;text-align:center;padding:32px"><span class="inline-spinner" style="width:32px;height:32px;border-width:3px;color:var(--accent)"></span></div>
      </div>
    </div>
  `;

  loadExploreUsers();

  // Search
  let searchTimeout;
  $('#explore-search')?.addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => filterExploreUsers(e.target.value.trim()), 300);
  });

  // Filter chips
  $$('#explore-filters .chip').forEach(chip => {
    chip.addEventListener('click', () => {
      $$('#explore-filters .chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      filterExploreUsers($('#explore-search')?.value || '', chip.dataset.filter);
    });
  });
}

async function loadExploreUsers() {
  try {
    const snap = await db.collection('users')
      .orderBy('joinedAt', 'desc')
      .limit(50)
      .get();
    state.users = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    displayExploreUsers(state.users.filter(u => u.id !== state.user?.uid));
  } catch (err) {
    console.error('Load users error:', err);
    $('#explore-users').innerHTML = `
      <div style="grid-column:1/-1" class="empty-state">
        <div class="empty-state-icon">🔍</div>
        <h3>No students found</h3>
        <p>Invite your friends to join Unino!</p>
      </div>
    `;
  }
}

function filterExploreUsers(query, majorFilter = 'all') {
  let filtered = state.users.filter(u => u.id !== state.user?.uid);
  if (query) {
    const q = query.toLowerCase();
    filtered = filtered.filter(u =>
      (u.displayName || '').toLowerCase().includes(q) ||
      (u.university || '').toLowerCase().includes(q) ||
      (u.major || '').toLowerCase().includes(q)
    );
  }
  if (majorFilter && majorFilter !== 'all') {
    filtered = filtered.filter(u => u.major === majorFilter);
  }
  displayExploreUsers(filtered);
}

function displayExploreUsers(users) {
  const container = $('#explore-users');
  if (!container) return;

  if (users.length === 0) {
    container.innerHTML = `
      <div style="grid-column:1/-1" class="empty-state">
        <div class="empty-state-icon">🔍</div>
        <h3>No students found</h3>
        <p>Try a different search or filter</p>
      </div>
    `;
    return;
  }

  container.innerHTML = users.map(u => `
    <div class="user-card" data-uid="${u.id}">
      ${avatarHTML(u.displayName, u.photoURL, 'avatar-lg')}
      <div class="user-card-name">${escapeHTML(u.displayName)}</div>
      <div class="user-card-uni">${escapeHTML(u.university || '')}</div>
      ${u.major ? `<div class="user-card-major">${escapeHTML(u.major)}</div>` : ''}
    </div>
  `).join('');

  container.querySelectorAll('.user-card').forEach(card => {
    card.addEventListener('click', () => showProfile(card.dataset.uid));
  });
}

// ══════════════════════════════════════════════════════
//  HUSTLE (MARKETPLACE) PAGE
// ══════════════════════════════════════════════════════
function renderHustle() {
  const content = $('#app-content');
  content.innerHTML = `
    <div class="hustle-page">
      <div class="hustle-header">
        <h2>💰 The Hustle</h2>
        <button class="btn-primary" id="create-listing-btn" style="padding:10px 16px;font-size:13px">+ Sell</button>
      </div>
      <div class="search-bar" style="margin-bottom:12px">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <input type="text" placeholder="Search marketplace..." id="hustle-search">
      </div>
      <div class="category-tabs" id="hustle-categories">
        <span class="chip active" data-cat="all">All</span>
        <span class="chip" data-cat="textbook">📚 Textbooks</span>
        <span class="chip" data-cat="electronics">💻 Electronics</span>
        <span class="chip" data-cat="furniture">🪑 Furniture</span>
        <span class="chip" data-cat="service">🛠️ Services</span>
        <span class="chip" data-cat="tutoring">📖 Tutoring</span>
        <span class="chip" data-cat="other">🏷️ Other</span>
      </div>
      <div class="listings-grid" id="listings-container">
        <div style="grid-column:1/-1;text-align:center;padding:32px"><span class="inline-spinner" style="width:32px;height:32px;border-width:3px;color:var(--accent)"></span></div>
      </div>
    </div>
  `;

  loadListings();
  $('#create-listing-btn')?.addEventListener('click', openCreateListingModal);

  // Search
  let searchTimeout;
  $('#hustle-search')?.addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => filterListings(e.target.value.trim()), 300);
  });

  // Category tabs
  $$('#hustle-categories .chip').forEach(chip => {
    chip.addEventListener('click', () => {
      $$('#hustle-categories .chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      filterListings($('#hustle-search')?.value || '', chip.dataset.cat);
    });
  });
}

async function loadListings() {
  try {
    const snap = await db.collection('listings')
      .where('status', '==', 'active')
      .orderBy('createdAt', 'desc')
      .limit(50)
      .get();
    state.listings = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    displayListings(state.listings);
  } catch (err) {
    console.error('Listings error:', err);
    $('#listings-container').innerHTML = `
      <div style="grid-column:1/-1" class="empty-state">
        <div class="empty-state-icon">🏪</div>
        <h3>Marketplace is empty</h3>
        <p>Be the first to list something for sale!</p>
      </div>
    `;
  }
}

function filterListings(query, category = 'all') {
  let filtered = [...state.listings];
  if (query) {
    const q = query.toLowerCase();
    filtered = filtered.filter(l =>
      (l.title || '').toLowerCase().includes(q) ||
      (l.description || '').toLowerCase().includes(q)
    );
  }
  if (category && category !== 'all') {
    filtered = filtered.filter(l => l.category === category);
  }
  displayListings(filtered);
}

function displayListings(listings) {
  const container = $('#listings-container');
  if (!container) return;

  if (listings.length === 0) {
    container.innerHTML = `
      <div style="grid-column:1/-1" class="empty-state">
        <div class="empty-state-icon">🏪</div>
        <h3>Nothing here yet</h3>
        <p>Try a different search or list something yourself!</p>
      </div>
    `;
    return;
  }

  container.innerHTML = listings.map(l => `
    <div class="listing-card" data-listing-id="${l.id}">
      ${l.imageURL
        ? `<img class="listing-image" src="${l.imageURL}" alt="${escapeHTML(l.title)}">`
        : `<div class="listing-image" style="display:flex;align-items:center;justify-content:center;font-size:48px;background:var(--bg-tertiary)">${getCategoryEmoji(l.category)}</div>`
      }
      <div class="listing-info">
        <div class="listing-title">${escapeHTML(l.title)}</div>
        <div class="listing-price">R${Number(l.price || 0).toLocaleString()}</div>
        <div class="listing-seller">
          ${avatarHTML(l.sellerName, l.sellerPhoto, 'avatar-sm')}
          ${escapeHTML(l.sellerName || 'Unknown')}
        </div>
      </div>
    </div>
  `).join('');

  container.querySelectorAll('.listing-card').forEach(card => {
    card.addEventListener('click', () => showListingDetail(card.dataset.listingId));
  });
}

function getCategoryEmoji(cat) {
  const map = { textbook: '📚', electronics: '💻', furniture: '🪑', service: '🛠️', tutoring: '📖', other: '🏷️' };
  return map[cat] || '🏷️';
}

function showListingDetail(listingId) {
  const listing = state.listings.find(l => l.id === listingId);
  if (!listing) return;
  const isOwn = listing.sellerId === state.user?.uid;

  openModal(`
    <div class="listing-detail">
      ${listing.imageURL
        ? `<img class="listing-detail-image" src="${listing.imageURL}" alt="">`
        : `<div class="listing-detail-image" style="display:flex;align-items:center;justify-content:center;font-size:80px">${getCategoryEmoji(listing.category)}</div>`
      }
      <div class="listing-detail-title">${escapeHTML(listing.title)}</div>
      <div class="listing-detail-price">R${Number(listing.price || 0).toLocaleString()}</div>
      <div class="listing-detail-desc">${escapeHTML(listing.description || 'No description')}</div>
      <div class="listing-detail-seller" data-uid="${listing.sellerId}">
        ${avatarHTML(listing.sellerName, listing.sellerPhoto, 'avatar-md')}
        <div class="listing-detail-seller-info">
          <div class="listing-detail-seller-name">${escapeHTML(listing.sellerName)}</div>
          <div class="listing-detail-seller-uni">${escapeHTML(listing.sellerUni || '')}</div>
        </div>
      </div>
      ${isOwn
        ? `<button class="btn-danger btn-full" id="delete-listing-btn">Remove Listing</button>`
        : `<button class="btn-primary btn-full" id="message-seller-btn">Message Seller</button>`
      }
    </div>
  `);

  // Seller profile click
  document.querySelector('.listing-detail-seller')?.addEventListener('click', () => {
    closeModal();
    showProfile(listing.sellerId);
  });

  // Message seller
  $('#message-seller-btn')?.addEventListener('click', () => {
    closeModal();
    startConversation(listing.sellerId, listing.sellerName, listing.sellerPhoto);
  });

  // Delete listing
  $('#delete-listing-btn')?.addEventListener('click', async () => {
    try {
      await db.collection('listings').doc(listingId).delete();
      showToast('Listing removed');
      closeModal();
      loadListings();
    } catch (err) {
      showToast('Could not delete listing');
    }
  });
}

function openCreateListingModal() {
  let pendingImage = '';

  openModal(`
    <div class="create-listing-form">
      <h2>Sell Something</h2>
      <div class="form-group">
        <label>Title *</label>
        <input type="text" id="listing-title" placeholder="What are you selling?" maxlength="100">
      </div>
      <div class="form-group">
        <label>Price (ZAR) *</label>
        <input type="number" id="listing-price" placeholder="0" min="0" step="1">
      </div>
      <div class="form-group">
        <label>Category *</label>
        <select id="listing-category">
          <option value="">Select category</option>
          <option value="textbook">📚 Textbook</option>
          <option value="electronics">💻 Electronics</option>
          <option value="furniture">🪑 Furniture</option>
          <option value="service">🛠️ Service</option>
          <option value="tutoring">📖 Tutoring</option>
          <option value="other">🏷️ Other</option>
        </select>
      </div>
      <div class="form-group">
        <label>Description</label>
        <textarea id="listing-desc" placeholder="Describe your item or service..." rows="3" style="resize:vertical"></textarea>
      </div>
      <div class="form-group">
        <label>Photo</label>
        <div class="image-preview-container" id="listing-image-preview">
          <img src="" alt="Preview" id="listing-preview-img">
          <button class="image-preview-remove" id="listing-remove-img">&times;</button>
        </div>
        <label class="attach-btn" for="listing-image-upload" style="display:inline-flex;margin-top:8px">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
          Add Photo
        </label>
        <input type="file" id="listing-image-upload" accept="image/*" hidden>
      </div>
      <button class="btn-primary btn-full" id="submit-listing-btn" style="margin-top:8px">List for Sale</button>
    </div>
  `);

  $('#listing-image-upload')?.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (file.size > 5 * 1024 * 1024) { showToast('Image too large (max 5MB)'); return; }
    pendingImage = await compressImage(file, 600, 0.7);
    $('#listing-preview-img').src = pendingImage;
    $('#listing-image-preview').style.display = 'block';
  });

  $('#listing-remove-img')?.addEventListener('click', () => {
    pendingImage = '';
    $('#listing-image-preview').style.display = 'none';
  });

  $('#submit-listing-btn')?.addEventListener('click', async () => {
    const title = $('#listing-title').value.trim();
    const price = parseFloat($('#listing-price').value) || 0;
    const category = $('#listing-category').value;
    const desc = $('#listing-desc').value.trim();

    if (!title || !category) return showToast('Title and category are required');

    const btn = $('#submit-listing-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="inline-spinner"></span>';

    try {
      await db.collection('listings').add({
        title,
        price,
        category,
        description: desc,
        imageURL: pendingImage || '',
        sellerId: state.user.uid,
        sellerName: state.profile.displayName,
        sellerPhoto: state.profile.photoURL || '',
        sellerUni: state.profile.university || '',
        status: 'active',
        createdAt: firebase.firestore.FieldValue.serverTimestamp()
      });
      showToast('Listed!');
      closeModal();
      loadListings();
    } catch (err) {
      showToast('Could not create listing');
      btn.disabled = false;
      btn.textContent = 'List for Sale';
    }
  });
}

// ══════════════════════════════════════════════════════
//  MESSAGES PAGE
// ══════════════════════════════════════════════════════
function renderMessages() {
  const content = $('#app-content');
  content.innerHTML = `
    <div class="messages-page">
      <div class="messages-header">
        <h2>Messages</h2>
      </div>
      <div id="convo-list">
        <div style="text-align:center;padding:32px"><span class="inline-spinner" style="width:32px;height:32px;border-width:3px;color:var(--accent)"></span></div>
      </div>
    </div>
  `;

  listenConversations();
}

function listenConversations() {
  const uid = state.user?.uid;
  if (!uid) return;

  const unsub = db.collection('conversations')
    .where('participants', 'array-contains', uid)
    .orderBy('lastMessageAt', 'desc')
    .limit(30)
    .onSnapshot(snap => {
      state.conversations = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      displayConversations();
    }, err => {
      console.error('Conversations error:', err);
      displayConversations();
    });
  state.unsubscribers.push(unsub);
}

function displayConversations() {
  const container = $('#convo-list');
  if (!container) return;

  if (state.conversations.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">💬</div>
        <h3>No messages yet</h3>
        <p>Start a conversation from someone\'s profile or a marketplace listing</p>
      </div>
    `;
    return;
  }

  const uid = state.user.uid;
  container.innerHTML = `<div class="convo-list">${
    state.conversations.map(c => {
      // Get the other participant's info
      const otherName = c.participantNames?.[uid === c.participants?.[0] ? 1 : 0] || 'User';
      const otherPhoto = c.participantPhotos?.[uid === c.participants?.[0] ? 1 : 0] || '';
      const unread = c.unreadCount?.[uid] || 0;

      return `
        <div class="convo-item" data-convo-id="${c.id}">
          ${avatarHTML(otherName, otherPhoto, 'avatar-md')}
          <div class="convo-info">
            <div class="convo-name">${escapeHTML(otherName)}</div>
            <div class="convo-last-msg">${escapeHTML(c.lastMessage || 'No messages yet')}</div>
          </div>
          <div class="convo-right">
            <div class="convo-time">${timeAgo(c.lastMessageAt)}</div>
            ${unread > 0 ? `<div class="convo-unread">${unread}</div>` : ''}
          </div>
        </div>
      `;
    }).join('')
  }</div>`;

  container.querySelectorAll('.convo-item').forEach(item => {
    item.addEventListener('click', () => openChat(item.dataset.convoId));
  });

  // Update chat badge
  const totalUnread = state.conversations.reduce((sum, c) => sum + (c.unreadCount?.[state.user.uid] || 0), 0);
  const badge = $('#chat-badge');
  if (badge) badge.textContent = totalUnread > 0 ? totalUnread : '';
}

// ─── Start Conversation ───────────────────────────────
async function startConversation(otherUid, otherName, otherPhoto) {
  const uid = state.user.uid;
  if (otherUid === uid) return showToast('That\'s your own profile!');

  // Check if conversation already exists
  try {
    const snap = await db.collection('conversations')
      .where('participants', 'array-contains', uid)
      .get();

    const existing = snap.docs.find(d => {
      const data = d.data();
      return data.participants?.includes(otherUid);
    });

    if (existing) {
      openChat(existing.id);
      return;
    }

    // Create new conversation
    const convoRef = await db.collection('conversations').add({
      participants: [uid, otherUid],
      participantNames: [state.profile.displayName, otherName],
      participantPhotos: [state.profile.photoURL || '', otherPhoto || ''],
      lastMessage: '',
      lastMessageAt: firebase.firestore.FieldValue.serverTimestamp(),
      unreadCount: { [uid]: 0, [otherUid]: 0 },
      createdAt: firebase.firestore.FieldValue.serverTimestamp()
    });

    openChat(convoRef.id);
  } catch (err) {
    console.error('Start conversation error:', err);
    showToast('Could not start conversation');
  }
}

// ─── Chat View ────────────────────────────────────────
let chatUnsub = null;

function openChat(convoId) {
  const convo = state.conversations.find(c => c.id === convoId);
  const uid = state.user.uid;

  // Determine other participant
  let otherName = 'User';
  let otherPhoto = '';
  if (convo) {
    const idx = convo.participants?.[0] === uid ? 1 : 0;
    otherName = convo.participantNames?.[idx] || 'User';
    otherPhoto = convo.participantPhotos?.[idx] || '';
  }

  // Show chat screen
  showScreen('chat-view');
  $('#chat-name').textContent = otherName;
  $('#chat-status').textContent = 'Online';
  const chatAvatarEl = $('#chat-avatar');
  chatAvatarEl.outerHTML = avatarHTML(otherName, otherPhoto, 'avatar-sm');

  // Clear old messages
  const msgContainer = $('#chat-messages');
  msgContainer.innerHTML = '<div style="text-align:center;padding:32px"><span class="inline-spinner" style="width:28px;height:28px;border-width:2px;color:var(--accent)"></span></div>';

  // Mark as read
  db.collection('conversations').doc(convoId).update({
    [`unreadCount.${uid}`]: 0
  }).catch(() => {});

  // Listen to messages
  if (chatUnsub) chatUnsub();
  chatUnsub = db.collection('conversations').doc(convoId)
    .collection('messages')
    .orderBy('createdAt', 'asc')
    .limit(100)
    .onSnapshot(snap => {
      const messages = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      renderChatMessages(messages);
    });

  // Back button
  $('#chat-back-btn').onclick = () => {
    if (chatUnsub) { chatUnsub(); chatUnsub = null; }
    showScreen('app-shell');
  };

  // Send message
  const input = $('#chat-input');
  const sendBtn = $('#chat-send-btn');

  const sendMessage = async () => {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';

    try {
      await db.collection('conversations').doc(convoId).collection('messages').add({
        text,
        senderId: uid,
        senderName: state.profile.displayName,
        createdAt: firebase.firestore.FieldValue.serverTimestamp()
      });

      // Get other user ID for unread count
      const otherUid = convo?.participants?.find(p => p !== uid) || '';

      await db.collection('conversations').doc(convoId).update({
        lastMessage: text,
        lastMessageAt: firebase.firestore.FieldValue.serverTimestamp(),
        [`unreadCount.${otherUid}`]: firebase.firestore.FieldValue.increment(1)
      });
    } catch (err) {
      showToast('Message failed to send');
    }
  };

  sendBtn.onclick = sendMessage;
  input.onkeypress = (e) => { if (e.key === 'Enter') sendMessage(); };

  // Image upload in chat
  $('#chat-img-input').onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const dataURL = await compressImage(file, 500, 0.6);
    try {
      const otherUid = convo?.participants?.find(p => p !== uid) || '';
      await db.collection('conversations').doc(convoId).collection('messages').add({
        text: '',
        imageURL: dataURL,
        senderId: uid,
        senderName: state.profile.displayName,
        createdAt: firebase.firestore.FieldValue.serverTimestamp()
      });
      await db.collection('conversations').doc(convoId).update({
        lastMessage: '📷 Photo',
        lastMessageAt: firebase.firestore.FieldValue.serverTimestamp(),
        [`unreadCount.${otherUid}`]: firebase.firestore.FieldValue.increment(1)
      });
    } catch (err) {
      showToast('Image failed to send');
    }
  };
}

function renderChatMessages(messages) {
  const container = $('#chat-messages');
  if (!container) return;
  const uid = state.user.uid;

  if (messages.length === 0) {
    container.innerHTML = '<div style="text-align:center;padding:32px;color:var(--text-tertiary);font-size:14px">Say hello! 👋</div>';
    return;
  }

  container.innerHTML = messages.map(m => {
    const isSent = m.senderId === uid;
    return `
      <div class="msg-bubble ${isSent ? 'msg-sent' : 'msg-received'}">
        ${m.imageURL ? `<img class="msg-image" src="${m.imageURL}" alt="Photo" onclick="openImageViewer('${m.imageURL}')">` : ''}
        ${m.text ? escapeHTML(m.text) : ''}
        <div class="msg-time">${formatTime(m.createdAt)}</div>
      </div>
    `;
  }).join('');

  // Scroll to bottom
  container.scrollTop = container.scrollHeight;
}

// ══════════════════════════════════════════════════════
//  PROFILE VIEW
// ══════════════════════════════════════════════════════
async function showProfile(uid) {
  showScreen('profile-view');
  const profileContent = $('#profile-content');
  profileContent.innerHTML = '<div style="text-align:center;padding:48px"><span class="inline-spinner" style="width:32px;height:32px;border-width:3px;color:var(--accent)"></span></div>';

  const isOwnProfile = uid === state.user?.uid;

  try {
    let profile;
    if (isOwnProfile && state.profile) {
      profile = state.profile;
    } else {
      const doc = await db.collection('users').doc(uid).get();
      if (!doc.exists) { profileContent.innerHTML = '<div class="empty-state"><h3>User not found</h3></div>'; return; }
      profile = { id: doc.id, ...doc.data() };
    }

    // Count posts for this user
    const postsSnap = await db.collection('posts')
      .where('authorId', '==', uid)
      .orderBy('createdAt', 'desc')
      .limit(20)
      .get();
    const userPosts = postsSnap.docs.map(d => ({ id: d.id, ...d.data() }));

    $('#profile-top-name').textContent = profile.displayName || '';

    profileContent.innerHTML = `
      <div class="profile-banner"></div>
      <div class="profile-avatar-wrapper">
        <div class="profile-avatar-large" style="background:${getAvatarColor(profile.displayName)}">
          ${profile.photoURL
            ? `<img src="${profile.photoURL}" alt="" onerror="this.remove()">`
            : getInitials(profile.displayName)
          }
        </div>
      </div>
      <div class="profile-info">
        <div class="profile-name">${escapeHTML(profile.displayName)}</div>
        <div class="profile-uni">${escapeHTML(profile.university || '')} ${profile.year ? '· ' + escapeHTML(profile.year) : ''}</div>
        ${profile.major ? `<div class="profile-major-pill">${escapeHTML(profile.major)}</div>` : ''}
        ${profile.bio ? `<div class="profile-bio">${escapeHTML(profile.bio)}</div>` : ''}
        <div class="profile-stats-row">
          <div class="profile-stat"><div class="profile-stat-num">${userPosts.length}</div><div class="profile-stat-label">Posts</div></div>
          <div class="profile-stat"><div class="profile-stat-num">${profile.friendsCount || 0}</div><div class="profile-stat-label">Friends</div></div>
        </div>
        <div class="profile-actions">
          ${isOwnProfile
            ? `<button class="btn-primary" id="edit-profile-btn">Edit Profile</button>
               <button class="btn-outline" id="logout-btn" style="color:var(--red);border-color:var(--red)">Log Out</button>`
            : `<button class="btn-primary" id="dm-profile-btn">Message</button>`
          }
        </div>
      </div>
      <div class="profile-posts-header">${isOwnProfile ? 'Your' : escapeHTML(profile.firstName || 'Their')} Posts</div>
      <div id="profile-posts-list">
        ${userPosts.length === 0
          ? '<div class="empty-state" style="padding:24px"><p style="color:var(--text-tertiary)">No posts yet</p></div>'
          : userPosts.map(p => postCardHTML(p)).join('')
        }
      </div>
    `;

    // Attach listeners
    if (isOwnProfile) {
      $('#edit-profile-btn')?.addEventListener('click', openEditProfileModal);
      $('#logout-btn')?.addEventListener('click', () => {
        auth.signOut();
        showScreen('auth-screen');
      });
    } else {
      $('#dm-profile-btn')?.addEventListener('click', () => {
        showScreen('app-shell');
        startConversation(uid, profile.displayName, profile.photoURL || '');
      });
    }

    // Post interactions within profile
    attachPostListeners();

  } catch (err) {
    console.error('Profile error:', err);
    profileContent.innerHTML = '<div class="empty-state"><h3>Could not load profile</h3></div>';
  }

  // Back button
  $('#profile-back-btn').onclick = () => showScreen('app-shell');
}

// ─── Edit Profile ─────────────────────────────────────
function openEditProfileModal() {
  const p = state.profile;
  let newPhotoData = '';

  openModal(`
    <div class="edit-profile-form">
      <h2>Edit Profile</h2>
      <div class="avatar-upload">
        <div class="avatar-upload-preview" id="edit-avatar-preview" style="background:${getAvatarColor(p.displayName)}">
          ${p.photoURL ? `<img src="${p.photoURL}" alt="">` : getInitials(p.displayName)}
        </div>
        <div>
          <label class="btn-outline" for="edit-avatar-upload" style="cursor:pointer;display:inline-block">Change Photo</label>
          <input type="file" id="edit-avatar-upload" accept="image/*" hidden>
          <div style="font-size:12px;color:var(--text-tertiary);margin-top:4px">Max 2MB, will be compressed</div>
        </div>
      </div>
      <div class="form-group">
        <label>Display Name</label>
        <input type="text" id="edit-name" value="${escapeHTML(p.displayName || '')}" maxlength="50">
      </div>
      <div class="form-group">
        <label>Bio</label>
        <textarea id="edit-bio" rows="3" maxlength="200" style="resize:vertical">${escapeHTML(p.bio || '')}</textarea>
      </div>
      <div class="form-group">
        <label>University</label>
        <input type="text" id="edit-uni" value="${escapeHTML(p.university || '')}">
      </div>
      <div class="form-group">
        <label>Major</label>
        <input type="text" id="edit-major" value="${escapeHTML(p.major || '')}">
      </div>
      <button class="btn-primary btn-full" id="save-profile-btn" style="margin-top:8px">Save Changes</button>
    </div>
  `);

  $('#edit-avatar-upload')?.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (file.size > 3 * 1024 * 1024) { showToast('Image too large'); return; }
    newPhotoData = await compressAvatar(file);
    $('#edit-avatar-preview').innerHTML = `<img src="${newPhotoData}" alt="">`;
  });

  $('#save-profile-btn')?.addEventListener('click', async () => {
    const btn = $('#save-profile-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="inline-spinner"></span>';

    const updates = {
      displayName: $('#edit-name').value.trim() || p.displayName,
      bio: $('#edit-bio').value.trim(),
      university: $('#edit-uni').value.trim(),
      major: $('#edit-major').value.trim(),
    };

    if (newPhotoData) {
      updates.photoURL = newPhotoData;
    }

    try {
      await db.collection('users').doc(state.user.uid).update(updates);
      // Update local state
      Object.assign(state.profile, updates);
      updateHeaderAvatar();
      showToast('Profile updated!');
      closeModal();
      showProfile(state.user.uid);
    } catch (err) {
      showToast('Could not save profile');
      btn.disabled = false;
      btn.textContent = 'Save Changes';
    }
  });
}

// ══════════════════════════════════════════════════════
//  MODAL & IMAGE VIEWER
// ══════════════════════════════════════════════════════
function openModal(html) {
  const overlay = $('#modal-overlay');
  const body = $('#modal-body');
  body.innerHTML = html;
  overlay.style.display = 'flex';
  overlay.onclick = (e) => { if (e.target === overlay) closeModal(); };
}

function closeModal() {
  const overlay = $('#modal-overlay');
  overlay.style.display = 'none';
  $('#modal-body').innerHTML = '';
}

function openImageViewer(src) {
  const viewer = $('#image-viewer');
  $('#image-viewer-img').src = src;
  viewer.style.display = 'flex';
}

function initImageViewer() {
  $('#image-viewer-close')?.addEventListener('click', () => {
    $('#image-viewer').style.display = 'none';
  });
  $('#image-viewer')?.addEventListener('click', (e) => {
    if (e.target === $('#image-viewer')) $('#image-viewer').style.display = 'none';
  });
}

// ══════════════════════════════════════════════════════
//  INITIALIZATION
// ══════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initImageViewer();
  initAuth();
  listenUserCount();

  // Hide loading screen after a moment
  setTimeout(() => {
    const loading = $('#loading-screen');
    if (loading.classList.contains('active') && !$('#app-shell').classList.contains('active')) {
      // Auth hasn't resolved yet, show auth screen
      // (auth.onAuthStateChanged will show correct screen)
    }
  }, 3000);

  // Fallback: if auth doesn't respond in 4s, show auth screen
  setTimeout(() => {
    if ($('#loading-screen').classList.contains('active')) {
      showScreen('auth-screen');
    }
  }, 4000);
});

// Make functions available globally for inline onclick handlers
window.submitComment = submitComment;
window.closeModal = closeModal;
window.openImageViewer = openImageViewer;
