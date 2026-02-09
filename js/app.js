// ============================================
// UNINO — Main Application Engine
// Full SPA: Home, Around, Academic, Hustle, Chat, Profile
// ============================================

const App = {
  // ============================
  // STATE
  // ============================
  state: {
    currentPage: 'home',
    user: null,
    privacyMode: 'online',
    theme: 'dark',
    aroundView: 'map',
    aroundFilter: 'all',
    hustleCategory: 'all',
    hustleSearch: '',
    academicTab: 'courses',
    messageTab: 'dm',
    activeChat: null,
    notificationsOpen: false,
    savedListings: new Set(),
    rsvpEvents: new Set(['e6']), // User already RSVP'd to their own event
    friendRequests: {},
  },

  // ============================
  // INITIALIZATION
  // ============================
  init() {
    this.state.user = MockData.currentUser;
    const saved = localStorage.getItem('unino-theme');
    if (saved) {
      this.state.theme = saved;
      document.documentElement.setAttribute('data-theme', saved);
    }
    // Pre-fill friend request states from connections
    MockData.connections.forEach(c => {
      this.state.friendRequests[c.userId] = c.status;
    });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') { this.closeModal(); this.closeChat(); this.closeNotifications(); }
    });
  },

  // ============================
  // AUTH
  // ============================
  login() {
    const btn = document.querySelector('#auth-screen .btn-primary');
    if (btn) { btn.textContent = 'Logging in…'; btn.disabled = true; }
    setTimeout(() => {
      document.getElementById('auth-screen').classList.remove('active');
      document.getElementById('app-shell').classList.add('active');
      this.navigate('home');
      this.renderNotifications();
      this.showToast('Welcome back, Alex! 👋');
    }, 700);
  },
  showSignup() {
    document.getElementById('login-form').classList.remove('active');
    document.getElementById('signup-form').classList.add('active');
  },
  showLogin() {
    document.getElementById('signup-form').classList.remove('active');
    document.getElementById('login-form').classList.add('active');
  },
  logout() {
    document.getElementById('app-shell').classList.remove('active');
    document.getElementById('auth-screen').classList.add('active');
    const b = document.querySelector('#login-form .btn-primary');
    if (b) { b.textContent = 'Log In'; b.disabled = false; }
  },

  // ============================
  // NAVIGATION
  // ============================
  navigate(page) {
    this.state.currentPage = page;
    this.closeNotifications();
    document.querySelectorAll('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.page === page));
    const el = document.getElementById('app-content');
    el.style.opacity = '0';
    el.style.transform = 'translateY(8px)';
    setTimeout(() => {
      switch (page) {
        case 'home':     el.innerHTML = this.renderHome(); break;
        case 'around':   el.innerHTML = this.renderAround(); break;
        case 'academic': el.innerHTML = this.renderAcademic(); break;
        case 'hustle':   el.innerHTML = this.renderHustle(); break;
        case 'messages': el.innerHTML = this.renderMessages(); break;
        case 'profile':  el.innerHTML = this.renderProfile(); break;
      }
      el.scrollTop = 0;
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    }, 120);
  },

  // ============================
  // HELPERS
  // ============================
  getGreeting() {
    const h = new Date().getHours();
    if (h < 12) return 'Good morning! Ready to conquer some classes?';
    if (h < 17) return 'Good afternoon! How\'s the grind going?';
    return 'Good evening! Time for some study or chill?';
  },
  formatTime(iso) {
    return new Date(iso).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
  },
  formatDate(iso) {
    return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  },
  relativeTime(iso) {
    const diff = Date.now() - new Date(iso).getTime();
    const m = Math.floor(diff / 60000);
    if (m < 1) return 'Just now';
    if (m < 60) return m + 'm ago';
    const h = Math.floor(m / 60);
    if (h < 24) return h + 'h ago';
    return Math.floor(h / 24) + 'd ago';
  },
  categoryIcon(cat) {
    const m = { textbook: '📚', service: '🛠️', furniture: '🪑', electronics: '📱', other: '📦' };
    return m[cat] || '📦';
  },
  courseIcon(code) {
    if (code.startsWith('CS'))   return { emoji: '💻', cls: 'cs' };
    if (code.startsWith('MATH')) return { emoji: '📐', cls: 'math' };
    if (code.startsWith('ENG'))  return { emoji: '✏️', cls: 'eng' };
    if (code.startsWith('PHIL')) return { emoji: '🤔', cls: 'phil' };
    return { emoji: '📖', cls: 'cs' };
  },

  // ============================
  // NOTIFICATIONS
  // ============================
  toggleNotifications() {
    this.state.notificationsOpen = !this.state.notificationsOpen;
    const panel = document.getElementById('notifications-panel');
    panel.classList.toggle('active', this.state.notificationsOpen);
    if (this.state.notificationsOpen) this.renderNotifications();
  },
  closeNotifications() {
    this.state.notificationsOpen = false;
    const p = document.getElementById('notifications-panel');
    if (p) p.classList.remove('active');
  },
  renderNotifications() {
    const icons = { friend_request: '👤', message: '💬', event: '📅', achievement: '🏆', marketplace: '🛍️' };
    const cls   = { friend_request: 'friend', message: 'message', event: 'event', achievement: 'achievement', marketplace: 'marketplace' };
    const list = document.getElementById('notif-list');
    if (!list) return;
    list.innerHTML = MockData.notifications.map(n => `
      <div class="notif-item ${n.read ? '' : 'unread'}">
        <div class="notif-icon ${cls[n.type]}">${icons[n.type]}</div>
        <div class="notif-content">
          <div class="notif-text"><strong>${n.from}</strong> ${n.message}</div>
          <div class="notif-time">${n.time}</div>
        </div>
      </div>
    `).join('');
  },
  clearNotifications() {
    MockData.notifications.forEach(n => n.read = true);
    document.getElementById('notif-badge').style.display = 'none';
    this.renderNotifications();
    this.showToast('All notifications cleared ✓');
  },

  // ============================
  // TOAST
  // ============================
  showToast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('active');
    setTimeout(() => t.classList.remove('active'), 2800);
  },

  // ============================
  // PRIVACY MODE
  // ============================
  setPrivacyMode(mode) {
    this.state.privacyMode = mode;
    this.state.user.privacyMode = mode;
    if (this.state.currentPage === 'home') this.navigate('home');
    else if (this.state.currentPage === 'around') this.navigate('around');
    const labels = { online: 'You\'re visible to nearby students', study: 'Study Mode — only study buddies see you', offline: 'You\'re hidden from the map' };
    this.showToast(labels[mode]);
  },

  // ============================
  // HOME PAGE
  // ============================
  renderHome() {
    const u = this.state.user;
    const online = MockData.users.filter(x => x.privacyMode !== 'offline').length;
    const todayEv = MockData.events.filter(e => {
      const d = new Date(e.startTime);
      return d.getMonth() === 1 && d.getDate() === 10;
    }).length;
    const unread = MockData.conversations.reduce((s, c) => s + c.unread, 0);

    return `
      <div class="status-bar">
        <div class="status-toggle">
          <button class="status-option ${this.state.privacyMode === 'online' ? 'active' : ''}" onclick="App.setPrivacyMode('online')">🟢 Online</button>
          <button class="status-option ${this.state.privacyMode === 'study' ? 'active study' : ''}" onclick="App.setPrivacyMode('study')">📖 Study</button>
          <button class="status-option ${this.state.privacyMode === 'offline' ? 'active offline' : ''}" onclick="App.setPrivacyMode('offline')">⚫ Off</button>
        </div>
        <div class="streak-badge">🔥 ${u.studyStreakDays} days</div>
      </div>

      <div class="page-section">
        <div class="welcome-banner animate-in">
          <div class="welcome-left">
            <h2>Hey, ${u.firstName}! 👋</h2>
            <p>${this.getGreeting()}</p>
          </div>
          <div class="karma-pill"><span class="karma-num">${u.karmaPoints}</span><span class="karma-lbl">Karma ✨</span></div>
        </div>
      </div>

      <div class="quick-stats">
        <div class="stat-card animate-in" onclick="App.navigate('around')" style="cursor:pointer">
          <div class="stat-number">${online}</div><div class="stat-label">Nearby</div>
        </div>
        <div class="stat-card animate-in" onclick="App.navigate('messages')" style="cursor:pointer">
          <div class="stat-number">${unread}</div><div class="stat-label">Unread</div>
        </div>
        <div class="stat-card animate-in" style="cursor:pointer">
          <div class="stat-number">${todayEv}</div><div class="stat-label">Events</div>
        </div>
      </div>

      <div class="page-section">
        <div class="section-header">
          <div><h3 class="section-title">Campus Pulse 🎯</h3><p class="section-subtitle">Upcoming events near you</p></div>
          <button class="text-btn" onclick="App.showAllEvents()">See All</button>
        </div>
        <div class="events-scroll">${MockData.events.map(e => this.renderEventCard(e)).join('')}</div>
      </div>

      <div class="page-section">
        <div class="section-header">
          <div><h3 class="section-title">People You May Know 👥</h3><p class="section-subtitle">Based on your courses</p></div>
        </div>
        <div class="suggested-scroll">${MockData.users.filter(x => x.privacyMode !== 'offline').slice(0, 6).map(x => this.renderSuggestedCard(x)).join('')}</div>
      </div>

      <div class="page-section">
        <div class="section-header"><h3 class="section-title">Activity Feed 📡</h3></div>
        ${this.renderActivityFeed()}
      </div>
    `;
  },

  renderEventCard(ev) {
    const t = this.formatTime(ev.startTime);
    const going = this.state.rsvpEvents.has(ev.id);
    return `
      <div class="event-card animate-in" onclick="App.openEventDetail('${ev.id}')">
        <span class="event-type-badge ${ev.type}">${ev.type.replace('_', ' ')}</span>
        <h4>${ev.title}</h4>
        <div class="event-meta">
          <div class="event-meta-row">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            ${t}
          </div>
          <div class="event-meta-row">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/><circle cx="12" cy="10" r="3"/></svg>
            ${ev.location}
          </div>
        </div>
        <div class="event-footer">
          <div class="event-attendees-mini">
            <div class="mini-avatar">${ev.creatorAvatar}</div>
            <div class="mini-avatar">+${Math.max(ev.attendees - 1, 0)}</div>
            <span class="attendee-count">${ev.attendees}/${ev.maxAttendees}</span>
          </div>
          <button class="rsvp-btn ${going ? 'going' : ''}" onclick="event.stopPropagation(); App.toggleRSVP('${ev.id}')">
            ${going ? '✓ Going' : 'RSVP'}
          </button>
        </div>
      </div>`;
  },

  renderSuggestedCard(u) {
    const st = this.state.friendRequests[u.id];
    let action;
    if (st === 'accepted') action = '<span class="suggested-badge connected">✓ Friends</span>';
    else if (st === 'pending') action = '<span class="suggested-badge pending">Pending</span>';
    else action = `<button class="btn-outline-accent btn-small" onclick="event.stopPropagation(); App.sendFriendRequest('${u.id}')">+ Add</button>`;
    return `
      <div class="suggested-card animate-in" onclick="App.openUserProfile('${u.id}')">
        <div class="user-avatar ${u.privacyMode === 'online' ? 'gradient' : ''}">${u.avatar}<span class="avatar-status ${u.privacyMode}"></span></div>
        <div class="suggested-name">${u.firstName} ${u.lastName.charAt(0)}.</div>
        <div class="suggested-major">${u.major}</div>
        ${action}
      </div>`;
  },

  renderActivityFeed() {
    const acts = [
      { icon: '🏆', text: '<strong>Priya Rao</strong> earned "Study Streak 21" badge', time: '2h ago', c: 'success' },
      { icon: '📚', text: '<strong>David Chen</strong> shared notes in MATH301', time: '3h ago', c: 'info' },
      { icon: '🛍️', text: '<strong>Emma Lee</strong> listed "Logo Design Package"', time: '5h ago', c: 'accent' },
      { icon: '📅', text: '<strong>Mike Thompson</strong> created "Startup Pitch Night"', time: '6h ago', c: 'warning' },
      { icon: '🤝', text: '<strong>Nina Patel</strong> and <strong>David Chen</strong> connected', time: '8h ago', c: 'success' },
    ];
    return `<div class="activity-feed">${acts.map((a, i) => `
      <div class="activity-item animate-in" style="animation-delay:${i * 0.05}s">
        <div class="activity-icon ${a.c}">${a.icon}</div>
        <div class="activity-content"><p>${a.text}</p><span class="activity-time">${a.time}</span></div>
      </div>`).join('')}</div>`;
  },

  // ============================
  // WHO'S AROUND
  // ============================
  renderAround() {
    const users = MockData.users.filter(u => {
      if (u.privacyMode === 'offline') return false;
      const f = this.state.aroundFilter;
      if (f === 'all') return true;
      if (f === 'buddy') return u.status === 'Looking for Study Buddy';
      if (f === 'free') return u.status === 'Free Now';
      if (f === 'cs') return u.major === 'Computer Science';
      return true;
    });

    return `
      <div class="around-header">
        <div class="flex items-center justify-between">
          <h3 class="section-title">Who's Around 📍</h3>
          <span class="privacy-pill ${this.state.privacyMode}">${this.state.privacyMode === 'online' ? '🟢 Visible' : this.state.privacyMode === 'study' ? '📖 Study' : '⚫ Hidden'}</span>
        </div>
        <div class="view-toggle">
          <button class="view-toggle-btn ${this.state.aroundView === 'map' ? 'active' : ''}" onclick="App.setAroundView('map')">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/></svg> Map
          </button>
          <button class="view-toggle-btn ${this.state.aroundView === 'list' ? 'active' : ''}" onclick="App.setAroundView('list')">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg> List
          </button>
        </div>
        <div class="filter-chips">
          ${['all','buddy','free','cs'].map(f => {
            const labels = { all: '👥 All', buddy: '🔍 Study Buddy', free: '✅ Free Now', cs: '💻 CS Major' };
            return `<button class="filter-chip ${this.state.aroundFilter === f ? 'active' : ''}" onclick="App.setAroundFilter('${f}')">${labels[f]}</button>`;
          }).join('')}
        </div>
      </div>
      ${this.state.aroundView === 'map' ? this.renderCampusMap(users) : ''}
      <div class="user-list" ${this.state.aroundView === 'map' ? 'style="padding-top:0"' : ''}>
        ${this.state.aroundView === 'map' ? `<div class="section-header" style="padding:16px 0 8px"><h3 class="section-title" style="font-size:16px">Closest to You</h3></div>` : ''}
        ${(this.state.aroundView === 'map' ? users.sort((a, b) => a.distance - b.distance).slice(0, 4) : users.sort((a, b) => a.distance - b.distance)).map(u => this.renderUserCard(u)).join('')}
        ${users.length === 0 ? '<div class="empty-state"><div class="empty-state-icon">👻</div><h3>Nobody matching filters</h3><p>Try broadening your search</p></div>' : ''}
      </div>`;
  },

  renderCampusMap(users) {
    const buildings = [
      { name: '📚 Library', x: 32, y: 12 },
      { name: '☕ Coffee Shop', x: 68, y: 14 },
      { name: '🏛️ Student Center', x: 50, y: 32 },
      { name: '💻 CS Building', x: 18, y: 48 },
      { name: '⚙️ Engineering Lab', x: 14, y: 72 },
      { name: '📐 Math Lab', x: 44, y: 62 },
      { name: '🎨 Art Building', x: 75, y: 55 },
      { name: '🧠 Psych Building', x: 68, y: 78 },
      { name: '🧪 Chem Lab', x: 30, y: 82 },
      { name: '🌳 Quad', x: 50, y: 46 },
    ];
    const positions = {
      u2: { x: 52, y: 28 }, u3: { x: 35, y: 15 }, u4: { x: 70, y: 17 },
      u5: { x: 70, y: 75 }, u6: { x: 77, y: 52 }, u7: { x: 46, y: 59 },
      u8: { x: 16, y: 69 }, u10: { x: 32, y: 79 },
    };
    const markers = users.map(u => {
      const p = positions[u.id];
      if (!p) return '';
      const cls = u.status === 'Looking for Study Buddy' ? 'buddy' : u.privacyMode === 'study' ? 'study' : 'online';
      return `<div class="map-marker ${cls}" style="left:${p.x}%;top:${p.y}%" onclick="App.openUserProfile('${u.id}')" title="${u.firstName} – ${u.status}">${u.avatar}</div>`;
    }).join('');

    return `
      <div class="map-container">
        <div class="map-grid"></div>
        <div class="map-paths"></div>
        ${buildings.map(b => `<div class="map-building" style="left:${b.x}%;top:${b.y}%">${b.name}</div>`).join('')}
        <div class="map-marker-you" style="left:48%;top:44%" title="You are here"></div>
        ${markers}
      </div>
      <div style="text-align:center;padding:8px;font-size:11px;color:var(--text-tertiary)">
        🔵 You&nbsp;&nbsp;&nbsp;🟣 Online&nbsp;&nbsp;&nbsp;🟡 Study&nbsp;&nbsp;&nbsp;🟢 Buddy
      </div>`;
  },

  renderUserCard(u) {
    const cls = u.status === 'Looking for Study Buddy' ? 'buddy' : u.status === 'Study Mode' ? 'study' : 'free';
    const lbl = u.status === 'Looking for Study Buddy' ? '🔍 Buddy' : u.status === 'Study Mode' ? '📖 Study' : '✅ Free';
    return `
      <div class="user-card animate-in" onclick="App.openUserProfile('${u.id}')">
        <div class="user-avatar">${u.avatar}<span class="avatar-status ${u.privacyMode}"></span></div>
        <div class="user-info">
          <div class="user-name">${u.firstName} ${u.lastName}</div>
          <div class="user-detail">${u.major} · ${u.location || 'Unknown'}</div>
          <div class="user-meta"><span class="user-tag ${cls}">${lbl}</span></div>
        </div>
        <div class="user-distance">${u.distance}m</div>
      </div>`;
  },

  setAroundView(v) { this.state.aroundView = v; this.navigate('around'); },
  setAroundFilter(f) { this.state.aroundFilter = f; this.navigate('around'); },

  // ============================
  // ACADEMIC HUB
  // ============================
  renderAcademic() {
    const tab = this.state.academicTab;
    return `
      <div class="tab-bar">
        <button class="tab-btn ${tab === 'courses' ? 'active' : ''}" onclick="App.setAcademicTab('courses')">📖 Courses</button>
        <button class="tab-btn ${tab === 'circles' ? 'active' : ''}" onclick="App.setAcademicTab('circles')">⭕ Circles</button>
        <button class="tab-btn ${tab === 'schedule' ? 'active' : ''}" onclick="App.setAcademicTab('schedule')">📅 Schedule</button>
      </div>
      <div class="page-section">
        ${tab === 'courses' ? this.renderCourses() : tab === 'circles' ? this.renderCircles() : this.renderSchedule()}
      </div>`;
  },

  renderCourses() {
    return `
      <div class="section-header"><h3 class="section-title">My Courses</h3><span class="text-btn">Spring 2026</span></div>
      <div style="display:flex;flex-direction:column;gap:10px">
        ${MockData.courses.map(c => {
          const ic = this.courseIcon(c.code);
          return `
            <div class="course-card animate-in" onclick="App.openCourseRoom('${c.id}')">
              <div class="course-icon ${ic.cls}">${ic.emoji}</div>
              <div class="course-info">
                <div class="course-code">${c.code}</div>
                <div class="course-name">${c.name}</div>
                <div class="course-meta">${c.instructor} · ${c.members} students</div>
              </div>
              ${c.unread > 0 ? `<div class="course-unread">${c.unread}</div>` : ''}
            </div>`;
        }).join('')}
      </div>`;
  },

  renderCircles() {
    return `
      <div class="section-header">
        <div><h3 class="section-title">Assignment Circles</h3><p class="section-subtitle">Temporary study groups</p></div>
        <button class="btn-outline-accent btn-small" onclick="App.showCreateCircle()">+ New</button>
      </div>
      <div style="display:flex;flex-direction:column;gap:10px">
        ${MockData.assignmentCircles.map(ac => {
          const members = ac.members.map(mid => {
            const mu = mid === 'u1' ? MockData.currentUser : MockData.users.find(x => x.id === mid);
            return mu ? (mu.avatar || mu.firstName.charAt(0) + mu.lastName.charAt(0)) : '?';
          });
          const daysLeft = Math.max(0, Math.floor((new Date(ac.dueDate) - new Date('2026-02-09')) / 86400000));
          return `
            <div class="circle-card animate-in" onclick="App.openCircleChat('${ac.id}')">
              <div class="circle-header">
                <span class="circle-course-badge">${ac.courseCode}</span>
                <span class="circle-due">${daysLeft === 0 ? '⚠️ Due today' : `📅 ${daysLeft}d left`}</span>
              </div>
              <h4>${ac.name}</h4>
              <div class="circle-footer">
                <div class="circle-members">${members.map(m => `<div class="mini-avatar">${m}</div>`).join('')}</div>
                <span class="circle-messages">💬 ${ac.messages} messages</span>
              </div>
            </div>`;
        }).join('')}
      </div>`;
  },

  renderSchedule() {
    const slots = [
      { time: '9:00 – 10:30 AM', course: 'CS201 — Data Structures', room: 'CS Building Room 101', friends: ['Priya R.', 'David C.'] },
      { time: '11:00 AM – 12:30 PM', course: 'MATH301 — Linear Algebra', room: 'Math Lab 204', friends: ['David C.', 'Nina P.'] },
      { time: '2:00 – 3:30 PM', course: 'CS301 — Operating Systems', room: 'CS Building Room 303', friends: ['Priya R.'] },
      { time: '4:00 – 5:00 PM', course: 'PHIL101 — Intro to Philosophy', room: 'Humanities 110', friends: [] },
    ];
    const freeOverlaps = [
      { time: '12:30 – 2:00 PM', label: 'Lunch Break', friends: ['Sarah J.', 'Priya R.', 'David C.'] },
      { time: '5:00 – 6:00 PM', label: 'Free Hour', friends: ['Nina P.', 'James W.'] },
    ];
    return `
      <div class="section-header"><h3 class="section-title">Today's Schedule</h3><span class="text-btn">Feb 10</span></div>
      <div style="display:flex;flex-direction:column;gap:10px">
        ${slots.map(s => `
          <div class="schedule-card animate-in">
            <div class="schedule-time">${s.time}</div>
            <h4>${s.course}</h4>
            <div style="font-size:12px;color:var(--text-tertiary);margin-top:2px">📍 ${s.room}</div>
            ${s.friends.length ? `<div class="schedule-friends">👥 ${s.friends.join(', ')} also enrolled</div>` : ''}
          </div>`).join('')}
      </div>

      <div class="section-header" style="margin-top:24px">
        <div><h3 class="section-title">Free Time Overlaps ☕</h3><p class="section-subtitle">Friends free at the same time</p></div>
      </div>
      <div style="display:flex;flex-direction:column;gap:10px">
        ${freeOverlaps.map(f => `
          <div class="schedule-card overlap animate-in">
            <div class="schedule-time" style="color:var(--success)">${f.time}</div>
            <h4>${f.label}</h4>
            <div class="schedule-friends">👥 ${f.friends.join(', ')}</div>
            <button class="btn-outline-accent btn-small" style="margin-top:8px" onclick="App.showToast('Invite sent! 🎉')">Invite to hang</button>
          </div>`).join('')}
      </div>`;
  },

  setAcademicTab(t) { this.state.academicTab = t; this.navigate('academic'); },

  // ============================
  // THE HUSTLE (MARKETPLACE)
  // ============================
  renderHustle() {
    const cats = ['all', 'textbook', 'service', 'furniture', 'electronics', 'other'];
    const catLabels = { all: '🔥 All', textbook: '📚 Books', service: '🛠️ Services', furniture: '🪑 Furniture', electronics: '📱 Tech', other: '📦 Other' };
    let listings = MockData.listings;
    if (this.state.hustleCategory !== 'all') listings = listings.filter(l => l.category === this.state.hustleCategory);
    if (this.state.hustleSearch) {
      const q = this.state.hustleSearch.toLowerCase();
      listings = listings.filter(l => l.title.toLowerCase().includes(q) || l.description.toLowerCase().includes(q));
    }

    return `
      <div class="hustle-header">
        <div class="section-header" style="margin-bottom:0"><h3 class="section-title">The Hustle 💰</h3></div>
        <div class="search-bar">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          <input type="text" placeholder="Search listings…" value="${this.state.hustleSearch}" oninput="App.hustleSearch(this.value)">
        </div>
        <div class="category-tabs">
          ${cats.map(c => `<button class="category-tab ${this.state.hustleCategory === c ? 'active' : ''}" onclick="App.setHustleCategory('${c}')">${catLabels[c]}</button>`).join('')}
        </div>
      </div>

      <div class="listing-grid">
        ${listings.map(l => `
          <div class="listing-card animate-in" onclick="App.openListingDetail('${l.id}')">
            <div class="listing-image"><span class="listing-category-icon">${this.categoryIcon(l.category)}</span>
              ${this.state.savedListings.has(l.id) ? '<div class="listing-saved-badge">♥</div>' : ''}
            </div>
            <div class="listing-body">
              <div class="listing-title">${l.title}</div>
              <div class="listing-price">$${l.price.toFixed(2)}</div>
              <div class="listing-seller">
                <div class="listing-seller-avatar">${l.sellerAvatar}</div>
                <span class="listing-seller-name">${l.sellerName}</span>
                <span class="listing-seller-rating">★ ${l.sellerRating}</span>
              </div>
            </div>
          </div>`).join('')}
        ${listings.length === 0 ? '<div class="empty-state" style="grid-column:1/-1"><div class="empty-state-icon">🔍</div><h3>No listings found</h3><p>Try different keywords or category</p></div>' : ''}
      </div>
      <button class="fab" onclick="App.openCreateListing()" aria-label="Create listing">+</button>`;
  },

  setHustleCategory(c) { this.state.hustleCategory = c; this.navigate('hustle'); },
  hustleSearch(v) { this.state.hustleSearch = v; this.navigate('hustle'); },

  // ============================
  // MESSAGES
  // ============================
  renderMessages() {
    const tab = this.state.messageTab;
    const convos = MockData.conversations.filter(c => tab === 'dm' ? c.type === 'dm' : c.type !== 'dm');

    return `
      <div class="messages-header">
        <h3 class="section-title">Messages 💬</h3>
        <div class="msg-tabs">
          <button class="msg-tab ${tab === 'dm' ? 'active' : ''}" onclick="App.setMessageTab('dm')">Direct</button>
          <button class="msg-tab ${tab === 'course' ? 'active' : ''}" onclick="App.setMessageTab('course')">Rooms</button>
        </div>
      </div>
      <div class="conversation-list">
        ${convos.map(c => {
          const isCourse = c.type !== 'dm';
          return `
            <div class="conversation-item animate-in" onclick="App.openChat('${c.id}')">
              <div class="conv-avatar ${isCourse ? 'course' : ''}">${c.avatar}</div>
              <div class="conv-content">
                <div class="conv-name">${c.name}</div>
                <div class="conv-last-msg">${c.lastMessage}</div>
              </div>
              <div class="conv-meta">
                <span class="conv-time">${this.relativeTime(c.lastMessageTime)}</span>
                ${c.unread > 0 ? `<span class="conv-unread">${c.unread}</span>` : ''}
              </div>
            </div>`;
        }).join('')}
        ${convos.length === 0 ? '<div class="empty-state"><div class="empty-state-icon">💬</div><h3>No conversations yet</h3><p>Start chatting with classmates!</p></div>' : ''}
      </div>`;
  },

  setMessageTab(t) { this.state.messageTab = t; this.navigate('messages'); },

  // ============================
  // CHAT VIEW
  // ============================
  openChat(convId) {
    const conv = MockData.conversations.find(c => c.id === convId);
    if (!conv) return;
    this.state.activeChat = conv;
    conv.unread = 0;
    const badge = document.getElementById('msg-badge');
    const total = MockData.conversations.reduce((s, c) => s + c.unread, 0);
    if (badge) { badge.textContent = total; badge.style.display = total > 0 ? 'flex' : 'none'; }

    document.getElementById('chat-avatar').textContent = conv.avatar;
    document.getElementById('chat-name').textContent = conv.name;
    document.getElementById('chat-status').textContent = conv.type === 'dm' ? 'Online' : `${conv.participantIds.length} members`;

    const messagesEl = document.getElementById('chat-messages');
    messagesEl.innerHTML = conv.messages.map(m => {
      const isMine = m.senderId === 'u1';
      return `
        <div class="message-bubble ${isMine ? 'sent' : 'received'}">
          ${!isMine && conv.type !== 'dm' ? `<div class="message-sender">${m.senderName}</div>` : ''}
          ${m.content}
          <div class="message-time">${this.formatTime(m.time)}</div>
        </div>`;
    }).join('');

    document.getElementById('chat-view').classList.add('active');
    document.getElementById('chat-input').value = '';
    setTimeout(() => { messagesEl.scrollTop = messagesEl.scrollHeight; }, 50);
  },
  closeChat() {
    document.getElementById('chat-view').classList.remove('active');
    this.state.activeChat = null;
  },
  sendMessage() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text || !this.state.activeChat) return;
    const now = new Date().toISOString();
    const msg = { id: 'm' + Date.now(), senderId: 'u1', senderName: 'You', content: text, time: now };
    this.state.activeChat.messages.push(msg);
    this.state.activeChat.lastMessage = text;
    this.state.activeChat.lastMessageTime = now;

    const el = document.getElementById('chat-messages');
    el.innerHTML += `
      <div class="message-bubble sent" style="animation:fadeInUp 0.2s ease">
        ${text}
        <div class="message-time">${this.formatTime(now)}</div>
      </div>`;
    input.value = '';
    el.scrollTop = el.scrollHeight;

    // Simulate typing + reply
    setTimeout(() => {
      const replies = [
        'That sounds great! 🙌', 'I agree, let\'s do it!', 'Sure, I\'ll be there 👍',
        'Haha nice 😂', 'Sounds good to me!', 'Let me check and get back to you',
        'Perfect, thanks!', 'Oh interesting, tell me more!', 'Can\'t wait! 🎉',
      ];
      const reply = replies[Math.floor(Math.random() * replies.length)];
      const replyTime = new Date().toISOString();
      const replyMsg = { id: 'm' + Date.now(), senderId: this.state.activeChat.participantIds.find(p => p !== 'u1') || 'u2', senderName: this.state.activeChat.name.split(' ')[0], content: reply, time: replyTime };
      this.state.activeChat.messages.push(replyMsg);
      el.innerHTML += `
        <div class="message-bubble received" style="animation:fadeInUp 0.2s ease">
          ${this.state.activeChat.type !== 'dm' ? `<div class="message-sender">${replyMsg.senderName}</div>` : ''}
          ${reply}
          <div class="message-time">${this.formatTime(replyTime)}</div>
        </div>`;
      el.scrollTop = el.scrollHeight;
    }, 1200 + Math.random() * 1500);
  },

  // ============================
  // PROFILE
  // ============================
  renderProfile() {
    const u = this.state.user;
    const friendCount = MockData.connections.filter(c => c.status === 'accepted').length;
    const myListings = MockData.listings.filter(l => l.sellerId === 'u1');

    return `
      <div class="profile-header">
        <div class="profile-avatar">${u.avatar}</div>
        <div class="profile-name">${u.firstName} ${u.lastName}</div>
        <div class="profile-handle">@${u.username}</div>
        <div class="profile-bio">${u.bio}</div>
        <div class="profile-stats">
          <div class="profile-stat"><div class="profile-stat-num">${friendCount}</div><div class="profile-stat-label">Friends</div></div>
          <div class="profile-stat"><div class="profile-stat-num">${u.karmaPoints}</div><div class="profile-stat-label">Karma</div></div>
          <div class="profile-stat"><div class="profile-stat-num">${u.studyStreakDays}</div><div class="profile-stat-label">🔥 Streak</div></div>
        </div>
        <div class="profile-actions">
          <button class="btn-secondary btn-small" onclick="App.showEditProfile()">✏️ Edit</button>
          <button class="btn-secondary btn-small" onclick="App.showStorefront()">🏪 Storefront</button>
        </div>
      </div>

      <div class="profile-section">
        <h3>Achievements 🏆</h3>
        <div class="achievements-grid">
          ${MockData.achievements.map(a => `
            <div class="achievement-item ${a.earned ? '' : 'locked'} animate-in">
              <div class="achievement-icon">${a.icon}</div>
              <div class="achievement-name">${a.name}</div>
            </div>`).join('')}
        </div>
      </div>

      <div class="profile-section">
        <h3>My Courses 📖</h3>
        <div style="display:flex;flex-wrap:wrap;gap:8px">
          ${MockData.courses.map(c => `<span class="filter-chip active">${c.code}</span>`).join('')}
        </div>
      </div>

      <div class="profile-section">
        <h3>Settings ⚙️</h3>
        <div class="settings-list">
          <div class="settings-item" onclick="App.toggleTheme()">
            <div class="settings-item-left">
              <div class="settings-icon">🌙</div>
              <div class="settings-item-text"><h4>Dark Mode</h4><p>Toggle light/dark theme</p></div>
            </div>
            <div class="toggle-switch ${this.state.theme === 'dark' ? 'active' : ''}" id="theme-toggle"></div>
          </div>
          <div class="settings-item">
            <div class="settings-item-left">
              <div class="settings-icon">📍</div>
              <div class="settings-item-text"><h4>Location Sharing</h4><p>Allow others to see you on the map</p></div>
            </div>
            <div class="toggle-switch ${this.state.user.locationSharing ? 'active' : ''}" onclick="App.toggleLocation(this)"></div>
          </div>
          <div class="settings-item">
            <div class="settings-item-left">
              <div class="settings-icon">🔔</div>
              <div class="settings-item-text"><h4>Notifications</h4><p>Push & in-app alerts</p></div>
            </div>
            <div class="toggle-switch active"></div>
          </div>
          <div class="settings-item" onclick="App.logout()">
            <div class="settings-item-left">
              <div class="settings-icon">🚪</div>
              <div class="settings-item-text"><h4 style="color:var(--danger)">Log Out</h4><p>Sign out of your account</p></div>
            </div>
          </div>
        </div>
      </div>`;
  },

  // ============================
  // MODALS
  // ============================
  openModal(title, bodyHTML) {
    document.getElementById('modal-title').textContent = title;
    document.getElementById('modal-body').innerHTML = bodyHTML;
    document.getElementById('modal-overlay').classList.add('active');
  },
  closeModal() {
    document.getElementById('modal-overlay').classList.remove('active');
  },

  // -- Listing Detail
  openListingDetail(id) {
    const l = MockData.listings.find(x => x.id === id);
    if (!l) return;
    const saved = this.state.savedListings.has(id);
    this.openModal(l.title, `
      <div class="detail-image">${this.categoryIcon(l.category)}</div>
      <div class="detail-title">${l.title}</div>
      <div class="detail-price">$${l.price.toFixed(2)}</div>
      <div class="detail-description">${l.description}</div>
      <div class="detail-seller" onclick="App.closeModal(); App.openUserProfile('${l.sellerId}')">
        <div class="user-avatar gradient">${l.sellerAvatar}</div>
        <div class="detail-seller-info">
          <h4>${l.sellerName}</h4>
          <p>★ ${l.sellerRating} · ${l.views} views</p>
        </div>
        <button class="btn-outline-accent btn-small">View</button>
      </div>
      <div class="detail-actions">
        <button class="btn-primary" onclick="App.closeModal(); App.messageSeller('${l.sellerId}')">💬 Message Seller</button>
        <button class="btn-secondary" onclick="App.toggleSaveListing('${l.id}')">
          ${saved ? '♥ Saved' : '♡ Save'}
        </button>
      </div>
      <div class="detail-views">👁 ${l.views} views · Listed ${this.formatDate(l.createdAt)}</div>
    `);
  },

  // -- Create Listing
  openCreateListing() {
    this.openModal('New Listing', `
      <div class="create-form">
        <div class="image-upload-area" onclick="App.showToast('Image upload — coming soon!')">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
          <span style="font-size:13px">Tap to add photos</span>
        </div>
        <div class="form-group">
          <label>Title</label>
          <input type="text" id="new-listing-title" placeholder="What are you selling?">
        </div>
        <div class="form-group">
          <label>Description</label>
          <textarea id="new-listing-desc" placeholder="Describe your item or service…" rows="3" style="resize:vertical"></textarea>
        </div>
        <div class="form-group">
          <label>Category</label>
          <select id="new-listing-cat">
            <option value="textbook">📚 Textbook</option>
            <option value="service">🛠️ Service</option>
            <option value="furniture">🪑 Furniture</option>
            <option value="electronics">📱 Electronics</option>
            <option value="other">📦 Other</option>
          </select>
        </div>
        <div class="form-group">
          <label>Price ($)</label>
          <input type="number" id="new-listing-price" placeholder="0.00" min="0" step="0.01">
        </div>
        <button class="btn-primary btn-full" onclick="App.submitListing()">🚀 Publish Listing</button>
      </div>
    `);
  },

  submitListing() {
    const title = document.getElementById('new-listing-title')?.value?.trim();
    const desc  = document.getElementById('new-listing-desc')?.value?.trim();
    const cat   = document.getElementById('new-listing-cat')?.value;
    const price = parseFloat(document.getElementById('new-listing-price')?.value);
    if (!title || !price) { this.showToast('Please fill in title and price'); return; }
    MockData.listings.unshift({
      id: 'l' + Date.now(), sellerId: 'u1', title, description: desc || '', category: cat,
      price, images: [], status: 'active', views: 0, createdAt: new Date().toISOString().split('T')[0],
      sellerName: 'Alex K.', sellerAvatar: 'AK', sellerRating: 4.8,
    });
    this.closeModal();
    this.navigate('hustle');
    this.showToast('Listing published! 🎉');
  },

  // -- User Profile
  openUserProfile(uid) {
    const u = MockData.users.find(x => x.id === uid);
    if (!u) return;
    const st = this.state.friendRequests[uid];
    const shared = MockData.courses.filter(c => u.courses && u.courses.some(uc => c.code === uc));
    let actionBtn;
    if (st === 'accepted') actionBtn = '<button class="btn-secondary btn-small" disabled>✓ Friends</button>';
    else if (st === 'pending') actionBtn = '<button class="btn-secondary btn-small" disabled>⏳ Pending</button>';
    else actionBtn = `<button class="btn-primary btn-small" onclick="App.sendFriendRequest('${uid}')">👋 Add Friend</button>`;

    this.openModal(`${u.firstName} ${u.lastName}`, `
      <div class="user-profile-modal">
        <div class="profile-avatar" style="background:var(--gradient-1);margin:0 auto 12px">${u.avatar}</div>
        <div class="profile-name">${u.firstName} ${u.lastName}</div>
        <div class="profile-handle">@${u.username}</div>
        <div class="profile-bio" style="margin-top:6px">${u.bio}</div>
        <div class="user-tags">
          <span class="filter-chip active">${u.major}</span>
          <span class="filter-chip">${u.status}</span>
          ${u.location ? `<span class="filter-chip">📍 ${u.location}</span>` : ''}
        </div>
        <div class="profile-stats" style="margin-top:16px">
          <div class="profile-stat"><div class="profile-stat-num">${u.karmaPoints}</div><div class="profile-stat-label">Karma</div></div>
          <div class="profile-stat"><div class="profile-stat-num">${u.studyStreakDays}</div><div class="profile-stat-label">🔥 Streak</div></div>
          <div class="profile-stat"><div class="profile-stat-num">${u.distance || '?'}m</div><div class="profile-stat-label">Away</div></div>
        </div>
        ${shared.length ? `<div style="margin-top:16px"><h4 style="font-size:13px;color:var(--text-secondary);margin-bottom:6px">Shared Courses</h4><div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:center">${shared.map(c => `<span class="filter-chip active">${c.code}</span>`).join('')}</div></div>` : ''}
        <div class="profile-action-btns" style="margin-top:16px;display:flex;gap:10px">
          ${actionBtn}
          <button class="btn-primary btn-small" onclick="App.closeModal(); App.messageSeller('${uid}')">💬 Message</button>
        </div>
      </div>
    `);
  },

  // -- Event Detail
  openEventDetail(eid) {
    const ev = MockData.events.find(x => x.id === eid);
    if (!ev) return;
    const going = this.state.rsvpEvents.has(eid);
    this.openModal(ev.title, `
      <span class="event-type-badge ${ev.type}" style="margin-bottom:12px;display:inline-flex">${ev.type.replace('_', ' ')}</span>
      <p style="color:var(--text-secondary);margin:12px 0">${ev.description}</p>
      <div style="display:flex;flex-direction:column;gap:8px;margin:16px 0">
        <div class="event-meta-row"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> ${this.formatTime(ev.startTime)} – ${this.formatTime(ev.endTime)}</div>
        <div class="event-meta-row"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/><circle cx="12" cy="10" r="3"/></svg> ${ev.location}</div>
        <div class="event-meta-row">👥 ${ev.attendees} / ${ev.maxAttendees} attending</div>
        <div class="event-meta-row">📅 Organized by ${ev.creatorName}</div>
      </div>
      <div class="detail-actions">
        <button class="btn-primary btn-full" onclick="App.toggleRSVP('${eid}')" id="modal-rsvp-btn">
          ${going ? '✓ You\'re Going!' : '🎯 RSVP — I\'m Going'}
        </button>
      </div>
    `);
  },

  // -- Course Room (opens chat)
  openCourseRoom(cid) {
    const course = MockData.courses.find(c => c.id === cid);
    if (!course) return;
    const conv = MockData.conversations.find(c => c.name.includes(course.code));
    if (conv) this.openChat(conv.id);
    else this.showToast(`Opening ${course.code} room… (demo)`);
  },
  openCircleChat(acid) {
    const ac = MockData.assignmentCircles.find(a => a.id === acid);
    if (!ac) return;
    this.showToast(`Opening "${ac.name}" circle chat… (demo)`);
  },

  // ============================
  // ACTIONS
  // ============================
  toggleRSVP(eid) {
    const ev = MockData.events.find(e => e.id === eid);
    if (!ev) return;
    if (this.state.rsvpEvents.has(eid)) {
      this.state.rsvpEvents.delete(eid);
      ev.attendees = Math.max(0, ev.attendees - 1);
      this.showToast('RSVP cancelled');
    } else {
      this.state.rsvpEvents.add(eid);
      ev.attendees++;
      this.showToast('You\'re going! 🎉');
    }
    // Re-render the current view
    if (this.state.currentPage === 'home') this.navigate('home');
    const btn = document.getElementById('modal-rsvp-btn');
    if (btn) btn.innerHTML = this.state.rsvpEvents.has(eid) ? '✓ You\'re Going!' : '🎯 RSVP — I\'m Going';
  },

  sendFriendRequest(uid) {
    this.state.friendRequests[uid] = 'pending';
    const u = MockData.users.find(x => x.id === uid);
    this.showToast(`Friend request sent to ${u ? u.firstName : 'user'}! 👋`);
    if (this.state.currentPage === 'home') this.navigate('home');
  },

  toggleSaveListing(lid) {
    if (this.state.savedListings.has(lid)) {
      this.state.savedListings.delete(lid);
      this.showToast('Removed from saved');
    } else {
      this.state.savedListings.add(lid);
      this.showToast('Saved! ♥');
    }
    this.closeModal();
    if (this.state.currentPage === 'hustle') this.navigate('hustle');
  },

  messageSeller(uid) {
    const u = uid === 'u1' ? MockData.currentUser : MockData.users.find(x => x.id === uid);
    if (!u) return;
    const existing = MockData.conversations.find(c => c.type === 'dm' && c.participantIds.includes(uid));
    if (existing) { this.openChat(existing.id); return; }
    // Create new conversation
    const newConv = {
      id: 'conv' + Date.now(), type: 'dm', name: `${u.firstName} ${u.lastName}`,
      participantIds: ['u1', uid], avatar: u.avatar, lastMessage: '', lastMessageTime: new Date().toISOString(),
      unread: 0, messages: [],
    };
    MockData.conversations.unshift(newConv);
    this.openChat(newConv.id);
  },

  toggleTheme() {
    this.state.theme = this.state.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', this.state.theme);
    localStorage.setItem('unino-theme', this.state.theme);
    const toggle = document.getElementById('theme-toggle');
    if (toggle) toggle.classList.toggle('active', this.state.theme === 'dark');
    this.showToast(`${this.state.theme === 'dark' ? '🌙 Dark' : '☀️ Light'} mode activated`);
  },

  toggleLocation(el) {
    this.state.user.locationSharing = !this.state.user.locationSharing;
    if (el) el.classList.toggle('active', this.state.user.locationSharing);
    this.showToast(this.state.user.locationSharing ? 'Location sharing ON 📍' : 'Location sharing OFF');
  },

  showAllEvents() {
    const evHTML = MockData.events.map(ev => {
      const going = this.state.rsvpEvents.has(ev.id);
      return `
        <div class="event-card" style="min-width:auto;margin-bottom:10px" onclick="App.closeModal(); App.openEventDetail('${ev.id}')">
          <span class="event-type-badge ${ev.type}">${ev.type.replace('_', ' ')}</span>
          <h4>${ev.title}</h4>
          <div class="event-meta">
            <div class="event-meta-row"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> ${this.formatTime(ev.startTime)}</div>
            <div class="event-meta-row"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/><circle cx="12" cy="10" r="3"/></svg> ${ev.location}</div>
          </div>
          <div class="event-footer">
            <span class="attendee-count">${ev.attendees}/${ev.maxAttendees}</span>
            <button class="rsvp-btn ${going ? 'going' : ''}" onclick="event.stopPropagation(); App.toggleRSVP('${ev.id}')">${going ? '✓ Going' : 'RSVP'}</button>
          </div>
        </div>`;
    }).join('');
    this.openModal('All Events 📅', `<div class="create-form">${evHTML}</div>`);
  },

  showCreateCircle() {
    this.openModal('New Assignment Circle', `
      <div class="create-form">
        <div class="form-group"><label>Course</label>
          <select>${MockData.courses.map(c => `<option value="${c.code}">${c.code} — ${c.name}</option>`).join('')}</select>
        </div>
        <div class="form-group"><label>Assignment Name</label><input type="text" placeholder="e.g. Problem Set 6"></div>
        <div class="form-group"><label>Due Date</label><input type="date" value="2026-02-20"></div>
        <button class="btn-primary btn-full" onclick="App.closeModal(); App.showToast('Circle created! 🎉')">Create Circle</button>
      </div>
    `);
  },

  showEditProfile() {
    const u = this.state.user;
    this.openModal('Edit Profile', `
      <div class="create-form">
        <div class="form-group"><label>First Name</label><input type="text" value="${u.firstName}"></div>
        <div class="form-group"><label>Last Name</label><input type="text" value="${u.lastName}"></div>
        <div class="form-group"><label>Bio</label><textarea rows="2">${u.bio}</textarea></div>
        <div class="form-group"><label>Major</label><input type="text" value="${u.major}"></div>
        <button class="btn-primary btn-full" onclick="App.closeModal(); App.showToast('Profile updated! ✨')">Save Changes</button>
      </div>
    `);
  },

  showStorefront() {
    const myListings = MockData.listings.filter(l => l.sellerId === 'u1');
    this.openModal('My Storefront 🏪', `
      <div class="storefront-header" style="padding:0;border:none;margin-bottom:16px">
        <div class="storefront-banner">Alex's Tech Shop</div>
        <div class="storefront-stats">
          <div class="storefront-stat"><div class="storefront-stat-num">${myListings.length}</div><div class="storefront-stat-label">Listings</div></div>
          <div class="storefront-stat"><div class="storefront-stat-num">★ 4.8</div><div class="storefront-stat-label">Rating</div></div>
          <div class="storefront-stat"><div class="storefront-stat-num">12</div><div class="storefront-stat-label">Sales</div></div>
        </div>
      </div>
      ${myListings.length > 0 ? myListings.map(l => `
        <div class="course-card" style="margin-bottom:8px">
          <div class="course-icon cs">${this.categoryIcon(l.category)}</div>
          <div class="course-info">
            <div class="course-name">${l.title}</div>
            <div class="course-meta">$${l.price.toFixed(2)} · ${l.views} views</div>
          </div>
        </div>`).join('') : '<div class="empty-state"><div class="empty-state-icon">🏪</div><h3>No listings yet</h3><p>Start selling to build your storefront!</p></div>'}
      <button class="btn-primary btn-full" style="margin-top:12px" onclick="App.closeModal(); App.openCreateListing()">+ Add Listing</button>
    `);
  },

  initMapInteractions() { /* placeholder for future map click handlers */ },
};

// ============================
// BOOT
// ============================
document.addEventListener('DOMContentLoaded', () => App.init());
