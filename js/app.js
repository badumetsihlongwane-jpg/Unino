/* ══════════════════════════════════════════════════════
 *  UNIBO — Campus Social Engine v5
 *  Firebase Auth + Firestore | base64 images
 *  Feed (Discover tabs), Explore (Radar/List + Modules),
 *  Marketplace, Messaging (fixed), Profiles (fixed)
 * ══════════════════════════════════════════════════════ */

// ─── State ───────────────────────────────────────
const state = { user: null, profile: null, page: 'feed', status: 'online', unsubs: [], lastMsgTab: 'dm' };

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

// ─── Custom Voice Note Player ────────────────────
let _vnCounter = 0;
const _vnAudios = {};

function renderVoiceMsg(audioURL) {
  const id = `vn-${++_vnCounter}`;
  return `<div class="vn-player" id="${id}" data-src="${audioURL}">
    <button class="vn-play-btn" onclick="toggleVN('${id}')">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
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
      if (o) { o.querySelector('.vn-play-btn').innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>'; o.classList.remove('playing'); }
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
      el.querySelector('.vn-play-btn').innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
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
    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/></svg>';
  } else {
    audio.pause();
    el.classList.remove('playing');
    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
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
  return esc(text).replace(/#(\w+)/g, '<span class="hashtag">#$1</span>');
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
  listenForNotifications();
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
      <button class="reels-fab" onclick="openReelsViewer()" title="Watch Reels">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="#fff" stroke="#fff" stroke-width="1.5"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2" fill="none" stroke="#fff" stroke-width="2"/></svg>
      </button>
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
      state.posts = visible;
      // Save scroll position before re-render to prevent "jump to top"
      const contentEl = document.getElementById('content');
      const savedScroll = contentEl ? contentEl.scrollTop : 0;
      renderPosts(visible);
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
      const isFriend = (state.profile.friends || []).includes(u.id);
      return `
        <div class="discover-card" onclick="openProfile('${u.id}')">
          <div class="discover-card-avatar">
            ${avatar(u.displayName, u.photoURL, 'avatar-lg')}
            ${online}
          </div>
          <div class="discover-card-name">${esc(u.displayName)}</div>
          <div class="discover-card-meta">${esc(u.major || 'Student')}</div>
          ${tag ? `<div class="discover-card-tag">${tag}</div>` : ''}
          <button class="discover-card-btn" onclick="event.stopPropagation();${isFriend ? `startChat('${u.id}','${esc(u.displayName)}','${u.photoURL || ''}')` : `sendFriendRequest('${u.id}','${esc(u.displayName)}','${u.photoURL || ''}')`}">${isFriend ? 'Message' : 'Add Friend'}</button>
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
      const grad = ev.gradient || 'linear-gradient(135deg,#6C5CE7,#A855F7)';
      const goingCount = (ev.going || []).length;
      return `
        <div class="discover-card event-card" style="background:${grad}" onclick="${ev.id ? `openEventDetail('${ev.id}')` : `toast('View on Campus map!')`}">
          <div style="font-size:36px;margin-bottom:8px">${ev.emoji || '📅'}</div>
          <div class="discover-card-name" style="color:#fff">${esc(ev.title)}</div>
          <div class="discover-card-meta" style="color:rgba(255,255,255,0.8)">${esc(ev.date || '')} ${esc(ev.time || '')}</div>
          <div class="discover-card-tag" style="background:rgba(255,255,255,0.2);color:#fff">📍 ${loc ? loc.name : esc(ev.location || '?')}</div>
          ${goingCount ? `<div style="font-size:11px;color:rgba(255,255,255,0.7);margin-top:4px">👥 ${goingCount} going</div>` : ''}
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
  hdr.innerHTML = `
    ${avatar(story.authorName, story.authorPhoto, 'avatar-sm')}
    <div><b>${esc(story.authorName)}</b><br><small>${timeAgo(story.createdAt)}</small></div>
  `;

  // Content
  const content = $('#story-viewer-content');
  if (story.type === 'video' && story.videoURL) {
    content.innerHTML = `
      <video src="${story.videoURL}" class="story-full-video" autoplay playsinline loop style="width:100%;height:100%;object-fit:cover"></video>
      ${story.caption ? `<div class="story-caption">${esc(story.caption)}</div>` : ''}
    `;
    content.style.background = '#000';
    // Video stories: auto-advance after video ends or 15s max
    clearTimeout(storyViewerData.timer);
    const vid = content.querySelector('video');
    if (vid) {
      vid.onended = () => advanceStory(1);
      storyViewerData.timer = setTimeout(() => advanceStory(1), 15000);
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
function renderQuoteEmbed(rp) {
  if (!rp) return '';
  const hasImg = rp.imageURL && rp.mediaType !== 'video';
  const hasVid = rp.videoURL || (rp.mediaType === 'video');
  const vidUrl = hasVid ? (rp.videoURL || rp.imageURL) : null;
  let vidHtml = '';
  if (hasVid && vidUrl) {
    const vpd = createVideoPlayer(vidUrl);
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
    const hasCollage = post.imageURLs && post.imageURLs.length > 1 && !post.repostOf;
    const hasVideo = post.videoURL || (post.mediaType === 'video');
    const hasImage = post.imageURL && !hasVideo && !hasCollage;
    const mediaURL = hasVideo ? (post.videoURL || post.imageURL) : post.imageURL;
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
          <div onclick="openProfile('${post.authorId}')" style="cursor:pointer">${avatar(post.authorName, post.authorPhoto, 'avatar-md')}</div>
          <div class="post-header-info">
            <div class="post-author-name" onclick="openProfile('${post.authorId}')">${esc(post.authorName)}</div>
            <div class="post-meta">${post.visibility === 'friends' ? '👫 ' : '🌍 '}${esc(post.authorUni || '')} · ${timeAgo(post.createdAt)}</div>
          </div>
          ${post.authorId === state.user.uid ? `<button class="icon-btn post-more-btn" onclick="showPostOptions('${post.id}')" title="Options" style="margin-left:auto;font-size:18px;color:var(--text-tertiary)">⋯</button>` : ''}
        </div>
        ${post.content ? `<div class="post-content">${formatContent(post.content)}</div>` : ''}
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
            <button class="reel-act-btn" onclick="closeReelsViewer();openComments('${p.id}')">
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
  let comments = [];
  try {
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

  _commentReplyTo = null;

  function renderComment(c, isReply = false) {
     const liked = (c.likes || []).includes(state.user.uid);
     const cReplies = replyMap[c.id] || [];
     
     return `
      <div class="comment-item ${isReply ? 'reply-item' : ''}" id="c-${c.id}">
        <div class="comment-avatar-col">
          ${avatar(c.authorName, c.authorPhoto, 'avatar-sm')}
        </div>
        <div class="comment-content-col">
           <div class="comment-bubble enhanced">
              <div class="comment-header">
                  <span class="comment-author" onclick="openProfile('${c.authorId}')">${esc(c.authorName)}</span>
              </div>
              <div class="comment-text">${esc(c.text)}</div>
           </div>
           <div class="comment-actions-row">
               <span class="comment-time">${timeAgo(c.createdAt)}</span>
               <button class="c-act ${liked?'liked':''}" onclick="toggleCommentLike('${c.id}','${postId}')">
                  ${liked ? 'Like' : 'Like'} ${c.likeCount > 0 ? c.likeCount : ''}
               </button>
               <button class="c-act" onclick="setCommentReply('${c.replyTo || c.id}','${esc(c.authorName)}')">Reply</button>
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
    <div class="modal-body comment-modal-body" style="display:flex;flex-direction:column;height:70vh;padding:0">
      <div id="comments-container" class="comments-scroll" style="flex:1;overflow-y:auto;padding:16px">
        ${renderCommentTree()}
      </div>
      <div id="comment-reply-indicator" class="reply-indicator" style="display:none">
        <span id="comment-reply-label"></span>
        <button onclick="clearCommentReply()">&times;</button>
      </div>
      <div class="comment-input-wrap" style="position:sticky;bottom:0;background:var(--bg-secondary);padding:12px 16px;border-top:1px solid var(--border);flex-shrink:0">
        <input type="text" id="comment-input" placeholder="Write a comment..." autocomplete="off">
        <button onclick="postComment('${postId}')">Post</button>
      </div>
    </div>
  `);
}

function setCommentReply(commentId, authorName) {
  _commentReplyTo = { id: commentId, authorName };
  const ind = $('#comment-reply-indicator');
  const label = $('#comment-reply-label');
  if (ind) { ind.style.display = 'flex'; }
  if (label) { label.textContent = `↩ Replying to ${authorName}`; }
  $('#comment-input')?.focus();
}

function clearCommentReply() {
  _commentReplyTo = null;
  const ind = $('#comment-reply-indicator');
  if (ind) ind.style.display = 'none';
}

async function postComment(postId) {
  const input = $('#comment-input'); const text = input?.value.trim(); if (!text) return;
  input.value = '';
  const replyTo = _commentReplyTo ? _commentReplyTo.id : null;
  _commentReplyTo = null;
  try {
    await db.collection('posts').doc(postId).collection('comments').add({
      text, authorId: state.user.uid, authorName: state.profile.displayName,
      authorPhoto: state.profile.photoURL || null, replyTo: replyTo || null,
      createdAt: FieldVal.serverTimestamp()
    });
    await db.collection('posts').doc(postId).update({ commentsCount: FieldVal.increment(1) });
    
    const pDoc = await db.collection('posts').doc(postId).get();
    if (pDoc.exists) addNotification(pDoc.data().authorId, 'comment', 'commented on your post', { postId });

    // Reopen to show the new comment
    openComments(postId);
  } catch (e) { console.error(e); toast('Failed'); }
}

// ─── Image Viewer ────────────────────────────────
function viewImage(url) { const v = $('#img-view'); if (!v) return; $('#img-full').src = url; v.style.display = 'flex'; }

// ─── Create Post ─────────────────────────────────
function openCreateModal() {
  let pendingFiles = []; // Array of File objects
  let pendingIsVideo = false;
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
      <div id="create-preview" class="media-preview" style="display:none">
        <div id="create-preview-content" class="collage-preview-grid"></div>
        <button class="image-preview-remove" onclick="document.getElementById('create-preview').style.display='none';window._createPendingFiles=[]">&times;</button>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;border-top:1px solid var(--border);padding-top:12px;margin-top:12px">
        <div style="display:flex;align-items:center;gap:8px">
          <label class="add-photo-btn" title="Photos (multiple)"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg><input type="file" hidden accept="image/*" id="create-file" multiple></label>
          <label class="add-photo-btn" title="Video"><svg width="22" height="22" viewBox="0 0 24 24" stroke="var(--accent)" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7" fill="var(--accent)"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2" fill="none"/></svg><input type="file" hidden accept="video/*" id="create-video-file"></label>
          <select id="create-visibility" style="padding:6px 10px;border-radius:100px;border:1px solid var(--border);background:var(--bg-tertiary);color:var(--text-primary);font-size:12px;font-weight:600">
            <option value="public">🌍 Public</option>
            <option value="friends">👫 Friends</option>
          </select>
        </div>
        <button class="btn-primary" id="create-submit" style="padding:10px 28px">Post</button>
      </div>
    </div>
  `);
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
  $('#create-file').onchange = e => {
    if (e.target.files.length) {
      pendingFiles = [...pendingFiles, ...Array.from(e.target.files)];
      pendingIsVideo = false;
      showPreviews();
    }
  };
  $('#create-video-file').onchange = e => {
    if (e.target.files[0]) {
      pendingFiles = [e.target.files[0]];
      pendingIsVideo = true;
      showPreviews();
    }
  };
  $('#create-submit').onclick = async () => {
    const text = $('#create-text').value.trim();
    if (!text && !pendingFiles.length) return toast('Post cannot be empty');
    const visibility = $('#create-visibility')?.value || 'public';
    closeModal(); toast('Uploading...');
    try {
      let mediaURL = null;
      let mediaType = 'text';
      let imageURLs = null;
      if (pendingFiles.length && pendingIsVideo) {
        mediaURL = await uploadToR2(pendingFiles[0], 'videos');
        mediaType = 'video';
      } else if (pendingFiles.length === 1) {
        mediaURL = await uploadToR2(pendingFiles[0], 'images');
        mediaType = 'image';
      } else if (pendingFiles.length > 1) {
        // Multi-image: upload all
        imageURLs = [];
        for (const f of pendingFiles) {
          const url = await uploadToR2(f, 'images');
          imageURLs.push(url);
        }
        mediaURL = imageURLs[0]; // First image as main
        mediaType = 'collage';
      }
      await db.collection('posts').add({
        content: text,
        imageURL: mediaType === 'image' || mediaType === 'collage' ? mediaURL : null,
        imageURLs: imageURLs || null,
        videoURL: mediaType === 'video' ? mediaURL : null,
        mediaType,
        authorId: state.user.uid, authorName: state.profile.displayName,
        authorPhoto: state.profile.photoURL || null, authorUni: state.profile.university || '',
        visibility,
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
        <button class="explore-toggle-btn" data-v="map">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
          Campus
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
  else if (exploreView === 'map') renderCampusMapView();
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
//  CAMPUS MAP — Events & Locations on visual map
// ══════════════════════════════════════════════════
const CAMPUS_LOCATIONS = [
  { id: 'library', name: 'Library', emoji: '📚', x: 30, y: 20 },
  { id: 'main-hall', name: 'Main Hall', emoji: '🏛', x: 50, y: 15 },
  { id: 'student-center', name: 'Student Center', emoji: '☕', x: 70, y: 25 },
  { id: 'cs-building', name: 'CS Building', emoji: '💻', x: 20, y: 50 },
  { id: 'sports-complex', name: 'Sports Complex', emoji: '⚽', x: 80, y: 50 },
  { id: 'amphitheatre', name: 'Amphitheatre', emoji: '🎭', x: 45, y: 45 },
  { id: 'quad', name: 'The Quad', emoji: '🌳', x: 55, y: 60 },
  { id: 'cafeteria', name: 'Cafeteria', emoji: '🍕', x: 35, y: 70 },
  { id: 'res-halls', name: 'Res Halls', emoji: '🏠', x: 75, y: 75 },
  { id: 'lab-block', name: 'Lab Block', emoji: '🔬', x: 15, y: 35 },
  { id: 'admin', name: 'Admin Block', emoji: '🏢', x: 60, y: 35 },
  { id: 'parking', name: 'Parking', emoji: '🅿️', x: 90, y: 85 },
];

let allCampusEvents = [];

async function loadCampusEvents() {
  try {
    const snap = await db.collection('events').orderBy('date','asc').limit(50).get();
    allCampusEvents = snap.docs.map(d => ({ id: d.id, ...d.data() }));
  } catch (e) {
    console.error(e);
    // Fallback seed events if collection is empty or fails
    allCampusEvents = [
      { title: 'Study Jam Session', emoji: '📚', date: '2026-02-11', time: '18:00', location: 'library', createdBy: 'system', going: [], gradient: 'linear-gradient(135deg,#6C5CE7,#A855F7)' },
      { title: 'Career Fair 2026', emoji: '💼', date: '2026-02-15', time: '09:00', location: 'main-hall', createdBy: 'system', going: [], gradient: 'linear-gradient(135deg,#7C3AED,#C084FC)' },
      { title: 'Pool Tournament', emoji: '🎱', date: '2026-02-14', time: '16:00', location: 'student-center', createdBy: 'system', going: [], gradient: 'linear-gradient(135deg,#8B5CF6,#D946EF)' },
      { title: 'Welcome Mixer', emoji: '🎉', date: '2026-02-14', time: '19:00', location: 'amphitheatre', createdBy: 'system', going: [], gradient: 'linear-gradient(135deg,#6366F1,#818CF8)' },
      { title: 'Hackathon', emoji: '💻', date: '2026-02-20', time: '08:00', location: 'cs-building', createdBy: 'system', going: [], gradient: 'linear-gradient(135deg,#7C3AED,#A855F7)' },
      { title: 'Open Mic Night', emoji: '🎤', date: '2026-02-18', time: '19:00', location: 'quad', createdBy: 'system', going: [], gradient: 'linear-gradient(135deg,#D946EF,#E879F9)' },
    ];
  }
}

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
      <div class="campus-map">
        ${CAMPUS_LOCATIONS.map(loc => {
          const evts = eventsByLoc[loc.id] || [];
          const hasEvents = evts.length > 0;
          return `
            <div class="campus-pin ${hasEvents ? 'has-events pulse' : ''}"
                 style="left:${loc.x}%;top:${loc.y}%"
                 onclick="openLocationDetail('${loc.id}')">
              <div class="campus-pin-icon">${loc.emoji}</div>
              ${hasEvents ? `<span class="campus-pin-badge">${evts.length}</span>` : ''}
              <div class="campus-pin-label">${loc.name}</div>
            </div>`;
        }).join('')}
      </div>

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
  openModal(`
    <div class="modal-header"><h2>Create Event</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      <div class="form-group"><label>Event Title</label><input type="text" id="ev-title" placeholder="e.g. Study Session"></div>
      <div class="form-group"><label>Emoji</label><input type="text" id="ev-emoji" value="📅" placeholder="📅" style="width:60px"></div>
      <div class="form-group"><label>Location</label>
        <select id="ev-location">
          ${CAMPUS_LOCATIONS.map(l => `<option value="${l.id}" ${l.id === presetLoc ? 'selected' : ''}>${l.emoji} ${l.name}</option>`).join('')}
        </select>
      </div>
      <div style="display:flex;gap:8px">
        <div class="form-group" style="flex:1"><label>Date</label><input type="date" id="ev-date"></div>
        <div class="form-group" style="flex:1"><label>Time</label><input type="time" id="ev-time"></div>
      </div>
      <div class="form-group"><label>Description (optional)</label><textarea id="ev-desc" placeholder="What's happening?" style="resize:none;height:60px"></textarea></div>
      <button class="btn-primary btn-full" id="ev-create-btn">Create Event</button>
    </div>
  `);
  $('#ev-create-btn').onclick = async () => {
    const title = $('#ev-title')?.value.trim();
    const emoji = $('#ev-emoji')?.value.trim() || '📅';
    const location = $('#ev-location')?.value;
    const date = $('#ev-date')?.value;
    const time = $('#ev-time')?.value || '';
    const desc = $('#ev-desc')?.value.trim() || '';
    if (!title || !date) return toast('Title and date required');
    closeModal(); toast('Creating event...');
    const gradients = ['linear-gradient(135deg,#6C5CE7,#A855F7)','linear-gradient(135deg,#7C3AED,#C084FC)','linear-gradient(135deg,#8B5CF6,#D946EF)','linear-gradient(135deg,#6366F1,#818CF8)','linear-gradient(135deg,#D946EF,#E879F9)'];
    try {
      await db.collection('events').add({
        title, emoji, location, date, time, description: desc,
        gradient: gradients[Math.floor(Math.random() * gradients.length)],
        createdBy: state.user.uid,
        creatorName: state.profile.displayName,
        going: [state.user.uid],
        createdAt: FieldVal.serverTimestamp()
      });
      toast('Event created!');
      await loadCampusEvents();
      renderCampusMapView();
    } catch (e) { toast('Failed'); console.error(e); }
  };
}

async function openEventDetail(eventId) {
  if (!eventId) return;
  try {
    const doc = await db.collection('events').doc(eventId).get();
    if (!doc.exists) return toast('Event not found');
    const ev = { id: doc.id, ...doc.data() };
    const loc = CAMPUS_LOCATIONS.find(l => l.id === ev.location);
    const amGoing = (ev.going || []).includes(state.user.uid);
    const goingCount = (ev.going || []).length;
    openModal(`
      <div class="modal-header"><h2>${ev.emoji || '📅'} Event</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
      <div class="modal-body">
        <div style="font-size:22px;font-weight:800;margin-bottom:8px">${esc(ev.title)}</div>
        <div style="display:flex;flex-wrap:wrap;gap:12px;font-size:13px;color:var(--text-secondary);margin-bottom:16px">
          <span>📍 ${loc ? loc.name : esc(ev.location)}</span>
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
      </div>
    `);
  } catch (e) { toast('Could not load event'); console.error(e); }
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
    if (e.target.files[0]) { window._sellFile = e.target.files[0]; $('#sell-preview img').src = localPreview(e.target.files[0]); $('#sell-preview').style.display = 'block'; }
  };
  $('#sell-submit').onclick = async () => {
    const title = $('#sell-title').value.trim(), price = $('#sell-price').value.trim();
    if (!title || !price) return toast('Title and price required');
    closeModal(); toast('Uploading...');
    try {
      let sellImgURL = null;
      if (window._sellFile) { sellImgURL = await uploadToR2(window._sellFile, 'listings'); }
      await db.collection('listings').add({
        title, price, category: $('#sell-cat').value, imageURL: sellImgURL,
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
        <button class="btn-primary" style="flex:1" onclick="closeModal();${(state.profile.friends || []).includes(item.sellerId) || item.sellerId === state.user.uid ? `startChat('${item.sellerId}','${esc(item.sellerName)}','')` : `sendFriendRequest('${item.sellerId}','${esc(item.sellerName)}','');toast('Add seller as friend first')`}">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          ${(state.profile.friends || []).includes(item.sellerId) ? 'Contact Seller' : 'Add Seller First'}
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
            let content = '';
            if (m.audioURL) content += renderVoiceMsg(m.audioURL);
            if (m.imageURL) content += `<img src="${m.imageURL}" class="msg-image" onclick="viewImage('${m.imageURL}')">`;
            if (m.text) content += esc(m.text);
            return `<div class="msg-row ${isMe ? 'msg-row-sent' : 'msg-row-received'}">
              ${!isMe ? `<div class="msg-avatar-wrap">${avatar(m.senderName || '?', m.senderPhoto, 'avatar-xs')}</div>` : ''}
              <div class="msg-bubble ${isMe ? 'msg-sent' : 'msg-received'}">
              ${!isMe ? `<div class="gchat-sender">${esc(m.senderName?.split(' ')[0] || '?')}</div>` : ''}
              ${content}
              <div class="msg-time">${m.createdAt ? timeAgo(m.createdAt) : ''}</div>
            </div></div>`;
          }).join('');
          msgs.scrollTop = msgs.scrollHeight;
        }
      });

    const sendGMsg = async () => {
      const input = $('#gchat-input');
      const text = input.value.trim();
      let audioURL = null;
      if (window._gchatVoiceBlob) {
        const af = new File([window._gchatVoiceBlob], `voice_${Date.now()}.webm`, { type: 'audio/webm' });
        audioURL = await uploadToR2(af, 'voice');
        window._gchatVoiceBlob = null;
      }
      if (!text && !audioURL) return;
      input.value = '';
      try {
        await db.collection(collection).doc(groupId).collection('messages').add({
          text: text || '', audioURL: audioURL || null,
          senderId: uid, senderName: state.profile.displayName,
          senderPhoto: state.profile.photoURL || null,
          createdAt: FieldVal.serverTimestamp()
        });
        await db.collection(collection).doc(groupId).update({
          lastMessage: audioURL ? '🎤 Voice' : text, updatedAt: FieldVal.serverTimestamp()
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
      state.lastMsgTab = tab.dataset.mt;
      if (tab.dataset.mt === 'dm') loadDMList();
      else if (tab.dataset.mt === 'groups') loadGroupList();
      else loadAssignmentGroups();
    };
  });
  // Restore last active tab
  const restoreTab = state.lastMsgTab || 'dm';
  const tabBtn = document.querySelector(`.msg-tab[data-mt="${restoreTab}"]`);
  if (tabBtn) { tabBtn.click(); } else { loadDMList(); }
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

// ══════════════════════════════════════════════════
//  FRIEND REQUEST SYSTEM
// ══════════════════════════════════════════════════
async function sendFriendRequest(toUid, toName, toPhoto) {
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
    toast('Friend request sent!');
    // Refresh current page to update button states
    if (state.page === 'feed') renderFeed();
    else if (state.page === 'explore') renderExplore();
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
    state.profile.friendRequests = data.friendRequests || [];
    state.profile.sentRequests = data.sentRequests || [];
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
  const requests = state.profile.friendRequests || [];
  const unreadCount = _notifications.filter(n => !n.read).length;
  const dot = $('#notif-dot');
  if (dot) dot.style.display = (requests.length > 0 || unreadCount > 0) ? 'block' : 'none';
}

function loadNotifications() {
  const dd = $('#notif-dropdown');
  const requests = state.profile.friendRequests || [];
  const notifs = _notifications;

  if (!requests.length && !notifs.length) {
    dd.innerHTML = `
      <div class="notif-header"><h3>Notifications</h3></div>
      <div style="padding:32px;text-align:center;color:var(--text-tertiary)">
        <div style="font-size:32px;margin-bottom:8px">🔔</div>
        <p>No new notifications</p>
      </div>`;
    return;
  }

  let html = '<div class="notif-header"><h3>Notifications</h3></div><div class="notif-scroll" style="max-height:400px;overflow-y:auto">';

  if (requests.length) {
    html += `<div style="padding:8px 16px;font-weight:600;font-size:13px;color:var(--text-secondary)">Friend Requests</div>`;
    html += requests.map(r => `
      <div class="notif-item unread">
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

  if (notifs.length) {
    if (requests.length) html += `<div style="height:1px;background:var(--border);margin:8px 0"></div>`;
    html += notifs.map(n => {
      const icon = n.type === 'like' ? '❤️' : n.type === 'comment' ? '💬' : '🔔';
      return `
       <div class="notif-item ${n.read ? '' : 'unread'}" ${n.payload?.postId ? `onclick="viewPost('${n.payload.postId}');markNotifRead('${n.id}')"` : ''}>
         <div style="position:relative">
           ${avatar(n.from.name, n.from.photo, 'avatar-md')}
           <div style="position:absolute;bottom:-2px;right:-2px;font-size:12px;background:var(--bg-secondary);border-radius:50%;padding:2px">${icon}</div>
         </div>
         <div class="notif-content">
           <div class="notif-text"><strong>${esc(n.from.name)}</strong> ${esc(n.text)}</div>
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

    openModal(`
      <div class="modal-header"><h2>Post</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
      <div class="modal-body" style="padding:16px">
        <div class="post-card" style="box-shadow:none;border:none;margin:0;padding:0">
          ${p.repostOf ? `<div style="padding-bottom:6px;margin-bottom:6px;font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:6px">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
            Reposted
          </div>` : ''}
          <div class="post-header">
            <div onclick="closeModal();openProfile('${p.authorId}')" style="cursor:pointer">${avatar(p.authorName, p.authorPhoto, 'avatar-md')}</div>
            <div class="post-header-info">
              <div class="post-author-name" onclick="closeModal();openProfile('${p.authorId}')">${esc(p.authorName || 'User')}</div>
              <div class="post-meta">${timeAgo(p.createdAt)}</div>
            </div>
          </div>
          ${p.content ? `<div class="post-content">${formatContent(p.content)}</div>` : ''}
          ${hasImage && mediaURL ? `<div class="post-media-wrap"><img src="${mediaURL}" class="post-image" onclick="viewImage('${mediaURL}')" style="max-height:300px"></div>` : ''}
          ${p.repostOf ? renderQuoteEmbed(p.repostOf) : ''}
          <div class="post-actions" style="border-top:1px solid var(--border);padding-top:12px;margin-top:12px">
            <button class="post-action ${liked ? 'liked' : ''}" onclick="toggleLike('${p.id}');closeModal()">❤ ${lc || 'Like'}</button>
            <button class="post-action" onclick="closeModal();openComments('${p.id}')">💬 ${cc || 'Comment'}</button>
            <button class="post-action" onclick="closeModal();openShareModal('${p.id}')">↗ Share</button>
          </div>
        </div>
      </div>
    `);
  } catch (e) { console.error(e); toast('Could not load post'); }
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
            if (m.audioURL) content += renderVoiceMsg(m.audioURL);
            if (m.imageURL) content += `<img src="${m.imageURL}" class="msg-image" onclick="viewImage('${m.imageURL}')">`;
            // Handle shared post messages
            if (m.type === 'share_post' && m.payload?.postId) {
              content = `<div class="shared-post-card" onclick="viewPost('${m.payload.postId}')">
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/></svg>
                  <span style="font-size:12px;font-weight:600;color:var(--accent)">Shared Post</span>
                </div>
                <div style="font-size:13px;color:var(--text-secondary)">Tap to view post</div>
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
            return `<div class="msg-row ${isMe ? 'msg-row-sent' : 'msg-row-received'}">
              ${!isMe ? `<div class="msg-avatar-wrap">${avatar(name, photo, 'avatar-xs')}</div>` : ''}
              <div class="msg-bubble ${isMe ? 'msg-sent' : 'msg-received'}">${content}<div class="msg-time">${ts ? timeAgo(ts) : ''}</div></div>
            </div>`;
          }).join('');
          msgs.scrollTop = msgs.scrollHeight;
        }
      });

    // Send message + image
    const input = $('#chat-input');
    let chatPendingImg = null;

    const sendMsg = async () => {
      const text = input.value.trim();
      let img = chatPendingImg;
      const chatFile = window._chatFile || null;
      if (!text && !img && !window._chatVoiceBlob) return;
      input.value = ''; chatPendingImg = null; window._chatFile = null;
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
        await db.collection('conversations').doc(convoId).collection('messages').add({
          text: text || '', imageURL: imageURL || null, audioURL: audioURL || null,
          senderId: uid, createdAt: FieldVal.serverTimestamp()
        });
        const otherUid = convo.participants.find(p => p !== uid);
        const lastMsg = audioURL ? '🎤 Voice' : imageURL ? (text || '📷 Photo') : text;
        await db.collection('conversations').doc(convoId).set({
          lastMessage: lastMsg, updatedAt: FieldVal.serverTimestamp(),
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
      showScreen('app');
      navigate('chat');
    };
  } catch (e) { console.error(e); toast('Could not open chat'); }
}

async function startChat(uid, name, photo) {
  if (uid === state.user.uid) return toast("That's you!");
  // Friends-only gate
  const myFriends = state.profile.friends || [];
  if (!myFriends.includes(uid)) return toast('Add as friend first to message');
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

      // Filter logic: If not me and not friend, hide friends-only posts
      if (uid !== state.user.uid) {
        const isFriend = (state.profile.friends || []).includes(uid);
        if (!isFriend) {
          posts = posts.filter(p => p.visibility !== 'friends');
        }
      }
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
                  friendBtn = `<button class="btn-outline" onclick="sendFriendRequest('${uid}','${esc(user.displayName)}','${user.photoURL || ''}');this.textContent='Pending…';this.disabled=true;this.style.opacity='0.6'">Add Friend</button>`;
                }
                const isFriendForChat = isFriend;
                const msgBtn = isFriendForChat
                  ? `<button class="btn-primary" onclick="startChat('${uid}','${esc(user.displayName)}','${user.photoURL || ''}')">Message</button>`
                  : `<button class="btn-outline" disabled style="opacity:0.5" title="Add as friend first">🔒 Message</button>`;
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
        else tc.innerHTML = renderProfileAbout(user);
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
  const _profPlayers = [];
  const html = `<div class="profile-posts">${posts.map(p => {
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
        ${avatar(user.displayName, user.photoURL, 'avatar-md')}
        <div class="post-header-info">
          <div class="post-author-name">${esc(user.displayName)}</div>
          <div class="post-meta">${timeAgo(p.createdAt)}</div>
        </div>
        ${isMe ? `<button class="icon-btn" onclick="showPostOptions('${p.id}')" style="margin-left:auto">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/></svg>
        </button>` : ''}
      </div>
      ${p.content ? `<div class="post-content">${formatContent(p.content)}</div>` : ''}
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
  requestAnimationFrame(() => _profPlayers.forEach(p => initPlayer(p.id)));
  return html;
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
      ${user.joinedAt ? `<div class="about-item"><span class="about-icon">🗓</span><div><div class="about-label">Joined</div><div class="about-value">${timeAgo(user.joinedAt)}</div></div></div>` : ''}
    </div>`;
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
  let newPhoto = null; let newPhotoFile = null;
  $('#edit-photo').onchange = async e => {
    if (e.target.files[0]) { newPhotoFile = e.target.files[0]; newPhoto = 'pending'; toast('Photo selected'); }
  };
  $('#edit-save').onclick = async () => {
    const name = $('#edit-name').value.trim();
    const bio = $('#edit-bio').value.trim();
    const modulesRaw = $('#edit-modules').value || '';
    const modules = modulesRaw.split(',').map(m => m.trim().toUpperCase()).filter(Boolean);
    if (!name) return toast('Name required');
    closeModal(); toast('Saving...');
    const updates = { displayName: name, bio, modules };
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
         payload: { postId },
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
    openStoryCreator, viewStory, closeStoryViewer, advanceStory,
    openCreateGroup, openGroupChat, joinGroup, loadStories,
    openCreateAssignmentGroup, openAssignmentDetail, joinAsg, requestJoinAsg,
    approveAsgRequest, rejectAsgRequest, removeFromAsg, leaveAsg,
    toggleAsgLock, archiveAsg, doArchiveAsg, autoFillAsg,
    openAsgPreferences, openAsgChat, loadAssignmentGroups,
    sendFriendRequest, acceptFriendRequest, rejectFriendRequest, unfriend,
    loadNotifications, setCommentReply, clearCommentReply,
    openCreateEvent, openEventDetail, openLocationDetail, toggleEventGoing,
    startVoiceRecord, cancelVoiceRecord, stopVoiceAndSend, openReelsViewer,
    toggleCommentLike, openShareModal, repost, openQuoteRepost, shareToFriend, viewPost, markNotifRead,
    closeReelsViewer, toggleReelPlay, reelLike,
    toggleVN, seekVN
  });
});
