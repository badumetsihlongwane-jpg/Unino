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

// ─── App Version ─────────────────────────────────
const APP_VERSION = 47;

// ─── Admin / Official Account ────────────────────
const ADMIN_EMAIL = 'admin@mynwu.ac.za';
const VERIFIED_UIDS = new Set(); // campus-verified users
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
let _presenceListenersAdded = false;
let _feedAutoplayObserver = null;
let _reelsObserver = null;
let _reelsSoundEnabled = false;
let _nativeBackListenerBound = false;
let _nativeNotificationListenersBound = false;
let _nativeShellReady = false;
let _nativeLocalNotificationsReady = false;
let _nativeAppIsActive = true;
let _nativePushReady = false;
let _nativePushToken = '';
let _nativePushListenersBound = false;
let _nativePushLastRegisterAt = 0;
let _nativePushRegisterInFlight = false;
let _nativeDmNotificationPrimed = false;
let _nativeGeneralNotificationPrimed = false;
let _nativeDmUnreadMap = {};
let _lastGatewayNotificationStatus = 'idle';
let _lastShadowSyncStatus = 'idle';
let _activeChatConvoId = '';
let _activeGroupChat = { id: '', collection: '' };
let _feedScrollTop = 0;
let _pendingFeedScrollRestore = null;
let _notifDropdownCloseHandler = null;
let _contentScrollGestureAt = 0;
let _pendingMapRoute = null;
let _activeMapRouteLayer = null;
let _activeMapRouteMarkers = [];
let _mapLibreMap = null;
let _mapLibreMarkers = [];
let _mapLibrePopups = [];
let _bootReady = false;
let _bootFallbackTimer = null;
let _activeMapRouteSummary = null;
let _feedRestorePendingPaint = false;
const _nativeGeneralNotifIds = new Set();
let _feedSearchQuery = '';
let _exploreSearchQuery = '';
let _pendingCommentImageFile = null;
let _pendingReelCommentImageFile = null;
let _pendingNativeNotificationOpen = null;
let _commentReactionPopover = null;
let _reelCommentReplyTo = null;
let _sendingReelComment = false;
let _lastFeedCommentSubmit = { key: '', at: 0 };
let _lastReelCommentSubmit = { key: '', at: 0 };
let _sessionRecoveryInFlight = false;
const _authorPhotoCache = {};
const _userContextCache = {};
let _feedInlineSuggestionSlotsUsed = 0;
let _feedInlineSuggestedUserIds = new Set();
let _postTopCommentCache = new Map();

const ANON_PERSONA_THEMES = ['Anonymous', 'Campus Ghost', 'Res Phantom'];
const SOFT_FILTER_RULES = [
  { regex: /\b(kill\s+yourself|kys)\b/gi, replacement: '[removed]' },
  { regex: /\b(rape|raped|rapist)\b/gi, replacement: '[removed]' },
  { regex: /\b(nudes?|dick\s?pic|send\s?nudes?)\b/gi, replacement: '[removed]' },
  { regex: /\b(i\s+will\s+kill\s+you|i\s+am\s+going\s+to\s+kill\s+you|shoot\s+you|stab\s+you)\b/gi, replacement: '[removed]' }
];

function isPermissionDeniedError(err) {
  return err?.code === 'permission-denied' || /missing or insufficient permissions/i.test(err?.message || '');
}

function isInvalidSessionError(err) {
  const code = String(err?.code || '').toLowerCase();
  const message = String(err?.message || '').toLowerCase();
  return [
    'auth/id-token-expired',
    'auth/user-token-expired',
    'auth/user-disabled',
    'auth/invalid-user-token',
    'auth/user-not-found',
    'auth/requires-recent-login'
  ].includes(code) || /securetoken|invalid grant|token.*expired|user token/i.test(message);
}

async function recoverInvalidSession(err, context = 'Firebase startup') {
  if (!isInvalidSessionError(err)) return false;
  console.error(`${context}:`, err);
  if (_sessionRecoveryInFlight) return true;
  _sessionRecoveryInFlight = true;
  try { await auth.signOut(); } catch (_) {}
  state.user = null;
  state.profile = null;
  VERIFIED_UIDS.clear();
  showScreen('auth-screen');
  toast('Session expired. Log in again.');
  _sessionRecoveryInFlight = false;
  return true;
}

function isVerifiedUser(uid) {
  if (!uid) return false;
  if (VERIFIED_UIDS.has(uid)) return true;
  if (uid === state.profile?.id) return _isAdmin || !!state.profile?.manualVerified;
  return false;
}
function verifiedBadge(uid) { return isVerifiedUser(uid) ? '<span class="verified-badge" title="Verified">✔</span>' : ''; }

function clampText(v = '', max = 80) {
  const t = (v || '').replace(/\s+/g, ' ').trim();
  return t.length > max ? `${t.slice(0, max - 1)}…` : t;
}

const REACTION_OPTIONS = ['❤️', '😂', '🔥', '😮', '👏'];
const APPWRITE_PUSH_SYNC_URLS = (window.UNINO_APPWRITE_SYNC_URLS || [window.UNINO_APPWRITE_SYNC_URL || ''])
  .map(url => (url || '').trim())
  .filter(Boolean);
const APPWRITE_EVENT_SYNC_URLS = (window.UNINO_APPWRITE_EVENT_SYNC_URLS || [window.UNINO_APPWRITE_EVENT_SYNC_URL || ''])
  .map(url => (url || '').trim())
  .filter(Boolean);

function shouldMirrorToAppwrite() {
  // Existing users remain on Firebase unless explicitly flagged.
  return !!state.profile?.appwritePrimary;
}

async function postToAppwriteBridge(url, payload) {
  if (!auth.currentUser) throw new Error('missing-auth-user');
  const mergedPayload = {
    ...(payload || {}),
    firebaseApiKey: window.UNINO_FIREBASE_WEB_API_KEY || ''
  };
  let idToken = await auth.currentUser.getIdToken();
  let resp = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${idToken}`
    },
    body: JSON.stringify(mergedPayload)
  });
  if (resp.status !== 401) return resp;

  // Retry once with a forced-refresh token to avoid stale-session 401s.
  idToken = await auth.currentUser.getIdToken(true);
  resp = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${idToken}`
    },
    body: JSON.stringify(mergedPayload)
  });
  return resp;
}

async function syncEventWithAppwrite(eventType, payload = {}) {
  if (!APPWRITE_EVENT_SYNC_URLS.length || !auth.currentUser || !shouldMirrorToAppwrite()) return;
  try {
    let lastErr = null;
    for (const url of APPWRITE_EVENT_SYNC_URLS) {
      try {
        const resp = await postToAppwriteBridge(url, { eventType, payload });
        if (resp.ok) {
          const body = await resp.clone().json().catch(() => null);
          const mirror = body?.result?.mirror;
          _lastShadowSyncStatus = mirror
            ? (mirror.mirrored
              ? `${eventType}:ok/${mirror.entity || 'entity'}`
              : `${eventType}:${mirror.reason || 'not-mirrored'}`)
            : `${eventType}:ok`;
          refreshBackendDebugStatus();
          return;
        }
        const detail = await resp.text().catch(() => '');
        lastErr = new Error(`sync status ${resp.status} from ${url}${detail ? `: ${detail.slice(0, 160)}` : ''}`);
      } catch (e) {
        lastErr = e;
      }
    }
    if (lastErr) throw lastErr;
  } catch (e) {
    _lastShadowSyncStatus = `${eventType}:error`;
    refreshBackendDebugStatus(`shadow detail: ${String(e?.message || e).slice(0, 140)}`);
    console.warn('Appwrite event sync skipped:', e?.message || e);
  }
}

async function syncPushTokenWithAppwrite(action, userId, token) {
  if (!APPWRITE_PUSH_SYNC_URLS.length || !userId || !token || !auth.currentUser) return;
  try {
    let lastErr = null;
    for (const url of APPWRITE_PUSH_SYNC_URLS) {
      try {
        const resp = await postToAppwriteBridge(url, {
          action,
          userId,
          token,
          platform: window.Capacitor?.getPlatform?.() || 'android'
        });
        if (resp.ok) return;
        const detail = await resp.text().catch(() => '');
        lastErr = new Error(`sync status ${resp.status} from ${url}${detail ? `: ${detail.slice(0, 160)}` : ''}`);
      } catch (e) {
        lastErr = e;
      }
    }
    if (lastErr) throw lastErr;
  } catch (e) {
    console.warn('Appwrite push sync skipped:', e?.message || e);
  }
}

function refreshBackendDebugStatus(extra = '') {
  const host = document.getElementById('backend-debug-status');
  if (!host) return;
  const platform = isNativeApp() ? (window.Capacitor?.getPlatform?.() || 'native') : 'web';
  const tokenSummary = _nativePushToken ? `${_nativePushToken.slice(0, 12)}...` : 'none';
  const notifSummary = isNativeApp()
    ? `local=${_nativeLocalNotificationsReady ? 'granted' : 'not-ready'}`
    : `web=${typeof Notification === 'undefined' ? 'unsupported' : Notification.permission}`;
  const base = [
    `platform=${platform}`,
    `appwriteMirror=${shouldMirrorToAppwrite() ? 'on' : 'off'}`,
    `notifGateway=${_lastGatewayNotificationStatus}`,
    `shadowSync=${_lastShadowSyncStatus}`,
    `pushReady=${_nativePushReady ? 'yes' : 'no'}`,
    `pushToken=${tokenSummary}`,
    notifSummary
  ].join(' | ');
  host.textContent = extra ? `${base}\n${extra}` : base;
  const mirrorBtn = document.getElementById('backend-mirror-toggle-btn');
  if (mirrorBtn) mirrorBtn.textContent = shouldMirrorToAppwrite() ? 'Disable Mirror' : 'Enable Mirror';
}

async function dispatchNotificationGateway(targetId, data = {}, options = {}) {
  const allowSelf = !!options.allowSelf;
  if (!targetId || (!allowSelf && targetId === state.user?.uid)) return { skipped: true, reason: 'self-or-missing-target' };

  const mode = APPWRITE_EVENT_SYNC_URLS.length ? 'appwrite-native-fcm' : 'firestore-only';
  const { docId = null } = options;
  let appwriteStatus = 'skipped';
  let appwriteDetail = '';

  const notifType = String(data.type || 'generic');
  const notifPayload = data.payload && typeof data.payload === 'object' ? data.payload : {};
  const kind = notifPayload.kind
    || (notifPayload.convoId ? 'dm'
      : notifPayload.groupId ? 'group'
      : notifPayload.streamId ? 'live'
      : notifPayload.postId ? 'post'
      : notifType === 'message' ? 'dm'
      : notifType === 'group' ? 'group'
      : 'app');
  const from = data.from && typeof data.from === 'object' ? data.from : {};
  const pushTitle = String(from.name || 'Unino').trim() || 'Unino';
  const pushBody = String(data.text || 'You have a new notification').trim() || 'You have a new notification';
  const bridgePayload = {
    mode,
    targetId,
    type: notifType,
    title: pushTitle,
    text: pushBody,
    body: pushBody,
    kind,
    channelId: (kind === 'dm' || kind === 'group') ? 'unibo-messages' : 'unibo-general',
    at: new Date().toISOString(),
    payload: notifPayload,
    from: {
      uid: String(from.uid || ''),
      name: pushTitle,
      photo: String(from.photo || '')
    }
  };

  if (APPWRITE_EVENT_SYNC_URLS.length && auth.currentUser) {
    try {
      let delivered = false;
      for (const url of APPWRITE_EVENT_SYNC_URLS) {
        const resp = await postToAppwriteBridge(url, {
          eventType: 'notification_dispatch',
          payload: bridgePayload
        });
        if (resp.ok) {
          const body = await resp.clone().json().catch(() => null);
          const push = body?.result?.push;
          appwriteStatus = push?.sent ? 'ok' : (push?.reason || 'ok');
          if (push && !push.sent && push.detail) appwriteDetail = String(push.detail);
          else if (push && !push.sent && push.reason) appwriteDetail = String(push.reason);
          delivered = !!push;
          if (delivered) break;
        } else {
          appwriteStatus = `http-${resp.status}`;
          appwriteDetail = await resp.text().catch(() => '');
        }
      }
      if (!delivered && appwriteStatus === 'skipped') appwriteStatus = 'failed';
    } catch (e) {
      appwriteStatus = 'error';
      appwriteDetail = e?.message || String(e);
    }
  }

  const col = db.collection('users').doc(targetId).collection('notifications');
  if (docId) await col.doc(docId).set(data);
  else await col.add(data);

  _lastGatewayNotificationStatus = `${mode}/${appwriteStatus}`;
  refreshBackendDebugStatus(appwriteDetail ? `notif detail: ${appwriteDetail.slice(0, 120)}` : '');

  return { ok: true, mode, appwriteStatus, appwriteDetail };
}

async function setAppwriteMirrorEnabled(enabled) {
  if (!state.user?.uid) return;
  try {
    await db.collection('users').doc(state.user.uid).set({
      appwritePrimary: !!enabled,
      appwriteMigrationPhase: enabled ? 'shadow' : 'firebase',
      appwriteUpdatedAt: FieldVal.serverTimestamp()
    }, { merge: true });
    if (state.profile) state.profile.appwritePrimary = !!enabled;
    refreshBackendDebugStatus(`appwriteMirror ${enabled ? 'enabled' : 'disabled'}`);
    toast(enabled ? 'Appwrite mirror enabled' : 'Appwrite mirror disabled');
  } catch (e) {
    console.error('setAppwriteMirrorEnabled failed', e);
    refreshBackendDebugStatus(`Mirror toggle failed: ${e?.message || e}`);
    toast('Could not update mirror setting');
  }
}

function toggleAppwriteMirror() {
  setAppwriteMirrorEnabled(!shouldMirrorToAppwrite());
}

function shadowSyncUserProfile(uid, profile = {}) {
  if (!uid) return;
  syncEventWithAppwrite('user_upsert', {
    uid,
    displayName: profile.displayName || '',
    email: profile.email || '',
    photoURL: profile.photoURL || '',
    major: profile.major || '',
    university: profile.university || '',
    updatedAt: new Date().toISOString()
  });
}

function shadowSyncPost(postId, post = {}) {
  if (!postId) return;
  syncEventWithAppwrite('post_upsert', {
    postId,
    authorId: post.authorId || '',
    authorName: post.authorName || '',
    content: post.content || '',
    mediaURL: post.videoURL || post.imageURL || '',
    visibility: post.visibility || 'public',
    createdAt: post.createdAt || new Date().toISOString(),
    updatedAt: new Date().toISOString()
  });
}

function shadowSyncComment(postId, commentId, comment = {}) {
  if (!postId || !commentId) return;
  syncEventWithAppwrite('comment_upsert', {
    postId,
    commentId,
    authorId: comment.authorId || '',
    authorName: comment.authorName || '',
    text: comment.text || '',
    createdAt: comment.createdAt || new Date().toISOString(),
    updatedAt: new Date().toISOString()
  });
}

function appwriteRootUrl(url) {
  try {
    const parsed = new URL(url);
    parsed.pathname = '/';
    parsed.search = '';
    parsed.hash = '';
    return parsed.toString();
  } catch {
    return url;
  }
}

async function runAppwriteBackendDiagnostics() {
  refreshBackendDebugStatus('Running Appwrite diagnostics...');
  const urls = [...new Set([...APPWRITE_PUSH_SYNC_URLS, ...APPWRITE_EVENT_SYNC_URLS])];
  if (!urls.length) {
    refreshBackendDebugStatus('No Appwrite bridge URLs configured.');
    return;
  }

  const lines = [];
  for (const url of urls) {
    const rootUrl = appwriteRootUrl(url);
    try {
      const rootResp = await fetch(rootUrl, { method: 'GET' });
      lines.push(`${rootUrl} -> GET / ${rootResp.status}`);
    } catch (e) {
      lines.push(`${rootUrl} -> GET / failed (${e?.message || 'network/cors'})`);
    }
  }

  const eventUrl = APPWRITE_EVENT_SYNC_URLS[0] || '';
  if (eventUrl && auth.currentUser) {
    try {
      const eventResp = await postToAppwriteBridge(eventUrl, {
        eventType: 'debug_probe',
        payload: {
          appVersion: APP_VERSION,
          platform: window.Capacitor?.getPlatform?.() || 'web',
          at: new Date().toISOString()
        }
      });
      let detail = '';
      if (!eventResp.ok) detail = await eventResp.text().catch(() => '');
      lines.push(`${eventUrl} -> POST /event-sync ${eventResp.status}${detail ? ` (${detail.slice(0, 120)})` : ''}`);
    } catch (e) {
      lines.push(`${eventUrl} -> POST /event-sync failed (${e?.message || 'network/cors'})`);
    }
  }

  refreshBackendDebugStatus(lines.join('\n'));
}

async function runNotificationDiagnostics() {
  refreshBackendDebugStatus('Running notification diagnostics...');
  const lines = [];
  if (isNativeApp()) {
    const pushPlugin = getCapacitorPlugin('PushNotifications');
    const localPlugin = getCapacitorPlugin('LocalNotifications');
    lines.push(`pushPlugin=${pushPlugin ? 'present' : 'missing'}`);
    lines.push(`localPlugin=${localPlugin ? 'present' : 'missing'}`);
    try {
      await requestLocalNotificationPermission();
      refreshPushRegistration(true);
      lines.push('triggered push registration refresh');
      sendDebugLocalNotification();
      lines.push('sent local test notification');
    } catch (e) {
      lines.push(`push refresh failed (${e?.message || 'unknown'})`);
    }
  } else {
    if (typeof Notification === 'undefined') {
      lines.push('browser Notification API unsupported');
    } else {
      try {
        if (Notification.permission === 'default') await Notification.requestPermission();
      } catch (_) {}
      lines.push(`browser notification permission=${Notification.permission}`);
      sendDebugLocalNotification();
      lines.push('sent browser/local test notification');
    }
  }
  refreshBackendDebugStatus(lines.join('\n'));
}

async function sendGatewayNotificationProbe() {
  if (!state.user?.uid) return;
  const probe = {
    type: 'gateway_probe',
    text: 'Notification gateway probe delivered',
    payload: { probe: true, at: Date.now() },
    read: false,
    createdAt: FieldVal.serverTimestamp(),
    from: {
      uid: state.user.uid,
      name: state.profile?.displayName || 'Unibo',
      photo: state.profile?.photoURL || null
    }
  };
  try {
    const result = await dispatchNotificationGateway(state.user.uid, probe, { allowSelf: true });
    refreshBackendDebugStatus(`Gateway probe: ${result.mode}/${result.appwriteStatus}`);
    toast('Gateway probe sent');
  } catch (e) {
    refreshBackendDebugStatus(`Gateway probe failed: ${e?.message || e}`);
    toast('Gateway probe failed');
  }
}

async function runShadowSyncProbe() {
  if (!auth.currentUser || !APPWRITE_EVENT_SYNC_URLS.length || !state.user?.uid) {
    refreshBackendDebugStatus('Shadow probe unavailable: missing auth/session/url');
    return;
  }
  try {
    const resp = await postToAppwriteBridge(APPWRITE_EVENT_SYNC_URLS[0], {
      eventType: 'user_upsert',
      payload: {
        uid: state.user.uid,
        displayName: state.profile?.displayName || '',
        email: state.profile?.email || auth.currentUser.email || '',
        photoURL: state.profile?.photoURL || '',
        major: state.profile?.major || '',
        university: state.profile?.university || '',
        updatedAt: new Date().toISOString(),
        probe: true
      }
    });
    const body = await resp.clone().json().catch(() => null);
    const mirror = body?.result?.mirror;
    if (resp.ok && mirror?.mirrored) {
      _lastShadowSyncStatus = `probe:ok/${mirror.entity || 'user'}`;
      refreshBackendDebugStatus(`shadow probe wrote row ${mirror.rowId || '(id n/a)'}`);
      toast('Shadow probe wrote to Appwrite');
      return;
    }
    _lastShadowSyncStatus = `probe:${mirror?.reason || `http-${resp.status}`}`;
    refreshBackendDebugStatus(`shadow probe detail: ${JSON.stringify(mirror || body || {}).slice(0, 180)}`);
    toast('Shadow probe did not write row');
  } catch (e) {
    _lastShadowSyncStatus = 'probe:error';
    refreshBackendDebugStatus(`shadow probe failed: ${String(e?.message || e).slice(0, 140)}`);
    toast('Shadow probe failed');
  }
}

function sendDebugLocalNotification() {
  scheduleLocalNotification({
    id: hashStringToId(`debug-local-${Date.now()}`),
    title: 'Unibo Debug Test',
    body: 'If you see this, local notifications are working.',
    channelId: 'unibo-general',
    actionTypeId: 'app-preview',
    extra: { kind: 'debug' }
  });
  toast('Debug notification sent');
}

function normalizeReactionMap(raw = {}, likes = []) {
  const map = {};
  if (raw && typeof raw === 'object') {
    Object.entries(raw).forEach(([uid, emoji]) => {
      if (uid && REACTION_OPTIONS.includes(emoji)) map[uid] = emoji;
    });
  }
  (likes || []).forEach(uid => {
    if (uid && !map[uid]) map[uid] = '❤️';
  });
  return map;
}

function getReactionSummary(raw = {}, likes = []) {
  const map = normalizeReactionMap(raw, likes);
  const counts = {};
  Object.values(map).forEach(emoji => { counts[emoji] = (counts[emoji] || 0) + 1; });
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  return { map, entries, total: Object.keys(map).length };
}

function getUserReaction(raw = {}, likes = []) {
  return normalizeReactionMap(raw, likes)[state.user?.uid || ''] || '';
}

function renderReactionSummary(raw = {}, likes = [], extraClass = '') {
  const { entries, total } = getReactionSummary(raw, likes);
  if (!total) return '';
  const cls = extraClass ? ` ${extraClass}` : '';
  return `<span class="reaction-summary${cls}">${entries.slice(0, 3).map(([emoji, count]) => `<span class="reaction-chip">${emoji}${count > 1 ? `<b>${count}</b>` : ''}</span>`).join('')}<span class="reaction-total">${total}</span></span>`;
}

function syncLocalPostReactionState(postId, reactions, likes) {
  state.posts = (state.posts || []).map(post => post.id === postId ? { ...post, reactions, likes } : post);
  _reelVideos = (_reelVideos || []).map(post => post.id === postId ? { ...post, reactions, likes } : post);
}

function getLocalPostById(postId) {
  return (state.posts || []).find(post => post.id === postId)
    || (_reelVideos || []).find(post => post.id === postId)
    || null;
}

function getLocalMessageById(scope, messageId) {
  const lookup = scope === 'group' ? _gMsgLookup : _dmMsgLookup;
  return lookup.get(messageId) || null;
}

function syncLocalMessageReactionState(scope, messageId, reactions, likes = []) {
  const lookup = scope === 'group' ? _gMsgLookup : _dmMsgLookup;
  const existing = lookup.get(messageId);
  if (!existing) return;
  lookup.set(messageId, { ...existing, reactions, likes });
}

function refreshMessageReactionUI(scope, primaryId, messageId, collection = '') {
  const message = getLocalMessageById(scope, messageId);
  const row = document.getElementById(`msg-${messageId}`);
  if (!message || !row) return;
  const stack = row.querySelector('.msg-stack');
  if (!stack) return;
  const reactionSummary = renderReactionSummary(message.reactions || {}, message.likes || [], 'msg-inline');
  let line = stack.querySelector('.msg-reaction-line');
  if (!reactionSummary) {
    if (line) line.remove();
    return;
  }
  const action = scope === 'group'
    ? `event.stopPropagation();openMessageActionSheet('group','${primaryId}','${messageId}','${collection}')`
    : `event.stopPropagation();openMessageActionSheet('dm','${primaryId}','${messageId}')`;
  if (!line) {
    line = document.createElement('div');
    line.className = 'msg-reaction-line';
    stack.appendChild(line);
  }
  line.setAttribute('onclick', action);
  line.innerHTML = reactionSummary;
}

function buildLocalReactionResult(post = {}, emoji = '') {
  const uid = state.user?.uid;
  if (!uid) return { reactions: post.reactions || {}, likes: post.likes || [], nextReaction: '' };
  const reactions = normalizeReactionMap(post.reactions, post.likes || []);
  const current = reactions[uid] || '';
  if (!emoji || current === emoji) delete reactions[uid];
  else reactions[uid] = emoji;
  return {
    reactions,
    likes: Object.entries(reactions).filter(([, value]) => value === '❤️').map(([userId]) => userId),
    nextReaction: reactions[uid] || ''
  };
}

function renderPostStatsMarkup(post = {}) {
  const reactionSummary = renderReactionSummary(post.reactions, post.likes || [], 'compact');
  const commentCount = post.commentsCount || 0;
  return `${reactionSummary ? `<span class="stat-item">${reactionSummary}</span>` : ''}${commentCount ? `<span class="stat-item">${commentCount} comment${commentCount > 1 ? 's' : ''}</span>` : ''}`;
}

function renderPostLikeMarkup(post = {}) {
  const liked = (post.likes || []).includes(state.user?.uid);
  const myReaction = getUserReaction(post.reactions, post.likes || []);
  const { total } = getReactionSummary(post.reactions, post.likes || []);
  return `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="${liked ? 'var(--red)' : 'none'}" stroke="${liked ? 'var(--red)' : 'currentColor'}" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
    ${myReaction && myReaction !== '❤️' ? myReaction : (total || 'Like')}
  `;
}

function renderCommentLikeMarkup(liked = false, total = 0, currentReaction = '') {
  const showEmoji = currentReaction && currentReaction !== '❤️';
  return `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="${liked ? 'var(--red)' : 'none'}" stroke="${liked ? 'var(--red)' : 'currentColor'}" stroke-width="2" aria-hidden="true"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
    ${showEmoji ? `<span class="comment-like-emoji">${currentReaction}</span>` : ''}
    ${total ? `<span class="comment-like-count">${total}</span>` : ''}
  `;
}

function refreshPostCardsUI(postId) {
  const post = getLocalPostById(postId);
  if (!post) return;
  const liked = (post.likes || []).includes(state.user?.uid);
  const myReaction = getUserReaction(post.reactions, post.likes || []);
  document.querySelectorAll(`.post-card[data-post-id="${postId}"]`).forEach(card => {
    const stats = card.querySelector('.post-stats');
    if (stats) stats.innerHTML = renderPostStatsMarkup(post);
    card.querySelectorAll('.post-like-action').forEach(btn => {
      btn.classList.toggle('liked', liked);
      btn.classList.toggle('reacted', !!myReaction && myReaction !== '❤️');
      btn.innerHTML = renderPostLikeMarkup(post);
    });
    card.querySelectorAll('.modal-like-action').forEach(btn => {
      btn.classList.toggle('liked', liked);
      btn.textContent = `❤ ${getReactionSummary(post.reactions, post.likes || []).total || 'Like'}`;
    });
  });
}

function initProfilePostInteractions() {
  const host = $('#profile-tab-content');
  if (!host) return;
  requestAnimationFrame(() => bindPostReactionLongPress(host));
}

async function updateDocReaction(ref, emoji, options = {}) {
  const uid = state.user?.uid;
  if (!uid) return null;
  let result = null;
  await db.runTransaction(async tx => {
    const snap = await tx.get(ref);
    if (!snap.exists) return;
    const data = snap.data() || {};
    const reactions = normalizeReactionMap(data.reactions, options.includeLikes ? (data.likes || []) : []);
    const current = reactions[uid] || '';
    if (!emoji || current === emoji) delete reactions[uid];
    else reactions[uid] = emoji;
    const update = { reactions };
    if (options.includeLikes) {
      update.likes = Object.entries(reactions).filter(([, value]) => value === '❤️').map(([userId]) => userId);
    }
    tx.update(ref, update);
    result = {
      before: data,
      reactions,
      likes: update.likes || data.likes || [],
      nextReaction: reactions[uid] || ''
    };
  });
  return result;
}

function refreshReelReactionUI(postId) {
  const post = (_reelVideos || []).find(item => item.id === postId);
  const slide = document.querySelector(`.reel-slide[data-post-id="${postId}"]`);
  if (!post || !slide) return;
  const { total } = getReactionSummary(post.reactions, post.likes || []);
  const liked = (post.likes || []).includes(state.user?.uid);
  const myReaction = getUserReaction(post.reactions, post.likes || []);
  const likeBtn = slide.querySelector('.reel-like-btn');
  if (likeBtn) {
    likeBtn.classList.toggle('liked', liked);
    const svg = likeBtn.querySelector('svg');
    if (svg) {
      svg.setAttribute('fill', liked ? '#ff4757' : 'none');
      svg.setAttribute('stroke', liked ? '#ff4757' : '#fff');
    }
    const count = likeBtn.querySelector('span');
    if (count) count.textContent = total || '';
  }
  const reactBtn = slide.querySelector('.reel-react-btn');
  if (reactBtn) {
    reactBtn.classList.toggle('reacted', !!myReaction && myReaction !== '❤️');
    const label = reactBtn.querySelector('span');
    if (label) label.textContent = myReaction && myReaction !== '❤️' ? myReaction : 'React';
  }
}

function getCapacitorPlugin(name) {
  return window.Capacitor?.Plugins?.[name] || null;
}

function isNativeApp() {
  const cap = window.Capacitor;
  if (!cap) return false;
  if (typeof cap.isNativePlatform === 'function') return cap.isNativePlatform();
  const platform = cap.getPlatform?.();
  return platform === 'android' || platform === 'ios';
}

function hashStringToId(value = '') {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = ((hash << 5) - hash) + value.charCodeAt(index);
    hash |= 0;
  }
  return Math.abs(hash % 2147480000) + 1;
}

function tokenDocId(token = '') {
  return btoa(token).replace(/[^a-zA-Z0-9]/g, '').slice(0, 120) || String(hashStringToId(token));
}

async function syncNativeStatusBar() {
  if (!isNativeApp()) return;
  const statusBar = getCapacitorPlugin('StatusBar');
  if (!statusBar) return;
  const darkTheme = document.documentElement.getAttribute('data-theme') === 'dark';
  try { await statusBar.setOverlaysWebView({ overlay: false }); } catch (e) {}
  try { await statusBar.setBackgroundColor({ color: darkTheme ? '#12121F' : '#FFFFFF' }); } catch (e) {}
  try { await statusBar.setStyle({ style: darkTheme ? 'LIGHT' : 'DARK' }); } catch (e) {}
  // Use the native-injected status bar height; fall back to a safe Android default
  const injected = getComputedStyle(document.documentElement).getPropertyValue('--native-status-bar').trim();
  const safePx = (injected && injected !== '0px') ? injected : '28px';
  document.documentElement.style.setProperty('--app-safe-top', safePx);
}

async function savePushTokenForCurrentUser(token) {
  if (!token || !state.user) return;
  _nativePushToken = token;
  try {
    const previousOwner = localStorage.getItem('unino-push-owner') || '';
    if (previousOwner && previousOwner !== state.user.uid) {
      await db.collection('users').doc(previousOwner).collection('pushTokens').doc(tokenDocId(token)).delete().catch(() => {});
    }
    await db.collection('users').doc(state.user.uid).collection('pushTokens').doc(tokenDocId(token)).set({
      token,
      platform: window.Capacitor?.getPlatform?.() || 'android',
      updatedAt: FieldVal.serverTimestamp(),
      createdAt: FieldVal.serverTimestamp()
    }, { merge: true });
    // Optional dual-write bridge for Appwrite migration (no-op unless configured).
    syncPushTokenWithAppwrite('upsert', state.user.uid, token).catch(() => {});
    localStorage.setItem('unino-push-owner', state.user.uid);
  } catch (e) {
    console.warn('Push token save failed:', e);
  }
}

async function removePushTokenForUser(userId, token = _nativePushToken) {
  if (!userId || !token) return;
  try {
    await db.collection('users').doc(userId).collection('pushTokens').doc(tokenDocId(token)).delete().catch(() => {});
    syncPushTokenWithAppwrite('delete', userId, token).catch(() => {});
    if ((localStorage.getItem('unino-push-owner') || '') === userId) localStorage.removeItem('unino-push-owner');
  } catch (e) {
    console.warn('Push token cleanup failed:', e);
  }
}

function appIsForeground() {
  return _nativeAppIsActive && !document.hidden;
}

function noteContentScrollGesture() {
  _contentScrollGestureAt = Date.now();
}

function isRecentContentScrollGesture(windowMs = 520) {
  return (Date.now() - (_contentScrollGestureAt || 0)) <= windowMs;
}

function closeNotifDropdown() {
  const dd = $('#notif-dropdown');
  if (dd) dd.style.display = 'none';
  if (_notifDropdownCloseHandler) {
    document.removeEventListener('click', _notifDropdownCloseHandler, true);
    _notifDropdownCloseHandler = null;
  }
}

function clearTransientUi() {
  const notifDropdown = $('#notif-dropdown');
  if (notifDropdown?.style.display === 'block') {
    notifDropdown.style.display = 'none';
    return true;
  }
  const imageView = $('#img-view');
  if (imageView?.style.display === 'block' || imageView?.style.display === 'flex') {
    closeGalleryViewer();
    return true;
  }
  const modal = $('#modal-bg');
  if (modal?.style.display === 'block' || modal?.style.display === 'flex') {
    closeModal();
    return true;
  }
  const storyViewer = $('#story-viewer');
  if (storyViewer?.style.display === 'block' || storyViewer?.style.display === 'flex') {
    closeStoryViewer();
    return true;
  }
  if (document.getElementById('reel-comments-panel')) {
    closeReelComments();
    return true;
  }
  if (document.getElementById('video-hub')) {
    closeVideoHub();
    return true;
  }
  return false;
}

function handleAppBackAction(options = {}) {
  const { fromPopstate = false, allowExit = false } = options;
  const appPlugin = getCapacitorPlugin('App');

  if (!state.user) {
    if (allowExit && isNativeApp()) {
      appPlugin?.exitApp?.();
      return true;
    }
    if (fromPopstate) history.pushState({ app: true }, '');
    return true;
  }

  if (clearTransientUi()) return true;

  if ($('#chat-view')?.classList.contains('active')) {
    $('#chat-back')?.click();
    return true;
  }

  if ($('#group-chat-view')?.classList.contains('active')) {
    $('#gchat-back')?.click();
    return true;
  }

  if ($('#profile-view')?.classList.contains('active')) {
    $('#prof-back')?.click();
    return true;
  }

  if ($('#settings-view')?.classList.contains('active')) {
    showScreen('app');
  if (_pendingNativeNotificationOpen) {
    const pendingOpen = { ..._pendingNativeNotificationOpen };
    _pendingNativeNotificationOpen = null;
    setTimeout(() => handleNativeNotificationOpen(pendingOpen.extra || pendingOpen, pendingOpen.actionId || 'tap'), 200);
  }
    return true;
  }

  if (state.page !== 'feed') {
    navigate('feed', { restoreFeed: true });
    return true;
  }

  if (allowExit && isNativeApp()) {
    appPlugin?.exitApp?.();
    return true;
  }

  if (fromPopstate) {
    history.pushState({ app: true, screen: 'app', page: 'feed' }, '');
  }
  return true;
}

async function scheduleLocalNotification(notification) {
  // Web fallback: use browser Notification API when not native
  if (!isNativeApp()) {
    if (typeof Notification !== 'undefined' && Notification.permission === 'granted') {
      try {
        const opts = { body: notification.body, icon: notification.largeIcon || '/icons/icon-192.png', tag: String(notification.id) };
        new Notification(notification.title, opts);
      } catch (_) {}
    }
    return;
  }
  if (!_nativeLocalNotificationsReady) return;
  const localNotifications = getCapacitorPlugin('LocalNotifications');
  if (!localNotifications) return;
  try {
    const notifPayload = {
      id: notification.id,
      title: notification.title,
      body: notification.body,
      schedule: { at: new Date(Date.now() + 50) },
      channelId: notification.channelId || 'unibo-general',
      actionTypeId: notification.actionTypeId || 'app-preview',
      extra: notification.extra || {}
    };
    if (notification.largeIcon) notifPayload.largeIcon = notification.largeIcon;
    await localNotifications.schedule({ notifications: [notifPayload] });
  } catch (e) {
    console.warn('Local notification failed:', e);
  }
}

async function initNativePushNotifications() {
  if (!isNativeApp() || !state.user) return;
  if (_nativePushRegisterInFlight) return;
  _nativePushRegisterInFlight = true;
  const pushNotifications = getCapacitorPlugin('PushNotifications');
  if (!pushNotifications) {
    console.warn('PushNotifications plugin unavailable — using local notification fallback');
    _nativePushReady = false;
    _nativePushRegisterInFlight = false;
    return;
  }

  if (!_nativePushListenersBound) {
    _nativePushListenersBound = true;

    pushNotifications.addListener('registration', token => {
      _nativePushReady = true;
      _nativePushLastRegisterAt = Date.now();
      savePushTokenForCurrentUser(token?.value || '');
    });

    pushNotifications.addListener('registrationError', error => {
      _nativePushReady = false;
      console.warn('Push registration error:', error);
      // Still allow local notification fallback to work
    });

    pushNotifications.addListener('pushNotificationReceived', notification => {
      const extra = notification?.data || {};
      scheduleLocalNotification({
        id: hashStringToId(`push-${notification?.id || notification?.title || Date.now()}`),
        title: notification?.title || 'Unibo',
        body: clampText(notification?.body || 'You have a new notification', 110),
        channelId: (extra.kind === 'dm' || extra.kind === 'group') ? 'unibo-messages' : 'unibo-general',
        actionTypeId: extra.kind === 'dm' ? 'dm-preview' : 'app-preview',
        extra
      });
    });

    pushNotifications.addListener('pushNotificationActionPerformed', event => {
      handleNativeNotificationOpen(event.notification?.data || {}, event.actionId || 'tap');
    });
  }

  try {
    const permStatus = await pushNotifications.checkPermissions();
    let receive = permStatus.receive;
    if (receive === 'prompt' || receive === 'prompt-with-rationale') {
      const requested = await pushNotifications.requestPermissions();
      receive = requested.receive;
    }
    if (receive !== 'granted') {
      _nativePushReady = false;
      console.warn('Push permission not granted:', receive);
      _nativePushRegisterInFlight = false;
      return;
    }
    await pushNotifications.register();
    _nativePushLastRegisterAt = Date.now();
  } catch (e) {
    _nativePushReady = false;
    console.warn('Push init failed:', e);
  } finally {
    _nativePushRegisterInFlight = false;
  }
}

function refreshPushRegistration(force = false) {
  if (!isNativeApp() || !state.user) return;
  const ageMs = Date.now() - (_nativePushLastRegisterAt || 0);
  // Refresh every 6 hours to reduce stale-token delivery failures.
  if (force || !_nativePushLastRegisterAt || ageMs > 6 * 60 * 60 * 1000) {
    initNativePushNotifications().catch(() => {});
  }
}

// Request local notification permission — called only after login.
async function requestLocalNotificationPermission() {
  if (isNativeApp()) {
    const localNotifications = getCapacitorPlugin('LocalNotifications');
    if (!localNotifications) return;
    try {
      const perm = await localNotifications.requestPermissions();
      _nativeLocalNotificationsReady = perm.display === 'granted';
    } catch (e) {
      _nativeLocalNotificationsReady = false;
    }
  } else if (typeof Notification !== 'undefined' && Notification.permission === 'default') {
    try { await Notification.requestPermission(); } catch (_) {}
  }
}

function handleNativeNotificationOpen(extra = {}, actionId = 'tap') {
  if (!extra) return;
  if (!state.user) {
    _pendingNativeNotificationOpen = { extra, actionId };
    return;
  }
  if (extra.notifDocId) markNotifRead(extra.notifDocId);
  if (extra.kind === 'dm' && extra.convoId) {
    openChat(extra.convoId);
    if (actionId === 'reply') setTimeout(() => $('#chat-input')?.focus(), 250);
    return;
  }
  if (extra.kind === 'group' && extra.groupId) {
    openGroupChat(extra.groupId, extra.collection || 'groups');
    if (actionId === 'reply') setTimeout(() => $('#gchat-input')?.focus(), 250);
    return;
  }
  if (extra.kind === 'live' && extra.streamId) {
    openLiveStreamFromFeed(extra.streamId, !!extra.profileId && extra.profileId === state.user?.uid);
    return;
  }
  if (extra.postId) {
    viewPost(extra.postId);
    return;
  }
  if (extra.profileId && extra.profileId !== 'anonymous') openProfile(extra.profileId);
}

function maybeNotifyForUnreadDMs(conversations = []) {
  if (!state.user) return;
  // Always track and notify — don't skip when FCM is ready since it may fail
  const nextUnreadMap = {};
  const myUid = state.user.uid;
  conversations.forEach(conversation => {
    nextUnreadMap[conversation.id] = (conversation.unread || {})[myUid] || 0;
  });

  if (!_nativeDmNotificationPrimed) {
    _nativeDmNotificationPrimed = true;
    // On first load, notify for any already-unread conversations (messages while app was closed)
    conversations.forEach(conversation => {
      const unread = nextUnreadMap[conversation.id] || 0;
      if (unread <= 0) return;
      if (_activeChatConvoId === conversation.id) return;
      const index = conversation.participants.indexOf(myUid) === 0 ? 1 : 0;
      const otherUid = conversation.participants[index];
      const anonymousPeer = !!((conversation.anonymous || {})[otherUid]);
      const senderName = anonymousPeer ? getAnonDisplayName(conversation, myUid, otherUid) : ((conversation.participantNames || [])[index] || 'New message');
      const senderPhoto = anonymousPeer ? null : ((conversation.participantPhotos || [])[index] || null);
      scheduleLocalNotification({
        id: hashStringToId(`dm-${conversation.id}-${unread}`),
        title: senderName,
        body: clampText(conversation.lastMessage || 'Sent you a message', 110),
        largeIcon: senderPhoto || undefined,
        channelId: 'unibo-messages',
        actionTypeId: 'dm-preview',
        extra: { kind: 'dm', convoId: conversation.id }
      });
    });
    _nativeDmUnreadMap = nextUnreadMap;
    return;
  }

  conversations.forEach(conversation => {
    const unread = nextUnreadMap[conversation.id] || 0;
    const previousUnread = _nativeDmUnreadMap[conversation.id] || 0;
    if (unread <= previousUnread) return;
    if (_activeChatConvoId === conversation.id && appIsForeground()) return;

    const index = conversation.participants.indexOf(myUid) === 0 ? 1 : 0;
    const otherUid = conversation.participants[index];
    const anonymousPeer = !!((conversation.anonymous || {})[otherUid]);
    const senderName = anonymousPeer ? getAnonDisplayName(conversation, myUid, otherUid) : ((conversation.participantNames || [])[index] || 'New message');
    const senderPhoto = anonymousPeer ? null : ((conversation.participantPhotos || [])[index] || null);
    scheduleLocalNotification({
      id: hashStringToId(`dm-${conversation.id}-${unread}`),
      title: senderName,
      body: clampText(conversation.lastMessage || 'Sent you a message', 110),
      largeIcon: senderPhoto || undefined,
      channelId: 'unibo-messages',
      actionTypeId: 'dm-preview',
      extra: { kind: 'dm', convoId: conversation.id }
    });
  });

  _nativeDmUnreadMap = nextUnreadMap;
}

function maybeNotifyForGeneralNotifications(notifications = []) {
  if (!state.user) return;
  // Always process notifications regardless of push status — FCM may fail silently
  if (!_nativeGeneralNotificationPrimed) {
    _nativeGeneralNotificationPrimed = true;
    // On first run, mark existing unreads as already seen to avoid flooding
    notifications.filter(n => !n.read).forEach(n => _nativeGeneralNotifIds.add(n.id));
    return;
  }

  notifications.forEach(notification => {
    if (notification.read || _nativeGeneralNotifIds.has(notification.id)) return;
    if (appIsForeground() && $('#notif-dropdown')?.style.display === 'block') {
      _nativeGeneralNotifIds.add(notification.id);
      return;
    }

    const fromName = notification.from?.name || 'Unibo';
    const fromPhoto = notification.from?.photo || null;
    const kind = notification.payload?.convoId
      ? 'dm'
      : notification.payload?.groupId
        ? 'group'
        : notification.payload?.streamId
          ? 'live'
          : notification.payload?.postId
            ? 'post'
            : 'app';
    scheduleLocalNotification({
      id: hashStringToId(`notif-${notification.id}`),
      title: fromName,
      body: clampText(notification.text || 'You have a new notification', 110),
      largeIcon: fromPhoto || undefined,
      channelId: kind === 'dm' || kind === 'group' ? 'unibo-messages' : 'unibo-general',
      actionTypeId: 'app-preview',
      extra: {
        kind,
        convoId: notification.payload?.convoId || '',
        groupId: notification.payload?.groupId || '',
        collection: notification.payload?.collection || 'groups',
        postId: notification.payload?.postId || '',
        streamId: notification.payload?.streamId || '',
        profileId: notification.from?.uid || '',
        notifDocId: notification.id
      }
    });
    _nativeGeneralNotifIds.add(notification.id);
  });
}

async function initNativeShell() {
  if (_nativeShellReady || !isNativeApp()) return;
  _nativeShellReady = true;
  await syncNativeStatusBar();

  const appPlugin = getCapacitorPlugin('App');
  const localNotifications = getCapacitorPlugin('LocalNotifications');

  if (appPlugin && !_nativeBackListenerBound) {
    _nativeBackListenerBound = true;
    appPlugin.addListener('backButton', () => {
      handleAppBackAction({ allowExit: true });
    });
    appPlugin.addListener('appStateChange', ({ isActive }) => {
      _nativeAppIsActive = !!isActive;
      if (isActive) {
        markActivity();
        refreshPushRegistration();
      }
      else stopAllVideos();
    });
  }

  // Local notification channels / action types are set up here but
  // permission is NOT requested until after the user logs in (enterApp).
  if (localNotifications) {
    try {
      await localNotifications.createChannel({
        id: 'unibo-messages',
        name: 'Messages',
        description: 'Direct and group message notifications',
        importance: 5,
        visibility: 1
      });
      await localNotifications.createChannel({
        id: 'unibo-general',
        name: 'Activity',
        description: 'General Unibo notifications',
        importance: 4,
        visibility: 1
      });
    } catch (e) {}
    try {
      await localNotifications.registerActionTypes({
        types: [
          {
            id: 'dm-preview',
            actions: [
              { id: 'open', title: 'Open' },
              { id: 'reply', title: 'Reply' }
            ]
          },
          {
            id: 'app-preview',
            actions: [
              { id: 'open', title: 'Open' }
            ]
          }
        ]
      });
    } catch (e) {}

    if (!_nativeNotificationListenersBound) {
      _nativeNotificationListenersBound = true;
      localNotifications.addListener('localNotificationActionPerformed', event => {
        handleNativeNotificationOpen(event.notification?.extra || {}, event.actionId || 'tap');
      });
    }
  }

  if (!_nativeNotificationListenersBound) {
    _nativeNotificationListenersBound = true;
    window.addEventListener('unino:native-notification-open', event => {
      const detail = event?.detail || {};
      handleNativeNotificationOpen(detail.extra || detail, detail.actionId || 'tap');
    });
    const pending = window.__UNINO_PENDING_NOTIFICATION || null;
    if (pending) {
      window.__UNINO_PENDING_NOTIFICATION = null;
      setTimeout(() => handleNativeNotificationOpen(pending.extra || pending, pending.actionId || 'tap'), 120);
    }
  }
}

function primeInlineVideoPreviews(root = document) {
  if (!root?.querySelectorAll) return;
  root.querySelectorAll('video.inline-video-preview').forEach(video => {
    if (video.dataset.previewBound === '1') return;
    video.dataset.previewBound = '1';
    const markReady = () => video.classList.add('ready');
    video.addEventListener('loadeddata', markReady, { once: true });
    video.addEventListener('canplay', markReady, { once: true });
    video.addEventListener('error', markReady, { once: true });
    if (video.readyState >= 2) markReady();
  });
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
  requestAnimationFrame(() => {
    msgsEl.scrollTop = msgsEl.scrollHeight;
    // Double-ensure on mobile after layout settles
    setTimeout(() => { msgsEl.scrollTop = msgsEl.scrollHeight; }, 50);
  });
}

function setupViewportFollow(msgsEl) {
  if (!msgsEl || !window.visualViewport) return () => {};
  let vvTimer;
  const onVv = () => {
    clearTimeout(vvTimer);
    vvTimer = setTimeout(() => {
      // When keyboard opens/closes, scroll chat to bottom
      scrollToLatest(msgsEl);
      // Ensure the chat view fits the visual viewport
      const chatView = msgsEl.closest('.screen');
      if (chatView) {
        chatView.style.height = window.visualViewport.height + 'px';
        chatView.style.top = window.visualViewport.offsetTop + 'px';
      }
    }, 60);
  };
  window.visualViewport.addEventListener('resize', onVv);
  window.visualViewport.addEventListener('scroll', onVv);
  return () => {
    window.visualViewport.removeEventListener('resize', onVv);
    window.visualViewport.removeEventListener('scroll', onVv);
    // Reset styles
    const chatView = msgsEl.closest('.screen');
    if (chatView) { chatView.style.height = ''; chatView.style.top = ''; }
  };
}

// ─── Helpers ─────────────────────────────────────
function colorFor(n) {
  let h = 0;
  for (let i = 0; i < (n || '').length; i++) h = n.charCodeAt(i) + ((h << 5) - h);
  return COLORS[Math.abs(h) % COLORS.length];
}

function normalizeModules(modules = []) {
  return (modules || [])
    .map(m => (m || '')
      .toString()
      .trim()
      .toUpperCase()
      .replace(/[^A-Z0-9\s-]/g, '')
      .replace(/[\s_-]+/g, '')
    )
    .filter(Boolean);
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

function getUserCoords(profile = {}) {
  const lat = Number(profile?.geoLat);
  const lng = Number(profile?.geoLng);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
  return { lat, lng };
}

function toRadians(deg) {
  return (deg * Math.PI) / 180;
}

function distanceKmBetween(a, b) {
  if (!a || !b) return Infinity;
  const earthRadiusKm = 6371;
  const dLat = toRadians(b.lat - a.lat);
  const dLng = toRadians(b.lng - a.lng);
  const s1 = Math.sin(dLat / 2);
  const s2 = Math.sin(dLng / 2);
  const aa = s1 * s1 + Math.cos(toRadians(a.lat)) * Math.cos(toRadians(b.lat)) * s2 * s2;
  return earthRadiusKm * 2 * Math.atan2(Math.sqrt(aa), Math.sqrt(1 - aa));
}

function formatDistanceText(distanceKm) {
  if (!Number.isFinite(distanceKm)) return '';
  if (distanceKm < 1) return `${Math.max(50, Math.round(distanceKm * 1000))} m away`;
  return `${distanceKm.toFixed(distanceKm < 10 ? 1 : 0)} km away`;
}

function getNearbySignal(meProfile = {}, otherProfile = {}) {
  const myCoords = getUserCoords(meProfile);
  const theirCoords = getUserCoords(otherProfile);
  if (myCoords && theirCoords) {
    const distanceKm = distanceKmBetween(myCoords, theirCoords);
    const score = distanceKm <= 0.35 ? 5
      : distanceKm <= 0.75 ? 4
      : distanceKm <= 1.5 ? 3
      : distanceKm <= 3 ? 2
      : distanceKm <= 5 ? 1
      : 0;
    return { score, distanceKm, source: 'gps' };
  }
  return {
    score: addressMatchScore(meProfile.address || '', otherProfile.address || ''),
    distanceKm: null,
    source: 'address'
  };
}

function getRadarCenterCoords() {
  return getUserCoords(state.profile) || { lat: -26.6840, lng: 27.0945 };
}

function getLocationPromptKey(uid = '') {
  return `unino-location-prompt-${uid}`;
}

async function saveCurrentGpsLocation(options = {}) {
  const { silent = false } = options;
  if (!state.user || !navigator.geolocation) {
    if (!silent) toast('GPS location is not available on this device');
    return null;
  }
  try {
    if (!silent) toast('Getting your GPS location...');
    const coords = await new Promise((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(resolve, reject, {
        enableHighAccuracy: true,
        maximumAge: 60000,
        timeout: 10000
      });
    });
    const lat = Number(coords.coords.latitude.toFixed(6));
    const lng = Number(coords.coords.longitude.toFixed(6));
    const nearestCampus = (typeof CAMPUS_LOCATIONS !== 'undefined' ? CAMPUS_LOCATIONS : []).reduce((best, loc) => {
      const dist = distanceKmBetween({ lat, lng }, { lat: loc.lat, lng: loc.lng });
      return !best || dist < best.distanceKm ? { ...loc, distanceKm: dist } : best;
    }, null);
    const updates = {
      geoLat: lat,
      geoLng: lng,
      geoSource: 'gps',
      geoUpdatedAt: FieldVal.serverTimestamp(),
      address: state.profile?.address || nearestCampus?.name || ''
    };
    await db.collection('users').doc(state.user.uid).update(updates);
    Object.assign(state.profile, {
      ...updates,
      geoUpdatedAt: new Date()
    });
    localStorage.setItem(getLocationPromptKey(state.user.uid), 'granted');
    const statusEl = $('#gps-location-status');
    if (statusEl) statusEl.textContent = `GPS saved: ${lat}, ${lng}`;
    if (state.page === 'explore') loadExploreUsers();
    if (!silent) toast('Location saved. Unibo will not track you continuously.');
    return { lat, lng };
  } catch (err) {
    if (await recoverInvalidSession(err, 'GPS save failed')) return null;
    console.error(err);
    if (!silent) openLocationHelpModal(getLocationErrorMessage(err));
    return null;
  }
}

function maybePromptForGpsLocation() {
  if (!state.user || !state.profile || getUserCoords(state.profile) || !navigator.geolocation) return;
  const promptKey = getLocationPromptKey(state.user.uid);
  if (localStorage.getItem(promptKey)) return;
  localStorage.setItem(promptKey, 'seen');
  setTimeout(async () => {
    const allow = window.confirm('Use your current GPS location for radar and nearby students? This saves your location once and does not track you continuously.');
    if (!allow) return;
    await saveCurrentGpsLocation();
  }, 700);
}

async function clearAppCache() {
  try {
    if ('caches' in window) {
      const keys = await caches.keys();
      await Promise.all(keys.map(key => caches.delete(key)));
    }
  } catch (e) { console.error('cache clear', e); }
  toast('Refreshing app...');
  setTimeout(() => {
    const url = new URL(window.location.href);
    url.searchParams.set('refresh', Date.now().toString());
    window.location.replace(url.toString());
  }, 250);
}

function getLocationErrorMessage(err) {
  if (!err) return 'Could not get your GPS location.';
  if (err.code === 1) return 'Location permission was denied. Turn on browser location permission for this site.';
  if (err.code === 2) return 'Your device location is off or unavailable. Turn on Location/GPS in your phone settings and try again.';
  if (err.code === 3) return 'Location lookup timed out. Move to a clearer area or retry.';
  return err.message || 'Could not get your GPS location.';
}

function openLocationHelpModal(message = '') {
  openModal(`
    <div class="modal-header"><h2>Location Needed</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" style="padding:18px 16px">
      <p style="font-size:14px;line-height:1.5;color:var(--text-secondary);margin-bottom:12px">${esc(message || 'Turn on location permissions so Unibo can save your current GPS position once for radar and nearby matching.')}</p>
      <div style="background:var(--bg-tertiary);border:1px solid var(--border);border-radius:12px;padding:12px;margin-bottom:14px;font-size:13px;color:var(--text-secondary);line-height:1.5">
        <div style="font-weight:600;color:var(--text-primary);margin-bottom:6px">How to fix it</div>
        <div>1. Turn on Location/GPS in your phone settings.</div>
        <div>2. Allow location access for your browser.</div>
        <div>3. Tap Retry below.</div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button class="btn-primary" onclick="closeModal();saveCurrentGpsLocation()">Retry GPS</button>
        <button class="btn-outline" onclick="closeModal();editProfile()">Open Edit Profile</button>
      </div>
    </div>
  `);
}

async function ensureUserContextCache(userIds = [], options = {}) {
  const force = !!options.force;
  const missing = [...new Set((userIds || []).filter(Boolean))].filter(uid => force || !_userContextCache[uid] || _userContextCache[uid]?.pending);
  if (!missing.length) return;
  await Promise.all(missing.map(async uid => {
    _userContextCache[uid] = { ...(force ? (_userContextCache[uid] || {}) : {}), pending: true };
    try {
      const doc = await db.collection('users').doc(uid).get();
      const next = doc.exists ? { id: doc.id, ...doc.data() } : null;
      _userContextCache[uid] = next;
      updateCachedUserRecord(uid, next || {});
    } catch (e) {
      console.error('user context', e);
      _userContextCache[uid] = null;
    }
  }));
}

function updateCachedUserRecord(userId, fields = {}) {
  if (!userId) return null;
  const next = { id: userId, ...(_userContextCache[userId] || {}), ...(fields || {}) };
  _userContextCache[userId] = next;
  const users = Array.isArray(_usersCache?.data) ? [..._usersCache.data] : [];
  const idx = users.findIndex(u => u.id === userId);
  if (idx >= 0) users[idx] = { ...users[idx], ...(fields || {}), id: userId };
  else users.unshift({ ...(fields || {}), id: userId });
  _usersCache = { data: users, expiresAt: Math.max(_usersCache?.expiresAt || 0, Date.now() + 60 * 1000) };
  return next;
}

function getResolvedUserIdentity(userId = '', fallbackName = 'User', fallbackPhoto = null) {
  const cached = _userContextCache[userId] || {};
  return {
    name: cached.displayName || cached.firstName || fallbackName || 'User',
    photo: cached.photoURL || fallbackPhoto || null,
    data: cached
  };
}

function buildInterestProfile() {
  const modules = new Set(normalizeModules(state.profile?.modules || []));
  const tags = new Set();
  (state.posts || []).forEach(existingPost => {
    if ((existingPost.likes || []).includes(state.user?.uid)) {
      normalizeModules(existingPost.moduleTags || []).forEach(tag => modules.add(tag));
      getPostHashTags(existingPost).forEach(tag => tags.add((tag || '').toLowerCase()));
    }
  });
  const tokens = new Set();
  modules.forEach(m => tokens.add(m.toLowerCase()));
  tags.forEach(t => tokens.add(t.toLowerCase()));
  return { modules, tags, tokens };
}

function textInterestScore(text = '', interestProfile = buildInterestProfile()) {
  const hay = (text || '').toLowerCase();
  if (!hay) return 0;
  let score = 0;
  interestProfile.modules.forEach(token => { if (hay.includes(token.toLowerCase())) score += 8; });
  interestProfile.tags.forEach(token => { if (hay.includes(token.toLowerCase())) score += 4; });
  return score;
}

function normalizeIdentityValue(value = '') {
  return String(value || '').trim().toLowerCase();
}


function normalizeSearchParts(value = '') {
  return String(value || '')
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

function userSearchScore(user = {}, query = '') {
  const parts = normalizeSearchParts(query);
  if (!parts.length) return 0;

  const displayName = user.displayName || `${user.firstName || ''} ${user.lastName || ''}`.trim();
  const nameParts = normalizeSearchParts(displayName);
  const majorParts = normalizeSearchParts(user.major || '');
  const uniParts = normalizeSearchParts(user.university || '');
  const addressParts = normalizeSearchParts(user.address || '');
  const moduleParts = normalizeSearchParts((user.modules || []).join(' '));
  const haystack = [...nameParts, ...majorParts, ...uniParts, ...addressParts, ...moduleParts];
  let score = 0;
  let matched = 0;
  const nameJoined = nameParts.join(' ');

  parts.forEach(part => {
    let partScore = 0;
    if (nameJoined === part) partScore = Math.max(partScore, 48);
    if (nameJoined.startsWith(part)) partScore = Math.max(partScore, 34);
    if (nameParts.some(token => token === part)) partScore = Math.max(partScore, 28);
    if (nameParts.some(token => token.startsWith(part))) partScore = Math.max(partScore, 22);
    if (moduleParts.some(token => token === part)) partScore = Math.max(partScore, 18);
    if (majorParts.some(token => token === part) || uniParts.some(token => token === part)) partScore = Math.max(partScore, 16);
    if (addressParts.some(token => token === part)) partScore = Math.max(partScore, 12);
    if (haystack.some(token => token.includes(part))) partScore = Math.max(partScore, 8);
    if (partScore > 0) {
      matched += 1;
      score += partScore;
    }
  });

  if (!matched) return 0;
  if (matched === parts.length) score += 14;
  if (displayName && normalizeSearchParts(displayName).join(' ').startsWith(parts.join(' '))) score += 10;
  return score;
}

function userMatchesQuery(user = {}, query = '') {
  return userSearchScore(user, query) > 0;
}

function genderAffinityScore(viewer = {}, candidate = {}) {
  const viewerGender = normalizeIdentityValue(viewer.gender);
  const viewerOrientation = normalizeIdentityValue(viewer.orientation);
  const candidateGender = normalizeIdentityValue(candidate.gender);
  if (!candidateGender) return 0;

  const isMale = g => g.includes('male') || g === 'm' || g === 'man';
  const isFemale = g => g.includes('female') || g === 'f' || g === 'woman';

  if (viewerOrientation.includes('lesbian')) return isFemale(candidateGender) ? 20 : -4;
  if (viewerOrientation.includes('gay')) {
    if (isMale(viewerGender)) return isMale(candidateGender) ? 20 : -4;
    if (isFemale(viewerGender)) return isFemale(candidateGender) ? 20 : -4;
    return 8;
  }
  if (viewerOrientation.includes('bi')) return 8;

  if (isMale(viewerGender)) return isFemale(candidateGender) ? 14 : -2;
  if (isFemale(viewerGender)) return isMale(candidateGender) ? 14 : -2;
  return 2;
}

function getCampusLocationById(locationId) {
  return CAMPUS_LOCATIONS.find(l => l.id === locationId) || null;
}

function getLocationDistanceBoost(coords) {
  const myCoords = getUserCoords(state.profile);
  if (!myCoords || !coords) return 0;
  const dist = distanceKmBetween(myCoords, coords);
  if (!Number.isFinite(dist)) return 0;
  if (dist <= 0.35) return 18;
  if (dist <= 0.75) return 14;
  if (dist <= 1.5) return 10;
  if (dist <= 3) return 6;
  if (dist <= 5) return 3;
  return 0;
}

function offsetLatLng(base, offsetIndex = 0, step = 0.00022) {
  const angle = (offsetIndex % 8) * (Math.PI / 4);
  const ring = Math.floor(offsetIndex / 8) + 1;
  return {
    lat: base.lat + Math.cos(angle) * step * ring,
    lng: base.lng + Math.sin(angle) * step * ring
  };
}

function resolveMapPoint(base, occupied = [], anchor = null) {
  if (!base) return null;
  const threshold = 0.00018;
  let candidate = { ...base };
  let attempt = 0;
  while (attempt < 24) {
    const tooCloseToAnchor = anchor && distanceKmBetween(candidate, anchor) < 0.045;
    const tooCloseToOthers = occupied.some(point => Math.abs(point.lat - candidate.lat) < threshold && Math.abs(point.lng - candidate.lng) < threshold);
    if (!tooCloseToAnchor && !tooCloseToOthers) break;
    attempt += 1;
    candidate = offsetLatLng(base, attempt);
  }
  occupied.push(candidate);
  return candidate;
}

function openCommentActionSheet(postId, commentId, source = 'feed') {
  const mode = arguments[3] || 'owner';
  const title = mode === 'post-owner' ? 'Moderate Comment' : 'Comment';
  const actionLabel = mode === 'admin' ? 'Admin Remove' : mode === 'post-owner' ? 'Remove Comment' : 'Delete Comment';
  openModal(`
    <div class="modal-header"><h2>${title}</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" style="padding:16px">
      <button class="btn-primary btn-full" style="background:var(--red);border:none" onclick="deleteCommentThread('${postId}','${commentId}','${source}')">${actionLabel}</button>
      <button class="btn-secondary btn-full" style="margin-top:8px" onclick="closeModal()">Cancel</button>
    </div>
  `);
}

function openPostReactionPicker(postId, source = 'feed') {
  const post = (state.posts || []).find(item => item.id === postId) || (_reelVideos || []).find(item => item.id === postId) || {};
  const current = getUserReaction(post.reactions, post.likes || []);
  openModal(`
    <div class="modal-header"><h2>React</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" style="padding:16px">
      <div class="reaction-picker-row">
        ${REACTION_OPTIONS.map(emoji => `<button class="reaction-option ${current === emoji ? 'active' : ''}" onclick="reactToPost('${postId}','${emoji}','${source}')">${emoji}</button>`).join('')}
      </div>
      <button class="btn-secondary btn-full" style="margin-top:10px" onclick="closeModal()">Cancel</button>
    </div>
  `);
}

async function reactToPost(postId, emoji, source = 'feed') {
  const localPost = getLocalPostById(postId);
  const previousState = localPost ? {
    reactions: normalizeReactionMap(localPost.reactions, localPost.likes || []),
    likes: [...(localPost.likes || [])]
  } : null;
  closeModal();
  if (localPost) {
    const optimistic = buildLocalReactionResult(localPost, emoji);
    syncLocalPostReactionState(postId, optimistic.reactions, optimistic.likes);
    refreshPostCardsUI(postId);
    if (source === 'reel') refreshReelReactionUI(postId);
  }
  try {
    const result = await updateDocReaction(db.collection('posts').doc(postId), emoji, { includeLikes: true });
    if (!result) return;
    syncLocalPostReactionState(postId, result.reactions, result.likes);
    refreshPostCardsUI(postId);
    syncEventWithAppwrite('post_reaction', {
      postId,
      emoji,
      nextReaction: result.nextReaction || '',
      reactedAt: Date.now()
    }).catch(() => {});
    if (result.nextReaction && result.before.authorId && result.before.authorId !== state.user?.uid) {
      addNotification(result.before.authorId, 'like', 'reacted to your post', { postId });
    }
    if (source === 'reel') refreshReelReactionUI(postId);
  } catch (e) {
    if (previousState) {
      syncLocalPostReactionState(postId, previousState.reactions, previousState.likes);
      refreshPostCardsUI(postId);
      if (source === 'reel') refreshReelReactionUI(postId);
    }
    console.error(e);
    toast('Could not react right now');
  }
}

function closeCommentReactionPopover() {
  if (_commentReactionPopover?.parentNode) _commentReactionPopover.parentNode.removeChild(_commentReactionPopover);
  _commentReactionPopover = null;
  document.removeEventListener('pointerdown', handleCommentReactionPopoverOutside, true);
  document.removeEventListener('scroll', closeCommentReactionPopover, true);
}

function handleCommentReactionPopoverOutside(event) {
  if (_commentReactionPopover && !_commentReactionPopover.contains(event.target)) closeCommentReactionPopover();
}

function openCommentReactionPicker(postId, commentId, source = 'feed', current = '', event = null) {
  closeCommentReactionPopover();
  const anchor = event?.currentTarget || event?.target;
  if (!anchor?.getBoundingClientRect) return;
  const pop = document.createElement('div');
  pop.className = 'comment-reaction-popover';
  pop.innerHTML = REACTION_OPTIONS.map(emoji => `<button class="reaction-option ${current === emoji ? 'active' : ''}" type="button" data-emoji="${emoji}">${emoji}</button>`).join('');
  pop.addEventListener('click', ev => {
    const btn = ev.target.closest('.reaction-option');
    if (!btn) return;
    ev.preventDefault();
    ev.stopPropagation();
    reactToComment(postId, commentId, btn.dataset.emoji || '', source);
  });
  document.body.appendChild(pop);
  const rect = anchor.getBoundingClientRect();
  const popRect = pop.getBoundingClientRect();
  let top = rect.top - popRect.height - 10;
  if (top < 12) top = rect.bottom + 10;
  let left = rect.left + (rect.width / 2) - (popRect.width / 2);
  left = Math.max(12, Math.min(left, window.innerWidth - popRect.width - 12));
  pop.style.top = `${top}px`;
  pop.style.left = `${left}px`;
  _commentReactionPopover = pop;
  setTimeout(() => {
    document.addEventListener('pointerdown', handleCommentReactionPopoverOutside, true);
    document.addEventListener('scroll', closeCommentReactionPopover, true);
  }, 0);
}

async function reactToComment(postId, commentId, emoji, source = 'feed') {
  const previous = _commentStateCache[commentId]
    ? { reactions: { ...(_commentStateCache[commentId].reactions || {}) }, likes: [...(_commentStateCache[commentId].likes || [])] }
    : null;
  closeCommentReactionPopover();
  if (previous) {
    const optimistic = buildLocalReactionResult(previous, emoji);
    _commentStateCache[commentId] = { reactions: optimistic.reactions, likes: optimistic.likes };
    refreshCommentReactionUI(commentId, optimistic.reactions, optimistic.likes);
  }
  try {
    const result = await updateDocReaction(db.collection('posts').doc(postId).collection('comments').doc(commentId), emoji, { includeLikes: true });
    if (result) {
      _commentStateCache[commentId] = { reactions: result.reactions || {}, likes: result.likes || [] };
      refreshCommentReactionUI(commentId, result.reactions, result.likes || []);
    }
    syncEventWithAppwrite('comment_reaction', {
      postId,
      commentId,
      emoji,
      reactedAt: Date.now()
    }).catch(() => {});
  } catch (e) {
    if (previous) {
      _commentStateCache[commentId] = previous;
      refreshCommentReactionUI(commentId, previous.reactions, previous.likes || []);
    }
    console.error(e);
    toast('Could not react right now');
  }
}

function refreshCommentReactionUI(commentId, reactions = {}, likes = []) {
  const normalizedReactions = normalizeReactionMap(reactions, likes || []);
  _commentStateCache[commentId] = { reactions: normalizedReactions, likes: [...(likes || [])] };
  const liked = (likes || []).includes(state.user?.uid);
  const currentReaction = normalizedReactions[state.user?.uid] || '';
  const total = getReactionSummary(normalizedReactions, likes).total;
  ['c-', 'rc-'].forEach(prefix => {
    const row = document.getElementById(`${prefix}${commentId}`);
    if (!row) return;
    const likeBtn = row.querySelector('.c-act.like-only');
    if (likeBtn) {
      likeBtn.classList.toggle('liked', liked);
      likeBtn.classList.toggle('reacted', !!currentReaction && currentReaction !== '❤️');
      likeBtn.classList.toggle('has-emoji', !!currentReaction && currentReaction !== '❤️');
      likeBtn.innerHTML = renderCommentLikeMarkup(liked, total, currentReaction);
    }
  });
}

function getMessageDocRef(scope, primaryId, messageId, collection = '') {
  if (scope === 'group') return db.collection(collection).doc(primaryId).collection('messages').doc(messageId);
  return db.collection('conversations').doc(primaryId).collection('messages').doc(messageId);
}

function openMessageActionSheet(scope, primaryId, messageId, collection = '') {
  const lookup = scope === 'group' ? _gMsgLookup : _dmMsgLookup;
  const message = lookup.get(messageId);
  if (!message) return;
  const current = getUserReaction(message.reactions);
  const isMine = message.senderId === state.user?.uid;
  openModal(`
    <div class="modal-header"><h2>Message</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" style="padding:16px">
      <div class="reaction-picker-row">
        ${REACTION_OPTIONS.map(emoji => `<button class="reaction-option ${current === emoji ? 'active' : ''}" onclick="reactToMessage('${scope}','${primaryId}','${messageId}','${emoji}','${collection}')">${emoji}</button>`).join('')}
      </div>
      ${isMine ? `<button class="btn-primary btn-full" style="margin-top:12px;background:var(--red);border:none" onclick="deleteMessage('${scope}','${primaryId}','${messageId}','${collection}')">Delete Message</button>` : ''}
      <button class="btn-secondary btn-full" style="margin-top:8px" onclick="closeModal()">Cancel</button>
    </div>
  `);
}

async function reactToMessage(scope, primaryId, messageId, emoji, collection = '') {
  const localMessage = getLocalMessageById(scope, messageId);
  if (!localMessage) return;
  const previousState = {
    reactions: normalizeReactionMap(localMessage.reactions, localMessage.likes || []),
    likes: [...(localMessage.likes || [])]
  };
  closeModal();
  try {
    const optimistic = buildLocalReactionResult(localMessage, emoji);
    syncLocalMessageReactionState(scope, messageId, optimistic.reactions, optimistic.likes);
    refreshMessageReactionUI(scope, primaryId, messageId, collection);
    const result = await updateDocReaction(getMessageDocRef(scope, primaryId, messageId, collection), emoji, { includeLikes: true });
    if (result) {
      syncLocalMessageReactionState(scope, messageId, result.reactions, result.likes);
      refreshMessageReactionUI(scope, primaryId, messageId, collection);
    }
  } catch (e) {
    syncLocalMessageReactionState(scope, messageId, previousState.reactions, previousState.likes);
    refreshMessageReactionUI(scope, primaryId, messageId, collection);
    console.error(e);
    toast('Could not react right now');
  }
}

async function syncThreadLastMessage(scope, primaryId, collection = '') {
  try {
    const parentRef = scope === 'group' ? db.collection(collection || 'groups').doc(primaryId) : db.collection('conversations').doc(primaryId);
    const snap = await parentRef.collection('messages').orderBy('createdAt', 'desc').limit(12).get();
    const items = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    const latest = items.find(m => !m.deleted && m.type !== 'deleted');
    let lastMessage = '';
    if (latest) {
      if (latest.locationPin?.label) lastMessage = `📍 ${latest.locationPin.label}`;
      else if (latest.type === 'poll' && latest.poll?.question) lastMessage = `📊 ${latest.poll.question}`;
      else if (latest.audioURL) lastMessage = '🎤 Voice';
      else if (latest.imageURL) lastMessage = latest.mediaType === 'video' ? '📹 Video' : (latest.text || '📷 Photo');
      else lastMessage = latest.text || '';
    }
    await parentRef.set({ lastMessage, updatedAt: FieldVal.serverTimestamp() }, { merge: true });
  } catch (_) {}
}

async function deleteMessage(scope, primaryId, messageId, collection = '') {
  try {
    const lookup = scope === 'group' ? _gMsgLookup : _dmMsgLookup;
    const message = lookup.get(messageId) || {};
    const ref = getMessageDocRef(scope, primaryId, messageId, collection);
    const hardDelete = message.type === 'poll' || !!message.locationPin;
    if (hardDelete) {
      await ref.delete();
      lookup.delete(messageId);
    } else {
      await ref.set({
        deleted: true,
        deletedAt: FieldVal.serverTimestamp(),
        text: '',
        imageURL: null,
        audioURL: null,
        type: 'deleted',
        payload: null,
        poll: null,
        locationPin: null,
        reactions: {}
      }, { merge: true });
    }
    await syncThreadLastMessage(scope, primaryId, collection);
    closeModal();
    toast('Message deleted');
  } catch (e) {
    console.error(e);
    toast('Could not delete message');
  }
}

function bindMessageLongPress(container, scope, primaryId, collection = '') {
  if (!container) return;
  container.querySelectorAll('.msg-bubble[data-message-id]').forEach(item => {
    const messageId = item.getAttribute('data-message-id') || '';
    if (!messageId) return;
    let timer = null;
    let didOpen = false;
    const start = () => {
      clearTimeout(timer);
      didOpen = false;
      item.classList.add('message-long-pressing');
      timer = setTimeout(() => {
        didOpen = true;
        item.classList.remove('message-long-pressing');
        if (navigator.vibrate) navigator.vibrate(18);
        openMessageActionSheet(scope, primaryId, messageId, collection);
      }, 360);
    };
    const clear = () => {
      clearTimeout(timer);
      timer = null;
      item.classList.remove('message-long-pressing');
    };
    item.oncontextmenu = e => {
      e.preventDefault();
      if (navigator.vibrate) navigator.vibrate(18);
      openMessageActionSheet(scope, primaryId, messageId, collection);
    };
    item.addEventListener('click', e => {
      if (!didOpen) return;
      e.preventDefault();
      e.stopPropagation();
      didOpen = false;
    }, true);
    item.addEventListener('touchstart', start, { passive: true });
    item.addEventListener('touchend', clear);
    item.addEventListener('touchmove', clear);
    item.addEventListener('touchcancel', clear);
    item.addEventListener('mousedown', start);
    item.addEventListener('mouseup', clear);
    item.addEventListener('mouseleave', clear);
  });
}

function bindPostReactionLongPress(container) {
  if (!container) return;
  container.querySelectorAll('.post-like-action[data-post-id]').forEach(item => {
    const postId = item.getAttribute('data-post-id') || '';
    const source = item.getAttribute('data-source') || 'feed';
    if (!postId) return;
    let timer = null;
    let didOpen = false;
    const start = () => {
      clearTimeout(timer);
      didOpen = false;
      timer = setTimeout(() => {
        didOpen = true;
        item.classList.add('react-holding');
        if (navigator.vibrate) navigator.vibrate(18);
        openPostReactionPicker(postId, source);
      }, 360);
    };
    const clear = () => {
      clearTimeout(timer);
      timer = null;
      item.classList.remove('react-holding');
    };
    item.oncontextmenu = e => {
      e.preventDefault();
      if (navigator.vibrate) navigator.vibrate(18);
      openPostReactionPicker(postId, source);
    };
    item.addEventListener('click', e => {
      if (!didOpen) return;
      e.preventDefault();
      e.stopPropagation();
      didOpen = false;
    }, true);
    item.addEventListener('touchstart', start, { passive: true });
    item.addEventListener('touchend', clear);
    item.addEventListener('touchmove', clear);
    item.addEventListener('touchcancel', clear);
    item.addEventListener('mousedown', start);
    item.addEventListener('mouseup', clear);
    item.addEventListener('mouseleave', clear);
  });
}

function bindCommentLongPress(container, postId, source = 'feed', postAuthorId = '') {
  if (!container) return;
  container.querySelectorAll('.comment-item[data-author-id]').forEach(item => {
    const authorId = item.getAttribute('data-author-id') || '';
    const commentId = item.getAttribute('data-comment-id') || '';
    const isCommentOwner = authorId === state.user?.uid;
    const isPostOwner = !!postAuthorId && postAuthorId === state.user?.uid;
    const canModerate = !!state.user?.uid && (_isAdmin || isCommentOwner || isPostOwner);
    if (!commentId || !canModerate) return;
    const mode = _isAdmin ? 'admin' : (isCommentOwner ? 'owner' : 'post-owner');
    let timer = null;
    let didOpen = false;
    const start = () => {
      clearTimeout(timer);
      didOpen = false;
      item.classList.add('comment-long-pressing');
      timer = setTimeout(() => {
        didOpen = true;
        item.classList.remove('comment-long-pressing');
        if (navigator.vibrate) navigator.vibrate(18);
        openCommentActionSheet(postId, commentId, source, mode);
      }, 360);
    };
    const clear = () => {
      clearTimeout(timer);
      timer = null;
      item.classList.remove('comment-long-pressing');
    };
    item.oncontextmenu = e => {
      e.preventDefault();
      if (navigator.vibrate) navigator.vibrate(18);
      openCommentActionSheet(postId, commentId, source, mode);
    };
    item.addEventListener('click', e => {
      if (!didOpen) return;
      e.preventDefault();
      e.stopPropagation();
      didOpen = false;
    }, true);
    item.addEventListener('touchstart', start, { passive: true });
    item.addEventListener('touchend', clear);
    item.addEventListener('touchmove', clear);
    item.addEventListener('touchcancel', clear);
    item.addEventListener('mousedown', start);
    item.addEventListener('mouseup', clear);
    item.addEventListener('mouseleave', clear);
  });
}

let _commentHeartHoldTimer = null;
let _commentHeartHoldConsumed = '';

function startCommentHeartHold(postId, commentId, source = 'feed', currentReaction = '', event) {
  if (event?.stopPropagation) event.stopPropagation();
  clearTimeout(_commentHeartHoldTimer);
  _commentHeartHoldConsumed = '';
  _commentHeartHoldTimer = setTimeout(() => {
    _commentHeartHoldConsumed = `${source}:${commentId}`;
    if (navigator.vibrate) navigator.vibrate(18);
    openCommentReactionPicker(postId, commentId, source, currentReaction || '', event);
  }, 360);
}

function endCommentHeartHold(event) {
  if (event?.stopPropagation) event.stopPropagation();
  clearTimeout(_commentHeartHoldTimer);
  _commentHeartHoldTimer = null;
}

function handleCommentHeartClick(postId, commentId, source = 'feed') {
  const key = `${source}:${commentId}`;
  if (_commentHeartHoldConsumed === key) {
    _commentHeartHoldConsumed = '';
    return;
  }
  if (source === 'reel') toggleReelCommentLike(commentId, postId);
  else toggleCommentLike(commentId, postId);
}

async function deleteCommentThread(postId, commentId, source = 'feed') {
  try {
    const uid = state.user?.uid;
    if (!uid) return toast('Sign in required');
    const postRef = db.collection('posts').doc(postId);
    const commentsRef = postRef.collection('comments');
    const [postDoc, commentDoc] = await Promise.all([
      postRef.get(),
      commentsRef.doc(commentId).get()
    ]);
    if (!postDoc.exists || !commentDoc.exists) {
      closeModal();
      return toast('Comment no longer exists');
    }
    const postAuthorId = postDoc.data()?.authorId || '';
    const commentAuthorId = commentDoc.data()?.authorId || '';
    const canDelete = _isAdmin || uid === commentAuthorId || uid === postAuthorId;
    if (!canDelete) {
      closeModal();
      return toast('You can only remove your own comments');
    }

    const snap = await commentsRef.limit(200).get();
    const comments = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    const toDelete = new Set([commentId]);
    let added = true;
    while (added) {
      added = false;
      comments.forEach(comment => {
        if (comment.replyTo && toDelete.has(comment.replyTo) && !toDelete.has(comment.id)) {
          toDelete.add(comment.id);
          added = true;
        }
      });
    }
    const batch = db.batch();
    toDelete.forEach(id => batch.delete(commentsRef.doc(id)));
    batch.update(db.collection('posts').doc(postId), { commentsCount: FieldVal.increment(-toDelete.size) });
    await batch.commit();
    closeModal();
    if (source === 'reel') openReelComments(postId, { scrollMode: 'preserve' });
    else openComments(postId, { scrollMode: 'preserve' });
    toast('Comment deleted');
  } catch (e) {
    console.error(e);
    toast('Could not delete comment');
  }
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

function allowAnonymousDMsFor(user = {}) {
  return user.allowAnonymousMessages !== false;
}

function bellNotifications(notifications = _notifications) {
  return (notifications || []).filter(notification => notification?.type !== 'message');
}

function anonNicknameKey(viewerUid, otherUid) {
  return `${viewerUid}_${otherUid}`;
}

function defaultAnonLabel(convoId = '') {
  const src = `${convoId || ''}`;
  let hash = 0;
  for (let i = 0; i < src.length; i++) hash = ((hash << 5) - hash) + src.charCodeAt(i);
  const code = String(Math.abs(hash % 1000)).padStart(3, '0');
  return `Anonymous #A${code}`;
}

function buildDefaultAnonIdentity(uid = '', firstName = '') {
  const source = `${uid || ''}${firstName || ''}`;
  let hash = 0;
  for (let i = 0; i < source.length; i++) hash = ((hash << 5) - hash) + source.charCodeAt(i);
  const idx = Math.abs(hash) % ANON_PERSONA_THEMES.length;
  const base = ANON_PERSONA_THEMES[idx] || 'Anonymous';
  const code = String(Math.abs(hash % 1000)).padStart(3, '0');
  return `${base} #A${code}`;
}

function getUserAnonIdentity(user = {}) {
  const custom = (user?.anonIdentity || '').trim();
  if (custom) return clampText(custom, 32);
  return buildDefaultAnonIdentity(user?.id || user?.uid || state.user?.uid || '', user?.firstName || '');
}

function getAnonymousLabelForPost(post = {}) {
  if (!post?.isAnonymous) return post?.authorName || 'User';
  const raw = (post.authorName || '').trim();
  if (raw && !/^anonymous$/i.test(raw)) return raw;
  return defaultAnonLabel(post.id || post.authorId || 'post');
}

function applySoftKeywordFilterText(raw = '') {
  let value = `${raw || ''}`;
  const flags = [];
  SOFT_FILTER_RULES.forEach(rule => {
    const matcher = new RegExp(rule.regex.source, rule.regex.flags);
    if (matcher.test(value)) {
      flags.push(rule.regex.source);
      value = value.replace(matcher, rule.replacement);
    }
  });
  return {
    text: value.trim(),
    flagged: flags.length > 0,
    flags
  };
}

function isPostShadowHiddenForViewer(post = {}, viewerUid = '') {
  if (!post?.shadowHidden) return false;
  if (!viewerUid) return true;
  if (post.authorId === viewerUid) return false;
  return !_isAdmin;
}

function renderPostContextTags(post = {}) {
  const chips = [];
  const author = _userContextCache[post.authorId] || null;
  const pingExpiry = post.locationPing?.expiresAt?.toDate
    ? post.locationPing.expiresAt.toDate()
    : (post.locationPing?.expiresAt ? new Date(post.locationPing.expiresAt) : null);
  const hasExplicitPing = !!post.locationPing?.enabled && (!pingExpiry || pingExpiry.getTime() > Date.now());
  if (author && hasExplicitPing) {
    chips.push({ text: '📍 Pin', className: 'clickable', onClick: `openPinnedPostOnRadar('${post.id}')` });
    const addr = (author.address || '').toLowerCase();
    if (/\bres\b|residence|hostel|hall/.test(addr)) chips.push({ text: '🏠 Res life' });
  }
  const sharedModule = normalizeModules(post.moduleTags || []).some(tag => normalizeModules(state.profile?.modules || []).includes(tag));
  if (sharedModule) chips.push({ text: '🎓 Same module' });
  const trendScore = getReactionSummary(post.reactions, post.likes || []).total + (post.commentsCount || 0) * 2;
  if (trendScore >= 12) chips.push({ text: '🔥 Trending' });
  if (!chips.length) return '';
  return `<div class="post-context-tags">${chips.slice(0, 4).map(chip => {
    const text = typeof chip === 'string' ? chip : chip.text;
    const cls = typeof chip === 'string' ? '' : (chip.className || '');
    const click = typeof chip === 'string' || !chip.onClick ? '' : ` onclick="${chip.onClick}"`;
    return `<button class="post-context-chip ${cls}" type="button"${click}>${esc(text)}</button>`;
  }).join('')}</div>`;
}

let _pendingRadarPinnedPoint = null;

function openPinnedPostOnRadar(postId) {
  const post = (state.posts || []).find(p => p.id === postId);
  if (!post) return;
  const context = _userContextCache[post.authorId] || {};
  const ping = post.locationPing || {};
  const lat = Number(ping.lat ?? context.geoLat ?? context.lat);
  const lng = Number(ping.lng ?? context.geoLng ?? context.lng);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
    toast('Pinned location not available yet');
    return;
  }
  const label = ping.label || post.authorName || 'Pinned';
  _pendingRadarPinnedPoint = {
    lat,
    lng,
    label
  };
  setPendingMapRoute({ lat, lng }, { label, targetId: postId, source: 'post' });
  exploreView = 'radar';
  navigate('explore');
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

// ─── Feed randomisation & seen-post tracking ─────
let _sessionPostSeeds = {};
function sessionSeed(postId) {
  if (_sessionPostSeeds[postId] == null) _sessionPostSeeds[postId] = Math.random();
  return _sessionPostSeeds[postId];
}
function resetFeedSeeds() { _sessionPostSeeds = {}; }

function getSeenPosts() {
  try { return JSON.parse(localStorage.getItem('unino_seen_posts') || '{}'); }
  catch { return {}; }
}
function markPostSeen(postId) {
  const seen = getSeenPosts();
  const entry = seen[postId] || { v: 0, t: 0 };
  entry.v++;
  entry.t = Date.now();
  seen[postId] = entry;
  // keep only most recent 500
  const keys = Object.keys(seen);
  if (keys.length > 500) {
    keys.sort((a, b) => seen[a].t - seen[b].t);
    keys.slice(0, keys.length - 500).forEach(k => delete seen[k]);
  }
  localStorage.setItem('unino_seen_posts', JSON.stringify(seen));
}
let _feedSeenObserver = null;

function extractHashTags(text = '') {
  const found = new Set();
  const regex = /#([A-Za-z][A-Za-z0-9_]{1,23})\b/g;
  const source = `${text || ''}`;
  let match;
  while ((match = regex.exec(source))) found.add(match[1].toLowerCase());
  return [...found].slice(0, 8);
}

function extractMentionHandles(text = '') {
  const found = new Set();
  const regex = /@([A-Za-z][A-Za-z0-9._-]{1,31})\b/g;
  let match;
  while ((match = regex.exec(text || ''))) {
    found.add((match[1] || '').toLowerCase());
  }
  return [...found].slice(0, 20);
}

function mentionHandleForName(name = '') {
  return String(name || '')
    .toLowerCase()
    .replace(/[^a-z0-9._-]/g, '');
}

async function openMentionProfileByHandle(handle = '') {
  const clean = String(handle || '').trim().toLowerCase();
  if (!clean) return;
  const local = (_usersCache?.data || []).find(u => mentionHandleForName(u.displayName || '') === clean);
  if (local?.id) {
    openProfile(local.id);
    return;
  }
  try {
    const users = await getUsersCache();
    const hit = (users || []).find(u => mentionHandleForName(u.displayName || '') === clean);
    if (hit?.id) {
      openProfile(hit.id);
      return;
    }
  } catch (_) {}
  toast('Profile not found');
}

function extractModuleTags(text = '', manualTags = '') {
  const found = new Set();
  extractHashTags(`${text} ${manualTags}`)
    .map(tag => tag.toUpperCase())
    .filter(tag => /^[A-Z]{3,5}\d{3}$/.test(tag))
    .forEach(tag => found.add(tag));
  normalizeModules((manualTags || '').split(',')).forEach(tag => found.add(tag));
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
  if (!rail || rail.classList.contains('collapsed')) return;
  const amount = rail.clientWidth * 0.82;
  const atEnd = rail.scrollLeft + rail.clientWidth >= rail.scrollWidth - 12;
  if (dir > 0 && atEnd) rail.scrollTo({ left: 0, behavior: 'smooth' });
  else rail.scrollBy({ left: amount * dir, behavior: 'smooth' });
}

function toggleTrendingRail() {
  const rail = document.getElementById('trending-post-scroll');
  const btn = document.querySelector('.trend-toggle-btn');
  if (!rail) return;
  const collapsed = rail.classList.toggle('collapsed');
  if (btn) btn.classList.toggle('collapsed', collapsed);
  if (collapsed && _trendingRailTimer) {
    clearInterval(_trendingRailTimer);
    _trendingRailTimer = null;
  } else if (!collapsed) {
    setupTrendingRail();
  }
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
    .filter(post => (post.content || post.imageURL) && !(post.videoURL || post.mediaType === 'video'))
    .sort((a, b) => ((getReactionSummary(b.reactions, b.likes || []).total + (b.commentsCount || 0) * 2) - (getReactionSummary(a.reactions, a.likes || []).total + (a.commentsCount || 0) * 2)))
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
          <button class="trend-toggle-btn" onclick="toggleTrendingRail()" title="Collapse / Expand">⌃</button>
        </div>
      </div>
      <div class="trending-post-scroll" id="trending-post-scroll">
        ${trending.map(post => `
          <div class="trending-post-card" onclick="viewPost('${post.id}')">
            ${post.videoURL || post.mediaType === 'video' ? `<div class="trending-post-media has-video"><video class="inline-video-preview" src="${post.videoURL || post.imageURL}" muted playsinline preload="metadata"></video><div class="trending-post-video-badge">▶</div></div>` : post.imageURL ? `<div class="trending-post-media"><img src="${post.imageURL}" alt=""></div>` : ''}
            <div class="trending-post-meta-top">
              <span>${post.isAnonymous ? esc(getAnonymousLabelForPost(post)) : esc(post.authorName || 'User')}</span>
              <span>${getReactionSummary(post.reactions, post.likes || []).total} reacts</span>
            </div>
            ${post.content ? `<div class="trending-post-copy">${formatContent((post.content || '').slice(0, 150) + ((post.content || '').length > 150 ? '...' : ''))}</div>` : ''}
          </div>
        `).join('')}
      </div>
    </div>
  `;
  primeInlineVideoPreviews(railHost);
  setupTrendingRail();
}

async function openModuleFeed(tag) {
  const moduleTag = normalizeModules([tag])[0] || '';
  if (!moduleTag) return;
  let posts = (state.posts || []).filter(post => normalizeModules(post.moduleTags || []).includes(moduleTag));
  if (!posts.length) {
    try {
      const snap = await db.collection('posts').orderBy('createdAt', 'desc').limit(50).get();
      posts = snap.docs.map(d => ({ id: d.id, ...d.data() })).filter(post => normalizeModules(post.moduleTags || []).includes(moduleTag));
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
              <div class="module-post-author">${post.isAnonymous ? esc(getAnonymousLabelForPost(post)) : esc(post.authorName || 'User')}</div>
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
              <div class="module-post-author">${post.isAnonymous ? esc(getAnonymousLabelForPost(post)) : esc(post.authorName || 'User')}</div>
              <div class="module-post-time">${timeAgo(post.createdAt)}</div>
            </div>
          </div>
          ${post.content ? `<div class="module-post-text">${renderExpandablePostContent(post.content, `tag-${post.id}`, 140)}</div>` : ''}
          ${renderPostHashTags(getPostHashTags(post).filter(item => item !== hashTag))}
        </div>`).join('') : '<div class="empty-state"><h3>No posts yet</h3><p>Use this tag in a post to start the thread.</p></div>'}
    </div>
  `);
}

async function openAnonPostActions(uid, postId = null) {
  if (!uid || uid === state.user.uid) return toast("That's you!");
  try {
    const userDoc = await db.collection('users').doc(uid).get();
    if (userDoc.exists && !allowAnonymousDMsFor(userDoc.data() || {})) {
      return toast('This user only accepts messages from friends');
    }
  } catch (_) {}
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

async function notifyRelevantModuleUsers(moduleTags = [], text = '', postId, isAnon = false) {
  const uniqueTags = [...new Set(moduleTags)].slice(0, 3);
  if (!uniqueTags.length || !postId) return;
  const notified = new Set();
  const notifTextFor = tag => /notes|summary|slides|past\s*paper|resource/i.test(text)
    ? `shared notes in ${tag}`
    : `posted in ${tag}`;
  try {
    const snap = await db.collection('users').limit(60).get();
    const promises = [];
    for (const doc of snap.docs) {
      if (doc.id === state.user.uid || notified.has(doc.id)) continue;
      const userData = doc.data() || {};
      const matchedTag = uniqueTags.find(tag => normalizeModules(userData.modules || []).includes(tag));
      if (!matchedTag) continue;
      notified.add(doc.id);
      promises.push(addNotification(doc.id, 'module', notifTextFor(matchedTag), { postId, moduleTag: matchedTag }, { anonymous: isAnon }));
    }
    await Promise.all(promises);
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
  // Generate 40 random waveform bar heights for visual fidelity
  const bars = Array.from({ length: 40 }, () => {
    const h = Math.max(4, Math.floor(Math.random() * 24) + 4);
    return `<div class="vn-waveform-bar" style="height:${h}px"></div>`;
  }).join('');
  return `<div class="vn-player" id="${id}" data-src="${audioURL}">
    <button class="vn-play-btn" onclick="toggleVN('${id}')">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
    </button>
    <div class="vn-waveform-wrap" onclick="seekVN(event,'${id}')">
      <div class="vn-waveform">${bars}</div>
      <div class="vn-waveform-played"></div>
      <div class="vn-scrub-dot"></div>
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
      const playedOverlay = el.querySelector('.vn-waveform-played');
      if (playedOverlay) playedOverlay.style.width = pct + '%';
      const scrubDot = el.querySelector('.vn-scrub-dot');
      if (scrubDot) scrubDot.style.left = pct + '%';
      el.querySelector('.vn-time').textContent = fmtDur(audio.duration - audio.currentTime);
    });
    audio.addEventListener('ended', () => {
      el.classList.remove('playing');
      el.querySelector('.vn-play-btn').innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
      const playedOverlay = el.querySelector('.vn-waveform-played');
      if (playedOverlay) playedOverlay.style.width = '0%';
      const scrubDot = el.querySelector('.vn-scrub-dot');
      if (scrubDot) scrubDot.style.left = '0%';
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
          if (uid === state.user.uid) {
            setTimeout(() => refreshBackendDebugStatus(), 0);
          }
}

function seekVN(e, id) {
  const el = document.getElementById(id);
  if (!el || !_vnAudios[id]) return;
  const track = el.querySelector('.vn-waveform-wrap');
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
  let html = esc(text);
  // Markdown-style links: [label](url) → clickable link
  html = html.replace(/\[([^\]]{1,80})\]\((https?:\/\/[^\s)]+)\)/g, (_, label, url) => {
    return `<a href="${url}" target="_blank" rel="noopener" class="post-link">${label}</a>`;
  });
  // Auto-linkify bare URLs (skip already-linked ones)
  html = html.replace(/(^|[^"=])(https?:\/\/[^\s<"]+)/g, (match, pre, url) => {
    const clean = url.replace(/[.,;:!?)]+$/, '');
    const display = clean.length > 40 ? clean.slice(0, 37) + '...' : clean;
    return `${pre}<a href="${clean}" target="_blank" rel="noopener" class="post-link">${display}</a>`;
  });
  // Hashtags
  html = html.replace(/#(\w+)/g, (_, rawTag) => {
    const tag = (rawTag || '').toUpperCase();
    if (/^[A-Z]{3,5}\d{3}$/.test(tag)) {
      return `<span class="hashtag module-hashtag" onclick="openModuleFeed('${tag}')">#${tag}</span>`;
    }
    return `<span class="hashtag" onclick="openTagFeed('${rawTag.toLowerCase()}')">#${rawTag}</span>`;
  });
  // Mentions
  html = html.replace(/(^|\s)@([A-Za-z][A-Za-z0-9._-]{1,31})\b/g, (_, lead, handle) => {
    const cleanHandle = (handle || '').toLowerCase();
    return `${lead}<span class="mention-handle" onclick="openMentionProfileByHandle('${cleanHandle}')">@${handle}</span>`;
  });
  return html;
}

// ─── Custom Video Player Engine ──────────────────
let _playerIdCounter = 0;
let _activePlayerDestroys = [];

function buildPlayerHTML(src, id) {
  return `
  <div class="unino-player show-controls is-loading" id="up-${id}" data-player-id="${id}">
    <video preload="metadata" playsinline webkit-playsinline disablepictureinpicture muted>
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
      <button class="up-btn up-download-btn" data-act="download" aria-label="Download video">
        <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><path d="M12 3v12"/><polyline points="7 11 12 16 17 11"/><rect x="4" y="18" width="16" height="3" rx="1"/></svg>
      </button>
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

  const markReady = () => {
    root.classList.add('ready');
    root.classList.remove('is-loading');
    if (vid) vid.style.visibility = 'visible';
    loader.classList.remove('active');
  };

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
      if (vid.muted && root.dataset.userMuted !== '1') {
        vid.muted = false;
        if (volSlider && (!Number.isFinite(parseFloat(volSlider.value)) || parseFloat(volSlider.value) <= 0)) volSlider.value = '1';
      }
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
  vid.addEventListener('loadeddata', markReady, { once: true });
  vid.addEventListener('ended', () => { root.classList.remove('playing'); root.classList.add('show-controls'); clearTimeout(controlsTimer); });
  vid.addEventListener('waiting', () => loader.classList.add('active'));
  vid.addEventListener('canplay', markReady);

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
  const _ac = new AbortController();
  const _acSig = { signal: _ac.signal };
  progressWrap.addEventListener('mousedown', (e) => {
    e.stopPropagation();
    scrubbing = true;
    isSeeking = true;
    seekFromEvent(e);
  });
  document.addEventListener('mousemove', (e) => {
    if (!scrubbing) return;
    seekFromEvent(e);
  }, _acSig);
  document.addEventListener('mouseup', () => {
    if (scrubbing) { scrubbing = false; isSeeking = false; }
  }, _acSig);
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
      root.dataset.userMuted = vid.muted ? '1' : '0';
    });
    const volBtn = root.querySelector('.vol-btn');
    if (volBtn) {
      volBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        vid.muted = !vid.muted;
        root.dataset.userMuted = vid.muted ? '1' : '0';
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

  // Download current media without leaving the app context.
  root.querySelector('[data-act="download"]')?.addEventListener('click', (e) => {
    e.stopPropagation();
    const activeSrc = vid.currentSrc || vid.getAttribute('src') || vid.querySelector('source')?.src || '';
    downloadUrlInApp(activeSrc);
  });

  // Fullscreen
  root.querySelector('[data-act="fullscreen"]')?.addEventListener('click', (e) => {
    e.stopPropagation();
    if (document.fullscreenElement === root) {
      document.exitFullscreen();
    