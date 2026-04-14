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
let _nativePendingFcmToken = '';
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
let _campusPlacePointsCache = [];
const _destinationSearchCache = new Map();
const _destinationSearchInflight = new Map();
const _reverseGeocodeCache = new Map();

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

async function sendAppwritePing() {
  refreshBackendDebugStatus('Sending Appwrite ping...');
  const urls = [...new Set([...APPWRITE_PUSH_SYNC_URLS, ...APPWRITE_EVENT_SYNC_URLS])];
  const probeUrl = urls[0] || '';
  if (!probeUrl) {
    refreshBackendDebugStatus('No Appwrite bridge URLs configured.');
    toast('No Appwrite bridge URL set');
    return;
  }

  const rootUrl = appwriteRootUrl(probeUrl);
  try {
    const resp = await fetch(rootUrl, { method: 'GET' });
    const body = await resp.clone().json().catch(() => null);
    const service = body?.service || 'bridge';
    const version = body?.version ? ` v${body.version}` : '';
    const detail = `${rootUrl} -> GET / ${resp.status} (${service}${version})`;
    refreshBackendDebugStatus(detail);
    toast(resp.ok ? 'Appwrite ping successful' : `Appwrite ping failed (${resp.status})`);
  } catch (e) {
    refreshBackendDebugStatus(`${rootUrl} -> ping failed (${e?.message || 'network/cors'})`);
    toast('Appwrite ping failed');
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

  const pushUrl = APPWRITE_PUSH_SYNC_URLS[0] || '';
  if (pushUrl && auth.currentUser && state.user?.uid && _nativePushToken) {
    try {
      const pushResp = await postToAppwriteBridge(pushUrl, {
        action: 'upsert',
        userId: state.user.uid,
        token: _nativePushToken,
        platform: window.Capacitor?.getPlatform?.() || 'android'
      });
      const body = await pushResp.clone().json().catch(() => null);
      const targetStatus = body?.result?.messagingTarget?.mode
        || body?.result?.messagingTarget?.reason
        || body?.error
        || '';
      lines.push(`${pushUrl} -> POST /push-sync ${pushResp.status}${targetStatus ? ` (${String(targetStatus).slice(0, 120)})` : ''}`);
    } catch (e) {
      lines.push(`${pushUrl} -> POST /push-sync failed (${e?.message || 'network/cors'})`);
    }
  } else if (pushUrl && state.user?.uid) {
    lines.push(`${pushUrl} -> POST /push-sync skipped (missing native token)`);
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

function queueNativeFcmTokenHint(token = '') {
  const hinted = String(token || '').trim();
  if (!hinted || hinted.length < 20) return;
  _nativePendingFcmToken = hinted;
  if (state.user?.uid) savePushTokenForCurrentUser(hinted).catch(() => {});
}

function consumeNativeFcmTokenHint() {
  const hinted = String(_nativePendingFcmToken || window.__UNINO_NATIVE_FCM_TOKEN || '').trim();
  if (!hinted || hinted.length < 20) return;
  queueNativeFcmTokenHint(hinted);
  window.__UNINO_NATIVE_FCM_TOKEN = '';
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
    window.addEventListener('unino:native-fcm-token', event => {
      queueNativeFcmTokenHint(event?.detail?.token || '');
    });
    const pending = window.__UNINO_PENDING_NOTIFICATION || null;
    if (pending) {
      window.__UNINO_PENDING_NOTIFICATION = null;
      setTimeout(() => handleNativeNotificationOpen(pending.extra || pending, pending.actionId || 'tap'), 120);
    }
    const pendingNativeToken = String(window.__UNINO_NATIVE_FCM_TOKEN || '').trim();
    if (pendingNativeToken) queueNativeFcmTokenHint(pendingNativeToken);
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

function getContactNickname(userId = '') {
  if (!userId || userId === state.user?.uid) return '';
  const value = (state.profile?.contactNicknames || {})[userId];
  return typeof value === 'string' ? value.trim() : '';
}

function formatConversationLastMessage(raw = '', convo = {}, viewerUid = '', otherName = 'User') {
  const text = String(raw || '').trim();
  if (!text) return '';

  if (text.startsWith('story_like::')) {
    const [, senderId = ''] = text.split('::');
    return senderId === viewerUid ? 'You liked their story' : `${otherName || 'They'} liked your story`;
  }

  if (text.startsWith('message_reaction::')) {
    const parts = text.split('::');
    const senderId = parts[1] || '';
    const targetSenderId = parts[2] || '';
    let emoji = parts[3] || '❤️';
    try { emoji = decodeURIComponent(emoji); } catch (_) {}
    if (senderId === viewerUid) return `You reacted ${emoji}`;
    if (targetSenderId === viewerUid) return `${otherName || 'They'} reacted ${emoji} to your message`;
    return `${otherName || 'They'} reacted ${emoji}`;
  }

  return text;
}

function getResolvedUserIdentity(userId = '', fallbackName = 'User', fallbackPhoto = null) {
  const cached = _userContextCache[userId] || {};
  const contactAlias = getContactNickname(userId);
  return {
    name: contactAlias || cached.displayName || cached.firstName || fallbackName || 'User',
    photo: cached.photoURL || fallbackPhoto || null,
    data: cached,
    alias: contactAlias
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
      if (result.nextReaction
        && localMessage.senderId
        && localMessage.senderId !== state.user?.uid
        && !['message_reaction', 'story_like'].includes(localMessage.type || '')) {
        emitMessageReactionNotice(scope, primaryId, localMessage, result.nextReaction, collection).catch(err => {
          console.warn('reaction notice failed', err);
        });
      }
    }
  } catch (e) {
    syncLocalMessageReactionState(scope, messageId, previousState.reactions, previousState.likes);
    refreshMessageReactionUI(scope, primaryId, messageId, collection);
    console.error(e);
    toast('Could not react right now');
  }
}

async function emitMessageReactionNotice(scope, primaryId, targetMessage = {}, emoji = '❤️', collection = '') {
  if (!primaryId || !emoji || !state.user?.uid) return;
  const actorUid = state.user.uid;
  const targetSenderId = targetMessage.senderId || '';
  if (!targetSenderId || targetSenderId === actorUid) return;

  const safeEmoji = `${emoji || '❤️'}`.trim() || '❤️';
  const payload = {
    targetMessageId: targetMessage.id || '',
    targetSenderId,
    emoji: safeEmoji
  };

  if (scope === 'group') {
    const groupRef = db.collection(collection || 'groups').doc(primaryId);
    await groupRef.collection('messages').add({
      text: `${state.profile?.displayName || 'Someone'} reacted ${safeEmoji}`,
      senderId: actorUid,
      senderName: state.profile?.displayName || 'Someone',
      senderPhoto: state.profile?.photoURL || null,
      senderAnon: false,
      type: 'message_reaction',
      payload,
      createdAt: FieldVal.serverTimestamp(),
      status: 'sent'
    });
    await syncThreadLastMessage('group', primaryId, collection || 'groups');
    return;
  }

  const convoRef = db.collection('conversations').doc(primaryId);
  await convoRef.collection('messages').add({
    text: '',
    senderId: actorUid,
    senderName: state.profile?.displayName || 'Someone',
    senderPhoto: state.profile?.photoURL || null,
    senderAnon: false,
    type: 'message_reaction',
    payload,
    createdAt: FieldVal.serverTimestamp(),
    status: 'sent'
  });

  let otherUid = '';
  try {
    const convoDoc = await convoRef.get();
    const participants = convoDoc.exists ? (convoDoc.data()?.participants || []) : [];
    otherUid = participants.find(uid => uid && uid !== actorUid) || '';
  } catch (_) {}

  const mergeData = {
    lastMessage: `message_reaction::${actorUid}::${targetSenderId}::${encodeURIComponent(safeEmoji)}`,
    updatedAt: FieldVal.serverTimestamp()
  };
  if (otherUid) mergeData.unread = { [otherUid]: FieldVal.increment(1), [actorUid]: 0 };
  await convoRef.set(mergeData, { merge: true });
  await addNotification(targetSenderId, 'message', `reacted ${safeEmoji} to your message`, { convoId: primaryId });
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
      else if (latest.type === 'story_like') {
        if (scope === 'group') {
          const who = latest.senderName || latest.senderId || 'Someone';
          lastMessage = `${who} liked a story`;
        } else {
          lastMessage = `story_like::${latest.senderId || ''}`;
        }
      }
      else if (latest.type === 'message_reaction') {
        const reactionEmoji = latest.payload?.emoji || '❤️';
        if (scope === 'group') {
          const who = latest.senderName || latest.senderId || 'Someone';
          lastMessage = `${who} reacted ${reactionEmoji}`;
        } else {
          lastMessage = `message_reaction::${latest.senderId || ''}::${latest.payload?.targetSenderId || ''}::${encodeURIComponent(reactionEmoji)}`;
        }
      }
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
    const commentsRef = db.collection('posts').doc(postId).collection('comments');
    const uid = state.user?.uid || '';
    if (!uid) return;
    const postSnap = await db.collection('posts').doc(postId).get();
    const postAuthorId = postSnap.exists ? (postSnap.data()?.authorId || '') : '';
    const snap = await commentsRef.limit(200).get();
    const comments = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    const targetComment = comments.find(comment => comment.id === commentId) || null;
    const canDelete = !!targetComment && (_isAdmin || targetComment.authorId === uid || postAuthorId === uid);
    if (!canDelete) {
      toast('Only the post owner or comment owner can delete this');
      return;
    }
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

function escJsString(s = '') {
  return String(s || '')
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/\u2028/g, ' ')
    .replace(/\u2029/g, ' ')
    .replace(/\r?\n/g, ' ');
}

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
              <span>${post.isAnonymous ? esc(getAnonymousLabelForPost(post)) : esc(getResolvedUserIdentity(post.authorId, post.authorName || 'User', post.authorPhoto || null).name)}</span>
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
            ${post.isAnonymous ? `<div class="avatar-sm anon-avatar">👻</div>` : avatar(getResolvedUserIdentity(post.authorId, post.authorName || 'User', post.authorPhoto || null).name, post.authorPhoto, 'avatar-sm')}
            <div>
              <div class="module-post-author">${post.isAnonymous ? esc(getAnonymousLabelForPost(post)) : esc(getResolvedUserIdentity(post.authorId, post.authorName || 'User', post.authorPhoto || null).name)}</div>
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
            ${post.isAnonymous ? `<div class="avatar-sm anon-avatar">👻</div>` : avatar(getResolvedUserIdentity(post.authorId, post.authorName || 'User', post.authorPhoto || null).name, post.authorPhoto, 'avatar-sm')}
            <div>
              <div class="module-post-author">${post.isAnonymous ? esc(getAnonymousLabelForPost(post)) : esc(getResolvedUserIdentity(post.authorId, post.authorName || 'User', post.authorPhoto || null).name)}</div>
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
    const currentSrc = vid.currentSrc
      || vid.getAttribute('src')
      || vid.querySelector('source')?.getAttribute('src')
      || '';
    downloadUrlInApp(currentSrc);
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
  document.addEventListener('fullscreenchange', onFsChange, _acSig);
  document.addEventListener('webkitfullscreenchange', onFsChange, _acSig);

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

  if (vid.readyState >= 2) markReady();
  else loader.classList.add('active');

  // Initial controls auto-hide
  controlsTimer = setTimeout(() => root.classList.remove('show-controls'), 4000);

  const destroy = () => { _ac.abort(); clearTimeout(controlsTimer); vid.pause(); vid.removeAttribute('src'); vid.load(); };
  return { root, vid, togglePlay, destroy };
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
  closeNotifDropdown();
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
const THEME_ASSETS = {
  light: { icon: `unino-icon-light.png?v=${APP_VERSION}`, themeColor: '#FFFFFF' },
  dark: { icon: `unino-icon-dark.png?v=${APP_VERSION}`, themeColor: '#12121F' }
};

function applyThemeAssets(theme = document.documentElement.getAttribute('data-theme') || 'light') {
  const mode = theme === 'dark' ? 'dark' : 'light';
  const asset = THEME_ASSETS[mode];
  document.querySelectorAll('[data-theme-logo]').forEach(img => {
    if (img.getAttribute('src') !== asset.icon) img.setAttribute('src', asset.icon);
  });
  const favicon = document.getElementById('theme-favicon');
  if (favicon) favicon.setAttribute('href', asset.icon);
  const themeMeta = document.getElementById('theme-color-meta');
  if (themeMeta) themeMeta.setAttribute('content', asset.themeColor);
}


function showBootStatus(message = 'Loading…') {
  const splash = $('#splash');
  if (splash) splash.classList.add('active');
  const title = document.querySelector('#splash .splash-status');
  if (title) title.textContent = message;
}

function markBootReady(message = '') {
  _bootReady = true;
  const splash = $('#splash');
  if (!splash) return;
  const status = document.querySelector('#splash .splash-status');
  if (status && message) status.textContent = message;
  requestAnimationFrame(() => {
    setTimeout(() => splash.classList.remove('active'), 180);
  });
}

function ensureBootFallback() {
  if (_bootFallbackTimer) clearTimeout(_bootFallbackTimer);
  _bootFallbackTimer = setTimeout(() => {
    if (_bootReady) return;
    showBootStatus('Still starting… reconnecting to your account.');
  }, 4500);
}

function initTheme() {
  const saved = localStorage.getItem('unino-theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
  applyThemeAssets(saved);
  syncNativeStatusBar().catch(() => {});
  $('#theme-btn')?.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('unino-theme', next);
    applyThemeAssets(next);
    syncNativeStatusBar().catch(() => {});
  });
}

// ══════════════════════════════════════════════════
//  AUTH
// ══════════════════════════════════════════════════
function initAuth() {
  const isAdminLoginEmail = email => (email || '').trim().toLowerCase() === ADMIN_EMAIL.toLowerCase();
  const loginEmailValue = () => ($('#l-email')?.value || '').trim();

  $$('.password-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = document.getElementById(btn.dataset.target || '');
      if (!target) return;
      const nextType = target.type === 'password' ? 'text' : 'password';
      target.type = nextType;
      btn.textContent = nextType === 'password' ? 'Show' : 'Hide';
    });
  });

  $('#forgot-pass-btn')?.addEventListener('click', async () => {
    const email = loginEmailValue();
    if (!email) return toast('Enter your email first');
    if (!isStudentEmail(email) && !isAdminLoginEmail(email)) return toast('Use your @mynwu.ac.za student email');
    const btn = $('#forgot-pass-btn');
    btn.disabled = true;
    try {
      await auth.sendPasswordResetEmail(email);
      toast('If that account exists, a recovery email has been sent. Check your inbox.');
    } catch (err) {
      if (err?.code === 'auth/user-not-found' || err?.code === 'auth/invalid-email') toast('If that account exists, a recovery email has been sent. Check your inbox.');
      else toast(friendlyErr(err.code));
    } finally {
      btn.disabled = false;
    }
  });

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
    try {
      await auth.signInWithEmailAndPassword(email, pass);
    }
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
    const modules = normalizeModules(modulesRaw.split(','));
    const gender = ($('#s-gender')?.value || '').trim();
    const orientation = ($('#s-orientation')?.value || '').trim();
    const address = $('#s-address')?.value.trim() || '';
    if (!fname || !lname || !email || !pass || !uni || !major || !address || !gender) return toast('All fields required');
    if (pass.length < 6) return toast('Password must be 6+ characters');
    if (!isStudentEmail(email)) return toast('Only @mynwu.ac.za emails can sign up');
    btn.disabled = true; btn.innerHTML = '<span class="inline-spinner"></span>';
    try {
      const cred = await auth.createUserWithEmailAndPassword(email, pass);
      const uid = cred.user.uid;
      const displayName = `${fname} ${lname}`;
      const anonIdentity = buildDefaultAnonIdentity(uid, fname);
      await db.collection('users').doc(uid).set({
        displayName, firstName: fname, lastName: lname,
        email, university: uni, major, year, modules, address, gender,
        orientation,
        bio: `${major} student at ${uni}`,
        photoURL: '', status: 'online', allowAutoFill: true,
        allowAnonymousMessages: true,
        anonIdentity,
        appwritePrimary: true,
        appwriteMigrationSource: 'cutover-new-users',
        appwriteJoinedAt: FieldVal.serverTimestamp(),
        isVerified: false,
        manualVerified: false,
        joinedAt: FieldVal.serverTimestamp(), friends: []
      });
      shadowSyncUserProfile(uid, { displayName, email, photoURL: '', major, university: uni });
      await cred.user.updateProfile({ displayName });
      toast('Account created.');
      db.collection('stats').doc('global').set({ totalUsers: FieldVal.increment(1) }, { merge: true }).catch(() => {});
      btn.disabled = false; btn.textContent = 'Create Account';
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
      _sessionRecoveryInFlight = false;
      try {
        await user.reload();
        await user.getIdToken(true);
      } catch (err) {
        if (await recoverInvalidSession(err, 'Auth refresh failed')) return;
      }
      const isAdminUser = isAdminLoginEmail(user.email || '');
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
      } catch (err) {
        if (await recoverInvalidSession(err, 'Profile load failed')) return;
        if (isPermissionDeniedError(err)) {
          console.error('Profile load denied:', err);
          toast('Could not access your profile. Log in again.');
          await auth.signOut().catch(() => {});
          return;
        }
        state.profile = { id: user.uid, displayName: user.displayName, email: user.email, status: 'online', modules: [] };
      }
      state.manualStatus = state.profile.manualStatus || state.profile.status || 'online';
      state.status = state.profile.status || state.manualStatus;
      // Admin detection
      _isAdmin = isAdminUser;
      if (_isAdmin || state.profile.manualVerified) VERIFIED_UIDS.add(user.uid);
      enterApp();
    } else {
      if (_nativePushToken && state.user?.uid) removePushTokenForUser(state.user.uid, _nativePushToken).catch(() => {});
      if (verifiedUsersUnsub) { verifiedUsersUnsub(); verifiedUsersUnsub = null; }
      if (_groupAlertUnsub) { _groupAlertUnsub(); _groupAlertUnsub = null; }
      if (notifUnsub) { notifUnsub(); notifUnsub = null; }
      if (generalNotifUnsub) { generalNotifUnsub(); generalNotifUnsub = null; }
      if (_unreadDMSub) { _unreadDMSub(); _unreadDMSub = null; }
      if (_onlineCountSub) { _onlineCountSub(); _onlineCountSub = null; }
      if (_onlineCountTimer) { clearInterval(_onlineCountTimer); _onlineCountTimer = null; }
      if (_presenceTimer) { clearInterval(_presenceTimer); _presenceTimer = null; }
      VERIFIED_UIDS.clear();
      _asgPendingAlerts = [];
      _dmUnreadCount = 0;
      _activeChatConvoId = '';
      _activeGroupChat = { id: '', collection: '' };
      _nativeDmUnreadMap = {};
      _nativePushReady = false;
      _nativePushToken = '';
      _nativeDmNotificationPrimed = false;
      _nativeGeneralNotificationPrimed = false;
      _nativeGeneralNotifIds.clear();
      state.user = null; state.profile = null; unsub(); showScreen('auth-screen'); markBootReady('Welcome back');
    }
  });

  // One-time fetch instead of permanent listener for auth screen count
  db.collection('stats').doc('global').get().then(doc => {
    const el = $('#auth-count'); if (el && !state.user) el.textContent = '0';
  }).catch(() => {});
}

function friendlyErr(code) {
  return { 'auth/user-not-found':'Account not found','auth/wrong-password':'Incorrect password',
    'auth/email-already-in-use':'Email already registered','auth/weak-password':'Password too weak',
    'auth/invalid-email':'Invalid email','auth/invalid-login-credentials':'Incorrect email or password',
    'auth/too-many-requests':'Too many attempts. Wait a bit and try again.',
    'auth/network-request-failed':'Network issue' }[code] || 'Something went wrong';
}

// ══════════════════════════════════════════════════
//  APP UPDATE CHECK
// ══════════════════════════════════════════════════
function checkForAppUpdate() {
  db.collection('appConfig').doc('version').get().then(snap => {
    if (!snap.exists) return;
    const data = snap.data();
    const latest = data.version || 0;
    if (latest <= APP_VERSION) return;
    const link = data.downloadUrl || '';
    const label = data.downloadLabel || 'Download Update';
    const msg = data.message || 'A new version of Unibo is available.';
    // Show banner
    let banner = document.getElementById('update-banner');
    if (banner) banner.remove();
    banner = document.createElement('div');
    banner.id = 'update-banner';
    banner.innerHTML = `<span>${msg}</span>` +
      (link ? `<a href="${encodeURI(link)}" target="_blank" rel="noopener">${label}</a>` : '') +
      `<button onclick="this.parentElement.remove()">&times;</button>`;
    document.body.appendChild(banner);
  }).catch(() => {});
}

// ══════════════════════════════════════════════════
//  ENTER APP
// ══════════════════════════════════════════════════
function enterApp() {
  showScreen('app');
  const content = $('#content');
  if (content) content.innerHTML = `<div class="app-loading-shell"><div class="app-loading-card"><span class="inline-spinner"></span><h3>Loading your campus feed…</h3><p>If your network is slow, the app will stay visible instead of going blank.</p></div></div>`;
  setupHeader(); setupNav(); setupStatusPill(); 
  initNativeShell().catch(() => {});
  // Request notification permissions eagerly then init push
  requestLocalNotificationPermission().then(() => {
    initNativePushNotifications();
    refreshPushRegistration(true);
  }).catch(() => {
    initNativePushNotifications();
    refreshPushRegistration(true);
  });
  consumeNativeFcmTokenHint();
  if (!_inAppBackInit) {
    // Fence: replace current history with app state, then push initial state
    history.replaceState({ app: true, screen: 'app', page: 'feed' }, '');
    _inAppBackInit = true;
  }
  navigate('feed');
  setTimeout(() => markBootReady('Loaded'), 220);
  listenForVerifiedUsers();
  listenForNotifications();
  listenForUnreadDMs();
  listenForGroupAlerts();
  setupPresenceTracking();
  listenForOnlineCount();
  cleanupExpiredStories();
  maybePromptForGpsLocation();
  checkForAppUpdate();
}

let _onlineCountSub = null;
let _onlineCountTimer = null;
function listenForOnlineCount() {
  if (_onlineCountSub) _onlineCountSub();
  if (_onlineCountTimer) clearInterval(_onlineCountTimer);
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
  // Only watch verified users instead of whole users collection.
  verifiedUsersUnsub = db.collection('users').where('manualVerified', '==', true).limit(120).onSnapshot(snap => {
    VERIFIED_UIDS.clear();
    snap.docs.forEach(d => VERIFIED_UIDS.add(d.id));
    if (_isAdmin && state.user?.uid) VERIFIED_UIDS.add(state.user.uid);
    // Refresh visible areas so badges appear as soon as verified list updates.
    if (state.page === 'feed' && Array.isArray(state.posts) && state.posts.length) {
      renderFeedResults(state.posts);
    }
    if (document.getElementById('profile-view')?.classList.contains('active')) {
      const pid = document.getElementById('prof-back')?.dataset?.uid;
      if (pid) openProfile(pid);
    }
  }, () => {
    if (_isAdmin && state.user?.uid) VERIFIED_UIDS.add(state.user.uid);
    if (state.page === 'feed' && Array.isArray(state.posts) && state.posts.length) {
      renderFeedResults(state.posts);
    }
  });
}

let _usersCache = { data: [], expiresAt: 0 };
let _usersCachePromise = null;

async function getUsersCache(options = {}) {
  const force = !!options.force;
  const now = Date.now();
  if (!force && now < _usersCache.expiresAt && _usersCache.data.length) return _usersCache.data;
  if (_usersCachePromise) return _usersCachePromise;
  _usersCachePromise = db.collection('users').get()
    .then(snap => {
      const users = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      _usersCache = { data: users, expiresAt: Date.now() + 3 * 60 * 1000 };
      return users;
    })
    .catch(() => _usersCache.data || [])
    .finally(() => { _usersCachePromise = null; });
  return _usersCachePromise;
}

function listenForUnreadDMs() {
  if (_unreadDMSub) _unreadDMSub();
  _unreadDMSub = db.collection('conversations')
    .where('participants', 'array-contains', state.user.uid)
    .onSnapshot(snap => {
      const uid = state.user.uid;
      const conversations = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      let total = 0;
      conversations.forEach(conversation => { total += (conversation.unread || {})[uid] || 0; });
      _dmUnreadCount = total;
      maybeNotifyForUnreadDMs(conversations);
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
      if (p === state.page) {
        if (p === 'feed') navigate('feed', { refresh: true, restoreFeed: false });
        if (p === 'chat') {
          const inGroupView = !!document.getElementById('group-chat-view')?.classList.contains('active');
          const inDmView = !!document.getElementById('chat-view')?.classList.contains('active');
          if (inGroupView || inDmView || state.lastMsgTab !== 'dm') {
            state.lastMsgTab = 'dm';
            showScreen('app');
            navigate('chat', { refresh: true, pushHistory: false });
          }
        }
        return;
      }
      if (p === 'chat') state.lastMsgTab = 'dm';
      navigate(p, { restoreFeed: p === 'feed' });
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
  } catch (e) {
    if (await recoverInvalidSession(e, 'Conversation presence sync failed')) return;
    console.error(e);
  }
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
  } catch (e) {
    if (await recoverInvalidSession(e, 'Presence update failed')) return;
    console.error(e);
  }
}

function markActivity() {
  _lastActivityAt = Date.now();
  if (state.manualStatus === 'online' && state.status !== 'online') refreshPresence().catch(() => {});
}

function setupPresenceTracking() {
  if (!_presenceListenersAdded) {
    _presenceListenersAdded = true;
    ['mousemove','keydown','click','touchstart','scroll'].forEach(evt => {
      document.addEventListener(evt, markActivity, { passive: true });
    });
    document.addEventListener('visibilitychange', () => {
      _nativeAppIsActive = !document.hidden;
      if (document.hidden) {
        stopAllVideos();
        if (state.manualStatus === 'online') {
          _lastActivityAt = 0;
          refreshPresence(true).catch(() => {});
        }
      } else {
        markActivity();
        refreshPresence(true).catch(() => {});
      }
    });
  }
  // Keep phone back navigation inside the app instead of leaving the site.
  if (!_inAppBackListenerBound) {
    window.addEventListener('popstate', () => handleAppBackAction({ fromPopstate: true }));
    _inAppBackListenerBound = true;
  }
  clearInterval(_presenceTimer);
  _presenceTimer = setInterval(() => refreshPresence().catch(() => {}), 30000);
  refreshPresence(true).catch(() => {});
}

// ─── Navigation ──────────────────────────────────
function navigate(page, options = {}) {
  const { pushHistory = true, refresh = true, restoreFeed = false } = options;
  const previousPage = state.page;
  if (previousPage === page && !refresh) return;
  closeNotifDropdown();
  if (previousPage === 'feed') {
    const contentEl = document.getElementById('content');
    if (contentEl) _feedScrollTop = contentEl.scrollTop || 0;
  }

  state.page = page; unsub();
  stopAllVideos();

  // Destroy leaked video players and observers
  _activePlayerDestroys.forEach(fn => fn());
  _activePlayerDestroys = [];
  if (_feedAutoplayObserver) { _feedAutoplayObserver.disconnect(); _feedAutoplayObserver = null; }

  // Clean up chat listeners if navigating away from chat view
  if (chatUnsub) { chatUnsub(); chatUnsub = null; }
  if (_anonUnsub) { _anonUnsub(); _anonUnsub = null; }
  if (_chatViewportCleanup) { _chatViewportCleanup(); _chatViewportCleanup = null; }
  _activeChatConvoId = '';

  // Clean up voice recording
  cancelVoiceRecord('nav');

  // Clean up audio cache
  Object.values(_vnAudios).forEach(a => { try { a.pause(); a.src = ''; } catch (_) {} });
  Object.keys(_vnAudios).forEach(k => delete _vnAudios[k]);

  // Prune caches to prevent unbounded growth
  const _pruneObj = (obj, max) => { const keys = Object.keys(obj); if (keys.length > max) keys.slice(0, keys.length - max).forEach(k => delete obj[k]); };
  _pruneObj(_authorPhotoCache, 200);
  _pruneObj(_userContextCache, 200);
  _pruneObj(_expandedPostKeys, 100);
  _pruneObj(_postTextStore, 100);
  _pruneObj(_postTextLimit, 100);

  // Keep map references in sync; stale handles can outlive screen transitions.
  destroyRadarMapInstance();
  if (page === 'feed') {
    _pendingFeedScrollRestore = restoreFeed ? _feedScrollTop : 0;
    _feedRestorePendingPaint = !!restoreFeed && _feedScrollTop > 0;
  }
  $$('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.p === page));
  switch (page) {
    case 'feed': renderFeed(); break;
    case 'explore': renderExplore(); break;
    case 'hustle': renderHustle(); break;
    case 'chat': renderMessages(); break;
  }
  // Push state for navigation
  if (_inAppBackInit && state.user && pushHistory) {
    history.pushState({ app: true, screen: 'app', page, msgTab: state.lastMsgTab }, '');
  }
}

// ─── Feed Search: People Suggestions ─────────────
function formatUserMetaLine(user = {}, maxLen = 56) {
  const raw = [user.major, user.university, user.address]
    .map(part => String(part || '').trim())
    .filter(Boolean)
    .join(' · ');
  return clampText(raw || 'Student', maxLen);
}

function renderFeedPeopleSuggestions(query) {
  let box = document.getElementById('feed-people-suggestions');
  if (!box) {
    const searchBar = document.querySelector('.feed-search-bar');
    if (!searchBar) return;
    box = document.createElement('div');
    box.id = 'feed-people-suggestions';
    box.className = 'feed-people-suggestions';
    searchBar.style.position = 'relative';
    searchBar.appendChild(box);
  }
  const q = String(query || '').trim();
  if (!q || q.startsWith('#')) { box.style.display = 'none'; return; }
  getUsersCache().then(users => {
    const hits = users
      .filter(u => u.id !== state.user?.uid)
      .map(u => ({ ...u, _searchScore: userSearchScore(u, q) }))
      .filter(u => u._searchScore > 0)
      .sort((a, b) => b._searchScore - a._searchScore || (a.displayName || '').localeCompare(b.displayName || ''))
      .slice(0, 5);
    if (!hits.length) { box.style.display = 'none'; return; }
    box.innerHTML = hits.map(u => {
      const resolved = getResolvedUserIdentity(u.id, u.displayName || 'User', u.photoURL || null);
      const displayName = resolved.name || u.displayName || 'User';
      const fullMeta = formatUserMetaLine(u, 120);
      return `
      <button type="button" class="feed-people-item" onmousedown="event.preventDefault();openProfile('${u.id}')">
        ${avatar(displayName, u.photoURL, 'avatar-sm')}
        <div class="feed-people-info">
          <span class="feed-people-name">${esc(displayName)}</span>
          <span class="feed-people-meta" title="${esc(fullMeta)}">${esc(formatUserMetaLine(u, 50))}</span>
        </div>
      </button>
    `;
    }).join('');
    box.style.display = 'block';
  }).catch(() => {});
}

function filterFeedPosts(posts = [], query = _feedSearchQuery) {
  const raw = (query || '').trim().toLowerCase();
  if (!raw) return posts;
  const needle = raw.replace(/^#/, '');
  const isTagSearch = raw.startsWith('#');
  return posts.filter(post => {
    const tags = getPostHashTags(post).map(tag => tag.toLowerCase());
    const modules = normalizeModules(post.moduleTags || []).map(tag => tag.toLowerCase());
    const contactAlias = getContactNickname(post.authorId || '');
    const haystack = [
      post.content || '',
      post.authorName || '',
      contactAlias,
      post.authorUni || '',
      ...(post.moduleTags || []),
      ...getPostHashTags(post)
    ].join(' ').toLowerCase();
    if (isTagSearch) return tags.some(tag => tag.includes(needle)) || modules.some(tag => tag.includes(needle));
    return haystack.includes(needle) || tags.some(tag => tag.includes(needle)) || modules.some(tag => tag.includes(needle));
  });
}

function clearFeedSearch() {
  _feedSearchQuery = '';
  const input = $('#feed-search-input');
  if (input) input.value = '';
  renderFeedResults(state.posts || []);
}

function renderFeedResults(posts = []) {
  const filtered = filterFeedPosts(posts, _feedSearchQuery);
  const meta = $('#feed-search-meta');
  if (meta) {
    if (_feedSearchQuery) {
      meta.style.display = 'flex';
      meta.innerHTML = `<span>${filtered.length} result${filtered.length === 1 ? '' : 's'} for <strong>${esc(_feedSearchQuery)}</strong></span><button class="feed-search-clear" onclick="clearFeedSearch()">Clear</button>`;
    } else {
      meta.style.display = 'none';
      meta.innerHTML = '';
    }
  }

  if (_feedSearchQuery) {
    const trends = $('#module-trends');
    const rail = $('#feed-trending-posts');
    if (trends) trends.innerHTML = '';
    if (rail) rail.innerHTML = '';
  } else {
    const trends = $('#module-trends');
    if (trends) trends.innerHTML = '';
    renderTrendingPostsRail(posts);
  }

  renderPosts(filtered);
}

// ══════════════════════════════════════════════════
//  FEED — Clean with unified Discover tabs
// ══════════════════════════════════════════════════
function renderFeed() {
  resetFeedSeeds(); // new random order every time feed is opened / refreshed
  const c = $('#content'), p = state.profile;
  const shouldDelayDiscover = !!_feedRestorePendingPaint;
  let discoverLoaded = !shouldDelayDiscover;
  const ensureDiscoverLoaded = () => {
    if (discoverLoaded) return;
    discoverLoaded = true;
    loadDiscoverPeople();
    loadFeedLiveSpotlight();
  };
  _feedInlineSuggestionSlotsUsed = 0;
  _feedInlineSuggestedUserIds = new Set();
  c.style.opacity = '';
  c.innerHTML = `
    <div class="feed-page">
      <div class="feed-toolbar">
        <div class="feed-live-chip"><span class="dot green"></span> <span id="feed-online">0</span> online</div>
        <div class="search-bar feed-search-bar">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          <input type="text" id="feed-search-input" placeholder="Search posts or #hashtags" value="${esc(_feedSearchQuery)}">
        </div>
      </div>
      <div id="feed-search-meta" class="feed-search-meta" style="display:none"></div>

      <div class="discover-section ${shouldDelayDiscover ? 'discover-section-pending' : ''}">
        <div class="discover-tabs">
          <button class="discover-tab active" data-dt="people">👥 People</button>
          <button class="discover-tab" data-dt="events">📅 Events</button>
        </div>
        <div class="discover-content" id="discover-content">
          ${shouldDelayDiscover
            ? '<div class="discover-empty"><span>⏳</span><p>Loading your feed…</p></div>'
            : '<div style="padding:20px;text-align:center"><span class="inline-spinner"></span></div>'}
        </div>
      </div>

      <div id="feed-live-spotlight" class="feed-live-spotlight" style="display:none"></div>

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
      <button class="reels-fab" onclick="openVideoHub()" title="Video Hub">
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
      if (!shouldDelayDiscover) loadFeedLiveSpotlight();
    };
  });

  if (!shouldDelayDiscover) {
    loadDiscoverPeople();
    loadFeedLiveSpotlight();
  }
  loadCampusEvents().then(() => {
    if (Array.isArray(state.posts) && state.posts.length) renderFeedResults(state.posts);
  }).catch(() => {});

  const searchInput = $('#feed-search-input');
  if (searchInput) {
    let timer = null;
    searchInput.addEventListener('input', e => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        _feedSearchQuery = e.target.value || '';
        renderFeedResults(state.posts || []);
        renderFeedPeopleSuggestions(e.target.value.trim());
      }, 120);
    });
    searchInput.addEventListener('focus', () => {
      if (searchInput.value.trim()) renderFeedPeopleSuggestions(searchInput.value.trim());
    });
    searchInput.addEventListener('blur', () => {
      setTimeout(() => { const box = $('#feed-people-suggestions'); if (box) box.style.display = 'none'; }, 200);
    });
  }

  // Real-time posts
  const u = db.collection('posts').orderBy('createdAt', 'desc').limit(50)
    .onSnapshot(async snap => {
      const myFriends = state.profile.friends || [];
      const uid = state.user.uid;
      const allPosts = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      const expiredPinned = [];
      const activePosts = allPosts.filter(post => {
        const expiry = post.locationPing?.expiresAt?.toDate ? post.locationPing.expiresAt.toDate() : (post.locationPing?.expiresAt ? new Date(post.locationPing.expiresAt) : null);
        const isExpiredPinned = !!post.locationPing?.enabled && expiry && expiry.getTime() <= Date.now();
        if (isExpiredPinned) expiredPinned.push(post.id);
        return !isExpiredPinned;
      });
      expiredPinned.forEach(pid => db.collection('posts').doc(pid).delete().catch(() => {}));
      // Filter: show public posts, own posts, and friends-only posts from friends
      const visible = activePosts.filter(post => {
        if (isPostShadowHiddenForViewer(post, uid)) return false;
        if (post.authorId === uid) return true;
        if (post.visibility === 'friends') return myFriends.includes(post.authorId);
        return true; // public or no visibility set
      });

      const previousPosts = state.posts || [];
      const previousOrder = new Map(previousPosts.map((post, index) => [post.id, index]));
      const likedTagPrefs = new Set();
      const likedModulePrefs = new Set(normalizeModules(state.profile.modules || []));
      (state.posts || []).forEach(existingPost => {
        if ((existingPost.likes || []).includes(uid)) {
          normalizeModules(existingPost.moduleTags || []).forEach(tag => likedModulePrefs.add(tag));
          getPostHashTags(existingPost).forEach(tag => likedTagPrefs.add(tag));
        }
      });
      await ensureUserContextCache(visible.map(post => post.authorId));
      const moderatedVisible = visible.filter(post => {
        const authorContext = _userContextCache[post.authorId] || {};
        if (post.authorId === uid) return true;
        if (_isAdmin) return true;
        return !authorContext.shadowMuted;
      });

      // Discovery ranking with relevancy, randomness per session, and seen-post demotion
      const seenMap = getSeenPosts();
      const scored = moderatedVisible.map(p => {
        const likes = (p.likes || []).length;
        const comments = p.commentsCount || 0;
        const isFriend = myFriends.includes(p.authorId);
        const authorContext = _userContextCache[p.authorId] || null;
        const nearbyBoost = authorContext ? getNearbySignal(state.profile, authorContext).score * 4 : 0;
        const sharedModules = normalizeModules(p.moduleTags || []).filter(tag => likedModulePrefs.has(tag)).length;
        const sharedTags = getPostHashTags(p).filter(tag => likedTagPrefs.has(tag)).length;
        const ageHrs = p.createdAt ? (Date.now() - (p.createdAt.toDate ? p.createdAt.toDate() : new Date(p.createdAt)).getTime()) / 3600000 : 999;
        const freshness = Math.max(0, 1 - ageHrs / 48); // decay over 48h
        const engagement = (likes * 2 + comments * 3) * 0.3;
        const friendBoost = isFriend ? 8 : 0;
        const interestBoost = sharedModules * 10 + sharedTags * 5;
        const randomFactor = sessionSeed(p.id) * 12;
        // Demote posts the user has already seen (more views → bigger penalty, caps at 25)
        const seenEntry = seenMap[p.id];
        const seenPenalty = seenEntry ? Math.min(seenEntry.v * 4, 25) : 0;
        return { ...p, _score: engagement + freshness * 15 + friendBoost + nearbyBoost + interestBoost + randomFactor - seenPenalty };
      });
      scored.sort((a, b) => {
        const ai = previousOrder.has(a.id) ? previousOrder.get(a.id) : -1;
        const bi = previousOrder.has(b.id) ? previousOrder.get(b.id) : -1;
        if (ai !== -1 && bi !== -1) return ai - bi;
        if (ai !== -1) return 1;
        if (bi !== -1) return -1;
        return b._score - a._score || (b.createdAt?.seconds || 0) - (a.createdAt?.seconds || 0);
      });

      const prevIds = previousPosts.map(post => post.id).join(',');
      const nextIds = scored.map(post => post.id).join(',');
      // ── Diff-based update: only re-render fully when post list changes ──
      const idsUnchanged = prevIds === nextIds;
      const onlyModifications = snap.docChanges().every(ch => ch.type === 'modified');

      state.posts = scored;

      if (idsUnchanged && onlyModifications) {
        // Just refresh the like/reaction UI on changed post cards — no full re-render
        snap.docChanges().forEach(ch => {
          const post = scored.find(p => p.id === ch.doc.id);
          if (!post) return;
          const card = document.getElementById('post-' + post.id);
          if (!card) return;
          const liked = (post.likes || []).includes(uid);
          const myReaction = getUserReaction(post.reactions, post.likes || []);
          const lc = getReactionSummary(post.reactions, post.likes || []).total;
          const likeBtn = card.querySelector('.post-like-action');
          if (likeBtn) {
            likeBtn.className = `post-action post-like-action ${liked ? 'liked' : ''} ${myReaction && myReaction !== '❤️' ? 'reacted' : ''}`;
            likeBtn.innerHTML = renderPostLikeMarkup(post);
          }
          const statsEl = card.querySelector('.post-stats');
          if (statsEl) statsEl.innerHTML = renderPostStatsMarkup(post);
        });
        window._lastLikedPost = null;
        c.style.opacity = '';
        _feedRestorePendingPaint = false;
        _pendingFeedScrollRestore = null;
        ensureDiscoverLoaded();
        return;
      }

      const contentEl = document.getElementById('content');
      const restoreScroll = _pendingFeedScrollRestore !== null
        ? _pendingFeedScrollRestore
        : (contentEl ? contentEl.scrollTop : _feedScrollTop);

      renderFeedResults(scored);
      if (contentEl) {
        requestAnimationFrame(() => {
          contentEl.scrollTop = restoreScroll;
          c.style.opacity = '';
          _feedRestorePendingPaint = false;
          ensureDiscoverLoaded();
        });
      } else {
        c.style.opacity = '';
        _feedRestorePendingPaint = false;
        ensureDiscoverLoaded();
      }
      _pendingFeedScrollRestore = null;
      window._lastLikedPost = null;
    }, err => {
      console.error('Feed listener error:', err);
      c.style.opacity = '';
      _feedRestorePendingPaint = false;
      ensureDiscoverLoaded();
      const postsEl = document.getElementById('feed-posts');
      if (postsEl) {
        postsEl.innerHTML = `<div class="empty-state"><div class="empty-state-icon">⚠️</div><h3>Could not load posts</h3><p>${isInvalidSessionError(err) ? 'Your session expired. Log in again.' : (isPermissionDeniedError(err) ? 'This account cannot load posts right now.' : 'Reopen the app or try again in a moment.')}</p></div>`;
      }
      if (isInvalidSessionError(err)) {
        recoverInvalidSession(err, 'Feed listener denied').catch(() => {});
      }
    });
  state.unsubs.push(u);
}

// ─── Discover: People tab ────────────────────────
function loadDiscoverPeople() {
  const el = $('#discover-content'); if (!el) return;
  const myMajor = state.profile.major || '';
  const myModules = normalizeModules(state.profile.modules || []);
  const myYear = state.profile.year || '';
  const blockedUsers = new Set(state.profile.blockedUsers || []);
  const blockedBy = new Set(state.profile.blockedBy || []);
  const myFriends = new Set(state.profile.friends || []);
  const sentRequests = new Set(state.profile.sentRequests || []);

  getUsersCache().then(allUsers => {
    let users = allUsers
      .filter(u => u.id !== state.user.uid && !blockedUsers.has(u.id) && !blockedBy.has(u.id))
      .filter(u => !myFriends.has(u.id) && !sentRequests.has(u.id));

    // Blend relevance with session randomness so suggestions rotate between feed visits.
    users = users.map(u => {
      let score = 0;
      const theirModules = normalizeModules(u.modules || []);
      const shared = myModules.filter(m => theirModules.includes(m));
      const nearby = getNearbySignal(state.profile, u);
      const nearbyScore = nearby.score;
      const isFriend = myFriends.has(u.id);
      if (shared.length) score += 36 + shared.length * 14;
      if (u.major === myMajor) score += 22;
      if (u.year && myYear && u.year === myYear) score += 8;
      if (nearbyScore > 0) score += 18 + nearbyScore * 7;
      if (u.status === 'online') score += 5;
      score += genderAffinityScore(state.profile, u);
      if (isFriend) score -= 18;
      if (!shared.length && u.major !== myMajor && nearbyScore === 0) score -= 12;
      score += sessionSeed(`discover-rank:${u.id}`) * 8;
      return { ...u, isFriend, score, sharedModules: shared, nearbyScore, distanceKm: nearby.distanceKm, nearbySource: nearby.source };
    });

    const relevant = users
      .filter(u => u.sharedModules.length || u.nearbyScore > 0 || u.major === myMajor || u.score >= 18)
      .sort((a, b) => b.score - a.score || (a.distanceKm || Infinity) - (b.distanceKm || Infinity));

    const relevantIds = new Set(relevant.map(u => u.id));
    const randomFallback = users
      .filter(u => !relevantIds.has(u.id))
      .sort((a, b) => sessionSeed(`discover-fallback:${a.id}`) - sessionSeed(`discover-fallback:${b.id}`));

    const relevantTake = Math.min(7, relevant.length);
    users = [...relevant.slice(0, relevantTake), ...randomFallback.slice(0, 10 - relevantTake)];
    if (users.length < 10) users.push(...relevant.slice(relevantTake, relevantTake + (10 - users.length)));
    users = users.slice(0, 10);

    if (!users.length) {
      el.innerHTML = `<div class="discover-empty"><span>👥</span><p>No students found yet. Invite friends!</p></div>`;
      return;
    }

    el.innerHTML = `<div class="discover-scroll">${users.map(u => {
      const tag = u.sharedModules?.length
        ? `🔗 ${u.sharedModules.length} shared module${u.sharedModules.length > 1 ? 's' : ''}`
        : Number.isFinite(u.distanceKm) ? `📍 ${u.distanceKm.toFixed(1)} km away`
        : u.nearbyScore > 0 ? '📍 Nearby area'
        : u.major ? `📚 ${esc(u.major)}` : '';
      const online = u.status === 'online' ? '<span class="online-dot"></span>' : '';
      const isFriend = (state.profile.friends || []).includes(u.id);
      const isPending = (state.profile.sentRequests || []).includes(u.id);
      const actionBtn = isFriend
        ? `<button class="discover-card-btn" onclick="event.stopPropagation();startChat('${u.id}','${esc(u.displayName)}','${u.photoURL || ''}')">Message</button>`
        : isPending
          ? `<button class="discover-card-btn" disabled style="opacity:0.6;cursor:not-allowed">Pending…</button>`
          : `<button type="button" class="discover-card-btn" onclick="event.preventDefault();event.stopPropagation();sendFriendRequest('${u.id}','${esc(u.displayName)}','${u.photoURL || ''}', this)">Add Friend</button>`;
      return `
        <div class="discover-card" onclick="if(event.target.closest('button')) return; openProfile('${u.id}')">
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
    const interestProfile = buildInterestProfile();
    const rankedEvents = [...events].map(ev => {
      const loc = getCampusLocationById(ev.location);
      const locationBoost = loc ? getLocationDistanceBoost({ lat: loc.lat, lng: loc.lng }) : 0;
      const goingBoost = Math.min(10, (ev.going || []).length * 1.5);
      const interestBoost = textInterestScore(`${ev.title || ''} ${ev.description || ''} ${ev.location || ''}`, interestProfile);
      return { ...ev, _discoverScore: locationBoost + goingBoost + interestBoost };
    }).sort((a, b) => b._discoverScore - a._discoverScore || (a.date || '').localeCompare(b.date || ''));
    el.innerHTML = `<div class="discover-scroll">${rankedEvents.slice(0, 8).map(ev => {
      const loc = CAMPUS_LOCATIONS.find(l => l.id === ev.location);
      const locName = loc ? loc.name : esc(ev.location || '?');
      const nearLabel = loc && getLocationDistanceBoost({ lat: loc.lat, lng: loc.lng }) > 0 ? ' · Near you' : '';
      const grad = ev.gradient || 'linear-gradient(135deg,#6C5CE7,#A855F7)';
      const goingCount = (ev.going || []).length;
      const thumb = (ev.imageURLs && ev.imageURLs.length) ? ev.imageURLs[0] : null;
      return `
        <div class="discover-card event-card" onclick="${ev.id ? `openEventDetail('${ev.id}')` : `toast('View on Campus map!')`}">
          ${thumb ? `<img src="${thumb}" style="width:100%;height:120px;object-fit:cover;border-radius:var(--radius);margin-bottom:8px">` : `<div style="background:${grad};width:100%;height:120px;border-radius:var(--radius);display:flex;align-items:center;justify-content:center;font-size:36px;margin-bottom:8px">${ev.emoji || '📅'}</div>`}
          <div class="discover-card-name">${esc(ev.title)}</div>
          <div class="discover-card-meta">${esc(ev.date || '')} ${esc(ev.time || '')}</div>
          <div class="discover-card-tag">📍 ${locName}${nearLabel}</div>
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
function cleanupExpiredStories() {
  const cutoff = new Date();
  db.collection('stories').where('expiresAt', '<=', cutoff).limit(50).get()
    .then(snap => { snap.docs.forEach(d => d.ref.delete().catch(() => {})); })
    .catch(() => {});
}

function loadStories() {
  const row = $('#stories-row') || $('#messages-stories-row'); if (!row) return;
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
        const identity = getResolvedUserIdentity(group.uid, s.authorName || s.authorFirstName || 'User', s.authorPhoto || null);
        const shortName = (identity.name || s.authorFirstName || s.authorName || '?').split(' ')[0] || '?';
        const name = isMe ? 'You' : esc(shortName);
        const hasNew = group.stories.some(st => !(st.viewedBy || []).includes(uid));
        row.insertAdjacentHTML('beforeend', `
          <div class="story-item ${hasNew ? 'has-unseen' : 'seen'}" onclick="event.stopPropagation();viewStory('${group.uid}')">
            <div class="story-avatar"><div class="story-avatar-inner">
              ${s.authorPhoto ? `<img src="${s.authorPhoto}" alt="" draggable="false">` : initials(identity.name || s.authorName)}
            </div></div>
            <div class="story-name">${name}</div>
          </div>
        `);
      });

    }).catch(() => {});
}

function setStoryComposerMode(hasMedia = false) {
  const textBg = $('#story-text-bg');
  const textInput = $('#story-text-input');
  const picker = $('#story-bg-picker');
  if (textBg) textBg.classList.toggle('story-caption-mode', !!hasMedia);
  if (textInput) textInput.placeholder = hasMedia ? 'Add a caption...' : 'Type your story...';
  if (picker) picker.style.display = hasMedia ? 'none' : 'flex';
}

function clearStoryMediaSelection() {
  window._storyFile = null;
  window._storyVideoFile = null;
  const preview = $('#story-media-preview');
  const content = $('#story-media-content');
  const input = $('#story-media-file');
  if (preview) preview.style.display = 'none';
  if (content) content.innerHTML = '';
  if (input) input.value = '';
  setStoryComposerMode(false);
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
        <div class="story-bg-picker" id="story-bg-picker">${bgOptions.map(c => `<button class="bg-dot" style="background:${c}" onclick="document.getElementById('story-text-bg').style.background='${c}';window._storyBg='${c}'"></button>`).join('')}</div>
        <div id="story-media-preview" style="display:none;margin-top:12px;position:relative;border-radius:var(--radius);overflow:hidden">
          <div id="story-media-content"></div>
          <button onclick="clearStoryMediaSelection()" style="position:absolute;top:6px;right:6px;width:28px;height:28px;border-radius:50%;background:rgba(0,0,0,0.6);color:#fff;border:none;font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center">&times;</button>
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
  setStoryComposerMode(false);
  $('#story-media-file').onchange = e => {
    const file = e.target.files[0];
    if (!file) return;
    const isVideo = file.type.startsWith('video/');
    const preview = $('#story-media-preview');
    const content = $('#story-media-content');
    if (isVideo) {
      window._storyVideoFile = file;
      window._storyFile = null;
      content.innerHTML = `<video src="${localPreview(file)}" style="width:100%;max-height:220px;object-fit:cover;border-radius:var(--radius);background:#000" autoplay muted loop playsinline></video>`;
    } else {
      window._storyFile = file;
      window._storyVideoFile = null;
      content.innerHTML = `<img src="${localPreview(file)}" style="width:100%;max-height:220px;object-fit:cover;border-radius:var(--radius)">`;
    }
    preview.style.display = 'block';
    setStoryComposerMode(true);
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
    const myFriends = state.profile.friends || [];
    snap.docs.forEach(d => {
      const s = { id: d.id, ...d.data() };
      // Only show own stories and friends' stories
      if (s.authorId !== state.user.uid && !myFriends.includes(s.authorId)) return;
      // Delete expired stories encountered (cleanup)
      if (s.expiresAt?.toDate && s.expiresAt.toDate() < new Date()) {
        db.collection('stories').doc(d.id).delete().catch(() => {});
        return;
      }
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
  const storyAuthorIdentity = getResolvedUserIdentity(story.authorId, story.authorName || 'User', story.authorPhoto || null);
  const viewCount = Math.max(0, (story.viewedBy || []).filter(uid => uid !== state.user.uid).length);
  hdr.innerHTML = `
    ${avatar(storyAuthorIdentity.name, storyAuthorIdentity.photo, 'avatar-sm')}
    <div><b>${esc(storyAuthorIdentity.name)}</b><br><small>${timeAgo(story.createdAt)}${isMyStory ? ` · <button class="story-insight-link" onclick="event.stopPropagation();openStoryInsights('${story.id}')">${viewCount} view${viewCount === 1 ? '' : 's'}</button>` : ''}</small></div>
    ${isMyStory ? `<button class="story-delete-btn" onclick="deleteStory('${story.id}')">Delete</button>` : ''}
  `;

  // Content
  const content = $('#story-viewer-content');
  let autoAdvanceMs = 5000;
  if (story.type === 'video' && story.videoURL) {
    content.innerHTML = `
      <video src="${story.videoURL}" class="story-full-video" autoplay muted playsinline loop style="width:100%;height:100%;object-fit:cover"></video>
      <button class="story-sound-toggle" id="story-sound-toggle" onpointerdown="event.stopPropagation()" onpointerup="event.stopPropagation()" onclick="toggleStoryViewerSound(event)">Unmute</button>
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

  // Story reply bar (only for others' stories)
  const replyBar = $('#story-reply-bar');
  const replyInput = $('#story-reply-input');
  const replySend = $('#story-reply-send');
  const storyCaption = content.querySelector('.story-caption');
  if (storyCaption) storyCaption.classList.toggle('with-reply', !isMyStory);
  if (replyBar) {
    if (!isMyStory) {
      replyBar.style.display = 'flex';
      replyInput.value = '';
      if (replySend) {
        replySend.type = 'button';
        replySend.innerHTML = '❤';
        replySend.title = 'Like story';
      }
      replyInput.onfocus = () => clearTimeout(storyViewerData.timer);
      replyInput.onblur = () => { if (!storyViewerData.paused) storyViewerData.timer = setTimeout(() => advanceStory(1), autoAdvanceMs); };
      const doReply = () => {
        const text = replyInput.value.trim();
        if (!text) return;
        replyInput.value = '';
        sendStoryReply(story, text);
      };
      replySend.onclick = () => toggleStoryLike(story);
      replyInput.onkeydown = e => { if (e.key === 'Enter') { e.preventDefault(); doReply(); } };
    } else {
      replyBar.style.display = 'none';
    }
  }

  // Navigation
  $('#story-prev').onclick = () => advanceStory(-1);
  $('#story-next').onclick = () => advanceStory(1);
  $('#story-close').onclick = closeStoryViewer;
  const storyContentEl = $('#story-viewer-content');
  if (storyContentEl) {
    storyContentEl.onpointerdown = () => pauseStoryViewer();
    storyContentEl.onpointerup = () => resumeStoryViewer(autoAdvanceMs);
    storyContentEl.onpointercancel = () => resumeStoryViewer(autoAdvanceMs);
    storyContentEl.onclick = () => {
      if (storyViewerData.paused) resumeStoryViewer(autoAdvanceMs);
      else pauseStoryViewer();
    };
  }
}

function pauseStoryViewer() {
  clearTimeout(storyViewerData.timer);
  storyViewerData.paused = true;
  const vid = $('#story-viewer-content video');
  if (vid && !vid.paused) vid.pause();
}

function resumeStoryViewer(autoAdvanceMs = 5000) {
  const wasPaused = !!storyViewerData.paused;
  storyViewerData.paused = false;
  const vid = $('#story-viewer-content video');
  if (vid && vid.paused) vid.play().catch(() => {});
  clearTimeout(storyViewerData.timer);
  if (wasPaused) storyViewerData.timer = setTimeout(() => advanceStory(1), autoAdvanceMs);
}

async function openStoryInsights(storyId = '') {
  try {
    const doc = await db.collection('stories').doc(storyId).get();
    if (!doc.exists) return;
    const story = { id: doc.id, ...doc.data() };
    const viewerIds = (story.viewedBy || []).filter(uid => uid && uid !== state.user?.uid);
    const likeIds = (story.likes || []).filter(Boolean);
    const ids = [...new Set([...viewerIds, ...likeIds])];
    await ensureUserContextCache(ids, { force: false });
    const renderRows = (arr = [], kind = 'Viewed') => arr.length
      ? arr.map(uid => {
          const info = resolveUserIdentity(uid, uid, null);
          return `<button class="cluster-user-row" onclick="closeModal();openProfile('${uid}')">${avatar(info.name, info.photo, 'avatar-sm')}<span>${esc(info.name)}</span><small style="margin-left:auto;color:var(--text-secondary)">${kind}</small></button>`;
        }).join('')
      : `<div class="empty-state" style="padding:10px 0"><p>No one yet</p></div>`;
    openModal(`
      <div class="modal-header"><h2>Story activity</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
      <div class="modal-body" style="padding:16px;display:grid;gap:14px">
        <div><h3 style="margin:0 0 8px">Views</h3>${renderRows(viewerIds, 'Viewed')}</div>
        <div><h3 style="margin:0 0 8px">Likes</h3>${renderRows(likeIds, 'Liked')}</div>
      </div>
    `);
  } catch (e) { console.error(e); }
}

async function sendStoryLikeMessage(story = {}) {
  const authorId = story.authorId || '';
  if (!authorId || authorId === state.user?.uid) return;
  try {
    const convoId = await ensureFriendDMConversation(authorId, story.authorName || 'User', story.authorPhoto || null);
    await db.collection('conversations').doc(convoId).collection('messages').add({
      text: '',
      senderId: state.user.uid,
      senderName: state.profile?.displayName || 'Someone',
      senderPhoto: state.profile?.photoURL || null,
      senderAnon: false,
      type: 'story_like',
      payload: {
        storyId: story.id || '',
        storyType: story.type || '',
        storyCaption: story.caption || story.text || ''
      },
      createdAt: FieldVal.serverTimestamp(),
      status: 'sent'
    });
    await db.collection('conversations').doc(convoId).set({
      lastMessage: `story_like::${state.user.uid}`,
      updatedAt: FieldVal.serverTimestamp(),
      unread: { [authorId]: FieldVal.increment(1), [state.user.uid]: 0 }
    }, { merge: true });
    await addNotification(authorId, 'message', 'liked your story', { convoId, storyId: story.id || '' });
  } catch (e) {
    console.error('story like message', e);
  }
}

async function toggleStoryLike(story = {}) {
  const storyId = story.id || '';
  if (!storyId || !state.user?.uid) return;
  const ref = db.collection('stories').doc(storyId);
  const myId = state.user.uid;
  const likes = Array.isArray(story.likes) ? [...story.likes] : [];
  const liked = likes.includes(myId);
  try {
    await ref.update({ likes: liked ? FieldVal.arrayRemove(myId) : FieldVal.arrayUnion(myId) });
    story.likes = liked ? likes.filter(id => id !== myId) : [...likes, myId];
    if (!liked) {
      await sendStoryLikeMessage(story);
    }
  } catch (e) {
    console.error(e);
  }
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
  // Stop any playing video/audio
  const vid = $('#story-viewer-content video');
  if (vid) { vid.pause(); vid.src = ''; }
  const aud = $('#story-viewer-content audio');
  if (aud) { aud.pause(); aud.src = ''; }
  $('#story-viewer').style.display = 'none';
}

function toggleStoryViewerSound(event) {
  if (event) {
    event.preventDefault();
    event.stopPropagation();
  }
  const vid = $('#story-viewer-content video');
  const btn = $('#story-sound-toggle');
  if (!vid || !btn) return;
  vid.muted = !vid.muted;
  btn.textContent = vid.muted ? 'Unmute' : 'Mute';
}

async function sendStoryReply(story, text) {
  const authorId = story.authorId;
  if (!authorId || authorId === state.user.uid) return;
  try {
    // Find or create a regular (non-anon) DM with the story author
    const snap = await db.collection('conversations').where('participants', 'array-contains', state.user.uid).get();
    let convoId = null;
    const existing = snap.docs.find(d => {
      const data = d.data();
      return data.participants.includes(authorId) && !data.isAnonymous;
    });
    if (existing) {
      convoId = existing.id;
    } else {
      // Create regular conversation
      const doc = await db.collection('conversations').add({
        participants: [state.user.uid, authorId],
        participantNames: [state.profile.displayName, story.authorName || 'User'],
        participantPhotos: [state.profile.photoURL || null, story.authorPhoto || null],
        lastMessage: '', updatedAt: FieldVal.serverTimestamp(),
        unread: { [authorId]: 0, [state.user.uid]: 0 },
        participantStatuses: { [state.user.uid]: state.status, [authorId]: 'offline' }
      });
      convoId = doc.id;
    }
    // Build a preview URL for the story
    const storyPreview = story.type === 'photo' ? story.imageURL : (story.type === 'video' ? story.videoURL : null);
    await db.collection('conversations').doc(convoId).collection('messages').add({
      text,
      senderId: state.user.uid,
      senderAnon: false,
      type: 'story_reply',
      payload: {
        storyId: story.id,
        storyType: story.type,
        storyPreview: storyPreview || null,
        storyCaption: story.caption || story.text || '',
        storyAuthorName: story.authorName || 'User'
      },
      createdAt: FieldVal.serverTimestamp(),
      status: 'sent'
    });
    const lastMsg = `Replied to story: ${text}`;
    await db.collection('conversations').doc(convoId).set({
      lastMessage: lastMsg, updatedAt: FieldVal.serverTimestamp(),
      unread: { [authorId]: FieldVal.increment(1), [state.user.uid]: 0 }
    }, { merge: true });
    await addNotification(authorId, 'message', 'sent you a story reply', { convoId });
    toast('Reply sent!');
  } catch (e) { console.error(e); toast('Failed to send reply'); }
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
  const allJson = JSON.stringify(urls).replace(/'/g, '&#39;').replace(/"/g, '&quot;');
  const cls = `collage-grid collage-${Math.min(count, 4)}`;
  return `<div class="${cls}">${urls.slice(0, 4).map((url, i) =>
    `<div class="collage-item${count > 4 && i === 3 ? ' collage-more-item' : ''}" onclick="openGallery(JSON.parse(this.closest('.collage-grid').dataset.urls),${i})">
      <img src="${url}" loading="lazy">
      ${count > 4 && i === 3 ? `<div class="collage-more-overlay">+${count - 4}</div>` : ''}
    </div>`
  ).join('')}</div>`.replace('class="collage-grid', `data-urls="${allJson}" class="collage-grid`);
}

// ─── Quote Embed Renderer ────────────────────────
const _pendingQuotePlayers = [];
function renderQuoteEmbed(rp, options = {}) {
  if (!rp) return '';
  const { repostStyle = false } = options;
  const hasVid = !!(rp.videoURL || (rp.mediaType === 'video') || (rp.imageURL && /\.(mp4|webm|mov|m4v)(\?|#|$)/i.test(rp.imageURL)));
  const hasImg = !!(rp.imageURL && !hasVid);
  const vidUrl = hasVid ? (rp.videoURL || rp.imageURL) : null;
  let vidHtml = '';
  if (hasVid && vidUrl) {
    const vpd = createVideoPlayer(vidUrl);
    _pendingQuotePlayers.push(vpd);
    vidHtml = `<div onclick="event.stopPropagation()" style="border-radius:8px;overflow:hidden;margin-top:8px">${vpd.html}</div>`;
  }
  const quoteImgMax = repostStyle ? 260 : 200;
  return `
    <div class="quote-embed${repostStyle ? ' quote-embed-repost' : ''}" onclick="${rp.id ? `viewPost('${rp.id}')` : ''}" style="cursor:pointer;border:1px solid var(--border);border-radius:var(--radius);padding:12px;margin:8px 0;background:var(--bg-secondary)">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
        ${avatar(rp.authorName || 'User', rp.authorPhoto, 'avatar-sm')}
        <span style="font-weight:600;font-size:13px">${esc(rp.authorName || 'User')}</span>
      </div>
      ${rp.content ? `<div style="font-size:13px;color:var(--text-secondary);margin-bottom:8px;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden">${esc(rp.content)}</div>` : ''}
      ${hasImg && rp.imageURL ? `<img src="${rp.imageURL}" style="width:100%;max-height:${quoteImgMax}px;object-fit:cover;border-radius:8px" onclick="event.stopPropagation();viewImage('${rp.imageURL}')">` : ''}
      ${vidHtml}
    </div>`;
}

function renderFeedInlineSuggestionCard(user = {}) {
  if (!user?.id) return '';
  const resolved = getResolvedUserIdentity(user.id, user.displayName || 'User', user.photoURL || null);
  const displayName = resolved.name || user.displayName || 'User';
  const meta = formatUserMetaLine(user, 48);
  const isFriend = (state.profile?.friends || []).includes(user.id);
  const isPending = (state.profile?.sentRequests || []).includes(user.id);
  const action = isFriend
    ? `<button class="discover-card-btn" onclick="event.stopPropagation();startChat('${user.id}','${esc(user.displayName || 'User')}','${user.photoURL || ''}')">Message</button>`
    : isPending
      ? `<button class="discover-card-btn" disabled style="opacity:0.6;cursor:not-allowed">Pending…</button>`
      : `<button type="button" class="discover-card-btn" onclick="event.preventDefault();event.stopPropagation();sendFriendRequest('${user.id}','${esc(displayName)}','${user.photoURL || ''}', this)">Add Friend</button>`;
  return `
    <div class="post-card feed-inline-suggestion" onclick="if(event.target.closest('button')) return; openProfile('${user.id}')">
      <div class="suggested-header" style="padding:14px 16px 6px;margin:0">
        <h3>People you may know</h3>
      </div>
      <div class="discover-card" style="margin:0 14px 14px">
        <div class="discover-card-avatar">
          ${avatar(displayName, user.photoURL, 'avatar-lg')}
          ${user.status === 'online' ? '<span class="online-dot"></span>' : ''}
        </div>
        <div class="discover-card-name">${esc(displayName)}</div>
        <div class="discover-card-meta" title="${esc(formatUserMetaLine(user, 120))}">${esc(meta)}${user.gender ? ` · ${esc(user.gender)}` : ''}</div>
        ${action}
      </div>
    </div>`;
}

function pickFeedSuggestedEvent() {
  if (!Array.isArray(allCampusEvents) || !allCampusEvents.length) return null;
  const now = Date.now();
  const upcoming = allCampusEvents.filter(ev => eventStartTimeMs(ev) >= now - (30 * 60 * 1000));
  return (upcoming[0] || allCampusEvents[0] || null);
}

function renderFeedInlineEventCard(eventDoc = {}) {
  if (!eventDoc?.id) return '';
  const thumb = (eventDoc.imageURLs && eventDoc.imageURLs.length) ? eventDoc.imageURLs[0] : '';
  const gradient = eventDoc.gradient || 'linear-gradient(135deg,#6C5CE7,#A855F7)';
  const loc = getCampusLocationById(eventDoc.location);
  const locationLabel = loc ? loc.name : (eventDoc.location || 'Campus');
  const goingCount = Array.isArray(eventDoc.going) ? eventDoc.going.length : 0;
  return `<div class="suggested-card suggested-event-card" onclick="openEventDetail('${eventDoc.id}')">
    <div class="suggested-event-thumb">${thumb
      ? `<img src="${thumb}" loading="lazy" decoding="async" alt="${esc(eventDoc.title || 'Event')}">`
      : `<div class="suggested-event-gradient" style="background:${gradient}">${eventDoc.emoji || '📅'}</div>`}</div>
    <div class="suggested-card-name">${esc(eventDoc.title || 'Campus Event')}</div>
    <div class="suggested-card-meta">📍 ${esc(locationLabel)}</div>
    <div class="suggested-card-meta">📅 ${esc(eventDoc.date || 'TBA')} ${eventDoc.time ? esc(eventDoc.time) : ''}</div>
    <button class="btn-sm btn-outline" style="pointer-events:none">${goingCount ? `${goingCount} going` : 'View event'}</button>
  </div>`;
}

function buildFeedSuggestedPeople(users = [], limit = 15) {
  const blockedUsers = new Set(state.profile?.blockedUsers || []);
  const blockedBy = new Set(state.profile?.blockedBy || []);
  const myFriends = new Set(state.profile?.friends || []);
  const sentRequests = new Set(state.profile?.sentRequests || []);
  const myModules = normalizeModules(state.profile?.modules || []);
  const myMajor = (state.profile?.major || '').toLowerCase();

  const candidates = (users || []).filter(u => {
    if (!u?.id || u.id === state.user?.uid) return false;
    if (_feedInlineSuggestedUserIds.has(u.id)) return false;
    if (myFriends.has(u.id) || sentRequests.has(u.id)) return false;
    if (blockedUsers.has(u.id) || blockedBy.has(u.id)) return false;
    return true;
  }).map(u => {
    const overlap = normalizeModules(u.modules || []).filter(tag => myModules.includes(tag)).length;
    const majorBoost = (u.major || '').toLowerCase() === myMajor ? 3 : 0;
    const nearby = getNearbySignal(state.profile || {}, u || {});
    const nearbyBoost = nearby.score > 0 ? (2 + nearby.score * 1.5) : 0;
    const genderBoost = genderAffinityScore(state.profile || {}, u || {}) * 0.3;
    const onlineBoost = u.status === 'online' ? 1.2 : 0;
    const relevance = overlap * 4.2 + majorBoost + nearbyBoost + genderBoost + onlineBoost;
    return { ...u, _relevance: relevance, _overlap: overlap };
  });

  if (!candidates.length) return [];
  const relevant = candidates
    .filter(u => u._overlap > 0 || u._relevance >= 6)
    .sort((a, b) => (b._relevance + sessionSeed(`feed-sugg-rel:${b.id}`) * 4) - (a._relevance + sessionSeed(`feed-sugg-rel:${a.id}`) * 4));
  const relevantIds = new Set(relevant.map(u => u.id));
  const randomPool = candidates
    .filter(u => !relevantIds.has(u.id))
    .sort((a, b) => sessionSeed(`feed-sugg-rand:${a.id}`) - sessionSeed(`feed-sugg-rand:${b.id}`));

  const takeRelevant = Math.min(relevant.length, Math.max(4, Math.floor(limit * 0.6)));
  const picks = [...relevant.slice(0, takeRelevant)];
  if (picks.length < limit) picks.push(...randomPool.slice(0, limit - picks.length));
  if (picks.length < limit) picks.push(...relevant.slice(takeRelevant, takeRelevant + (limit - picks.length)));

  return picks
    .slice(0, limit)
    .sort((a, b) => sessionSeed(`feed-sugg-order:${a.id}`) - sessionSeed(`feed-sugg-order:${b.id}`));
}

function renderFeedInlineSuggestionRail(users = []) {
  const picks = (users || []).slice(0, 10);
  const eventPicks = [...(allCampusEvents || [])].slice(0, Math.min(2, Math.max(0, 6 - picks.length)));
  if (!picks.length && !eventPicks.length) return '';
  return `
    <div class="post-card feed-inline-suggestion feed-inline-suggestion-rail">
      <div class="suggested-header" style="padding:14px 16px 6px;margin:0">
        <h3>Suggestions</h3>
      </div>
      <div class="suggested-list">${picks.map(user => {
        const resolved = getResolvedUserIdentity(user.id, user.displayName || 'User', user.photoURL || null);
        const displayName = resolved.name || user.displayName || 'User';
        const meta = formatUserMetaLine(user, 44);
        const isFriend = (state.profile?.friends || []).includes(user.id);
        const isPending = (state.profile?.sentRequests || []).includes(user.id);
        const action = isFriend
          ? `<button class="btn-sm btn-outline" onclick="event.stopPropagation();startChat('${user.id}','${esc(user.displayName || 'User')}','${user.photoURL || ''}')">Message</button>`
          : isPending
            ? `<button class="btn-sm btn-outline" disabled style="opacity:0.6;cursor:not-allowed">Pending…</button>`
            : `<button type="button" class="btn-sm btn-primary" onclick="event.preventDefault();event.stopPropagation();sendFriendRequest('${user.id}','${esc(displayName)}','${user.photoURL || ''}', this)">Add</button>`;
          return `<div class="suggested-card" onclick="if(event.target.closest('button')) return; openProfile('${user.id}')">
          ${avatar(displayName, user.photoURL, 'avatar-lg')}
          <div class="suggested-card-name">${esc(displayName)}</div>
          <div class="suggested-card-meta" title="${esc(formatUserMetaLine(user, 120))}">${esc(meta)}${user.gender ? ` · ${esc(user.gender)}` : ''}</div>
          ${action}
        </div>`;
      }).join('')}
      ${eventPicks.map(ev => {
        const loc = getCampusLocationById(ev.location);
        const thumb = (ev.imageURLs && ev.imageURLs.length) ? ev.imageURLs[0] : '';
        return `<div class="suggested-card suggested-card-event" onclick="openEventDetail('${ev.id || ''}')">
          ${thumb ? `<img src="${thumb}" class="suggested-event-thumb" loading="lazy">` : `<div class="suggested-event-emoji">${esc(ev.emoji || '📅')}</div>`}
          <div class="suggested-card-name">${esc(ev.title || 'Campus event')}</div>
          <div class="suggested-card-meta">${esc(loc ? loc.name : (ev.location || 'Event'))}</div>
          <button class="btn-sm btn-outline" onclick="event.preventDefault();event.stopPropagation();openEventDetail('${ev.id || ''}')">Open</button>
        </div>`;
      }).join('')}</div>
    </div>`;
}

function renderPostCommentPreview(postId) {
  const cached = _postTopCommentCache.get(postId);
  const hidden = cached ? '' : 'style="display:none"';
  return `<button class="post-comment-preview" onclick="openComments('${postId}')" ${hidden}><span class="post-comment-preview-label">Top comment</span><span class="post-comment-preview-text">${esc(cached || '')}</span></button>`;
}

async function hydratePostCommentPreviews(posts = []) {
  const needs = posts.map(p => p.id).filter(id => id && !_postTopCommentCache.has(id)).slice(0, 8);
  await Promise.all(needs.map(async postId => {
    try {
      const snap = await db.collection('posts').doc(postId).collection('comments').limit(25).get();
      const comments = snap.docs.map(d => ({ id: d.id, ...d.data() })).filter(c => !c.replyTo && (c.text || '').trim());
      if (!comments.length) {
        _postTopCommentCache.set(postId, '');
        return;
      }
      comments.sort((a, b) => {
        const aScore = getReactionSummary(a.reactions, a.likes || []).total;
        const bScore = getReactionSummary(b.reactions, b.likes || []).total;
        return bScore - aScore || (b.createdAt?.seconds || 0) - (a.createdAt?.seconds || 0);
      });
      _postTopCommentCache.set(postId, clampText(comments[0].text || '', 90));
    } catch (_) {
      _postTopCommentCache.set(postId, '');
    }
  }));
}

// ─── Render Posts ────────────────────────────────
function renderPosts(posts) {
  const el = $('#feed-posts'); if (!el) return;
  if (!posts.length) {
    el.innerHTML = _feedSearchQuery
      ? `<div class="empty-state"><div class="empty-state-icon">🔎</div><h3>No matches found</h3><p>Try another keyword or search with a hashtag like #nwu</p></div>`
      : `<div class="empty-state"><div class="empty-state-icon">📝</div><h3>No posts yet</h3><p>Be the first to share something!</p></div>`;
    return;
  }
  const _videoPlayers = [];
  const postCards = posts.map(post => {
    const liked = (post.likes || []).includes(state.user.uid);
    const myReaction = getUserReaction(post.reactions, post.likes || []);
    const reactionSummary = renderReactionSummary(post.reactions, post.likes || [], 'compact');
    const lc = getReactionSummary(post.reactions, post.likes || []).total, cc = post.commentsCount || 0;
    const canAnonMessage = post.isAnonymous && post.authorId !== state.user.uid;
    const hasCollage = post.imageURLs && post.imageURLs.length > 1 && !post.repostOf;
    const hasVideo = post.videoURL || (post.mediaType === 'video');
    const hasImage = post.imageURL && !hasVideo && !hasCollage;
    const mediaURL = hasVideo ? (post.videoURL || post.imageURL) : post.imageURL;
    const rawAuthorPhoto = post.isAnonymous ? null : (post.authorPhoto || resolvePostAuthorPhoto(post));
    const authorIdentity = post.isAnonymous
      ? { name: getAnonymousLabelForPost(post), photo: null }
      : getResolvedUserIdentity(post.authorId, post.authorName || 'User', rawAuthorPhoto);
    const displayAuthorName = authorIdentity.name || post.authorName || 'User';
    const displayAuthorPhoto = post.isAnonymous ? null : (authorIdentity.photo || rawAuthorPhoto);
    let videoPlayerData = null;
    if (hasVideo && mediaURL) {
      videoPlayerData = createVideoPlayer(mediaURL);
      _videoPlayers.push(videoPlayerData);
    }
    return `
      <div class="post-card" id="post-${post.id}" data-post-id="${post.id}">
        ${post.repostOf ? `<div class="repost-badge">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
          <span>Reposted by ${esc(displayAuthorName)}</span>
        </div>` : ''}
        <div class="post-header">
          ${post.isAnonymous
            ? `<div class="avatar-md anon-avatar" onclick="openAnonPostActions('${post.authorId}', '${post.id}')" style="cursor:pointer">👻</div>`
            : `<div class="feed-author-avatar" data-author-id="${post.authorId}" data-author-name="${esc(displayAuthorName)}" onclick="openProfile('${post.authorId}')" style="cursor:pointer">${avatar(displayAuthorName, displayAuthorPhoto, 'avatar-md')}</div>`}
          <div class="post-header-info">
            <div class="post-author-name" ${post.isAnonymous ? `onclick="openAnonPostActions('${post.authorId}', '${post.id}')" style="cursor:pointer"` : `onclick="openProfile('${post.authorId}')"`}>${post.isAnonymous ? esc(getAnonymousLabelForPost(post)) : esc(displayAuthorName) + verifiedBadge(post.authorId)}</div>
            <div class="post-meta">${post.visibility === 'friends' ? '👫 ' : post.isAnonymous ? '👻 ' : '🌍 '}${post.isAnonymous ? '' : esc(post.authorUni || '')}${post.isAnonymous ? '' : ' · '}${timeAgo(post.createdAt)}</div>
          </div>
          ${!post.isAnonymous && post.authorId === state.user.uid ? `<button class="icon-btn post-more-btn" onclick="showPostOptions('${post.id}')" title="Options" style="margin-left:auto;font-size:18px;color:var(--text-tertiary)">⋯</button>` : ''}
        </div>
        ${post.content ? renderExpandablePostContent(post.content, `feed-${post.id}`, 180) : ''}
        ${renderPostModuleTags(post.moduleTags || [])}
        ${renderPostHashTags(getPostHashTags(post).filter(tag => !(post.moduleTags || []).includes(tag.toUpperCase())))}
        ${renderPostContextTags(post)}
        ${post.poll ? renderInteractivePoll(post.poll, 'post', post.id, '') : ''}
        ${!post.repostOf && hasImage ? `<div class="post-media-wrap"><img src="${mediaURL}" class="post-image" loading="lazy" decoding="async" onclick="viewImage('${mediaURL}')"></div>` : ''}
        ${hasCollage ? renderCollage(post.imageURLs) : ''}
        ${!post.repostOf && hasVideo && videoPlayerData ? videoPlayerData.html : ''}
        ${post.repostOf ? renderQuoteEmbed(post.repostOf, { repostStyle: true }) : ''}
        <div class="post-engagement">
          <div class="post-stats">${renderPostStatsMarkup(post)}</div>
          <div class="post-actions">
            ${canAnonMessage ? `<button class="post-action anon-inline-action" onclick="openAnonPostActions('${post.authorId}', '${post.id}')">👻 Reply</button>` : ''}
            <button class="post-action post-like-action ${liked ? 'liked' : ''} ${myReaction && myReaction !== '❤️' ? 'reacted' : ''}" data-post-id="${post.id}" data-source="feed" onclick="toggleLike('${post.id}')">${renderPostLikeMarkup(post)}</button>
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
        ${renderPostCommentPreview(post.id)}
      </div>`;
  });

  const maxSuggestionsLeft = Math.max(0, 2 - _feedInlineSuggestionSlotsUsed);
  if (maxSuggestionsLeft > 0 && Array.isArray(_usersCache?.data) && _usersCache.data.length) {
    const pool = buildFeedSuggestedPeople(_usersCache.data || [], 15);
    const seedBase = `${posts[0]?.id || 'seed'}:${posts.length}:${_feedInlineSuggestionSlotsUsed}`;
    const featuredEventForFeed = pickFeedSuggestedEvent();
    const shouldInject = !!featuredEventForFeed || (posts.length >= 10 && scoreSeed(seedBase) > 0.42);
    const desired = shouldInject ? Math.min(1, maxSuggestionsLeft) : 0;
    for (let i = 0; i < desired; i++) {
      const picks = pool.slice(0, 15);
      if (!picks.length && !featuredEventForFeed) break;
      const spread = Math.max(4, Math.min(9, posts.length - 2));
      const offset = Math.max(3, Math.round(scoreSeed(`${seedBase}:slot:${i}`) * spread));
      const slot = Math.min(postCards.length, offset + i * 7);
      postCards.splice(slot, 0, renderFeedInlineSuggestionRail(picks));
      picks.forEach(p => _feedInlineSuggestedUserIds.add(p.id));
      _feedInlineSuggestionSlotsUsed += 1;
    }
  }

  el.innerHTML = postCards.join('');
  primeMediaCache((posts || []).flatMap(post => [post.imageURL, post.videoURL]).filter(Boolean), 8).catch(() => {});
  el.classList.remove('feed-ready');
  requestAnimationFrame(() => el.classList.add('feed-ready'));
  hydratePostCommentPreviews(posts).then(() => {
    posts.forEach(post => {
      const card = document.querySelector(`.post-card[data-post-id="${post.id}"] .post-comment-preview`);
      const text = _postTopCommentCache.get(post.id) || '';
      if (!card || !text) return;
      card.innerHTML = `<span class="post-comment-preview-label">Top comment</span><span class="post-comment-preview-text">${esc(text)}</span>`;
      card.style.display = 'flex';
    });
  });

  // Initialize all custom video players after DOM update
  requestAnimationFrame(() => {
    _activePlayerDestroys.forEach(fn => fn());
    _activePlayerDestroys = [];
    _videoPlayers.forEach(p => {
      const result = initPlayer(p.id);
      if (result?.destroy) _activePlayerDestroys.push(result.destroy);
    });
    _pendingQuotePlayers.forEach(p => {
      const result = initPlayer(p.id);
      if (result?.destroy) _activePlayerDestroys.push(result.destroy);
    });
    _pendingQuotePlayers.length = 0;
    setupFeedVideoAutoplay();
    bindPostReactionLongPress(el);

    // Track which posts the user actually sees (scrolls into view)
    if (_feedSeenObserver) _feedSeenObserver.disconnect();
    _feedSeenObserver = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          const pid = e.target.dataset.postId;
          if (pid) markPostSeen(pid);
          _feedSeenObserver.unobserve(e.target);
        }
      });
    }, { threshold: 0.5 });
    el.querySelectorAll('.post-card[data-post-id]').forEach(card => _feedSeenObserver.observe(card));
  });
}

function openLiveStreamFromFeed(streamId = '', isOwn = false) {
  if (!streamId) return;
  openVideoHub('live');
  setTimeout(() => {
    if (isOwn) openHostLiveView(streamId);
    else joinLiveStream(streamId);
  }, 90);
}

function loadFeedLiveSpotlight() {
  const host = document.getElementById('feed-live-spotlight');
  if (!host) return;
  host.innerHTML = '<div class="feed-live-card feed-live-card-loading"><span class="inline-spinner"></span><span>Checking who is live…</span></div>';
  host.style.display = 'block';
  db.collection('liveStreams')
    .where('status', 'in', ['live', 'starting'])
    .limit(6)
    .get()
    .then(snap => {
      const streams = snap.docs
        .map(d => ({ id: d.id, ...d.data() }))
        .sort((a, b) => (b.currentViewerCount || 0) - (a.currentViewerCount || 0) || (b.updatedAt?.seconds || 0) - (a.updatedAt?.seconds || 0));
      if (!streams.length) {
        host.style.display = 'none';
        host.innerHTML = '';
        return;
      }
      const picks = streams.slice(0, 3);
      host.innerHTML = `
        <div class="feed-live-head">
          <div class="feed-live-title">🔴 Live now</div>
          <button class="btn-outline btn-sm" onclick="openVideoHub('live')">Open live</button>
        </div>
        <div class="feed-live-row">
          ${picks.map(stream => {
            const isOwn = stream.hostUid === state.user?.uid;
            const viewers = Number(stream.currentViewerCount || 0);
            const cover = stream.thumbnailUrl || stream.hostPhotoURL || '';
            return `
              <button type="button" class="feed-live-card" onclick="openLiveStreamFromFeed('${stream.id}', ${isOwn ? 'true' : 'false'})">
                <div class="feed-live-media">
                  ${cover ? `<img src="${cover}" alt="">` : `<div class="feed-live-fallback">${avatar(stream.hostName || 'Live', stream.hostPhotoURL || null, 'avatar-lg')}</div>`}
                  <span class="feed-live-badge">● LIVE</span>
                </div>
                <div class="feed-live-copy">
                  <div class="feed-live-card-title">${esc(stream.title || 'Untitled stream')}</div>
                  <div class="feed-live-card-meta">${esc(stream.hostName || 'User')} · ${viewers} watching</div>
                </div>
              </button>`;
          }).join('')}
        </div>
      `;
    })
    .catch(err => {
      console.warn('feed live spotlight', err);
      host.style.display = 'none';
      host.innerHTML = '';
    });
}

// ═══════════════════════════════════════════════════
//  VIDEO HUB — LIVE | CLIPS  (replaces old Reels viewer)
// ═══════════════════════════════════════════════════
let _reelsActive = false;
let _reelVideos = [];
let _videoHubTab = 'clips'; // 'live' | 'clips'
let _liveStreams = [];
let _liveUnsub = null;
let _hostStream = null;      // MediaStream when hosting
let _hostStreamId = null;     // Firestore doc id
let _hostFacingMode = 'user'; // 'user' (front) or 'environment' (back)
let _viewerPeerConns = {};    // { viewerUid: RTCPeerConnection }
let _hostPeerConn = null;     // viewer-side peer connection
let _liveCommentsUnsub = null;
let _liveViewerHeartbeat = null;
let _liveViewerPresenceId = null;
let _liveViewerCountTimer = null;
let _liveViewingStreamId = null;
let _liveViewerMirrored = false;

// ─── Open Video Hub ──────────────────────────────
function openVideoHub(tab) {
  stopAllVideos();
  if (tab) _videoHubTab = tab;
  const existing = document.getElementById('video-hub');
  if (existing) existing.remove();

  const hub = document.createElement('div');
  hub.id = 'video-hub';
  hub.className = 'video-hub';
  hub.innerHTML = `
    <div class="vh-header">
      <button class="vh-close" onclick="closeVideoHub()">&times;</button>
      <div class="vh-tabs">
        <button class="vh-tab ${_videoHubTab === 'live' ? 'active' : ''}" onclick="switchVideoHubTab('live')">LIVE</button>
        <button class="vh-tab ${_videoHubTab === 'clips' ? 'active' : ''}" onclick="switchVideoHubTab('clips')">CLIPS</button>
      </div>
      <div style="width:40px"></div>
    </div>
    <div class="vh-body" id="vh-body">
      <div style="padding:40px;text-align:center"><span class="inline-spinner" style="width:28px;height:28px;color:var(--accent)"></span></div>
    </div>
  `;
  document.body.appendChild(hub);

  if (_videoHubTab === 'clips') loadClipsTab();
  else loadLiveTab();
}

function closeVideoHub() {
  // If host is live, end the stream first
  if (_hostStreamId) {
    db.collection('liveStreams').doc(_hostStreamId).update({
      status: 'ended',
      endedAt: FieldVal.serverTimestamp(),
      endedReason: 'host_ended',
      updatedAt: FieldVal.serverTimestamp()
    }).catch(() => {});
    Object.values(_viewerPeerConns).forEach(pc => pc.close());
    _viewerPeerConns = {};
    if (_hostStream) { _hostStream.getTracks().forEach(t => t.stop()); _hostStream = null; }
    _hostStreamId = null;
    toast('Stream ended');
  }
  // If viewer, close peer connection
  if (_hostPeerConn) { _hostPeerConn.close(); _hostPeerConn = null; }
  if (_liveViewerPresenceId) {
    // Best-effort mark viewer as inactive
    db.collection('liveStreams').doc(_liveViewingStreamId || '_').collection('viewers').doc(_liveViewerPresenceId).update({ isActive: false }).catch(() => {});
    _liveViewerPresenceId = null;
    _liveViewingStreamId = null;
  }
  stopLiveListeners();
  const el = document.getElementById('video-hub');
  if (el) {
    el.querySelectorAll('video').forEach(v => { v.pause(); v.srcObject = null; });
    el.remove();
  }
  _reelsActive = false;
  if (_reelsObserver) { _reelsObserver.disconnect(); _reelsObserver = null; }
}

function switchVideoHubTab(tab) {
  _videoHubTab = tab;
  document.querySelectorAll('.vh-tab').forEach(t => t.classList.toggle('active', t.textContent.trim() === tab.toUpperCase()));
  if (tab === 'clips') loadClipsTab();
  else loadLiveTab();
}

// ─── CLIPS TAB (existing reels) ──────────────────
function loadClipsTab() {
  const body = document.getElementById('vh-body');
  if (!body) return;
  body.innerHTML = '<div style="padding:40px;text-align:center"><span class="inline-spinner" style="width:28px;height:28px;color:var(--accent)"></span></div>';

  db.collection('posts').orderBy('createdAt', 'desc').limit(100).get().then(snap => {
    const allPosts = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    _reelVideos = allPosts.filter(p => p.videoURL || p.mediaType === 'video');
    _reelVideos = _reelVideos.map(p => {
      const likes = (p.likes || []).length;
      const comments = p.commentsCount || 0;
      return { ...p, _score: (likes + comments) * 0.3 + Math.random() * 10 };
    }).sort((a, b) => b._score - a._score);

    if (!_reelVideos.length) {
      body.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🎬</div><h3>No clips yet</h3><p>Post a video to see it here!</p></div>';
      return;
    }
    _reelsActive = true;
    renderClipsContent(body);
  }).catch(e => { console.error(e); body.innerHTML = '<div class="empty-state"><div class="empty-state-icon">⚠️</div><h3>Could not load clips</h3></div>'; });
}

function renderClipsContent(body) {
  body.innerHTML = `
    <div class="reels-scroll" id="reels-scroll">
      ${_reelVideos.map((p, i) => {
        const url = p.videoURL || p.imageURL;
        const liked = (p.likes || []).includes(state.user.uid);
        const myReaction = getUserReaction(p.reactions, p.likes || []);
        const lc = getReactionSummary(p.reactions, p.likes || []).total;
        const cc = p.commentsCount || 0;
        const reelIdentity = p.isAnonymous
          ? { name: getAnonymousLabelForPost(p), photo: null }
          : getResolvedUserIdentity(p.authorId, p.authorName || 'User', p.authorPhoto || null);
        return `
        <div class="reel-slide" data-idx="${i}" data-post-id="${p.id}">
          <video class="reel-video" src="${url}" loop playsinline preload="metadata" muted></video>
          <div class="reel-overlay-bottom">
            <div class="reel-author" ${p.isAnonymous ? '' : `onclick="closeVideoHub();openProfile('${p.authorId}')"`}>
              ${p.isAnonymous ? `<div class="avatar-sm anon-avatar">👻</div>` : avatar(reelIdentity.name, reelIdentity.photo, 'avatar-sm')}
              <span class="reel-author-name">${p.isAnonymous ? esc(getAnonymousLabelForPost(p)) : esc(reelIdentity.name)}</span>
            </div>
            ${p.content ? `<p class="reel-caption">${esc(p.content)}</p>` : ''}
          </div>
          <div class="reel-actions">
            <button class="reel-act-btn reel-like-btn ${liked ? 'liked' : ''}" onclick="reelLike('${p.id}', this)">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="${liked ? '#ff4757' : 'none'}" stroke="${liked ? '#ff4757' : '#fff'}" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
              <span>${lc || ''}</span>
            </button>
            <button class="reel-act-btn reel-react-btn ${myReaction && myReaction !== '❤️' ? 'reacted' : ''}" onclick="event.stopPropagation();openPostReactionPicker('${p.id}','reel')">
              <span>${myReaction && myReaction !== '❤️' ? myReaction : 'React'}</span>
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

  const scrollEl = document.getElementById('reels-scroll');
  if (_reelsObserver) { _reelsObserver.disconnect(); _reelsObserver = null; }
  _reelsObserver = new IntersectionObserver(entries => {
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
  scrollEl.querySelectorAll('.reel-slide').forEach(slide => _reelsObserver.observe(slide));

  requestAnimationFrame(() => {
    const firstVid = scrollEl.querySelector('.reel-slide:first-child .reel-video');
    if (firstVid) { firstVid.muted = false; firstVid.play().catch(() => {}); }
  });
}

// Keep old function name for backwards compat
function openReelsViewer() { openVideoHub('clips'); }
function closeReelsViewer() { closeVideoHub(); }

function toggleReelPlay(overlay) {
  const panel = document.getElementById('reel-comments-panel');
  if (panel) {
    closeReelComments();
    return;
  }
  const vid = overlay.parentElement.querySelector('.reel-video');
  if (!vid) return;
  if (vid.paused) {
    vid.muted = false;
    vid.play().catch(() => {});
    overlay.classList.remove('paused');
  } else {
    vid.pause();
    overlay.classList.add('paused');
  }
}

function toggleReelsSound() {
  _reelsSoundEnabled = !_reelsSoundEnabled;
  document.querySelectorAll('.reel-video').forEach(video => { video.muted = !_reelsSoundEnabled; });
}

// ─── LIVE TAB ────────────────────────────────────
function loadLiveTab() {
  const body = document.getElementById('vh-body');
  if (!body) return;
  body.innerHTML = `
    <div class="live-tab-content">
      <button class="go-live-btn" onclick="openGoLiveModal()">
        <div class="go-live-icon">📡</div>
        <span>Go Live</span>
      </button>
      <div id="live-streams-list">
        <div style="padding:40px;text-align:center"><span class="inline-spinner" style="width:28px;height:28px;color:var(--accent)"></span></div>
      </div>
    </div>
  `;
  subscribeLiveStreams();
}

function subscribeLiveStreams() {
  stopLiveListeners();
  _liveUnsub = db.collection('liveStreams')
    .where('status', 'in', ['live', 'starting'])
    .orderBy('startedAt', 'desc')
    .onSnapshot(snap => {
      _liveStreams = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      renderLiveStreamsList();
    }, err => {
      console.error('Live streams listener error:', err);
      const el = document.getElementById('live-streams-list');
      if (el) el.innerHTML = '<div class="empty-state"><div class="empty-state-icon">⚠️</div><h3>Could not load live streams</h3></div>';
    });
}

function stopLiveListeners() {
  if (_liveUnsub) { _liveUnsub(); _liveUnsub = null; }
  if (_liveCommentsUnsub) { _liveCommentsUnsub(); _liveCommentsUnsub = null; }
  if (_liveReactionsUnsub) { _liveReactionsUnsub(); _liveReactionsUnsub = null; }
  if (_liveViewerHeartbeat) { clearInterval(_liveViewerHeartbeat); _liveViewerHeartbeat = null; }
  if (_liveViewerCountTimer) { clearInterval(_liveViewerCountTimer); _liveViewerCountTimer = null; }
}

function renderLiveStreamsList() {
  const el = document.getElementById('live-streams-list');
  if (!el) return;
  if (!_liveStreams.length) {
    el.innerHTML = '<div class="empty-state" style="padding:30px;text-align:center"><div class="empty-state-icon">📡</div><h3>No one is live right now</h3><p>Tap "Go Live" to start streaming!</p></div>';
    return;
  }
  el.innerHTML = _liveStreams.map(s => {
    const viewers = s.currentViewerCount || 0;
    const isOwn = s.hostUid === state.user.uid;
    return `
      <div class="live-stream-card" onclick="${isOwn ? `openHostLiveView('${s.id}')` : `joinLiveStream('${s.id}')`}">
        <div class="live-card-thumb">
          ${s.thumbnailUrl ? `<img src="${s.thumbnailUrl}" alt="">` : `<div class="live-card-placeholder">${avatar(s.hostName, s.hostPhotoURL, 'avatar-lg')}</div>`}
          <div class="live-badge-overlay"><span class="live-badge">● LIVE</span><span class="live-viewers">${viewers} watching</span></div>
        </div>
        <div class="live-card-info">
          <div class="live-card-row">
            ${avatar(s.hostName, s.hostPhotoURL, 'avatar-sm')}
            <div>
              <div class="live-card-title">${esc(s.title || 'Untitled stream')}</div>
              <div class="live-card-host">${esc(s.hostName || 'User')}${s.category ? ` · ${esc(s.category)}` : ''}</div>
            </div>
          </div>
        </div>
      </div>`;
  }).join('');
}

// ─── GO LIVE MODAL ───────────────────────────────
function openGoLiveModal() {
  // Clean up any prior camera preview
  if (_hostStream) { _hostStream.getTracks().forEach(t => t.stop()); _hostStream = null; }

  const modalBg = document.getElementById('modal-bg');
  const modalInner = document.getElementById('modal-inner');
  if (!modalBg || !modalInner) return;
  modalInner.innerHTML = `
    <div class="go-live-modal">
      <h2>Go Live</h2>
      <div class="form-group">
        <label>Title</label>
        <input type="text" id="live-title" placeholder="What's your stream about?" maxlength="100">
      </div>
      <div class="form-group">
        <label>Category</label>
        <select id="live-category">
          <option value="general">General</option>
          <option value="social">Social</option>
          <option value="sports">Sports</option>
          <option value="campus_event">Campus Event</option>
          <option value="marketplace">Marketplace</option>
          <option value="study">Study</option>
          <option value="music">Music</option>
        </select>
      </div>
      <div class="form-group">
        <label>Visibility</label>
        <select id="live-visibility">
          <option value="public">Public</option>
          <option value="campus_only">Campus Only</option>
        </select>
      </div>
      <div id="live-camera-preview" class="live-camera-preview">
        <video id="live-preview-video" autoplay muted playsinline></video>
        <div class="live-preview-placeholder" id="live-preview-placeholder">Camera preview will appear here</div>
      </div>
      <div class="live-modal-actions">
        <button class="btn-secondary" onclick="closeModal()">Cancel</button>
        <button class="btn-primary go-live-start-btn" id="go-live-start-btn" onclick="startLiveStream()">🔴 Go Live</button>
      </div>
    </div>
  `;
  modalBg.style.display = 'flex';

  // Start camera preview
  if (!navigator.mediaDevices?.getUserMedia) {
    toast('Camera is not supported on this device');
    return;
  }
  navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: true })
    .then(stream => {
      _hostStream = stream;
      const prevVideo = document.getElementById('live-preview-video');
      if (prevVideo) {
        prevVideo.srcObject = stream;
        prevVideo.play().catch(() => {});
        const placeholder = document.getElementById('live-preview-placeholder');
        if (placeholder) placeholder.style.display = 'none';
      }
    })
    .catch(err => {
      console.error('Camera access denied:', err);
      if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
        toast(isNativeApp()
          ? 'Camera blocked — open Settings → Apps → Unino → Permissions and enable Camera & Microphone'
          : 'Camera access denied. Allow camera in your browser settings and reload');
      } else if (err?.name === 'NotFoundError' || err?.name === 'DevicesNotFoundError') {
        toast('No camera found on this device');
      } else if (err?.name === 'NotReadableError' || err?.name === 'TrackStartError') {
        toast('Camera is in use by another app. Close it and try again');
      } else {
        toast('Could not access camera');
      }
    });
}

// ─── START LIVE STREAM ───────────────────────────
async function startLiveStream() {
  if (!_hostStream) { toast('Camera not ready'); return; }
  const titleEl = document.getElementById('live-title');
  const catEl = document.getElementById('live-category');
  const visEl = document.getElementById('live-visibility');
  const btn = document.getElementById('go-live-start-btn');
  if (btn) btn.disabled = true;

  const title = titleEl?.value.trim() || 'Untitled stream';
  const category = catEl?.value || 'general';
  const visibility = visEl?.value || 'public';
  const uid = state.user.uid;
  const p = state.profile;

  try {
    // Check if already live
    const existingSnap = await db.collection('liveStreams').where('hostUid', '==', uid).where('status', 'in', ['live', 'starting']).get();
    if (!existingSnap.empty) { toast('You already have an active stream'); if (btn) btn.disabled = false; return; }

    const streamDoc = await db.collection('liveStreams').add({
      hostUid: uid,
      hostName: p.displayName || 'User',
      hostPhotoURL: p.photoURL || null,
      title,
      category,
      visibility,
      status: 'live',
      campus: p.university || '',
      startedAt: FieldVal.serverTimestamp(),
      createdAt: FieldVal.serverTimestamp(),
      updatedAt: FieldVal.serverTimestamp(),
      currentViewerCount: 0,
      peakViewerCount: 0,
      totalViews: 0,
      commentCount: 0,
      reactionCount: 0,
      reportCount: 0,
      tags: [],
      isFeatured: false,
      thumbnailUrl: null,
      endedReason: null,
      endedAt: null
    });

    _hostStreamId = streamDoc.id;
    closeModal();
    openHostLiveView(_hostStreamId);
    toast('You are live! 🔴');
  } catch (e) {
    console.error('Failed to start live:', e);
    toast('Failed to go live');
    if (btn) btn.disabled = false;
  }
}

// ─── HOST LIVE VIEW ──────────────────────────────
function openHostLiveView(streamId) {
  const hub = document.getElementById('video-hub');
  if (!hub) { openVideoHub('live'); return; }

  const body = document.getElementById('vh-body');
  if (!body) return;

  body.innerHTML = `
    <div class="live-viewer-screen">
      <video id="host-live-video" class="live-video host-cam" autoplay muted playsinline webkit-playsinline></video>
      <div class="live-top-bar">
        <div class="live-info-pill">
          <span class="live-dot">●</span> LIVE
          <span class="live-viewer-count" id="live-viewer-count">0</span>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="#fff" stroke="none"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/></svg>
        </div>
        <button class="end-live-btn" onclick="endLiveStream('${streamId}')">End</button>
      </div>
      <div class="live-comments-overlay" id="live-comments-overlay"></div>
      <div class="live-bottom-bar">
        <div class="live-comment-input-wrap">
          <input type="text" id="live-comment-input" placeholder="Say something..." maxlength="200" autocomplete="off">
          <button class="live-send-btn" onclick="sendLiveComment('${streamId}')">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          </button>
        </div>
        <button class="live-react-btn" onclick="sendLiveReaction('${streamId}')">❤️</button>
        <button class="live-flip-btn" onclick="switchLiveCamera('${streamId}')" title="Flip camera">🔄</button>
      </div>
    </div>
  `;

  // Show host camera in the video element
  const hostVideo = document.getElementById('host-live-video');
  if (hostVideo && _hostStream) {
    hostVideo.srcObject = _hostStream;
    hostVideo.classList.toggle('host-cam', _hostFacingMode === 'user');
  }

  // Listen for viewers wanting to connect (WebRTC signaling)
  listenForViewerOffers(streamId);
  // Listen for live comments
  subscribeLiveComments(streamId);
  // Listen for reactions from all clients
  subscribeLiveReactions(streamId);
  // Update viewer count periodically
  startViewerCountUpdater(streamId, true);

  const input = document.getElementById('live-comment-input');
  if (input) {
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); sendLiveComment(streamId); }
    });
  }
}

// ─── SWITCH CAMERA (front/back) ──────────────────
async function switchLiveCamera() {
  const previousFacingMode = _hostFacingMode;
  const currentVideoTrack = _hostStream?.getVideoTracks?.()[0] || null;
  const currentAudioTrack = _hostStream?.getAudioTracks?.()[0] || null;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices().catch(() => []);
    const videoInputs = devices.filter(device => device.kind === 'videoinput');
    let targetConstraints;
    if (videoInputs.length > 1) {
      const currentId = currentVideoTrack?.getSettings?.().deviceId || '';
      const currentIndex = Math.max(0, videoInputs.findIndex(device => device.deviceId === currentId));
      const nextDevice = videoInputs[(currentIndex + 1) % videoInputs.length];
      _hostFacingMode = _hostFacingMode === 'user' ? 'environment' : 'user';
      targetConstraints = { deviceId: { exact: nextDevice.deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } };
    } else {
      _hostFacingMode = _hostFacingMode === 'user' ? 'environment' : 'user';
      targetConstraints = { facingMode: { ideal: _hostFacingMode }, width: { ideal: 1280 }, height: { ideal: 720 } };
    }

    const videoOnlyStream = await navigator.mediaDevices.getUserMedia({ video: targetConstraints, audio: false }).catch(async () => {
      return navigator.mediaDevices.getUserMedia({ video: { facingMode: _hostFacingMode }, audio: false });
    });
    const newVideoTrack = videoOnlyStream.getVideoTracks()[0];
    if (!newVideoTrack) throw new Error('missing-video-track');

    await Promise.all(Object.values(_viewerPeerConns).map(async pc => {
      const videoSender = pc.getSenders().find(sender => sender.track?.kind === 'video');
      if (videoSender) await videoSender.replaceTrack(newVideoTrack).catch(() => {});
    }));

    const oldVideoTracks = _hostStream?.getVideoTracks?.() || [];
    _hostStream = new MediaStream([newVideoTrack, ...(currentAudioTrack ? [currentAudioTrack] : [])]);
    const hostVideo = document.getElementById('host-live-video');
    if (hostVideo) {
      hostVideo.srcObject = _hostStream;
      hostVideo.classList.toggle('host-cam', _hostFacingMode === 'user');
      await hostVideo.play?.().catch(() => {});
    }
    oldVideoTracks.forEach(track => { try { track.stop(); } catch (_) {} });
  } catch (e) {
    console.error('Camera switch error:', e);
    toast('Could not switch camera');
    _hostFacingMode = previousFacingMode;
  }
}

function toggleLiveViewerMirror() {
  _liveViewerMirrored = !_liveViewerMirrored;
  const video = document.getElementById('viewer-live-video');
  if (video) video.classList.toggle('viewer-mirrored', _liveViewerMirrored);
}

// ─── VIEWER JOIN STREAM ──────────────────────────
async function joinLiveStream(streamId) {
  const hub = document.getElementById('video-hub');
  if (!hub) { openVideoHub('live'); return; }

  const body = document.getElementById('vh-body');
  if (!body) return;

  body.innerHTML = `
    <div class="live-viewer-screen">
      <video id="viewer-live-video" class="live-video" autoplay playsinline webkit-playsinline></video>
      <div class="live-connecting-overlay" id="live-connecting">
        <span class="inline-spinner" style="width:36px;height:36px;color:#fff"></span>
        <p style="color:#fff;margin-top:12px">Connecting to stream...</p>
      </div>
      <div class="live-top-bar">
        <div class="live-info-pill">
          <span class="live-dot">●</span> LIVE
          <span class="live-viewer-count" id="live-viewer-count">0</span>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="#fff" stroke="none"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/></svg>
        </div>
        <button class="leave-live-btn" onclick="leaveLiveStream('${streamId}')">✕</button>
      </div>
      <div class="live-comments-overlay" id="live-comments-overlay"></div>
      <div class="live-bottom-bar">
        <div class="live-comment-input-wrap">
          <input type="text" id="live-comment-input" placeholder="Say something..." maxlength="200" autocomplete="off">
          <button class="live-send-btn" onclick="sendLiveComment('${streamId}')">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          </button>
        </div>
        <button class="live-react-btn" onclick="sendLiveReaction('${streamId}')">❤️</button>
        <button class="live-view-mirror-btn" onclick="toggleLiveViewerMirror()" title="Mirror only my view">⇋</button>
      </div>
    </div>
  `;

  _liveViewerMirrored = false;

  // Register viewer presence
  const uid = state.user.uid;
  try {
    const presDoc = await db.collection('liveStreams').doc(streamId).collection('viewers').doc(uid).set({
      uid,
      displayName: state.profile.displayName || '',
      joinedAt: FieldVal.serverTimestamp(),
      lastSeenAt: FieldVal.serverTimestamp(),
      isActive: true
    });
    _liveViewerPresenceId = uid;
  } catch (e) { console.error('Viewer presence error:', e); }

  // Heartbeat every 30s
  _liveViewingStreamId = streamId;
  if (_liveViewerHeartbeat) { clearInterval(_liveViewerHeartbeat); _liveViewerHeartbeat = null; }
  _liveViewerHeartbeat = setInterval(() => {
    db.collection('liveStreams').doc(streamId).collection('viewers').doc(uid).update({
      lastSeenAt: FieldVal.serverTimestamp(),
      isActive: true
    }).catch(() => {});
  }, 30000);

  // Increment total views
  db.collection('liveStreams').doc(streamId).update({ totalViews: FieldVal.increment(1) }).catch(() => {});

  // WebRTC: create offer to host
  connectToHost(streamId);
  subscribeLiveComments(streamId);
  subscribeLiveReactions(streamId);
  startViewerCountUpdater(streamId, false);

  const input = document.getElementById('live-comment-input');
  if (input) {
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); sendLiveComment(streamId); }
    });
  }
}

// ─── WebRTC SIGNALING ────────────────────────────
const ICE_SERVERS = [{ urls: 'stun:stun.l.google.com:19302' }, { urls: 'stun:stun1.l.google.com:19302' }];

// HOST: listen for viewer offers
function listenForViewerOffers(streamId) {
  db.collection('liveStreams').doc(streamId).collection('signals')
    .where('type', '==', 'offer')
    .onSnapshot(snap => {
      snap.docChanges().forEach(async change => {
        if (change.type !== 'added') return;
        const data = change.doc.data();
        const viewerUid = data.senderUid;
        if (_viewerPeerConns[viewerUid]) return; // already handling

        const pc = new RTCPeerConnection({ iceServers: ICE_SERVERS });
        _viewerPeerConns[viewerUid] = pc;

        // Add host's stream tracks
        if (_hostStream) {
          _hostStream.getTracks().forEach(track => pc.addTrack(track, _hostStream));
        }

        // Send ICE candidates to viewer
        pc.onicecandidate = e => {
          if (e.candidate) {
            db.collection('liveStreams').doc(streamId).collection('signals').add({
              type: 'host-ice',
              targetUid: viewerUid,
              candidate: e.candidate.toJSON(),
              createdAt: FieldVal.serverTimestamp()
            }).catch(() => {});
          }
        };

        try {
          await pc.setRemoteDescription(new RTCSessionDescription(data.sdp));
          const answer = await pc.createAnswer();
          await pc.setLocalDescription(answer);

          await db.collection('liveStreams').doc(streamId).collection('signals').add({
            type: 'answer',
            targetUid: viewerUid,
            senderUid: state.user.uid,
            sdp: { type: answer.type, sdp: answer.sdp },
            createdAt: FieldVal.serverTimestamp()
          });
        } catch (e) { console.error('Host signaling error:', e); }

        // Listen for viewer ICE candidates
        db.collection('liveStreams').doc(streamId).collection('signals')
          .where('type', '==', 'viewer-ice')
          .where('targetUid', '==', 'host')
          .onSnapshot(iceSnap => {
            iceSnap.docChanges().forEach(async ic => {
              if (ic.type !== 'added') return;
              const iceData = ic.doc.data();
              if (iceData.senderUid !== viewerUid) return;
              try {
                await pc.addIceCandidate(new RTCIceCandidate(iceData.candidate));
              } catch (e) { /* ignore late candidates */ }
            });
          });
      });
    });
}

// VIEWER: send offer to host
async function connectToHost(streamId) {
  const pc = new RTCPeerConnection({ iceServers: ICE_SERVERS });
  _hostPeerConn = pc;
  const sessionStartMs = Date.now();
  const uid = state.user.uid;

  pc.ontrack = e => {
    const video = document.getElementById('viewer-live-video');
    if (video && e.streams[0]) {
      video.srcObject = e.streams[0];
      video.muted = false;
      video.play().catch(() => {});
      const overlay = document.getElementById('live-connecting');
      if (overlay) overlay.style.display = 'none';
    }
  };

  pc.onicecandidate = e => {
    if (e.candidate) {
      db.collection('liveStreams').doc(streamId).collection('signals').add({
        type: 'viewer-ice',
        senderUid: uid,
        targetUid: 'host',
        candidate: e.candidate.toJSON(),
        createdAt: FieldVal.serverTimestamp()
      }).catch(() => {});
    }
  };

  pc.onconnectionstatechange = () => {
    const s = pc.connectionState;
    console.log('[WebRTC viewer] connectionState:', s);
    if (s === 'failed') {
      // Attempt ICE restart once
      if (!pc._iceRestarted) {
        pc._iceRestarted = true;
        pc.restartIce();
        pc.createOffer({ iceRestart: true }).then(offer => {
          pc.setLocalDescription(offer).then(() => {
            db.collection('liveStreams').doc(streamId).collection('signals').add({
              type: 'offer',
              senderUid: uid,
              sdp: { type: offer.type, sdp: offer.sdp },
              createdAt: FieldVal.serverTimestamp()
            }).catch(() => {});
          });
        }).catch(() => {});
        return;
      }
      const overlay = document.getElementById('live-connecting');
      if (overlay) { overlay.style.display = 'flex'; overlay.innerHTML = '<p style="color:#fff">Stream disconnected</p>'; }
    } else if (s === 'disconnected') {
      // Temporary — WebRTC may recover on its own
      console.log('[WebRTC viewer] disconnected, waiting for recovery...');
    } else if (s === 'connected') {
      const overlay = document.getElementById('live-connecting');
      if (overlay) overlay.style.display = 'none';
    }
  };

  // Add a transceiver so we can receive video+audio
  pc.addTransceiver('video', { direction: 'recvonly' });
  pc.addTransceiver('audio', { direction: 'recvonly' });

  try {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    await db.collection('liveStreams').doc(streamId).collection('signals').add({
      type: 'offer',
      senderUid: uid,
      sdp: { type: offer.type, sdp: offer.sdp },
      createdAt: FieldVal.serverTimestamp()
    });


    // Listen for host answer — skip docs older than this session (stale rejoin signals)
    db.collection('liveStreams').doc(streamId).collection('signals')
      .where('type', '==', 'answer')
      .where('targetUid', '==', uid)
      .onSnapshot(snap => {
        snap.docChanges().forEach(async change => {
          if (change.type !== 'added') return;
          const ansData = change.doc.data();
          const createdMs = ansData.createdAt?.toMillis?.() || 0;
          if (createdMs && createdMs < sessionStartMs - 5000) return; // skip stale from prior session
          try {
            await pc.setRemoteDescription(new RTCSessionDescription(ansData.sdp));
          } catch (e) { console.error('Viewer answer error:', e); }
        });
      });

    // Listen for host ICE candidates — skip stale docs from prior sessions
    db.collection('liveStreams').doc(streamId).collection('signals')
      .where('type', '==', 'host-ice')
      .where('targetUid', '==', uid)
      .onSnapshot(snap => {
        snap.docChanges().forEach(async change => {
          if (change.type !== 'added') return;
          const iceDoc = change.doc.data();
          const createdMs = iceDoc.createdAt?.toMillis?.() || 0;
          if (createdMs && createdMs < sessionStartMs - 5000) return; // skip stale
          try {
            await pc.addIceCandidate(new RTCIceCandidate(iceDoc.candidate));
          } catch (e) { /* ignore */ }
        });
      });
  } catch (e) {
    console.error('Viewer offer error:', e);
    toast('Could not connect to stream');
  }

  // Fallback timeout: if no video after 15s, show message
  setTimeout(() => {
    const video = document.getElementById('viewer-live-video');
    if (video && !video.srcObject) {
      const overlay = document.getElementById('live-connecting');
      if (overlay) overlay.innerHTML = '<p style="color:#fff">Could not connect to stream. The host may have ended.</p>';
    }
  }, 15000);
}

// ─── END LIVE STREAM (host) ──────────────────────
async function endLiveStream(streamId) {
  try {
    await db.collection('liveStreams').doc(streamId).update({
      status: 'ended',
      endedAt: FieldVal.serverTimestamp(),
      endedReason: 'host_ended',
      updatedAt: FieldVal.serverTimestamp()
    });

    // Clean up peer connections
    Object.values(_viewerPeerConns).forEach(pc => pc.close());
    _viewerPeerConns = {};

    // Stop host camera
    if (_hostStream) { _hostStream.getTracks().forEach(t => t.stop()); _hostStream = null; }
    _hostStreamId = null;

    stopLiveListeners();
    toast('Stream ended');
    loadLiveTab();
  } catch (e) { console.error(e); toast('Failed to end stream'); }
}

// ─── LEAVE LIVE STREAM (viewer) ──────────────────
async function leaveLiveStream(streamId) {
  if (_hostPeerConn) { _hostPeerConn.close(); _hostPeerConn = null; }
  if (_liveViewerPresenceId) {
    db.collection('liveStreams').doc(streamId).collection('viewers').doc(_liveViewerPresenceId).update({ isActive: false }).catch(() => {});
    _liveViewerPresenceId = null;
  }
  _liveViewingStreamId = null;
  _liveViewerMirrored = false;
  stopLiveListeners();
  loadLiveTab();
}

// ─── LIVE COMMENTS ───────────────────────────────
function subscribeLiveComments(streamId) {
  if (_liveCommentsUnsub) _liveCommentsUnsub();
  _liveCommentsUnsub = db.collection('liveStreams').doc(streamId).collection('comments')
    .orderBy('createdAt', 'desc')
    .limit(13)
    .onSnapshot(snap => {
      const comments = snap.docs.map(d => ({ id: d.id, ...d.data() })).reverse();
      const overlay = document.getElementById('live-comments-overlay');
      if (!overlay) return;
      // Keep only comment divs, preserve floating hearts
      const hearts = overlay.querySelectorAll('.live-floating-heart');
      overlay.innerHTML = comments.map(c => `
        <div class="live-comment ${c.uid === state.user.uid ? 'own' : ''}">
          <strong>${esc(c.displayName || 'User')}</strong> ${esc(c.text || '')}
        </div>
      `).join('');
      hearts.forEach(h => overlay.appendChild(h));
      overlay.scrollTop = overlay.scrollHeight;
    });
}

async function sendLiveComment(streamId) {
  const input = document.getElementById('live-comment-input');
  const text = input?.value.trim();
  if (!text) return;
  input.value = '';
  try {
    await db.collection('liveStreams').doc(streamId).collection('comments').add({
      uid: state.user.uid,
      displayName: state.profile.displayName || 'User',
      photoURL: state.profile.photoURL || null,
      text,
      createdAt: FieldVal.serverTimestamp(),
      isDeleted: false
    });
    await db.collection('liveStreams').doc(streamId).update({ commentCount: FieldVal.increment(1) });
  } catch (e) { console.error(e); }
}

// ─── LIVE REACTIONS ──────────────────────────────
let _liveReactionsUnsub = null;

function sendLiveReaction(streamId) {
  // Instant local heart for feedback
  const overlay = document.getElementById('live-comments-overlay');
  if (overlay) {
    const heart = document.createElement('div');
    heart.className = 'live-floating-heart';
    heart.textContent = '❤️';
    heart.style.left = (65 + Math.random() * 25) + '%';
    overlay.appendChild(heart);
    setTimeout(() => heart.remove(), 2000);
  }
  // Write reaction event to Firestore so all clients see it
  db.collection('liveStreams').doc(streamId).collection('reactions').add({
    uid: state.user.uid,
    type: '❤️',
    createdAt: FieldVal.serverTimestamp()
  }).catch(() => {});
  db.collection('liveStreams').doc(streamId).update({ reactionCount: FieldVal.increment(1) }).catch(() => {});
}

function subscribeLiveReactions(streamId) {
  if (_liveReactionsUnsub) _liveReactionsUnsub();
  _liveReactionsUnsub = db.collection('liveStreams').doc(streamId).collection('reactions')
    .orderBy('createdAt', 'desc')
    .limit(1)
    .onSnapshot(snap => {
      snap.docChanges().forEach(change => {
        if (change.type !== 'added') return;
        const data = change.doc.data();
        // Skip own reactions (already shown locally)
        if (data.uid === state.user?.uid) return;
        const overlay = document.getElementById('live-comments-overlay');
        if (!overlay) return;
        const heart = document.createElement('div');
        heart.className = 'live-floating-heart';
        heart.textContent = '❤️';
        heart.style.left = (65 + Math.random() * 25) + '%';
        overlay.appendChild(heart);
        setTimeout(() => heart.remove(), 2000);
      });
    });
}

// ─── VIEWER COUNT ────────────────────────────────
function startViewerCountUpdater(streamId, isHost) {
  const updateCount = () => {
    db.collection('liveStreams').doc(streamId).collection('viewers')
      .where('isActive', '==', true)
      .get()
      .then(snap => {
        const count = snap.size;
        const el = document.getElementById('live-viewer-count');
        if (el) el.textContent = count;
        if (isHost) {
          db.collection('liveStreams').doc(streamId).update({
            currentViewerCount: count,
            peakViewerCount: FieldVal.increment(0) // we'll handle peak separately
          }).catch(() => {});
          // Update peak
          db.collection('liveStreams').doc(streamId).get().then(doc => {
            if (doc.exists && count > (doc.data().peakViewerCount || 0)) {
              doc.ref.update({ peakViewerCount: count }).catch(() => {});
            }
          }).catch(() => {});
        }
      }).catch(() => {});
  };
  updateCount();
  if (_liveViewerCountTimer) clearInterval(_liveViewerCountTimer);
  _liveViewerCountTimer = setInterval(updateCount, 15000);
}

// ─── Inline Reel Comments ────────────────────────
async function openReelComments(postId, options = {}) {
  const reelsScroll = document.querySelector('.reels-scroll');
  const scrollPos = reelsScroll ? reelsScroll.scrollTop : 0;
  const existing = document.getElementById('reel-comments-panel');
  const prevList = existing?.querySelector('#reel-comments-list');
  const prevListScroll = prevList ? prevList.scrollTop : 0;
  
  _pendingReelCommentImageFile = null;
  _reelCommentReplyTo = null;

  if (reelsScroll) reelsScroll.scrollTop = scrollPos;

  const panel = existing || document.createElement('div');
  panel.id = 'reel-comments-panel';
  panel.className = 'reel-comments-panel is-loading';
  panel.onclick = e => e.stopPropagation();
  panel.innerHTML = `
    <div class="reel-comments-header">
      <h3>Comments</h3>
      <button class="icon-btn" onclick="closeReelComments()">✕</button>
    </div>
    <div class="reel-comments-list" id="reel-comments-list">
      <div style="display:flex;align-items:center;justify-content:center;padding:24px 0"><span class="inline-spinner" style="width:24px;height:24px"></span></div>
    </div>
  `;
  if (!existing) (document.getElementById('video-hub') || document.body).appendChild(panel);
  document.getElementById('video-hub')?.classList.add('reel-comments-open');
  requestAnimationFrame(() => panel.classList.add('is-open'));

  let comments = [];
  try {
    const snap = await db.collection('posts').doc(postId).collection('comments').limit(100).get();
    comments = snap.docs.map(d => ({ id: d.id, ...d.data() }));
  } catch (e) { console.error(e); }

  comments.forEach(c => { c.likeCount = getReactionSummary(c.reactions, c.likes || []).total; });
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
    const myReaction = getUserReaction(c.reactions, c.likes || []);
    const cReplies = replyMap[c.id] || [];
    const target = c.replyTo ? commentById[c.replyTo] : null;
    const fromLabel = c.authorId === state.user.uid ? 'me' : (c.authorName || 'User');
    const toLabel = target ? (target.authorId === state.user.uid ? 'me' : (target.authorName || 'User')) : '';
    return `
      <div class="comment-item ${isReply ? 'reply-item' : ''}" id="rc-${c.id}" data-comment-id="${c.id}" data-author-id="${c.authorId || ''}">
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
            <button class="c-act" onclick="setReelCommentReply('${c.id}','${esc(c.authorName || 'User')}')">Reply</button>
            <button class="c-act like-only ${liked ? 'liked' : ''}" onclick="handleCommentHeartClick('${postId}','${c.id}','reel')" onmousedown="startCommentHeartHold('${postId}','${c.id}','reel','${myReaction || ''}', event)" onmouseup="endCommentHeartHold(event)" onmouseleave="endCommentHeartHold(event)" ontouchstart="startCommentHeartHold('${postId}','${c.id}','reel','${myReaction || ''}', event)" ontouchend="endCommentHeartHold(event)" ontouchcancel="endCommentHeartHold(event)">${renderCommentLikeMarkup(liked, c.likeCount || 0, myReaction || '')}</button>
          </div>
          ${cReplies.length ? `
            <button class="toggle-replies-btn" onclick="toggleCommentReplies(this)">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
              View ${cReplies.length} repl${cReplies.length === 1 ? 'y' : 'ies'}
            </button>
            <div class="comment-replies" style="display:none">${cReplies.map(r => renderComment(r, true)).join('')}</div>` : ''}
        </div>
      </div>`;
  };

  panel.className = 'reel-comments-panel is-open';
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
      <div class="chat-bar comment-chat-bar reel-comment-chat-bar">
        <label class="add-photo-btn chat-attach-btn comment-attach-btn" title="Add sticker/image">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
          <input type="file" hidden accept="image/*" id="reel-comment-image-input">
        </label>
        <textarea id="reel-comment-input" placeholder="Add a comment..." autocomplete="off" style="resize:none;overflow-y:auto;max-height:84px;min-height:40px;font-family:inherit;font-size:16px;line-height:1.3"></textarea>
        <button class="send-btn" onclick="postReelComment('${postId}')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg></button>
      </div>
    </div>
  `;

  const list = document.getElementById('reel-comments-list');
  if (list) {
    const reelPost = (_reelVideos || []).find(p => p.id === postId) || (state.posts || []).find(p => p.id === postId) || {};
    bindCommentLongPress(list, postId, 'reel', reelPost.authorId || '');
    requestAnimationFrame(() => {
      if (options.focusCommentId) {
        document.getElementById(`rc-${options.focusCommentId}`)?.scrollIntoView({ block: 'nearest' });
      } else if (options.scrollMode === 'preserve') {
        list.scrollTop = prevListScroll;
      } else {
        list.scrollTop = list.scrollHeight;
      }
    });
  }

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
  document.getElementById('video-hub')?.classList.remove('reel-comments-open');
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
    const docRef = await db.collection('posts').doc(postId).collection('comments').add({
      text: text || '', imageURL,
      authorId: state.user.uid,
      authorName: state.profile.displayName,
      authorPhoto: state.profile.photoURL || null,
      likes: [],
      replyTo: replyTo || null,
      createdAt: FieldVal.serverTimestamp()
    });
    shadowSyncComment(postId, docRef.id, {
      authorId: state.user.uid,
      authorName: state.profile.displayName,
      text: text || '',
      createdAt: new Date().toISOString()
    });
    await db.collection('posts').doc(postId).update({ commentsCount: FieldVal.increment(1) });
    _postTopCommentCache.set(postId, clampText(text || 'Photo comment', 90));
    _pendingReelCommentImageFile = null;
    _reelCommentReplyTo = null;
    clearReelCommentImage();
    clearReelCommentReply();
    openReelComments(postId, { focusCommentId: docRef.id, scrollMode: 'preserve' });
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
  await reactToComment(postId, commentId, '❤️', 'reel');
}

async function reelLike(pid, btn) {
  await reactToPost(pid, '❤️', 'reel');
}

// ─── Auto-play videos on scroll in feed ─────────
function setupFeedVideoAutoplay() {
  if (_feedAutoplayObserver) { _feedAutoplayObserver.disconnect(); _feedAutoplayObserver = null; }
  const feedEl = document.getElementById('feed-posts');
  if (!feedEl) return;
  _feedAutoplayObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const vid = entry.target.querySelector('video');
      if (!vid) return;
      if (entry.isIntersecting && entry.intersectionRatio >= 0.6) {
        vid.muted = false;
        vid.play().catch(() => {
          // Browser policy fallback if autoplay with sound is blocked.
          vid.muted = true;
          vid.play().catch(() => {});
        });
      } else {
        vid.pause();
      }
    });
  }, { threshold: [0.6] });
  feedEl.querySelectorAll('.unino-player').forEach(p => _feedAutoplayObserver.observe(p));
}

// ─── Like ────────────────────────────────────────
async function toggleLike(pid) {
  window._lastLikedPost = pid;
  await reactToPost(pid, '❤️', 'feed');
}

// ─── Comments with Replies ────────────────────────────────────
let _commentReplyTo = null; // { id, authorName } or null
let _sendingComment = false;
const _commentStateCache = {};

async function toggleCommentLike(cid, pid) {
  await reactToComment(pid, cid, '❤️', 'feed');
}

async function openComments(postId, options = {}) {
  const modalBg = $('#modal-bg');
  const modalInner = $('#modal-inner');
  const existingList = $('#comments-container');
  const prevListScroll = existingList ? existingList.scrollTop : 0;
  const isExistingCommentsModal = modalBg?.style.display === 'flex' && !!modalInner?.querySelector('.comment-modal-body');
  const loadingHtml = `
    <div class="modal-header"><h2>Comments</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body comment-modal-body" style="display:flex;align-items:center;justify-content:center;height:72vh;padding:0">
      <span class="inline-spinner" style="width:28px;height:28px;color:var(--accent)"></span>
    </div>
  `;
  if (isExistingCommentsModal) {
    modalInner.innerHTML = loadingHtml;
  } else {
    openModal(loadingHtml);
  }
  let postData = null;
  let comments = [];
  try {
    const postDoc = await db.collection('posts').doc(postId).get();
    postData = postDoc.exists ? postDoc.data() : null;
    const snap = await db.collection('posts').doc(postId).collection('comments').limit(100).get();
    comments = snap.docs.map(d => ({ id: d.id, ...d.data() }));
  } catch (e) { console.error(e); }

  // Process reactions
  comments.forEach(c => { c.likeCount = getReactionSummary(c.reactions, c.likes || []).total; });

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
  comments.forEach(c => {
    commentById[c.id] = c;
    _commentStateCache[c.id] = {
      reactions: normalizeReactionMap(c.reactions, c.likes || []),
      likes: [...(c.likes || [])]
    };
  });

  _commentReplyTo = null;
  _pendingCommentImageFile = null;
  const forceAnon = !!postData?.isAnonymous && postData?.authorId === state.user.uid;
  const supportsAnonChoice = !!postData?.isAnonymous && postData?.authorId !== state.user.uid;
  _commentAnonChoice = forceAnon ? true : (supportsAnonChoice ? true : false);

  function renderComment(c, isReply = false) {
     const liked = (c.likes || []).includes(state.user.uid);
      const myReaction = getUserReaction(c.reactions, c.likes || []);
     const cReplies = replyMap[c.id] || [];
      const hiddenIdentity = !!c.isAnonymous || (!!postData?.isAnonymous && c.authorId === postData.authorId);
      const displayName = hiddenIdentity ? 'Anonymous' : c.authorName;
      const displayPhoto = hiddenIdentity ? null : c.authorPhoto;
      const target = c.replyTo ? commentById[c.replyTo] : null;
      const targetHidden = !!target && (!!target.isAnonymous || (!!postData?.isAnonymous && target.authorId === postData.authorId));
      const targetDisplayName = target ? (targetHidden ? 'Anonymous' : (target.authorName || 'User')) : '';
      const replyLabel = target ? `${displayName} > ${target.authorId === state.user.uid ? 'me' : targetDisplayName}` : displayName;
     
     return `
      <div class="comment-item ${isReply ? 'reply-item' : ''}" id="c-${c.id}" data-comment-id="${c.id}" data-author-id="${c.authorId || ''}">
        <div class="comment-avatar-col">
          ${hiddenIdentity ? `<div class="avatar-sm anon-avatar">👻</div>` : avatar(displayName, displayPhoto, 'avatar-sm')}
        </div>
        <div class="comment-content-col">
           <div class="comment-bubble enhanced">
              <div class="comment-header">
                  <span class="comment-author" ${hiddenIdentity && c.authorId !== state.user.uid ? `onclick="openAnonPostActions('${c.authorId}')" style="cursor:pointer"` : hiddenIdentity ? '' : `onclick="openProfile('${c.authorId}')"`}>${esc(replyLabel)}</span>
              </div>
              <div class="comment-text">${esc(c.text)}</div>
              ${c.imageURL ? `<img src="${c.imageURL}" class="comment-inline-image" onclick="viewImage('${c.imageURL}')">` : ''}
           </div>
           <div class="comment-actions-row">
               <span class="comment-time">${timeAgo(c.createdAt)}</span>
            <button class="c-act" onclick="setCommentReply('${c.id}','${esc(displayName)}')">Reply</button>
            <button class="c-act like-only ${liked?'liked':''}" onclick="handleCommentHeartClick('${postId}','${c.id}','feed')" onmousedown="startCommentHeartHold('${postId}','${c.id}','feed','${myReaction || ''}', event)" onmouseup="endCommentHeartHold(event)" onmouseleave="endCommentHeartHold(event)" ontouchstart="startCommentHeartHold('${postId}','${c.id}','feed','${myReaction || ''}', event)" ontouchend="endCommentHeartHold(event)" ontouchcancel="endCommentHeartHold(event)">${renderCommentLikeMarkup(liked, c.likeCount || 0, myReaction || '')}</button>
           </div>
           ${cReplies.length ? `
             <button class="toggle-replies-btn" onclick="toggleCommentReplies(this)">
               <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
               View ${cReplies.length} repl${cReplies.length === 1 ? 'y' : 'ies'}
             </button>
             <div class="comment-replies" style="display:none">
               ${cReplies.map(r => renderComment(r, true)).join('')}
             </div>` : ''}
        </div>
      </div>`;
  }

  function renderCommentTree() {
    if (!comments.length) return '<p class="empty-msg" style="text-align:center;padding:20px;color:var(--text-tertiary)">No comments. be the first.</p>';
    return topLevel.map(c => renderComment(c)).join('');
  }

  const commentModalHtml = `
    <div class="modal-header"><h2>Comments</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body comment-modal-body" style="display:flex;flex-direction:column;height:72vh;padding:0">
      <div id="comments-container" class="comments-scroll" style="flex:1;overflow-y:auto;padding:16px 16px 8px">
        ${renderCommentTree()}
      </div>
      <div id="comment-reply-indicator" class="reply-indicator" style="display:none">
        <span id="comment-reply-label"></span>
        <button onclick="clearCommentReply()">&times;</button>
      </div>
      <div class="comment-input-wrap modern" style="position:sticky;bottom:0;flex-shrink:0">
        ${postData?.isAnonymous ? `<div style="display:flex;align-items:flex-start;gap:8px;font-size:12px;color:var(--text-secondary);margin-bottom:8px">
          <input type="checkbox" id="comment-anon-toggle" ${_commentAnonChoice ? 'checked' : ''} ${forceAnon ? 'disabled' : ''} onchange="setCommentAnonChoice(this.checked)" style="margin-top:2px;flex-shrink:0">
          <span>${forceAnon ? 'Your comments stay anonymous on your anonymous post' : 'Comment anonymously on this anonymous post'}</span>
        </div>` : ''}
        <div id="comment-img-preview" class="comment-img-preview" style="display:none"></div>
        <div class="chat-bar comment-chat-bar">
          <label class="add-photo-btn chat-attach-btn comment-attach-btn" title="Add sticker/image">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
            <input type="file" hidden accept="image/*" id="comment-image-input">
          </label>
          <textarea id="comment-input" placeholder="Write a comment..." autocomplete="off" style="resize:none;overflow-y:auto;max-height:84px;min-height:40px;font-family:inherit;font-size:16px;line-height:1.3"></textarea>
          <button class="send-btn" onclick="postComment('${postId}')"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg></button>
        </div>
      </div>
    </div>
  `;

  const activeInner = $('#modal-inner');
  if (!activeInner) return;
  activeInner.innerHTML = commentModalHtml;

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

  const commentsList = $('#comments-container');
  if (commentsList) {
    bindCommentLongPress(commentsList, postId, 'feed', postData?.authorId || '');
    requestAnimationFrame(() => {
      if (options.focusCommentId) {
        document.getElementById(`c-${options.focusCommentId}`)?.scrollIntoView({ block: 'nearest' });
      } else if (options.scrollMode === 'preserve') {
        commentsList.scrollTop = prevListScroll;
      }
    });
  }
}

function setCommentAnonChoice(next) {
  _commentAnonChoice = !!next;
}

function toggleCommentReplies(btn) {
  const repliesDiv = btn.nextElementSibling;
  if (!repliesDiv) return;
  const isHidden = repliesDiv.style.display === 'none';
  repliesDiv.style.display = isHidden ? 'flex' : 'none';
  btn.classList.toggle('expanded', isHidden);
  const count = repliesDiv.querySelectorAll('.comment-item').length;
  const svg = isHidden
    ? '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 15 12 9 18 15"/></svg>'
    : '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>';
  btn.innerHTML = `${svg} ${isHidden ? 'Hide' : 'View'} ${count} repl${count === 1 ? 'y' : 'ies'}`;
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
    const docRef = await db.collection('posts').doc(postId).collection('comments').add({
      text: text || '', imageURL,
      authorId: state.user.uid, authorName: commentAnon ? 'Anonymous' : state.profile.displayName,
      authorPhoto: commentAnon ? null : (state.profile.photoURL || null), isAnonymous: commentAnon,
      likes: [],
      replyTo: replyTo || null,
      createdAt: FieldVal.serverTimestamp()
    });
    shadowSyncComment(postId, docRef.id, {
      authorId: state.user.uid,
      authorName: commentAnon ? 'Anonymous' : state.profile.displayName,
      text: text || '',
      createdAt: new Date().toISOString()
    });
    await db.collection('posts').doc(postId).update({ commentsCount: FieldVal.increment(1) });
    
    if (postData) addNotification(postData.authorId, 'comment', 'commented on your post', { postId }, { anonymous: commentAnon });

    // Silent UI update: append new comment without rebuilding modal.
    const list = $('#comments-container');
    if (list) {
      const displayName = commentAnon ? 'Anonymous' : state.profile.displayName;
      const displayPhoto = commentAnon ? null : (state.profile.photoURL || null);
      const tempId = `tmp-${Date.now()}`;
      const node = document.createElement('div');
      node.className = 'comment-item';
      node.id = `c-${tempId}`;
      node.dataset.commentId = tempId;
      node.dataset.authorId = state.user.uid;
      _commentStateCache[tempId] = { reactions: {}, likes: [] };
      node.innerHTML = `
        <div class="comment-avatar-col">${commentAnon ? '<div class="avatar-sm anon-avatar">👻</div>' : avatar(displayName, displayPhoto, 'avatar-sm')}</div>
        <div class="comment-content-col">
          <div class="comment-bubble enhanced">
            <div class="comment-header"><span class="comment-author">${esc(displayName)}</span></div>
            <div class="comment-text">${esc(text || '')}</div>
            ${imageURL ? `<img src="${imageURL}" class="comment-inline-image" onclick="viewImage('${imageURL}')">` : ''}
          </div>
          <div class="comment-actions-row">
            <span class="comment-time">Just now</span>
            <button class="c-act" onclick="setCommentReply('${tempId}','${esc(displayName)}')">Reply</button>
            <button class="c-act like-only" type="button" disabled title="Syncing...">${renderCommentLikeMarkup(false, 0, '')}</button>
          </div>
        </div>
      `;
      if (replyTo) {
        const parentItem = list.querySelector(`#c-${replyTo}`);
        if (parentItem) {
          let parentReplyWrap = parentItem.querySelector('.comment-replies');
          let toggleBtn = parentItem.querySelector('.toggle-replies-btn');
          if (!parentReplyWrap) {
            const contentCol = parentItem.querySelector('.comment-content-col');
            toggleBtn = document.createElement('button');
            toggleBtn.className = 'toggle-replies-btn expanded';
            toggleBtn.setAttribute('onclick', 'toggleCommentReplies(this)');
            toggleBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 15 12 9 18 15"/></svg> Hide 1 reply';
            parentReplyWrap = document.createElement('div');
            parentReplyWrap.className = 'comment-replies';
            parentReplyWrap.style.display = 'flex';
            if (contentCol) {
              contentCol.appendChild(toggleBtn);
              contentCol.appendChild(parentReplyWrap);
            }
          }
          const nextCount = (parentReplyWrap?.querySelectorAll('.comment-item').length || 0) + 1;
          if (toggleBtn) {
            toggleBtn.classList.add('expanded');
            toggleBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 15 12 9 18 15"/></svg> Hide ${nextCount} repl${nextCount === 1 ? 'y' : 'ies'}`;
          }
          if (parentReplyWrap) {
            parentReplyWrap.style.display = 'flex';
            parentReplyWrap.appendChild(node);
          } else {
            list.prepend(node);
          }
        } else {
          list.prepend(node);
        }
      } else {
        list.prepend(node);
      }
    }

    _pendingCommentImageFile = null;
    _commentReplyTo = null;
    clearCommentImage();
    clearCommentReply();
    state.posts = (state.posts || []).map(p => p.id === postId ? { ...p, commentsCount: Number(p.commentsCount || 0) + 1 } : p);
    refreshPostCardsUI(postId);
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
let _galleryUrls = [];
let _galleryIdx = 0;
function viewImage(url) { openGallery([url], 0); }

function inferDownloadFilename(url = '', fallbackBase = 'unino-media') {
  try {
    const clean = String(url || '').split('?')[0].split('#')[0];
    const last = clean.split('/').pop() || '';
    if (last && /\.[a-zA-Z0-9]{2,5}$/.test(last)) return last;
  } catch (_) {}
  return `${fallbackBase}-${Date.now()}.jpg`;
}

function isLikelyVideoUrl(url = '', mime = '') {
  const lowerUrl = String(url || '').toLowerCase();
  const lowerMime = String(mime || '').toLowerCase();
  return lowerMime.includes('video') || /\.(mp4|webm|mov|m4v|avi|mkv)(\?|#|$)/.test(lowerUrl);
}

async function downloadUrlInApp(url = '') {
  const target = String(url || '').trim();
  if (!target) return;
  let started = false;
  try {
    const response = await fetch(target, { mode: 'cors', cache: 'no-cache' });
    if (!response.ok) throw new Error(`http-${response.status}`);
    const blob = await response.blob();
    if (!blob.size) throw new Error('empty-blob');
    const objectUrl = URL.createObjectURL(blob);
    const fallbackBase = isLikelyVideoUrl(target, blob.type) ? 'unino-video' : 'unino-image';
    const filename = inferDownloadFilename(target, fallbackBase);
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = filename;
    link.rel = 'noopener';
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(objectUrl), 1500);
    started = true;
    toast('Download started');
  } catch (e) {
    console.warn('download failed', e);
  }
  if (started) return;
  try {
    const filename = inferDownloadFilename(target, isLikelyVideoUrl(target) ? 'unino-video' : 'unino-image');
    const link = document.createElement('a');
    link.href = target;
    link.download = filename;
    link.rel = 'noopener';
    link.target = '_blank';
    document.body.appendChild(link);
    link.click();
    link.remove();
    toast('Opening download...');
  } catch (_) {
    toast('Could not download media');
  }
}

function downloadCurrentGalleryMedia() {
  const url = _galleryUrls[_galleryIdx] || '';
  if (!url) return;
  downloadUrlInApp(url);
}

function closeGalleryViewer() {
  const v = $('#img-view');
  if (v) v.style.display = 'none';
  document.body.classList.remove('image-view-open');
  _galleryUrls = [];
}

function guessDownloadFilename(url = '', fallback = 'download') {
  try {
    const clean = String(url || '').split('#')[0].split('?')[0];
    const last = clean.split('/').pop() || '';
    if (last && /\.[a-z0-9]{2,8}$/i.test(last)) return last;
  } catch (_) {}
  return fallback;
}

async function downloadMediaUrl(url = '', fallback = 'download') {
  const target = String(url || '').trim();
  if (!target) return;
  try {
    const name = guessDownloadFilename(target, fallback);
    const a = document.createElement('a');
    a.href = target;
    a.download = name;
    a.rel = 'noopener';
    a.target = '_blank';
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch (e) {
    console.error(e);
    toast('Could not start download');
  }
}

function downloadCurrentGalleryImage() {
  if (!_galleryUrls.length) return;
  downloadMediaUrl(_galleryUrls[_galleryIdx], 'image');
}

function openVideoDownloadPrompt(url = '') {
  const target = String(url || '').trim();
  if (!target) return;
  openModal(`
    <div class="modal-header"><h2>Video</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" style="padding:16px;display:grid;gap:10px">
      <button class="btn-primary btn-full" onclick="closeModal();downloadMediaUrl(${JSON.stringify(target)}, 'video')">Download</button>
      <button class="btn-secondary btn-full" onclick="closeModal()">Cancel</button>
    </div>
  `);
}

const _mediaCacheWarm = new Map();
async function primeMediaCache(urls = [], limit = 6) {
  if (!('caches' in window)) return;
  const list = [...new Set((urls || []).filter(Boolean))]
    .filter(url => !isLikelyVideoUrl(url))
    .slice(0, limit);
  if (!list.length) return;
  try {
    const cache = await caches.open('unino-media-v1');
    const now = Date.now();
    await Promise.all(list.map(async url => {
      const last = _mediaCacheWarm.get(url) || 0;
      if (now - last < 15 * 60 * 1000) return;
      _mediaCacheWarm.set(url, now);
      const match = await cache.match(url);
      if (match) return;
      try {
        const resp = await fetch(url, { cache: 'no-store', mode: 'cors' });
        if (resp.ok) await cache.put(url, resp.clone());
      } catch (_) {}
    }));
  } catch (_) {}
}
function openGallery(urls, startIdx = 0) {
  _galleryUrls = urls || [];
  _galleryIdx = startIdx;
  const v = $('#img-view'); if (!v) return;
  _renderGalleryFrame();
  document.body.classList.add('image-view-open');
  v.style.display = 'flex';
}
function _renderGalleryFrame() {
  if (!_galleryUrls.length) return;
  const currentUrl = _galleryUrls[_galleryIdx];
  $('#img-full').src = currentUrl;
  const counter = $('#img-counter');
  const prev = $('#img-prev');
  const next = $('#img-next');
  const downloadBtn = $('#img-download');
  if (downloadBtn) {
    downloadBtn.style.display = currentUrl ? 'flex' : 'none';
  }
  if (_galleryUrls.length > 1) {
    counter.textContent = `${_galleryIdx + 1} / ${_galleryUrls.length}`;
    counter.style.display = 'block';
    prev.style.display = _galleryIdx > 0 ? 'flex' : 'none';
    next.style.display = _galleryIdx < _galleryUrls.length - 1 ? 'flex' : 'none';
  } else {
    counter.style.display = 'none';
    prev.style.display = 'none';
    next.style.display = 'none';
  }
}

// ─── Create Post ─────────────────────────────────
function openCreateModal() {
  let pendingFiles = [];
  let pendingIsVideo = false;
  let locationPing = { enabled: false, label: '', lat: null, lng: null, expiresAt: null };
  let postPoll = null;
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
      <div class="create-author-row">
        <div class="create-author-main">
          <div id="create-author-avatar">${avatar(state.profile.displayName, state.profile.photoURL, 'avatar-md')}</div>
          <div>
            <div id="create-author-name" style="font-weight:600">${esc(state.profile.displayName)}</div>
            <div style="font-size:12px;color:var(--text-secondary)">Posting to ${esc(state.profile.university || 'NWU')}</div>
          </div>
        </div>
        <div class="create-author-actions">
          <button type="button" class="btn-outline create-anon-toggle" id="create-anon-toggle" aria-pressed="false" title="Toggle anonymous posting">👻 Anon off</button>
        </div>
      </div>
      <textarea id="create-text" placeholder="What's on your mind?" style="width:100%;min-height:100px;border:none;background:transparent;color:var(--text-primary);font-size:16px;resize:none;outline:none"></textarea>
      <div class="create-post-hint">Use #MATH301-style tags inside your post so students in that module can discover it. Choose Anonymous if you want the post hidden from your identity.</div>
      <div id="create-mention-suggestions" class="create-mention-suggestions" style="display:none"></div>
      <div id="create-preview" class="media-preview" style="display:none">
        <div id="create-preview-content" class="collage-preview-grid"></div>
        <button class="image-preview-remove" onclick="document.getElementById('create-preview').style.display='none';window._createPendingFiles=[]">&times;</button>
      </div>
      <div class="create-post-actions-row">
        <div class="create-post-actions-left">
          <label class="add-photo-btn" title="Media"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg><input type="file" hidden accept="image/*,video/*" id="create-media-file" multiple></label>
          <button type="button" class="btn-outline" id="create-pin-location" style="padding:7px 10px;font-size:12px">📍 Pin</button>
          <button type="button" class="btn-outline" id="create-poll-btn" style="padding:7px 10px;font-size:12px">📊 Poll</button>
          <select id="create-visibility" style="padding:6px 10px;border-radius:100px;border:1px solid var(--border);background:var(--bg-tertiary);color:var(--text-primary);font-size:12px;font-weight:600">
            <option value="public">🌍 Public</option>
            <option value="friends">👫 Friends</option>
          </select>
        </div>
        <button class="btn-primary" id="create-submit" style="padding:10px 28px">Post</button>
      </div>
      <div id="create-location-pill" style="display:none;margin-top:10px;font-size:12px;color:var(--text-secondary)"></div>
      <div id="create-poll-pill" class="create-poll-pill" style="display:none;margin-top:8px"></div>
      <div id="create-poll-editor" class="create-poll-editor" style="display:none;margin-top:12px">
        <div class="form-group"><label>Poll question</label><input type="text" id="post-poll-question" placeholder="Ask something"></div>
        <div class="form-group"><label>Options</label><textarea id="post-poll-options" style="height:110px;resize:none" placeholder="One option per line">Yes
No</textarea></div>
        <div class="create-poll-editor-actions">
          <button type="button" class="btn-outline" id="cancel-post-poll">Cancel</button>
          <button type="button" class="btn-primary" id="save-post-poll">Save Poll</button>
        </div>
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
    let createAnon = false;
    const anonIdentity = getUserAnonIdentity(state.profile || {});
    const mentionMap = new Map();
    const normalizeCreatePoll = (questionRaw = '', optionsRaw = '') => {
      const question = String(questionRaw || '').trim().slice(0, 160);
      const options = Array.from(new Set(String(optionsRaw || '').split('\n').map(v => v.trim()).filter(Boolean))).slice(0, 6);
      return { question, options };
    };
    const setPollEditorVisible = (visible = false) => {
      const editor = $('#create-poll-editor');
      if (!editor) return;
      editor.style.display = visible ? 'block' : 'none';
      if (!visible) return;
      const qInput = $('#post-poll-question');
      const oInput = $('#post-poll-options');
      if (qInput) qInput.value = postPoll?.question || '';
      if (oInput) oInput.value = (postPoll?.options?.length ? postPoll.options : ['Yes', 'No']).join('\n');
      qInput?.focus();
    };

    const refreshLocationUI = () => {
      const pill = $('#create-location-pill');
      if (!pill) return;
      if (!locationPing.enabled) {
        pill.style.display = 'none';
        pill.textContent = '';
        return;
      }
      const label = locationPing.label || 'Current location';
      pill.style.display = 'block';
      pill.textContent = `📍 ${label} · expires in 24h`;
    };

    const refreshPollUI = () => {
      const pill = $('#create-poll-pill');
      const pollBtn = $('#create-poll-btn');
      if (pollBtn) pollBtn.textContent = postPoll?.question ? '📊 Edit poll' : '📊 Poll';
      if (!pill) return;
      if (!postPoll?.question) {
        pill.style.display = 'none';
        pill.innerHTML = '';
        return;
      }
      pill.style.display = 'flex';
      pill.innerHTML = `<strong>📊 ${esc(postPoll.question)}</strong><div class="create-poll-pill-actions"><button type="button" class="create-pill-action" data-act="edit">Edit</button><button type="button" class="create-pill-action" data-act="remove">Remove</button></div>`;
      pill.querySelector('[data-act="edit"]')?.addEventListener('click', () => setPollEditorVisible(true));
      pill.querySelector('[data-act="remove"]')?.addEventListener('click', () => {
        postPoll = null;
        setPollEditorVisible(false);
        refreshPollUI();
      });
    };

    const applyAnonUi = () => {
      const btn = $('#create-anon-toggle');
      if (!btn) return;
      btn.setAttribute('aria-pressed', createAnon ? 'true' : 'false');
      btn.textContent = createAnon ? '👻 Anon on' : '👻 Anon off';
      btn.classList.toggle('anon-active', createAnon);
      const nameEl = $('#create-author-name');
      if (nameEl) nameEl.textContent = createAnon ? anonIdentity : (state.profile.displayName || 'User');
      const avatarEl = $('#create-author-avatar');
      if (avatarEl) {
        avatarEl.innerHTML = createAnon
          ? '<div class="avatar-md anon-avatar">👻</div>'
          : avatar(state.profile.displayName, state.profile.photoURL, 'avatar-md');
      }
    };

    const renderMentionSuggestions = async () => {
      const input = $('#create-text');
      const box = $('#create-mention-suggestions');
      if (!input || !box) return;
      const value = input.value || '';
      const atIndex = value.lastIndexOf('@');
      if (atIndex < 0) { box.style.display = 'none'; return; }
      const tail = value.slice(atIndex + 1);
      if (!tail || /\s/.test(tail)) { box.style.display = 'none'; return; }
      const q = tail.trim().toLowerCase();
      if (q.length < 1) { box.style.display = 'none'; return; }
      const users = await getUsersCache().catch(() => []);
      const hits = users
        .filter(u => u.id !== state.user.uid)
        .map(u => ({ user: u, score: userSearchScore(u, q) }))
        .filter(entry => entry.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 6)
        .map(entry => entry.user);
      if (!hits.length) { box.style.display = 'none'; return; }
      box.innerHTML = hits.map(u => `
        <button type="button" class="create-mention-item" data-uid="${u.id}" data-name="${esc(u.displayName || 'User')}">
          ${avatar(u.displayName, u.photoURL, 'avatar-sm')}
          <span>${esc(u.displayName || 'User')}</span>
        </button>
      `).join('');
      box.style.display = 'block';
      box.querySelectorAll('.create-mention-item').forEach(btn => {
        btn.onclick = () => {
          const name = btn.getAttribute('data-name') || 'User';
          const uid = btn.getAttribute('data-uid') || '';
          const handle = mentionHandleForName(name);
          const next = `${value.slice(0, atIndex)}@${handle} `;
          input.value = next;
          mentionMap.set(handle, uid);
          box.style.display = 'none';
          input.focus();
        };
      });
    };

    const showPreviews = () => {
      const pc = $('#create-preview-content');
      if (!pendingFiles.length) { $('#create-preview').style.display = 'none'; return; }
      if (pendingIsVideo) {
        pc.innerHTML = `<video src="${localPreview(pendingFiles[0])}" style="width:100%;max-height:200px;border-radius:var(--radius);background:#000" autoplay muted loop playsinline></video>`;
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
    if ($('#create-media-file')) $('#create-media-file').onchange = e => {
      const files = Array.from(e.target.files || []);
      if (!files.length) return;
      const pickedVideo = files.find(file => file.type?.startsWith('video/'));
      if (pickedVideo) {
        pendingFiles = [pickedVideo];
        pendingIsVideo = true;
      } else {
        pendingFiles = files;
        pendingIsVideo = false;
      }
      showPreviews();
    };
    if ($('#create-anon-toggle')) {
      $('#create-anon-toggle').onclick = () => {
        createAnon = !createAnon;
        applyAnonUi();
      };
      applyAnonUi();
    }
    const createTextInput = $('#create-text');
    if (createTextInput) {
      createTextInput.addEventListener('input', () => { renderMentionSuggestions().catch(() => {}); });
      createTextInput.addEventListener('blur', () => {
        setTimeout(() => {
          const box = $('#create-mention-suggestions');
          if (box) box.style.display = 'none';
        }, 120);
      });
      createTextInput.addEventListener('focus', () => { renderMentionSuggestions().catch(() => {}); });
    }
    if ($('#create-pin-location')) $('#create-pin-location').onclick = () => {
      if (!navigator.geolocation) return toast('Location is not available on this device');
      toast('Getting location...');
      navigator.geolocation.getCurrentPosition(pos => {
        const lat = Number(pos.coords.latitude.toFixed(6));
        const lng = Number(pos.coords.longitude.toFixed(6));
        const nearestCampus = (typeof CAMPUS_LOCATIONS !== 'undefined' ? CAMPUS_LOCATIONS : []).reduce((best, loc) => {
          const dist = distanceKmBetween({ lat, lng }, { lat: loc.lat, lng: loc.lng });
          return !best || dist < best.dist ? { loc, dist } : best;
        }, null);
        locationPing = {
          enabled: true,
          label: nearestCampus?.loc?.name || `${lat}, ${lng}`,
          lat,
          lng,
          expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000)
        };
        refreshLocationUI();
        toast('Pinned for 24h');
      }, () => toast('Could not get location'), {
        enableHighAccuracy: true,
        timeout: 12000,
        maximumAge: 0
      });
    };
    const savePostPoll = () => {
      const questionRaw = $('#post-poll-question')?.value || '';
      const optionsRaw = $('#post-poll-options')?.value || '';
      const parsed = normalizeCreatePoll(questionRaw, optionsRaw);
      if (!parsed.question || parsed.options.length < 2) return toast('Add a question and at least 2 options');
      postPoll = { question: parsed.question, options: parsed.options, votes: {} };
      setPollEditorVisible(false);
      refreshPollUI();
    };
    if ($('#create-poll-btn')) $('#create-poll-btn').onclick = () => setPollEditorVisible(true);
    if ($('#save-post-poll')) $('#save-post-poll').onclick = savePostPoll;
    if ($('#cancel-post-poll')) $('#cancel-post-poll').onclick = () => {
      setPollEditorVisible(false);
      refreshPollUI();
    };
    refreshLocationUI();
    refreshPollUI();
    if ($('#create-submit')) $('#create-submit').onclick = async () => {
      const text = $('#create-text').value.trim();
      const moderation = applySoftKeywordFilterText(text);
      const safeText = moderation.text;
      const moduleTags = extractModuleTags(text);
      const hashTags = extractHashTags(text);
      const mentionHandles = extractMentionHandles(text);
      const pollEditor = $('#create-poll-editor');
      const pollEditorOpen = !!pollEditor && pollEditor.style.display !== 'none';
      if (pollEditorOpen) {
        const questionRaw = $('#post-poll-question')?.value || '';
        const optionsRaw = $('#post-poll-options')?.value || '';
        const parsed = normalizeCreatePoll(questionRaw, optionsRaw);
        const hasDraft = !!parsed.question || !!String(optionsRaw || '').trim();
        if (hasDraft) {
          if (!parsed.question || parsed.options.length < 2) return toast('Add a poll question and at least 2 options, or remove it');
          postPoll = { question: parsed.question, options: parsed.options, votes: {} };
          refreshPollUI();
        }
      }
      if (!safeText && !pendingFiles.length && !postPoll) return toast('Post cannot be empty');
      const visibility = $('#create-visibility')?.value || 'public';
      const isAnon = !!createAnon;
      const anonIdentity = getUserAnonIdentity(state.profile || {});
      const postPollPayload = postPoll?.question ? { question: postPoll.question, options: [...(postPoll.options || [])], votes: {} } : null;
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
          content: safeText,
          imageURL: mediaType === 'image' || mediaType === 'collage' ? mediaURL : null,
          imageURLs: imageURLs || null,
          videoURL: mediaType === 'video' ? mediaURL : null,
          mediaType,
          authorId: state.user.uid,
          authorName: isAnon ? anonIdentity : state.profile.displayName,
          authorPhoto: isAnon ? null : (state.profile.photoURL || null),
          authorUni: state.profile.university || '',
          moduleTags,
          hashTags,
          mentionHandles,
          moderationFlags: moderation.flags,
          moderationReviewNeeded: moderation.flagged,
          locationPing: locationPing.enabled ? locationPing : null,
          isAnonymous: isAnon || false,
          visibility,
          poll: postPollPayload,
          createdAt: FieldVal.serverTimestamp(), likes: [], commentsCount: 0
        });
        shadowSyncPost(postRef.id, {
          authorId: state.user.uid,
          authorName: isAnon ? anonIdentity : state.profile.displayName,
          content: safeText,
          imageURL: mediaType === 'image' || mediaType === 'collage' ? mediaURL : null,
          videoURL: mediaType === 'video' ? mediaURL : null,
          visibility,
          createdAt: new Date().toISOString()
        });
        if (moduleTags.length) notifyRelevantModuleUsers(moduleTags, safeText, postRef.id, isAnon);
        if (mentionHandles.length) {
          const users = await getUsersCache().catch(() => []);
          const byName = new Map(users.map(u => [mentionHandleForName(u.displayName || ''), u.id]));
          const targets = new Set();
          mentionHandles.forEach(handle => {
            if (mentionMap.has(handle)) targets.add(mentionMap.get(handle));
            else if (byName.has(handle)) targets.add(byName.get(handle));
          });
          targets.forEach(uid => {
            if (!uid || uid === state.user.uid) return;
            addNotification(uid, 'mention', 'mentioned you in a post', { postId: postRef.id }, { anonymous: isAnon });
          });
        }
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
        const eventPayload = {
          title, location: locationText, date, time, description: desc,
          imageURLs,
          gradient: gradients[Math.floor(Math.random() * gradients.length)],
          createdBy: state.user.uid,
          creatorName: state.profile.displayName,
          going: [state.user.uid],
          createdAt: FieldVal.serverTimestamp()
        };
        const evRef = await db.collection('events').add(eventPayload);
        await upsertEventChatGroup({ id: evRef.id, ...eventPayload });
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
  if (!['radar', 'list'].includes(exploreView)) exploreView = 'radar';
  const c = $('#content');
  c.innerHTML = `
    <div class="explore-page">
      <div class="explore-toggle">
        <button class="explore-toggle-btn ${exploreView === 'radar' ? 'active' : ''}" data-v="radar">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
          Radar
        </button>
        <button class="explore-toggle-btn ${exploreView === 'list' ? 'active' : ''}" data-v="list">
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
      exploreView = btn.dataset.v || 'radar';
      syncExploreToggleButtons();
      renderExploreView();
    };
  });
  loadExploreUsers();
}

async function loadExploreUsers() {
  try {
    // Load events in parallel with users
    const [snap] = await Promise.all([
      db.collection('users').get(),
      loadCampusEvents()
    ]);
    const myMajor = state.profile.major || '';
    const myModules = normalizeModules(state.profile.modules || []);
    const myYear = state.profile.year || '';
    const blockedUsers = new Set(state.profile.blockedUsers || []);
    const blockedBy = new Set(state.profile.blockedBy || []);
    const myFriends = new Set(state.profile.friends || []);

    allExploreUsers = snap.docs
      .map(d => ({ id: d.id, ...d.data() }))
      .filter(u => u.id !== state.user.uid && !blockedUsers.has(u.id) && !blockedBy.has(u.id))
      .map(u => {
        const uModules = normalizeModules(u.modules || []);
        const shared = myModules.filter(m => uModules.includes(m));
        const nearby = getNearbySignal(state.profile, u);
        const nearbyScore = nearby.score;
        let proximity = 'far';
        if (shared.length > 0) proximity = 'module';
        else if (nearbyScore > 0) proximity = 'nearby';
        else if (u.major === myMajor) proximity = 'course';
        const isFriend = myFriends.has(u.id);
        const affinityScore = (shared.length * 20)
          + (u.major === myMajor ? 18 : 0)
          + (u.year && myYear && u.year === myYear ? 6 : 0)
          + (nearbyScore * 8)
          + (u.status === 'online' ? 4 : 0)
          + (isFriend ? 14 : 0)
          + genderAffinityScore(state.profile, u);
        return { ...u, isFriend, sharedModules: shared, proximity, nearbyScore, distanceKm: nearby.distanceKm, nearbySource: nearby.source, affinityScore };
      })
      .sort((a, b) => b.affinityScore - a.affinityScore || (a.distanceKm || Infinity) - (b.distanceKm || Infinity));
    renderExploreView();
  } catch (e) {
    console.error(e);
    const body = $('#explore-body');
    if (body) body.innerHTML = '<div class="empty-state"><h3>Could not load students</h3></div>';
  }
}

function syncExploreToggleButtons() {
  $$('.explore-toggle-btn').forEach(button => {
    button.classList.toggle('active', button.dataset.v === exploreView);
  });
}

function openCampusMapView() {
  exploreView = 'radar';
  if (state.page !== 'explore') {
    navigate('explore');
    return;
  }
  syncExploreToggleButtons();
  renderExploreView();
}

function renderExploreView() {
  syncExploreToggleButtons();
  if (exploreView === 'list') renderListView();
  else renderRadarView();
}

function switchToExploreList(filter = 'all', queryRaw = '') {
  exploreView = 'list';
  const query = (queryRaw || '').trim();
  _exploreSearchQuery = query;
  syncExploreToggleButtons();
  renderListView(query, filter);
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
      <div class="radar-overlay" id="radar-overlay">
        <div class="radar-ring r3"></div>
        <div class="radar-ring r2"></div>
        <div class="radar-ring r1"></div>
        <div class="radar-sweep-anim"></div>
      </div>
    </div>
    <div id="map-route-panel" class="map-route-panel" style="display:none"></div>
    <div id="radar-dynamic"></div>
  `;

  const applyRadarFilter = (queryRaw = '') => {
    const searchQuery = (queryRaw || '').trim();
    let filteredUsers = allExploreUsers.map(u => ({ ...u, _searchScore: userSearchScore(u, searchQuery) }));
    if (searchQuery) {
      filteredUsers = filteredUsers
        .filter(u => u._searchScore > 0)
        .sort((a, b) => b._searchScore - a._searchScore || (b.affinityScore || 0) - (a.affinityScore || 0));
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
          <div class="proximity-header"><h3>🔗 Shared Modules</h3><span class="proximity-count">${moduleUsers.length}</span><button class="btn-outline btn-sm" onclick="switchToExploreList('module', '${esc(searchQuery)}')">View all</button></div>
          <div class="proximity-scroll">${moduleUsers.map(u => proximityCard(u)).join('')}</div>
        </div>` : ''}

        <div class="proximity-section">
          <div class="proximity-header"><h3>📍 Nearby Area</h3><span class="proximity-count">${nearbyUsers.length}</span><button class="btn-outline btn-sm" onclick="switchToExploreList('nearby', '${esc(searchQuery)}')">View all</button></div>
          <div class="proximity-scroll">
            ${nearbyUsers.length ? nearbyUsers.map(u => proximityCard(u)).join('')
              : '<p style="padding:12px;color:var(--text-tertiary);font-size:13px">No one found yet</p>'}
          </div>
        </div>

        ${courseUsers.length ? `
        <div class="proximity-section">
          <div class="proximity-header"><h3>📚 Same Course</h3><span class="proximity-count">${courseUsers.length}</span><button class="btn-outline btn-sm" onclick="switchToExploreList('course', '${esc(searchQuery)}')">View all</button></div>
          <div class="proximity-scroll">${courseUsers.map(u => proximityCard(u)).join('')}</div>
        </div>` : ''}

        ${otherUsers.length ? `
        <div class="proximity-section">
          <div class="proximity-header"><h3>🎓 Other Students</h3><span class="proximity-count">${otherUsers.length}</span><button class="btn-outline btn-sm" onclick="switchToExploreList('all', '${esc(searchQuery)}')">View all</button></div>
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

  let radarSuggestionSeq = 0;
  const renderRadarSuggestions = async (queryRaw = '') => {
    const box = $('#radar-suggestions');
    if (!box) return;
    box.onmousedown = e => e.preventDefault();
    const rawQuery = (queryRaw || '').trim();
    const q = normalizePlaceKey(rawQuery);
    if (!q) {
      box.style.display = 'none';
      box.innerHTML = '';
      return;
    }
    const seq = ++radarSuggestionSeq;

    const peopleHits = allExploreUsers
      .map(u => ({ ...u, _searchScore: userSearchScore(u, rawQuery) }))
      .filter(u => u._searchScore > 0)
      .sort((a, b) => b._searchScore - a._searchScore || (b.affinityScore || 0) - (a.affinityScore || 0))
      .slice(0, 4)
      .map(u => ({ type: 'person', uid: u.id, label: u.displayName || 'User', sub: u.major || u.address || '' }));

    const campusPlaces = CAMPUS_LOCATIONS
      .map(loc => ({
        type: 'place',
        label: loc.name,
        sub: 'Campus location',
        lat: Number(loc.lat),
        lng: Number(loc.lng),
        locationId: loc.id,
        source: 'campus'
      }))
      .filter(place => scorePlaceQueryMatch(place.label, rawQuery) > 0);

    const geojsonPlaces = (await ensureCampusPlacePoints())
      .filter(place => scorePlaceQueryMatch(place.label, rawQuery) > 0)
      .map(place => ({ type: 'place', ...place }));

    const remotePlaces = await searchOpenStreetMapDestinations(rawQuery, 4);
    if (seq !== radarSuggestionSeq) return;

    const placeHits = dedupePlaceCandidates([
      ...campusPlaces,
      ...geojsonPlaces,
      ...(remotePlaces || [])
    ])
      .map(place => ({ ...place, type: 'place', _score: scorePlaceQueryMatch(place.label || '', rawQuery) + (place.source === 'campus' ? 10 : 0) }))
      .filter(place => place._score > 0)
      .sort((a, b) => b._score - a._score)
      .slice(0, 5);

    if (!peopleHits.length && !placeHits.length) {
      box.style.display = 'none';
      box.innerHTML = '';
      return;
    }

    const peopleMarkup = peopleHits.map(h => {
      const user = allExploreUsers.find(u => u.id === h.uid);
      const photo = user?.photoURL || null;
      const avatarHTML = photo ? `<img src="${photo}" alt="" class="radar-suggestion-avatar">` : `<div class="radar-suggestion-avatar">${initials(h.label)}</div>`;
      return `
        <button type="button" class="radar-suggestion-item" data-v="${esc(h.label)}" data-type="person" data-uid="${h.uid || ''}">
          ${avatarHTML}
          <div class="radar-suggestion-text">
            <span class="radar-suggestion-main">${esc(h.label)}</span>
            ${h.sub ? `<span class="radar-suggestion-sub">${esc(h.sub)}</span>` : ''}
          </div>
        </button>
      `;
    }).join('');

    const placeMarkup = placeHits.map(h => `
      <button type="button" class="radar-suggestion-item" data-v="${esc(h.label)}" data-type="place" data-lat="${Number(h.lat)}" data-lng="${Number(h.lng)}">
        <span class="radar-suggestion-icon">📍</span>
        <div class="radar-suggestion-text">
          <span class="radar-suggestion-main">${esc(h.label)}</span>
          ${h.sub ? `<span class="radar-suggestion-sub">${esc(h.sub)}</span>` : ''}
        </div>
      </button>
    `).join('');

    box.innerHTML = `${peopleHits.length ? `<div class="radar-suggestion-sub" style="padding:6px 12px 2px;font-weight:700;opacity:.75">People</div>${peopleMarkup}` : ''}${placeHits.length ? `<div class="radar-suggestion-sub" style="padding:8px 12px 2px;font-weight:700;opacity:.75">Places</div>${placeMarkup}` : ''}`;
    box.style.display = 'block';
    box.querySelectorAll('.radar-suggestion-item').forEach(btn => {
      let handled = false;
      const handleSelect = () => {
        if (handled) return;
        handled = true;
        const selected = btn.getAttribute('data-v') || '';
        const kind = btn.getAttribute('data-type') || '';
        const uid = btn.getAttribute('data-uid') || '';
        if (kind === 'person' && uid) {
          box.style.display = 'none';
          openProfile(uid);
          return;
        }
        if (kind === 'place') {
          const lat = Number(btn.getAttribute('data-lat'));
          const lng = Number(btn.getAttribute('data-lng'));
          if (!Number.isFinite(lat) || !Number.isFinite(lng)) return;
          const inp = $('#radar-search');
          if (inp) inp.value = selected;
          window._radarSearchQuery = selected;
          _exploreSearchQuery = selected;
          box.style.display = 'none';
          routeToMapPoint(lat, lng, selected || 'Destination');
          return;
        }
        const inp = $('#radar-search');
        if (inp) inp.value = selected;
        window._radarSearchQuery = selected;
        _exploreSearchQuery = selected;
        renderRadarSuggestions(selected);
        applyRadarFilter(selected);
      };
      btn.onclick = event => {
        event.preventDefault();
        handleSelect();
      };
      btn.onpointerdown = event => {
        event.preventDefault();
        handleSelect();
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
    syncExploreToggleButtons();
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



function destroyRadarMapInstance() {
  (_mapLibreMarkers || []).forEach(marker => { try { marker.remove(); } catch (_) {} });
  (_mapLibrePopups || []).forEach(popup => { try { popup.remove(); } catch (_) {} });
  _mapLibreMarkers = [];
  _mapLibrePopups = [];
  if (_mapLibreMap) {
    try { _mapLibreMap.remove(); } catch (_) {}
    _mapLibreMap = null;
  }
  if (_leafletMap) {
    try { _leafletMap.remove(); } catch (_) {}
    _leafletMap = null;
  }
}

function createMapLibreMarkerHTML(kind = 'user', options = {}) {
  if (kind === 'me') return `<div class="map-user-pin map-me-pin">${state.profile.photoURL ? `<img src="${state.profile.photoURL}">` : `<span>${initials(state.profile.displayName)}</span>`}</div>`;
  if (kind === 'user') return `<div class="map-user-pin ${options.cls || ''}">${options.photoURL ? `<img src="${options.photoURL}">` : `<span>${initials(options.label || 'User')}</span>`}</div>`;
  if (kind === 'cluster') return `<div class="map-user-cluster">${Number(options.count || 0)}</div>`;
  return `<div class="map-pin-wrap ${options.hasEvents ? 'has-events' : ''} ${options.floating ? 'is-floating' : ''}"><span class="map-pin-emoji">${options.emoji || '📍'}</span>${options.count ? `<span class="map-pin-count">${options.count}</span>` : ''}${options.label ? `<span class="map-pin-label">${esc(options.label)}</span>` : ''}</div>`;
}

function buildRadarUserClusters(users = [], center = null) {
  const grid = users.length > 42 ? 0.00042 : 0.00028;
  const buckets = new Map();
  users.forEach(u => {
    const coords = getUserCoords(u);
    if (!coords) return;
    const distanceKm = distanceKmBetween(center, coords);
    if (!Number.isFinite(distanceKm) || distanceKm > 5) return;
    const key = `${Math.round(coords.lat / grid)}:${Math.round(coords.lng / grid)}`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(u);
  });
  return Array.from(buckets.values());
}

function openRadarClusterPreview(encoded = '') {
  try {
    const users = JSON.parse(decodeURIComponent(encoded || '[]')) || [];
    if (!users.length) return;
    openModal(`
      <div class="modal-header"><h2>Nearby people</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
      <div class="modal-body" style="padding:16px;display:grid;gap:10px">${users.slice(0,18).map(u => `
        <button class="cluster-user-row" onclick="closeModal();showUserPreview('${u.id}')">
          ${avatar(u.displayName || 'User', u.photoURL || '', 'avatar-sm')}
          <span>${esc(u.displayName || 'User')}</span>
        </button>`).join('')}
      </div>
    `);
  } catch (_) {}
}

function attachMapLibreMarker(lng, lat, html, onClick) {
  if (!_mapLibreMap || typeof maplibregl === 'undefined') return null;
  const el = document.createElement('div');
  el.className = 'maplibre-custom-marker';
  el.innerHTML = html;
  if (onClick) {
    el.style.cursor = 'pointer';
    el.addEventListener('click', ev => {
      ev.stopPropagation();
      onClick();
    });
  }
  const marker = new maplibregl.Marker({ element: el, anchor: 'center' }).setLngLat([lng, lat]).addTo(_mapLibreMap);
  _mapLibreMarkers.push(marker);
  return marker;
}

async function fetchOsrmRoute(start, end) {
  try {
    const url = `https://router.project-osrm.org/route/v1/foot/${start.lng},${start.lat};${end.lng},${end.lat}?overview=full&geometries=geojson`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('route-http');
    const data = await res.json();
    const route = data?.routes?.[0];
    if (!route?.geometry?.coordinates?.length) throw new Error('route-empty');
    const points = route.geometry.coordinates.map(([lng, lat]) => ({ lat, lng }));
    const distanceKm = (route.distance || 0) / 1000;
    const rawWalkMinutes = Math.max(1, Math.round((route.duration || 0) / 60));
    const pacedWalkMinutes = Math.max(1, Math.round((distanceKm / 4.8) * 60));
    const impliedSpeedKmh = rawWalkMinutes > 0 ? (distanceKm / (rawWalkMinutes / 60)) : 0;
    const walkMinutes = impliedSpeedKmh > 8 ? pacedWalkMinutes : rawWalkMinutes;
    return {
      points,
      distanceKm,
      walkMinutes,
      direct: false,
      source: 'osrm'
    };
  } catch (_) {
    return null;
  }
}

async function buildBestRoute(startCoords = null, endCoords = null) {
  const safeStart = startCoords || getRadarCenterCoords();
  const safeEnd = endCoords || safeStart;
  const osrm = await fetchOsrmRoute(safeStart, safeEnd);
  if (osrm) return osrm;
  return buildCampusRoute(safeStart, safeEnd);
}

function syncRadarOverlayToUser() {
  const overlay = document.getElementById('radar-overlay');
  const wrap = document.querySelector('.radar-map-wrap');
  if (!overlay || !wrap) return;
  const rect = wrap.getBoundingClientRect();
  let x = Math.round(rect.width / 2);
  let y = Math.round(rect.height / 2);
  const myCoords = getUserCoords(state.profile) || getRadarCenterCoords();
  if (_mapLibreMap && typeof _mapLibreMap.project === 'function' && Number.isFinite(myCoords?.lat) && Number.isFinite(myCoords?.lng)) {
    try {
      const projected = _mapLibreMap.project([myCoords.lng, myCoords.lat]);
      const canvasRect = _mapLibreMap.getCanvas()?.getBoundingClientRect?.();
      if (projected && Number.isFinite(projected.x) && Number.isFinite(projected.y) && canvasRect) {
        x = Math.round(projected.x + (canvasRect.left - rect.left));
        y = Math.round(projected.y + (canvasRect.top - rect.top));
      }
    } catch (_) {}
  }
  overlay.style.left = `${x}px`;
  overlay.style.top = `${y}px`;
}

function setRadarRouteFilterState(active = false) {
  const dynamic = $('#radar-dynamic');
  if (dynamic) dynamic.classList.toggle('route-focus-active', !!active);
  const overlay = document.querySelector('.radar-overlay');
  if (overlay) overlay.style.display = active ? 'none' : 'block';
}

function renderRadarMap(moduleUsers, nearbyUsers, courseUsers, otherUsers, eventsByLoc) {
  requestAnimationFrame(() => {
    const el = document.getElementById('radar-map');
    if (!el) return;
    clearActiveMapRoute({ clearPending: false });
    destroyRadarMapInstance();

    const center = getRadarCenterCoords();
    const routeOnlyMode = !!_pendingMapRoute;
    setRadarRouteFilterState(routeOnlyMode);

    if (typeof maplibregl !== 'undefined') {
      const mapRef = new maplibregl.Map({
        container: 'radar-map',
        style: 'https://tiles.openfreemap.org/styles/bright',
        center: [center.lng, center.lat],
        zoom: routeOnlyMode ? 18 : 16.6,
        pitch: routeOnlyMode ? 56 : 48,
        bearing: -17.6,
        attributionControl: false,
        canvasContextAttributes: { antialias: true }
      });
      _mapLibreMap = mapRef;
      _leafletMap = mapRef;
      mapRef.addControl(new maplibregl.NavigationControl({ visualizePitch: true, showZoom: true, showCompass: true }), 'top-right');
      mapRef.on('load', () => {
        if (_mapLibreMap !== mapRef) return;
        mapRef.on('styleimagemissing', evt => {
          if (_mapLibreMap !== mapRef) return;
          const missingId = evt?.id;
          if (!missingId || mapRef.hasImage(missingId)) return;
          try {
            mapRef.addImage(missingId, { width: 1, height: 1, data: new Uint8Array([0, 0, 0, 0]) });
          } catch (_) {}
        });
        try {
          const layers = mapRef.getStyle().layers || [];
          const labelLayerId = (layers.find(l => l.type === 'symbol' && l.layout && l.layout['text-field']) || {}).id;
          mapRef.addSource('openfreemap-buildings', { type: 'vector', url: 'https://tiles.openfreemap.org/planet' });
          if (!mapRef.getLayer('3d-buildings')) {
            mapRef.addLayer({
              id: '3d-buildings',
              source: 'openfreemap-buildings',
              'source-layer': 'building',
              type: 'fill-extrusion',
              minzoom: 15,
              filter: ['!=', ['get', 'hide_3d'], true],
              paint: {
                'fill-extrusion-color': ['interpolate', ['linear'], ['coalesce', ['get', 'render_height'], 0], 0, '#d9d3c9', 60, '#c7b8a0', 120, '#b89e7a', 240, '#8f6f47'],
                'fill-extrusion-height': ['interpolate', ['linear'], ['zoom'], 15, 0, 16, ['coalesce', ['get', 'render_height'], 0]],
                'fill-extrusion-base': ['interpolate', ['linear'], ['zoom'], 15, 0, 16, ['coalesce', ['get', 'render_min_height'], 0]],
                'fill-extrusion-opacity': 0.9
              }
            }, labelLayerId);
          }
        } catch (_) {}

        if (_mapLibreMap !== mapRef) return;
        mountCampusGeoJsonLayers(mapRef).catch(() => {});
        attachMapLibreMarker(center.lng, center.lat, createMapLibreMarkerHTML('me'));
        syncRadarOverlayToUser();
        mapRef.on('move', syncRadarOverlayToUser);
        mapRef.on('render', syncRadarOverlayToUser);

        if (!routeOnlyMode && _pendingRadarPinnedPoint && Number.isFinite(_pendingRadarPinnedPoint.lat) && Number.isFinite(_pendingRadarPinnedPoint.lng)) {
          attachMapLibreMarker(_pendingRadarPinnedPoint.lng, _pendingRadarPinnedPoint.lat, createMapLibreMarkerHTML('pin', { emoji: '📌', hasEvents: true, floating: true, label: _pendingRadarPinnedPoint.label || 'Pinned' }), () => routeToMapPoint(_pendingRadarPinnedPoint.lat, _pendingRadarPinnedPoint.lng, _pendingRadarPinnedPoint.label || 'Pinned'));
          if (_mapLibreMap === mapRef) mapRef.flyTo({ center: [_pendingRadarPinnedPoint.lng, _pendingRadarPinnedPoint.lat], zoom: 17.9, speed: 0.8 });
          _pendingRadarPinnedPoint = null;
        }

        if (!routeOnlyMode) {
          CAMPUS_LOCATIONS.forEach(loc => {
            const evts = eventsByLoc[loc.id] || [];
            attachMapLibreMarker(loc.lng, loc.lat, createMapLibreMarkerHTML('pin', { emoji: loc.emoji, hasEvents: evts.length > 0, count: evts.length, label: loc.name, floating: true }), () => openLocationDetail(loc.id));
          });
        }

        const usersToPlot = routeOnlyMode ? [] : [...moduleUsers, ...nearbyUsers, ...courseUsers, ...otherUsers.slice(0, 8)];
        const plottedIds = new Set();
        const occupiedPoints = [{ lat: center.lat, lng: center.lng }];
        usersToPlot.forEach((u, i) => {
          const coords = getUserCoords(u);
          if (!coords) return;
          const distanceKm = distanceKmBetween(center, coords);
          if (distanceKm > 5) return;
          plottedIds.add(u.id);
          const displayPoint = resolveMapPoint(coords, occupiedPoints, center);
          const cls = u.proximity === 'module' ? 'pin-module' : (u.proximity === 'nearby' || u.proximity === 'course') ? 'pin-campus' : 'pin-far';
          attachMapLibreMarker(displayPoint.lng, displayPoint.lat, createMapLibreMarkerHTML('user', { photoURL: u.photoURL, label: u.displayName, cls }), () => showUserPreview(u.id));
        });
        usersToPlot.filter(u => !plottedIds.has(u.id)).forEach((u, i) => {
          const baseR = u.proximity === 'module' ? 0.001 : u.proximity === 'nearby' ? 0.0018 : u.proximity === 'course' ? 0.0025 : 0.004;
          const angle = (i / Math.max(1, usersToPlot.length)) * Math.PI * 2 + (i * 0.7);
          const rawPoint = { lat: center.lat + Math.cos(angle) * baseR * (0.6 + Math.random() * 0.8), lng: center.lng + Math.sin(angle) * baseR * (0.6 + Math.random() * 0.8) };
          const displayPoint = resolveMapPoint(rawPoint, occupiedPoints, center);
          const cls = u.proximity === 'module' ? 'pin-module' : (u.proximity === 'nearby' || u.proximity === 'course') ? 'pin-campus' : 'pin-far';
          attachMapLibreMarker(displayPoint.lng, displayPoint.lat, createMapLibreMarkerHTML('user', { photoURL: u.photoURL, label: u.displayName, cls }), () => showUserPreview(u.id));
        });

        if (_pendingMapRoute) drawPendingMapRouteOnCurrentMap();
        else updateMapRoutePanel(null);
      });
      return;
    }

    if (typeof L === 'undefined') return;
  });
}

function renderRadarDots(users, radius, type) {
  if (!users.length) return '';
  return users.slice(0, 8).map((u, i) => {
    const angle = (i / Math.min(users.length, 8)) * Math.PI * 2 - Math.PI / 2;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    const bg = colorFor(u.displayName);
    return `<div class="radar-dot ${type}" style="transform:translate(${x}px,${y}px);background:${bg}" onclick="showUserPreview('${u.id}')" title="${esc(u.displayName)}">
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
    const allowsAnon = allowAnonymousDMsFor(user);
    const modules = (user.modules || []).slice(0, 3);
    const myCoords = getUserCoords(state.profile);
    const theirCoords = getUserCoords(user);
    const distanceText = !isMe && myCoords && theirCoords ? formatDistanceText(distanceKmBetween(myCoords, theirCoords)) : '';
    const detailChips = [
      user.year ? `🎓 ${esc(user.year)}` : '',
      distanceText ? `🧭 ${esc(distanceText)}` : ''
    ].filter(Boolean);
    openModal(`
      <div class="modal-body" style="text-align:center;padding:24px">
        <div style="margin-bottom:12px">${avatar(user.displayName, user.photoURL, 'avatar-xl')}</div>
        <div style="font-size:18px;font-weight:700;margin-bottom:4px">${esc(user.displayName)}</div>
        <div style="font-size:13px;color:var(--text-secondary);margin-bottom:4px">${esc(user.major || 'Student')}${user.university ? ' · ' + esc(user.university) : ''}${user.gender ? ' · ' + esc(user.gender) : ''}</div>
        ${detailChips.length ? `<div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin:10px 0 12px">${detailChips.map(label => `<span style="display:inline-flex;align-items:center;gap:6px;padding:7px 10px;border-radius:999px;background:var(--bg-tertiary);font-size:12px;color:var(--text-secondary)">${label}</span>`).join('')}</div>` : ''}
        ${user.bio ? `<p style="font-size:13px;color:var(--text-secondary);margin-bottom:12px;line-height:1.4">${esc(user.bio)}</p>` : ''}
        ${modules.length ? `<div style="display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin-bottom:16px">${modules.map(m => `<span class="module-chip">${esc(m)}</span>`).join('')}</div>` : ''}
        <div style="display:flex;gap:8px;justify-content:center">
          ${isMe ? '' : isFriend
            ? `<button class="btn-primary" onclick="closeModal();startChat('${uid}','${esc(user.displayName)}','${user.photoURL || ''}')">Message</button>`
            : `${allowsAnon ? `<button class="btn-outline anon-msg-btn" onclick="closeModal();startAnonChat('${uid}','${esc(user.displayName)}','${user.photoURL || ''}', true)">👻 Anonymous</button>` : `<button class="btn-outline" disabled style="opacity:0.6">Anon Off</button>`}
               ${isPending
                 ? `<button class="btn-outline" disabled style="opacity:0.6">Pending…</button>`
                 : `<button class="btn-outline" onclick="closeModal();sendFriendRequest('${uid}','${esc(user.displayName)}','${user.photoURL || ''}')">Add Friend</button>`}`}
          <button class="btn-secondary" onclick="closeModal();openProfile('${uid}')">View Profile</button>
        </div>
      </div>
    `);
  } catch (e) { toast('Could not load user'); console.error(e); }
}

function renderListView(initialQuery = _exploreSearchQuery || window._radarSearchQuery || '', initialFilter = 'all') {
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
  renderExploreGrid(_exploreSearchQuery, initialFilter || 'all');
  let timer;
  $('#explore-search')?.addEventListener('input', e => {
    clearTimeout(timer); timer = setTimeout(() => {
      _exploreSearchQuery = e.target.value;
      renderExploreGrid(_exploreSearchQuery, document.querySelector('#explore-body .filter-chips .chip.active')?.dataset.f || 'all');
    }, 160);
  });
  $$('#explore-body .filter-chips .chip').forEach(ch => {
    ch.onclick = () => {
      $$('#explore-body .filter-chips .chip').forEach(c2 => c2.classList.remove('active'));
      ch.classList.add('active');
      renderExploreGrid($('#explore-search')?.value, ch.dataset.f);
    };
  });
  const presetChip = document.querySelector(`#explore-body .filter-chips .chip[data-f="${initialFilter || 'all'}"]`);
  if (presetChip) {
    $$('#explore-body .filter-chips .chip').forEach(c2 => c2.classList.remove('active'));
    presetChip.classList.add('active');
  }
}

async function renderExploreGrid(query = '', filter = 'all') {
  const grid = $('#explore-grid'); if (!grid) return;
  const blockedUsers = new Set(state.profile.blockedUsers || []);
  const blockedBy = new Set(state.profile.blockedBy || []);
  let users = [...allExploreUsers];

  if (query) {
    const rawQuery = String(query || '').trim();
    const q = rawQuery.toLowerCase();
    users = users
      .map(u => ({ ...u, _searchScore: userSearchScore(u, rawQuery) }))
      .filter(u => u._searchScore > 0)
      .sort((a, b) => b._searchScore - a._searchScore || (b.affinityScore || 0) - (a.affinityScore || 0));
    // If no local matches and query is 2+ chars, search Firestore directly
    if (!users.length && q.length >= 2) {
      grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><span class="inline-spinner" style="width:24px;height:24px"></span></div>';
      try {
        const snap = await db.collection('users').get();
        const uid = state.user.uid;
        users = snap.docs
          .map(d => ({ id: d.id, ...d.data() }))
          .filter(u => u.id !== uid && !blockedUsers.has(u.id) && !blockedBy.has(u.id))
          .map(u => ({ ...u, _searchScore: userSearchScore(u, rawQuery) }))
          .filter(u => u._searchScore > 0)
          .sort((a, b) => b._searchScore - a._searchScore || (a.displayName || '').localeCompare(b.displayName || ''));
      } catch (_) {}
    }
  }
  if (filter === 'nearby') users = users.filter(u => u.proximity === 'nearby');
  else if (filter === 'module') users = users.filter(u => u.sharedModules?.length > 0);
  else if (filter === 'course') users = users.filter(u => u.major === state.profile.major);

  if (!users.length) {
    grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><h3>No matches</h3><p>Try a different search</p></div>';
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

const CAMPUS_ROUTE_EDGES = [
  ['library', 'lab-block'],
  ['library', 'main-hall'],
  ['lab-block', 'cs-building'],
  ['cs-building', 'amphitheatre'],
  ['cs-building', 'cafeteria'],
  ['main-hall', 'amphitheatre'],
  ['main-hall', 'student-center'],
  ['main-hall', 'admin'],
  ['student-center', 'admin'],
  ['student-center', 'sports-complex'],
  ['student-center', 'res-halls'],
  ['admin', 'quad'],
  ['amphitheatre', 'quad'],
  ['quad', 'cafeteria'],
  ['quad', 'res-halls'],
  ['sports-complex', 'res-halls'],
  ['res-halls', 'parking']
];

let _campusGeoJsonCache = null;
let _campusGeoJsonLoadPromise = null;

async function loadCampusGeoJson() {
  if (_campusGeoJsonCache) return _campusGeoJsonCache;
  if (_campusGeoJsonLoadPromise) return _campusGeoJsonLoadPromise;
  _campusGeoJsonLoadPromise = (async () => {
    const candidates = ['NWUCAMP.geojson', './NWUCAMP.geojson', '/NWUCAMP.geojson'];
    for (const path of candidates) {
      try {
        const res = await fetch(path, { cache: 'force-cache' });
        if (!res.ok) continue;
        const data = await res.json();
        if (data && Array.isArray(data.features)) return data;
      } catch (_) {}
    }
    return null;
  })()
    .then(data => {
      _campusGeoJsonCache = data && Array.isArray(data.features) ? data : null;
      return _campusGeoJsonCache;
    })
    .catch(() => null)
    .finally(() => { _campusGeoJsonLoadPromise = null; });
  return _campusGeoJsonLoadPromise;
}

function normalizePlaceKey(value = '') {
  return String(value || '')
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function scorePlaceQueryMatch(label = '', query = '') {
  const cleanLabel = normalizePlaceKey(label);
  const cleanQuery = normalizePlaceKey(query);
  if (!cleanLabel || !cleanQuery) return 0;
  if (cleanLabel === cleanQuery) return 120;
  if (cleanLabel.startsWith(cleanQuery)) return 90;
  if (cleanLabel.includes(cleanQuery)) return 70;
  const parts = cleanQuery.split(' ').filter(Boolean);
  if (!parts.length) return 0;
  const matched = parts.filter(part => cleanLabel.includes(part)).length;
  if (!matched) return 0;
  return 40 + (matched * 12);
}

async function ensureCampusPlacePoints() {
  if (_campusPlacePointsCache.length) return _campusPlacePointsCache;
  const geojson = await loadCampusGeoJson();
  if (!geojson) {
    _campusPlacePointsCache = [];
    return _campusPlacePointsCache;
  }
  const labels = labelPointsFromPolygons(geojson)?.features || [];
  _campusPlacePointsCache = labels
    .map(feature => {
      const coords = feature?.geometry?.coordinates || [];
      const lng = Number(coords[0]);
      const lat = Number(coords[1]);
      const label = String(feature?.properties?.name || '').trim();
      if (!Number.isFinite(lat) || !Number.isFinite(lng) || !label) return null;
      return { label, sub: 'Campus building', lat, lng, source: 'geojson' };
    })
    .filter(Boolean);
  return _campusPlacePointsCache;
}

function dedupePlaceCandidates(candidates = []) {
  const seen = new Set();
  const deduped = [];
  (candidates || []).forEach(item => {
    const lat = Number(item?.lat);
    const lng = Number(item?.lng);
    const label = String(item?.label || '').trim();
    if (!Number.isFinite(lat) || !Number.isFinite(lng) || !label) return;
    const key = `${normalizePlaceKey(label)}|${lat.toFixed(5)}|${lng.toFixed(5)}`;
    if (seen.has(key)) return;
    seen.add(key);
    deduped.push({ ...item, lat, lng, label });
  });
  return deduped;
}

function reverseGeocodeSummary(address = {}) {
  const road = address.road || address.pedestrian || address.footway || address.path || '';
  const area = address.suburb || address.neighbourhood || address.city_district || address.town || address.city || address.village || '';
  return [road, area].filter(Boolean).join(', ');
}

async function reverseGeocodeLabel(lat, lng) {
  const key = `${Number(lat).toFixed(5)},${Number(lng).toFixed(5)}`;
  if (_reverseGeocodeCache.has(key)) return _reverseGeocodeCache.get(key);
  try {
    const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&addressdetails=1&zoom=18&lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lng)}&accept-language=en`;
    const res = await fetch(url, { headers: { Accept: 'application/json' } });
    if (!res.ok) throw new Error(`reverse-${res.status}`);
    const data = await res.json();
    const address = data?.address || {};
    const primary = String(
      address.building
      || address.amenity
      || address.attraction
      || address.university
      || address.college
      || address.road
      || address.pedestrian
      || address.neighbourhood
      || data?.name
      || ''
    ).trim();
    const sub = reverseGeocodeSummary(address);
    const out = {
      label: primary || 'Pinned location',
      sub: sub || ''
    };
    _reverseGeocodeCache.set(key, out);
    return out;
  } catch (_) {
    const fallback = { label: 'Pinned location', sub: '' };
    _reverseGeocodeCache.set(key, fallback);
    return fallback;
  }
}

async function resolveLocationPinLabel(lat, lng) {
  const point = { lat: Number(lat), lng: Number(lng) };
  if (!Number.isFinite(point.lat) || !Number.isFinite(point.lng)) {
    return { label: 'Pinned location', sub: '' };
  }

  const campusNearest = CAMPUS_LOCATIONS
    .map(loc => ({
      label: loc.name,
      sub: 'Campus location',
      lat: Number(loc.lat),
      lng: Number(loc.lng),
      distanceKm: distanceKmBetween(point, { lat: Number(loc.lat), lng: Number(loc.lng) })
    }))
    .sort((a, b) => a.distanceKm - b.distanceKm)[0];

  const buildingNearest = (await ensureCampusPlacePoints())
    .map(place => ({
      ...place,
      distanceKm: distanceKmBetween(point, { lat: place.lat, lng: place.lng })
    }))
    .sort((a, b) => a.distanceKm - b.distanceKm)[0];

  const bestKnown = [buildingNearest, campusNearest]
    .filter(Boolean)
    .sort((a, b) => a.distanceKm - b.distanceKm)[0] || null;

  if (buildingNearest && Number.isFinite(buildingNearest.distanceKm) && buildingNearest.distanceKm <= 0.2) {
    return {
      label: buildingNearest.label || 'Pinned location',
      sub: buildingNearest.sub || 'Campus building'
    };
  }

  const reverse = await reverseGeocodeLabel(point.lat, point.lng);
  const reverseKey = normalizePlaceKey(reverse?.label || '');
  const reverseGeneric = !reverseKey || reverseKey === 'pinned location' || reverseKey === 'location';

  if (!reverseGeneric && reverse?.label) {
    return {
      label: reverse.label,
      sub: reverse.sub || ''
    };
  }

  if (campusNearest && Number.isFinite(campusNearest.distanceKm) && campusNearest.distanceKm <= 0.1) {
    return {
      label: campusNearest.label || 'Pinned location',
      sub: campusNearest.sub || ''
    };
  }

  if (bestKnown && Number.isFinite(bestKnown.distanceKm) && bestKnown.distanceKm <= 0.2) {
    return {
      label: bestKnown.label || 'Pinned location',
      sub: bestKnown.sub || ''
    };
  }

  return {
    label: reverse?.label || 'Pinned location',
    sub: reverse?.sub || ''
  };
}

async function searchOpenStreetMapDestinations(queryRaw = '', limit = 4) {
  const query = normalizePlaceKey(queryRaw);
  if (!query || query.length < 3) return [];
  if (_destinationSearchCache.has(query)) return _destinationSearchCache.get(query) || [];
  if (_destinationSearchInflight.has(query)) return _destinationSearchInflight.get(query);

  const task = (async () => {
    try {
      const url = `https://nominatim.openstreetmap.org/search?format=jsonv2&addressdetails=1&limit=${Math.max(1, Math.min(8, limit))}&countrycodes=za&q=${encodeURIComponent(queryRaw)}`;
      const res = await fetch(url, { headers: { Accept: 'application/json' } });
      if (!res.ok) throw new Error(`search-${res.status}`);
      const rows = await res.json();
      const mapped = (rows || []).map(row => {
        const lat = Number(row?.lat);
        const lng = Number(row?.lon);
        if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
        const address = row?.address || {};
        const label = String(
          address.building
          || address.amenity
          || address.attraction
          || address.road
          || address.neighbourhood
          || row?.name
          || (row?.display_name || '').split(',')[0]
          || ''
        ).trim();
        if (!label) return null;
        const sub = reverseGeocodeSummary(address) || String(row?.display_name || '').split(',').slice(1, 3).join(', ').trim();
        return { type: 'place', label, sub, lat, lng, source: 'nominatim' };
      }).filter(Boolean);
      const deduped = dedupePlaceCandidates(mapped).slice(0, limit);
      _destinationSearchCache.set(query, deduped);
      return deduped;
    } catch (_) {
      _destinationSearchCache.set(query, []);
      return [];
    } finally {
      _destinationSearchInflight.delete(query);
    }
  })();

  _destinationSearchInflight.set(query, task);
  return task;
}

function getCampusExtent(geojson) {
  let minLng = Infinity, minLat = Infinity, maxLng = -Infinity, maxLat = -Infinity;
  (geojson?.features || []).forEach(feature => {
    const coords = feature?.geometry?.coordinates || [];
    coords.forEach(poly => poly.forEach(ring => ring.forEach(coord => {
      const [lng, lat] = coord || [];
      if (!Number.isFinite(lng) || !Number.isFinite(lat)) return;
      if (lng < minLng) minLng = lng;
      if (lat < minLat) minLat = lat;
      if (lng > maxLng) maxLng = lng;
      if (lat > maxLat) maxLat = lat;
    })));
  });
  if (!Number.isFinite(minLng)) return null;
  return { center: [(minLng + maxLng) / 2, (minLat + maxLat) / 2], bounds: [[minLng, minLat], [maxLng, maxLat]] };
}

function polygonCentroid(ring = []) {
  let area = 0, cx = 0, cy = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    const [x1, y1] = ring[i] || [];
    const [x2, y2] = ring[i + 1] || [];
    if (![x1, y1, x2, y2].every(Number.isFinite)) continue;
    const f = x1 * y2 - x2 * y1;
    area += f;
    cx += (x1 + x2) * f;
    cy += (y1 + y2) * f;
  }
  area *= 0.5;
  return area ? [cx / (6 * area), cy / (6 * area)] : (ring[0] || [0, 0]);
}

function labelPointsFromPolygons(geojson) {
  return {
    type: 'FeatureCollection',
    features: (geojson?.features || []).map((feature, index) => {
      const polys = feature?.geometry?.coordinates || [];
      let bestRing = null;
      let bestSize = -Infinity;
      polys.forEach(poly => {
        const outerRing = poly?.[0];
        if (!outerRing || outerRing.length < 4) return;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        outerRing.forEach(([x, y]) => {
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        });
        const boxSize = (maxX - minX) * (maxY - minY);
        if (boxSize > bestSize) { bestSize = boxSize; bestRing = outerRing; }
      });
      const center = polygonCentroid(bestRing || []);
      return {
        type: 'Feature',
        properties: { name: feature?.properties?.name || `Building ${index + 1}` },
        geometry: { type: 'Point', coordinates: center }
      };
    })
  };
}

async function mountCampusGeoJsonLayers(map) {
  if (!map || typeof map.getStyle !== 'function') return;
  const geojson = await loadCampusGeoJson();
  if (!geojson || map.getSource('campus-geojson')) return;
  const labelPoints = labelPointsFromPolygons(geojson);
  map.addSource('campus-geojson', { type: 'geojson', data: geojson });
  map.addSource('campus-label-points', { type: 'geojson', data: labelPoints });
  map.addLayer({
    id: 'campus-fill',
    type: 'fill',
    source: 'campus-geojson',
    paint: {
      'fill-color': 'rgba(108,92,231,0)',
      'fill-opacity': 0,
      'fill-outline-color': 'rgba(108,92,231,0)'
    }
  });
  map.addLayer({
    id: 'campus-outline',
    type: 'line',
    source: 'campus-geojson',
    paint: { 'line-color': 'rgba(108,92,231,0)', 'line-opacity': 0, 'line-width': 1 }
  });
  map.addLayer({
    id: 'campus-name-labels',
    type: 'symbol',
    source: 'campus-label-points',
    minzoom: 16,
    layout: {
      'text-field': ['get', 'name'],
      'text-size': ['interpolate', ['linear'], ['zoom'], 16, 11, 18.5, 16],
      'text-font': ['Open Sans Bold'],
      'text-anchor': 'center',
      'text-allow-overlap': true,
      'text-ignore-placement': true
    },
    paint: {
      'text-color': '#171717',
      'text-halo-color': 'rgba(255,255,255,0.96)',
      'text-halo-width': 2.2
    }
  });
  map.on('click', 'campus-name-labels', e => {
    const feature = e.features && e.features[0];
    if (!feature) return;
    const [lng, lat] = feature.geometry.coordinates || [];
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return;
    const name = feature.properties?.name || 'Building';
    const encodedName = encodeURIComponent(name || 'Building');
    new maplibregl.Popup({ closeButton: false, offset: 18 })
      .setLngLat([lng, lat])
      .setHTML(`<div class="campus-map-popup"><div class="campus-map-popup-title">${esc(name)}</div><div class="campus-map-popup-sub">Choose an action</div><div class="campus-map-popup-actions"><button class="map-popup-btn" onclick="routeToMapPointEncoded(${lat},${lng}, '${encodedName}')">Route here</button><button class="map-popup-btn secondary" onclick="closeModal();openCreateEventAtMapPointEncoded(${lat},${lng}, '${encodedName}')">Create event here</button></div></div>`)
      .addTo(map);
  });
  map.on('mouseenter', 'campus-name-labels', () => { map.getCanvas().style.cursor = 'pointer'; });
  map.on('mouseleave', 'campus-name-labels', () => { map.getCanvas().style.cursor = ''; });
}

function setPendingMapRoute(target = {}, options = {}) {
  const lat = Number(target.lat);
  const lng = Number(target.lng);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return false;
  _pendingMapRoute = {
    target: { lat, lng },
    label: options.label || 'Destination',
    targetId: options.targetId || '',
    source: options.source || 'manual'
  };
  return true;
}

function calculateRouteDistance(points = []) {
  let total = 0;
  for (let i = 1; i < points.length; i++) total += distanceKmBetween(points[i - 1], points[i]);
  return total;
}

function dedupeRoutePoints(points = []) {
  return points.filter((point, index, arr) => {
    if (!point || !Number.isFinite(point.lat) || !Number.isFinite(point.lng)) return false;
    if (index === 0) return true;
    const prev = arr[index - 1];
    return !prev || Math.abs(prev.lat - point.lat) > 0.000001 || Math.abs(prev.lng - point.lng) > 0.000001;
  });
}

function nearestCampusNode(coords = null, maxDistanceKm = 0.75) {
  if (!coords) return null;
  let best = null;
  CAMPUS_LOCATIONS.forEach(loc => {
    const distanceKm = distanceKmBetween(coords, loc);
    if (!best || distanceKm < best.distanceKm) best = { loc, distanceKm };
  });
  if (!best || best.distanceKm > maxDistanceKm) return null;
  return best;
}

function shortestCampusPath(startId = '', endId = '') {
  if (!startId || !endId) return [];
  if (startId === endId) return [getCampusLocationById(startId)].filter(Boolean);
  const graph = {};
  CAMPUS_LOCATIONS.forEach(loc => { graph[loc.id] = []; });
  CAMPUS_ROUTE_EDGES.forEach(([left, right]) => {
    const a = getCampusLocationById(left);
    const b = getCampusLocationById(right);
    if (!a || !b) return;
    const distanceKm = distanceKmBetween(a, b);
    graph[left].push({ id: right, distanceKm });
    graph[right].push({ id: left, distanceKm });
  });

  const distances = {};
  const previous = {};
  const remaining = new Set(Object.keys(graph));
  Object.keys(graph).forEach(id => { distances[id] = Number.POSITIVE_INFINITY; });
  distances[startId] = 0;

  while (remaining.size) {
    let current = '';
    let bestDistance = Number.POSITIVE_INFINITY;
    remaining.forEach(id => {
      if (distances[id] < bestDistance) {
        bestDistance = distances[id];
        current = id;
      }
    });
    if (!current) break;
    remaining.delete(current);
    if (current === endId) break;
    (graph[current] || []).forEach(edge => {
      if (!remaining.has(edge.id)) return;
      const nextDistance = distances[current] + edge.distanceKm;
      if (nextDistance < distances[edge.id]) {
        distances[edge.id] = nextDistance;
        previous[edge.id] = current;
      }
    });
  }

  if (!Number.isFinite(distances[endId])) return [];
  const path = [];
  let cursor = endId;
  while (cursor) {
    path.unshift(cursor);
    if (cursor === startId) break;
    cursor = previous[cursor];
  }
  return path.map(getCampusLocationById).filter(Boolean);
}

function buildCampusRoute(startCoords = null, endCoords = null) {
  const safeStart = startCoords || getRadarCenterCoords();
  const safeEnd = endCoords || safeStart;
  const startNode = nearestCampusNode(safeStart);
  const endNode = nearestCampusNode(safeEnd);
  let direct = true;
  let points = [safeStart, safeEnd];

  if (startNode && endNode) {
    const pathNodes = shortestCampusPath(startNode.loc.id, endNode.loc.id);
    if (pathNodes.length) {
      direct = false;
      points = dedupeRoutePoints([
        safeStart,
        ...pathNodes.map(node => ({ lat: node.lat, lng: node.lng })),
        safeEnd
      ]);
    }
  }

  const distanceKm = calculateRouteDistance(points);
  const walkMinutes = Math.max(1, Math.round((distanceKm / 4.8) * 60));
  return { points, distanceKm, walkMinutes, direct };
}

function updateMapRoutePanel(summary = null) {
  const host = document.getElementById('map-route-panel');
  if (!host) return;
  if (!summary) {
    host.style.display = 'none';
    host.innerHTML = '';
    return;
  }
  const usingCampusFallback = !getUserCoords(state.profile);
  host.innerHTML = `
    <div class="map-route-card">
      <div class="map-route-copy">
        <div class="map-route-title">Route to ${esc(summary.label || 'destination')}</div>
        <div class="map-route-meta">
          <span>🧭 ${esc(formatDistanceText(summary.distanceKm) || 'Nearby')}</span>
          <span>🚶 ${summary.walkMinutes} min walk</span>
          <span>${summary.source === 'osrm' ? '🗺 Real route' : (summary.direct ? '↗ Direct line' : '🛣 Campus route')}</span>
        </div>
        ${usingCampusFallback ? '<div class="map-route-note">Using campus center. Save GPS in profile for better routing.</div>' : ''}
      </div>
      <div class="map-route-actions">
        <button class="btn-primary btn-sm" onclick="clearMapRoute()">Clear</button>
      </div>
    </div>
  `;
  host.style.display = 'block';
}

function clearActiveMapRoute(options = {}) {
  const { clearPending = true } = options;
  if (_mapLibreMap && _activeMapRouteLayer === 'route-line') {
    try { if (_mapLibreMap.getLayer('route-line')) _mapLibreMap.removeLayer('route-line'); } catch (_) {}
    try { if (_mapLibreMap.getSource('route-line')) _mapLibreMap.removeSource('route-line'); } catch (_) {}
  } else if (_leafletMap && _activeMapRouteLayer) {
    try { _leafletMap.removeLayer(_activeMapRouteLayer); } catch (_) {}
  }
  (_activeMapRouteMarkers || []).forEach(marker => {
    try { marker?.remove ? marker.remove() : _leafletMap?.removeLayer(marker); } catch (_) {}
  });
  _activeMapRouteLayer = null;
  _activeMapRouteMarkers = [];
  _activeMapRouteSummary = null;
  if (clearPending) _pendingMapRoute = null;
  setRadarRouteFilterState(false);
  updateMapRoutePanel(null);
}

function clearMapRoute() {
  clearActiveMapRoute({ clearPending: true });
}

async function drawPendingMapRouteOnCurrentMap() {
  const mapLibreRef = _mapLibreMap;
  const leafletRef = _leafletMap;
  if (!_pendingMapRoute || (!leafletRef && !mapLibreRef)) {
    if (!_pendingMapRoute) updateMapRoutePanel(null);
    return;
  }
  clearActiveMapRoute({ clearPending: false });
  const start = getUserCoords(state.profile) || getRadarCenterCoords();
  const route = await buildBestRoute(start, _pendingMapRoute.target);
  if (!_pendingMapRoute) return;
  if ((mapLibreRef && _mapLibreMap !== mapLibreRef) || (leafletRef && _leafletMap !== leafletRef)) {
    setTimeout(() => ensurePendingRouteVisible(1), 120);
    return;
  }
  const coordinates = route.points.map(point => [point.lng, point.lat]);
  if (!coordinates.length) return;

  if (mapLibreRef && _mapLibreMap === mapLibreRef && typeof maplibregl !== 'undefined') {
    try {
      if (mapLibreRef.getLayer('route-line')) mapLibreRef.removeLayer('route-line');
      if (mapLibreRef.getSource('route-line')) mapLibreRef.removeSource('route-line');
      mapLibreRef.addSource('route-line', {
        type: 'geojson',
        data: { type: 'Feature', geometry: { type: 'LineString', coordinates } }
      });
      mapLibreRef.addLayer({
        id: 'route-line',
        type: 'line',
        source: 'route-line',
        paint: {
          'line-color': '#6C5CE7',
          'line-width': ['interpolate',['linear'],['zoom'],15,5,18,8],
          'line-opacity': 0.92
        }
      });
      _activeMapRouteLayer = 'route-line';
      const startMarker = attachMapLibreMarker(start.lng, start.lat, `<div class="route-endpoint route-start">You</div>`);
      const endMarker = attachMapLibreMarker(_pendingMapRoute.target.lng, _pendingMapRoute.target.lat, `<div class="route-endpoint route-end">${esc(_pendingMapRoute.label || 'Destination')}</div>`);
      _activeMapRouteMarkers = [startMarker, endMarker].filter(Boolean);
      const bounds = coordinates.reduce((b, coord) => b.extend(coord), new maplibregl.LngLatBounds(coordinates[0], coordinates[0]));
      if (!bounds.isEmpty()) mapLibreRef.fitBounds(bounds, { padding: 54, duration: 700, pitch: 56 });
    } catch (_) {
      setTimeout(() => ensurePendingRouteVisible(1), 120);
      return;
    }
  } else if (leafletRef && _leafletMap === leafletRef && typeof L !== 'undefined') {
    const latLngs = route.points.map(point => [point.lat, point.lng]);
    _activeMapRouteLayer = L.polyline(latLngs, { color: '#6C5CE7', weight: 5, opacity: 0.9, lineCap: 'round', lineJoin: 'round', dashArray: route.direct ? '10 10' : null }).addTo(leafletRef);
    const startMarker = L.circleMarker([start.lat, start.lng], { radius: 7, color: '#111827', weight: 2, fillColor: '#22C55E', fillOpacity: 1 }).addTo(leafletRef);
    const endMarker = L.circleMarker([_pendingMapRoute.target.lat, _pendingMapRoute.target.lng], { radius: 7, color: '#111827', weight: 2, fillColor: '#8B5CF6', fillOpacity: 1 }).addTo(leafletRef);
    _activeMapRouteMarkers = [startMarker, endMarker];
    const bounds = L.latLngBounds(latLngs);
    if (bounds.isValid()) leafletRef.fitBounds(bounds.pad(0.18), { animate: true, duration: 0.45 });
  } else {
    setTimeout(() => ensurePendingRouteVisible(1), 120);
    return;
  }

  _activeMapRouteSummary = { ...route, label: _pendingMapRoute.label || 'Destination' };
  setRadarRouteFilterState(true);
  updateMapRoutePanel(_activeMapRouteSummary);
}

function ensurePendingRouteVisible(attempt = 0) {
  if (!_pendingMapRoute) return;
  if (attempt > 24) return;
  if (state.page !== 'explore') {
    setTimeout(() => ensurePendingRouteVisible(attempt + 1), 120);
    return;
  }
  if (!_mapLibreMap && !_leafletMap) {
    renderExploreView();
    setTimeout(() => ensurePendingRouteVisible(attempt + 1), 120);
    return;
  }
  drawPendingMapRouteOnCurrentMap().catch(() => {
    setTimeout(() => ensurePendingRouteVisible(attempt + 1), 120);
  });
}

function openCreateEventAtMapPoint(lat, lng, label = 'Pinned location') {
  const name = (label || 'Pinned location').trim();
  navigate('create');
  setTimeout(() => {
    const title = document.getElementById('create-event-title') || document.getElementById('event-title') || document.querySelector('input[placeholder*="event" i]');
    const loc = document.getElementById('create-event-location') || document.getElementById('event-location') || document.querySelector('input[placeholder*="location" i]');
    if (loc && !loc.value) loc.value = name;
    if (title && !title.value) title.value = `${name} meetup`;
    window._pendingRadarPinnedPoint = { lat, lng, label: name };
  }, 80);
}

function routeToMapPoint(lat, lng, label = 'Destination') {
  const safeLat = Number(lat);
  const safeLng = Number(lng);
  if (!Number.isFinite(safeLat) || !Number.isFinite(safeLng)) return toast('Could not build route');
  closeModal();
  closeChatPlusMenus();
  if (!setPendingMapRoute({ lat: safeLat, lng: safeLng }, { label, source: 'manual' })) return toast('Could not build route');
  exploreView = 'radar';
  if (!document.getElementById('app')?.classList.contains('active')) showScreen('app');
  if (state.page !== 'explore') navigate('explore');
  ensurePendingRouteVisible(0);
  setTimeout(() => ensurePendingRouteVisible(1), 180);
}

function routeToMapPointEncoded(lat, lng, encodedLabel = '') {
  let label = 'Destination';
  try {
    label = decodeURIComponent(String(encodedLabel || '')) || 'Destination';
  } catch (_) {}
  routeToMapPoint(lat, lng, label);
}

function openCreateEventAtMapPointEncoded(lat, lng, encodedLabel = '') {
  let label = 'Pinned location';
  try {
    label = decodeURIComponent(String(encodedLabel || '')) || 'Pinned location';
  } catch (_) {}
  openCreateEventAtMapPoint(lat, lng, label);
}

function routeToCampusLocation(locationId = '') {
  const loc = getCampusLocationById(locationId);
  if (!loc) return toast('Location not found');
  closeModal();
  closeChatPlusMenus();
  routeToMapPoint(loc.lat, loc.lng, loc.name || 'Destination');
}

let allCampusEvents = [];

function eventStartTimeMs(eventDoc = {}) {
  const dateRaw = (eventDoc.date || '').trim();
  if (!dateRaw) return Number.POSITIVE_INFINITY;
  const timeRaw = (eventDoc.time || '').trim();
  const iso = /^\d{4}-\d{2}-\d{2}$/.test(dateRaw)
    ? `${dateRaw}T${timeRaw || '23:59'}:00`
    : `${dateRaw} ${timeRaw || '23:59'}`;
  const parsed = new Date(iso).getTime();
  return Number.isFinite(parsed) ? parsed : Number.POSITIVE_INFINITY;
}

function eventGroupId(eventId = '') {
  return `event-${eventId}`;
}

async function upsertEventChatGroup(eventData = {}) {
  const eventId = eventData.id || '';
  if (!eventId) return;
  const members = Array.from(new Set((eventData.going || []).filter(Boolean)));
  if (!members.length) {
    await db.collection('groups').doc(eventGroupId(eventId)).delete().catch(() => {});
    return;
  }

  const userDocs = await Promise.all(members.map(uid => db.collection('users').doc(uid).get().catch(() => null)));
  const memberNames = {};
  const memberPhotos = {};
  userDocs.forEach((doc, idx) => {
    const uid = members[idx];
    if (!uid) return;
    if (doc?.exists) {
      const data = doc.data() || {};
      memberNames[uid] = data.displayName || 'Student';
      memberPhotos[uid] = data.photoURL || '';
    } else if (uid === state.user?.uid) {
      memberNames[uid] = state.profile?.displayName || 'Student';
      memberPhotos[uid] = state.profile?.photoURL || '';
    }
  });

  await db.collection('groups').doc(eventGroupId(eventId)).set({
    name: `Event: ${eventData.title || 'Campus Event'}`,
    description: `Auto group for event attendees${eventData.date ? ` · ${eventData.date}` : ''}${eventData.time ? ` ${eventData.time}` : ''}`,
    type: 'event',
    isEventGroup: true,
    eventId,
    eventImage: (eventData.imageURLs && eventData.imageURLs.length ? eventData.imageURLs[0] : '') || eventData.coverURL || eventData.imageURL || '',
    eventDate: eventData.date || '',
    eventTime: eventData.time || '',
    createdBy: eventData.createdBy || state.user.uid,
    admins: [eventData.createdBy || state.user.uid],
    members,
    memberNames,
    memberPhotos,
    lastMessage: eventData.lastMessage || 'Event attendee group created',
    updatedAt: FieldVal.serverTimestamp(),
    createdAt: FieldVal.serverTimestamp()
  }, { merge: true });
}

async function loadCampusEvents() {
  try {
    let snap;
    try {
      snap = await db.collection('events').orderBy('date', 'asc').limit(80).get();
    } catch (orderedErr) {
      // Some legacy docs can break ordered queries; fall back so events still render.
      console.warn('Ordered events query failed, using fallback:', orderedErr?.message || orderedErr);
      snap = await db.collection('events').limit(80).get();
    }
    const now = Date.now();
    const expiredDocs = [];
    const upcoming = [];
    snap.docs.forEach(doc => {
      const data = { id: doc.id, ...doc.data() };
      const startAt = eventStartTimeMs(data);
      if (Number.isFinite(startAt) && startAt < now) expiredDocs.push(doc.ref);
      else upcoming.push(data);
    });
    if (expiredDocs.length) {
      await Promise.all(expiredDocs.map(ref => ref.delete().catch(() => {})));
      await Promise.all(expiredDocs.map(ref => db.collection('groups').doc(eventGroupId(ref.id)).delete().catch(() => {})));
    }
    allCampusEvents = upcoming.sort((a, b) => eventStartTimeMs(a) - eventStartTimeMs(b));
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
      <div id="map-route-panel" class="map-route-panel" style="display:none"></div>

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

  clearActiveMapRoute({ clearPending: false });
  if (_leafletMap) { _leafletMap.remove(); _leafletMap = null; }

  const myCoords = getUserCoords(state.profile) || { lat: -26.6840, lng: 27.0945 };

  // NWU Potchefstroom campus center
  _leafletMap = L.map('leaflet-map', { zoomControl: false }).setView([myCoords.lat, myCoords.lng], 16);
  L.control.zoom({ position: 'topright' }).addTo(_leafletMap);

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
    maxZoom: 20,
    subdomains: 'abcd'
  }).addTo(_leafletMap);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png', {
    maxZoom: 20, subdomains: 'abcd', pane: 'overlayPane'
  }).addTo(_leafletMap);

  const myIcon = L.divIcon({
    className: 'leaflet-user-pin',
    html: `<div class="map-user-pin map-me-pin">${state.profile.photoURL ? `<img src="${state.profile.photoURL}">` : `<span>${initials(state.profile.displayName)}</span>`}</div>`,
    iconSize: [36, 36],
    iconAnchor: [18, 18]
  });
  L.marker([myCoords.lat, myCoords.lng], { icon: myIcon }).addTo(_leafletMap).bindPopup('<b>You</b>');

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
    marker.bindPopup(`<div class="map-popup-card"><b>${loc.emoji} ${loc.name}</b>${hasEvents ? `<br><small>${evts.length} event${evts.length > 1 ? 's' : ''}</small>` : ''}<div class="map-popup-actions"><button class="map-popup-btn" onclick="routeToCampusLocation('${loc.id}')">Route here</button><button class="map-popup-btn secondary" onclick="openLocationDetail('${loc.id}')">Details</button></div></div>`);
  });

  // User pins - show friends on map
  allExploreUsers.forEach(u => {
    const coords = getUserCoords(u);
    if (!coords) return;
    const userIcon = L.divIcon({
      className: 'leaflet-user-pin',
      html: `<div class="map-user-pin">${u.photoURL ? `<img src="${u.photoURL}">` : `<span>${initials(u.displayName)}</span>`}</div>`,
      iconSize: [28, 28],
      iconAnchor: [14, 14]
    });
    const m = L.marker([coords.lat, coords.lng], { icon: userIcon }).addTo(_leafletMap);
    m.bindPopup(`<b>${esc(u.displayName)}</b><br><small>${esc(u.major || '')}</small>`);
    m.on('click', () => showUserPreview(u.id));
  });

  if (_pendingMapRoute) drawPendingMapRouteOnCurrentMap();
  else updateMapRoutePanel(null);

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
      ` : `<div class="empty-state"><h3>No events here</h3><p>Nothing happening at ${esc(loc.name)} yet</p></div>`}
      <button class="btn-outline btn-full" style="margin-top:12px" onclick="closeModal();routeToCampusLocation('${locationId}')">🧭 Route Here</button>
      <button class="btn-primary btn-full" style="margin-top:10px" onclick="closeModal();openCreateEvent('${locationId}')">+ Create Event Here</button>
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

function renderStarString(score = 0) {
  const rounded = Math.max(1, Math.min(5, Math.round(Number(score) || 0)));
  return `${'★'.repeat(rounded)}${'☆'.repeat(5 - rounded)}`;
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
            <span>${esc(ev.creatorName || 'Unibo')}</span>
          </div>
        </div>
        <button class="btn-primary btn-full" onclick="toggleEventGoing('${ev.id}');closeModal()">
          ${amGoing ? '✓ Going — Tap to Cancel' : "I'm Going!"}
        </button>
        ${loc ? `<button class="btn-outline btn-full" style="margin-top:10px" onclick="closeModal();routeToCampusLocation('${loc.id}')">🧭 Route Here</button>` : ''}
        ${amGoing ? `<button class="btn-outline btn-full" style="margin-top:10px" onclick="closeModal();navigate('chat');setTimeout(() => openGroupChat('${eventGroupId(ev.id)}','groups'), 120)">Open Event Group Chat</button>` : ''}
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
    await db.collection('groups').doc(eventGroupId(eventId)).delete().catch(() => {});
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
    const eventData = doc.data() || {};
    const going = eventData.going || [];
    let nextGoing = going;
    if (going.includes(uid)) {
      await db.collection('events').doc(eventId).update({ going: FieldVal.arrayRemove(uid) });
      nextGoing = going.filter(id => id !== uid);
      toast('RSVP cancelled');
    } else {
      await db.collection('events').doc(eventId).update({ going: FieldVal.arrayUnion(uid) });
      nextGoing = Array.from(new Set([...going, uid]));
      toast("You're going! 🎉");
    }
    await upsertEventChatGroup({ id: eventId, ...eventData, going: nextGoing });
    await loadCampusEvents();
    if (state.page === 'explore') renderExploreView();
    if (state.page === 'feed') loadDiscoverEvents();
  } catch (e) { toast('Failed'); console.error(e); }
}

// ══════════════════════════════════════════════════
//  HUSTLE (Marketplace)
// ══════════════════════════════════════════════════
function renderHustle() {
  const c = $('#content');
  c.innerHTML = `
    <div class="hustle-page">
      <div class="hustle-header"><h2>Marketplace</h2><div class="hustle-header-actions"><button class="btn-outline btn-sm" onclick="openSavedCartView()">Saved</button><button class="btn-primary btn-sm" onclick="openSellModal()">+ Sell</button></div></div>
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
    await ensureUserContextCache(items.map(item => item.sellerId));
    const interestProfile = buildInterestProfile();
    
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
    items = items.map(item => {
      const sellerContext = _userContextCache[item.sellerId] || null;
      const nearbyBoost = sellerContext ? getNearbySignal(state.profile, sellerContext).score * 5 : 0;
      const interestBoost = textInterestScore(`${item.title || ''} ${item.category || ''} ${item.description || ''}`, interestProfile);
      const rank = nearbyBoost + interestBoost + ((item.createdAt?.seconds || 0) / 100000);
      const syntheticViews = Math.round(((item.interestCount || 0) * 6) + scoreSeed(item.id || '') * 40 + 8);
      const syntheticRating = Number(item.ratingAvg || (3.6 + scoreSeed(item.sellerId || item.id || '') * 1.3));
      return {
        ...item,
        _rank: rank,
        _views: Math.max(item.views || 0, syntheticViews),
        _ratingAvg: Math.min(5, Math.max(1, syntheticRating)),
        _ratingCount: Number(item.ratingCount || 0),
        _isHot: rank > 15 || (item.interestCount || 0) >= 2,
        _sellingFast: (item.interestCount || 0) >= 4
      };
    });
    items.sort((a, b) => b._rank - a._rank || (b.createdAt?.seconds || 0) - (a.createdAt?.seconds || 0));

    if (!items.length) {
      grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1"><div class="empty-state-icon">🛒</div><h3>No listings yet</h3><p>Be the first to sell something!</p></div>`;
      return;
    }
    grid.innerHTML = items.map(item => `
      <div class="listing-card" onclick="openProductDetail('${item.id}')" oncontextmenu="event.preventDefault();openHustleQuickActions('${item.id}')">
        ${item.imageURL ? `<img class="listing-image" src="${item.imageURL}" loading="lazy">` : '<div class="listing-placeholder">📦</div>'}
        <div class="listing-info">
          <div class="listing-meta-tags">
            ${item._isHot ? '<span class="listing-meta-tag hot">🔥 Hot</span>' : ''}
            <span class="listing-meta-tag">⏳ ${timeAgo(item.createdAt)}</span>
            <span class="listing-meta-tag">👀 ${item._views} views</span>
            ${item._sellingFast ? '<span class="listing-meta-tag fast">⚡ Selling fast</span>' : ''}
          </div>
          <div class="listing-price">R${esc(String(item.price))}</div>
          <div class="listing-title">${esc(item.title)}</div>
          <div class="listing-rating">${renderStarString(item._ratingAvg)}${item._ratingCount ? ` <span>(${item._ratingCount})</span>` : ''}</div>
          <div class="listing-seller">${avatar(item.sellerName, (_userContextCache[item.sellerId]?.photoURL || null), 'avatar-sm')}<span>${esc(item.sellerName)}${(_userContextCache[item.sellerId] && getNearbySignal(state.profile, _userContextCache[item.sellerId]).score > 0) ? ' · Nearby' : ''}</span></div>
        </div>
      </div>
    `).join('');
    // store items for detail view
    window._hustleItems = {};
    items.forEach(i => window._hustleItems[i.id] = i);
  } catch (e) { grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><h3>Error loading</h3></div>'; }
}

async function saveListingToProfile(itemId) {
  if (!itemId) return;
  try {
    await db.collection('users').doc(state.user.uid).set({
      savedListings: FieldVal.arrayUnion(itemId)
    }, { merge: true });
    state.profile.savedListings = Array.from(new Set([...(state.profile.savedListings || []), itemId]));
    if (typeof window._savedCartRefresh === 'function') window._savedCartRefresh();
    toast('Saved listing');
  } catch (e) {
    console.error(e);
    toast('Could not save listing');
  }
}

async function removeSavedListingFromProfile(itemId) {
  if (!itemId) return;
  try {
    await db.collection('users').doc(state.user.uid).set({
      savedListings: FieldVal.arrayRemove(itemId)
    }, { merge: true });
    state.profile.savedListings = (state.profile.savedListings || []).filter(id => id !== itemId);
    if (typeof window._savedCartRefresh === 'function') window._savedCartRefresh();
    toast('Removed');
  } catch (e) {
    console.error(e);
    toast('Could not remove');
  }
}

async function openSavedCartView() {
  openModal(`
    <div class="modal-header"><h2>Saved Cart</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      <div class="form-group"><label>Budget (R)</label><input type="number" id="cart-budget" placeholder="e.g. 1200"></div>
      <div id="saved-cart-list"><div style="text-align:center;padding:18px"><span class="inline-spinner"></span></div></div>
      <div id="saved-cart-total" style="margin-top:12px;font-weight:700"></div>
    </div>
  `);
  const budgetInput = $('#cart-budget');
  const key = `unino-budget-${state.user.uid}`;
  if (budgetInput) budgetInput.value = localStorage.getItem(key) || '';

  const renderSaved = async () => {
    const holder = $('#saved-cart-list');
    const totalEl = $('#saved-cart-total');
    if (!holder || !totalEl) return;
    const ids = state.profile.savedListings || [];
    if (!ids.length) {
      holder.innerHTML = '<p style="color:var(--text-secondary)">No saved listings yet.</p>';
      totalEl.textContent = 'Total: R0';
      return;
    }
    const docs = await Promise.all(ids.slice(0, 30).map(id => db.collection('listings').doc(id).get().catch(() => null)));
    const items = docs.filter(d => d?.exists).map(d => ({ id: d.id, ...d.data() }));
    const total = items.reduce((sum, item) => sum + Number(item.price || 0), 0);
    const budget = Number(budgetInput?.value || 0);
    holder.innerHTML = items.map(item => `
      <div class="saved-cart-row" style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid var(--border);gap:10px">
        <div style="display:flex;align-items:center;gap:10px;min-width:0">
          ${item.imageURL ? `<img src="${item.imageURL}" alt="" style="width:44px;height:44px;border-radius:8px;object-fit:cover;border:1px solid var(--border)">` : ''}
          <div style="min-width:0">
          <div style="font-weight:600">${esc(item.title || 'Listing')}</div>
          <div style="font-size:12px;color:var(--text-secondary)">R${esc(String(item.price || 0))}</div>
          </div>
        </div>
        <div style="display:flex;gap:8px">
          <button class="btn-outline" style="padding:6px 10px;font-size:12px" onclick="openProductDetail('${item.id}')">View</button>
          <button class="btn-outline saved-cart-remove" title="Remove from saved" aria-label="Remove from saved" style="padding:6px 10px;font-size:12px" onclick="removeSavedListingFromProfile('${item.id}')">&times;</button>
        </div>
      </div>
    `).join('');
    totalEl.textContent = `Total: R${total.toFixed(2)}${budget > 0 ? ` · ${total <= budget ? 'Within budget' : 'Over budget'}` : ''}`;
  };

  budgetInput?.addEventListener('input', () => {
    localStorage.setItem(key, budgetInput.value || '');
    renderSaved();
  });
  window._savedCartRefresh = renderSaved;
  renderSaved();
}

function shareListingCard(itemId) {
  const item = (window._hustleItems || {})[itemId];
  if (!item) return;
  const text = `${item.title} · R${item.price} on Unibo`;
  if (navigator.share) {
    navigator.share({ title: item.title, text }).catch(() => {});
  } else {
    navigator.clipboard?.writeText(text).then(() => toast('Listing info copied')).catch(() => toast('Share not available'));
  }
}

function openHustleQuickActions(itemId) {
  const item = (window._hustleItems || {})[itemId];
  if (!item) return;
  const isSelf = item.sellerId === state.user.uid;
  const isFriend = (state.profile.friends || []).includes(item.sellerId);
  openModal(`
    <div class="modal-header"><h2>Quick Actions</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" style="padding:16px">
      <div style="font-weight:700;margin-bottom:10px">${esc(item.title)}</div>
      <button class="btn-outline btn-full" style="margin-bottom:8px" onclick="saveListingToProfile('${itemId}')">❤️ Save</button>
      ${isSelf ? '' : `<button class="btn-outline btn-full" style="margin-bottom:8px" onclick="closeModal();${isFriend ? `startChat('${item.sellerId}','${esc(item.sellerName)}','')` : `hustleBuyInterest('${itemId}')`}">💬 Chat</button>`}
      <button class="btn-outline btn-full" onclick="shareListingCard('${itemId}')">📤 Share</button>
    </div>
  `);
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
  const myCoords = getUserCoords(state.profile);
  const sellerContext = _userContextCache[item.sellerId] || null;
  const sellerCoords = getUserCoords(sellerContext || {});
  const distanceText = (!isSelf && myCoords && sellerCoords) ? formatDistanceText(distanceKmBetween(myCoords, sellerCoords)) : '';
  const myRating = Number((item.userRatings || {})[state.user.uid] || 0);

  openModal(`
    <div class="modal-header"><h2>Product Details</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      ${item.imageURL ? `<div style="border-radius:var(--radius);overflow:hidden;margin-bottom:16px"><img src="${item.imageURL}" style="width:100%;max-height:280px;object-fit:cover;cursor:pointer" onclick="viewImage('${item.imageURL}')"></div>` : ''}
      <div style="font-size:24px;font-weight:800;color:var(--accent);margin-bottom:4px">R${esc(String(item.price))}</div>
      <div style="font-size:18px;font-weight:700;margin-bottom:12px">${esc(item.title)}</div>
      ${item.description ? `<p style="font-size:14px;line-height:1.5;color:var(--text-secondary);margin-bottom:12px">${esc(item.description)}</p>` : ''}
      <div id="product-seller-row" style="display:flex;align-items:center;gap:8px;margin-bottom:12px;cursor:pointer" onclick="openProfile('${item.sellerId}')">
        <span id="product-seller-avatar">${avatar(item.sellerName, (sellerContext?.photoURL || null), 'avatar-sm')}</span>
        <div>
          <div id="product-seller-name" style="font-weight:600;font-size:14px">${esc(item.sellerName)}</div>
          <div style="font-size:12px;color:var(--text-secondary)">Seller</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;padding:10px 0;border-top:1px solid var(--border);font-size:13px;color:var(--text-secondary)">
        <span>📁 ${esc(item.category || 'Other')}</span>
        <span>·</span>
        <span>📅 ${timeAgo(item.createdAt)}</span>
        <span id="product-distance-meta">${distanceText ? `· 📍 ${esc(distanceText)}` : ''}</span>
      </div>
      <div style="margin-top:10px;padding:10px;border:1px solid var(--border);border-radius:12px;background:var(--bg-tertiary)">
        <div style="font-size:12px;color:var(--text-secondary);margin-bottom:8px">Rate this listing</div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <div class="product-rating-stars" id="product-rating-stars">${renderProductRatingStars(itemId, myRating)}</div>
          <input type="hidden" id="product-rating-value" value="${myRating || ''}">
          <button class="btn-outline" onclick="saveListingToProfile('${itemId}')" style="padding:8px 12px">Save</button>
        </div>
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

  if (!distanceText && !isSelf && myCoords) {
    db.collection('users').doc(item.sellerId).get().then(doc => {
      if (!doc.exists) return;
      const seller = { id: doc.id, ...doc.data() };
      _userContextCache[item.sellerId] = seller;
      const sellerAvatarEl = document.getElementById('product-seller-avatar');
      if (sellerAvatarEl) sellerAvatarEl.innerHTML = avatar(seller.displayName || item.sellerName || 'Seller', seller.photoURL || null, 'avatar-sm');
      const sellerNameEl = document.getElementById('product-seller-name');
      if (sellerNameEl) sellerNameEl.textContent = seller.displayName || item.sellerName || 'Seller';
      const sellerCoordsLive = getUserCoords(seller);
      const sellerDistanceText = sellerCoordsLive ? formatDistanceText(distanceKmBetween(myCoords, sellerCoordsLive)) : '';
      const target = document.getElementById('product-distance-meta');
      if (target) target.textContent = sellerDistanceText ? `· 📍 ${sellerDistanceText}` : '';
    }).catch(() => {});
  }
}

function renderProductRatingStars(itemId, value = 0) {
  const score = Number(value || 0);
  return [1, 2, 3, 4, 5].map(i => {
    const active = i <= score ? 'active' : '';
    return `<button type="button" class="product-rate-star ${active}" onclick="setProductRating('${itemId}', ${i})" aria-label="Rate ${i} stars">★</button>`;
  }).join('');
}

const _listingRatingBusy = new Set();

function setProductRating(itemId, value) {
  const input = $('#product-rating-value');
  if (input) input.value = String(value);
  const wrap = $('#product-rating-stars');
  if (wrap) wrap.innerHTML = renderProductRatingStars(itemId, value);
  submitListingRating(itemId, value, { silent: true, keepOpen: true });
}

async function submitListingRating(itemId, explicitScore = null, options = {}) {
  const score = Number(explicitScore != null ? explicitScore : ($('#product-rating-value')?.value || 0));
  if (!itemId || !score || score < 1 || score > 5) return toast('Choose a rating first');
  if (_listingRatingBusy.has(itemId)) return;
  _listingRatingBusy.add(itemId);
  try {
    const ref = db.collection('listings').doc(itemId);
    await db.runTransaction(async tx => {
      const snap = await tx.get(ref);
      if (!snap.exists) throw new Error('missing');
      const data = snap.data() || {};
      const userRatings = data.userRatings || {};
      const prev = Number(userRatings[state.user.uid] || 0);
      const ratingCount = Number(data.ratingCount || 0);
      const ratingTotal = Number(data.ratingTotal || 0);
      const nextCount = prev ? ratingCount : ratingCount + 1;
      const nextTotal = ratingTotal - prev + score;
      tx.update(ref, {
        userRatings: { ...userRatings, [state.user.uid]: score },
        ratingCount: nextCount,
        ratingTotal: nextTotal,
        ratingAvg: nextCount ? Number((nextTotal / nextCount).toFixed(2)) : 0
      });
    });
    const local = (window._hustleItems || {})[itemId];
    if (local) {
      const userRatings = local.userRatings || {};
      const prev = Number(userRatings[state.user.uid] || 0);
      const ratingCount = Number(local.ratingCount || 0);
      const ratingTotal = Number(local.ratingTotal || 0);
      const nextCount = prev ? ratingCount : ratingCount + 1;
      const nextTotal = ratingTotal - prev + score;
      local.userRatings = { ...userRatings, [state.user.uid]: score };
      local.ratingCount = nextCount;
      local.ratingTotal = nextTotal;
      local.ratingAvg = nextCount ? Number((nextTotal / nextCount).toFixed(2)) : 0;
    }
    if (!options.silent) toast('Rating saved');
    if (!options.keepOpen) {
      closeModal();
      loadListings();
    }
  } catch (e) {
    console.error(e);
    toast('Could not save rating');
  } finally {
    _listingRatingBusy.delete(itemId);
  }
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
    await addNotification(sellerId, 'hustle_interest', `is interested in your listing "${item.title}"`, {
      itemId,
      itemTitle: item.title,
      itemPrice: item.price
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

function renderEventGroupChatBanner(group = {}) {
  if (!group?.isEventGroup) return '';
  const members = Array.isArray(group.members) ? group.members : [];
  const attendeePreview = members.slice(0, 5).map(memberId => {
    const fallbackName = (group.memberNames || {})[memberId] || 'Student';
    const fallbackPhoto = (group.memberPhotos || {})[memberId] || null;
    const identity = getResolvedUserIdentity(memberId, fallbackName, fallbackPhoto);
    return `<button type="button" class="event-chat-attendee" onclick="event.stopPropagation();openProfile('${memberId}')">${avatar(identity.name, identity.photo, 'avatar-xs')}</button>`;
  }).join('');
  return `
    <div class="event-chat-banner clickable" onclick="${group.eventId ? `openEventDetail('${group.eventId}')` : `openGroupDetail('${group.id}')`}">
      <div class="event-chat-banner-media">
        ${group.eventImage ? `<img src="${group.eventImage}" alt="">` : '<div class="event-chat-banner-fallback">📅</div>'}
      </div>
      <div class="event-chat-banner-copy">
        <div class="event-chat-banner-title">${esc(group.name || 'Event Chat')}</div>
        <div class="event-chat-banner-meta">${group.eventDate ? `📅 ${esc(group.eventDate)}` : ''}${group.eventTime ? ` · 🕐 ${esc(group.eventTime)}` : ''}</div>
        <div class="event-chat-banner-members">
          <div class="event-chat-attendees">${attendeePreview}</div>
          <span>${members.length} going</span>
        </div>
        <div class="event-chat-banner-actions">
          ${group.eventId ? `<button class="btn-outline btn-sm" onclick="event.stopPropagation();openEventDetail('${group.eventId}')">Event details</button>` : ''}
          <button class="btn-primary btn-sm" onclick="event.stopPropagation();openGroupDetail('${group.id}')">Group info</button>
        </div>
      </div>
    </div>
  `;
}


function closeChatPlusMenus() {
  $$('.chat-plus-menu').forEach(menu => {
    menu.classList.remove('open');
    menu.style.display = 'none';
  });
}

function ensureChatPlusMenu(menu, scope = 'dm') {
  const fileInput = scope === 'group' ? $('#gchat-file-input') : $('#chat-file-input');
  if (!menu || menu.dataset.ready === '1') return;
  menu.innerHTML = `
    <button type="button" class="chat-plus-option modern" data-act="upload"><span class="chat-plus-option-icon">🖼️</span><span>UPLOAD</span></button>
    <button type="button" class="chat-plus-option modern" data-act="poll"><span class="chat-plus-option-icon">📊</span><span>POLL</span></button>
    <button type="button" class="chat-plus-option modern" data-act="pin"><span class="chat-plus-option-icon">📍</span><span>PIN</span></button>
  `;
  menu.dataset.ready = '1';
  menu.querySelectorAll('.chat-plus-option').forEach(btn => {
    btn.addEventListener('click', () => {
      const act = btn.dataset.act || '';
      closeChatPlusMenus();
      if (act === 'upload') { fileInput?.click(); return; }
      if (act === 'poll') { openChatPollComposer(scope); return; }
      if (act === 'pin') { sendChatLocationPin(scope); }
    });
  });
}

function openChatPlusMenu(scope = 'dm') {
  const menu = scope === 'group' ? $('#gchat-plus-menu') : $('#chat-plus-menu');
  if (!menu) return;
  ensureChatPlusMenu(menu, scope);
  const isOpen = menu.classList.contains('open');
  closeChatPlusMenus();
  if (isOpen) return;
  menu.style.display = 'grid';
  menu.classList.add('open');
}

document.addEventListener('pointerdown', event => {
  if (event.target.closest('.chat-plus-wrap')) return;
  closeChatPlusMenus();
}, true);
document.addEventListener('scroll', () => closeChatPlusMenus(), true);

function renderInteractivePoll(poll = {}, scope = 'post', primaryId = '', messageId = '') {
  const options = Array.isArray(poll.options) ? poll.options : [];
  const voters = poll.votes || {};
  const total = Object.keys(voters).length;
  const myVote = voters[state.user?.uid] || '';
  const isFeedPoll = scope === 'post';
  const totalLabel = total ? `${total} total vote${total === 1 ? '' : 's'}` : 'No votes yet';
  return `
    <div class="poll-card modern-poll ${isFeedPoll ? 'feed-poll' : ''}" data-poll-scope="${scope}" data-poll-primary="${primaryId}" data-poll-message="${messageId}">
      ${isFeedPoll ? `<div class="feed-poll-head"><span class="feed-poll-chip">Campus poll</span><span class="feed-poll-total">${totalLabel}</span></div>` : ''}
      ${poll.question ? `<div class="poll-question">${esc(poll.question)}</div>` : ''}
      <div class="poll-options">
        ${options.map(opt => {
          const count = Object.values(voters).filter(v => v === opt).length;
          const pct = total ? Math.round((count / total) * 100) : 0;
          const encodedOpt = encodeURIComponent(opt || '');
          return `<button class="poll-option ${myVote === opt ? 'active' : ''}" data-opt="${esc(opt)}" onclick="voteOnPollEncoded('${scope}','${primaryId}','${messageId}','${encodedOpt}')"><span class="poll-option-main"><span class="poll-option-text">${esc(opt)}</span><span class="poll-option-meta">${count} vote${count === 1 ? '' : 's'}${total ? ` · ${pct}%` : ''}</span></span><span class="poll-option-fill" style="width:${Math.max(myVote === opt ? 12 : 0, pct)}%"></span></button>`;
        }).join('')}
      </div>
      <div class="poll-footer">${isFeedPoll ? 'Tap an option to vote' : totalLabel}</div>
    </div>`;
}

function voteOnPollEncoded(scope = 'post', primaryId = '', messageId = '', encodedOption = '') {
  let option = '';
  try {
    option = decodeURIComponent(String(encodedOption || ''));
  } catch (_) {
    option = String(encodedOption || '');
  }
  voteOnPoll(scope, primaryId, messageId, option);
}

async function voteOnPoll(scope = 'post', primaryId = '', messageId = '', option = '') {
  try {
    let ref = null;
    if (scope === 'post') ref = db.collection('posts').doc(primaryId);
    else if (scope === 'group') ref = db.collection(_activeGroupChat.collection || 'groups').doc(primaryId).collection('messages').doc(messageId);
    else if (scope === 'dm') ref = db.collection('conversations').doc(primaryId).collection('messages').doc(messageId);
    if (!ref) return;
    const host = document.querySelector(`[data-poll-scope="${scope}"][data-poll-primary="${primaryId}"][data-poll-message="${messageId}"]`);
    if (host) {
      host.querySelectorAll('.poll-option').forEach(btn => {
        const active = btn.getAttribute('data-opt') === option;
        btn.classList.toggle('active', active);
      });
    }
    const snap = await ref.get();
    if (!snap.exists) return;
    const data = snap.data() || {};
    const poll = data.poll || { options: [], votes: {} };
    const votes = { ...(poll.votes || {}) };
    votes[state.user.uid] = option;
    await ref.update({ poll: { ...poll, votes } });
  } catch (e) { console.error(e); toast('Could not save vote'); }
}

function openChatPollComposer(scope = 'dm') {
  openModal(`
    <div class="modal-header"><h2>Create poll</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body modern-poll-composer">
      <div class="form-group"><label>Question</label><input type="text" id="chat-poll-question" placeholder="What should we do after class?"></div>
      <div class="form-group"><label>Options</label><textarea id="chat-poll-options" placeholder="One option per line" style="height:110px;resize:none">Yes
No</textarea></div>
      <div class="poll-composer-tip">Tip: keep options short so voting looks clean in chat.</div>
      <button class="btn-primary btn-full" id="chat-poll-send-btn" onclick="submitChatPoll('${scope}')">Send poll</button>
    </div>
  `);
}

async function submitChatPoll(scope = 'dm') {
  if (window._sendingChatPoll) return;
  const question = ($('#chat-poll-question')?.value || '').trim();
  const options = Array.from(new Set(($('#chat-poll-options')?.value || '').split('\n').map(v => v.trim()).filter(Boolean))).slice(0, 6);
  if (!question || options.length < 2) return toast('Add a question and at least 2 options');
  const btn = $('#chat-poll-send-btn');
  window._sendingChatPoll = true;
  if (btn) { btn.disabled = true; btn.textContent = 'Sending...'; }
  closeModal();
  try {
    if (scope === 'group') {
      const { id, collection } = _activeGroupChat || {};
      if (!id) return;
      await db.collection(collection || 'groups').doc(id).collection('messages').add({ text: '', poll: { question, options, votes: {} }, senderId: state.user.uid, senderName: state.profile.displayName, senderPhoto: state.profile.photoURL || null, createdAt: FieldVal.serverTimestamp() });
      await db.collection(collection || 'groups').doc(id).set({ lastMessage: `📊 ${question}`, updatedAt: FieldVal.serverTimestamp() }, { merge: true });
    } else {
      const convoId = _activeChatConvoId;
      if (!convoId) return;
      const convoDoc = await db.collection('conversations').doc(convoId).get();
      const convo = convoDoc.data() || {};
      const otherUid = (convo.participants || []).find(id => id !== state.user.uid) || '';
      await db.collection('conversations').doc(convoId).collection('messages').add({ text: '', poll: { question, options, votes: {} }, senderId: state.user.uid, senderAnon: !!(convo.anonymous || {})[state.user.uid], createdAt: FieldVal.serverTimestamp(), status: 'sent' });
      await db.collection('conversations').doc(convoId).set({ lastMessage: `📊 ${question}`, updatedAt: FieldVal.serverTimestamp(), unread: otherUid ? { [otherUid]: FieldVal.increment(1), [state.user.uid]: 0 } : {} }, { merge: true });
    }
  } catch (e) { console.error(e); toast('Could not send poll'); }
  finally { window._sendingChatPoll = false; }
}

async function sendChatLocationPin(scope = 'dm') {
  if (!navigator.geolocation) return toast('Location unavailable');
  navigator.geolocation.getCurrentPosition(async pos => {
    const lat = Number(pos.coords.latitude.toFixed(6));
    const lng = Number(pos.coords.longitude.toFixed(6));
    const resolved = await resolveLocationPinLabel(lat, lng).catch(() => ({ label: 'Pinned location', sub: '' }));
    const locationPin = {
      lat,
      lng,
      label: resolved?.label || 'Pinned location',
      subLabel: resolved?.sub || ''
    };
    try {
      if (scope === 'group') {
        const { id, collection } = _activeGroupChat || {};
        if (!id) return;
        await db.collection(collection || 'groups').doc(id).collection('messages').add({ text: '', locationPin, senderId: state.user.uid, senderName: state.profile.displayName, senderPhoto: state.profile.photoURL || null, createdAt: FieldVal.serverTimestamp() });
        await db.collection(collection || 'groups').doc(id).set({ lastMessage: `📍 ${locationPin.label}`, updatedAt: FieldVal.serverTimestamp() }, { merge: true });
      } else {
        const convoId = _activeChatConvoId;
        if (!convoId) return;
        const convoDoc = await db.collection('conversations').doc(convoId).get();
        const convo = convoDoc.data() || {};
        const otherUid = (convo.participants || []).find(id => id !== state.user.uid) || '';
        await db.collection('conversations').doc(convoId).collection('messages').add({ text: '', locationPin, senderId: state.user.uid, senderAnon: !!(convo.anonymous || {})[state.user.uid], createdAt: FieldVal.serverTimestamp(), status: 'sent' });
        await db.collection('conversations').doc(convoId).set({ lastMessage: `📍 ${locationPin.label}`, updatedAt: FieldVal.serverTimestamp(), unread: otherUid ? { [otherUid]: FieldVal.increment(1), [state.user.uid]: 0 } : {} }, { merge: true });
      }
    } catch (e) { console.error(e); toast('Could not send location'); }
  }, () => toast('Could not get location'), { enableHighAccuracy: true, timeout: 12000, maximumAge: 0 });
}

function openLocationPinPreview(encoded = '') {
  try {
    const pin = JSON.parse(decodeURIComponent(encoded || '')) || {};
    const lat = Number(pin.lat);
    const lng = Number(pin.lng);
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return;
    const encodedLabel = encodeURIComponent(pin.label || 'Pinned location');
    openModal(`
      <div class="modal-header"><h2>📍 ${esc(pin.label || 'Pinned location')}</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
      <div class="modal-body">
        <div class="location-preview-card">
          ${pin.subLabel ? `<div class="location-preview-meta" style="margin-bottom:4px">${esc(pin.subLabel)}</div>` : ''}
          <div class="location-preview-meta">${lat.toFixed(5)}, ${lng.toFixed(5)}</div>
          <div class="location-preview-actions">
            <button class="btn-primary" onclick="closeModal();routeToMapPointEncoded(${lat},${lng}, '${encodedLabel}')">Route here</button>
          </div>
        </div>
      </div>
    `);
  } catch (_) {}
}

function openPinnedMapPreview(lat, lng, label = 'Pinned location') {
  routeToMapPoint(lat, lng, label);
}

function renderLocationPinCard(pin = {}) {
  const lat = Number(pin.lat);
  const lng = Number(pin.lng);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return '';
  const encoded = encodeURIComponent(JSON.stringify(pin));
  const sub = (pin.subLabel || '').trim();
  return `<button class="chat-location-card modern" onclick="openLocationPinPreview('${encoded}')"><div class="chat-location-icon">📍</div><div class="chat-location-copy"><div class="chat-location-title">${esc(pin.label || 'Pinned location')}</div><div class="chat-location-sub">${esc(sub || 'Tap to route here')}</div></div><span class="chat-location-cta">Route</span></button>`;
}


async function openGroupChat(groupId, collection = 'groups') {
  try {
    const gDoc = await db.collection(collection).doc(groupId).get();
    if (!gDoc.exists) return toast('Group not found');
    const group = { id: groupId, ...gDoc.data() };
    const uid = state.user.uid;
    const gName = group.name || group.groupTitle || 'Group';
    const gType = group.type || 'study';
    await ensureUserContextCache(group.members || []);
    const gEmoji = collection === 'assignmentGroups' ? '📋' : (gType === 'study' ? '📚' : gType === 'project' ? '💻' : gType === 'module' ? '🧩' : '🎉');

    showScreen('group-chat-view');
    _activeGroupChat = { id: groupId, collection };
    _activeChatConvoId = '';
    const groupHeaderAction = group.isEventGroup && group.eventId ? `openEventDetail('${group.eventId}')` : `openGroupDetail('${group.id}')`;
    $('#gchat-hdr-info').innerHTML = `
      <div class="group-header-info clickable" onclick="${groupHeaderAction}">
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
        const bannerHtml = renderEventGroupChatBanner(group);
        if (!messages.length) {
          msgs.innerHTML = `${bannerHtml}<div style="text-align:center;padding:32px;opacity:0.5">Start the conversation! 💬</div>`;
        } else {
          msgs.innerHTML = bannerHtml + messages.map((m, idx) => {
            const isMe = m.senderId === uid;
            let content = '';
            if (m.deleted || m.type === 'deleted') {
              content = '<span class="msg-deleted">Message deleted</span>';
            }
            if (m.audioURL) content += renderVoiceMsg(m.audioURL);
            if (m.poll) content += renderInteractivePoll(m.poll, 'group', groupId, m.id);
            if (m.locationPin) content = renderLocationPinCard(m.locationPin);
            if (m.imageURL) {
              const isVideoMedia = m.mediaType === 'video' || /\.(mp4|webm|mov|m4v)(\?|#|$)/i.test(m.imageURL || '');
              if (isVideoMedia) {
                content += `<video src="${m.imageURL}" class="msg-image" style="background:#000" controls playsinline preload="metadata"></video>`;
              } else {
                content += `<img src="${m.imageURL}" class="msg-image" onclick="viewImage('${m.imageURL}')">`;
              }
            }
            if (!m.deleted && m.text) content += esc(m.text);
            // Support both new and legacy replies: infer original sender from replyToId when needed.
            const replyToSenderId = m.replyToSenderId || _gMsgLookup.get(m.replyToId || '')?.senderId;
            const replyDisplayName = replyToSenderId === uid ? 'me' : (m.replyToName || 'Message');
            const replyMeta = m.replyToText
              ? `<div class="msg-reply-snippet">↩ ${esc(replyDisplayName)}: ${esc(clampText(m.replyToText, 50))}</div>`
              : '';
            const newCls = (idx === messages.length - 1 && isMe) ? 'msg-new' : '';
            const reactionSummary = renderReactionSummary(m.reactions || {}, [], 'msg-inline');
            const senderIdentity = getResolvedUserIdentity(m.senderId, m.senderName || '?', m.senderPhoto || null);
            return `<div class="msg-row ${isMe ? 'msg-row-sent' : 'msg-row-received'}" id="msg-${m.id}">
              ${!isMe ? `<div class="msg-avatar-wrap">${avatar(senderIdentity.name || '?', senderIdentity.photo, 'avatar-xs')}</div>` : ''}
              <div class="msg-stack ${isMe ? 'msg-stack-sent' : 'msg-stack-received'}">
              <div class="msg-bubble ${isMe ? 'msg-sent' : 'msg-received'} ${newCls} ${m.locationPin ? 'msg-bubble-pin' : ''}" data-message-id="${m.id}">
              ${!isMe ? `<div class="gchat-sender">${esc((senderIdentity.name || '?').split(' ')[0] || '?')}</div>` : ''}
              ${m.replyToId && m.replyToText ? `<div class="msg-reply-snippet" onclick="jumpToMessage('${m.replyToId}','gchat-msgs')">↩ ${esc(replyDisplayName)}: ${esc(clampText(m.replyToText, 50))}</div>` : ''}
              ${content}
              ${m.deleted ? '' : `<button class="msg-reply-btn" title="Reply" aria-label="Reply" onclick="setGroupReply('${m.id}')"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 17 4 12 9 7"></polyline><path d="M20 18v-2a4 4 0 0 0-4-4H4"></path></svg></button>`}
              <div class="msg-time">${m.createdAt ? timeAgo(m.createdAt) : ''}</div>
            </div>
            ${reactionSummary ? `<div class="msg-reaction-line" onclick="event.stopPropagation();openMessageActionSheet('group','${groupId}','${m.id}','${collection}')">${reactionSummary}</div>` : ''}</div></div>`;
          }).join('');
          bindMessageLongPress(msgs, 'group', groupId, collection);
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
    let gchatPendingImg = null;

    const sendGMsg = async () => {
      const input = gInput || $('#gchat-input');
      const text = input?.value.trim();
      const moderation = applySoftKeywordFilterText(text || '');
      const safeText = moderation.text;
      const img = gchatPendingImg;
      const gchatFile = window._gchatFile || null;
      let audioURL = null;
      if (window._gchatVoiceBlob) {
        const af = new File([window._gchatVoiceBlob], `voice_${Date.now()}.webm`, { type: 'audio/webm' });
        audioURL = await uploadToR2(af, 'voice');
        window._gchatVoiceBlob = null;
      }
      if (!safeText && !img && !audioURL) return;
      const replyPayload = _gReplyTo ? {
        replyToId: _gReplyTo.id,
        replyToText: _gReplyTo.text,
        replyToName: _gReplyTo.name,
        replyToSenderId: _gReplyTo.senderId
      } : {};
      input.value = '';
      gchatPendingImg = null;
      window._gchatFile = null;
      resizeGInput();
      input.focus();
      _gReplyTo = null;
      if (gReply) gReply.style.display = 'none';
      const preview = $('#gchat-img-preview');
      if (preview) preview.style.display = 'none';
      try {
        let imageURL = null;
        let mediaType = null;
        if (gchatFile) {
          imageURL = await uploadToR2(gchatFile, 'chat-images');
          mediaType = (gchatFile.type || '').startsWith('video/') ? 'video' : 'image';
        }
        await db.collection(collection).doc(groupId).collection('messages').add({
          text: safeText || '', imageURL: imageURL || null, mediaType: mediaType || null, audioURL: audioURL || null,
          moderationFlags: moderation.flags,
          senderId: uid, senderName: state.profile.displayName,
          senderPhoto: state.profile.photoURL || null,
          ...replyPayload,
          createdAt: FieldVal.serverTimestamp()
        });
        await db.collection(collection).doc(groupId).update({
          lastMessage: audioURL ? '🎤 Voice' : imageURL ? (mediaType === 'video' ? '📹 Video' : (safeText || '📷 Photo')) : safeText,
          updatedAt: FieldVal.serverTimestamp()
        });
      } catch (e) { console.error(e); }
    };
    $('#gchat-send').onclick = sendGMsg;
    const groupPlusBtn = $('#gchat-plus-btn'); if (groupPlusBtn) groupPlusBtn.onclick = (e) => { e.stopPropagation(); openChatPlusMenu('group'); };
    $('#gchat-input').onfocus = () => setTimeout(() => scrollToLatest(msgs), 100);
    $('#gchat-input').onblur = () => setTimeout(() => scrollToLatest(msgs), 150);

    const gchatFileInput = $('#gchat-file-input');
    if (gchatFileInput) {
      gchatFileInput.onchange = e => {
        if (e.target.files[0]) {
          window._gchatFile = e.target.files[0];
          gchatPendingImg = localPreview(e.target.files[0]);
          const preview = $('#gchat-img-preview');
          if (preview) {
            preview.querySelector('img').src = gchatPendingImg;
            preview.style.display = 'block';
          }
        }
      };
    }

    $('#gchat-back').onclick = () => {
      _activeGroupChat = { id: '', collection: '' };
      if (gchatUnsub) { gchatUnsub(); gchatUnsub = null; }
      if (_gchatViewportCleanup) { _gchatViewportCleanup(); _gchatViewportCleanup = null; }
      window._gchatFile = null;
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
  const inGroupsView = state.lastMsgTab === 'groups';
  c.innerHTML = `
    <div class="messages-page">
      <div class="messages-header"><h2>Messages</h2><div class="messages-header-actions"><button class="btn-outline btn-sm" id="messages-groups-toggle" title="Open groups">${inGroupsView ? 'Chats' : 'Groups'}</button><button class="icon-btn anon-pref-btn" id="messages-anon-pref" title="Anonymous message setting"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l7 4v5c0 5-3.5 7.5-7 9-3.5-1.5-7-4-7-9V7l7-4z"/><path d="M9 12l2 2 4-4"/></svg></button></div></div>
      <div class="stories-row messages-stories-row" id="messages-stories-row">
        <div class="story-item add-story" onclick="openStoryCreator()">
          <div class="story-avatar"><div class="story-avatar-inner">+</div></div>
          <div class="story-name">Story</div>
        </div>
      </div>
      <div id="msg-tab-content">
        <div class="convo-list" id="convo-list">
          <div style="padding:40px;text-align:center"><span class="inline-spinner"></span></div>
        </div>
      </div>
      <button class="archive-fab" id="archive-fab" onclick="toggleArchiveDmView()" aria-label="Archived chats" title="Show archived chats">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="21 8 21 21 3 21 3 8"/><rect x="1" y="3" width="22" height="5"/><line x1="10" y1="12" x2="14" y2="12"/></svg>
        <span class="archive-fab-badge" id="archive-fab-badge" style="display:none"></span>
      </button>
    </div>
  `;
  // Compute DM unread count for the tab badge
  _updateDMTabBadge();
  refreshChatBadge();
  const anonPrefBtn = $('#messages-anon-pref');
  if (anonPrefBtn) {
    anonPrefBtn.onclick = openAnonDmSettings;
    updateAnonPrefButton('messages-anon-pref');
  }
  const groupsToggleBtn = $('#messages-groups-toggle');
  if (groupsToggleBtn) {
    groupsToggleBtn.onclick = toggleMessagesGroupsView;
  }
  if (!inGroupsView) loadStories();
  bindMessagesScrollEffects();
  syncMessagesStoriesVisibility();

  const restoreTab = state.lastMsgTab || 'dm';
  if (restoreTab === 'archived') {
    loadArchivedDMList();
  } else if (restoreTab === 'groups') {
    loadGroups();
  } else {
    loadDMList();
  }
  updateMessagesGroupsToggle();
  updateArchiveFabState();
  syncMessagesStoriesVisibility();
}

function syncMessagesStoriesVisibility() {
  const page = document.querySelector('.messages-page');
  const row = document.getElementById('messages-stories-row');
  const hideStories = state.lastMsgTab === 'groups';
  if (page) page.classList.toggle('stories-collapsed', hideStories);
  if (row) row.style.display = hideStories ? 'none' : '';
}

function updateMessagesGroupsToggle() {
  const btn = $('#messages-groups-toggle');
  if (!btn) return;
  btn.textContent = state.lastMsgTab === 'groups' ? 'Chats' : 'Groups';
}

function toggleMessagesGroupsView() {
  if (state.lastMsgTab === 'groups') {
    state.lastMsgTab = 'dm';
    loadDMList();
  } else {
    state.lastMsgTab = 'groups';
    loadGroups();
  }
  updateMessagesGroupsToggle();
  updateArchiveFabState();
  syncMessagesStoriesVisibility();
}

function toggleArchiveDmView() {
  if (state.lastMsgTab === 'archived') {
    state.lastMsgTab = state.lastMsgTabBeforeArchive || 'dm';
    if (state.lastMsgTab === 'groups') loadGroups();
    else loadDMList();
  } else {
    state.lastMsgTabBeforeArchive = state.lastMsgTab === 'archived' ? 'dm' : (state.lastMsgTab || 'dm');
    state.lastMsgTab = 'archived';
    loadArchivedDMList();
  }
  updateMessagesGroupsToggle();
  updateArchiveFabState();
  syncMessagesStoriesVisibility();
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
  syncMessagesStoriesVisibility();
}

function bindMessagesScrollEffects() {
  const page = document.querySelector('.messages-page');
  const host = document.getElementById('msg-tab-content');
  if (!page || !host || host.dataset.storyScrollBound === '1') return;
  let lastTop = 0;
  let lastObservedTop = 0;
  let collapsed = false;
  let ticking = false;
  page.classList.toggle('stories-collapsed', state.lastMsgTab === 'groups');

  const applyState = (st) => {
    if (state.lastMsgTab === 'groups') {
      page.classList.add('stories-collapsed');
      lastTop = st;
      return;
    }
    const delta = st - lastTop;
    if (!collapsed && st > 44 && delta > 3) collapsed = true;
    if (collapsed && (delta < -2 || st < 12)) collapsed = false;
    page.classList.toggle('stories-collapsed', collapsed);
    lastTop = st;
  };

  host.addEventListener('scroll', e => {
    const target = e.target;
    if (!(target instanceof HTMLElement) || !target.classList.contains('convo-list')) return;
    lastObservedTop = target.scrollTop || 0;
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      applyState(lastObservedTop);
      ticking = false;
    });
  }, true);
  host.dataset.storyScrollBound = '1';
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
      <div id="event-chat-groups" style="padding:8px 12px 0"></div>
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
  loadEventChatGroups();
  loadAsgList('my');
}

async function loadEventChatGroups() {
  const host = $('#event-chat-groups');
  if (!host) return;
  const uid = state.user?.uid;
  if (!uid) { host.innerHTML = ''; return; }
  try {
    const snap = await db.collection('groups').where('members', 'array-contains', uid).limit(40).get();
    const groups = snap.docs
      .map(d => ({ id: d.id, ...d.data() }))
      .filter(g => g.isEventGroup)
      .sort((a, b) => (b.updatedAt?.seconds || 0) - (a.updatedAt?.seconds || 0));
    if (!groups.length) {
      host.innerHTML = '';
      return;
    }
    host.innerHTML = `
      <div style="padding:6px 4px 8px;font-size:12px;font-weight:700;color:var(--text-secondary)">EVENT GROUPS</div>
      ${groups.map(g => `
        <div class="convo-item" style="margin-bottom:8px" onclick="openGroupChat('${g.id}','groups')">
          <div class="convo-avatar">${g.eventImage ? `<img class="avatar-md group-avatar-photo" src="${g.eventImage}" alt="">` : '<div class="avatar-md group-avatar-icon">📅</div>'}</div>
          <div class="convo-info">
            <div class="convo-name">${esc(g.name || 'Event Group')}</div>
            <div class="convo-last-msg">${esc(g.lastMessage || 'Event chat')}</div>
          </div>
          <div class="convo-right">
            <div class="convo-time">${timeAgo(g.updatedAt)}</div>
            <div style="font-size:11px;color:var(--text-tertiary)">${(g.members || []).length} members</div>
          </div>
        </div>
      `).join('')}
    `;
  } catch (e) {
    console.error(e);
    host.innerHTML = '';
  }
}

async function loadAsgList(filter = 'my') {
  const el = $('#asg-list'); if (!el) return;
  const uid = state.user.uid;
  const myModules = normalizeModules(state.profile.modules || []);
  try {
    let snap;
    if (filter === 'mine') {
      snap = await db.collection('assignmentGroups').where('members', 'array-contains', uid).limit(30).get();
    } else {
      snap = await db.collection('assignmentGroups').where('status', '==', 'open').limit(50).get();
    }
    let groups = snap.docs.map(d => ({ id: d.id, ...d.data() }));
    if (filter === 'my') {
      groups = groups.filter(g => myModules.includes(normalizeModules([g.moduleCode])[0] || ''));
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
    moduleCode = normalizeModules([moduleCode])[0] || '';
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
    const filtered = q ? modulePeople.map(u => ({ user: u, score: userSearchScore(u, q) })).filter(entry => entry.score > 0).sort((a, b) => b.score - a.score).map(entry => entry.user) : modulePeople;
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
      await acceptFriendRequest(toUid, toName, toPhoto);
      if (btnEl) {
        btnEl.textContent = '✓ Friends';
        btnEl.disabled = true;
        btnEl.style.opacity = '0.6';
      }
      return;
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
    addNotification(toUid, 'friend_request', 'sent you a friend request', {});
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
    addNotification(fromUid, 'friend_accept', 'accepted your friend request', {});
    // Update local state
    state.profile.friends = [...(state.profile.friends || []), fromUid];
    state.profile.friendRequests = (state.profile.friendRequests || []).filter(r => r.uid !== fromUid);
    toast(`You and ${fromName} are now friends!`);
    // Only refresh dropdown if it's actually open
    const dd = $('#notif-dropdown');
    if (dd && dd.style.display === 'block') {
      loadNotifications();
      if (!(state.profile.friendRequests || []).length) dd.style.display = 'none';
    }
  } catch (e) { toast('Failed'); console.error(e); }
}

function isConversationWithUser(data = {}, otherUid, { anonymous = null } = {}) {
  const participants = data.participants || [];
  if (!participants.includes(otherUid) || !participants.includes(state.user?.uid)) return false;
  if (anonymous === null) return true;
  return !!data.isAnonymous === !!anonymous;
}

async function ensureFriendDMConversation(otherUid, otherName, otherPhoto) {
  const myUid = state.user.uid;
  const snap = await db.collection('conversations').where('participants', 'array-contains', myUid).get();
  const existing = snap.docs.find(d => isConversationWithUser(d.data() || {}, otherUid, { anonymous: false }));
  if (existing) {
    await existing.ref.set({ archived: FieldVal.arrayRemove(myUid, otherUid) }, { merge: true }).catch(() => {});
    return existing.id;
  }
  const ref = await db.collection('conversations').add({
    participants: [myUid, otherUid],
    participantNames: [state.profile.displayName, otherName || 'Friend'],
    participantPhotos: [state.profile.photoURL || null, otherPhoto || null],
    lastMessage: '',
    updatedAt: FieldVal.serverTimestamp(),
    unread: { [otherUid]: 0, [myUid]: 0 },
    participantStatuses: { [myUid]: state.status || 'online', [otherUid]: 'offline' },
    isAnonymous: false,
    anonymous: { [myUid]: false, [otherUid]: false },
    archived: []
  });
  return ref.id;
}

async function openFriendsList(uid = state.user?.uid, name = state.profile?.displayName || 'Friends') {
  if (!uid) return;
  try {
    const userDoc = uid === state.user?.uid
      ? { exists: true, data: () => state.profile }
      : await db.collection('users').doc(uid).get();
    if (!userDoc.exists) return toast('Could not load friends');
    const user = userDoc.data() || {};
    const friends = user.friends || [];
    openModal(`
      <div class="modal-header"><h2>${esc(name)}'s Friends</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
      <div class="modal-body" style="padding:16px"><div id="friends-list-modal"><div style="text-align:center;padding:24px"><span class="inline-spinner"></span></div></div></div>
    `);
    const host = $('#friends-list-modal');
    if (!host) return;
    if (!friends.length) {
      host.innerHTML = '<div class="empty-state"><h3>No friends yet</h3></div>';
      return;
    }
    const docs = await Promise.all(friends.map(friendId => db.collection('users').doc(friendId).get().catch(() => null)));
    const users = docs.filter(doc => doc?.exists).map(doc => ({ id: doc.id, ...doc.data() }));
    host.innerHTML = users.length ? users.map(friend => {
      const isMe = friend.id === state.user?.uid;
      const isFriend = (state.profile?.friends || []).includes(friend.id);
      const canMessage = !isMe && isFriend;
      return `
        <div style="display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid var(--border)">
          <div onclick="closeModal();openProfile('${friend.id}')" style="cursor:pointer">${avatar(friend.displayName, friend.photoURL, 'avatar-sm')}</div>
          <div style="flex:1;min-width:0">
            <div style="font-weight:600;font-size:14px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(friend.displayName)}${verifiedBadge(friend.id)}</div>
            <div style="font-size:12px;color:var(--text-secondary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(friend.major || 'Student')}</div>
          </div>
          <button class="btn-outline btn-sm" onclick="closeModal();openProfile('${friend.id}')">View</button>
          ${canMessage ? `<button class="btn-primary btn-sm" onclick="closeModal();startChat('${friend.id}','${esc(friend.displayName)}','${friend.photoURL || ''}')">Message</button>` : ''}
        </div>`;
    }).join('') : '<div class="empty-state"><h3>No friends yet</h3></div>';
  } catch (e) {
    console.error(e);
    toast('Could not load friends');
  }
}

async function rejectFriendRequest(fromUid) {
  const uid = state.user.uid;
  try {
    closeNotifDropdown();
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
    const data = doc.data() || {};
    const prevName = state.profile?.displayName || '';
    const prevPhoto = state.profile?.photoURL || '';
    Object.assign(state.profile, data);
    state.profile.friends = data.friends || [];
    state.profile.friendRequests = sanitizeFriendRequests(data.friendRequests || []);
    state.profile.sentRequests = data.sentRequests || [];
    state.profile.blockedUsers = data.blockedUsers || [];
    state.profile.blockedBy = data.blockedBy || [];
    state.profile.allowAnonymousMessages = data.allowAnonymousMessages !== false;
    updateCachedUserRecord(state.user.uid, state.profile || {});
    setupHeader();
    updateAnonPrefButton('messages-anon-pref');
    if ((data.displayName || '') !== prevName || (data.photoURL || '') !== prevPhoto) {
      if (state.page === 'chat') refreshCurrentMessageList();
    }
    updateNotifBadge();
    const dd = $('#notif-dropdown');
    if (dd && dd.style.display === 'block') loadNotifications();
  }, err => {
    console.warn('User listener error:', err);
    recoverInvalidSession(err, 'User profile listener denied').catch(() => {});
  });

  generalNotifUnsub = db.collection('users').doc(state.user.uid).collection('notifications')
    .orderBy('createdAt', 'desc')
    .limit(30)
    .onSnapshot(snap => {
      _notifications = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      _notifications.sort((a,b) => (b.createdAt?.seconds||0) - (a.createdAt?.seconds||0));
      _notifications = _notifications.slice(0, 20);
      maybeNotifyForGeneralNotifications(bellNotifications(_notifications));
      updateNotifBadge();
      const dd = $('#notif-dropdown');
      if (dd && dd.style.display === 'block') loadNotifications();
    }, err => {
      console.warn('Notif listener error:', err);
      _notifications = [];
      updateNotifBadge();
      recoverInvalidSession(err, 'Notifications listener denied').catch(() => {});
    });
}

function updateNotifBadge() {
  const requests = sanitizeFriendRequests(state.profile.friendRequests || []);
  const bellNotifs = bellNotifications(_notifications);
  const unreadCount = bellNotifs.filter(n => !n.read).length;
  const pendingAsg = _asgPendingAlerts.reduce((sum, g) => sum + (g.pendingRequests || []).length, 0);
  const revealCount = bellNotifs.filter(n => n.type === 'reveal_request' && !n.read).length;
  const dot = $('#notif-dot');
  if (dot) dot.style.display = (requests.length > 0 || unreadCount > 0 || pendingAsg > 0 || revealCount > 0) ? 'block' : 'none';
}

function loadNotifications() {
  const dd = $('#notif-dropdown');
  const requests = sanitizeFriendRequests(state.profile.friendRequests || []);
  const asgAlerts = _asgPendingAlerts;
  const notifs = bellNotifications(_notifications);
  
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
      const convoId = n.payload?.convoId || '';
      return `
      <div class="notif-item ${n.read ? '' : 'unread'}" onclick="closeNotifDropdown();openChat('${convoId}');markNotifRead('${n.id}')">
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
      <div class="notif-item unread" onclick="closeNotifDropdown();openProfile('${r.uid}')">
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
      <div class="notif-item unread" onclick="closeNotifDropdown();openGroupDetail('${g.id}')">
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
      const icon = n.type === 'like' ? '❤️' : n.type === 'comment' ? '💬' : n.type === 'module' ? '📚' : n.type === 'group' ? '📋' : n.type === 'message' ? '✉️' : n.type === 'friend_request' ? '👋' : n.type === 'friend_accept' ? '🤝' : n.type === 'live' ? '🔴' : '🔔';
      const isOwnLive = !!n.from?.uid && n.from.uid === state.user?.uid;
      const clickAction = n.payload?.convoId
        ? `closeNotifDropdown();openChat('${n.payload.convoId}');markNotifRead('${n.id}')`
        : n.payload?.streamId
          ? `closeNotifDropdown();openLiveStreamFromFeed('${n.payload.streamId}', ${isOwnLive ? 'true' : 'false'});markNotifRead('${n.id}')`
          : n.payload?.postId
            ? `closeNotifDropdown();viewPost('${n.payload.postId}');markNotifRead('${n.id}')`
            : n.payload?.groupId
              ? `closeNotifDropdown();openGroupDetail('${n.payload.groupId}');markNotifRead('${n.id}')`
              : n.from?.uid && n.from.uid !== 'anonymous'
                ? `closeNotifDropdown();openProfile('${n.from.uid}');markNotifRead('${n.id}')`
                : `closeNotifDropdown();markNotifRead('${n.id}')`;
      const from = n.from || { name: 'Unibo', photo: null };
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

async function markAllBellNotifsRead() {
  if (!state.user?.uid) return;
  const unreadIds = bellNotifications(_notifications).filter(n => !n.read).map(n => n.id);
  if (!unreadIds.length) return;
  const unreadSet = new Set(unreadIds);
  _notifications = _notifications.map(n => unreadSet.has(n.id) ? { ...n, read: true } : n);
  updateNotifBadge();
  await Promise.all(unreadIds.map(id =>
    db.collection('users').doc(state.user.uid).collection('notifications').doc(id).update({ read: true }).catch(() => {})
  ));
}

async function addNotification(targetId, type, text, payload, { anonymous = false, docId = null, fromOverride = null } = {}) {
  if (targetId === state.user.uid) return;
  try {
    const from = fromOverride || {
      uid: anonymous ? 'anonymous' : state.user.uid,
      name: anonymous ? 'Anonymous' : state.profile.displayName,
      photo: anonymous ? null : (state.profile.photoURL || null)
    };
    const data = {
      type, text, payload, read: false, createdAt: FieldVal.serverTimestamp(),
      from
    };
    await dispatchNotificationGateway(targetId, data, { docId });
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
    const rawAuthorPhoto = p.isAnonymous ? null : (p.authorPhoto || resolvePostAuthorPhoto(p));
    const authorIdentity = p.isAnonymous
      ? { name: getAnonymousLabelForPost(p), photo: null }
      : getResolvedUserIdentity(p.authorId, p.authorName || 'User', rawAuthorPhoto);
    const displayAuthorName = authorIdentity.name || p.authorName || 'User';
    const displayAuthorPhoto = p.isAnonymous ? null : (authorIdentity.photo || rawAuthorPhoto);

    let videoPlayerData = null;
    if (hasVideo && mediaURL) { videoPlayerData = createVideoPlayer(mediaURL); }

    openModal(`
      <div class="modal-header"><h2>Post</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
      <div class="modal-body" style="padding:16px">
        <div class="post-card" data-post-id="${p.id}" style="box-shadow:none;border:none;margin:0;padding:0">
          ${p.repostOf ? `<div class="repost-badge" style="margin:-0 -0 10px">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
            <span>Reposted by ${esc(displayAuthorName)}</span>
          </div>` : ''}
          <div class="post-header">
            ${p.isAnonymous ? `<div class="avatar-md anon-avatar" onclick="closeModal();openAnonPostActions('${p.authorId}')" style="cursor:pointer">👻</div>` : `<div onclick="closeModal();openProfile('${p.authorId}')" style="cursor:pointer">${avatar(displayAuthorName, displayAuthorPhoto, 'avatar-md')}</div>`}
            <div class="post-header-info">
              <div class="post-author-name" ${p.isAnonymous ? `onclick="closeModal();openAnonPostActions('${p.authorId}')" style="cursor:pointer"` : `onclick="closeModal();openProfile('${p.authorId}')"`}>${p.isAnonymous ? esc(getAnonymousLabelForPost(p)) : esc(displayAuthorName) + verifiedBadge(p.authorId)}</div>
              <div class="post-meta">${timeAgo(p.createdAt)}</div>
            </div>
          </div>
          ${p.content ? renderExpandablePostContent(p.content, `modal-${p.id}`, 180) : ''}
          ${renderPostModuleTags(p.moduleTags || [])}
          ${renderPostHashTags(getPostHashTags(p).filter(tag => !(p.moduleTags || []).includes(tag.toUpperCase())))}
          ${hasImage && mediaURL ? `<div class="post-media-wrap"><img src="${mediaURL}" class="post-image" onclick="viewImage('${mediaURL}')" style="max-height:300px"></div>` : ''}
          ${!p.repostOf && hasVideo && videoPlayerData ? videoPlayerData.html : ''}
          ${p.repostOf ? renderQuoteEmbed(p.repostOf, { repostStyle: true }) : ''}
          <div class="post-actions" style="border-top:1px solid var(--border);padding-top:12px;margin-top:12px">
            ${p.isAnonymous && p.authorId !== state.user.uid ? `<button class="post-action anon-inline-action" onclick="closeModal();openAnonPostActions('${p.authorId}')">👻 Message</button>` : ''}
            <button class="post-action post-like-action modal-like-action ${liked ? 'liked' : ''}" data-post-id="${p.id}" onclick="toggleLike('${p.id}')">❤ ${lc || 'Like'}</button>
            <button class="post-action" onclick="closeModal();openComments('${p.id}')">💬 ${cc || 'Comment'}</button>
            <button class="post-action" onclick="closeModal();openShareModal('${p.id}')">↗ Share</button>
          </div>
        </div>
      </div>
    `);

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
  syncMessagesStoriesVisibility();
  container.innerHTML = `<div class="convo-list" id="convo-list"><div style="padding:40px;text-align:center"><span class="inline-spinner"></span></div></div>`;

  unsub();
  const u = db.collection('conversations')
    .where('participants', 'array-contains', state.user.uid)
    .onSnapshot(async snap => {
      const convos = snap.docs
        .map(d => ({ id: d.id, ...d.data() }))
        .sort((a, b) => (b.updatedAt?.seconds || 0) - (a.updatedAt?.seconds || 0));

      const el = $('#convo-list'); if (!el) return;
      if (!convos.length) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">💬</div><h3>No chats yet</h3><p>Visit a profile to start a conversation</p></div>`;
        return;
      }

      const uid = state.user.uid;
      const otherUids = convos.map(c => c.participants.find(p => p !== uid)).filter(Boolean);
      await ensureUserContextCache(otherUids);
      let archivedUnread = 0;
      convos.forEach(c => {
        if ((c.archived || []).includes(uid)) archivedUnread += (c.unread || {})[uid] || 0;
      });
      const archBadge = $('#archive-fab-badge');
      if (archBadge) {
        archBadge.textContent = archivedUnread || '';
        archBadge.style.display = archivedUnread ? 'flex' : 'none';
      }
      el.innerHTML = convos.map(c => {
        const idx = c.participants.indexOf(uid) === 0 ? 1 : 0;
        const otherUid = c.participants[idx];
        const rawName = (c.participantNames || [])[idx] || 'User';
        const rawPhoto = (c.participantPhotos || [])[idx] || null;
        const otherStatus = (c.participantStatuses || {})[otherUid] || 'offline';
        const theirAnon = !!((c.anonymous || {})[otherUid]);
        const resolved = getResolvedUserIdentity(otherUid, rawName, rawPhoto);
        const displayName = theirAnon ? getAnonDisplayName(c, uid, otherUid) : resolved.name;
        const displayPhoto = theirAnon ? null : resolved.photo;
        const previewText = formatConversationLastMessage(c.lastMessage || '', c, uid, displayName);
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
              <div class="convo-last-msg">${esc(previewText || 'Start chatting...')}</div>
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
  syncMessagesStoriesVisibility();
  container.innerHTML = `<div class="convo-list" id="convo-list"><div style="padding:40px;text-align:center"><span class="inline-spinner"></span></div></div>`;

  unsub();
  const u = db.collection('conversations')
    .where('participants', 'array-contains', state.user.uid)
    .onSnapshot(async snap => {
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
      const otherUids = convos.map(c => c.participants.find(p => p !== uid)).filter(Boolean);
      await ensureUserContextCache(otherUids);
      el.innerHTML = convos.map(c => {
        const idx = c.participants.indexOf(uid) === 0 ? 1 : 0;
        const otherUid = c.participants[idx];
        const rawName = (c.participantNames || [])[idx] || 'User';
        const rawPhoto = (c.participantPhotos || [])[idx] || null;
        const otherStatus = (c.participantStatuses || {})[otherUid] || 'offline';
        const theirAnon = !!((c.anonymous || {})[otherUid]);
        const resolved = getResolvedUserIdentity(otherUid, rawName, rawPhoto);
        const displayName = theirAnon ? getAnonDisplayName(c, uid, otherUid) : resolved.name;
        const displayPhoto = theirAnon ? null : resolved.photo;
        const previewText = formatConversationLastMessage(c.lastMessage || '', c, uid, displayName);
        const avatarHtml = theirAnon
          ? '<div class="avatar-md anon-avatar">👻</div>'
          : avatar(displayName, displayPhoto, 'avatar-md');
        const unread = (c.unread || {})[uid] || 0;
        return `
          <div class="convo-item ${unread ? 'unread' : ''}" onclick="openChat('${c.id}')">
            <div class="convo-avatar">${avatarHtml}${otherStatus === 'online' ? '<span class="online-indicator"></span>' : ''}</div>
            <div class="convo-info">
              <div class="convo-name">${esc(displayName)}</div>
              <div class="convo-last-msg">${esc(previewText || 'Start chatting...')}</div>
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
  const rawOtherName = (convo.participantNames || [])[idx] || realName || 'User';
  const rawOtherPhoto = (convo.participantPhotos || [])[idx] || realPhoto || null;
  const resolvedIdentity = getResolvedUserIdentity(otherUid, rawOtherName, rawOtherPhoto);
  const otherName = resolvedIdentity.name;
  const otherPhoto = resolvedIdentity.photo;
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
      <div onclick="openProfile('${otherUid}')" style="cursor:pointer">${avatar(otherName, otherPhoto, 'avatar-sm')}</div>
      <div onclick="openProfile('${otherUid}')" style="cursor:pointer"><h3 style="font-size:15px;font-weight:700">${esc(otherName)}</h3><span style="font-size:11px;color:var(--text-tertiary)">${otherStatus === 'online' ? 'Online' : otherStatus === 'study' ? 'Studying' : 'Offline'}</span></div>
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
        }).then(() => syncConversationIdentities(convo._id)).catch(() => {});
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

    // Show accept button for anon chat recipient
    if (convo.isAnonymous && !convo.anonAccepted && convo.anonStartedBy && convo.anonStartedBy !== uid) {
      const acceptHtml = `<button class="btn-primary btn-sm" style="margin-left:8px" onclick="acceptAnonChat('${convo._id}')">Accept Chat</button>`;
      banner.innerHTML += acceptHtml;
      banner.style.display = 'flex';
    }
  }
}

async function acceptAnonChat(convoId) {
  try {
    await db.collection('conversations').doc(convoId).update({ anonAccepted: true });
    toast('Chat accepted — they can now send more messages');
  } catch (e) { toast('Failed'); console.error(e); }
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
    if (currentAnon) await syncConversationIdentities(convoId);
    toast(!currentAnon ? `You are now ${getUserAnonIdentity(state.profile || {})}` : 'Identity revealed');
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
    await addNotification(otherUid, 'reveal_request', 'Someone wants to reveal their identity in an anonymous chat', { convoId }, { anonymous: true });
    
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

async function syncConversationIdentities(convoId) {
  try {
    const convoSnap = await db.collection('conversations').doc(convoId).get();
    if (!convoSnap.exists) return;
    const convo = convoSnap.data() || {};
    const participants = convo.participants || [];
    if (participants.length !== 2) return;
    const anonMap = convo.anonymous || {};
    const anyStillAnon = participants.some(participantId => !!anonMap[participantId]);
    const userDocs = await Promise.all(participants.map(participantId => db.collection('users').doc(participantId).get()));
    const participantNames = userDocs.map(doc => doc.exists ? (doc.data().displayName || doc.data().firstName || 'User') : 'User');
    const participantPhotos = userDocs.map(doc => doc.exists ? (doc.data().photoURL || null) : null);
    await db.collection('conversations').doc(convoId).update({
      participantNames,
      participantPhotos,
      isAnonymous: anyStillAnon
    });
  } catch (e) {
    console.warn('syncConversationIdentities failed', e);
  }
}

function updateAnonPrefButton(buttonId = 'messages-anon-pref') {
  const btn = document.getElementById(buttonId);
  if (!btn) return;
  const enabled = allowAnonymousDMsFor(state.profile || {});
  btn.classList.toggle('pref-on', enabled);
  btn.classList.toggle('pref-off', !enabled);
  btn.title = enabled ? 'Anonymous messages allowed' : 'Anonymous messages disabled';
}

function openAnonDmSettings() {
  const enabled = allowAnonymousDMsFor(state.profile || {});
  openModal(`
    <div class="modal-header"><h2>Anonymous Messages</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      <div class="about-item" style="margin-bottom:14px">
        <span class="about-icon">👻</span>
        <div>
          <div class="about-label">Current Rule</div>
          <div class="about-value">${enabled ? 'Non-friends can start anonymous chats' : 'Only friends can message you'}</div>
        </div>
      </div>
      <button class="btn-primary btn-full" onclick="setAllowAnonymousMessages(${enabled ? 'false' : 'true'})">${enabled ? 'Turn Off Anonymous Messages' : 'Turn On Anonymous Messages'}</button>
      <button class="btn-secondary btn-full" style="margin-top:10px" onclick="closeModal()">Close</button>
    </div>
  `);
}

async function setAllowAnonymousMessages(enabled) {
  try {
    await db.collection('users').doc(state.user.uid).update({ allowAnonymousMessages: !!enabled });
    state.profile.allowAnonymousMessages = !!enabled;
    updateAnonPrefButton('messages-anon-pref');
    closeModal();
    toast(enabled ? 'Anonymous messages enabled' : 'Anonymous messages turned off');
  } catch (e) {
    console.error(e);
    toast('Failed to update setting');
  }
}

async function openChat(convoId) {
  showScreen('chat-view');
  _activeChatConvoId = convoId;
  _activeGroupChat = { id: '', collection: '' };
  const msgs = $('#chat-msgs');
  if (msgs) msgs.innerHTML = '<div style="text-align:center;padding:32px"><span class="inline-spinner"></span></div>';
  try {
    const convoDoc = await db.collection('conversations').doc(convoId).get();
    if (!convoDoc.exists) return toast('Chat not found');
    const convo = convoDoc.data();
    convo._id = convoId;
    const uid = state.user.uid;
    const idx = convo.participants.indexOf(uid) === 0 ? 1 : 0;
    const otherUid = convo.participants[idx];
    await ensureUserContextCache([otherUid], { force: true });
    const resolvedIdentity = getResolvedUserIdentity(otherUid, (convo.participantNames || [])[idx] || 'User', (convo.participantPhotos || [])[idx] || null);
    const name = resolvedIdentity.name;
    const photo = resolvedIdentity.photo;


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
            if (m.deleted || m.type === 'deleted') {
              content = '<span class="msg-deleted">Message deleted</span>';
            }
            if (m.audioURL) content += renderVoiceMsg(m.audioURL);
            if (m.poll) content += renderInteractivePoll(m.poll, 'dm', convoId, m.id);
            if (m.locationPin) content = renderLocationPinCard(m.locationPin);
            if (m.imageURL) {
              const isVideoMedia = m.mediaType === 'video' || /\.(mp4|webm|mov|m4v)(\?|#|$)/i.test(m.imageURL || '');
              if (isVideoMedia) content += `<video src="${m.imageURL}" class="msg-image" style="background:#000" controls playsinline preload="metadata"></video>`;
              else content += `<img src="${m.imageURL}" class="msg-image" onclick="viewImage('${m.imageURL}')">`;
            }
            // Handle shared post messages
            if (!m.deleted && m.type === 'share_post' && m.payload?.postId) {
              const pl = m.payload;
              let mediaPreview = '';
              if (pl.mediaURL && pl.mediaType === 'video') {
                mediaPreview = `<div style="position:relative;border-radius:8px;overflow:hidden;margin-bottom:6px;max-height:140px">
                  <video class="inline-video-preview" src="${pl.mediaURL}" style="width:100%;max-height:140px;object-fit:cover;display:block" preload="metadata" muted playsinline></video>
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
            } else if (!m.deleted && m.type === 'story_reply' && m.payload) {
              const sp = m.payload;
              let storyThumb = '';
              if (sp.storyPreview && sp.storyType === 'photo') {
                storyThumb = `<img src="${sp.storyPreview}" style="width:100%;max-height:100px;object-fit:cover;border-radius:6px;margin-bottom:6px;display:block">`;
              } else if (sp.storyPreview && sp.storyType === 'video') {
                storyThumb = `<div style="position:relative;border-radius:6px;overflow:hidden;margin-bottom:6px;max-height:100px">
                  <video src="${sp.storyPreview}" style="width:100%;max-height:100px;object-fit:cover;display:block" preload="metadata" muted></video>
                  <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.25)">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="white"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                  </div>
                </div>`;
              }
              const captionSnip = sp.storyCaption ? `<div style="font-size:11px;color:var(--text-tertiary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(sp.storyCaption)}</div>` : '';
              content = `<div style="background:var(--bg-secondary);border-radius:8px;padding:8px;margin-bottom:6px;border-left:3px solid var(--accent)">
                <div style="display:flex;align-items:center;gap:4px;margin-bottom:4px">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                  <span style="font-size:11px;font-weight:600;color:var(--accent)">Replied to story</span>
                </div>
                ${storyThumb}${captionSnip}
              </div>${esc(m.text)}`;
            } else if (!m.deleted && m.type === 'story_like') {
              const otherShort = (name || 'They').split(' ')[0] || 'They';
              const label = isMe ? '❤️ You liked their story' : `❤️ ${esc(otherShort)} liked your story`;
              content = `<div class="msg-system-pill msg-system-pill-story">${label}</div>`;
            } else if (!m.deleted && m.type === 'message_reaction' && m.payload) {
              const otherShort = (name || 'They').split(' ')[0] || 'They';
              const reactionEmoji = esc(m.payload.emoji || '❤️');
              const targetMine = m.payload.targetSenderId === uid;
              const label = isMe
                ? `${reactionEmoji} You reacted to their message`
                : targetMine
                  ? `${reactionEmoji} ${esc(otherShort)} reacted to your message`
                  : `${reactionEmoji} ${esc(otherShort)} reacted`;
              content = `<div class="msg-system-pill msg-system-pill-reaction">${label}</div>`;
            } else if (!m.deleted && m.text && !m.text.startsWith('shared post::')) {
              content += esc(m.text);
            } else if (!m.deleted && m.text && m.text.startsWith('shared post::')) {
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
            const liveIdentity = getResolvedUserIdentity(m.senderId, name, photo);
            const displayName = (!isMe && senderAnon) ? 'Anonymous' : liveIdentity.name;
            const displayPhoto = (!isMe && senderAnon) ? null : liveIdentity.photo;
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
            const reactionSummary = renderReactionSummary(m.reactions || {}, [], 'msg-inline');
            const allowReply = !m.deleted && !['story_like', 'message_reaction'].includes(m.type || '');
            return `${dateSep}<div class="msg-row ${isMe ? 'msg-row-sent' : 'msg-row-received'}" id="msg-${m.id}">
              ${!isMe ? `<div class="msg-avatar-wrap">${avatarHTML}</div>` : ''}
              <div class="msg-stack ${isMe ? 'msg-stack-sent' : 'msg-stack-received'}"><div class="msg-bubble ${isMe ? 'msg-sent' : 'msg-received'} ${newCls} ${m.locationPin ? 'msg-bubble-pin' : ''}" data-message-id="${m.id}">${m.replyToId && m.replyToText ? `<div class="msg-reply-snippet" onclick="jumpToMessage('${m.replyToId}','chat-msgs')">↩ ${esc(replyDisplayName)}: ${esc(clampText(m.replyToText, 50))}</div>` : ''}${content}${allowReply ? `<button class="msg-reply-btn" title="Reply" aria-label="Reply" onclick="setDmReply('${m.id}')"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 17 4 12 9 7"></polyline><path d="M20 18v-2a4 4 0 0 0-4-4H4"></path></svg></button>` : ''}<div class="msg-time">${ts ? chatTime(ts) : ''}${statusIcon}</div></div>${reactionSummary ? `<div class="msg-reaction-line" onclick="event.stopPropagation();openMessageActionSheet('dm','${convoId}','${m.id}')">${reactionSummary}</div>` : ''}</div>
            </div>`;
          }).join('');
          primeInlineVideoPreviews(msgs);
          bindMessageLongPress(msgs, 'dm', convoId);
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

      // Per-recipient anonymous message limit
      if (convo.isAnonymous && convo.anonStartedBy === uid && !convo.anonAccepted) {
        const currentCount = convo.anonMsgCount || 0;
        if (currentCount >= 2) {
          toast('Limit reached — wait for them to accept the chat');
          return;
        }
      }
      
      const text = input.value.trim();
      const moderation = applySoftKeywordFilterText(text);
      const safeText = moderation.text;
      let img = chatPendingImg;
      const chatFile = window._chatFile || null;
      if (!safeText && !img && !window._chatVoiceBlob) return;
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
        let mediaType = null;
        if (chatFile) {
          imageURL = await uploadToR2(chatFile, 'chat-images');
          mediaType = (chatFile.type || '').startsWith('video/') ? 'video' : 'image';
        }
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
        let freshConvoData = null;
        try {
          const freshConvo = await db.collection('conversations').doc(convoId).get();
          if (freshConvo.exists) {
            freshConvoData = freshConvo.data();
            senderAnon = !!(freshConvoData.anonymous || {})[uid];
          }
        } catch (_) {}
        await db.collection('conversations').doc(convoId).collection('messages').add({
          text: safeText || '', imageURL: imageURL || null, mediaType: mediaType || null, audioURL: audioURL || null,
          moderationFlags: moderation.flags,
          senderId: uid, senderAnon, ...replyPayload,
          createdAt: FieldVal.serverTimestamp(), status: 'sent'
        });
        const lastMsg = audioURL ? '🎤 Voice' : imageURL ? (mediaType === 'video' ? '📹 Video' : (safeText || '📷 Photo')) : safeText;
        const mergeData = {
          lastMessage: lastMsg, updatedAt: FieldVal.serverTimestamp(),
          unread: { [otherUid]: FieldVal.increment(1), [uid]: 0 }
        };
        // Increment anon msg counter for initiator
        if (freshConvoData?.isAnonymous && freshConvoData?.anonStartedBy === uid && !freshConvoData?.anonAccepted) {
          mergeData.anonMsgCount = FieldVal.increment(1);
          convo.anonMsgCount = (convo.anonMsgCount || 0) + 1;
        }
        await db.collection('conversations').doc(convoId).set(mergeData, { merge: true });
        await addNotification(otherUid, 'message', 'sent you a message', { convoId });
      } catch (e) { console.error(e); }
    };
    $('#chat-send').onclick = sendMsg;
    const dmPlusBtn = $('#chat-plus-btn'); if (dmPlusBtn) dmPlusBtn.onclick = (e) => { e.stopPropagation(); openChatPlusMenu('dm'); };
    input.onfocus = () => setTimeout(() => scrollToLatest(msgs), 100);
    input.onblur = () => setTimeout(() => scrollToLatest(msgs), 150);

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
      _activeChatConvoId = '';
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
    const existing = snap.docs.find(d => isConversationWithUser(d.data() || {}, uid, { anonymous: false }));
    if (existing) {
      await existing.ref.set({ archived: FieldVal.arrayRemove(state.user.uid, uid) }, { merge: true }).catch(() => {});
      openChat(existing.id);
    }
    else {
      const doc = await db.collection('conversations').add({
        participants: [state.user.uid, uid],
        participantNames: [state.profile.displayName, name],
        participantPhotos: [state.profile.photoURL || null, photo || null],
        lastMessage: '', updatedAt: FieldVal.serverTimestamp(),
        unread: { [uid]: 0, [state.user.uid]: 0 },
        participantStatuses: { [state.user.uid]: state.status, [uid]: 'offline' },
        isAnonymous: false,
        anonymous: { [state.user.uid]: false, [uid]: false },
        archived: []
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
    let targetAllowsAnon = true;
    const userDoc = await db.collection('users').doc(uid).get();
    if (userDoc.exists) {
      const userData = userDoc.data() || {};
      if (!targetName || targetName === 'Anonymous' || targetName.includes('Anonymous')) {
        targetName = userData.displayName || userData.firstName || 'User';
        targetPhoto = userData.photoURL || '';
      }
      targetAllowsAnon = allowAnonymousDMsFor(userData);
    }
    if (!targetAllowsAnon) return toast('This user only accepts messages from friends');

    // Check for existing anon conversation
    const snap = await db.collection('conversations').where('participants', 'array-contains', state.user.uid).get();
    const existing = snap.docs
      .map(doc => ({ id: doc.id, ...doc.data() }))
      .filter(data => {
        const stillAnonymous = !!((data.anonymous || {})[state.user.uid]) || !!((data.anonymous || {})[uid]);
        return (data.participants || []).includes(uid) && data.isAnonymous && stillAnonymous;
      })
      .sort((a, b) => (b.updatedAt?.seconds || 0) - (a.updatedAt?.seconds || 0))[0];
    if (existing) { openChat(existing.id); return; }

    // Increment anon usage
    await db.collection('users').doc(state.user.uid).update({
      anonUsageToday: todayCount + 1,
      anonUsageDate: today
    });
    state.profile.anonUsageToday = todayCount + 1;
    state.profile.anonUsageDate = today;

    // Create anonymous conversation
    const myAnonIdentity = getUserAnonIdentity(state.profile || {});
    const doc = await db.collection('conversations').add({
      participants: [state.user.uid, uid],
      participantNames: [myAnonIdentity, defaultAnonLabel(uid)],
      participantPhotos: [null, null],
      lastMessage: '', updatedAt: FieldVal.serverTimestamp(),
      unread: { [uid]: 0, [state.user.uid]: 0 },
      participantStatuses: { [state.user.uid]: state.status, [uid]: 'offline' },
      isAnonymous: true,
      anonymous: { [state.user.uid]: true, [uid]: true },
      anonStartedBy: state.user.uid,
      anonAccepted: false,
      anonMsgCount: 0,
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

async function editContactNickname(targetUid, fallbackName = 'User') {
  if (!targetUid || targetUid === state.user?.uid) return;
  try {
    const current = getContactNickname(targetUid);
    const identity = getResolvedUserIdentity(targetUid, fallbackName || 'User', null);
    const next = window.prompt('Set a private contact name for this person', current || identity.name || 'User');
    if (next === null) return;
    const cleaned = clampText(next.trim(), 42);
    const updates = {};
    updates[`contactNicknames.${targetUid}`] = cleaned || FieldVal.delete();
    await db.collection('users').doc(state.user.uid).update(updates);
    state.profile.contactNicknames = { ...(state.profile.contactNicknames || {}) };
    if (cleaned) state.profile.contactNicknames[targetUid] = cleaned;
    else delete state.profile.contactNicknames[targetUid];
    if (state.page === 'feed') renderFeedResults(state.posts || []);
    if (state.page === 'chat') refreshCurrentMessageList();
    toast(cleaned ? 'Contact name saved' : 'Contact name removed');
  } catch (e) {
    console.error(e);
    toast('Could not save contact name');
  }
}

function showConvoActions(convoId, displayName, otherUid) {
  const hasContactAlias = !!getContactNickname(otherUid);
  openModal(`
    <div class="modal-header"><h2>Chat Actions</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" style="padding:16px">
      <p style="margin-bottom:16px;color:var(--text-secondary);font-size:13px">Manage conversation with <strong>${displayName}</strong></p>
      <button class="btn-outline btn-full" style="margin-bottom:8px" onclick="editContactNickname('${otherUid}')">${hasContactAlias ? '✎ Edit Contact Name' : '✎ Set Contact Name'}</button>
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

    const isMe = uid === state.user.uid;
    const profileIdentity = isMe
      ? { name: user.displayName || 'You', photo: user.photoURL || null }
      : getResolvedUserIdentity(uid, user.displayName || 'User', user.photoURL || null);
    const profileDisplayName = profileIdentity.name || user.displayName || 'User';

    $('#prof-top-name').textContent = profileDisplayName;

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

    const modules = user.modules || [];
    const showFriendCount = user.showFriendsCount !== false && isMe; // only show to self unless they opted in

    // KEY FIX: avatar-wrap is INSIDE profile-cover so position:absolute works relative to cover
    body.innerHTML = `
      <div class="profile-cover">
        <div class="profile-avatar-wrap">
          <div class="profile-avatar-large${user.photoURL ? ' clickable' : ''}" ${user.photoURL ? `onclick="viewImage('${user.photoURL}')"` : ''}>
            ${user.photoURL ? `<img src="${user.photoURL}" alt="">` : initials(profileDisplayName)}
          </div>
          ${user.status === 'online' ? '<div class="avatar-online-dot"></div>' : ''}
        </div>
      </div>

      <div class="profile-info">
        <div class="profile-name">${esc(profileDisplayName)}${verifiedBadge(uid)}</div>
        <div class="profile-handle">${esc(user.major || '')}${user.major && user.university ? ' · ' : ''}${esc(user.university || '')}</div>
        <div class="profile-badges">
          ${user.year ? `<span class="profile-badge">🎓 ${esc(user.year)}</span>` : ''}
          ${user.gender ? `<span class="profile-badge">🧬 ${esc(user.gender)}</span>` : ''}
          ${isMe && user.address ? `<span class="profile-badge">📍 ${esc(user.address)}</span>` : ''}
        </div>
        ${user.bio ? `<p class="profile-bio">${esc(user.bio)}</p>` : ''}
        ${modules.length ? `<div class="profile-modules">${modules.map(m => `<span class="module-chip clickable" onclick="openModuleFeed('${esc(m)}')">${esc(m)}</span>`).join('')}</div>` : ''}

        <div class="profile-stats">
          <div class="profile-stat"><div class="stat-num">${posts.length}</div><div class="stat-label">Posts</div></div>
          ${showFriendCount ? `<button class="profile-stat profile-stat-btn" onclick="openFriendsList('${uid}','${esc(user.displayName)}')"><div class="stat-num">${(user.friends || []).length}</div><div class="stat-label">Friends</div></button>` : ''}
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
                  ? `<button class="btn-primary" onclick="startChat('${uid}','${esc(user.displayName || 'User')}','${user.photoURL || ''}')">Message</button>`
                  : `${allowAnonymousDMsFor(user) ? `<button class="btn-outline anon-msg-btn" onclick="startAnonChat('${uid}','${esc(user.displayName)}','${user.photoURL || ''}', true)">👻 Anonymous Message</button>` : `<button class="btn-outline" disabled style="opacity:0.6">Anonymous Off</button>`}`;
                const contactBtn = `<button class="btn-outline" onclick="editContactNickname('${uid}')">${getContactNickname(uid) ? 'Edit Contact Name' : 'Set Contact Name'}</button>`;
                return `${msgBtn}\n               ${friendBtn}\n               ${contactBtn}`;
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
    initProfilePostInteractions();

    // Wire tabs
    $$('.profile-tab').forEach(tab => {
      tab.onclick = () => {
        $$('.profile-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const tc = $('#profile-tab-content');
        if (tab.dataset.pt === 'posts') {
          tc.innerHTML = renderProfilePosts(posts, user);
          initProfilePostInteractions();
        }
        else if (tab.dataset.pt === 'photos') {
          tc.innerHTML = renderProfilePhotos(posts);
          primeInlineVideoPreviews(tc);
        } else {
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
    const liked = (p.likes || []).includes(state.user.uid);
    const myReaction = getUserReaction(p.reactions, p.likes || []);
    const commentCount = p.commentsCount || 0;
    const hasVideo = p.videoURL || (p.mediaType === 'video');
    const hasImage = p.imageURL && !hasVideo;
    const mediaURL = hasVideo ? (p.videoURL || p.imageURL) : p.imageURL;
    let videoPlayerData = null;
    if (hasVideo && mediaURL) {
      videoPlayerData = createVideoPlayer(mediaURL);
      _profPlayers.push(videoPlayerData);
    }
    return `
    <div class="post-card" data-post-id="${p.id}">
      ${p.repostOf ? `<div style="padding-bottom:6px;margin-bottom:6px;font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:6px">
         <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
         Reposted by ${esc(user.displayName)}
       </div>` : ''}
      <div class="post-header">
        ${p.isAnonymous ? `<div class="avatar-md anon-avatar">👻</div>` : avatar(user.displayName, user.photoURL, 'avatar-md')}
        <div class="post-header-info">
          <div class="post-author-name">${p.isAnonymous ? esc(getAnonymousLabelForPost(p)) : esc(user.displayName)}</div>
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
      ${p.repostOf ? renderQuoteEmbed(p.repostOf, { repostStyle: true }) : ''}
      <div class="post-engagement">
        <div class="post-stats">${renderPostStatsMarkup(p)}</div>
      </div>
      <div class="post-actions">
        <button class="post-action post-like-action ${liked ? 'liked' : ''} ${myReaction && myReaction !== '❤️' ? 'reacted' : ''}" data-post-id="${p.id}" data-source="profile" onclick="toggleLike('${p.id}')">${renderPostLikeMarkup(p)}</button>
        <button class="post-action" onclick="openComments('${p.id}')">💬 ${commentCount || 'Comment'}</button>
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

async function notifyAdminOfReport(reportData = {}) {
  try {
    const adminSnap = await db.collection('users').where('email', '==', ADMIN_EMAIL).limit(1).get();
    if (adminSnap.empty) return;
    const adminId = adminSnap.docs[0].id;
    if (!adminId || adminId === state.user?.uid) return;
    await addNotification(
      adminId,
      'report',
      `flagged a post for ${reportData.reason || 'review'}`,
      {
        postId: reportData.postId || '',
        reason: reportData.reason || '',
        reportsCount: reportData.reportsCount || 0,
        kind: 'post_report'
      }
    );
  } catch (e) {
    console.warn('notifyAdminOfReport failed', e);
  }
}

async function submitPostReport(postId) {
  const reason = document.querySelector('input[name="report-reason"]:checked')?.value;
  if (!reason) return toast('Select a reason');
  try {
    const postRef = db.collection('posts').doc(postId);
    const postDoc = await postRef.get();
    if (!postDoc.exists) return toast('Post no longer exists');
    const post = postDoc.data() || {};
    if ((post.reportedBy || []).includes(state.user.uid)) {
      closeModal();
      return toast('You already reported this post');
    }
    const nextReportsCount = (post.reportsCount || 0) + 1;
    const updates = {
      reportsCount: FieldVal.increment(1),
      reportedBy: FieldVal.arrayUnion(state.user.uid),
      lastReportReason: reason,
      lastReportedAt: FieldVal.serverTimestamp()
    };
    if (nextReportsCount >= 3) {
      updates.shadowHidden = true;
      updates.moderationStatus = 'shadow-hidden';
    }
    await postRef.update(updates);
    await db.collection('reports').add({
      type: 'post',
      postId,
      reporterId: state.user.uid,
      targetAuthorId: post.authorId || '',
      reason,
      createdAt: FieldVal.serverTimestamp(),
      status: 'open'
    });
    notifyAdminOfReport({ postId, reason, reportsCount: nextReportsCount });
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
      await _wipeCollection(`users/${d.id}/pushTokens`);
    }

    // Then wipe main collections
    const collections = ['posts', 'groups', 'conversations', 'events', 'assignmentGroups', 'stories', 'listings', 'stats', 'users'];
    for (const col of collections) await _wipeCollection(col);

    toast('Kill switch complete. Data wiped.');
    navigate('feed');
  setTimeout(() => markBootReady('Loaded'), 220);
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
      <div class="about-item"><span class="about-icon">🧬</span><div><div class="about-label">Gender</div><div class="about-value">${esc(user.gender || 'Not set')}</div></div></div>
      <div class="about-item"><span class="about-icon">📅</span><div><div class="about-label">Year</div><div class="about-value">${esc(user.year || 'Not set')}</div></div></div>
      ${isMe && user.address ? `<div class="about-item"><span class="about-icon">📍</span><div><div class="about-label">Location</div><div class="about-value">${esc(user.address)}</div></div></div>` : ''}
      ${modules.length ? `<div class="about-item"><span class="about-icon">🧩</span><div><div class="about-label">Modules</div><div class="about-modules">${modules.map(m => `<span class="module-chip">${esc(m)}</span>`).join('')}</div></div></div>` : ''}
      ${isMe ? `<div class="about-item"><span class="about-icon">👻</span><div><div class="about-label">Anonymous Messages</div><div class="about-value">${allowAnonymousDMsFor(user) ? 'Allowed from non-friends' : 'Friends only'}</div></div></div>` : ''}
      ${user.joinedAt ? `<div class="about-item"><span class="about-icon">🗓</span><div><div class="about-label">Joined</div><div class="about-value">${timeAgo(user.joinedAt)}</div></div></div>` : ''}
      ${isMe ? `<div class="about-item"><span class="about-icon">🚫</span><div><div class="about-label">Blocked Users</div><div id="blocked-users-list">${blockedUsers.length ? '<span class="inline-spinner"></span>' : 'None'}</div></div></div>` : ''}
      ${isMe ? `<div class="about-item backend-debug-card"><span class="about-icon">🧪</span><div><div class="about-label">Backend Diagnostics</div><div class="about-value">Appwrite + Notifications</div><div id="backend-debug-status" class="backend-debug-status">Tap a test below to run checks.</div><div class="backend-debug-actions"><button class="btn-outline btn-sm" onclick="runAppwriteBackendDiagnostics()">Test Appwrite</button><button class="btn-outline btn-sm" onclick="sendAppwritePing()">Send Ping</button><button class="btn-outline btn-sm" onclick="runNotificationDiagnostics()">Test Notifications</button><button class="btn-outline btn-sm" onclick="sendDebugLocalNotification()">Send Local Test</button><button class="btn-outline btn-sm" onclick="sendGatewayNotificationProbe()">Gateway Probe</button><button class="btn-outline btn-sm" onclick="runShadowSyncProbe()">Shadow Probe</button><button class="btn-outline btn-sm" id="backend-mirror-toggle-btn" onclick="toggleAppwriteMirror()">${shouldMirrorToAppwrite() ? 'Disable Mirror' : 'Enable Mirror'}</button></div></div></div>` : ''}
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
    return `<div class="photo-grid-item" onclick="viewImage('${m.url}')"><video class="inline-video-preview" src="${m.url}" preload="metadata" muted playsinline></video><div class="photo-grid-play">▶</div></div>`;
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
  const currentAnonIdentity = getUserAnonIdentity(p);
  const gpsStatus = getUserCoords(p)
    ? `GPS saved: ${Number(p.geoLat).toFixed(5)}, ${Number(p.geoLng).toFixed(5)}`
    : 'No GPS location saved yet';
  openModal(`
    <div class="modal-header"><h2>Edit Profile</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body">
      <div class="form-group"><label>Display Name</label><input type="text" id="edit-name" value="${esc(p.displayName)}"></div>
      <div class="form-group"><label>Bio</label><textarea id="edit-bio">${esc(p.bio || '')}</textarea></div>
      <div class="form-group"><label>Location / Res</label><input type="text" id="edit-address" value="${esc(p.address || '')}" placeholder="e.g. Potch Main Campus"></div>
      <div class="form-group" style="margin-top:-6px">
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <button type="button" class="btn-outline" onclick="saveCurrentGpsLocation()">Use Current GPS</button>
        </div>
        <p id="gps-location-status" style="color:var(--text-secondary);font-size:12px;margin-top:8px">${esc(gpsStatus)}</p>
        <p style="color:var(--text-tertiary);font-size:11px;margin-top:6px">Current GPS is saved once as your main radar location. Unibo does not track you continuously.</p>
      </div>
      <div class="form-group"><label>Modules (comma-separated)</label><input type="text" id="edit-modules" value="${esc(mods)}" placeholder="MAT101, COS132, PHY121"></div>
      <div class="form-group"><label>Gender</label><input type="text" id="edit-gender" value="${esc(p.gender || '')}" placeholder="e.g. female, male, non-binary, prefer not to say"></div>
      <div class="form-group"><label>Profile Photo</label><input type="file" accept="image/*" id="edit-photo"></div>
      <div class="form-group" style="display:flex;align-items:center;gap:8px">
        <input type="checkbox" id="edit-autofill" style="width:auto" ${p.allowAutoFill !== false ? 'checked' : ''}>
        <label for="edit-autofill" style="margin:0;font-size:14px">Allow auto-fill into groups</label>
      </div>
      <p style="color:var(--text-tertiary);font-size:11px;margin:-8px 0 12px">When enabled, group hosts can auto-fill you into their groups for your modules.</p>
      <div class="form-group" style="display:flex;align-items:center;gap:8px">
        <input type="checkbox" id="edit-anon-dm" style="width:auto" ${allowAnonymousDMsFor(p) ? 'checked' : ''}>
        <label for="edit-anon-dm" style="margin:0;font-size:14px">Allow anonymous messages from non-friends</label>
      </div>
      <p style="color:var(--text-tertiary);font-size:11px;margin:-8px 0 12px">If turned off, only friends can start chats with you.</p>
      <div class="form-group"><label>Anonymous Identity</label>
        <input type="text" id="edit-anon-custom" maxlength="32" value="${esc(currentAnonIdentity)}" placeholder="e.g. Campus Ghost #A23">
      </div>
      <div class="form-group"><label>Quick Persona Theme</label>
        <select id="edit-anon-identity-theme">
          <option value="__custom">Keep typed identity</option>
          ${ANON_PERSONA_THEMES.map(theme => `<option value="${esc(theme)}">${esc(theme)}</option>`).join('')}
        </select>
      </div>
      <p style="color:var(--text-tertiary);font-size:11px;margin:-8px 0 12px">Use the text box above directly, or pick a quick persona to auto-fill it.</p>
      <button type="button" class="btn-secondary btn-full" onclick="closeModal();openFriendsList('${state.user.uid}','${esc(p.displayName)}')" style="margin-bottom:12px">View Friends</button>
      <button class="btn-primary btn-full" id="edit-save">Save</button>
    </div>
  `);
  const themeSelect = $('#edit-anon-identity-theme');
  const customInput = $('#edit-anon-custom');
  const existingBase = ANON_PERSONA_THEMES.find(theme => currentAnonIdentity.toLowerCase().startsWith(theme.toLowerCase()));
  if (themeSelect) themeSelect.value = existingBase || '__custom';
  if (themeSelect) {
    themeSelect.onchange = () => {
      if (themeSelect.value !== '__custom' && customInput) {
        customInput.value = `${themeSelect.value} #A${String(Math.floor(scoreSeed(state.user.uid) * 1000)).padStart(3, '0')}`;
      }
    };
  }
  let newPhoto = null; let newPhotoFile = null;
  $('#edit-photo').onchange = async e => {
    if (e.target.files[0]) { newPhotoFile = e.target.files[0]; newPhoto = 'pending'; toast('Photo selected'); }
  };
  $('#edit-save').onclick = async () => {
    const name = $('#edit-name').value.trim();
    const bio = $('#edit-bio').value.trim();
    const address = $('#edit-address')?.value.trim() || '';
    const modulesRaw = $('#edit-modules').value || '';
    const modules = normalizeModules(modulesRaw.split(','));
    const gender = ($('#edit-gender')?.value || '').trim();
    if (!name) return toast('Name required');
    closeModal(); toast('Saving...');
    const allowAutoFill = $('#edit-autofill')?.checked !== false;
    const allowAnonymousMessages = $('#edit-anon-dm')?.checked !== false;
    const selectedTheme = $('#edit-anon-identity-theme')?.value || '__custom';
    let anonIdentity = ($('#edit-anon-custom')?.value || '').trim();
    if (!anonIdentity && selectedTheme !== '__custom') anonIdentity = `${selectedTheme} #A${String(Math.floor(scoreSeed(state.user.uid) * 1000)).padStart(3, '0')}`;
    if (!anonIdentity) anonIdentity = buildDefaultAnonIdentity(state.user.uid, state.profile?.firstName || '');
    anonIdentity = clampText(anonIdentity, 32);
    const updates = { displayName: name, bio, modules, address, gender, allowAutoFill, allowAnonymousMessages, anonIdentity };
    if (newPhotoFile) { updates.photoURL = await uploadToR2(newPhotoFile, 'profile'); }
    try {
      await db.collection('users').doc(state.user.uid).update(updates);
      Object.assign(state.profile, updates);
      updateCachedUserRecord(state.user.uid, { ...state.profile, ...updates });
      shadowSyncUserProfile(state.user.uid, { ...state.profile, ...updates });
      if (name !== state.user.displayName) await state.user.updateProfile({ displayName: name });
      setupHeader();
      if (state.page === 'chat') { loadDMList().catch(() => {}); }
      toast('Profile updated!'); openProfile(state.user.uid);
    } catch (e) { toast('Failed'); console.error(e); }
  };
}

// ─── Voice Recording ─────────────────────────────
let _voiceRecorder = null;
let _voiceChunks = [];
let _voiceInterval = null;
let _voiceStartTime = 0;
let _voiceContext = ''; // '' for DM, 'gchat' for group

async function ensureMicrophoneStream() {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast('Microphone is not supported on this device');
    return null;
  }
  try {
    return await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
      toast(isNativeApp()
        ? 'Microphone blocked — open Settings → Apps → Unino → Permissions and enable Microphone'
        : 'Microphone access denied. Allow microphone in your browser settings');
    } else {
      toast('Could not access microphone');
    }
    return null;
  }
}

async function startVoiceRecord(ctx = '') {
  if (_voiceRecorder && _voiceRecorder.state !== 'inactive') return;
  _voiceContext = ctx;
  const stream = await ensureMicrophoneStream();
  if (!stream) return;
  try {
    _voiceRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
  } catch (_) {
    try {
      _voiceRecorder = new MediaRecorder(stream);
    } catch (err) {
      stream.getTracks().forEach(track => track.stop());
      toast('Voice notes are not supported on this device');
      return;
    }
  }
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

async function doLogout() {
  if (state.user?.uid && _nativePushToken) await removePushTokenForUser(state.user.uid, _nativePushToken).catch(() => {});
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
  // Stop live camera preview if active and not yet streaming
  if (_hostStream && !_hostStreamId) {
    _hostStream.getTracks().forEach(t => t.stop());
    _hostStream = null;
  }
  $('#modal-bg').style.display = 'none';
  $('#modal-inner').innerHTML = '';
}

// ─── Share System ────────────────────────────────
async function openShareModal(postId) {
  openModal(`
    <div class="modal-header"><h2>Share Post</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
    <div class="modal-body" style="padding:16px">
       <button class="btn-primary btn-full" style="margin-bottom:12px;background:var(--accent);color:white;border:none;padding:12px;border-radius:12px;font-weight:600;width:100%" onclick="openQuoteRepost('${postId}')">🔄 Repost</button>
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
      <div class="modal-header"><h2>Repost</h2><button class="icon-btn" onclick="closeModal()">&times;</button></div>
      <div class="modal-body" style="padding:16px">
        <div style="display:flex;gap:10px;margin-bottom:12px">
          ${avatar(state.profile.displayName, state.profile.photoURL, 'avatar-md')}
          <div><div style="font-weight:600">${esc(state.profile.displayName)}</div><div style="font-size:12px;color:var(--text-secondary)">Reposting this post</div></div>
        </div>
        <textarea id="quote-text" placeholder="Add your thoughts…" style="width:100%;min-height:80px;border:none;background:transparent;color:var(--text-primary);font-size:15px;resize:none;outline:none;margin-bottom:12px"></textarea>
        <div class="quote-embed-preview" style="border:1px solid var(--border);border-radius:var(--radius);padding:12px;background:var(--bg-secondary)">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            ${avatar(orig.authorName, orig.authorPhoto, 'avatar-sm')}
            <span style="font-weight:600;font-size:13px">${esc(orig.authorName || 'User')}</span>
          </div>
          ${orig.content ? `<div style="font-size:13px;color:var(--text-secondary);margin-bottom:8px;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden">${esc(orig.content)}</div>` : ''}
          ${hasImg ? `<img src="${orig.imageURL}" style="width:100%;max-height:120px;object-fit:cover;border-radius:8px">` : ''}
          ${hasVid && vidUrl ? `<video class="inline-video-preview ready" src="${vidUrl}" style="width:100%;max-height:120px;object-fit:cover;border-radius:8px;background:#000" autoplay muted loop playsinline preload="metadata"></video>` : ''}
        </div>
        <div style="display:flex;justify-content:flex-end;margin-top:12px">
          <button class="btn-primary" id="quote-submit" style="padding:10px 28px">Repost</button>
        </div>
      </div>
    `);
    $('#quote-submit').onclick = async () => {
      const quoteText = ($('#quote-text')?.value || '').trim();
      closeModal(); toast('Reposting…');
      try {
        const postRef = await db.collection('posts').add({
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
        shadowSyncPost(postRef.id, {
          authorId: state.user.uid,
          authorName: state.profile.displayName,
          content: quoteText,
          visibility: 'public',
          createdAt: new Date().toISOString()
        });
        toast('Reposted!');
        navigate('feed');
  setTimeout(() => markBootReady('Loaded'), 220);
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
            <div class="pref-person-name">${esc(u.displayName)}${u.manualVerified || VERIFIED_UIDS.has(u.id) ? '<span class="verified-badge">\u2714</span>' : ''}</div>
            <div class="pref-person-meta">${esc(u.email || '')} \u00b7 ${esc(u.major || '')} \u00b7 ${u.status || 'offline'}</div>
          </div>
        </div>
      `).join('');
    }
    $('#admin-users-list').innerHTML = renderAdminUsers(users);
    $('#admin-user-search').oninput = (e) => {
      const q = (e.target.value || '').toLowerCase();
      const filtered = q ? users.map(u => ({ user: u, score: userSearchScore(u, q) + (((u.email||'').toLowerCase().includes(q.toLowerCase())) ? 80 : 0) })).filter(entry => entry.score > 0).sort((a, b) => b.score - a.score).map(entry => entry.user) : users;
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
          .map(u => ({ user: u, score: userSearchScore(u, q) + (((u.email||'').toLowerCase().includes(q.toLowerCase())) ? 80 : 0) }))
          .filter(entry => entry.score > 0)
          .sort((a, b) => b.score - a.score)
          .map(entry => entry.user);
        $('#verify-results').innerHTML = users.map(u => `
          <div class="pref-person" style="cursor:pointer">
            ${avatar(u.displayName, u.photoURL, 'avatar-sm')}
            <div class="pref-person-info">
              <div class="pref-person-name">${esc(u.displayName)}${u.manualVerified || VERIFIED_UIDS.has(u.id) ? '<span class="verified-badge">\u2714</span>' : ''}</div>
              <div class="pref-person-meta">${esc(u.email || '')}</div>
            </div>
            <button class="btn-sm ${u.manualVerified ? 'btn-ghost' : 'btn-primary'}" onclick="event.stopPropagation();doVerifyUser('${u.id}', ${!u.manualVerified})">${u.manualVerified ? 'Unverify' : 'Verify'}</button>
          </div>
        `).join('') || '<p style="color:var(--text-tertiary)">No matches.</p>';
      } catch (e) { console.error(e); }
    }, 300);
  };
}

async function doVerifyUser(uid, verify) {
  if (!_isAdmin) return;
  try {
    await db.collection('users').doc(uid).update({ manualVerified: verify, isVerified: verify });
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
    await Promise.all(usersSnap.docs.map(d => addNotification(
      d.id,
      'admin',
      text,
      { admin: true },
      {
        fromOverride: {
          uid: state.user.uid,
          name: 'Unibo Admin',
          photo: state.profile.photoURL || null
        }
      }
    )));
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
  initNativeShell().catch(() => {});

  showBootStatus('Starting Unibo…');
  ensureBootFallback();
  ensureChatPlusMenu($('#chat-plus-menu'), 'dm');
  ensureChatPlusMenu($('#gchat-plus-menu'), 'group');
  $('#chat-msgs')?.addEventListener('pointerdown', closeChatPlusMenus, true);
  $('#gchat-msgs')?.addEventListener('pointerdown', closeChatPlusMenus, true);
  $('#chat-msgs')?.addEventListener('scroll', closeChatPlusMenus, true);
  $('#gchat-msgs')?.addEventListener('scroll', closeChatPlusMenus, true);

  // Image viewer close + gallery navigation + swipe
  $('#img-close')?.addEventListener('click', () => { closeGalleryViewer(); });
  $('#img-download')?.addEventListener('click', () => { downloadCurrentGalleryMedia(); });
  $('#img-prev')?.addEventListener('click', () => { if (_galleryIdx > 0) { _galleryIdx--; _renderGalleryFrame(); } });
  $('#img-next')?.addEventListener('click', () => { if (_galleryIdx < _galleryUrls.length - 1) { _galleryIdx++; _renderGalleryFrame(); } });
  // Touch swipe support for gallery
  let _galTouchX = 0;
  const imgView = $('#img-view');
  if (imgView) {
    imgView.addEventListener('touchstart', e => { _galTouchX = e.touches[0].clientX; }, { passive: true });
    imgView.addEventListener('touchend', e => {
      const dx = e.changedTouches[0].clientX - _galTouchX;
      if (Math.abs(dx) > 60 && _galleryUrls.length > 1) {
        if (dx < 0 && _galleryIdx < _galleryUrls.length - 1) { _galleryIdx++; _renderGalleryFrame(); }
        else if (dx > 0 && _galleryIdx > 0) { _galleryIdx--; _renderGalleryFrame(); }
      }
    });
  }
  document.addEventListener('contextmenu', e => {
    const video = e.target && e.target.closest ? e.target.closest('video') : null;
    if (!video) return;
    const src = video.currentSrc || video.src || video.getAttribute('src') || '';
    if (!src) return;
    e.preventDefault();
    openVideoDownloadPrompt(src);
  }, true);

  // Notifications dropdown toggle
  $('#notif-btn')?.addEventListener('click', (e) => {
    e.stopPropagation();
    requestLocalNotificationPermission().catch(() => {});
    const dd = $('#notif-dropdown');
    if (dd.style.display === 'block') { closeNotifDropdown(); return; }
    loadNotifications();
    dd.style.display = 'block';
    markAllBellNotifsRead().then(() => {
      if (dd.style.display === 'block') loadNotifications();
    }).catch(() => {});
    _notifDropdownCloseHandler = ev => {
      const trigger = $('#notif-btn');
      if (!dd.contains(ev.target) && ev.target !== trigger && !trigger?.contains(ev.target)) {
        closeNotifDropdown();
      }
    };
    setTimeout(() => document.addEventListener('click', _notifDropdownCloseHandler, true), 10);
  });

  // Close notification dropdown only for real user scroll gestures, not auto scroll/restore.
  const contentEl = $('#content');
  if (contentEl) {
    ['touchstart', 'touchmove', 'wheel', 'pointerdown'].forEach(eventName => {
      contentEl.addEventListener(eventName, () => noteContentScrollGesture(), { passive: true, capture: true });
    });
    contentEl.addEventListener('scroll', () => {
      const dd = $('#notif-dropdown');
      if (!dd || dd.style.display !== 'block') return;
      if (isRecentContentScrollGesture()) closeNotifDropdown();
    }, true);
  }

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
    setCommentAnonChoice, editAnonNickname, editContactNickname,
    toggleCommentReplies,
    saveCurrentGpsLocation, clearAppCache,
    openCreateEvent, openEventDetail, openLocationDetail, toggleEventGoing, deleteEvent,
    switchToExploreList,
    startAnonChat, removeEventImage, showUserPreview, openModuleFeed, openTagFeed, openAnonPostActions,
    openHustleQuickActions, saveListingToProfile, shareListingCard,
    startVoiceRecord, cancelVoiceRecord, stopVoiceAndSend, openReelsViewer,
    openVideoHub, closeVideoHub, switchVideoHubTab, openGoLiveModal, startLiveStream,
    joinLiveStream, leaveLiveStream, endLiveStream, sendLiveComment, sendLiveReaction, openHostLiveView, switchLiveCamera,
    toggleCommentLike, openShareModal, repost, openQuoteRepost, shareToFriend, viewPost, markNotifRead,
    clearFeedSearch,
    clearCommentImage, clearReelCommentImage, toggleReelCommentLike,
    setReelCommentReply, clearReelCommentReply,
    closeReelsViewer, toggleReelPlay, toggleReelsSound, reelLike, togglePostExpand, shiftTrendingRail, toggleTrendingRail,
    openPostReactionPicker, reactToPost, openCommentReactionPicker, reactToComment,
    openMessageActionSheet, reactToMessage, deleteMessage,
    toggleVN, seekVN,
    openAdminPanel, adminViewAllGroups, adminViewAllUsers, adminVerifyUser, doVerifyUser,
    adminModeratePosts, adminDeletePost, adminBroadcastPrompt, adminSendBroadcast,
    reportPost, submitPostReport, showAdminDataClear, adminDataClearStepTwo, doAdminDataClear,
    showConvoActions, archiveConvo, deleteConvo, blockUserFromChat, unblockUser, requestReveal,
    unarchiveConvo, loadArchivedDMList, toggleArchiveDmView, loadBlockedUsersList,
    openAnonDmSettings, setAllowAnonymousMessages, toggleStoryViewerSound, clearStoryMediaSelection, closeNotifDropdown,
    clearMapRoute, routeToCampusLocation, routeToMapPoint, openCampusMapView, openLocationPinPreview, openCreateEventAtMapPoint,
    downloadCurrentGalleryImage, downloadMediaUrl, openVideoDownloadPrompt, openStoryInsights,
    openMentionProfileByHandle,
    runAppwriteBackendDiagnostics, sendAppwritePing, runNotificationDiagnostics, sendDebugLocalNotification, sendGatewayNotificationProbe, runShadowSyncProbe,
    toggleAppwriteMirror, setAppwriteMirrorEnabled
  });
});
