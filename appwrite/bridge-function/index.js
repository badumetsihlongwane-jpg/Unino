import crypto from 'node:crypto';
import {
  Client,
  Databases,
  ID,
  MessagePriority,
  Messaging,
  MessagingProviderType,
  Query,
  Users
} from 'node-appwrite';

const JSON_HEADERS = { 'content-type': 'application/json; charset=utf-8' };
const FUNCTION_VERSION = '2026-03-17-shadow-v4';
const _schemaCache = new Map();

function corsHeaders(req) {
  const origin = (req?.headers?.origin || req?.headers?.Origin || '').trim();
  // Allow known app/web origins. Fallback to wildcard for easier staged migration diagnostics.
  const allowOrigin = origin || '*';
  return {
    'access-control-allow-origin': allowOrigin,
    'access-control-allow-methods': 'GET,POST,OPTIONS',
    'access-control-allow-headers': 'Content-Type, Authorization, X-Requested-With',
    'access-control-max-age': '86400',
    vary: 'Origin'
  };
}

function json(req, res, status, payload) {
  return res.send(JSON.stringify(payload), status, {
    ...JSON_HEADERS,
    ...corsHeaders(req)
  });
}

function routePath(req) {
  const raw = req.path || req.url || '/';
  try {
    const parsed = new URL(raw, 'http://localhost');
    return parsed.pathname || '/';
  } catch {
    return raw.split('?')[0] || '/';
  }
}

function extractBearerFromHeaders(headers = {}) {
  const keys = Object.keys(headers || {});
  const direct = headers.authorization || headers.Authorization || headers['x-authorization'] || '';
  if (typeof direct === 'string' && direct.startsWith('Bearer ')) return direct.slice(7);
  for (const key of keys) {
    const value = headers[key];
    if (typeof value === 'string' && value.startsWith('Bearer ')) return value.slice(7);
  }
  return '';
}

async function verifyFirebaseToken(idToken, apiKeyOverride = '') {
  const firebaseApiKey = String(apiKeyOverride || process.env.FIREBASE_WEB_API_KEY || '').trim();
  if (!firebaseApiKey) throw new Error('Missing FIREBASE_WEB_API_KEY (env or request body.firebaseApiKey)');
  const resp = await fetch(`https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=${encodeURIComponent(firebaseApiKey)}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ idToken })
  });
  if (!resp.ok) {
    let reason = '';
    try {
      const payload = await resp.json();
      reason = payload?.error?.message || payload?.error || '';
    } catch {
      reason = (await resp.text().catch(() => '')) || '';
    }
    throw new Error(`Firebase token lookup failed (${resp.status})${reason ? `: ${reason}` : ''}`);
  }
  const data = await resp.json();
  const user = data?.users?.[0];
  if (!user?.localId) throw new Error('Invalid Firebase token');
  return { uid: user.localId, email: user.email || '' };
}

function makeClient() {
  const endpoint = process.env.APPWRITE_ENDPOINT || process.env.APPWRITE_FUNCTION_API_ENDPOINT || '';
  const project = process.env.APPWRITE_PROJECT_ID || process.env.APPWRITE_FUNCTION_PROJECT_ID || '';
  const key = process.env.APPWRITE_API_KEY || process.env.APPWRITE_FUNCTION_API_KEY || '';
  if (!endpoint || !project || !key) throw new Error('Missing Appwrite env (APPWRITE_ENDPOINT / APPWRITE_PROJECT_ID / APPWRITE_API_KEY)');
  const client = new Client().setEndpoint(endpoint).setProject(project).setKey(key);
  return client;
}

function getClientEnv() {
  const endpoint = process.env.APPWRITE_ENDPOINT || process.env.APPWRITE_FUNCTION_API_ENDPOINT || '';
  const project = process.env.APPWRITE_PROJECT_ID || process.env.APPWRITE_FUNCTION_PROJECT_ID || '';
  const key = process.env.APPWRITE_API_KEY || process.env.APPWRITE_FUNCTION_API_KEY || '';
  return { endpoint, project, key };
}

async function fetchSchema(endpoint, project, key, path) {
  const url = `${endpoint.replace(/\/$/, '')}${path}`;
  const resp = await fetch(url, {
    method: 'GET',
    headers: {
      'X-Appwrite-Project': project,
      'X-Appwrite-Key': key,
      'content-type': 'application/json'
    }
  });
  if (!resp.ok) return null;
  return resp.json().catch(() => null);
}

async function getTableSchema(dbId, tableId) {
  const cacheKey = `${dbId}:${tableId}`;
  if (_schemaCache.has(cacheKey)) return _schemaCache.get(cacheKey);

  const { endpoint, project, key } = getClientEnv();
  if (!endpoint || !project || !key) return null;

  // Support both Appwrite Databases API generations.
  const candidates = [
    `/databases/${encodeURIComponent(dbId)}/tables/${encodeURIComponent(tableId)}/columns`,
    `/databases/${encodeURIComponent(dbId)}/collections/${encodeURIComponent(tableId)}/attributes`
  ];

  for (const path of candidates) {
    const data = await fetchSchema(endpoint, project, key, path);
    if (!data) continue;
    const columns = Array.isArray(data?.columns) ? data.columns : [];
    const attributes = Array.isArray(data?.attributes) ? data.attributes : [];
    const fields = columns.length ? columns : attributes;
    if (!fields.length) continue;
    const normalized = fields.map(field => ({
      key: String(field.key || field.$id || '').trim(),
      required: !!field.required,
      type: String(field.type || field.format || 'string').toLowerCase()
    })).filter(field => field.key);
    if (normalized.length) {
      _schemaCache.set(cacheKey, normalized);
      return normalized;
    }
  }

  return null;
}

function defaultForType(type) {
  if (type.includes('int') || type.includes('float') || type.includes('double') || type.includes('number')) return 0;
  if (type.includes('bool')) return false;
  if (type.includes('datetime') || type.includes('date') || type.includes('time')) return new Date().toISOString();
  return '';
}

async function normalizePayloadForTable(dbId, tableId, payload = {}) {
  const schema = await getTableSchema(dbId, tableId);
  if (!schema) return payload;

  const out = {};
  const keys = new Set(schema.map(field => field.key));
  for (const [key, value] of Object.entries(payload || {})) {
    if (!keys.has(key)) continue;
    out[key] = value;
  }

  // Ensure required keys are present when possible.
  for (const field of schema) {
    if (!field.required) continue;
    if (out[field.key] !== undefined && out[field.key] !== null) continue;
    out[field.key] = defaultForType(field.type);
  }

  return out;
}

async function upsertPushTarget({ userId, token, platform }) {
  const dbId = process.env.APPWRITE_DB_ID || '';
  const tableId = process.env.APPWRITE_PUSH_TABLE_ID || '';
  if (!dbId || !tableId) return { skipped: true, reason: 'missing-db-config' };

  const databases = new Databases(makeClient());
  const existing = await dbList(databases, dbId, tableId, [
    Query.equal('userId', userId),
    Query.equal('token', token),
    Query.limit(1)
  ]);
  const existingRows = extractItems(existing);

  if ((existing?.total || 0) > 0 && existingRows.length) {
    const row = existingRows[0];
    await dbUpdate(databases, dbId, tableId, row.$id, {
      platform: platform || 'android',
      updatedAt: new Date().toISOString(),
      active: true
    });
    return { upserted: true, rowId: row.$id, mode: 'update' };
  }

  const created = await dbCreate(databases, dbId, tableId, ID.unique(), {
    userId,
    token,
    platform: platform || 'android',
    active: true,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  });
  return { upserted: true, rowId: created.$id, mode: 'create' };
}

function toAppwriteUserId(firebaseUid = '') {
  // Keep deterministic IDs under Appwrite max length constraints.
  return `f_${String(firebaseUid || '').replace(/[^a-zA-Z0-9_-]/g, '_')}`.slice(0, 36);
}

function buildTargetId(token = '', platform = 'android') {
  const digest = crypto.createHash('sha1').update(String(token || '')).digest('hex').slice(0, 24);
  return `pt_${String(platform || 'android').slice(0, 6)}_${digest}`.slice(0, 36);
}

async function ensureMessagingUser(users, appwriteUserId, email = '') {
  try {
    await users.get(appwriteUserId);
    return { ok: true, mode: 'exists', userId: appwriteUserId };
  } catch (e) {
    if (Number(e?.code || 0) !== 404) throw e;
  }

  const normalizedEmail = String(email || '').trim();
  if (!normalizedEmail || !normalizedEmail.includes('@')) {
    return { ok: false, skipped: true, reason: 'missing-email-for-user-create', userId: appwriteUserId };
  }

  await users.create(appwriteUserId, normalizedEmail);
  return { ok: true, mode: 'created', userId: appwriteUserId };
}

async function ensureUserPushTarget({ firebaseUid, token, platform = 'android', email = '' }) {
  const users = new Users(makeClient());
  const appwriteUserId = toAppwriteUserId(firebaseUid);
  const userEnsure = await ensureMessagingUser(users, appwriteUserId, email);
  if (!userEnsure.ok) return { ok: false, skipped: true, reason: userEnsure.reason, appwriteUserId };

  const targetId = buildTargetId(token, platform);
  try {
    const target = await users.createTarget(
      appwriteUserId,
      targetId,
      MessagingProviderType.Push,
      token,
      undefined,
      `unino-${platform}`
    );
    return { ok: true, mode: 'create', appwriteUserId, targetId: target.$id || targetId };
  } catch (e) {
    if (Number(e?.code || 0) !== 409) throw e;
    const target = await users.updateTarget(appwriteUserId, targetId, token, undefined, `unino-${platform}`);
    return { ok: true, mode: 'update', appwriteUserId, targetId: target.$id || targetId };
  }
}

async function listPushTargetsForFirebaseUid(firebaseUid) {
  const users = new Users(makeClient());
  const appwriteUserId = toAppwriteUserId(firebaseUid);
  try {
    const list = await users.listTargets(appwriteUserId, [Query.limit(100)]);
    const targets = Array.isArray(list?.targets) ? list.targets : [];
    const pushTargets = targets.filter(target => String(target?.providerType || '') === 'push');
    return {
      ok: true,
      appwriteUserId,
      targets: pushTargets.map(target => ({ id: target.$id, identifier: target.identifier || '' }))
    };
  } catch (e) {
    if (Number(e?.code || 0) === 404) return { ok: true, appwriteUserId, targets: [] };
    throw e;
  }
}

async function dispatchPushViaMessaging({ targetFirebaseUid, title, body, data = {} }) {
  const messaging = new Messaging(makeClient());
  const targetList = await listPushTargetsForFirebaseUid(targetFirebaseUid);
  const targetIds = (targetList.targets || []).map(target => target.id).filter(Boolean);

  if (!targetIds.length) {
    return {
      sent: false,
      reason: 'no-targets',
      appwriteUserId: targetList.appwriteUserId,
      targetCount: 0
    };
  }

  const message = await messaging.createPush(
    ID.unique(),
    String(title || 'Unino notification'),
    String(body || ''),
    [],
    [],
    targetIds,
    data,
    '',
    '',
    '',
    '',
    '',
    1,
    false,
    undefined,
    true,
    true,
    MessagePriority.High
  );

  return {
    sent: true,
    messageId: message?.$id || '',
    providerType: message?.providerType || 'push',
    appwriteUserId: targetList.appwriteUserId,
    targetCount: targetIds.length
  };
}

async function deletePushTarget({ userId, token }) {
  const dbId = process.env.APPWRITE_DB_ID || '';
  const tableId = process.env.APPWRITE_PUSH_TABLE_ID || '';
  if (!dbId || !tableId) return { skipped: true, reason: 'missing-db-config' };

  const databases = new Databases(makeClient());
  const existing = await dbList(databases, dbId, tableId, [
    Query.equal('userId', userId),
    Query.equal('token', token),
    Query.limit(10)
  ]);
  const existingRows = extractItems(existing);

  await Promise.all(existingRows.map(row => dbDelete(databases, dbId, tableId, row.$id)));
  return { deleted: true, count: existingRows.length };
}

async function logEvent({ uid, eventType, payload }) {
  const dbId = process.env.APPWRITE_DB_ID || '';
  const tableId = process.env.APPWRITE_EVENTS_TABLE_ID || '';
  if (!dbId || !tableId) return { skipped: true, reason: 'missing-events-config' };

  const databases = new Databases(makeClient());
  const row = await dbCreate(databases, dbId, tableId, ID.unique(), {
    uid,
    eventType,
    payload: JSON.stringify(payload || {}),
    createdAt: new Date().toISOString()
  });
  return { logged: true, rowId: row.$id };
}

function extractItems(listResult) {
  if (!listResult || typeof listResult !== 'object') return [];
  if (Array.isArray(listResult.rows)) return listResult.rows;
  if (Array.isArray(listResult.documents)) return listResult.documents;
  return [];
}

async function dbList(databases, dbId, tableOrCollectionId, queries = []) {
  if (typeof databases.listRows === 'function') return databases.listRows(dbId, tableOrCollectionId, queries);
  if (typeof databases.listDocuments === 'function') return databases.listDocuments(dbId, tableOrCollectionId, queries);
  throw new Error('No compatible list method found on Databases client');
}

async function dbCreate(databases, dbId, tableOrCollectionId, docId, payload) {
  if (typeof databases.createRow === 'function') return databases.createRow(dbId, tableOrCollectionId, docId, payload);
  if (typeof databases.createDocument === 'function') return databases.createDocument(dbId, tableOrCollectionId, docId, payload);
  throw new Error('No compatible create method found on Databases client');
}

async function dbUpdate(databases, dbId, tableOrCollectionId, docId, payload) {
  if (typeof databases.updateRow === 'function') return databases.updateRow(dbId, tableOrCollectionId, docId, payload);
  if (typeof databases.updateDocument === 'function') return databases.updateDocument(dbId, tableOrCollectionId, docId, payload);
  throw new Error('No compatible update method found on Databases client');
}

async function dbDelete(databases, dbId, tableOrCollectionId, docId) {
  if (typeof databases.deleteRow === 'function') return databases.deleteRow(dbId, tableOrCollectionId, docId);
  if (typeof databases.deleteDocument === 'function') return databases.deleteDocument(dbId, tableOrCollectionId, docId);
  throw new Error('No compatible delete method found on Databases client');
}

function getShadowConfig() {
  return {
    dbId: process.env.APPWRITE_DB_ID || 'unibo_db',
    usersTableId: process.env.APPWRITE_USERS_TABLE_ID || 'users',
    postsTableId: process.env.APPWRITE_POSTS_TABLE_ID || 'posts',
    messagesTableId: process.env.APPWRITE_MESSAGES_TABLE_ID || 'messages'
  };
}

async function upsertRowById(databases, dbId, tableId, rowId, payload) {
  try {
    await dbUpdate(databases, dbId, tableId, rowId, payload);
    return { mode: 'update', rowId };
  } catch (e) {
    if (Number(e?.code || 0) !== 404) throw e;
    const created = await dbCreate(databases, dbId, tableId, rowId, payload);
    return { mode: 'create', rowId: created.$id || rowId };
  }
}

async function mirrorEventToCoreTables({ eventType, payload = {} }) {
  const { dbId, usersTableId, postsTableId, messagesTableId } = getShadowConfig();
  if (!dbId) return { mirrored: false, reason: 'missing-db-id' };

  const databases = new Databases(makeClient());
  const now = new Date().toISOString();

  if (eventType === 'user_upsert') {
    const uid = String(payload.uid || '').trim();
    if (!uid) return { mirrored: false, reason: 'missing-uid' };
    const rowId = `u_${uid}`.slice(0, 36);
    const userPayload = await normalizePayloadForTable(dbId, usersTableId, {
      uid,
      displayName: String(payload.displayName || ''),
      email: String(payload.email || ''),
      photoURL: String(payload.photoURL || ''),
      major: String(payload.major || ''),
      university: String(payload.university || ''),
      updatedAt: String(payload.updatedAt || now)
    });
    const result = await upsertRowById(databases, dbId, usersTableId, rowId, userPayload);
    return { mirrored: true, entity: 'user', ...result };
  }

  if (eventType === 'post_upsert') {
    const postId = String(payload.postId || '').trim();
    if (!postId) return { mirrored: false, reason: 'missing-postId' };
    const rowId = `p_${postId}`.slice(0, 36);
    const postPayload = await normalizePayloadForTable(dbId, postsTableId, {
      postId,
      authorId: String(payload.authorId || ''),
      authorName: String(payload.authorName || ''),
      content: String(payload.content || ''),
      mediaURL: String(payload.mediaURL || ''),
      visibility: String(payload.visibility || 'public'),
      createdAt: String(payload.createdAt || now),
      updatedAt: String(payload.updatedAt || now)
    });
    const result = await upsertRowById(databases, dbId, postsTableId, rowId, postPayload);
    return { mirrored: true, entity: 'post', ...result };
  }

  if (eventType === 'comment_upsert') {
    const commentId = String(payload.commentId || '').trim();
    const postId = String(payload.postId || '').trim();
    if (!commentId || !postId) return { mirrored: false, reason: 'missing-comment-or-post-id' };
    const rowId = `c_${commentId}`.slice(0, 36);
    const commentPayload = await normalizePayloadForTable(dbId, messagesTableId, {
      messageType: 'comment',
      commentId,
      postId,
      authorId: String(payload.authorId || ''),
      authorName: String(payload.authorName || ''),
      text: String(payload.text || ''),
      createdAt: String(payload.createdAt || now),
      updatedAt: String(payload.updatedAt || now)
    });
    const result = await upsertRowById(databases, dbId, messagesTableId, rowId, commentPayload);
    return { mirrored: true, entity: 'comment', ...result };
  }

  return { mirrored: false, reason: 'event-not-mapped' };
}

export default async ({ req, res, log, error }) => {
  const method = (req.method || 'GET').toUpperCase();
  const path = routePath(req);

  if (method === 'OPTIONS') {
    return json(req, res, 204, { ok: true });
  }

  if (method === 'GET') {
    return json(req, res, 200, {
      ok: true,
      service: 'unino-appwrite-bridge',
      version: FUNCTION_VERSION,
      status: 'ready',
      path,
      routes: ['/push-sync', '/event-sync']
    });
  }

  if (method !== 'POST') {
    return json(req, res, 405, { ok: false, error: 'method-not-allowed' });
  }

  let body = {};
  try {
    body = req.bodyRaw ? JSON.parse(req.bodyRaw) : (req.body || {});
  } catch {
    return json(req, res, 400, { ok: false, error: 'invalid-json' });
  }

  const bearer = extractBearerFromHeaders(req.headers || {})
    || String(body.idToken || body.firebaseIdToken || body.token || '');
  if (!bearer) {
    return json(req, res, 401, {
      ok: false,
      error: 'missing-auth',
      hint: 'Provide Authorization: Bearer <firebaseIdToken> or body.idToken for manual execution tests'
    });
  }

  try {
    const auth = await verifyFirebaseToken(bearer, body.firebaseApiKey || '');

    if (path === '/push-sync') {
      const { action = '', userId = '', token = '', platform = 'android' } = body;
      if (!['upsert', 'delete'].includes(action) || !userId || !token) {
        return json(req, res, 400, { ok: false, error: 'invalid-payload' });
      }
      if (auth.uid !== userId) return json(req, res, 403, { ok: false, error: 'uid-mismatch' });

      const result = action === 'upsert'
        ? await upsertPushTarget({ userId, token, platform })
        : await deletePushTarget({ userId, token });

      let messagingTarget = { skipped: true, reason: 'delete-or-not-attempted' };
      if (action === 'upsert') {
        try {
          messagingTarget = await ensureUserPushTarget({
            firebaseUid: userId,
            token,
            platform,
            email: auth.email || ''
          });
        } catch (e) {
          messagingTarget = { ok: false, reason: 'messaging-target-failed', detail: String(e?.message || e) };
        }
      }

      log(`push-sync ${action} uid=${userId}`);
      return json(req, res, 200, { ok: true, route: 'push-sync', result: { pushTable: result, messagingTarget } });
    }

    if (path === '/event-sync') {
      const { eventType = '', payload = {} } = body;
      if (!eventType) return json(req, res, 400, { ok: false, error: 'missing-event-type' });
      const eventLog = await logEvent({ uid: auth.uid, eventType, payload });
      let mirror = { mirrored: false, reason: 'not-attempted' };
      try {
        mirror = await mirrorEventToCoreTables({ eventType, payload });
      } catch (e) {
        mirror = { mirrored: false, reason: 'mirror-failed', detail: String(e?.message || e) };
      }
      let push = { sent: false, reason: 'not-attempted' };
      if (eventType === 'notification_dispatch') {
        try {
          push = await dispatchPushViaMessaging({
            targetFirebaseUid: String(payload.targetId || '').trim(),
            title: String(payload.type || 'notification').replace(/_/g, ' ').slice(0, 80),
            body: String(payload.text || '').slice(0, 300),
            data: {
              type: String(payload.type || 'generic'),
              at: String(payload.at || new Date().toISOString()),
              payload: payload.payload && typeof payload.payload === 'object' ? payload.payload : {}
            }
          });
        } catch (e) {
          push = { sent: false, reason: 'push-dispatch-failed', detail: String(e?.message || e) };
        }
      }
      log(`event-sync ${eventType} uid=${auth.uid}`);
      return json(req, res, 200, { ok: true, route: 'event-sync', result: { eventLog, mirror, push } });
    }

    return json(req, res, 404, { ok: false, error: 'route-not-found', path });
  } catch (e) {
    const msg = String(e?.message || e);
    if (msg.includes('Firebase token lookup failed') || msg.includes('Invalid Firebase token')) {
      return json(req, res, 401, {
        ok: false,
        error: 'invalid-auth',
        detail: msg,
        hint: 'Use Firebase ID token from firebase.auth().currentUser.getIdToken(true) and a valid FIREBASE_WEB_API_KEY (env or body.firebaseApiKey)'
      });
    }
    error(`bridge-error: ${e?.message || e}`);
    return json(req, res, 500, { ok: false, error: 'internal', detail: String(e?.message || e) });
  }
};
