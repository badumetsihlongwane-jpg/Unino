import { Client, Databases, ID, Query } from 'node-appwrite';

const JSON_HEADERS = { 'content-type': 'application/json; charset=utf-8' };
const FUNCTION_VERSION = '2026-03-17-shadow-v2';

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
    const result = await upsertRowById(databases, dbId, usersTableId, rowId, {
      uid,
      displayName: String(payload.displayName || ''),
      email: String(payload.email || ''),
      photoURL: String(payload.photoURL || ''),
      major: String(payload.major || ''),
      university: String(payload.university || ''),
      updatedAt: String(payload.updatedAt || now)
    });
    return { mirrored: true, entity: 'user', ...result };
  }

  if (eventType === 'post_upsert') {
    const postId = String(payload.postId || '').trim();
    if (!postId) return { mirrored: false, reason: 'missing-postId' };
    const rowId = `p_${postId}`.slice(0, 36);
    const result = await upsertRowById(databases, dbId, postsTableId, rowId, {
      postId,
      authorId: String(payload.authorId || ''),
      authorName: String(payload.authorName || ''),
      content: String(payload.content || ''),
      mediaURL: String(payload.mediaURL || ''),
      visibility: String(payload.visibility || 'public'),
      createdAt: String(payload.createdAt || now),
      updatedAt: String(payload.updatedAt || now)
    });
    return { mirrored: true, entity: 'post', ...result };
  }

  if (eventType === 'comment_upsert') {
    const commentId = String(payload.commentId || '').trim();
    const postId = String(payload.postId || '').trim();
    if (!commentId || !postId) return { mirrored: false, reason: 'missing-comment-or-post-id' };
    const rowId = `c_${commentId}`.slice(0, 36);
    const result = await upsertRowById(databases, dbId, messagesTableId, rowId, {
      messageType: 'comment',
      commentId,
      postId,
      authorId: String(payload.authorId || ''),
      authorName: String(payload.authorName || ''),
      text: String(payload.text || ''),
      createdAt: String(payload.createdAt || now),
      updatedAt: String(payload.updatedAt || now)
    });
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

      log(`push-sync ${action} uid=${userId}`);
      return json(req, res, 200, { ok: true, route: 'push-sync', result });
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
      log(`event-sync ${eventType} uid=${auth.uid}`);
      return json(req, res, 200, { ok: true, route: 'event-sync', result: { eventLog, mirror } });
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
