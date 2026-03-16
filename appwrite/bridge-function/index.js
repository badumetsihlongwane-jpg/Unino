import { Client, Databases, ID, Query } from 'node-appwrite';

const JSON_HEADERS = { 'content-type': 'application/json; charset=utf-8' };

function json(res, status, payload) {
  return res.send(JSON.stringify(payload), status, JSON_HEADERS);
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

async function verifyFirebaseToken(idToken) {
  const firebaseApiKey = process.env.FIREBASE_WEB_API_KEY || '';
  if (!firebaseApiKey) throw new Error('Missing FIREBASE_WEB_API_KEY env');
  const resp = await fetch(`https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=${encodeURIComponent(firebaseApiKey)}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ idToken })
  });
  if (!resp.ok) throw new Error(`Firebase token lookup failed (${resp.status})`);
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
  const existing = await databases.listRows(dbId, tableId, [
    Query.equal('userId', userId),
    Query.equal('token', token),
    Query.limit(1)
  ]);

  if (existing.total > 0) {
    const row = existing.rows[0];
    await databases.updateRow(dbId, tableId, row.$id, {
      platform: platform || 'android',
      updatedAt: new Date().toISOString(),
      active: true
    });
    return { upserted: true, rowId: row.$id, mode: 'update' };
  }

  const created = await databases.createRow(dbId, tableId, ID.unique(), {
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
  const existing = await databases.listRows(dbId, tableId, [
    Query.equal('userId', userId),
    Query.equal('token', token),
    Query.limit(10)
  ]);

  await Promise.all((existing.rows || []).map(row => databases.deleteRow(dbId, tableId, row.$id)));
  return { deleted: true, count: existing.total || 0 };
}

async function logEvent({ uid, eventType, payload }) {
  const dbId = process.env.APPWRITE_DB_ID || '';
  const tableId = process.env.APPWRITE_EVENTS_TABLE_ID || '';
  if (!dbId || !tableId) return { skipped: true, reason: 'missing-events-config' };

  const databases = new Databases(makeClient());
  const row = await databases.createRow(dbId, tableId, ID.unique(), {
    uid,
    eventType,
    payload: JSON.stringify(payload || {}),
    createdAt: new Date().toISOString()
  });
  return { logged: true, rowId: row.$id };
}

export default async ({ req, res, log, error }) => {
  const method = (req.method || 'GET').toUpperCase();
  const path = routePath(req);

  if (method === 'OPTIONS') {
    return json(res, 204, { ok: true });
  }

  if (method === 'GET' && path === '/') {
    return json(res, 200, { ok: true, service: 'unino-appwrite-bridge', status: 'ready' });
  }

  if (method !== 'POST') {
    return json(res, 405, { ok: false, error: 'method-not-allowed' });
  }

  let body = {};
  try {
    body = req.bodyRaw ? JSON.parse(req.bodyRaw) : (req.body || {});
  } catch {
    return json(res, 400, { ok: false, error: 'invalid-json' });
  }

  const bearer = extractBearerFromHeaders(req.headers || {}) || String(body.idToken || '');
  if (!bearer) {
    return json(res, 401, {
      ok: false,
      error: 'missing-auth',
      hint: 'Provide Authorization: Bearer <firebaseIdToken> or body.idToken for manual execution tests'
    });
  }

  try {
    const auth = await verifyFirebaseToken(bearer);

    if (path === '/push-sync') {
      const { action = '', userId = '', token = '', platform = 'android' } = body;
      if (!['upsert', 'delete'].includes(action) || !userId || !token) {
        return json(res, 400, { ok: false, error: 'invalid-payload' });
      }
      if (auth.uid !== userId) return json(res, 403, { ok: false, error: 'uid-mismatch' });

      const result = action === 'upsert'
        ? await upsertPushTarget({ userId, token, platform })
        : await deletePushTarget({ userId, token });

      log(`push-sync ${action} uid=${userId}`);
      return json(res, 200, { ok: true, route: 'push-sync', result });
    }

    if (path === '/event-sync') {
      const { eventType = '', payload = {} } = body;
      if (!eventType) return json(res, 400, { ok: false, error: 'missing-event-type' });
      const result = await logEvent({ uid: auth.uid, eventType, payload });
      log(`event-sync ${eventType} uid=${auth.uid}`);
      return json(res, 200, { ok: true, route: 'event-sync', result });
    }

    return json(res, 404, { ok: false, error: 'route-not-found', path });
  } catch (e) {
    error(`bridge-error: ${e?.message || e}`);
    return json(res, 500, { ok: false, error: 'internal', detail: String(e?.message || e) });
  }
};
