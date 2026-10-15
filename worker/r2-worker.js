/**
 * Cloudflare R2 Worker — with CORS support
 * 
 * Deploy this to your "app-media" worker to fix browser uploads.
 * 
 * HOW TO UPDATE:
 * 1. Go to https://dash.cloudflare.com → Workers & Pages → app-media
 * 2. Click "Edit Code" (or "Quick Edit")
 * 3. Replace ALL the code with this file's contents
 * 4. Click "Save and Deploy"
 * 
 * IMPORTANT: Check your R2 bucket binding name in Settings → Variables → R2 Bucket Bindings.
 * Update the BINDING_NAME constant below to match YOUR binding name.
 * Common names: MY_BUCKET, BUCKET, R2_BUCKET, APP_MEDIA, media
 */

// ⚠️ CHANGE THIS to match your R2 binding name in Cloudflare dashboard
const BINDING_NAME = 'MY_BUCKET';
const DEFAULT_FIREBASE_API_KEY = 'AIzaSyClbenDcEzwjAiEO8x8-5atWsrFKrQpdgc';
const ADMIN_EMAIL = 'admin@mynwu.ac.za';

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, HEAD, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Range, Authorization',
  'Access-Control-Expose-Headers': 'Content-Length, Content-Range, Content-Type',
  'Access-Control-Max-Age': '86400',
};

function getBucket(env) {
  // Try the configured name first, then common alternatives
  return env[BINDING_NAME] || env.BUCKET || env.MY_BUCKET || env.R2_BUCKET || env.APP_MEDIA || env.media || env.r2;
}

async function verifyFirebaseToken(idToken, apiKey) {
  if (!idToken) throw new Error('Missing token');
  const key = apiKey || DEFAULT_FIREBASE_API_KEY;
  const resp = await fetch(`https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=${encodeURIComponent(key)}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ idToken })
  });
  if (!resp.ok) {
    let err = 'Invalid Firebase token';
    try {
      const data = await resp.json();
      err = data?.error?.message || err;
    } catch (_) {}
    throw new Error(err);
  }
  const data = await resp.json();
  const user = data?.users?.[0];
  if (!user?.localId) throw new Error('User not found in token');
  return { uid: user.localId, email: user.email || '' };
}

function extractBearer(header) {
  if (typeof header !== 'string') return '';
  const trimmed = header.trim();
  return trimmed.startsWith('Bearer ') ? trimmed.slice(7).trim() : '';
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const key = decodeURIComponent(url.pathname.slice(1)); // Remove leading /

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const bucket = getBucket(env);
    if (!bucket) {
      const bindings = Object.keys(env).filter(k => typeof env[k] === 'object').join(', ');
      return new Response(
        `R2 bucket not found. Available bindings: [${bindings}]. Update BINDING_NAME in worker code.`,
        { status: 500, headers: { ...CORS_HEADERS, 'Content-Type': 'text/plain' } }
      );
    }

    // Require Firebase authentication for mutating operations (PUT & DELETE)
    if (request.method === 'PUT' || request.method === 'DELETE') {
      const authHeader = request.headers.get('Authorization') || '';
      const token = extractBearer(authHeader);

      if (!token) {
        return new Response('Unauthorized: Missing or invalid Authorization Bearer header', {
          status: 401,
          headers: { ...CORS_HEADERS, 'Content-Type': 'text/plain' }
        });
      }

      let authUser = null;
      try {
        authUser = await verifyFirebaseToken(token, env.FIREBASE_WEB_API_KEY);
      } catch (e) {
        return new Response(`Unauthorized: ${e.message || 'Token verification failed'}`, {
          status: 401,
          headers: { ...CORS_HEADERS, 'Content-Type': 'text/plain' }
        });
      }

      // Enforce path ownership: Key format is typically Images/{userId}/... or {userId}/...
      const segments = key.split('/').filter(Boolean);
      const pathOwnerUid = (segments[0] === 'Images' ? segments[1] : segments[0]) || '';
      const isAdmin = (authUser.email || '').toLowerCase() === ADMIN_EMAIL.toLowerCase();

      // If the path includes a user ID, only that user or an admin can upload/delete
      if (pathOwnerUid && pathOwnerUid !== authUser.uid && !isAdmin) {
        return new Response('Forbidden: Cannot modify files belonging to another user', {
          status: 403,
          headers: { ...CORS_HEADERS, 'Content-Type': 'text/plain' }
        });
      }

      // Upload file
      if (request.method === 'PUT') {
        try {
          const body = await request.arrayBuffer();
          await bucket.put(key, body, {
            httpMetadata: {
              contentType: request.headers.get('Content-Type') || 'application/octet-stream',
            },
          });
          return new Response('OK', {
            status: 200,
            headers: { ...CORS_HEADERS, 'Content-Type': 'text/plain' },
          });
        } catch (e) {
          return new Response('Upload failed: ' + e.message, {
            status: 500,
            headers: CORS_HEADERS,
          });
        }
      }

      // Delete file
      if (request.method === 'DELETE') {
        try {
          await bucket.delete(key);
          return new Response('Deleted', {
            status: 200,
            headers: CORS_HEADERS,
          });
        } catch (e) {
          return new Response('Delete failed: ' + e.message, {
            status: 500,
            headers: CORS_HEADERS,
          });
        }
      }
    }

    // Download / stream file (GET and HEAD)
    if (request.method === 'GET' || request.method === 'HEAD') {
      try {
        const object = await bucket.get(key);
        if (!object) {
          return new Response('Not found', { status: 404, headers: CORS_HEADERS });
        }
        const headers = new Headers(CORS_HEADERS);
        headers.set('Content-Type', object.httpMetadata?.contentType || 'application/octet-stream');
        headers.set('Content-Length', object.size);
        headers.set('Cache-Control', 'public, max-age=86400');
        headers.set('ETag', object.httpEtag);
        headers.set('Accept-Ranges', 'bytes');

        // For HEAD, return headers only (browsers need this for video/audio)
        if (request.method === 'HEAD') {
          return new Response(null, { status: 200, headers });
        }

        // Support Range requests for video/audio seeking
        const range = request.headers.get('Range');
        if (range) {
          const [, start, end] = range.match(/bytes=(\d+)-(\d*)/) || [];
          if (start !== undefined) {
            const s = parseInt(start);
            const e = end ? parseInt(end) : object.size - 1;
            const slice = await bucket.get(key, { range: { offset: s, length: e - s + 1 } });
            headers.set('Content-Range', `bytes ${s}-${e}/${object.size}`);
            headers.set('Content-Length', e - s + 1);
            return new Response(slice.body, { status: 206, headers });
          }
        }

        return new Response(object.body, { headers });
      } catch (e) {
        return new Response('Error: ' + e.message, { status: 500, headers: CORS_HEADERS });
      }
    }

    return new Response('Method not allowed', {
      status: 405,
      headers: CORS_HEADERS,
    });
  },
};




