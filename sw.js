// Unibo PWA Service Worker — Lightweight, network-first (never traps stale code or Firebase traffic)
const SW_VERSION = 'unibo-pwa-v1';

self.addEventListener('install', event => {
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== SW_VERSION).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Always prefer network so Unibo updates land instantly; never intercept cross-origin APIs
self.addEventListener('fetch', event => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  event.respondWith(
    fetch(req).catch(() => caches.match(req))
  );
});

// Ready for iOS 16.4+ & desktop Web Push notifications
self.addEventListener('push', event => {
  let payload = {};
  try {
    payload = event.data ? event.data.json() : {};
  } catch (_) {
    payload = { title: 'Unibo', body: event.data ? event.data.text() : 'New campus notification' };
  }
  const title = payload.title || 'Unibo';
  const options = {
    body: payload.body || 'You have a new campus update',
    icon: payload.senderPhoto || '/unino-icon.png',
    badge: '/favicon.png',
    data: payload.data || {}
  };
  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener('notificationclick', event => {
  event.notification.close();
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(clientList => {
      for (const client of clientList) {
        if ('focus' in client) return client.focus();
      }
      if (self.clients.openWindow) return self.clients.openWindow('/');
    })
  );
});
