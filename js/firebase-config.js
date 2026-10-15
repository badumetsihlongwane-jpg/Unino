/* ══════════════════════════════════════════════════════
 *  UNINO — Firebase Configuration
 * ══════════════════════════════════════════════════════
 *
 *  Firestore security rules are defined in `firestore.rules`
 *  and deployed via `firebase deploy --only firestore:rules`.
 */

const firebaseConfig = {
  apiKey: "AIzaSyClbenDcEzwjAiEO8x8-5atWsrFKrQpdgc",
  authDomain: "unino-b215f.firebaseapp.com",
  projectId: "unino-b215f",
  storageBucket: "unino-b215f.firebasestorage.app",
  messagingSenderId: "174182514516",
  appId: "1:174182514516:web:8e91c12857fc01707bd2c1",
  measurementId: "G-SHQ1FB28Y5"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const auth = firebase.auth();
const db = firebase.firestore();
db.settings({
  merge: true
});
// Storage disabled (requires paid plan) — using base64 data URLs instead
const storage = null;

// Public Firebase Web API key, used by Appwrite bridge to verify Firebase ID tokens.
window.UNINO_FIREBASE_WEB_API_KEY = window.UNINO_FIREBASE_WEB_API_KEY || firebaseConfig.apiKey;

// IndexedDB multi-tab persistence in this compat setup is noisy and unstable in-browser.
// Leave Firestore online-only here until the cache API migration is done.

// Public Appwrite metadata (safe in client). Do NOT put Appwrite API keys here.
const DEFAULT_APPWRITE_ENDPOINT = 'https://fra.cloud.appwrite.io/v1';
const DEFAULT_APPWRITE_PROJECT_ID = '69dd43b6002414cdfb70';
const DEFAULT_APPWRITE_PROJECT_NAME = 'unibo';

window.UNINO_APPWRITE_ENDPOINT = window.UNINO_APPWRITE_ENDPOINT || DEFAULT_APPWRITE_ENDPOINT;
window.UNINO_APPWRITE_PROJECT_ID = window.UNINO_APPWRITE_PROJECT_ID || DEFAULT_APPWRITE_PROJECT_ID;
window.UNINO_APPWRITE_PROJECT_NAME = window.UNINO_APPWRITE_PROJECT_NAME || DEFAULT_APPWRITE_PROJECT_NAME;
const DEFAULT_APPWRITE_PUSH_PROVIDER_ID = '69dd5e4600355126b5b0';
window.UNINO_APPWRITE_PUSH_PROVIDER_ID = window.UNINO_APPWRITE_PUSH_PROVIDER_ID || DEFAULT_APPWRITE_PUSH_PROVIDER_ID;

// Bridge URLs are derived from a single base run URL, for example:
//   window.UNINO_APPWRITE_BRIDGE_BASE_URL = 'https://<functionId>.fra.appwrite.run'
const DEFAULT_APPWRITE_BRIDGE_BASE_URL = 'https://69dd474c000ed9db9d03.fra.appwrite.run';
const appwriteBridgeBase = String(
  window.UNINO_APPWRITE_BRIDGE_BASE_URL
  || window.UNINO_APPWRITE_BRIDGE_RUN_URL
  || DEFAULT_APPWRITE_BRIDGE_BASE_URL
).trim().replace(/\/+$/, '');

window.UNINO_APPWRITE_SYNC_URL = window.UNINO_APPWRITE_SYNC_URL || (appwriteBridgeBase ? `${appwriteBridgeBase}/push-sync` : '');
window.UNINO_APPWRITE_SYNC_URLS = window.UNINO_APPWRITE_SYNC_URLS || (
  window.UNINO_APPWRITE_SYNC_URL ? [window.UNINO_APPWRITE_SYNC_URL] : []
);

window.UNINO_APPWRITE_EVENT_SYNC_URL = window.UNINO_APPWRITE_EVENT_SYNC_URL || (appwriteBridgeBase ? `${appwriteBridgeBase}/event-sync` : '');
window.UNINO_APPWRITE_EVENT_SYNC_URLS = window.UNINO_APPWRITE_EVENT_SYNC_URLS || (
  window.UNINO_APPWRITE_EVENT_SYNC_URL ? [window.UNINO_APPWRITE_EVENT_SYNC_URL] : []
);
