/* ══════════════════════════════════════════════════════
 *  UNINO — Firebase Configuration
 * ══════════════════════════════════════════════════════
 *
 *  FIRESTORE SECURITY RULES (paste in Firestore → Rules tab):
 *
 *  rules_version = '2';.. and can
 *  service cloud.firestore {
 *    match /databases/{database}/documents {
 *      match /users/{userId} {
 *        allow read: if request.auth != null;
 *        allow create: if request.auth != null && request.auth.uid == userId;
 *        allow update: if request.auth != null && request.auth.uid == userId;
 *      }
 *      match /users/{userId}/pushTokens/{tokenId} {
 *        allow read, write: if request.auth != null && request.auth.uid == userId;
 *      }
 *      match /posts/{postId} {
 *        allow read: if request.auth != null;
 *        allow create: if request.auth != null;
 *        allow update, delete: if request.auth != null && request.auth.uid == resource.data.authorId;
 *      }
 *      match /posts/{postId}/comments/{commentId} {
 *        allow read, write: if request.auth != null;
 *      }
 *      match /posts/{postId}/likes/{likeId} {
 *        allow read, write: if request.auth != null;
 *      }
 *      match /conversations/{convoId} {
 *        allow read, write: if request.auth != null;
 *      }
 *      match /conversations/{convoId}/messages/{msgId} {
 *        allow read, write: if request.auth != null;
 *      }
 *      match /listings/{listingId} {
 *        allow read: if request.auth != null;
 *        allow create: if request.auth != null;
 *        allow update, delete: if request.auth != null && request.auth.uid == resource.data.sellerId;
 *      }
 *      match /stats/{doc} {
 *        allow read: if true;
 *        allow write: if request.auth != null;
 *      }
 *      match /stories/{storyId} {
 *        allow read: if request.auth != null;
 *        allow create: if request.auth != null;
 *        allow update: if request.auth != null;
 *      }
 *      match /groups/{groupId} {
 *        allow read: if request.auth != null;
 *        allow create: if request.auth != null;
 *        allow update: if request.auth != null;
 *      }
 *      match /groups/{groupId}/messages/{msgId} {
 *        allow read, write: if request.auth != null;
 *      }
 *      match /assignmentGroups/{groupId} {
 *        allow read: if request.auth != null;
 *        allow create: if request.auth != null;
 *        allow update: if request.auth != null;
 *      }
 *      match /assignmentGroups/{groupId}/messages/{msgId} {
 *        allow read, write: if request.auth != null;
 *      }
 *      match /events/{eventId} {
 *        allow read: if request.auth != null;
 *        allow create: if request.auth != null;
 *        allow update: if request.auth != null;
 *      }
 *    }
 *  }
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
  experimentalAutoDetectLongPolling: true,
  useFetchStreams: false,
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

// Bridge URLs are derived from a single base run URL, for example:
//   window.UNINO_APPWRITE_BRIDGE_BASE_URL = 'https://<functionId>.fra.appwrite.run'
const appwriteBridgeBase = String(
  window.UNINO_APPWRITE_BRIDGE_BASE_URL
  || window.UNINO_APPWRITE_BRIDGE_RUN_URL
  || ''
).trim().replace(/\/+$/, '');

window.UNINO_APPWRITE_SYNC_URL = window.UNINO_APPWRITE_SYNC_URL || (appwriteBridgeBase ? `${appwriteBridgeBase}/push-sync` : '');
window.UNINO_APPWRITE_SYNC_URLS = window.UNINO_APPWRITE_SYNC_URLS || (
  window.UNINO_APPWRITE_SYNC_URL ? [window.UNINO_APPWRITE_SYNC_URL] : []
);

window.UNINO_APPWRITE_EVENT_SYNC_URL = window.UNINO_APPWRITE_EVENT_SYNC_URL || (appwriteBridgeBase ? `${appwriteBridgeBase}/event-sync` : '');
window.UNINO_APPWRITE_EVENT_SYNC_URLS = window.UNINO_APPWRITE_EVENT_SYNC_URLS || (
  window.UNINO_APPWRITE_EVENT_SYNC_URL ? [window.UNINO_APPWRITE_EVENT_SYNC_URL] : []
);
