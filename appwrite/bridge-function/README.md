# Unino Appwrite Bridge Function

This function enables a safe hybrid migration:
- Firebase remains live for existing users.
- New users flagged as `appwritePrimary` can mirror reactions/events to Appwrite.
- Push token lifecycle can be mirrored to Appwrite.

## Routes

- `GET /`
  - Health check, returns `{ ok: true, status: "ready" }`

- `POST /push-sync`
  - Body:
    - `action`: `upsert` or `delete`
    - `userId`: Firebase UID
    - `token`: FCM token
    - `platform`: `android` / `ios` / `web`
  - Header:
    - `Authorization: Bearer <firebaseIdToken>`

- `POST /event-sync`
  - Body:
    - `eventType`: e.g. `post_reaction`, `comment_reaction`
    - `payload`: object
  - Header:
    - `Authorization: Bearer <firebaseIdToken>`

## Required Appwrite Function Env Vars

- `APPWRITE_ENDPOINT` = `https://syd.cloud.appwrite.io/v1`
- `APPWRITE_PROJECT_ID` = your Appwrite project ID
- `APPWRITE_API_KEY` = server key (keep secret)
- `FIREBASE_WEB_API_KEY` = Firebase web API key for token verification

## Optional Env Vars (for persistence)

- `APPWRITE_DB_ID`
- `APPWRITE_PUSH_TABLE_ID`
- `APPWRITE_EVENTS_TABLE_ID`
- `APPWRITE_USERS_TABLE_ID` (default: `users`)
- `APPWRITE_POSTS_TABLE_ID` (default: `posts`)
- `APPWRITE_MESSAGES_TABLE_ID` (default: `messages`)

If DB/table vars are not set, routes return success with `skipped` responses.

## Shadow Sync Events

`POST /event-sync` now mirrors selected events directly into core tables:

- `user_upsert` -> `users`
- `post_upsert` -> `posts`
- `comment_upsert` -> `messages` (stored with `messageType: "comment"`)

Response includes mirror status in `result.mirror` for diagnostics.

## Appwrite Console Setup

1. Function settings:
   - Runtime: Node
   - Entrypoint: `index.js`
   - Execute permission: allow your app traffic (for testing: Any)
2. Add env vars listed above.
3. Deploy.
4. Verify:
   - `GET /` should return 200.

## Client Config Already Wired

In `js/firebase-config.js`:
- `window.UNINO_APPWRITE_SYNC_URL` points to `/push-sync`
- `window.UNINO_APPWRITE_EVENT_SYNC_URL` points to `/event-sync`

These calls are non-blocking and do not break Firebase behavior.
