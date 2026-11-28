# Unino Appwrite Bridge Function

This serverless bridge handles Firebase-authenticated backend operations for Unibo:
- **Profile Pictures on Azure Blob Storage** (`profilesunibo` / `profile-pictures`) with server-side validation, read-only Blob SAS URLs, and automatic orphan cleanup.
- **Push Notifications via Appwrite FCM Messaging** (high-priority data-only FCM pushes for native Android `MessagingStyle` circular avatars & inline sent-image previews).
- **Shadow Sync** for staged Firebase -> Appwrite migration.

## Routes

- `GET /`
  - Health check, returns `{ ok: true, version, status: "ready", azureProfilePicturesReady, routes }`

- `POST /profile-picture`
  - Header: `Authorization: Bearer <firebaseIdToken>`
  - Body: `{ dataUrl, prevBlobName?, prevPhotoUrl? }`
  - Validates JPEG/PNG/WebP magic bytes (max 2 MB), uploads a versioned immutable blob to Azure Blob Storage (`{uid}/{timestamp}_{rand}.{ext}`), deletes the user's previous owned blob to prevent orphans, and returns a read-only HTTPS Blob SAS URL.

- `POST /delete-profile-picture` (or `DELETE /profile-picture`)
  - Header: `Authorization: Bearer <firebaseIdToken>`
  - Body: `{ blobName?, photoUrl? }`
  - Verifies UID ownership and deletes the user's profile picture blob from Azure Blob Storage.

- `POST /push-sync`
  - Header: `Authorization: Bearer <firebaseIdToken>`
  - Body: `{ action: "upsert" | "delete", userId, token, platform }`

- `POST /event-sync`
  - Header: `Authorization: Bearer <firebaseIdToken>`
  - Body: `{ eventType, payload }`

- `GET /avatar/:fileId` & `GET /profile-picture/:uid/:file`
  - Public read proxies for legacy Appwrite avatars and Azure profile blobs.

## Required Environment Variables

### Core Bridge
- `APPWRITE_ENDPOINT` = `https://fra.cloud.appwrite.io/v1`
- `APPWRITE_PROJECT_ID` = your Appwrite project ID
- `APPWRITE_API_KEY` = server key (keep secret)
- `FIREBASE_WEB_API_KEY` = Firebase web API key for token verification

### Azure Blob Storage (Profile Pictures)
- `AZURE_STORAGE_CONNECTION_STRING` = Azure Storage connection string (server-side secret only — never expose in client code or repository)
- `AZURE_STORAGE_ACCOUNT_NAME` = `profilesunibo`
- `AZURE_STORAGE_ACCOUNT_KEY` = Azure Storage account key (server-side secret only)
- `AZURE_STORAGE_CONTAINER` = `profile-pictures`

> All 4 `AZURE_STORAGE_*` variables are stored in GitHub Actions Repository Secrets and synced automatically to Appwrite Function `69dd474b002ce3934b75` via `.github/workflows/sync-appwrite-azure-vars.yml`.

## Optional Env Vars (for persistence)

- `APPWRITE_DB_ID`
- `APPWRITE_PUSH_TABLE_ID`
- `APPWRITE_EVENTS_TABLE_ID`
- `APPWRITE_USERS_TABLE_ID` (default: `users`)
- `APPWRITE_POSTS_TABLE_ID` (default: `posts`)
- `APPWRITE_MESSAGES_TABLE_ID` (default: `messages`)
