const admin = require('firebase-admin');
const { logger } = require('firebase-functions');
const { onDocumentCreated } = require('firebase-functions/v2/firestore');
const { onRequest } = require('firebase-functions/v2/https');

admin.initializeApp();

const db = admin.firestore();
const messaging = admin.messaging();

function clampText(value = '', max = 120) {
  const text = String(value || '').replace(/\s+/g, ' ').trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function messagePreview(message = {}) {
  if (message.audioURL) return '🎤 Voice message';
  if (message.imageURL) return message.text ? `📷 ${clampText(message.text, 80)}` : '📷 Photo';
  if (message.type === 'share_post') return '↗ Shared a post';
  return clampText(message.text || 'New message', 120);
}

function cleanDataMap(data = {}) {
  return Object.fromEntries(Object.entries(data).filter(([, value]) => value !== undefined && value !== null).map(([key, value]) => [key, String(value)]));
}

function tokenDocId(token = '') {
  return Buffer.from(token).toString('base64url').slice(0, 120) || 'token';
}


function normalizeKey(value = '') {
  return String(value || '').trim().toLowerCase();
}

function chunk(items = [], size = 400) {
  const rows = [];
  for (let i = 0; i < items.length; i += size) rows.push(items.slice(i, i + size));
  return rows;
}

async function getLiveNotificationRecipients(stream = {}) {
  const hostUid = stream.hostUid || '';
  if (!hostUid) return [];
  const visibility = normalizeKey(stream.visibility || 'public');
  const campusKey = normalizeKey(stream.campus || stream.university || '');
  const userSnap = await db.collection('users').get();
  return userSnap.docs
    .map(doc => ({ id: doc.id, ...doc.data() }))
    .filter(user => user.id && user.id !== hostUid)
    .filter(user => {
      if (visibility !== 'campus_only') return true;
      const userCampus = normalizeKey(user.university || user.campus || user.address || '');
      if (!campusKey) return true;
      return !!userCampus && (userCampus.includes(campusKey) || campusKey.includes(userCampus));
    })
    .map(user => user.id);
}

async function getUserTokens(userId) {
  const snap = await db.collection('users').doc(userId).collection('pushTokens').get();
  return snap.docs
    .map(doc => ({ id: doc.id, ...doc.data() }))
    .filter(entry => typeof entry.token === 'string' && entry.token.length > 20);
}

async function pruneInvalidTokens(userId, tokens = [], responses = []) {
  const invalidCodes = new Set([
    'messaging/invalid-registration-token',
    'messaging/registration-token-not-registered'
  ]);
  const deletes = responses.flatMap((response, index) => {
    if (response.success) return [];
    const code = response.error?.code || '';
    if (!invalidCodes.has(code)) return [];
    const token = tokens[index];
    if (!token?.token) return [];
    const docId = token.id || tokenDocId(token.token);
    return [db.collection('users').doc(userId).collection('pushTokens').doc(docId).delete().catch(() => {})];
  });
  await Promise.all(deletes);
}

async function sendPushToUser(userId, payload) {
  const tokenRows = await getUserTokens(userId);
  if (!tokenRows.length) return;

  const channelId = payload.channelId || 'unibo-general';
  const title = payload.title || 'Unibo';
  const body = payload.body || 'You have a new notification';
  const rawImageUrl = typeof payload.imageUrl === 'string' ? payload.imageUrl.trim() : '';
  const imageUrl = /^https?:\/\//i.test(rawImageUrl) ? rawImageUrl : undefined;
  const androidIcon = String(payload.androidIcon || 'ic_notification_small').trim() || 'ic_notification_small';
  const androidColor = String(payload.androidColor || '#6D28D9').trim() || '#6D28D9';
  const clickAction = String(payload.clickAction || 'OPEN_UNIBO').trim() || 'OPEN_UNIBO';
  const mergedData = cleanDataMap({
    title,
    body,
    channelId,
    imageUrl: imageUrl || '',
    icon: androidIcon,
    color: androidColor,
    clickAction,
    ...(payload.data || {})
  });

  const androidRows = tokenRows.filter(row => String(row.platform || 'android').toLowerCase() === 'android');
  const otherRows = tokenRows.filter(row => String(row.platform || '').toLowerCase() !== 'android');

  if (androidRows.length) {
    try {
      // Hybrid payload: notification for closed-app visibility + data for deep-link routing.
      const androidResult = await messaging.sendEachForMulticast({
        tokens: androidRows.map(row => row.token),
        notification: {
          title,
          body
        },
        data: mergedData,
        android: {
          priority: 'high',
          ttl: 60 * 60 * 1000,
          notification: {
            title,
            body,
            channelId,
            sound: 'default',
            imageUrl,
            clickAction,
            icon: androidIcon,
            color: androidColor
          }
        }
      });
      await pruneInvalidTokens(userId, androidRows, androidResult.responses || []);
    } catch (error) {
      logger.error('FCM android send failed', { userId, error });
    }
  }

  if (otherRows.length) {
    try {
      const result = await messaging.sendEachForMulticast({
        tokens: otherRows.map(row => row.token),
        notification: {
          title,
          body
        },
        data: mergedData,
        android: {
          priority: 'high',
          notification: {
            channelId,
            sound: 'default',
            imageUrl,
            clickAction,
            icon: androidIcon,
            color: androidColor
          }
        },
        apns: {
          payload: {
            aps: {
              sound: 'default'
            }
          }
        }
      });
      await pruneInvalidTokens(userId, otherRows, result.responses || []);
    } catch (error) {
      logger.error('FCM multi-platform send failed', { userId, error });
    }
  }
}

exports.onDirectMessageCreated = onDocumentCreated('conversations/{convoId}/messages/{msgId}', async event => {
  const message = event.data?.data();
  const convoId = event.params.convoId;
  if (!message || !convoId) return;

  const convoSnap = await db.collection('conversations').doc(convoId).get();
  if (!convoSnap.exists) return;
  const convo = convoSnap.data() || {};
  const senderId = message.senderId;
  const recipients = (convo.participants || []).filter(uid => uid && uid !== senderId);
  if (!recipients.length) return;

  const senderIndex = (convo.participants || []).indexOf(senderId);
  const senderName = message.senderAnon
    ? 'Anonymous'
    : ((convo.participantNames || [])[senderIndex] || 'New message');
  const senderPhoto = message.senderAnon ? '' : ((convo.participantPhotos || [])[senderIndex] || '');

  await Promise.all(recipients.map(uid => sendPushToUser(uid, {
    title: senderName,
    body: messagePreview(message),
    channelId: 'unibo-messages',
    imageUrl: senderPhoto || undefined,
    data: {
      kind: 'dm',
      convoId,
      profileId: message.senderAnon ? 'anonymous' : senderId,
      senderName
    }
  })));
});

function createGroupMessageTrigger(collectionName) {
  return onDocumentCreated(`${collectionName}/{groupId}/messages/{msgId}`, async event => {
    const message = event.data?.data();
    const groupId = event.params.groupId;
    if (!message || !groupId) return;

    const groupSnap = await db.collection(collectionName).doc(groupId).get();
    if (!groupSnap.exists) return;
    const group = groupSnap.data() || {};
    const senderId = message.senderId;
    const recipients = (group.members || []).filter(uid => uid && uid !== senderId);
    if (!recipients.length) return;

    const groupName = group.name || group.groupTitle || group.assignmentTitle || 'Group';
    const senderName = message.senderName || 'Someone';

    await Promise.all(recipients.map(uid => sendPushToUser(uid, {
      title: `${groupName}`,
      body: `${senderName}: ${messagePreview(message)}`,
      channelId: 'unibo-messages',
      imageUrl: message.senderPhoto || undefined,
      data: {
        kind: 'group',
        groupId,
        collection: collectionName,
        profileId: senderId,
        senderName
      }
    })));
  });
}

exports.onGroupMessageCreated = createGroupMessageTrigger('groups');
exports.onAssignmentGroupMessageCreated = createGroupMessageTrigger('assignmentGroups');

exports.onUserNotificationCreated = onDocumentCreated('users/{userId}/notifications/{notifId}', async event => {
  const notification = event.data?.data();
  const userId = event.params.userId;
  const notifId = event.params.notifId;
  if (!notification || !userId || !notifId) return;

  const pushMeta = notification.pushMeta && typeof notification.pushMeta === 'object' ? notification.pushMeta : {};
  const appwriteAlreadySent = pushMeta.appwritePushSent === true || String(pushMeta.appwriteStatus || '') === 'ok';
  if (appwriteAlreadySent) {
    logger.info('Skipping Firebase push (already delivered via Appwrite)', {
      userId,
      notifId,
      mode: String(pushMeta.mode || 'appwrite-native-fcm')
    });
    return;
  }

  const from = notification.from || {};
  const payload = notification.payload || {};
  const kind = payload.convoId
    ? 'dm'
    : payload.groupId
      ? 'group'
      : payload.streamId
        ? 'live'
        : payload.postId
          ? 'post'
          : 'app';

  await sendPushToUser(userId, {
    title: from.name || 'Unibo',
    body: clampText(notification.text || 'You have a new notification', 120),
    channelId: kind === 'dm' || kind === 'group' ? 'unibo-messages' : 'unibo-general',
    imageUrl: notification.imageUrl || from.photo || undefined,
    androidIcon: notification.androidIcon || 'ic_notification_small',
    androidColor: notification.androidColor || '#6D28D9',
    clickAction: notification.clickAction || 'OPEN_UNIBO',
    data: {
      kind,
      convoId: payload.convoId || '',
      groupId: payload.groupId || '',
      collection: payload.collection || 'groups',
      postId: payload.postId || '',
      streamId: payload.streamId || '',
      profileId: from.uid || '',
      notifDocId: notifId
    }
  });
});

exports.appwritePushSync = onRequest({ cors: true }, async (req, res) => {
  if (req.method !== 'POST') {
    res.status(405).json({ ok: false, error: 'method-not-allowed' });
    return;
  }

  const authHeader = String(req.headers.authorization || '');
  const bearer = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : '';
  if (!bearer) {
    res.status(401).json({ ok: false, error: 'missing-auth' });
    return;
  }

  const { action = '', userId = '', token = '', platform = 'android' } = req.body || {};
  if (!['upsert', 'delete'].includes(action) || !userId || !token) {
    res.status(400).json({ ok: false, error: 'invalid-payload' });
    return;
  }

  try {
    const decoded = await admin.auth().verifyIdToken(bearer);
    if (decoded.uid !== userId) {
      res.status(403).json({ ok: false, error: 'uid-mismatch' });
      return;
    }

    const endpoint = process.env.APPWRITE_PUSH_SYNC_URL || '';
    if (!endpoint) {
      // Bridge disabled by config; keep client path successful.
      res.status(204).send();
      return;
    }

    const syncSecret = process.env.APPWRITE_PUSH_SYNC_SECRET || '';
    const syncResp = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(syncSecret ? { 'x-sync-secret': syncSecret } : {})
      },
      body: JSON.stringify({
        action,
        userId,
        token,
        platform,
        source: 'firebase'
      })
    });

    if (!syncResp.ok) {
      const body = await syncResp.text().catch(() => '');
      logger.warn('Appwrite sync failed', { status: syncResp.status, body: body.slice(0, 300) });
      res.status(502).json({ ok: false, error: 'bridge-failed', status: syncResp.status });
      return;
    }

    res.status(200).json({ ok: true });
  } catch (error) {
    logger.error('appwritePushSync error', error);
    res.status(500).json({ ok: false, error: 'internal' });
  }
});

exports.onLiveStreamCreated = onDocumentCreated('liveStreams/{streamId}', async event => {
  const stream = event.data?.data();
  const streamId = event.params.streamId;
  if (!stream || !streamId) return;
  if (!['live', 'starting'].includes(String(stream.status || 'live'))) return;

  const recipients = await getLiveNotificationRecipients(stream);
  if (!recipients.length) return;

  const hostName = clampText(stream.hostName || 'Someone', 40);
  const streamTitle = clampText(stream.title || 'Untitled stream', 70);
  const text = streamTitle ? `is live now — ${streamTitle}` : 'is live now';
  const payload = {
    streamId,
    title: stream.title || 'Untitled stream',
    visibility: stream.visibility || 'public'
  };
  const from = {
    uid: stream.hostUid || '',
    name: stream.hostName || 'Someone',
    photo: stream.hostPhotoURL || null
  };

  for (const ids of chunk(recipients, 400)) {
    const batch = db.batch();
    ids.forEach(userId => {
      const ref = db.collection('users').doc(userId).collection('notifications').doc(`live_${streamId}`);
      batch.set(ref, {
        type: 'live',
        text,
        payload,
        read: false,
        createdAt: admin.firestore.FieldValue.serverTimestamp(),
        from
      }, { merge: true });
    });
    await batch.commit();
  }

  logger.info('Live notifications queued', { streamId, recipients: recipients.length, hostName, visibility: stream.visibility || 'public' });
});
