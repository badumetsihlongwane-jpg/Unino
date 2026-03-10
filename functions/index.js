const admin = require('firebase-admin');
const { logger } = require('firebase-functions');
const { onDocumentCreated } = require('firebase-functions/v2/firestore');

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
    return [db.collection('users').doc(userId).collection('pushTokens').doc(tokenDocId(token.token)).delete().catch(() => {})];
  });
  await Promise.all(deletes);
}

async function sendPushToUser(userId, payload) {
  const tokenRows = await getUserTokens(userId);
  if (!tokenRows.length) return;

  const tokens = tokenRows.map(row => row.token);
  const multicast = {
    tokens,
    notification: {
      title: payload.title,
      body: payload.body
    },
    data: cleanDataMap(payload.data),
    android: {
      priority: 'high',
      notification: {
        channelId: payload.channelId || 'unibo-general',
        sound: 'default',
        imageUrl: payload.imageUrl || undefined,
        clickAction: 'OPEN_UNIBO'
      }
    },
    apns: {
      payload: {
        aps: {
          sound: 'default'
        }
      }
    }
  };

  try {
    const result = await messaging.sendEachForMulticast(multicast);
    await pruneInvalidTokens(userId, tokenRows, result.responses || []);
  } catch (error) {
    logger.error('FCM send failed', { userId, error });
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

  const from = notification.from || {};
  const payload = notification.payload || {};
  const kind = payload.convoId
    ? 'dm'
    : payload.groupId
      ? 'group'
      : payload.postId
        ? 'post'
        : 'app';

  await sendPushToUser(userId, {
    title: from.name || 'Unibo',
    body: clampText(notification.text || 'You have a new notification', 120),
    channelId: kind === 'dm' || kind === 'group' ? 'unibo-messages' : 'unibo-general',
    imageUrl: from.photo || undefined,
    data: {
      kind,
      convoId: payload.convoId || '',
      groupId: payload.groupId || '',
      collection: payload.collection || 'groups',
      postId: payload.postId || '',
      profileId: from.uid || '',
      notifDocId: notifId
    }
  });
});