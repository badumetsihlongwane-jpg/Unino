import { databases, APPWRITE_DB_ID, ID, Query } from '../config/appwrite.js';
import { state } from '../config/app-config.js';

export const DbService = {
  // ─── POSTS & FEED ──────────────────────────────────────
  async getPosts({ limit = 20, cursor = null, campus = null, moduleTag = null } = {}) {
    const queries = [
      Query.orderDesc('createdAt'),
      Query.limit(limit)
    ];

    if (cursor) {
      queries.push(Query.cursorAfter(cursor));
    }
    if (campus) {
      queries.push(Query.equal('campus', campus));
    }
    if (moduleTag) {
      queries.push(Query.search('moduleTags', moduleTag));
    }

    try {
      const res = await databases.listDocuments(APPWRITE_DB_ID, 'posts', queries);
      return res.documents;
    } catch (e) {
      console.warn('getPosts failed:', e);
      return [];
    }
  },

  async createPost({ content, mediaURL = '', visibility = 'public', isAnonymous = false, moduleTags = [] }) {
    if (!state.user) throw new Error('Must be logged in to post');
    const postId = ID.unique();
    const now = new Date().toISOString();

    const postDoc = {
      authorId: state.user.$id,
      authorName: isAnonymous ? 'Anonymous' : (state.profile?.displayName || 'User'),
      authorPhoto: isAnonymous ? '' : (state.profile?.photoURL || ''),
      content: content.trim(),
      mediaURL: mediaURL || '',
      visibility,
      isAnonymous: !!isAnonymous,
      moduleTags: moduleTags || [],
      likesCount: 0,
      commentsCount: 0,
      createdAt: now,
      updatedAt: now
    };

    return await databases.createDocument(APPWRITE_DB_ID, 'posts', postId, postDoc);
  },

  async deletePost(postId) {
    return await databases.deleteDocument(APPWRITE_DB_ID, 'posts', postId);
  },

  // ─── CHAT & MESSAGES ───────────────────────────────────
  async getConversations(userId) {
    if (!userId) return [];
    try {
      const res = await databases.listDocuments(APPWRITE_DB_ID, 'conversations', [
        Query.search('participants', userId),
        Query.orderDesc('updatedAt'),
        Query.limit(50)
      ]);
      return res.documents;
    } catch (e) {
      console.warn('getConversations failed:', e);
      return [];
    }
  },

  async getMessages(convoId, limit = 50) {
    try {
      const res = await databases.listDocuments(APPWRITE_DB_ID, 'messages', [
        Query.equal('convoId', convoId),
        Query.orderAsc('createdAt'),
        Query.limit(limit)
      ]);
      return res.documents;
    } catch (e) {
      console.warn('getMessages failed:', e);
      return [];
    }
  },

  async sendMessage(convoId, { text = '', mediaURL = '', replyTo = null }) {
    if (!state.user) throw new Error('Must be logged in to send messages');
    const msgId = ID.unique();
    const now = new Date().toISOString();

    const msgDoc = {
      convoId,
      senderId: state.user.$id,
      senderName: state.profile?.displayName || 'User',
      senderPhoto: state.profile?.photoURL || '',
      text: text.trim(),
      mediaURL: mediaURL || '',
      replyTo: replyTo ? JSON.stringify(replyTo) : '',
      createdAt: now
    };

    const created = await databases.createDocument(APPWRITE_DB_ID, 'messages', msgId, msgDoc);

    // Update conversation preview and timestamp
    try {
      await databases.updateDocument(APPWRITE_DB_ID, 'conversations', convoId, {
        lastMessage: text.slice(0, 100),
        lastSenderId: state.user.$id,
        updatedAt: now
      });
    } catch (_) {}

    return created;
  },

  // ─── MARKETPLACE / HUSTLE ──────────────────────────────
  async getListings(limit = 30) {
    try {
      const res = await databases.listDocuments(APPWRITE_DB_ID, 'listings', [
        Query.orderDesc('createdAt'),
        Query.limit(limit)
      ]);
      return res.documents;
    } catch (e) {
      console.warn('getListings failed:', e);
      return [];
    }
  },

  // ─── 24H STORIES ───────────────────────────────────────
  async getStories() {
    const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    try {
      const res = await databases.listDocuments(APPWRITE_DB_ID, 'stories', [
        Query.greaterThan('createdAt', yesterday),
        Query.orderDesc('createdAt'),
        Query.limit(100)
      ]);
      return res.documents;
    } catch (e) {
      console.warn('getStories failed:', e);
      return [];
    }
  }
};
