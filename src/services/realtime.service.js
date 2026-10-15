import { realtime, APPWRITE_DB_ID } from '../config/appwrite.js';

class RealtimeServiceManager {
  constructor() {
    this.activeSubscriptions = new Map();
  }

  /**
   * Subscribe to one or more Appwrite Realtime channels.
   * @param {string|string[]} channels Channel or array of channels (e.g. `databases.unibo_db.collections.posts.documents`)
   * @param {Function} callback Handler for incoming events
   * @param {string} [tag='general'] Optional tag to group subscriptions for lifecycle cleanup
   * @returns {Function} Unsubscribe function
   */
  subscribe(channels, callback, tag = 'general') {
    const channelList = Array.isArray(channels) ? channels : [channels];
    const subId = `sub_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    try {
      const unsub = realtime.subscribe(channelList, response => {
        try {
          callback(response);
        } catch (err) {
          console.error('Realtime callback error:', err);
        }
      });

      const entry = { unsub, channels: channelList, tag };
      this.activeSubscriptions.set(subId, entry);

      return () => {
        if (this.activeSubscriptions.has(subId)) {
          unsub();
          this.activeSubscriptions.delete(subId);
        }
      };
    } catch (e) {
      console.warn('Realtime subscription failed:', e);
      return () => {};
    }
  }

  /**
   * Helper to subscribe to collection documents
   */
  subscribeToCollection(collectionId, callback, tag = 'general') {
    const channel = `databases.${APPWRITE_DB_ID}.collections.${collectionId}.documents`;
    return this.subscribe(channel, callback, tag);
  }

  /**
   * Unsubscribe all subscriptions registered under a specific lifecycle tag
   */
  unsubscribeTag(tag) {
    for (const [subId, entry] of this.activeSubscriptions.entries()) {
      if (entry.tag === tag) {
        try {
          entry.unsub();
        } catch (_) {}
        this.activeSubscriptions.delete(subId);
      }
    }
  }

  /**
   * Clean up all active realtime subscriptions (e.g. on user logout)
   */
  unsubscribeAll() {
    for (const [_, entry] of this.activeSubscriptions.entries()) {
      try {
        entry.unsub();
      } catch (_) {}
    }
    this.activeSubscriptions.clear();
  }
}

export const RealtimeService = new RealtimeServiceManager();
