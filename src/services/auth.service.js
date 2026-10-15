import { account, databases, APPWRITE_DB_ID, ID, Query } from '../config/appwrite.js';
import { state } from '../config/app-config.js';

export const AuthService = {
  async getCurrentSession() {
    try {
      const user = await account.get();
      state.user = user;
      const profile = await this.getUserProfile(user.$id);
      state.profile = profile || {
        displayName: user.name || 'User',
        email: user.email,
        uid: user.$id
      };
      return { user, profile: state.profile };
    } catch (e) {
      state.user = null;
      state.profile = null;
      return null;
    }
  },

  async login(email, password) {
    // Delete any active session first to avoid multi-session errors
    try {
      await account.deleteSession('current');
    } catch (_) {}
    const session = await account.createEmailPasswordSession(email, password);
    return this.getCurrentSession();
  },

  async signup(email, password, name, extraData = {}) {
    const userId = ID.unique();
    await account.create(userId, email, password, name);
    await account.createEmailPasswordSession(email, password);

    // Create user profile in Appwrite database
    const profileData = {
      uid: userId,
      displayName: name,
      email,
      major: extraData.major || '',
      faculty: extraData.faculty || '',
      residence: extraData.residence || '',
      campus: extraData.campus || 'potch',
      bio: extraData.bio || '',
      photoURL: extraData.photoURL || '',
      isVerified: false,
      manualVerified: false,
      isAdmin: false,
      status: 'online',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    try {
      await databases.createDocument(APPWRITE_DB_ID, 'users', userId, profileData);
    } catch (e) {
      console.warn('Could not auto-create users collection document:', e);
    }

    return this.getCurrentSession();
  },

  async logout() {
    try {
      await account.deleteSession('current');
    } catch (e) {
      console.warn('Session delete error:', e);
    }
    state.user = null;
    state.profile = null;
  },

  async getUserProfile(userId) {
    try {
      return await databases.getDocument(APPWRITE_DB_ID, 'users', userId);
    } catch (e) {
      // Fallback query if document ID differs from user ID
      try {
        const res = await databases.listDocuments(APPWRITE_DB_ID, 'users', [
          Query.equal('uid', userId),
          Query.limit(1)
        ]);
        return res.documents[0] || null;
      } catch (_) {
        return null;
      }
    }
  },

  async updateProfile(userId, updates = {}) {
    updates.updatedAt = new Date().toISOString();
    try {
      const updated = await databases.updateDocument(APPWRITE_DB_ID, 'users', userId, updates);
      if (state.profile && state.user?.$id === userId) {
        state.profile = { ...state.profile, ...updated };
      }
      return updated;
    } catch (e) {
      console.error('Update profile failed:', e);
      throw e;
    }
  }
};
