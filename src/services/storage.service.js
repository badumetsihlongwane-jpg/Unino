import { storage, APPWRITE_STORAGE_BUCKET_ID, ID } from '../config/appwrite.js';

export const StorageService = {
  async uploadFile(file, bucketId = APPWRITE_STORAGE_BUCKET_ID) {
    if (!file) throw new Error('No file provided for upload');
    const fileId = ID.unique();
    const result = await storage.createFile(bucketId, fileId, file);
    const viewUrl = storage.getFileView(bucketId, result.$id).toString();
    return {
      fileId: result.$id,
      url: viewUrl,
      name: result.name,
      sizeOriginal: result.sizeOriginal,
      mimeType: result.mimeType
    };
  },

  async deleteFile(fileId, bucketId = APPWRITE_STORAGE_BUCKET_ID) {
    if (!fileId) return;
    try {
      await storage.deleteFile(bucketId, fileId);
    } catch (e) {
      console.warn('Storage delete failed:', e);
    }
  },

  getFileViewUrl(fileId, bucketId = APPWRITE_STORAGE_BUCKET_ID) {
    if (!fileId) return '';
    return storage.getFileView(bucketId, fileId).toString();
  },

  getFilePreviewUrl(fileId, width = 400, height = 400, bucketId = APPWRITE_STORAGE_BUCKET_ID) {
    if (!fileId) return '';
    return storage.getFilePreview(bucketId, fileId, width, height).toString();
  }
};
