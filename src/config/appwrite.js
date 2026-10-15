import { Client, Account, Databases, Storage, Realtime, ID, Query } from 'appwrite';

export const APPWRITE_ENDPOINT = window.UNINO_APPWRITE_ENDPOINT || 'https://fra.cloud.appwrite.io/v1';
export const APPWRITE_PROJECT_ID = window.UNINO_APPWRITE_PROJECT_ID || '69dd43b6002414cdfb70';
export const APPWRITE_DB_ID = 'unibo_db';
export const APPWRITE_STORAGE_BUCKET_ID = 'unibo_media';

export const client = new Client();
client.setEndpoint(APPWRITE_ENDPOINT).setProject(APPWRITE_PROJECT_ID);

export const account = new Account(client);
export const databases = new Databases(client);
export const storage = new Storage(client);
export const realtime = new Realtime(client);
export { ID, Query };
