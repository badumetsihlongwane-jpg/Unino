package com.unino.app;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Build;
import android.text.TextUtils;
import android.util.LruCache;

import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;
import androidx.core.app.Person;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.core.content.pm.ShortcutInfoCompat;
import androidx.core.content.pm.ShortcutManagerCompat;
import androidx.core.graphics.drawable.IconCompat;

import com.google.firebase.messaging.FirebaseMessagingService;
import com.google.firebase.messaging.RemoteMessage;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class UninoFirebaseMessagingService extends FirebaseMessagingService {
    public static final String CHANNEL_MESSAGES = "unibo-messages";
    public static final String CHANNEL_GENERAL = "unibo-general";
    private static final String PREFS_PUSH = "unino_push";
    private static final String PREF_FCM_TOKEN = "fcm_token";
    private static final String BRIDGE_AVATAR_BASE = "https://69dd474c000ed9db9d03.fra.appwrite.run/avatar/";
    private static final Pattern PRIVATE_AVATAR_PATTERN = Pattern.compile(
        "/storage/buckets/6b06c81b00165508f719/files/([a-zA-Z0-9_-]+)/(?:view|preview|download)",
        Pattern.CASE_INSENSITIVE
    );
    private static final int[] INITIAL_COLORS = new int[]{
        0xFF6D28D9, 0xFF2563EB, 0xFF0D9488, 0xFFD97706, 0xFFDB2777, 0xFF059669, 0xFF4F46E5
    };
    private static final LruCache<String, byte[]> AVATAR_BYTES_CACHE = new LruCache<>(16);

    @Override
    public void onMessageReceived(RemoteMessage remoteMessage) {
        super.onMessageReceived(remoteMessage);

        ensureNotificationChannels(this);

        RemoteMessage.Notification notification = remoteMessage.getNotification();
        Map<String, String> data = remoteMessage.getData();

        String title = firstNonBlank(
            data != null ? valueOrDefault(data.get("title"), "") : "",
            notification != null ? valueOrDefault(notification.getTitle(), "") : "",
            "Unibo"
        );
        String body = firstNonBlank(
            data != null ? valueOrDefault(data.get("body"), "") : "",
            data != null ? valueOrDefault(data.get("text"), "") : "",
            notification != null ? valueOrDefault(notification.getBody(), "") : "",
            "You have a new notification"
        );

        showSystemNotification(title, body, resolveChannelId(data), data, remoteMessage);
    }

    @Override
    public void onNewToken(String token) {
        super.onNewToken(token);
        ensureNotificationChannels(this);
        storeFcmToken(this, token);
        // Final sync to Firestore/Appwrite still happens through the JS bridge path.
    }

    public static void storeFcmToken(Context context, String token) {
        if (context == null || TextUtils.isEmpty(token)) return;
        context.getSharedPreferences(PREFS_PUSH, Context.MODE_PRIVATE)
            .edit()
            .putString(PREF_FCM_TOKEN, token)
            .apply();
    }

    public static String getStoredFcmToken(Context context) {
        if (context == null) return "";
        return context.getSharedPreferences(PREFS_PUSH, Context.MODE_PRIVATE)
            .getString(PREF_FCM_TOKEN, "");
    }

    public static void ensureNotificationChannels(Context context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O || context == null) return;
        NotificationManager manager = context.getSystemService(NotificationManager.class);
        if (manager == null) return;

        ensureChannel(manager, CHANNEL_MESSAGES, "Messages", "Direct and group message notifications", NotificationManager.IMPORTANCE_HIGH);
        ensureChannel(manager, CHANNEL_GENERAL, "Activity", "General Unibo notifications", NotificationManager.IMPORTANCE_HIGH);
    }

    private static void ensureChannel(NotificationManager manager, String channelId, String name, String description, int importance) {
        NotificationChannel existing = manager.getNotificationChannel(channelId);
        if (existing != null) return;

        NotificationChannel channel = new NotificationChannel(channelId, name, importance);
        channel.setDescription(description);
        channel.enableVibration(true);
        channel.enableLights(true);
        channel.setLightColor(0xFF6D28D9);
        channel.setLockscreenVisibility(android.app.Notification.VISIBILITY_PUBLIC);
        manager.createNotificationChannel(channel);
    }

    private String resolveChannelId(Map<String, String> data) {
        if (data == null) return CHANNEL_GENERAL;
        String explicitChannel = valueOrDefault(data.get("channelId"), "");
        if (CHANNEL_MESSAGES.equalsIgnoreCase(explicitChannel)) return CHANNEL_MESSAGES;
        if (CHANNEL_GENERAL.equalsIgnoreCase(explicitChannel)) return CHANNEL_GENERAL;

        String kind = valueOrDefault(data.get("kind"), "");
        return ("dm".equalsIgnoreCase(kind) || "group".equalsIgnoreCase(kind)) ? CHANNEL_MESSAGES : CHANNEL_GENERAL;
    }

    private int getNotificationId(RemoteMessage remoteMessage, Map<String, String> data, String title, String body) {
        if (data != null) {
            String notifDocId = data.get("notifDocId");
            if (!TextUtils.isEmpty(notifDocId)) return Math.abs(notifDocId.hashCode());
            String docId = data.get("docId");
            if (!TextUtils.isEmpty(docId)) return Math.abs(docId.hashCode());
            String msgId = data.get("msgId");
            if (!TextUtils.isEmpty(msgId)) return Math.abs(msgId.hashCode());
            String convoId = data.get("convoId");
            if (!TextUtils.isEmpty(convoId)) return Math.abs(("convo_" + convoId).hashCode());
            String groupId = data.get("groupId");
            if (!TextUtils.isEmpty(groupId)) return Math.abs(("group_" + groupId).hashCode());
        }
        if (remoteMessage != null && !TextUtils.isEmpty(remoteMessage.getMessageId())) {
            return Math.abs(remoteMessage.getMessageId().hashCode());
        }
        return Math.abs((title + ":" + body).hashCode());
    }

    private void showSystemNotification(String title, String body, String channelId, Map<String, String> data, RemoteMessage remoteMessage) {
        ensureNotificationChannels(this);

        if (Build.VERSION.SDK_INT >= 33) {
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.POST_NOTIFICATIONS) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                return;
            }
        }

        Intent launchIntent = buildLaunchIntent(data);
        int notificationId = getNotificationId(remoteMessage, data, title, body);

        PendingIntent pendingIntent = PendingIntent.getActivity(
            this,
            notificationId,
            launchIntent,
            PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
        );

        String kind = data != null ? valueOrDefault(data.get("kind"), "") : "";
        String convoId = data != null ? valueOrDefault(data.get("convoId"), "") : "";
        String groupId = data != null ? valueOrDefault(data.get("groupId"), "") : "";
        String groupName = data != null ? valueOrDefault(data.get("groupName"), "") : "";
        String profileId = data != null ? valueOrDefault(data.get("profileId"), "") : "";
        String senderName = firstNonBlank(
            data != null ? valueOrDefault(data.get("senderName"), "") : "",
            title,
            "Unibo"
        );
        String senderPhoto = resolvePublicAvatarUrl(data != null ? valueOrDefault(data.get("senderPhoto"), "") : "");
        String mediaUrl = data != null ? valueOrDefault(data.get("mediaUrl"), "") : "";

        boolean isDm = "dm".equalsIgnoreCase(kind) || (!TextUtils.isEmpty(convoId) && !"group".equalsIgnoreCase(kind));
        boolean isGroup = "group".equalsIgnoreCase(kind) || !TextUtils.isEmpty(groupId);
        boolean isAnonymous = "anonymous".equalsIgnoreCase(profileId) || "Anonymous".equalsIgnoreCase(senderName);
        boolean isSystem = "timetable".equalsIgnoreCase(kind) || "timetable".equalsIgnoreCase(profileId);

        // Fetch sender profile photo (NEVER used as a BigPicture sent-image preview)
        byte[] avatarBytes = null;
        if (!isAnonymous && !isSystem && senderPhoto.startsWith("https://")) {
            avatarBytes = fetchAvatarBytesCached(senderPhoto);
        }

        // Fetch actual sent photo attachment ONLY when mediaUrl is explicitly present
        Bitmap sentImageBitmap = null;
        Uri sentImageUri = null;
        if (mediaUrl.startsWith("https://")) {
            sentImageBitmap = fetchScaledBitmap(mediaUrl, 4 * 1024 * 1024, 1024, 4000);
            if (sentImageBitmap != null) {
                sentImageUri = saveBitmapForNotificationMedia(sentImageBitmap, notificationId);
            }
        }

        NotificationCompat.Builder builder = new NotificationCompat.Builder(this, channelId)
            .setSmallIcon(R.drawable.ic_notification_small)
            .setColor(0xFF6D28D9)
            .setContentTitle(title)
            .setContentText(body)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setDefaults(NotificationCompat.DEFAULT_ALL)
            .setVibrate(new long[]{0, 250, 250, 250})
            .setAutoCancel(true)
            .setContentIntent(pendingIntent)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setCategory((isDm || isGroup) ? NotificationCompat.CATEGORY_MESSAGE : NotificationCompat.CATEGORY_SOCIAL);

        boolean useConversationStyle = !isAnonymous && !isSystem && (isDm || isGroup || avatarBytes != null);

        if (useConversationStyle) {
            Bitmap rawAvatar = avatarBytes != null ? decodeSampledBitmap(avatarBytes, 256) : null;
            Bitmap adaptiveAvatar = rawAvatar != null
                ? toAdaptiveAvatarBitmap(rawAvatar, 192)
                : createInitialsAvatarBitmap(senderName, 192);
            Bitmap legacyCircle = (Build.VERSION.SDK_INT < Build.VERSION_CODES.P && rawAvatar != null)
                ? toCircularBitmap(rawAvatar, 128)
                : null;
            if (rawAvatar != null) rawAvatar.recycle();

            IconCompat senderIcon = adaptiveAvatar != null
                ? IconCompat.createWithAdaptiveBitmap(adaptiveAvatar)
                : null;

            String senderKey = !TextUtils.isEmpty(profileId) ? ("user_" + sanitizeId(profileId)) : ("name_" + Math.abs(senderName.hashCode()));
            Person.Builder senderBuilder = new Person.Builder()
                .setKey(senderKey)
                .setName(senderName);
            if (senderIcon != null) {
                senderBuilder.setIcon(senderIcon);
            }
            Person senderPerson = senderBuilder.build();

            Person selfPerson = new Person.Builder()
                .setKey("unibo_self")
                .setName("You")
                .build();

            // Publish a long-lived dynamic conversation shortcut so Android 11+ & Samsung One UI
            // render the sender's circular profile photo on the LEFT instead of the app/location icon.
            String shortcutId = buildConversationShortcutId(isDm, isGroup, convoId, groupId, profileId, notificationId);
            String shortcutLabel = (isGroup && !TextUtils.isEmpty(groupName)) ? groupName : senderName;
            boolean shortcutPublished = pushConversationShortcut(shortcutId, shortcutLabel, senderPerson, senderIcon, buildLaunchIntent(data));
            if (shortcutPublished) {
                builder.setShortcutId(shortcutId);
            }

            String displayBody = body;
            if (isGroup && !TextUtils.isEmpty(groupName)) {
                String suffix = " in " + groupName;
                if (displayBody.endsWith(suffix) && displayBody.length() > suffix.length()) {
                    displayBody = displayBody.substring(0, displayBody.length() - suffix.length()).trim();
                }
            }

            if (sentImageBitmap != null && (sentImageUri == null || Build.VERSION.SDK_INT < Build.VERSION_CODES.P)) {
                // Fallback on Android < 9 or if FileProvider URI unavailable — show sent photo in BigPictureStyle
                NotificationCompat.BigPictureStyle bigPic = new NotificationCompat.BigPictureStyle()
                    .bigPicture(sentImageBitmap)
                    .setSummaryText(displayBody);
                bigPic.bigLargeIcon((Bitmap) null);
                builder.setStyle(bigPic);
                if (legacyCircle != null) {
                    builder.setLargeIcon(legacyCircle);
                }
            } else {
                NotificationCompat.MessagingStyle messagingStyle = new NotificationCompat.MessagingStyle(selfPerson);
                if (isGroup && !TextUtils.isEmpty(groupName)) {
                    messagingStyle.setGroupConversation(true);
                    messagingStyle.setConversationTitle(groupName);
                } else {
                    messagingStyle.setGroupConversation(false);
                }

                long now = System.currentTimeMillis();
                if (sentImageUri != null) {
                    String caption = extractPhotoCaption(displayBody);
                    if (!TextUtils.isEmpty(caption)) {
                        messagingStyle.addMessage(new NotificationCompat.MessagingStyle.Message(caption, now - 1, senderPerson));
                    }
                    NotificationCompat.MessagingStyle.Message mediaMessage =
                        new NotificationCompat.MessagingStyle.Message(
                            !TextUtils.isEmpty(caption) ? caption : "📷 Photo",
                            now,
                            senderPerson
                        );
                    mediaMessage.setData("image/jpeg", sentImageUri);
                    messagingStyle.addMessage(mediaMessage);
                } else {
                    messagingStyle.addMessage(new NotificationCompat.MessagingStyle.Message(displayBody, now, senderPerson));
                }

                builder.setStyle(messagingStyle);

                // On Android 9+ (API 28+), avoid calling setLargeIcon alongside MessagingStyle so
                // Samsung One UI does not render a duplicate avatar thumbnail on the right side.
                if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P && legacyCircle != null) {
                    builder.setLargeIcon(legacyCircle);
                }
            }
        } else {
            // System, timetable, or anonymous notification
            if (sentImageBitmap != null) {
                NotificationCompat.BigPictureStyle bigPic = new NotificationCompat.BigPictureStyle()
                    .bigPicture(sentImageBitmap)
                    .setSummaryText(body);
                bigPic.bigLargeIcon((Bitmap) null);
                builder.setStyle(bigPic);
                builder.setLargeIcon(sentImageBitmap);
            } else {
                builder.setStyle(new NotificationCompat.BigTextStyle().bigText(body));
                try {
                    Bitmap appLogo = BitmapFactory.decodeResource(getResources(), R.mipmap.ic_launcher);
                    if (appLogo != null) {
                        builder.setLargeIcon(appLogo);
                    }
                } catch (Exception ignored) {}
            }
        }

        try {
            NotificationManagerCompat.from(this).notify(notificationId, builder.build());
        } catch (Exception ignored) {}
    }

    private Intent buildLaunchIntent(Map<String, String> data) {
        Intent launchIntent = new Intent(this, MainActivity.class)
            .setAction("OPEN_UNIBO")
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        if (data != null) {
            for (Map.Entry<String, String> entry : data.entrySet()) {
                if (entry.getKey() == null || entry.getValue() == null) continue;
                launchIntent.putExtra(entry.getKey(), entry.getValue());
            }
        }
        return launchIntent;
    }

    private String buildConversationShortcutId(boolean isDm, boolean isGroup, String convoId, String groupId, String profileId, int notificationId) {
        if (isDm && !TextUtils.isEmpty(convoId)) return "unibo_dm_" + sanitizeId(convoId);
        if (isGroup && !TextUtils.isEmpty(groupId)) return "unibo_grp_" + sanitizeId(groupId);
        if (!TextUtils.isEmpty(profileId)) return "unibo_usr_" + sanitizeId(profileId);
        return "unibo_msg_" + notificationId;
    }

    private String sanitizeId(String raw) {
        if (raw == null) return "";
        String cleaned = raw.replaceAll("[^a-zA-Z0-9_-]", "_");
        return cleaned.length() > 48 ? cleaned.substring(0, 48) : cleaned;
    }

    private boolean pushConversationShortcut(String shortcutId, String label, Person person, IconCompat icon, Intent intent) {
        try {
            if (TextUtils.isEmpty(shortcutId) || TextUtils.isEmpty(label)) return false;
            ShortcutInfoCompat.Builder sb = new ShortcutInfoCompat.Builder(this, shortcutId)
                .setShortLabel(label)
                .setLongLived(true)
                .setPerson(person)
                .setIntent(intent);
            if (icon != null) {
                sb.setIcon(icon);
            }
            ShortcutManagerCompat.pushDynamicShortcut(this, sb.build());
            return true;
        } catch (Exception ignored) {
            return false;
        }
    }

    private String extractPhotoCaption(String body) {
        if (TextUtils.isEmpty(body)) return "";
        String trimmed = body.trim();
        if (trimmed.startsWith("📷 Photo:")) {
            return trimmed.substring("📷 Photo:".length()).trim();
        }
        if ("📷 Photo".equals(trimmed) || "sent a photo".equalsIgnoreCase(trimmed)) {
            return "";
        }
        return trimmed;
    }

    private String resolvePublicAvatarUrl(String urlStr) {
        if (TextUtils.isEmpty(urlStr)) return "";
        Matcher m = PRIVATE_AVATAR_PATTERN.matcher(urlStr);
        if (m.find() && m.group(1) != null) {
            return BRIDGE_AVATAR_BASE + m.group(1);
        }
        return urlStr;
    }

    private byte[] fetchAvatarBytesCached(String urlStr) {
        if (TextUtils.isEmpty(urlStr)) return null;
        byte[] cached = AVATAR_BYTES_CACHE.get(urlStr);
        if (cached != null && cached.length > 0) return cached;
        byte[] fetched = fetchBytes(urlStr, 512 * 1024, 3000);
        if (fetched != null && fetched.length > 0) {
            AVATAR_BYTES_CACHE.put(urlStr, fetched);
        }
        return fetched;
    }

    private Bitmap fetchScaledBitmap(String urlStr, int maxBytes, int maxDimPx, int timeoutMs) {
        byte[] bytes = fetchBytes(urlStr, maxBytes, timeoutMs);
        if (bytes == null || bytes.length == 0) return null;
        return decodeSampledBitmap(bytes, maxDimPx);
    }

    private byte[] fetchBytes(String urlStr, int maxBytes, int timeoutMs) {
        HttpURLConnection conn = null;
        InputStream in = null;
        try {
            URL url = new URL(urlStr);
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(timeoutMs);
            conn.setReadTimeout(timeoutMs);
            conn.setInstanceFollowRedirects(true);
            conn.setDoInput(true);
            conn.setRequestProperty("User-Agent", "Unibo-Android/1.0");
            if (conn.getResponseCode() != HttpURLConnection.HTTP_OK) return null;

            in = conn.getInputStream();
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            byte[] chunk = new byte[4096];
            int total = 0;
            int read;
            while ((read = in.read(chunk)) != -1) {
                total += read;
                if (total > maxBytes) return null;
                buffer.write(chunk, 0, read);
            }
            byte[] bytes = buffer.toByteArray();
            return bytes.length > 0 ? bytes : null;
        } catch (Exception ignored) {
            return null;
        } finally {
            try { if (in != null) in.close(); } catch (Exception ignored) {}
            if (conn != null) conn.disconnect();
        }
    }

    private Bitmap decodeSampledBitmap(byte[] bytes, int targetMaxDim) {
        try {
            BitmapFactory.Options bounds = new BitmapFactory.Options();
            bounds.inJustDecodeBounds = true;
            BitmapFactory.decodeByteArray(bytes, 0, bytes.length, bounds);
            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null;

            int sample = 1;
            int maxDim = Math.max(bounds.outWidth, bounds.outHeight);
            while (maxDim / (sample * 2) >= targetMaxDim) sample *= 2;

            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inSampleSize = sample;
            return BitmapFactory.decodeByteArray(bytes, 0, bytes.length, opts);
        } catch (Exception ignored) {
            return null;
        }
    }

    private Uri saveBitmapForNotificationMedia(Bitmap bitmap, int notificationId) {
        if (bitmap == null) return null;
        FileOutputStream fos = null;
        try {
            File dir = new File(getCacheDir(), "notif_media");
            if (!dir.exists()) dir.mkdirs();
            pruneOldMediaFiles(dir, 12);

            File outFile = new File(dir, "img_" + notificationId + "_" + System.currentTimeMillis() + ".jpg");
            fos = new FileOutputStream(outFile);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 86, fos);
            fos.flush();

            Uri contentUri = FileProvider.getUriForFile(this, getPackageName() + ".fileprovider", outFile);
            String[] uiPackages = new String[]{
                "com.android.systemui",
                "com.samsung.android.app.aodservice",
                "com.samsung.android.app.cocktailbarservice"
            };
            for (String pkg : uiPackages) {
                try {
                    grantUriPermission(pkg, contentUri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
                } catch (Exception ignored) {}
            }
            return contentUri;
        } catch (Exception ignored) {
            return null;
        } finally {
            try { if (fos != null) fos.close(); } catch (Exception ignored) {}
        }
    }

    private void pruneOldMediaFiles(File dir, int maxKeep) {
        try {
            File[] files = dir.listFiles();
            if (files == null || files.length <= maxKeep) return;
            Arrays.sort(files, new Comparator<File>() {
                @Override
                public int compare(File a, File b) {
                    return Long.compare(b.lastModified(), a.lastModified());
                }
            });
            for (int i = maxKeep; i < files.length; i++) {
                files[i].delete();
            }
        } catch (Exception ignored) {}
    }

    private Bitmap toAdaptiveAvatarBitmap(Bitmap src, int sizePx) {
        int side = Math.min(src.getWidth(), src.getHeight());
        int x = (src.getWidth() - side) / 2;
        int y = (src.getHeight() - side) / 2;
        Rect srcRect = new Rect(x, y, x + side, y + side);

        Bitmap output = Bitmap.createBitmap(sizePx, sizePx, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(output);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        canvas.drawBitmap(src, srcRect, new Rect(0, 0, sizePx, sizePx), paint);
        return output;
    }

    private Bitmap createInitialsAvatarBitmap(String name, int sizePx) {
        String initials = extractInitials(name);
        int bgColor = INITIAL_COLORS[Math.abs(initials.hashCode() + (name != null ? name.hashCode() : 0)) % INITIAL_COLORS.length];

        Bitmap output = Bitmap.createBitmap(sizePx, sizePx, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(output);
        canvas.drawColor(bgColor);

        Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(sizePx * 0.34f);
        textPaint.setTypeface(Typeface.create(Typeface.DEFAULT, Typeface.BOLD));
        textPaint.setTextAlign(Paint.Align.CENTER);

        Paint.FontMetrics fm = textPaint.getFontMetrics();
        float cy = (sizePx / 2f) - ((fm.ascent + fm.descent) / 2f);
        canvas.drawText(initials, sizePx / 2f, cy, textPaint);
        return output;
    }

    private String extractInitials(String name) {
        if (TextUtils.isEmpty(name)) return "U";
        String[] parts = name.trim().split("\\s+");
        if (parts.length == 0 || TextUtils.isEmpty(parts[0])) return "U";
        String first = parts[0].substring(0, 1).toUpperCase();
        String second = (parts.length > 1 && !TextUtils.isEmpty(parts[1]))
            ? parts[1].substring(0, 1).toUpperCase()
            : "";
        return first + second;
    }

    private Bitmap toCircularBitmap(Bitmap src, int sizePx) {
        int side = Math.min(src.getWidth(), src.getHeight());
        int x = (src.getWidth() - side) / 2;
        int y = (src.getHeight() - side) / 2;
        Bitmap output = Bitmap.createBitmap(sizePx, sizePx, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(output);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        float radius = sizePx / 2f;
        canvas.drawCircle(radius, radius, radius, paint);
        paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.SRC_IN));
        Rect srcRect = new Rect(x, y, x + side, y + side);
        Rect dstRect = new Rect(0, 0, sizePx, sizePx);
        canvas.drawBitmap(src, srcRect, dstRect, paint);
        return output;
    }

    private String firstNonBlank(String... items) {
        if (items != null) {
            for (String item : items) {
                if (!TextUtils.isEmpty(item)) return item;
            }
        }
        return "";
    }

    private String valueOrDefault(String value, String fallback) {
        if (value == null) return fallback;
        String trimmed = value.trim();
        return trimmed.isEmpty() ? fallback : trimmed;
    }
}
