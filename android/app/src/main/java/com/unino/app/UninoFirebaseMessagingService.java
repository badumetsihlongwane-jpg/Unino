package com.unino.app;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.os.Build;
import android.text.TextUtils;

import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;

import com.google.firebase.messaging.FirebaseMessagingService;
import com.google.firebase.messaging.RemoteMessage;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Map;

public class UninoFirebaseMessagingService extends FirebaseMessagingService {
    public static final String CHANNEL_MESSAGES = "unibo-messages";
    public static final String CHANNEL_GENERAL = "unibo-general";
    private static final String PREFS_PUSH = "unino_push";
    private static final String PREF_FCM_TOKEN = "fcm_token";

    @Override
    public void onMessageReceived(RemoteMessage remoteMessage) {
        super.onMessageReceived(remoteMessage);

        ensureNotificationChannels(this);

        RemoteMessage.Notification notification = remoteMessage.getNotification();
        Map<String, String> data = remoteMessage.getData();

        String title = firstNonBlank(
            data != null ? valueOrDefault(data.get("title"), "") : "",
            notification != null ? valueOrDefault(notification.getTitle(), "") : "",
            "Unino"
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
        ensureChannel(manager, CHANNEL_GENERAL, "Activity", "General Unino notifications", NotificationManager.IMPORTANCE_HIGH);
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
            if (androidx.core.content.ContextCompat.checkSelfPermission(this, android.Manifest.permission.POST_NOTIFICATIONS) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                return;
            }
        }

        Intent launchIntent = new Intent(this, MainActivity.class)
            .setAction("OPEN_UNIBO")
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);

        if (data != null) {
            for (Map.Entry<String, String> entry : data.entrySet()) {
                if (entry.getKey() == null || entry.getValue() == null) continue;
                launchIntent.putExtra(entry.getKey(), entry.getValue());
            }
        }

        int notificationId = getNotificationId(remoteMessage, data, title, body);

        PendingIntent pendingIntent = PendingIntent.getActivity(
            this,
            notificationId,
            launchIntent,
            PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
        );

        NotificationCompat.Builder builder = new NotificationCompat.Builder(this, channelId)
            .setSmallIcon(R.drawable.ic_notification_small)
            .setColor(0xFF6D28D9)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(new NotificationCompat.BigTextStyle().bigText(body))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setDefaults(NotificationCompat.DEFAULT_ALL)
            .setVibrate(new long[]{0, 250, 250, 250})
            .setAutoCancel(true)
            .setContentIntent(pendingIntent)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setCategory(NotificationCompat.CATEGORY_MESSAGE);

        Bitmap largeIcon = null;
        String senderPhoto = data != null ? valueOrDefault(data.get("senderPhoto"), "") : "";
        if (senderPhoto.startsWith("https://")) {
            largeIcon = fetchCircularAvatar(senderPhoto);
        }
        if (largeIcon == null) {
            try {
                largeIcon = BitmapFactory.decodeResource(getResources(), R.mipmap.ic_launcher);
            } catch (Exception ignored) {}
        }
        if (largeIcon != null) {
            builder.setLargeIcon(largeIcon);
        }

        try {
            NotificationManagerCompat.from(this).notify(notificationId, builder.build());
        } catch (Exception ignored) {}
    }

    private Bitmap fetchCircularAvatar(String urlStr) {
        HttpURLConnection conn = null;
        InputStream in = null;
        try {
            URL url = new URL(urlStr);
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(2500);
            conn.setReadTimeout(2500);
            conn.setInstanceFollowRedirects(true);
            conn.setDoInput(true);
            if (conn.getResponseCode() != HttpURLConnection.HTTP_OK) return null;

            in = conn.getInputStream();
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            byte[] chunk = new byte[4096];
            int total = 0;
            int read;
            final int MAX_BYTES = 512 * 1024;
            while ((read = in.read(chunk)) != -1) {
                total += read;
                if (total > MAX_BYTES) return null;
                buffer.write(chunk, 0, read);
            }
            byte[] bytes = buffer.toByteArray();
            if (bytes.length == 0) return null;

            BitmapFactory.Options bounds = new BitmapFactory.Options();
            bounds.inJustDecodeBounds = true;
            BitmapFactory.decodeByteArray(bytes, 0, bytes.length, bounds);
            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null;

            int sample = 1;
            int minDim = Math.min(bounds.outWidth, bounds.outHeight);
            while (minDim / (sample * 2) >= 128) sample *= 2;

            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inSampleSize = sample;
            Bitmap decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.length, opts);
            if (decoded == null) return null;

            Bitmap circle = toCircularBitmap(decoded, 128);
            if (circle != decoded) decoded.recycle();
            return circle;
        } catch (Exception ignored) {
            return null;
        } finally {
            try { if (in != null) in.close(); } catch (Exception ignored) {}
            if (conn != null) conn.disconnect();
        }
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
