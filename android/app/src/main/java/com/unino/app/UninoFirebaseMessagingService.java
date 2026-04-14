package com.unino.app;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.text.TextUtils;

import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;

import com.google.firebase.messaging.FirebaseMessagingService;
import com.google.firebase.messaging.RemoteMessage;

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
            valueOrDefault(data.get("title"), ""),
            notification != null ? valueOrDefault(notification.getTitle(), "") : "",
            "Unino"
        );
        String body = firstNonBlank(
            valueOrDefault(data.get("body"), ""),
            notification != null ? valueOrDefault(notification.getBody(), "") : "",
            "You have a new notification"
        );

        showSystemNotification(title, body, resolveChannelId(data), data);
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
        ensureChannel(manager, CHANNEL_GENERAL, "Activity", "General Unino notifications", NotificationManager.IMPORTANCE_DEFAULT);
    }

    private static void ensureChannel(NotificationManager manager, String channelId, String name, String description, int importance) {
        NotificationChannel existing = manager.getNotificationChannel(channelId);
        if (existing != null) return;

        NotificationChannel channel = new NotificationChannel(channelId, name, importance);
        channel.setDescription(description);
        manager.createNotificationChannel(channel);
    }

    private String resolveChannelId(Map<String, String> data) {
        String explicitChannel = valueOrDefault(data.get("channelId"), "");
        if (!TextUtils.isEmpty(explicitChannel)) return explicitChannel;

        String kind = valueOrDefault(data.get("kind"), "");
        return ("dm".equalsIgnoreCase(kind) || "group".equalsIgnoreCase(kind)) ? CHANNEL_MESSAGES : CHANNEL_GENERAL;
    }

    private void showSystemNotification(String title, String body, String channelId, Map<String, String> data) {
        ensureNotificationChannels(this);

        Intent launchIntent = new Intent(this, MainActivity.class)
            .setAction("OPEN_UNIBO")
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);

        if (data != null) {
            for (Map.Entry<String, String> entry : data.entrySet()) {
                if (entry.getKey() == null || entry.getValue() == null) continue;
                launchIntent.putExtra(entry.getKey(), entry.getValue());
            }
        }

        PendingIntent pendingIntent = PendingIntent.getActivity(
            this,
            (int) (System.currentTimeMillis() & 0x7fffffff),
            launchIntent,
            PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
        );

        NotificationCompat.Builder builder = new NotificationCompat.Builder(this, channelId)
            .setSmallIcon(R.drawable.ic_notification_small)
            .setColor(0xFF6D28D9)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(new NotificationCompat.BigTextStyle().bigText(body))
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setDefaults(NotificationCompat.DEFAULT_ALL)
            .setAutoCancel(true)
            .setContentIntent(pendingIntent)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setCategory(NotificationCompat.CATEGORY_MESSAGE);

        NotificationManagerCompat.from(this).notify((int) (System.currentTimeMillis() & 0x7fffffff), builder.build());
    }

    private String firstNonBlank(String first, String second, String fallback) {
        if (!TextUtils.isEmpty(first)) return first;
        if (!TextUtils.isEmpty(second)) return second;
        return fallback;
    }

    private String valueOrDefault(String value, String fallback) {
        if (value == null) return fallback;
        String trimmed = value.trim();
        return trimmed.isEmpty() ? fallback : trimmed;
    }
}
