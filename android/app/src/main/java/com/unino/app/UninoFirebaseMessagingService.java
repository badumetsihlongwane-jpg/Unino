package com.unino.app;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.os.Build;

import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;

import com.google.firebase.messaging.FirebaseMessagingService;
import com.google.firebase.messaging.RemoteMessage;

import java.util.Map;

public class UninoFirebaseMessagingService extends FirebaseMessagingService {
    public static final String CHANNEL_MESSAGES = "unibo-messages";
    public static final String CHANNEL_GENERAL = "unibo-general";

    @Override
    public void onMessageReceived(RemoteMessage remoteMessage) {
        super.onMessageReceived(remoteMessage);

        ensureNotificationChannels(this);

        RemoteMessage.Notification notification = remoteMessage.getNotification();
        Map<String, String> data = remoteMessage.getData();

        // Let the OS render notification payloads in background. We only render
        // a local fallback for data payloads to avoid duplicate notifications.
        if (notification != null) {
            return;
        }

        String title = valueOrDefault(data.get("title"), "Unino");
        String body = valueOrDefault(data.get("body"), "You have a new notification");
        String channelId = resolveChannelId(data);

        showSystemNotification(title, body, channelId, data);
    }

    @Override
    public void onNewToken(String token) {
        super.onNewToken(token);
        ensureNotificationChannels(this);
        // Token sync remains handled by the JS Capacitor registration path.
    }

    public static void ensureNotificationChannels(Context context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O || context == null) return;
        NotificationManager manager = context.getSystemService(NotificationManager.class);
        if (manager == null) return;

        ensureChannel(manager, CHANNEL_MESSAGES, "Messages", "Direct and group message notifications", NotificationManager.IMPORTANCE_HIGH);
        ensureChannel(manager, CHANNEL_GENERAL, "Activity", "General Unibo notifications", NotificationManager.IMPORTANCE_DEFAULT);
    }

    private static void ensureChannel(NotificationManager manager, String channelId, String name, String description, int importance) {
        NotificationChannel existing = manager.getNotificationChannel(channelId);
        if (existing != null) return;

        NotificationChannel channel = new NotificationChannel(channelId, name, importance);
        channel.setDescription(description);
        manager.createNotificationChannel(channel);
    }

    private String resolveChannelId(Map<String, String> data) {
        String channelId = valueOrDefault(data.get("channelId"), "");
        if (!channelId.isEmpty()) return channelId;
        String kind = valueOrDefault(data.get("kind"), "");
        return ("dm".equals(kind) || "group".equals(kind)) ? CHANNEL_MESSAGES : CHANNEL_GENERAL;
    }

    private void showSystemNotification(String title, String body, String channelId, Map<String, String> data) {
        ensureNotificationChannels(this);

        Intent launchIntent = new Intent(this, MainActivity.class)
            .setAction("OPEN_UNIBO")
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);

        if (data != null) {
            for (Map.Entry<String, String> entry : data.entrySet()) {
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
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(new NotificationCompat.BigTextStyle().bigText(body))
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setAutoCancel(true)
            .setContentIntent(pendingIntent)
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
            .setCategory(NotificationCompat.CATEGORY_MESSAGE);

        NotificationManagerCompat.from(this).notify((int) (System.currentTimeMillis() & 0x7fffffff), builder.build());
    }

    private String valueOrDefault(String value, String fallback) {
        if (value == null) return fallback;
        String trimmed = value.trim();
        return trimmed.isEmpty() ? fallback : trimmed;
    }
}
