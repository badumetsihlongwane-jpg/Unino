package com.unino.app;

import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;

import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.WindowInsetsControllerCompat;

import org.json.JSONException;
import org.json.JSONObject;

import android.content.Intent;

import com.getcapacitor.BridgeActivity;
import com.google.firebase.FirebaseApp;
import com.google.firebase.messaging.FirebaseMessaging;
import io.appwrite.Client;

import java.lang.reflect.Method;

public class MainActivity extends BridgeActivity {
	private static final String TAG = "MainActivity";
	private static final String APPWRITE_ENDPOINT = "https://fra.cloud.appwrite.io/v1";
	private static final String APPWRITE_PROJECT_ID = "69dd43b6002414cdfb70";

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		UninoFirebaseMessagingService.ensureNotificationChannels(this);

		try {
			FirebaseApp.initializeApp(this);
			FirebaseMessaging.getInstance().getToken().addOnCompleteListener(task -> {
				if (!task.isSuccessful()) {
					Log.w(TAG, "Unable to fetch FCM token", task.getException());
					return;
				}
				String token = task.getResult();
				UninoFirebaseMessagingService.storeFcmToken(this, token);
				dispatchNativeFcmToken(token);
				Log.i(TAG, "FCM token fetched");
			});
		} catch (Exception e) {
			Log.w(TAG, "Firebase init/token fetch failed", e);
		}

		try {
			Client appwriteClient = new Client(this);
			invokeClientSetter(appwriteClient, "setEndpoint", APPWRITE_ENDPOINT);
			invokeClientSetter(appwriteClient, "setProject", APPWRITE_PROJECT_ID);
			Log.i(TAG, "Appwrite client initialized");
		} catch (Exception e) {
			Log.w(TAG, "Appwrite client init failed", e);
		}

		dispatchNativeFcmToken(UninoFirebaseMessagingService.getStoredFcmToken(this));

		Window window = getWindow();

		// Clear any fullscreen / translucent flags the splash theme may have set
		window.clearFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
		window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS);
		window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);

		// Content must NOT draw behind the status bar
		WindowCompat.setDecorFitsSystemWindows(window, true);
		window.setStatusBarColor(Color.parseColor("#12121F"));

		WindowInsetsControllerCompat controller =
				WindowCompat.getInsetsController(window, window.getDecorView());
		if (controller != null) {
			controller.setAppearanceLightStatusBars(false);
			controller.show(WindowInsetsCompat.Type.statusBars());
		}

		// Ensure the content root respects system window insets
		View content = findViewById(android.R.id.content);
		if (content != null) {
			content.setFitsSystemWindows(true);
			content.requestApplyInsets();
		}

		// Inject actual status-bar height as a CSS variable so the WebView can use it
		int sbHeight = 0;
		int resId = getResources().getIdentifier("status_bar_height", "dimen", "android");
		if (resId > 0) sbHeight = getResources().getDimensionPixelSize(resId);
		float density = getResources().getDisplayMetrics().density;
		final int cssPx = Math.round(sbHeight / density);

		getBridge().getWebView().post(() ->
			getBridge().getWebView().evaluateJavascript(
				"document.documentElement.style.setProperty('--native-status-bar','"+cssPx+"px')",
				null
			)
		);

		dispatchNotificationIntent(getIntent());
	}
	@Override
	protected void onNewIntent(Intent intent) {
		super.onNewIntent(intent);
		setIntent(intent);
		dispatchNotificationIntent(intent);
	}

	private void dispatchNotificationIntent(Intent intent) {
		if (intent == null || intent.getExtras() == null || getBridge() == null) return;
		JSONObject payload = new JSONObject();
		for (String key : intent.getExtras().keySet()) {
			Object value = intent.getExtras().get(key);
			try {
				payload.put(key, value == null ? JSONObject.NULL : String.valueOf(value));
			} catch (JSONException ignored) {}
		}
		if (payload.length() == 0) return;
		String json = payload.toString().replace("\\", "\\\\").replace("'", "\\'");
		String script = "window.__UNINO_PENDING_NOTIFICATION={extra:JSON.parse('" + json + "'),actionId:'tap'};window.dispatchEvent(new CustomEvent('unino:native-notification-open',{detail:window.__UNINO_PENDING_NOTIFICATION}));";
		getBridge().getWebView().postDelayed(() -> getBridge().getWebView().evaluateJavascript(script, null), 300);
	}

	private void dispatchNativeFcmToken(String token) {
		if (token == null || token.trim().isEmpty() || getBridge() == null) return;
		String safeToken = token.replace("\\", "\\\\").replace("'", "\\'");
		String script = "window.__UNINO_NATIVE_FCM_TOKEN='" + safeToken + "';window.dispatchEvent(new CustomEvent('unino:native-fcm-token',{detail:{token:'" + safeToken + "'}}));";
		getBridge().getWebView().post(() -> getBridge().getWebView().evaluateJavascript(script, null));
	}

	private void invokeClientSetter(Client client, String methodName, String value) throws Exception {
		for (Method method : Client.class.getMethods()) {
			if (method.getName().equals(methodName)
				&& method.getParameterCount() == 1
				&& method.getParameterTypes()[0] == String.class) {
				method.invoke(client, value);
				return;
			}
		}
		throw new NoSuchMethodException("No matching " + methodName + "(String) on Appwrite Client");
	}
}
