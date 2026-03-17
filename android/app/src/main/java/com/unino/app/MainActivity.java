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

import com.getcapacitor.BridgeActivity;
import io.appwrite.Client;

public class MainActivity extends BridgeActivity {
	private static final String TAG = "MainActivity";

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		Client appwriteClient = new Client(this)
			.setEndpoint("https://syd.cloud.appwrite.io/v1")
			.setProject("69b4202c00370d4748d6");
		new Thread(() -> {
			try {
				appwriteClient.ping();
				Log.i(TAG, "Appwrite ping success");
			} catch (Exception e) {
				Log.w(TAG, "Appwrite ping failed", e);
			}
		}).start();

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
	}
}
