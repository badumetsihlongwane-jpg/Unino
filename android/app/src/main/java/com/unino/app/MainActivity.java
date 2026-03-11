package com.unino.app;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.webkit.PermissionRequest;
import android.webkit.WebChromeClient;
import android.webkit.WebView;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.WindowInsetsControllerCompat;

import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
	private static final int MIC_PERMISSION_CODE = 1001;
	private PermissionRequest pendingPermissionRequest;

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

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

		// Grant microphone/camera permissions to the WebView when requested by getUserMedia
		getBridge().getWebView().setWebChromeClient(new WebChromeClient() {
			@Override
			public void onPermissionRequest(final PermissionRequest request) {
				String[] resources = request.getResources();
				boolean needsMic = false;
				for (String r : resources) {
					if (PermissionRequest.RESOURCE_AUDIO_CAPTURE.equals(r)) {
						needsMic = true;
					}
				}
				if (needsMic) {
					if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.RECORD_AUDIO)
							== PackageManager.PERMISSION_GRANTED) {
						request.grant(resources);
					} else {
						pendingPermissionRequest = request;
						ActivityCompat.requestPermissions(MainActivity.this,
								new String[]{Manifest.permission.RECORD_AUDIO}, MIC_PERMISSION_CODE);
					}
				} else {
					request.grant(resources);
				}
			}

			@Override
			public void onPermissionRequestCanceled(PermissionRequest request) {
				pendingPermissionRequest = null;
			}
		});
	}

	@Override
	public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
		super.onRequestPermissionsResult(requestCode, permissions, grantResults);
		if (requestCode == MIC_PERMISSION_CODE && pendingPermissionRequest != null) {
			if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
				pendingPermissionRequest.grant(pendingPermissionRequest.getResources());
			} else {
				pendingPermissionRequest.deny();
			}
			pendingPermissionRequest = null;
		}
	}
}
