import '../css/styles.css';
import { AuthService } from './services/auth.service.js';
import { FeedFeature } from './features/feed/feed.js';
import { ChatFeature } from './features/chat/chat.js';
import { MarketplaceFeature } from './features/marketplace/marketplace.js';
import { ProfileFeature } from './features/profile/profile.js';
import { state } from './config/app-config.js';
import { $, $$, toast, closeModal, openModal } from './lib/dom.js';

// Expose safe global bridges for any remaining dynamic template actions
window.closeModal = closeModal;
window.openModal = openModal;
window.toast = toast;

let _activeFeature = null;

const ROUTES = {
  feed: FeedFeature,
  chat: ChatFeature,
  messages: ChatFeature,
  marketplace: MarketplaceFeature,
  hustle: MarketplaceFeature,
  profile: ProfileFeature
};

export async function navigateTo(pageName) {
  if (state.page === pageName && _activeFeature) return;

  // Cleanup old active feature to prevent memory leaks and listener accumulation
  if (_activeFeature && typeof _activeFeature.unmount === 'function') {
    _activeFeature.unmount();
  }

  state.page = pageName;

  // Update DOM navigation state
  $$('.nav-item').forEach(el => {
    el.classList.toggle('active', el.dataset.page === pageName);
  });

  $$('.page-section').forEach(el => {
    el.classList.toggle('active', el.id === `page-${pageName}`);
  });

  // Mount new feature
  const nextFeature = ROUTES[pageName];
  if (nextFeature && typeof nextFeature.mount === 'function') {
    _activeFeature = nextFeature;
    await nextFeature.mount();
  }
}

function initNavigation() {
  $$('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
      const page = btn.dataset.page;
      if (page) navigateTo(page);
    });
  });
}

function initAuthForms() {
  const loginForm = $('#login-form');
  const signupForm = $('#signup-form');
  const toSignup = $('#to-signup');
  const toLogin = $('#to-login');

  if (toSignup) {
    toSignup.addEventListener('click', e => {
      e.preventDefault();
      loginForm?.classList.remove('active');
      signupForm?.classList.add('active');
    });
  }

  if (toLogin) {
    toLogin.addEventListener('click', e => {
      e.preventDefault();
      signupForm?.classList.remove('active');
      loginForm?.classList.add('active');
    });
  }

  if (loginForm) {
    loginForm.addEventListener('submit', async e => {
      e.preventDefault();
      const email = $('#l-email')?.value.trim();
      const pass = $('#l-pass')?.value;
      const btn = $('#l-btn');
      if (btn) btn.disabled = true;

      try {
        await AuthService.login(email, pass);
        toast('Welcome back to Unibo!');
        bootstrapApp();
      } catch (err) {
        toast(err.message || 'Login failed');
      } finally {
        if (btn) btn.disabled = false;
      }
    });
  }

  if (signupForm) {
    signupForm.addEventListener('submit', async e => {
      e.preventDefault();
      const name = $('#s-name')?.value.trim() || 'Student';
      const email = $('#s-email')?.value.trim();
      const pass = $('#s-pass')?.value;
      const btn = $('#s-btn');
      if (btn) btn.disabled = true;

      try {
        await AuthService.signup(email, pass, name);
        toast('Account created! Welcome to Unibo.');
        bootstrapApp();
      } catch (err) {
        toast(err.message || 'Sign up failed');
      } finally {
        if (btn) btn.disabled = false;
      }
    });
  }
}

async function bootstrapApp() {
  const splash = $('#splash');
  const authScreen = $('#auth-screen');
  const appShell = $('#app-shell');

  try {
    const session = await AuthService.getCurrentSession();
    if (splash) splash.classList.remove('active');

    if (session) {
      if (authScreen) authScreen.classList.remove('active');
      if (appShell) appShell.classList.add('active');
      initNavigation();
      await navigateTo('feed');
    } else {
      if (appShell) appShell.classList.remove('active');
      if (authScreen) authScreen.classList.add('active');
      initAuthForms();
    }
  } catch (e) {
    console.error('App bootstrap failed:', e);
    if (splash) splash.classList.remove('active');
    if (authScreen) authScreen.classList.add('active');
    initAuthForms();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  bootstrapApp();
});
