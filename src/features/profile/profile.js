import { AuthService } from '../../services/auth.service.js';
import { state } from '../../config/app-config.js';
import { $, esc, escAttr, toast, avatar, openModal, closeModal } from '../../lib/dom.js';

export const ProfileFeature = {
  mount() {
    this.render();
    this.bindEvents();
  },

  unmount() {},

  render() {
    const profile = state.profile;
    if (!profile) return;

    const nameEl = $('#p-name');
    const emailEl = $('#p-email');
    const majorEl = $('#p-major');
    const avatarEl = $('#p-avatar-wrap');

    if (nameEl) nameEl.textContent = profile.displayName || 'Student';
    if (emailEl) emailEl.textContent = profile.email || '';
    if (majorEl) majorEl.textContent = profile.major || 'NWU Student';
    if (avatarEl) avatarEl.innerHTML = avatar(profile.displayName, profile.photoURL, 'avatar-xl');
  },

  bindEvents() {
    const editBtn = $('#p-edit-btn');
    if (editBtn && !editBtn._bound) {
      editBtn._bound = true;
      editBtn.addEventListener('click', () => this.showEditModal());
    }

    const logoutBtn = $('#p-logout-btn');
    if (logoutBtn && !logoutBtn._bound) {
      logoutBtn._bound = true;
      logoutBtn.addEventListener('click', async () => {
        if (!confirm('Log out of Unibo?')) return;
        await AuthService.logout();
        window.location.reload();
      });
    }
  },

  showEditModal() {
    const profile = state.profile || {};
    openModal(`
      <div class="modal-header">
        <h2>Edit Profile</h2>
        <button class="icon-btn" onclick="closeModal()">&times;</button>
      </div>
      <div class="modal-body" style="padding:16px">
        <label style="font-size:12px;font-weight:600;display:block;margin-bottom:4px">Display Name</label>
        <input type="text" id="edit-p-name" value="${escAttr(profile.displayName || '')}" style="width:100%;margin-bottom:12px">

        <label style="font-size:12px;font-weight:600;display:block;margin-bottom:4px">Degree / Major</label>
        <input type="text" id="edit-p-major" value="${escAttr(profile.major || '')}" style="width:100%;margin-bottom:12px">

        <label style="font-size:12px;font-weight:600;display:block;margin-bottom:4px">Bio</label>
        <textarea id="edit-p-bio" rows="3" style="width:100%;margin-bottom:16px">${esc(profile.bio || '')}</textarea>

        <button class="btn-primary btn-full" id="save-profile-btn">Save Changes</button>
      </div>
    `);

    $('#save-profile-btn')?.addEventListener('click', async () => {
      const displayName = $('#edit-p-name')?.value.trim();
      const major = $('#edit-p-major')?.value.trim();
      const bio = $('#edit-p-bio')?.value.trim();

      if (!state.user) return;
      try {
        await AuthService.updateProfile(state.user.$id, { displayName, major, bio });
        toast('Profile updated');
        closeModal();
        this.render();
      } catch (e) {
        toast('Failed to save profile');
      }
    });
  }
};
