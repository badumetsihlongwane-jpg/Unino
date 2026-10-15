export const APP_VERSION = 52;
export const APP_NAME = 'Unibo';
export const ADMIN_EMAIL = 'admin@mynwu.ac.za';

export const COLORS = [
  '#6C5CE7', '#8B5CF6', '#A855F7', '#7C3AED', '#6366F1',
  '#818CF8', '#C084FC', '#D946EF', '#E879F9', '#A78BFA'
];

export const CAMPUS_COORDS = {
  potch: { lat: -26.6995, lng: 27.0945, label: 'Potchefstroom' },
  vaaldri: { lat: -26.7153, lng: 27.8767, label: 'Vanderbijlpark' },
  mafikeng: { lat: -25.8242, lng: 25.6136, label: 'Mahikeng' }
};

export const state = {
  user: null,
  profile: null,
  page: 'feed',
  status: 'online',
  manualStatus: 'online',
  unsubs: [],
  lastMsgTab: 'dm'
};
