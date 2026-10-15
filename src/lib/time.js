export function timeAgo(input) {
  if (!input) return '';
  let date = input;
  if (typeof input?.toDate === 'function') date = input.toDate();
  else if (typeof input === 'string' || typeof input === 'number') date = new Date(input);
  if (!(date instanceof Date) || isNaN(date)) return '';

  const diffMs = Date.now() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return 'just now';
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHours = Math.floor(diffMin / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) return `${diffDays}d ago`;
  const diffWeeks = Math.floor(diffDays / 7);
  if (diffWeeks < 4) return `${diffWeeks}w ago`;
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export function formatTime(input) {
  if (!input) return '';
  let date = input;
  if (typeof input?.toDate === 'function') date = input.toDate();
  else if (typeof input === 'string' || typeof input === 'number') date = new Date(input);
  if (!(date instanceof Date) || isNaN(date)) return '';

  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
