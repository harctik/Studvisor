export const WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

/** Parse "2026-05-25" or "2026-05-25 18:24:46" as a local calendar date. */
export function parseDate(value) {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(value || '')
  return match ? new Date(+match[1], +match[2] - 1, +match[3]) : null
}

export function formatDate(value, options = { day: 'numeric', month: 'short', year: 'numeric' }) {
  const date = parseDate(value)
  return date ? date.toLocaleDateString('en-IN', options) : value || '—'
}

export function toISODate(date = new Date()) {
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${date.getFullYear()}-${month}-${day}`
}

/** Whole days from today; negative once the date has passed. */
export function daysUntil(value) {
  const date = parseDate(value)
  if (!date) return null
  const today = parseDate(toISODate())
  return Math.round((date - today) / 86400000)
}

export const money = (amount) => `₹${Number(amount || 0).toLocaleString('en-IN')}`

export const plural = (count, word, many = `${word}s`) => `${count} ${count === 1 ? word : many}`

/** Only let http(s) links through; anything else (such as javascript:) is dropped. */
export function safeUrl(value) {
  try {
    const url = new URL(value, window.location.origin)
    return url.protocol === 'http:' || url.protocol === 'https:' ? url.href : null
  } catch {
    return null
  }
}
