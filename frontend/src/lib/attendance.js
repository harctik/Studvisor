import { plural } from './format'

export const TARGET = 75

/** What the 75% rule means for this record, using whole classes only. */
export function standing(present, total) {
  if (!total) return 'No classes have been recorded yet.'
  if (present * 4 >= total * 3) {
    const spare = Math.floor((present * 4) / 3) - total
    return spare > 0
      ? `You can miss ${plural(spare, 'more class', 'more classes')} and stay at ${TARGET}%.`
      : `Missing the next class would take you below ${TARGET}%.`
  }
  const needed = total * 3 - present * 4
  return `Attend the next ${plural(needed, 'class', 'classes')} in a row to get back to ${TARGET}%.`
}
