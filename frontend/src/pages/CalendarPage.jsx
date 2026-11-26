import { useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import ErpLayout from '../components/ErpLayout'
import { Empty, Load, Panel } from '../components/kit'
import { formatDate, parseDate, toISODate } from '../lib/format'
import useApi from '../lib/useApi'
import { fetchCalendar } from '../services/api'

const DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

function Month({ days }) {
  const today = toISODate()
  const [cursor, setCursor] = useState(() => new Date(new Date().getFullYear(), new Date().getMonth(), 1))

  const byDate = new Map(days.map((d) => [d.date, d]))
  const year = cursor.getFullYear()
  const month = cursor.getMonth()
  const blanks = (new Date(year, month, 1).getDay() + 6) % 7
  const length = new Date(year, month + 1, 0).getDate()
  const dates = Array.from({ length }, (_, i) => toISODate(new Date(year, month, i + 1)))
  const noted = dates.map((d) => byDate.get(d)).filter((d) => d && (d.holiday || d.note))
  const move = (step) => setCursor(new Date(year, month + step, 1))

  return (
    <>
      <Panel
        flush
        title="Month view"
        action={(
          <div className="cal-nav">
            <button type="button" className="icon-btn" onClick={() => move(-1)} aria-label="Previous month"><ChevronLeft size={18} /></button>
            <h2 aria-live="polite">{cursor.toLocaleDateString('en-IN', { month: 'long', year: 'numeric' })}</h2>
            <button type="button" className="icon-btn" onClick={() => move(1)} aria-label="Next month"><ChevronRight size={18} /></button>
          </div>
        )}
      >
        <div className="month">
          {DOW.map((d) => <div key={d} className="dow">{d}</div>)}
          {Array.from({ length: blanks }, (_, i) => <div key={`blank-${i}`} />)}
          {dates.map((date, i) => {
            const info = byDate.get(date)
            return (
              <div key={date} className={`day ${info && !info.working ? 'day-off' : ''} ${date === today ? 'day-today' : ''}`}>
                <b>{i + 1}</b>
                {info?.holiday && <small className="holiday">{typeof info.holiday === 'string' ? info.holiday : 'Holiday'}</small>}
                {info?.note && <small>{info.note}</small>}
              </div>
            )
          })}
        </div>
      </Panel>

      <p className="meta">
        Shaded days are non-working. {dates.filter((d) => byDate.get(d)?.working).length} working days this month.
        {!dates.some((d) => byDate.has(d)) && ' No calendar has been published for this month.'}
      </p>

      <Panel title="Holidays and notes this month">
        {noted.length ? (
          <ul className="list">
            {noted.map((d) => (
              <li key={d.date}>
                <div className="list-main">
                  <strong>{typeof d.holiday === 'string' ? d.holiday : d.note || 'Holiday'}</strong>
                  <span>{formatDate(d.date, { weekday: 'long', day: 'numeric', month: 'long' })}{typeof d.holiday === 'string' && d.note ? ` · ${d.note}` : ''}</span>
                </div>
              </li>
            ))}
          </ul>
        ) : <Empty title="Nothing marked for this month" />}
      </Panel>
    </>
  )
}

export default function CalendarPage() {
  const state = useApi(fetchCalendar)
  return (
    <ErpLayout title="Calendar" subtitle="Working days and holidays for the academic year.">
      <Load state={state}>
        {({ calendar }) => <Month days={calendar.filter((d) => parseDate(d.date))} />}
      </Load>
    </ErpLayout>
  )
}
