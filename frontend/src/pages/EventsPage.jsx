import ErpLayout from '../components/ErpLayout'
import { Empty, Load, Panel, useAction } from '../components/kit'
import { daysUntil, formatDate } from '../lib/format'
import useApi from '../lib/useApi'
import { fetchEvents, rsvpEvent } from '../services/api'

function EventList({ events, children }) {
  return (
    <ul className="list">
      {events.map((e) => (
        <li key={e.id} className="stack">
          <div className="date-tile">
            <strong>{formatDate(e.date, { day: 'numeric' })}</strong>
            <span>{formatDate(e.date, { month: 'short' })}</span>
          </div>
          <div className="list-main">
            <strong>{e.title}</strong>
            {e.description && <span>{e.description}</span>}
            <span>{[formatDate(e.date), e.venue].filter(Boolean).join(' · ')}</span>
          </div>
          {children?.(e)}
        </li>
      ))}
    </ul>
  )
}

export default function EventsPage() {
  const state = useApi(fetchEvents)
  const { busy, run, banner } = useAction(state.reload)

  return (
    <ErpLayout title="Events" subtitle="What's on around campus.">
      {banner}
      <Load state={state}>
        {({ events }) => {
          const sorted = [...events].sort((a, b) => a.date.localeCompare(b.date))
          const upcoming = sorted.filter((e) => daysUntil(e.date) >= 0)
          const past = sorted.filter((e) => daysUntil(e.date) < 0).reverse()
          return (
            <>
              <Panel title="Coming up">
                {upcoming.length ? (
                  <EventList events={upcoming}>
                    {(e) => (
                      <button type="button" className="btn btn-small" disabled={busy === e.id} onClick={() => run(e.id, () => rsvpEvent(e.id))}>RSVP</button>
                    )}
                  </EventList>
                ) : <Empty title="No upcoming events" />}
              </Panel>
              {past.length > 0 && <Panel title="Past events"><EventList events={past} /></Panel>}
            </>
          )
        }}
      </Load>
    </ErpLayout>
  )
}
