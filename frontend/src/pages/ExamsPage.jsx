import ErpLayout from '../components/ErpLayout'
import { Empty, Load, Panel, Table } from '../components/kit'
import { daysUntil, formatDate, plural } from '../lib/format'
import useApi from '../lib/useApi'
import { fetchExams } from '../services/api'

const columns = (upcoming) => [
  { label: 'Date', cell: (e) => <>{formatDate(e.date)}{upcoming && <span className="cell-sub">{daysUntil(e.date) === 0 ? 'Today' : `In ${plural(daysUntil(e.date), 'day')}`}</span>}</> },
  { label: 'Subject', cell: (e) => e.subject },
  { label: 'Type', cell: (e) => e.type },
  { label: 'Time', cell: (e) => e.time || '—' },
  { label: 'Venue', cell: (e) => e.venue || '—' },
]

export default function ExamsPage() {
  const state = useApi(fetchExams)
  return (
    <ErpLayout title="Exams" subtitle="Dates, times and halls.">
      <Load state={state}>
        {({ exams }) => {
          const sorted = [...exams].sort((a, b) => a.date.localeCompare(b.date))
          const upcoming = sorted.filter((e) => daysUntil(e.date) >= 0)
          const past = sorted.filter((e) => daysUntil(e.date) < 0).reverse()
          return (
            <>
              <Panel flush title="Coming up">
                <Table rows={upcoming} rowKey={(e) => e.id} columns={columns(true)} empty={<Empty title="No exams scheduled" />} />
              </Panel>
              {past.length > 0 && (
                <Panel flush title="Past exams">
                  <Table rows={past} rowKey={(e) => e.id} columns={columns(false)} />
                </Panel>
              )}
            </>
          )
        }}
      </Load>
    </ErpLayout>
  )
}
