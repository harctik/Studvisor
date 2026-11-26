import { useState } from 'react'
import ErpLayout from '../components/ErpLayout'
import { Badge, Empty, Load, Tabs, useAction } from '../components/kit'
import { daysUntil, formatDate, plural } from '../lib/format'
import useApi from '../lib/useApi'
import { fetchAssignments, submitAssignment } from '../services/api'

function due(assignment) {
  if (assignment.submitted) return <Badge tone="good">Submitted</Badge>
  const days = daysUntil(assignment.due_date)
  if (days === null) return <Badge>No due date</Badge>
  if (days < 0) return <Badge tone="bad">Overdue by {plural(-days, 'day')}</Badge>
  if (days === 0) return <Badge tone="warn">Due today</Badge>
  return <Badge tone={days <= 3 ? 'warn' : 'plain'}>Due in {plural(days, 'day')}</Badge>
}

export default function AssignmentsPage() {
  const state = useApi(fetchAssignments)
  const { busy, run, banner } = useAction(state.reload)
  const [view, setView] = useState('open')

  // This records the submission only; the server has no file upload.
  function markSubmitted(assignment) {
    const sure = window.confirm(`Mark "${assignment.title}" as submitted? No file is uploaded from here, and this can't be undone.`)
    if (sure) run(assignment.id, () => submitAssignment(assignment.id), 'Marked as submitted.')
  }

  return (
    <ErpLayout title="Assignments" subtitle="Hand your work to your lecturer, then record it here.">
      <Tabs label="Filter assignments" value={view} onChange={setView} tabs={[['open', 'To submit'], ['done', 'Submitted']]} />
      {banner}
      <Load state={state}>
        {({ assignments }) => {
          const list = assignments
            .filter((a) => Boolean(a.submitted) === (view === 'done'))
            .sort((a, b) => String(a.due_date).localeCompare(String(b.due_date)))
          if (!list.length) return <Empty title={view === 'done' ? 'Nothing submitted yet' : 'Nothing waiting to be submitted'} />
          return (
            <div className="cards">
              {list.map((a) => (
                <article key={a.id} className="card">
                  <div className="card-top"><h2>{a.title}</h2>{due(a)}</div>
                  {a.description && <p className="pre">{a.description}</p>}
                  <div className="card-foot">
                    <span className="meta">Due {formatDate(a.due_date)}{a.max_marks ? ` · ${a.max_marks} marks` : ''}</span>
                    {!a.submitted && (
                      <button type="button" className="btn btn-small" disabled={busy === a.id} onClick={() => markSubmitted(a)}>Mark as submitted</button>
                    )}
                  </div>
                </article>
              ))}
            </div>
          )
        }}
      </Load>
    </ErpLayout>
  )
}
