import { useState } from 'react'
import ErpLayout from '../components/ErpLayout'
import { Badge, Empty, Field, Load, Meter, Panel, Stats, Table, Tabs } from '../components/kit'
import { TARGET, standing } from '../lib/attendance'
import { formatDate } from '../lib/format'
import useApi from '../lib/useApi'
import {
  errorMessage, fetchHeatmap, fetchMissedClasses, fetchOverallAttendance, fetchSubjectAttendance,
  simulateBunk,
} from '../services/api'

const HEAT_LABEL = { full: 'Present all day', partial: 'Missed some classes', absent: 'Absent all day' }

function Planner({ subjects }) {
  const [subjectId, setSubjectId] = useState(subjects[0]?.subject_id ?? '')
  const [count, setCount] = useState(2)
  const [result, setResult] = useState(null)
  const [problem, setProblem] = useState(null)

  async function check(event) {
    event.preventDefault()
    setProblem(null)
    try {
      setResult(await simulateBunk(Number(count), Number(subjectId)))
    } catch (error) {
      setResult(null)
      setProblem(errorMessage(error))
    }
  }

  const edit = (setter) => (event) => {
    setter(event.target.value)
    setResult(null)
  }

  return (
    <form className="form" onSubmit={check}>
      <div className="form-grid">
        <Field label="Subject">
          <select value={subjectId} onChange={edit(setSubjectId)}>
            {subjects.map((s) => <option key={s.subject_id} value={s.subject_id}>{s.subject}</option>)}
          </select>
        </Field>
        <Field label="Classes you would miss">
          <input type="number" min="1" max="30" value={count} onChange={edit(setCount)} required />
        </Field>
      </div>
      <div className="form-actions"><button type="submit" className="btn btn-primary">Check</button></div>
      {problem && <p className="notice notice-bad" role="alert">{problem}</p>}
      {result && (
        <p className={`notice ${result.simulated_pct >= TARGET ? 'notice-good' : 'notice-bad'}`} role="status">
          Your attendance would go from {result.current_pct}% to {result.simulated_pct}% ({result.simulated_present} of {result.simulated_total} classes).
        </p>
      )}
    </form>
  )
}

export default function AttendancePage() {
  const [view, setView] = useState('subjects')
  const overall = useApi(fetchOverallAttendance)
  const subjects = useApi(fetchSubjectAttendance)
  const missed = useApi(fetchMissedClasses)
  const heatmap = useApi(fetchHeatmap)

  return (
    <ErpLayout title="Attendance" subtitle={`You need ${TARGET}% in each subject.`}>
      <Load state={overall}>
        {({ total, present, absent, percentage }) => (
          <>
            <Stats items={[
              { label: 'Attendance', value: total ? `${percentage}%` : '—', tone: total ? (percentage >= TARGET ? 'good' : 'bad') : undefined },
              { label: 'Classes held', value: total },
              { label: 'Attended', value: present },
              { label: 'Missed', value: absent },
            ]} />
            {total > 0 && <p className="meta">{standing(present, total)}</p>}
          </>
        )}
      </Load>

      <Tabs label="Attendance view" value={view} onChange={setView} tabs={[['subjects', 'By subject'], ['days', 'By day'], ['missed', 'Missed classes']]} />

      {view === 'subjects' && (
        <Load state={subjects}>
          {({ subjects: list }) => (
            <>
              <Panel flush title="Subjects, lowest first">
                <Table
                  rows={list}
                  rowKey={(s) => s.subject_id}
                  empty={<Empty title="No attendance recorded yet" />}
                  columns={[
                    { label: 'Subject', cell: (s) => <>{s.subject}<span className="cell-sub">{s.code}</span></> },
                    { label: 'Progress', cell: (s) => <div style={{ minWidth: 130 }}><Meter percent={s.percentage} target={TARGET} /></div> },
                    { label: 'Attended', numeric: true, cell: (s) => `${s.present} / ${s.total}` },
                    { label: 'Percent', numeric: true, cell: (s) => <Badge tone={s.below_75 ? 'bad' : 'good'}>{s.percentage}%</Badge> },
                  ]}
                />
              </Panel>
              {list.length > 0 && <Panel title="What if I miss a few classes?"><Planner subjects={list} /></Panel>}
            </>
          )}
        </Load>
      )}

      {view === 'days' && (
        <Panel title="Each recorded day">
          <Load state={heatmap}>
            {({ heatmap: days }) => (days.length ? (
              <>
                <div className="heat">
                  {days.map((d) => {
                    const text = `${formatDate(d.date)}: ${d.present} of ${d.total} classes`
                    return <span key={d.date} className={d.status} title={text} role="img" aria-label={text} />
                  })}
                </div>
                <div className="legend">
                  {Object.entries(HEAT_LABEL).map(([key, text]) => <span key={key}><i className={`key ${key}`} />{text}</span>)}
                </div>
              </>
            ) : <Empty title="No days recorded yet" />)}
          </Load>
        </Panel>
      )}

      {view === 'missed' && (
        <Panel flush title="Recently missed (up to 50)">
          <Load state={missed}>
            {({ missed: list }) => (
              <Table
                rows={list}
                empty={<Empty title="You haven't missed any classes" />}
                columns={[
                  { label: 'Date', cell: (m) => formatDate(m.date, { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' }) },
                  { label: 'Hour', numeric: true, cell: (m) => m.hour },
                  { label: 'Subject', cell: (m) => m.subject },
                ]}
              />
            )}
          </Load>
        </Panel>
      )}
    </ErpLayout>
  )
}
