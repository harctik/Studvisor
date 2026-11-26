import { Link } from 'react-router-dom'
import ErpLayout from '../components/ErpLayout'
import { Badge, Empty, Load, Meter, Panel } from '../components/kit'
import { TARGET, standing } from '../lib/attendance'
import { formatDate, money, toISODate, daysUntil } from '../lib/format'
import useApi from '../lib/useApi'
import {
  fetchAnnouncements, fetchAssignments, fetchBunkAlerts, fetchCGPA, fetchExams, fetchFeeSummary,
  fetchOverallAttendance, fetchTimetable,
} from '../services/api'

function SideLink({ to, label, state, value }) {
  return (
    <Link to={to}>
      <span>{label}</span>
      <strong>{state.data ? value(state.data) : '—'}</strong>
    </Link>
  )
}

export default function DashboardPage() {
  const attendance = useApi(fetchOverallAttendance)
  const alerts = useApi(fetchBunkAlerts)
  const timetable = useApi(fetchTimetable)
  const cgpa = useApi(fetchCGPA)
  const fees = useApi(fetchFeeSummary)
  const assignments = useApi(fetchAssignments)
  const exams = useApi(fetchExams)
  const announcements = useApi(fetchAnnouncements)

  const now = new Date()
  const weekday = now.toLocaleDateString('en-US', { weekday: 'long' })

  return (
    <ErpLayout title="Dashboard" subtitle={formatDate(toISODate(now), { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })}>
      <Load state={attendance}>
        {({ present, total, percentage }) => (
          <section className="hero">
            <div>
              <p className="hero-label">Overall attendance</p>
              <p className="hero-figure">{total ? `${percentage}%` : '—'}</p>
              {total > 0 && <Meter percent={percentage} target={TARGET} />}
              <p className="hero-note">
                {total > 0 && `${present} of ${total} classes. `}
                {standing(present, total)}
              </p>
            </div>
            <div className="hero-side">
              <SideLink to="/gpa" label="CGPA" state={cgpa} value={(d) => (d.semesters.length ? d.cgpa : '—')} />
              <SideLink to="/assignments" label="Assignments to submit" state={assignments} value={(d) => d.assignments.filter((a) => !a.submitted).length} />
              <SideLink to="/fees" label="Fees pending" state={fees} value={(d) => money(d.pending)} />
            </div>
          </section>
        )}
      </Load>

      <div className="split">
        <Panel title={`${weekday}'s classes`} action={<Link to="/timetable">Full timetable</Link>}>
          <Load state={timetable}>
            {({ timetable: slots }) => {
              const today = slots.filter((s) => s.day === weekday).sort((a, b) => a.hour - b.hour)
              if (!today.length) return <Empty title="No classes today" />
              return (
                <ul className="list">
                  {today.map((slot) => (
                    <li key={slot.hour}>
                      <div className="list-main">
                        <strong>{slot.subject}</strong>
                        <span>{[slot.faculty, slot.room].filter(Boolean).join(' · ')}</span>
                      </div>
                      <Badge>Hour {slot.hour}</Badge>
                    </li>
                  ))}
                </ul>
              )
            }}
          </Load>
        </Panel>

        <Panel title="Subjects to watch" action={<Link to="/attendance">All subjects</Link>}>
          <Load state={alerts}>
            {({ alerts: list }) => (list.length ? (
              <ul className="list">
                {list.map((alert) => (
                  <li key={alert.subject}>
                    <div className="list-main"><strong>{alert.subject}</strong></div>
                    <Badge tone={alert.level === 'critical' ? 'bad' : 'warn'}>{alert.percentage}%</Badge>
                  </li>
                ))}
              </ul>
            ) : <Empty title="No attendance warnings" />)}
          </Load>
        </Panel>

        <Panel title="Coming exams" action={<Link to="/exams">All exams</Link>}>
          <Load state={exams}>
            {({ exams: list }) => {
              const coming = list.filter((e) => daysUntil(e.date) >= 0).sort((a, b) => a.date.localeCompare(b.date)).slice(0, 4)
              if (!coming.length) return <Empty title="No exams scheduled" />
              return (
                <ul className="list">
                  {coming.map((exam) => (
                    <li key={exam.id}>
                      <div className="list-main">
                        <strong>{exam.subject}</strong>
                        <span>{[exam.type, exam.time, exam.venue].filter(Boolean).join(' · ')}</span>
                      </div>
                      <Badge>{formatDate(exam.date, { day: 'numeric', month: 'short' })}</Badge>
                    </li>
                  ))}
                </ul>
              )
            }}
          </Load>
        </Panel>

        <Panel title="Announcements" action={<Link to="/announcements">All announcements</Link>}>
          <Load state={announcements}>
            {({ announcements: list }) => (list.length ? (
              <ul className="list">
                {[...list].sort((a, b) => b.date.localeCompare(a.date)).slice(0, 4).map((item) => (
                  <li key={item.id}>
                    <div className="list-main">
                      <strong>{item.title}</strong>
                      <span>{formatDate(item.date)}</span>
                    </div>
                  </li>
                ))}
              </ul>
            ) : <Empty title="No announcements" />)}
          </Load>
        </Panel>
      </div>
    </ErpLayout>
  )
}
