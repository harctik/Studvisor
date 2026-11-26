import ErpLayout from '../components/ErpLayout'
import { Empty, Load, Meter, Panel } from '../components/kit'
import { TARGET } from '../lib/attendance'
import useApi from '../lib/useApi'
import { fetchCGPA, fetchMarks, fetchSubjectAttendance } from '../services/api'

function Bars({ rows }) {
  return (
    <div className="bars">
      {rows.map((row) => (
        <div key={row.key} className="bar-row">
          <div><span>{row.label}</span><strong>{row.value}</strong></div>
          <Meter percent={row.percent} target={row.target} />
        </div>
      ))}
    </div>
  )
}

/** Weighted average percentage for each subject across its assessments. */
function bySubject(marks) {
  const totals = new Map()
  marks.forEach((m) => {
    const sum = totals.get(m.code) || { subject: m.subject, obtained: 0, max: 0 }
    totals.set(m.code, { ...sum, obtained: sum.obtained + m.obtained, max: sum.max + m.max })
  })
  return [...totals].map(([code, s]) => {
    const percent = s.max ? Math.round((s.obtained / s.max) * 1000) / 10 : 0
    return { key: code, label: s.subject, percent, value: `${percent}%` }
  }).sort((a, b) => a.percent - b.percent)
}

export default function AnalyticsPage() {
  const attendance = useApi(fetchSubjectAttendance)
  const marks = useApi(fetchMarks)
  const gpa = useApi(fetchCGPA)

  return (
    <ErpLayout title="Analytics" subtitle="Your attendance, marks and grade points side by side.">
      <div className="split">
        <Panel title="Attendance by subject">
          <Load state={attendance}>
            {({ subjects }) => (subjects.length
              ? <Bars rows={subjects.map((s) => ({ key: s.subject_id, label: s.subject, percent: s.percentage, value: `${s.percentage}%`, target: TARGET }))} />
              : <Empty title="No attendance recorded yet" />)}
          </Load>
        </Panel>
        <Panel title="Marks by subject, lowest first">
          <Load state={marks}>
            {({ marks: list }) => (list.length ? <Bars rows={bySubject(list)} /> : <Empty title="No marks published yet" />)}
          </Load>
        </Panel>
        <Panel title="SGPA by semester">
          <Load state={gpa}>
            {({ semesters }) => (semesters.length
              ? <Bars rows={[...semesters].sort((a, b) => a.semester - b.semester).map((s) => ({ key: s.semester, label: `Semester ${s.semester}`, percent: s.sgpa * 10, value: s.sgpa }))} />
              : <Empty title="No grade points yet" />)}
          </Load>
        </Panel>
      </div>
    </ErpLayout>
  )
}
