import { useState } from 'react'
import ErpLayout from '../components/ErpLayout'
import { Badge, Empty, Field, Load, Panel, Stats, Table } from '../components/kit'
import useApi from '../lib/useApi'
import { errorMessage, fetchCGPA, predictCgpa } from '../services/api'

const SCALE = 'O is 90% and above (10 points), then A+ 80, A 70, B+ 60, B 50 and C 40. Below 40% is F.'

function GradeLookup() {
  const [target, setTarget] = useState(75)
  const [result, setResult] = useState(null)
  const [problem, setProblem] = useState(null)

  async function check(event) {
    event.preventDefault()
    setProblem(null)
    try {
      setResult(await predictCgpa(Number(target)))
    } catch (error) {
      setResult(null)
      setProblem(errorMessage(error))
    }
  }

  return (
    <form className="form" onSubmit={check}>
      <div className="search">
        <Field label="Percentage">
          <input type="number" min="0" max="100" value={target} onChange={(e) => { setTarget(e.target.value); setResult(null) }} required />
        </Field>
        <button type="submit" className="btn btn-primary" style={{ alignSelf: 'flex-end', height: 45 }}>Show grade</button>
      </div>
      {problem && <p className="notice notice-bad" role="alert">{problem}</p>}
      {result && (
        <p className="notice notice-good" role="status">
          {result.target_percentage}% earns grade {result.projected_grade}, worth {result.projected_grade_point} points.
        </p>
      )}
      <p className="meta">{SCALE}</p>
    </form>
  )
}

export default function GPAPage() {
  const state = useApi(fetchCGPA)
  return (
    <ErpLayout title="GPA" subtitle="Semester and cumulative grade points.">
      <Load state={state}>
        {({ cgpa, total_credits: credits, semesters }) => (semesters.length ? (
          <>
            <Stats items={[
              { label: 'CGPA', value: cgpa },
              { label: 'Credits counted', value: credits },
              { label: 'Semesters', value: semesters.length },
            ]} />
            {semesters.map((s) => (
              <Panel flush key={s.semester} title={`Semester ${s.semester}`} action={<Badge tone="lime">SGPA {s.sgpa}</Badge>}>
                <Table
                  rows={s.subjects}
                  rowKey={(row) => row.name}
                  columns={[
                    { label: 'Subject', cell: (row) => row.name },
                    { label: 'Credits', numeric: true, cell: (row) => row.credits },
                    { label: 'Percent', numeric: true, cell: (row) => `${row.percentage}%` },
                    { label: 'Grade', numeric: true, cell: (row) => row.grade },
                    { label: 'Points', numeric: true, cell: (row) => row.points },
                  ]}
                />
              </Panel>
            ))}
          </>
        ) : <Panel><Empty title="No GPA yet">It is calculated once marks are published.</Empty></Panel>)}
      </Load>
      <Panel title="What grade is a percentage worth?"><GradeLookup /></Panel>
    </ErpLayout>
  )
}
