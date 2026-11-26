import ErpLayout from '../components/ErpLayout'
import { Empty, Load, Panel, Table } from '../components/kit'
import useApi from '../lib/useApi'
import { fetchMarks } from '../services/api'

export default function ResultsPage() {
  const state = useApi(fetchMarks)
  return (
    <ErpLayout title="Marks" subtitle="Published assessment marks.">
      <Load state={state}>
        {({ marks }) => {
          if (!marks.length) {
            return <Panel><Empty title="No marks published yet">Marks appear here once your lecturers publish them.</Empty></Panel>
          }
          const semesters = [...new Set(marks.map((m) => m.semester))].sort((a, b) => b - a)
          return semesters.map((semester) => (
            <Panel flush key={semester} title={`Semester ${semester}`}>
              <Table
                rows={marks.filter((m) => m.semester === semester)}
                rowKey={(m) => m.id}
                columns={[
                  { label: 'Subject', cell: (m) => <>{m.subject}<span className="cell-sub">{m.code}</span></> },
                  { label: 'Assessment', cell: (m) => m.assessment },
                  { label: 'Marks', numeric: true, cell: (m) => `${m.obtained} / ${m.max}` },
                  { label: 'Percent', numeric: true, cell: (m) => `${m.percentage}%` },
                ]}
              />
            </Panel>
          ))
        }}
      </Load>
    </ErpLayout>
  )
}
