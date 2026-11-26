import ErpLayout from '../components/ErpLayout'
import { Empty, Load, Panel, Table } from '../components/kit'
import useApi from '../lib/useApi'
import { fetchSyllabus } from '../services/api'

export default function SyllabusPage() {
  const state = useApi(fetchSyllabus)
  return (
    <ErpLayout title="Syllabus" subtitle="Units and topics for each subject.">
      <Load state={state}>
        {({ syllabus }) => (syllabus.length ? syllabus.map((s) => (
          <Panel flush key={s.code} title={`${s.subject} (${s.code})`}>
            <Table
              rows={s.topics}
              empty={<Empty title="No topics added for this subject yet" />}
              columns={[
                { label: 'Unit', cell: (t) => t.unit },
                { label: 'Topic', cell: (t) => t.name },
                { label: 'Hours', numeric: true, cell: (t) => t.hours },
              ]}
            />
          </Panel>
        )) : <Panel><Empty title="No syllabus published yet" /></Panel>)}
      </Load>
    </ErpLayout>
  )
}
