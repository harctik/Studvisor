import ErpLayout from '../components/ErpLayout'
import { Badge, Empty, Load, Panel, Table, useAction } from '../components/kit'
import { formatDate } from '../lib/format'
import useApi from '../lib/useApi'
import { applyToDrive, fetchApplications, fetchDrives } from '../services/api'

export default function PlacementPage() {
  const drives = useApi(fetchDrives)
  const applications = useApi(fetchApplications)
  const { busy, run, banner } = useAction(() => Promise.all([drives.reload(), applications.reload()]))
  // Applications come back without a drive id, so match on company and role.
  const applied = new Set((applications.data?.applications || []).map((a) => `${a.company}|${a.role}`))

  return (
    <ErpLayout title="Placements" subtitle="Campus recruitment drives and your applications.">
      {banner}
      <Load state={drives}>
        {({ drives: list }) => (list.length ? (
          <div className="cards">
            {list.map((d) => {
              const done = applied.has(`${d.company}|${d.role}`)
              return (
                <article key={d.id} className="card">
                  <div className="card-top"><h2>{d.company}</h2>{d.package ? <Badge tone="lime">{d.package} LPA</Badge> : null}</div>
                  <p>{d.role}</p>
                  <p className="meta">{formatDate(d.date)}{d.min_cgpa ? ` · Minimum CGPA ${d.min_cgpa}` : ''}</p>
                  <div className="card-foot">
                    <button type="button" className="btn btn-small" disabled={done || busy === d.id}
                      onClick={() => window.confirm(`Apply to ${d.company} for ${d.role}?`) && run(d.id, () => applyToDrive(d.id))}>
                      {done ? 'Applied' : 'Apply'}
                    </button>
                  </div>
                </article>
              )
            })}
          </div>
        ) : <Panel><Empty title="No drives open right now" /></Panel>)}
      </Load>

      <Panel flush title="Your applications">
        <Load state={applications}>
          {({ applications: list }) => (
            <Table
              rows={list}
              empty={<Empty title="You haven't applied to any drive" />}
              columns={[
                { label: 'Company', cell: (a) => a.company },
                { label: 'Role', cell: (a) => a.role },
                { label: 'Status', cell: (a) => <Badge>{a.status}</Badge> },
              ]}
            />
          )}
        </Load>
      </Panel>
    </ErpLayout>
  )
}
