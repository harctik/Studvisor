import ErpLayout from '../components/ErpLayout'
import { Badge, Empty, Load, Panel, Stats, Table } from '../components/kit'
import { formatDate } from '../lib/format'
import useApi from '../lib/useApi'
import { fetchAchievements, fetchMerit } from '../services/api'

export default function AchievementsPage() {
  const badges = useApi(fetchAchievements)
  const merit = useApi(fetchMerit)

  return (
    <ErpLayout title="Achievements" subtitle="Badges and merit points you have earned.">
      <Load state={merit}>
        {(m) => <Stats items={[{ label: 'Merit points', value: m.points }, { label: 'Tier', value: m.tier || '—' }]} />}
      </Load>
      <Load state={badges}>
        {({ achievements }) => (achievements.length ? (
          <div className="cards">
            {achievements.map((a) => (
              <article key={`${a.badge}-${a.earned}`} className="card">
                <div className="card-top"><h2>{a.badge}</h2><Badge tone="lime">{a.xp} XP</Badge></div>
                <p className="meta">Earned {formatDate(a.earned)}</p>
              </article>
            ))}
          </div>
        ) : <Panel><Empty title="No badges yet" /></Panel>)}
      </Load>
      <Panel flush title="Points history">
        <Load state={merit}>
          {({ history }) => (
            <Table
              rows={history}
              empty={<Empty title="No points awarded yet" />}
              columns={[
                { label: 'Date', cell: (h) => formatDate(h.date) },
                { label: 'Reason', cell: (h) => h.reason },
                { label: 'Points', numeric: true, cell: (h) => (h.points > 0 ? `+${h.points}` : h.points) },
              ]}
            />
          )}
        </Load>
      </Panel>
    </ErpLayout>
  )
}
