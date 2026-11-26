import ErpLayout from '../components/ErpLayout'
import { Badge, Empty, Load, Panel, Stats } from '../components/kit'
import useApi from '../lib/useApi'
import { fetchLeaderboard, fetchMerit } from '../services/api'

export default function LeaderboardPage() {
  const board = useApi(fetchLeaderboard)
  const merit = useApi(fetchMerit)

  return (
    <ErpLayout title="Leaderboard" subtitle="Students ranked by merit points.">
      <Load state={merit}>
        {(m) => <Stats items={[{ label: 'Your merit points', value: m.points }, { label: 'Your tier', value: m.tier || '—' }]} />}
      </Load>
      <Panel flush>
        <Load state={board}>
          {({ leaderboard }) => (leaderboard.length ? (
            <div className="table-scroll">
              <table>
                <thead><tr><th>Rank</th><th>Student</th><th>Department</th><th>Tier</th><th className="num">Points</th></tr></thead>
                <tbody>
                  {leaderboard.map((row) => (
                    <tr key={row.rank}>
                      <td><span className={`rank ${row.rank <= 3 ? 'rank-top' : ''}`}>{row.rank}</span></td>
                      <td>{row.name}</td>
                      <td>{row.department}</td>
                      <td><Badge>{row.tier}</Badge></td>
                      <td className="num">{row.points}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <Empty title="No one is ranked yet" />)}
        </Load>
      </Panel>
    </ErpLayout>
  )
}
