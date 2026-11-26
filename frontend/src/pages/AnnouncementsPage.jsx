import ErpLayout from '../components/ErpLayout'
import { Badge, Empty, Load } from '../components/kit'
import { formatDate } from '../lib/format'
import useApi from '../lib/useApi'
import { fetchAnnouncements } from '../services/api'

export default function AnnouncementsPage() {
  const state = useApi(fetchAnnouncements)
  return (
    <ErpLayout title="Announcements" subtitle="Notices from the college.">
      <Load state={state}>
        {({ announcements }) => (announcements.length ? (
          <div className="form">
            {[...announcements].sort((a, b) => b.date.localeCompare(a.date)).map((item) => (
              <article key={item.id} className="card">
                <div className="card-top">
                  <h2>{item.title}</h2>
                  {item.scope && <Badge>{item.scope === 'all' ? 'Everyone' : item.scope}</Badge>}
                </div>
                <p className="pre">{item.content}</p>
                <p className="meta">{formatDate(item.date)}</p>
              </article>
            ))}
          </div>
        ) : <Empty title="No announcements yet" />)}
      </Load>
    </ErpLayout>
  )
}
