import { useState } from 'react'
import ErpLayout from '../components/ErpLayout'
import { Badge, Empty, Field, Load, Panel, Tabs } from '../components/kit'
import { formatDate } from '../lib/format'
import useApi from '../lib/useApi'
import { createAnonPost, errorMessage, fetchAnonPosts } from '../services/api'

const CATEGORIES = ['General', 'Confessions', 'Questions', 'Clubs', 'Funny', 'Vent']

export default function AnonChatPage() {
  const state = useApi(fetchAnonPosts)
  const [filter, setFilter] = useState('All')
  const [content, setContent] = useState('')
  const [category, setCategory] = useState('General')
  const [notice, setNotice] = useState(null)
  const [posting, setPosting] = useState(false)

  async function post(event) {
    event.preventDefault()
    if (!content.trim()) return
    setPosting(true)
    setNotice(null)
    try {
      const result = await createAnonPost(category, content.trim())
      setNotice(result.moderated
        ? { tone: 'warn', text: result.warning || 'Your post is held for review.' }
        : { tone: 'good', text: 'Posted.' })
      setContent('')
      await state.reload()
    } catch (error) {
      setNotice({ tone: 'bad', text: errorMessage(error) })
    } finally {
      setPosting(false)
    }
  }

  return (
    <ErpLayout title="Campus wall" subtitle="Posts appear without your name. Keep it kind.">
      <Panel title="Write a post">
        <form className="form" onSubmit={post}>
          <Field label="Your post"><textarea value={content} onChange={(e) => setContent(e.target.value)} maxLength={1000} required /></Field>
          <div className="toolbar">
            <label className="search" style={{ alignItems: 'center', maxWidth: 280 }}>
              <span className="meta">Category</span>
              <select value={category} onChange={(e) => setCategory(e.target.value)}>{CATEGORIES.map((c) => <option key={c}>{c}</option>)}</select>
            </label>
            <button type="submit" className="btn btn-primary" disabled={posting || !content.trim()}>{posting ? 'Posting…' : 'Post'}</button>
          </div>
          {notice && <p className={`notice notice-${notice.tone}`} role="status">{notice.text}</p>}
        </form>
      </Panel>

      <Tabs label="Filter posts" value={filter} onChange={setFilter} tabs={['All', ...CATEGORIES].map((c) => [c, c])} />

      <Load state={state}>
        {({ posts }) => {
          const list = posts.filter((p) => filter === 'All' || String(p.category).toLowerCase() === filter.toLowerCase())
          if (!list.length) return <Panel><Empty title="No posts here yet" /></Panel>
          return (
            <div className="form">
              {list.map((p) => (
                <article key={p.id} className="card">
                  <p className="pre">{p.content}</p>
                  <div className="card-foot" style={{ justifyContent: 'flex-start' }}>
                    <Badge>{p.category}</Badge>
                    <span className="meta">{formatDate(p.date)}</span>
                  </div>
                </article>
              ))}
            </div>
          )
        }}
      </Load>
    </ErpLayout>
  )
}
