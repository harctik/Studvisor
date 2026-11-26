import { useState } from 'react'
import ErpLayout from '../components/ErpLayout'
import { Badge, Dialog, Empty, Field, Load, Panel, Tabs, useAction } from '../components/kit'
import { formatDate } from '../lib/format'
import useApi from '../lib/useApi'
import { createLostFoundItem, errorMessage, fetchLostFound } from '../services/api'

function NewReport({ onClose, onSent }) {
  const [form, setForm] = useState({ type: 'lost', item_name: '', description: '', category: 'General', location: '' })
  const [problem, setProblem] = useState(null)
  const [sending, setSending] = useState(false)
  const set = (key) => (event) => setForm((current) => ({ ...current, [key]: event.target.value }))

  async function submit(event) {
    event.preventDefault()
    if (!form.item_name.trim()) return setProblem('Say what the item is.')
    setSending(true)
    setProblem(null)
    try {
      await createLostFoundItem({ ...form, item_name: form.item_name.trim() })
      onSent()
    } catch (error) {
      setProblem(errorMessage(error))
      setSending(false)
    }
    return undefined
  }

  return (
    <Dialog title="Report an item" onClose={onClose}>
      <form className="form" onSubmit={submit}>
        <div className="form-grid">
          <Field label="I have">
            <select value={form.type} onChange={set('type')}><option value="lost">Lost something</option><option value="found">Found something</option></select>
          </Field>
          <Field label="Category">
            <select value={form.category} onChange={set('category')}>
              {['General', 'Electronics', 'ID and cards', 'Books', 'Clothing', 'Keys'].map((c) => <option key={c}>{c}</option>)}
            </select>
          </Field>
        </div>
        <Field label="Item"><input value={form.item_name} onChange={set('item_name')} maxLength={150} required /></Field>
        <Field label="Where"><input value={form.location} onChange={set('location')} maxLength={150} required /></Field>
        <Field label="Description"><textarea value={form.description} onChange={set('description')} maxLength={1000} required /></Field>
        {problem && <p className="notice notice-bad" role="alert">{problem}</p>}
        <div className="form-actions">
          <button type="button" className="btn" onClick={onClose}>Cancel</button>
          <button type="submit" className="btn btn-primary" disabled={sending}>{sending ? 'Posting…' : 'Post report'}</button>
        </div>
      </form>
    </Dialog>
  )
}

export default function LostFoundPage() {
  const state = useApi(fetchLostFound)
  const { run, banner } = useAction(state.reload)
  const [view, setView] = useState('all')
  const [writing, setWriting] = useState(false)

  return (
    <ErpLayout title="Lost and found" subtitle="Items reported around campus."
      actions={<button type="button" className="btn btn-primary" onClick={() => setWriting(true)}>Report an item</button>}>
      <Tabs label="Filter items" value={view} onChange={setView} tabs={[['all', 'All'], ['lost', 'Lost'], ['found', 'Found']]} />
      {banner}
      <Load state={state}>
        {({ items }) => {
          const list = items.filter((i) => view === 'all' || i.type === view)
          if (!list.length) return <Panel><Empty title="No items reported" /></Panel>
          return (
            <div className="cards">
              {list.map((i) => (
                <article key={i.id} className="card">
                  <div className="card-top"><h2>{i.item}</h2><Badge tone={i.type === 'found' ? 'good' : 'warn'}>{i.type === 'found' ? 'Found' : 'Lost'}</Badge></div>
                  {i.description && <p className="pre">{i.description}</p>}
                  <p className="meta">{[i.location, formatDate(i.date), i.status].filter(Boolean).join(' · ')}</p>
                </article>
              ))}
            </div>
          )
        }}
      </Load>
      {writing && (
        <NewReport onClose={() => setWriting(false)} onSent={() => { setWriting(false); run('sent', async () => ({ message: 'Report posted.' })) }} />
      )}
    </ErpLayout>
  )
}
