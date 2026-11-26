import { useState } from 'react'
import ErpLayout from '../components/ErpLayout'
import { Badge, Dialog, Empty, Field, Load, Panel, useAction } from '../components/kit'
import useApi from '../lib/useApi'
import { createComplaint, errorMessage, fetchComplaints } from '../services/api'

const CATEGORIES = [['general', 'General'], ['academic', 'Academic'], ['facility', 'Facilities'], ['hostel', 'Hostel'], ['transport', 'Transport']]
const URGENCY = ['Low', 'Medium', 'High']
const urgencyTone = (value) => ({ High: 'bad', Critical: 'bad', Medium: 'warn' }[value] || 'plain')
const statusTone = (value) => (/resolved|closed/i.test(value) ? 'good' : 'plain')

function NewComplaint({ onClose, onSent }) {
  const [form, setForm] = useState({ title: '', description: '', category: 'general', urgency: 'Medium' })
  const [problem, setProblem] = useState(null)
  const [sending, setSending] = useState(false)
  const set = (key) => (event) => setForm((current) => ({ ...current, [key]: event.target.value }))

  async function submit(event) {
    event.preventDefault()
    if (!form.title.trim() || !form.description.trim()) return setProblem('Add a title and a description.')
    setSending(true)
    setProblem(null)
    try {
      const result = await createComplaint({ ...form, title: form.title.trim(), description: form.description.trim() })
      const raised = result.ai_urgency_assigned && result.ai_urgency_assigned !== form.urgency
      onSent(raised ? `Complaint sent. The server set its urgency to ${result.ai_urgency_assigned}.` : 'Complaint sent.')
    } catch (error) {
      setProblem(errorMessage(error))
      setSending(false)
    }
    return undefined
  }

  return (
    <Dialog title="New complaint" onClose={onClose}>
      <form className="form" onSubmit={submit}>
        <Field label="Title"><input value={form.title} onChange={set('title')} maxLength={200} required /></Field>
        <Field label="What happened?"><textarea value={form.description} onChange={set('description')} maxLength={2000} required /></Field>
        <div className="form-grid">
          <Field label="Category">
            <select value={form.category} onChange={set('category')}>{CATEGORIES.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select>
          </Field>
          <Field label="Urgency">
            <select value={form.urgency} onChange={set('urgency')}>{URGENCY.map((u) => <option key={u}>{u}</option>)}</select>
          </Field>
        </div>
        <p className="meta">The server may raise the urgency based on what you write.</p>
        {problem && <p className="notice notice-bad" role="alert">{problem}</p>}
        <div className="form-actions">
          <button type="button" className="btn" onClick={onClose}>Cancel</button>
          <button type="submit" className="btn btn-primary" disabled={sending}>{sending ? 'Sending…' : 'Send complaint'}</button>
        </div>
      </form>
    </Dialog>
  )
}

export default function ComplaintsPage() {
  const state = useApi(fetchComplaints)
  const { run, banner } = useAction(state.reload)
  const [writing, setWriting] = useState(false)

  return (
    <ErpLayout title="Complaints" subtitle="Report a problem and follow the reply. This list shows only yours."
      actions={<button type="button" className="btn btn-primary" onClick={() => setWriting(true)}>New complaint</button>}>
      {banner}
      <Load state={state}>
        {({ complaints }) => (complaints.length ? (
          <div className="form">
            {complaints.map((c) => (
              <article key={c.id} className="card">
                <div className="card-top">
                  <h2>{c.title}</h2>
                  <Badge tone={statusTone(c.status)}>{c.status}</Badge>
                </div>
                <div className="card-foot" style={{ justifyContent: 'flex-start' }}>
                  <Badge>{c.category}</Badge>
                  <Badge tone={urgencyTone(c.urgency)}>{c.urgency} urgency</Badge>
                </div>
                <p className="pre">{c.response ? `Reply: ${c.response}` : 'No reply yet.'}</p>
              </article>
            ))}
          </div>
        ) : <Panel><Empty title="You haven't sent any complaints" /></Panel>)}
      </Load>
      {writing && (
        <NewComplaint onClose={() => setWriting(false)} onSent={(message) => { setWriting(false); run('sent', async () => ({ message })) }} />
      )}
    </ErpLayout>
  )
}
