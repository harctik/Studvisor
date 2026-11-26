import { useState } from 'react'
import ErpLayout from '../components/ErpLayout'
import { Badge, Empty, Field, Load, Panel, Stats, Table, useAction } from '../components/kit'
import { formatDate, toISODate } from '../lib/format'
import useApi from '../lib/useApi'
import { applyLeaveRequest, fetchLeaveBalance, fetchLeaveRequests, withdrawLeaveRequest } from '../services/api'

const TYPES = ['Medical', 'Personal', 'OD', 'Other']
const TONE = { Approved: 'good', Rejected: 'bad', Pending: 'warn' }
const blank = () => ({ leave_type: 'Medical', from_date: toISODate(), to_date: toISODate(), reason: '' })

export default function LeavePage() {
  const requests = useApi(fetchLeaveRequests)
  const balance = useApi(fetchLeaveBalance)
  const { busy, run, banner } = useAction(() => Promise.all([requests.reload(), balance.reload()]))
  const [form, setForm] = useState(blank)
  const [problem, setProblem] = useState(null)

  const set = (key) => (event) => setForm((current) => ({ ...current, [key]: event.target.value }))

  async function submit(event) {
    event.preventDefault()
    // The server accepts any dates, so the order is checked here.
    if (form.to_date < form.from_date) return setProblem('The end date is before the start date.')
    if (!form.reason.trim()) return setProblem('Add a reason for the request.')
    setProblem(null)
    const sent = await run('apply', () => applyLeaveRequest({ ...form, reason: form.reason.trim() }), 'Request sent.')
    if (sent) setForm(blank())
    return undefined
  }

  return (
    <ErpLayout title="Leave and OD" subtitle="Apply for leave or on-duty and follow each request.">
      <Load state={balance}>
        {({ balance: types }) => (
          <Stats items={Object.entries(types).map(([type, b]) => ({
            label: `${type} requests left`, value: b.remaining, note: `${b.used} of ${b.limit} used`,
          }))} />
        )}
      </Load>

      {banner}

      <Panel title="New request">
        <form className="form" onSubmit={submit}>
          <div className="form-grid">
            <Field label="Type">
              <select value={form.leave_type} onChange={set('leave_type')}>
                {TYPES.map((t) => <option key={t}>{t}</option>)}
              </select>
            </Field>
            <Field label="From"><input type="date" value={form.from_date} onChange={set('from_date')} required /></Field>
            <Field label="To"><input type="date" value={form.to_date} min={form.from_date} onChange={set('to_date')} required /></Field>
          </div>
          <Field label="Reason"><textarea value={form.reason} onChange={set('reason')} maxLength={500} required /></Field>
          {problem && <p className="notice notice-bad" role="alert">{problem}</p>}
          <div className="form-actions">
            <button type="submit" className="btn btn-primary" disabled={busy === 'apply'}>{busy === 'apply' ? 'Sending…' : 'Send request'}</button>
          </div>
        </form>
      </Panel>

      <Panel flush title="Your requests">
        <Load state={requests}>
          {({ leaves }) => (
            <Table
              rows={leaves}
              rowKey={(l) => l.id}
              empty={<Empty title="You haven't made any requests" />}
              columns={[
                { label: 'Type', cell: (l) => l.type },
                { label: 'Dates', cell: (l) => (l.from === l.to ? formatDate(l.from) : `${formatDate(l.from)} to ${formatDate(l.to)}`) },
                { label: 'Reason', cell: (l) => l.reason },
                { label: 'Status', cell: (l) => <Badge tone={TONE[l.status]}>{l.status}</Badge> },
                {
                  label: 'Action',
                  cell: (l) => l.status === 'Pending' && (
                    <button type="button" className="btn btn-small" disabled={busy === l.id}
                      onClick={() => window.confirm('Withdraw this request?') && run(l.id, () => withdrawLeaveRequest(l.id), 'Request withdrawn.')}>
                      Withdraw
                    </button>
                  ),
                },
              ]}
            />
          )}
        </Load>
      </Panel>
    </ErpLayout>
  )
}
