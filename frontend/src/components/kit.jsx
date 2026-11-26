import { useEffect, useRef, useState } from 'react'
import { X } from 'lucide-react'
import { errorMessage } from '../services/api'

/** Loading and failure states, so a broken request never looks like an empty list. */
export function Load({ state, children }) {
  const { data, error, loading, reload } = state
  if (loading) return <p className="loading" role="status">Loading…</p>
  const failure = error && (
    <div className="notice notice-bad" role="alert">
      <span>{error}</span>
      <button type="button" className="btn btn-small" onClick={reload}>Try again</button>
    </div>
  )
  if (data === null) return failure
  return (
    <>
      {failure}
      {children(data)}
    </>
  )
}

export function Panel({ title, action, flush, className = '', children }) {
  return (
    <section className={`panel ${flush ? 'panel-flush' : ''} ${className}`}>
      {(title || action) && (
        <header className="panel-head">
          {title ? <h2>{title}</h2> : <span />}
          {action}
        </header>
      )}
      {children}
    </section>
  )
}

export function Stats({ items }) {
  return (
    <dl className="stats">
      {items.map(({ label, value, note, tone }) => (
        <div key={label} className={`stat ${tone ? `stat-${tone}` : ''}`}>
          <dt>{label}</dt>
          <dd>{value}</dd>
          {note && <p>{note}</p>}
        </div>
      ))}
    </dl>
  )
}

export function Empty({ title, children }) {
  return (
    <div className="empty">
      <strong>{title}</strong>
      {children && <p>{children}</p>}
    </div>
  )
}

export function Badge({ tone = 'plain', children }) {
  return <span className={`badge badge-${tone}`}>{children}</span>
}

/** columns: [{ label, cell(row), numeric }] */
export function Table({ columns, rows, rowKey, empty }) {
  if (!rows.length) return empty || <Empty title="Nothing here yet" />
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            {columns.map((c) => <th key={c.label} className={c.numeric ? 'num' : undefined}>{c.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={rowKey ? rowKey(row) : index}>
              {columns.map((c) => <td key={c.label} className={c.numeric ? 'num' : undefined}>{c.cell(row)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Tabs({ tabs, value, onChange, label }) {
  return (
    <div className="tabs" role="group" aria-label={label}>
      {tabs.map(([id, text]) => (
        <button key={id} type="button" aria-pressed={value === id} onClick={() => onChange(id)}>{text}</button>
      ))}
    </div>
  )
}

export function Meter({ percent, target }) {
  const width = Math.max(0, Math.min(100, percent))
  return (
    <div className={`meter ${target && percent < target ? 'meter-low' : ''}`}>
      <span style={{ width: `${width}%` }} />
      {target ? <i style={{ left: `${target}%` }} /> : null}
    </div>
  )
}

export function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  )
}

/** Native dialog: focus trapping and Escape come from the browser. */
export function Dialog({ title, onClose, children }) {
  const ref = useRef(null)

  useEffect(() => {
    const dialog = ref.current
    if (dialog && !dialog.open) dialog.showModal()
  }, [])

  return (
    <dialog ref={ref} className="dialog" onClose={onClose} aria-label={title}>
      <header className="panel-head">
        <h2>{title}</h2>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Close"><X size={18} /></button>
      </header>
      {children}
    </dialog>
  )
}

/** Run a write request, show what the server said, then refresh the page data. */
export function useAction(reload) {
  const [busy, setBusy] = useState(null)
  const [notice, setNotice] = useState(null)

  async function run(key, request, success) {
    setBusy(key)
    setNotice(null)
    try {
      const result = await request()
      setNotice({ tone: 'good', text: success || result?.message || 'Done.' })
      await reload?.()
      return true
    } catch (error) {
      setNotice({ tone: 'bad', text: errorMessage(error) })
      return false
    } finally {
      setBusy(null)
    }
  }

  const banner = notice && (
    <div className={`notice notice-${notice.tone}`} role="status">
      <span>{notice.text}</span>
      <button type="button" className="icon-btn" onClick={() => setNotice(null)} aria-label="Dismiss"><X size={16} /></button>
    </div>
  )

  return { busy, run, banner }
}
