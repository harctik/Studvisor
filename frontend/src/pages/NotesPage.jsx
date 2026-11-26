import ErpLayout from '../components/ErpLayout'
import { Empty, Load, Panel, useAction } from '../components/kit'
import { safeUrl } from '../lib/format'
import useApi from '../lib/useApi'
import { fetchNotes, rateNote } from '../services/api'

export default function NotesPage() {
  const state = useApi(fetchNotes)
  const { busy, run, banner } = useAction(state.reload)

  return (
    <ErpLayout title="Notes" subtitle="Shared study material.">
      {banner}
      <Load state={state}>
        {({ notes }) => (notes.length ? (
          <div className="cards">
            {notes.map((note) => {
              const url = safeUrl(note.file_url || '')
              return (
                <article key={note.id} className="card">
                  <h2>{note.title}</h2>
                  {note.file_url && url
                    ? <a href={url} target="_blank" rel="noreferrer">Open file</a>
                    : <p className="meta">No file link</p>}
                  <div className="card-foot">
                    {[[true, 'Helpful', note.helpful], [false, 'Not helpful', note.not_helpful]].map(([helpful, label, count]) => (
                      <button key={label} type="button" className="btn btn-small" disabled={busy === note.id}
                        onClick={() => run(note.id, () => rateNote(note.id, helpful), 'Thanks for rating.')}>
                        {label} ({count})
                      </button>
                    ))}
                  </div>
                </article>
              )
            })}
          </div>
        ) : <Panel><Empty title="No notes shared yet" /></Panel>)}
      </Load>
    </ErpLayout>
  )
}
