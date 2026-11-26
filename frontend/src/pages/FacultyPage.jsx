import { useState } from 'react'
import ErpLayout from '../components/ErpLayout'
import { Empty, Load, Panel } from '../components/kit'
import useApi from '../lib/useApi'
import { fetchFaculty } from '../services/api'

export default function FacultyPage() {
  const state = useApi(fetchFaculty)
  const [text, setText] = useState('')
  const needle = text.trim().toLowerCase()

  return (
    <ErpLayout title="Faculty" subtitle="Who teaches what, and where to find them.">
      <div className="search" role="search">
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Name, department or subject code" aria-label="Search faculty" />
      </div>
      <Load state={state}>
        {({ faculty }) => {
          const list = faculty.filter((f) => [f.name, f.department, f.subjects].join(' ').toLowerCase().includes(needle))
          if (!list.length) return <Panel><Empty title={needle ? 'No one matches that search' : 'The directory is empty'} /></Panel>
          return (
            <div className="cards">
              {list.map((f) => (
                <article key={f.id} className="card">
                  <div>
                    <h2>{f.name}</h2>
                    <p className="meta">{[f.designation, f.department].filter(Boolean).join(' · ')}</p>
                  </div>
                  {f.subjects && <p>Teaches {f.subjects.split(',').map((s) => s.trim()).join(', ')}</p>}
                  {f.cabin && <p className="meta">Cabin {f.cabin.replace(/^cabin\s*/i, '')}</p>}
                  {f.email && <a href={`mailto:${f.email}`}>{f.email}</a>}
                </article>
              ))}
            </div>
          )
        }}
      </Load>
    </ErpLayout>
  )
}
