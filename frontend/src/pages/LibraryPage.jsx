import { useRef, useState } from 'react'
import ErpLayout from '../components/ErpLayout'
import { Badge, Empty, Load, Panel, Table, Tabs, useAction } from '../components/kit'
import { formatDate, money } from '../lib/format'
import useApi from '../lib/useApi'
import { fetchMyBooks, issueBook, returnBook, searchLibrary } from '../services/api'

export default function LibraryPage() {
  const [view, setView] = useState('catalog')
  const [text, setText] = useState('')
  const query = useRef('')
  const catalog = useApi(() => searchLibrary(query.current))
  const mine = useApi(fetchMyBooks)
  const { busy, run, banner } = useAction(() => Promise.all([catalog.reload(), mine.reload()]))

  function search(event) {
    event.preventDefault()
    query.current = text.trim()
    catalog.reload()
  }

  return (
    <ErpLayout title="Library" subtitle="Find books and keep track of the ones you have.">
      <div className="toolbar">
        <Tabs label="Library view" value={view} onChange={setView} tabs={[['catalog', 'Catalogue'], ['mine', 'My books']]} />
        {view === 'catalog' && (
          <form className="search" onSubmit={search} role="search">
            <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Title, author or ISBN" aria-label="Search the catalogue" />
            <button type="submit" className="btn">Search</button>
          </form>
        )}
      </div>
      {banner}

      {view === 'catalog' ? (
        <Panel flush>
          <Load state={catalog}>
            {({ books }) => (
              <Table
                rows={books}
                rowKey={(b) => b.id}
                empty={<Empty title="No books match that search" />}
                columns={[
                  { label: 'Title', cell: (b) => <>{b.title}<span className="cell-sub">{b.author}</span></> },
                  { label: 'Category', cell: (b) => b.category || '—' },
                  { label: 'Shelf', cell: (b) => b.shelf_location || '—' },
                  { label: 'Copies', numeric: true, cell: (b) => b.available_copies },
                  {
                    label: 'Action',
                    cell: (b) => (
                      <button type="button" className="btn btn-small" disabled={b.available_copies < 1 || busy === `issue-${b.id}`}
                        onClick={() => run(`issue-${b.id}`, () => issueBook(b.id))}>
                        {b.available_copies < 1 ? 'Unavailable' : 'Borrow'}
                      </button>
                    ),
                  },
                ]}
              />
            )}
          </Load>
        </Panel>
      ) : (
        <Panel flush>
          <Load state={mine}>
            {({ issued_books: books }) => (
              <Table
                rows={books}
                rowKey={(b) => b.issue_id}
                empty={<Empty title="You haven't borrowed any books" />}
                columns={[
                  { label: 'Title', cell: (b) => b.title },
                  { label: 'Borrowed', cell: (b) => formatDate(b.issue_date) },
                  { label: 'Due', cell: (b) => formatDate(b.due_date) },
                  { label: 'Fine', numeric: true, cell: (b) => (b.fine ? money(b.fine) : '—') },
                  { label: 'Status', cell: (b) => <Badge tone={b.status === 'returned' || b.status === 'Returned' ? 'good' : 'plain'}>{b.status}</Badge> },
                  {
                    label: 'Action',
                    cell: (b) => String(b.status).toLowerCase() !== 'returned' && (
                      <button type="button" className="btn btn-small" disabled={busy === `return-${b.issue_id}`}
                        onClick={() => window.confirm(`Return "${b.title}"?`) && run(`return-${b.issue_id}`, () => returnBook(b.issue_id), 'Book returned.')}>
                        Return
                      </button>
                    ),
                  },
                ]}
              />
            )}
          </Load>
        </Panel>
      )}
    </ErpLayout>
  )
}
