import { useEffect, useState } from 'react'
import Sidebar from './Sidebar'
import Header from './Header'

export default function ErpLayout({ title, subtitle, actions, children }) {
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    document.title = `${title} · Studvisor`
  }, [title])

  useEffect(() => {
    if (!menuOpen) return undefined
    const close = (event) => event.key === 'Escape' && setMenuOpen(false)
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [menuOpen])

  return (
    <div className="app">
      <a className="skip" href="#content">Skip to content</a>
      <Sidebar open={menuOpen} onClose={() => setMenuOpen(false)} />
      {menuOpen && <div className="scrim" onClick={() => setMenuOpen(false)} />}
      <div className="app-body">
        <Header menuOpen={menuOpen} onMenu={() => setMenuOpen(true)} />
        <main id="content" className="page">
          <div className="page-head">
            <div>
              <h1>{title}</h1>
              {subtitle && <p>{subtitle}</p>}
            </div>
            {actions}
          </div>
          {children}
        </main>
      </div>
    </div>
  )
}
