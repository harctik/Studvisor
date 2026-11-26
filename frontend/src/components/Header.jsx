import { Menu } from 'lucide-react'
import { fetchProfile } from '../services/api'
import useApi from '../lib/useApi'

const initials = (name = '') => name.split(/\s+/).slice(0, 2).map((part) => part[0]).join('').toUpperCase()

export default function Header({ menuOpen, onMenu }) {
  const { data: student } = useApi(fetchProfile)
  const name = student && (student.full_name || student.username)

  return (
    <header className="topbar">
      <button type="button" className="icon-btn menu-btn" onClick={onMenu} aria-label="Open menu" aria-expanded={menuOpen} aria-controls="sidebar">
        <Menu size={22} />
      </button>
      {student && (
        <div className="who">
          <div>
            <strong>{name}</strong>
            <span>
              {[student.roll_number || student.username, student.department, student.semester && `Semester ${student.semester}`]
                .filter(Boolean)
                .join(' · ')}
            </span>
          </div>
          <span className="avatar" aria-hidden="true">{initials(name)}</span>
        </div>
      )}
    </header>
  )
}
