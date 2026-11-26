import { NavLink, Link, useNavigate } from 'react-router-dom'
import {
  Award, Banknote, BarChart3, BookMarked, BookOpen, Briefcase, CalendarCheck, CalendarDays,
  ClipboardList, Clock, FileText, GraduationCap, LayoutGrid, LogOut, Megaphone, MessageSquare,
  MessagesSquare, PencilRuler, PieChart, Search, Sigma, Trophy, UserRound, Vote, X, Ticket,
} from 'lucide-react'
import Mark from './Mark'
import useAuthStore from '../stores/authStore'

export const NAV = [
  ['Today', [
    ['/dashboard', 'Dashboard', LayoutGrid],
    ['/timetable', 'Timetable', Clock],
    ['/calendar', 'Calendar', CalendarDays],
    ['/announcements', 'Announcements', Megaphone],
  ]],
  ['Academics', [
    ['/attendance', 'Attendance', CalendarCheck],
    ['/leave', 'Leave and OD', FileText],
    ['/assignments', 'Assignments', ClipboardList],
    ['/exams', 'Exams', PencilRuler],
    ['/results', 'Marks', BarChart3],
    ['/gpa', 'GPA', Sigma],
    ['/analytics', 'Analytics', PieChart],
    ['/syllabus', 'Syllabus', BookOpen],
    ['/notes', 'Notes', BookMarked],
  ]],
  ['Services', [
    ['/fees', 'Fees', Banknote],
    ['/library', 'Library', GraduationCap],
    ['/placement', 'Placements', Briefcase],
    ['/faculty', 'Faculty', UserRound],
    ['/complaints', 'Complaints', MessageSquare],
    ['/lost-found', 'Lost and found', Search],
  ]],
  ['Campus', [
    ['/campus-wall', 'Campus wall', MessagesSquare],
    ['/events', 'Events', Ticket],
    ['/polls', 'Polls', Vote],
    ['/leaderboard', 'Leaderboard', Trophy],
    ['/achievements', 'Achievements', Award],
  ]],
]

export default function Sidebar({ open, onClose }) {
  const navigate = useNavigate()
  const logout = useAuthStore((state) => state.logout)

  function signOut() {
    logout()
    navigate('/login')
  }

  return (
    <aside id="sidebar" className={`sidebar ${open ? 'is-open' : ''}`}>
      <div className="sidebar-top">
        <Link to="/dashboard" className="brand">
          <span className="brand-tile"><Mark size={27} /></span>
          Studvisor
        </Link>
        <button type="button" className="icon-btn sidebar-close" onClick={onClose} aria-label="Close menu">
          <X size={20} />
        </button>
      </div>

      <nav aria-label="Main">
        {NAV.map(([group, items]) => (
          <div key={group} className="nav-group">
            <p>{group}</p>
            {items.map(([to, label, Icon]) => (
              <NavLink key={to} to={to} onClick={onClose}>
                <Icon size={17} aria-hidden="true" />
                {label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      <button type="button" className="sign-out" onClick={signOut}>
        <LogOut size={17} aria-hidden="true" />
        Sign out
      </button>
    </aside>
  )
}
