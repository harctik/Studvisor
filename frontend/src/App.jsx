import { Link, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import ErrorBoundary from './components/ErrorBoundary'
import FloatingChatWidget from './components/FloatingChatWidget'
import Mark from './components/Mark'
import AchievementsPage from './pages/AchievementsPage'
import AnalyticsPage from './pages/AnalyticsPage'
import AnnouncementsPage from './pages/AnnouncementsPage'
import AnonChatPage from './pages/AnonChatPage'
import AssignmentsPage from './pages/AssignmentsPage'
import AttendancePage from './pages/AttendancePage'
import CalendarPage from './pages/CalendarPage'
import ComplaintsPage from './pages/ComplaintsPage'
import DashboardPage from './pages/DashboardPage'
import EventsPage from './pages/EventsPage'
import ExamsPage from './pages/ExamsPage'
import FacultyPage from './pages/FacultyPage'
import FeesPage from './pages/FeesPage'
import GPAPage from './pages/GPAPage'
import LeaderboardPage from './pages/LeaderboardPage'
import LeavePage from './pages/LeavePage'
import LibraryPage from './pages/LibraryPage'
import LoginPage from './pages/LoginPage'
import LostFoundPage from './pages/LostFoundPage'
import NotesPage from './pages/NotesPage'
import PlacementPage from './pages/PlacementPage'
import PollsPage from './pages/PollsPage'
import ResultsPage from './pages/ResultsPage'
import SyllabusPage from './pages/SyllabusPage'
import TimetablePage from './pages/TimetablePage'
import { getToken } from './utils/auth'
import './studvisor.css'

// [path, page, context sent to the study assistant]
const PAGES = [
  ['/dashboard', DashboardPage],
  ['/timetable', TimetablePage, 'timetable'],
  ['/calendar', CalendarPage, 'calendar'],
  ['/announcements', AnnouncementsPage],
  ['/attendance', AttendancePage, 'attendance'],
  ['/leave', LeavePage, 'od'],
  ['/assignments', AssignmentsPage],
  ['/exams', ExamsPage],
  ['/results', ResultsPage, 'results'],
  ['/gpa', GPAPage],
  ['/analytics', AnalyticsPage, 'analytics'],
  ['/syllabus', SyllabusPage],
  ['/notes', NotesPage],
  ['/fees', FeesPage],
  ['/library', LibraryPage],
  ['/placement', PlacementPage],
  ['/faculty', FacultyPage],
  ['/complaints', ComplaintsPage],
  ['/lost-found', LostFoundPage],
  ['/campus-wall', AnonChatPage, 'campus'],
  ['/events', EventsPage],
  ['/polls', PollsPage],
  ['/leaderboard', LeaderboardPage],
  ['/achievements', AchievementsPage],
]

function Protected({ context = 'dashboard', children }) {
  const location = useLocation()
  if (!getToken()) return <Navigate to="/login" replace />
  return (
    <>
      {/* keyed so each screen starts fresh, including its error state */}
      <ErrorBoundary key={location.pathname}>{children}</ErrorBoundary>
      <FloatingChatWidget contextPage={context} />
    </>
  )
}

function NotFoundPage() {
  return (
    <div className="standalone">
      <span className="brand-tile brand-tile-large"><Mark size={46} /></span>
      <h1>Page not found</h1>
      <p>That address doesn't match any screen in Studvisor.</p>
      <Link to="/dashboard" className="btn btn-primary">Go to the dashboard</Link>
    </div>
  )
}

export default function App() {
  return (
    <ErrorBoundary>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        {PAGES.map(([path, Page, context]) => (
          <Route key={path} path={path} element={<Protected context={context}><Page /></Protected>} />
        ))}
        <Route path="/hub" element={<Navigate to="/complaints" replace />} />
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </ErrorBoundary>
  )
}
