import { useEffect, useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { Eye, EyeOff } from 'lucide-react'
import Mark from '../components/Mark'
import { Field } from '../components/kit'
import { errorMessage, loginUser } from '../services/api'
import useAuthStore from '../stores/authStore'
import { getToken } from '../utils/auth'

export default function LoginPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()
  const login = useAuthStore((state) => state.login)

  useEffect(() => {
    document.title = 'Sign in · Studvisor'
  }, [])

  if (getToken()) return <Navigate to="/dashboard" replace />

  async function handleSubmit(event) {
    event.preventDefault()
    setLoading(true)
    setError('')
    try {
      const data = await loginUser({ username: username.trim(), password })
      if (data.role && data.role !== 'student') {
        setError('That is a staff account. These screens are for students only.')
        return
      }
      login(data.access_token, 'student', data.user || { name: username })
      navigate('/dashboard')
    } catch (failure) {
      setError(failure.response?.status === 401 ? 'That username and password do not match.' : errorMessage(failure))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login">
      <section className="login-art">
        <Mark size={180} />
        <p>Attendance, marks, fees and campus life for students, in one place.</p>
      </section>

      <main className="login-main">
        <form onSubmit={handleSubmit}>
          <div className="brand">
            <span className="brand-tile"><Mark size={27} /></span>
            Studvisor
          </div>
          <h1>Sign in</h1>
          <p className="login-sub">Use your student account.</p>

          {error && <p className="notice notice-bad" role="alert">{error}</p>}

          <Field label="Username">
            <input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" autoCapitalize="none" required />
          </Field>

          <div className="password-row">
            <Field label="Password">
              <input type={showPassword ? 'text' : 'password'} value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required />
            </Field>
            <button type="button" className="icon-btn" onClick={() => setShowPassword((value) => !value)} aria-label={showPassword ? 'Hide password' : 'Show password'}>
              {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
            </button>
          </div>

          <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
            {loading ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
      </main>
    </div>
  )
}
