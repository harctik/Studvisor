import { useState } from 'react'
import { useNavigate, Navigate } from 'react-router-dom'
import { Eye, EyeOff } from 'lucide-react'
import useAuthStore from '../stores/authStore'
import { getToken } from '../utils/auth'
import { loginUser } from '../services/api'

export default function LoginPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()
  const { login } = useAuthStore()

  if (getToken()) return <Navigate to="/dashboard" replace />

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const data = await loginUser({ username, password })
      login(data.access_token, data.role || 'student', data.user || { name: username })
      navigate('/dashboard')
    } catch (err) {
      setError('Invalid credentials. Please verify your identity.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="linways-login-wrapper">
      <div className="linways-login-left"></div>
      
      <div className="linways-login-right">
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', flex: 1, justifyContent: 'center' }}>
          <h1 className="linways-login-title">Get Started with AMS</h1>
          <p className="linways-login-subtitle">Sign in to your student account</p>

          {error && <div className="error-box">{error}</div>}

          <div className="linways-input-group">
            <label>Login</label>
            <input 
              type="text" 
              placeholder="User Name" 
              value={username} 
              onChange={e => setUsername(e.target.value)}
              required 
            />
          </div>

          <div className="linways-input-group" style={{ flexDirection: 'row', alignItems: 'center' }}>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <label>Password</label>
              <input 
                type={showPassword ? 'text' : 'password'} 
                placeholder="Credentials" 
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
              />
            </div>
            <div onClick={() => setShowPassword(!showPassword)} style={{ cursor: 'pointer', color: '#666', padding: '0 8px' }}>
              {showPassword ? <EyeOff size={20}/> : <Eye size={20}/>}
            </div>
          </div>

          <button type="submit" className="linways-btn-primary" disabled={loading}>
            {loading ? 'Signing in...' : 'Sign in'}
          </button>

          <button type="button" className="linways-btn-outline">
            <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Google_%22G%22_Logo.svg" width="18" height="18" alt="Google" />
            Sign in with Google
          </button>

          <button type="button" className="linways-btn-outline">
            <img src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" width="18" height="18" alt="Microsoft" />
            Sign in with Microsoft
          </button>

          <div className="linways-forgot">Forgot Password?</div>
        </form>

        <div className="linways-app-links">
          <p style={{ marginBottom: '8px' }}>Download the app now</p>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '12px' }}>
            <img src="https://upload.wikimedia.org/wikipedia/commons/7/78/Google_Play_Store_badge_EN.svg" height="30" alt="Play Store" />
            <img src="https://developer.apple.com/app-store/marketing/guidelines/images/badge-example-preferred.png" height="30" alt="App Store" />
          </div>
          <div style={{ marginTop: '24px', fontSize: '0.75rem', color: '#888' }}>
            Powered by Linways Technologies Pvt.Ltd.
          </div>
        </div>
      </div>
    </div>
  )
}
