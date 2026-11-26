import { useEffect, useState } from 'react'
import { MessageCircle, X } from 'lucide-react'
import ChatBox from './ChatBox'

export default function FloatingChatWidget({ contextPage = 'dashboard' }) {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (!open) return undefined
    const close = (event) => event.key === 'Escape' && setOpen(false)
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [open])

  return (
    <div className="assistant">
      {open && <ChatBox contextPage={contextPage} />}
      <button type="button" className="assistant-trigger" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        {open ? <X size={20} /> : <MessageCircle size={20} />}
        <span>{open ? 'Close' : 'Ask'}</span>
      </button>
    </div>
  )
}
