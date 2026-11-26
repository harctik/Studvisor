import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { SendHorizontal, Trash2 } from 'lucide-react'
import {
  clearChatHistory, errorMessage, fetchChatHistory, fetchChatSuggestions, sendChatMessage,
  streamChatMessage,
} from '../services/api'

const links = {
  a: ({ node, children, ...props }) => <a {...props} target="_blank" rel="noreferrer">{children}</a>,
}

export default function ChatBox({ contextPage = 'dashboard' }) {
  const [messages, setMessages] = useState([])
  const [suggestions, setSuggestions] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [problem, setProblem] = useState(null)
  const logRef = useRef(null)

  useEffect(() => {
    let active = true
    fetchChatHistory()
      .then(({ messages: history }) => {
        const earlier = history.slice(0, 10).reverse().flatMap((item) => [
          { from: 'you', text: item.query },
          { from: 'bot', text: item.response },
        ])
        if (active) setMessages((current) => [...earlier, ...current])
      })
      .catch((error) => active && setProblem(errorMessage(error)))
    fetchChatSuggestions()
      .then((data) => active && setSuggestions(data.suggestions))
      .catch(() => {}) // suggestions are optional; typing still works without them
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    const log = logRef.current
    if (log) log.scrollTop = log.scrollHeight
  }, [messages, problem])

  async function ask(question) {
    const query = question.trim()
    if (!query || sending) return
    setInput('')
    setProblem(null)
    setSending(true)
    setMessages((current) => [...current, { from: 'you', text: query }, { from: 'bot', text: '' }])
    const answer = (text) => setMessages((current) => [...current.slice(0, -1), { from: 'bot', text }])

    try {
      const text = await streamChatMessage(query, { contextPage, onText: answer })
      if (!text) answer((await sendChatMessage(query, contextPage)).response)
    } catch (error) {
      setMessages((current) => current.slice(0, -2))
      setProblem(error.response ? errorMessage(error) : "The assistant didn't answer. Please try again.")
      setInput(query)
    } finally {
      setSending(false)
    }
  }

  async function clear() {
    try {
      await clearChatHistory()
      setMessages([])
      setProblem(null)
    } catch (error) {
      setProblem(errorMessage(error))
    }
  }

  return (
    <section className="chat" aria-label="Study assistant">
      <header>
        <strong>Assistant</strong>
        <button type="button" className="icon-btn" onClick={clear} disabled={sending} aria-label="Clear chat history">
          <Trash2 size={16} />
        </button>
      </header>

      <div className="chat-log" ref={logRef} aria-live="polite">
        {!messages.length && <p className="chat-hint">Ask about your attendance, marks or exam eligibility.</p>}
        {messages.map((message, index) => (
          <div key={index} className={`bubble bubble-${message.from}`}>
            {message.from === 'bot'
              ? (message.text ? <ReactMarkdown remarkPlugins={[remarkGfm]} components={links}>{message.text}</ReactMarkdown> : 'Looking that up…')
              : message.text}
          </div>
        ))}
        {problem && <p className="notice notice-bad" role="alert">{problem}</p>}
      </div>

      {suggestions.length > 0 && (
        <div className="chips">
          {suggestions.slice(0, 4).map((text) => (
            <button key={text} type="button" onClick={() => ask(text)} disabled={sending}>{text}</button>
          ))}
        </div>
      )}

      <form onSubmit={(event) => { event.preventDefault(); ask(input) }}>
        <input value={input} onChange={(event) => setInput(event.target.value)} placeholder="Type a question" aria-label="Your question" />
        <button type="submit" className="btn btn-primary" disabled={sending || !input.trim()} aria-label="Send">
          <SendHorizontal size={17} />
        </button>
      </form>
    </section>
  )
}
