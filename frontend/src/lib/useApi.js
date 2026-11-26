import { useCallback, useEffect, useRef, useState } from 'react'
import { errorMessage } from '../services/api'

/** Load data on mount. `reload` refetches without blanking what is on screen. */
export default function useApi(load) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  const loadRef = useRef(load)
  loadRef.current = load

  const reload = useCallback(async () => {
    try {
      const data = await loadRef.current()
      setState({ data, error: null, loading: false })
    } catch (error) {
      setState((previous) => ({ data: previous.data, error: errorMessage(error), loading: false }))
    }
  }, [])

  useEffect(() => {
    reload()
  }, [reload])

  return { ...state, reload }
}
