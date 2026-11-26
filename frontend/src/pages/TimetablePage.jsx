import ErpLayout from '../components/ErpLayout'
import { Empty, Load, Panel } from '../components/kit'
import { WEEKDAYS } from '../lib/format'
import useApi from '../lib/useApi'
import { fetchTimetable } from '../services/api'

export default function TimetablePage() {
  const state = useApi(fetchTimetable)
  const today = new Date().toLocaleDateString('en-US', { weekday: 'long' })

  return (
    <ErpLayout title="Timetable" subtitle="Your weekly classes. Today's column is highlighted.">
      <Panel flush>
        <Load state={state}>
          {({ timetable }) => {
            if (!timetable.length) return <Empty title="No timetable published yet" />
            const days = WEEKDAYS.filter((day) => timetable.some((slot) => slot.day === day))
            const hours = [...new Set(timetable.map((slot) => slot.hour))].sort((a, b) => a - b)
            return (
              <div className="table-scroll">
                <table className="week">
                  <thead>
                    <tr>
                      <th scope="col">Hour</th>
                      {days.map((day) => <th key={day} scope="col" className={day === today ? 'is-today' : undefined}>{day}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {hours.map((hour) => (
                      <tr key={hour}>
                        <th scope="row">Hour {hour}</th>
                        {days.map((day) => (
                          <td key={day} className={day === today ? 'is-today' : undefined}>
                            {timetable.filter((s) => s.day === day && s.hour === hour).map((slot) => (
                              <div key={slot.code} className="slot">
                                <strong>{slot.subject}</strong>
                                <span>{slot.code}{slot.room ? ` · ${slot.room}` : ''}</span>
                                <span>{slot.faculty}</span>
                              </div>
                            ))}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          }}
        </Load>
      </Panel>
    </ErpLayout>
  )
}
