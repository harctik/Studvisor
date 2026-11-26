import ErpLayout from '../components/ErpLayout'
import { Empty, Load, Meter, Panel, useAction } from '../components/kit'
import { plural } from '../lib/format'
import useApi from '../lib/useApi'
import { fetchPolls, votePoll } from '../services/api'

export default function PollsPage() {
  const state = useApi(fetchPolls)
  const { busy, run, banner } = useAction(state.reload)

  return (
    <ErpLayout title="Polls" subtitle="One vote per poll.">
      {banner}
      <Load state={state}>
        {({ polls }) => (polls.length ? (
          <div className="cards">
            {polls.map((poll) => {
              const total = poll.options.reduce((sum, o) => sum + o.votes, 0)
              return (
                <article key={poll.id} className="card">
                  <h2>{poll.question}</h2>
                  <div className="vote">
                    {poll.options.map((option) => (
                      <div key={option.id} className="vote-option">
                        <div>
                          <span>{option.text}</span>
                          <button type="button" className="btn btn-small" disabled={busy === poll.id} aria-label={`Vote for ${option.text}`}
                            onClick={() => run(poll.id, () => votePoll(poll.id, option.id), 'Vote counted.')}>
                            Vote
                          </button>
                        </div>
                        <Meter percent={total ? (option.votes / total) * 100 : 0} />
                        <span className="meta">{plural(option.votes, 'vote')}</span>
                      </div>
                    ))}
                  </div>
                  <p className="meta">{plural(total, 'vote')} in total</p>
                </article>
              )
            })}
          </div>
        ) : <Panel><Empty title="No polls are open" /></Panel>)}
      </Load>
    </ErpLayout>
  )
}
