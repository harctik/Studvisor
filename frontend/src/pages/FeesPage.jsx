import { useState } from 'react'
import ErpLayout from '../components/ErpLayout'
import { Badge, Empty, Load, Panel, Stats, Table, Tabs } from '../components/kit'
import { formatDate, money } from '../lib/format'
import useApi from '../lib/useApi'
import { fetchFeeSummary, fetchMyFees, fetchPaymentHistory } from '../services/api'

const tone = (status) => ({ paid: 'good', partial: 'warn', overdue: 'bad' }[String(status).toLowerCase()] || 'plain')

export default function FeesPage() {
  const [view, setView] = useState('fees')
  const summary = useApi(fetchFeeSummary)
  const fees = useApi(fetchMyFees)
  const payments = useApi(fetchPaymentHistory)

  return (
    <ErpLayout title="Fees" subtitle="What is due and what you have paid. Online payment isn't available here.">
      <Load state={summary}>
        {(s) => (
          <Stats items={[
            { label: 'Total fees', value: money(s.total_due) },
            { label: 'Paid', value: money(s.total_paid), tone: 'good' },
            { label: 'Pending', value: money(s.pending), tone: s.pending > 0 ? 'warn' : undefined },
            { label: 'Overdue items', value: s.overdue_count, tone: s.overdue_count ? 'bad' : undefined },
          ]} />
        )}
      </Load>

      <Tabs label="Fee view" value={view} onChange={setView} tabs={[['fees', 'Fees'], ['payments', 'Payments']]} />

      {view === 'fees' ? (
        <Panel flush>
          <Load state={fees}>
            {({ fees: list }) => (
              <Table
                rows={list}
                rowKey={(f) => f.id}
                empty={<Empty title="No fees assigned" />}
                columns={[
                  { label: 'Fee', cell: (f) => f.name },
                  { label: 'Due date', cell: (f) => formatDate(f.due_date) },
                  { label: 'Amount', numeric: true, cell: (f) => money(f.due) },
                  { label: 'Paid', numeric: true, cell: (f) => money(f.paid) },
                  { label: 'Remaining', numeric: true, cell: (f) => money(f.remaining) },
                  { label: 'Status', cell: (f) => <Badge tone={tone(f.status)}>{f.status}</Badge> },
                ]}
              />
            )}
          </Load>
        </Panel>
      ) : (
        <Panel flush>
          <Load state={payments}>
            {({ payments: list }) => (
              <Table
                rows={list}
                rowKey={(p) => p.id}
                empty={<Empty title="No payments recorded" />}
                columns={[
                  { label: 'Date', cell: (p) => formatDate(p.date) },
                  { label: 'Method', cell: (p) => p.method || '—' },
                  { label: 'Reference', cell: (p) => p.txn_id || '—' },
                  { label: 'Amount', numeric: true, cell: (p) => money(p.amount) },
                ]}
              />
            )}
          </Load>
        </Panel>
      )}
    </ErpLayout>
  )
}
