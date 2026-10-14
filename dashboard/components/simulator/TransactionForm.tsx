'use client';

import { useState, useEffect } from 'react';
import type { TransactionInput } from '@/lib/types';
import { exampleTransactions } from '@/lib/mock-data';
import styles from './TransactionForm.module.css';

interface TransactionFormProps {
  onSubmit: (input: TransactionInput) => void;
  isLoading?: boolean;
}

// Must match services/ai-engine/app/schemas.py::PaymentFormat enum exactly
const PAYMENT_FORMATS = [
  { value: 'ACH',          label: 'ACH' },
  { value: 'Wire',         label: 'Wire Transfer' },
  { value: 'Cash',         label: 'Cash' },
  { value: 'Cheque',       label: 'Cheque' },
  { value: 'Credit Card',  label: 'Credit Card' },
  { value: 'Debit Card',   label: 'Debit Card' },
  { value: 'Bitcoin',      label: 'Bitcoin' },
  { value: 'Reinvestment', label: 'Reinvestment' },
  { value: 'Other',        label: 'Other' },
];

const DEFAULT_TRANSACTION: TransactionInput = {
  transaction_id: 'TXN-20220901-934281',
  from_account: '837B70300',
  to_account: '837B70350',
  amount: 5107.168000000001,
  timestamp: '2022-09-01T00:38:00+00:00',
  payment_format: 'ACH',
  currency: 'Mexican Peso',
};

function makeTransactionId(): string {
  const ts = new Date().toISOString().slice(0, 10).replace(/-/g, '');
  const rand = Math.random().toString(36).slice(2, 8).toUpperCase();
  return `TXN-${ts}-${rand}`;
}

export default function TransactionForm({ onSubmit, isLoading = false }: TransactionFormProps) {
  const [form, setForm] = useState<TransactionInput>(DEFAULT_TRANSACTION);
  const [jsonMode, setJsonMode] = useState(false);
  const [jsonText, setJsonText] = useState(JSON.stringify(DEFAULT_TRANSACTION, null, 2));
  const [jsonError, setJsonError] = useState('');

  function handleChange(field: keyof TransactionInput, value: string | number) {
    setForm(f => ({ ...f, [field]: value }));
  }

  function loadExample(key: keyof typeof exampleTransactions) {
    const ex = {
      ...exampleTransactions[key],
    };
    setForm(ex);
    setJsonText(JSON.stringify(ex, null, 2));
    setJsonError('');
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (jsonMode) {
      try {
        const parsed: TransactionInput = JSON.parse(jsonText);
        // Ensure required fields present
        if (!parsed.transaction_id) parsed.transaction_id = DEFAULT_TRANSACTION.transaction_id;
        if (!parsed.timestamp) parsed.timestamp = DEFAULT_TRANSACTION.timestamp;
        if (!parsed.currency) parsed.currency = DEFAULT_TRANSACTION.currency;
        onSubmit(parsed);
        setJsonError('');
      } catch {
        setJsonError('Invalid JSON — please check syntax');
      }
    } else {
      onSubmit({
        ...form,
        transaction_id: form.transaction_id || DEFAULT_TRANSACTION.transaction_id,
        timestamp: form.timestamp || DEFAULT_TRANSACTION.timestamp,
        currency: form.currency || DEFAULT_TRANSACTION.currency,
      });
    }
  }

  return (
    <form onSubmit={handleSubmit} className={styles.form} id="transaction-form">
      {/* Header */}
      <div className={styles.header}>
        <h3 className={styles.title}>Transaction Input</h3>
        <button
          type="button"
          className={`btn btn-ghost ${styles.modeToggle}`}
          onClick={() => setJsonMode(m => !m)}
          id="btn-toggle-json-mode"
        >
          {jsonMode ? '⊞ Form' : '{ } JSON'}
        </button>
      </div>



      {jsonMode ? (
        <div className={styles.jsonArea}>
          <label className="label">Raw JSON Payload</label>
          <textarea
            className={`input input-mono ${styles.jsonTextarea}`}
            value={jsonText}
            onChange={e => { setJsonText(e.target.value); setJsonError(''); }}
            rows={12}
            spellCheck={false}
            id="json-input"
          />
          {jsonError && <p className={styles.error}>{jsonError}</p>}
        </div>
      ) : (
        <div className={styles.fields}>
          <div className={styles.twoCol}>
            <div className={styles.fieldGroup}>
              <label className="label" htmlFor="from-account">From Account</label>
              <input id="from-account" className="input" value={form.from_account}
                onChange={e => handleChange('from_account', e.target.value)} required
                placeholder="e.g. ACC_MU4821" />
            </div>
            <div className={styles.fieldGroup}>
              <label className="label" htmlFor="to-account">To Account</label>
              <input id="to-account" className="input" value={form.to_account}
                onChange={e => handleChange('to_account', e.target.value)} required
                placeholder="e.g. ACC_HB1142" />
            </div>
          </div>
          <div className={styles.twoCol}>
            <div className={styles.fieldGroup}>
              <label className="label" htmlFor="amount">Amount</label>
              <input id="amount" className="input" type="number" min={0.01} step="any"
                value={form.amount}
                onChange={e => handleChange('amount', parseFloat(e.target.value) || 0)} required />
            </div>
            <div className={styles.fieldGroup}>
              <label className="label" htmlFor="payment-format">Payment Format</label>
              <select id="payment-format" className="input"
                value={form.payment_format}
                onChange={e => handleChange('payment_format', e.target.value)}>
                {PAYMENT_FORMATS.map(f => (
                  <option key={f.value} value={f.value}>{f.label}</option>
                ))}
              </select>
            </div>
          </div>
          <div className={styles.twoCol}>
            <div className={styles.fieldGroup}>
              <label className="label" htmlFor="currency">Currency</label>
              <input id="currency" className="input" value={form.currency}
                onChange={e => handleChange('currency', e.target.value)}
                placeholder="e.g. Mexican Peso" maxLength={30} />
            </div>
            <div className={styles.fieldGroup}>
              <label className="label" htmlFor="txn-id">Transaction ID</label>
              <input id="txn-id" className="input input-mono" value={form.transaction_id}
                onChange={e => handleChange('transaction_id', e.target.value)}
                placeholder="TXN-YYYYMMDD-XXXXXX" />
            </div>
          </div>
          <div className={styles.fieldGroup}>
            <label className="label" htmlFor="timestamp">Timestamp (ISO 8601)</label>
            <input id="timestamp" className="input input-mono" value={form.timestamp}
              onChange={e => handleChange('timestamp', e.target.value)}
              placeholder="YYYY-MM-DDTHH:mm:ssZ" />
          </div>

          {/* Schema note */}
          <p className={styles.schemaNote}>
            ℹ Payload sent to <span>AI Engine /score</span> + <span>Graph /graph-score</span> in parallel.
          </p>
        </div>
      )}

      <button
        type="submit"
        className={`btn btn-primary ${styles.submitBtn}`}
        disabled={isLoading}
        id="btn-score-transaction"
      >
        {isLoading ? (
          <><span className="spinner" style={{ width: 16, height: 16 }} /> Scoring…</>
        ) : (
          <>▶ Score Transaction</>
        )}
      </button>
    </form>
  );
}
