import React, { useMemo, useState } from 'react';
import Card from './ui/Card';
import Button from './ui/Button';
import Input from './ui/Input';
import Spinner from './ui/Spinner';
import { analyzeAlert } from '../services/api';
import { normalizeAnalysisResponse } from '../lib/analysis';

export default function SemanticSearch() {
  const [query, setQuery] = useState('');
  const [items, setItems] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState('');

  const canSearch = query.trim().length >= 6 && !isSearching;

  const sortedItems = useMemo(() => {
    return [...items].sort((a, b) => b.similarity - a.similarity);
  }, [items]);

  async function onSearch() {
    const text = query.trim();
    if (text.length < 6) return;

    setIsSearching(true);
    setError('');
    setItems([]);

    try {
      const data = await analyzeAlert(text);
      const normalized = normalizeAnalysisResponse(data, text);
      setItems(normalized.semanticSimilar);
    } catch (e) {
      setError('Semantic search failed. Check the API and try again.');
    } finally {
      setIsSearching(false);
    }
  }

  return (
    <Card>
      <div className="border-b border-slate-200/70 dark:border-slate-800 px-5 py-4">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Semantic Search</h3>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Find similar historical alerts by meaning (uses the same `POST /analyze` response’s similarity results).
        </p>
      </div>

      <div className="px-5 py-4 space-y-3">
        <div className="flex gap-2">
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search similar alerts…"
            aria-label="Semantic search query"
          />
          <Button onClick={onSearch} disabled={!canSearch}>
            {isSearching ? (
              <span className="inline-flex items-center gap-2">
                <Spinner className="border-white/70" size={16} />
                Searching
              </span>
            ) : (
              'Search'
            )}
          </Button>
        </div>

        {error ? <div className="text-sm text-rose-600 dark:text-rose-400">{error}</div> : null}

        <div className="space-y-2">
          {isSearching ? (
            <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/40 px-4 py-3 text-sm text-slate-600 dark:text-slate-300">
              Retrieving similarity results…
            </div>
          ) : sortedItems.length === 0 ? (
            <div className="rounded-lg border border-dashed border-slate-200 dark:border-slate-800 px-4 py-6 text-sm text-slate-500 dark:text-slate-400">
              Enter a query to see similar alerts.
            </div>
          ) : (
            sortedItems.map((r, idx) => (
              <div
                key={`${idx}-${r.text}`}
                className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white/60 dark:bg-slate-900/30 px-4 py-3 hover:bg-white dark:hover:bg-slate-900/50 transition-colors"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="text-sm text-slate-900 dark:text-slate-100">{r.text}</div>
                  <div className="shrink-0 rounded-full bg-slate-100 dark:bg-slate-800 px-2 py-1 text-xs font-medium text-slate-700 dark:text-slate-200">
                    {(r.similarity * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="mt-2 h-1.5 w-full rounded-full bg-slate-200 dark:bg-slate-800">
                  <div
                    className="h-1.5 rounded-full bg-indigo-600"
                    style={{ width: `${Math.round(r.similarity * 100)}%` }}
                  />
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </Card>
  );
}
