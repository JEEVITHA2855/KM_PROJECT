import React, { useEffect, useMemo, useState } from 'react';
import AlertResult from './AlertResult';
import AlertHistory from './AlertHistory';
import SemanticSearch from './SemanticSearch';
import Card from './ui/Card';
import Button from './ui/Button';
import Spinner from './ui/Spinner';
import Textarea from './ui/Textarea';
import { analyzeAlert } from '../services/api';
import { normalizeAnalysisResponse } from '../lib/analysis';

const Dashboard = () => {
  const [alertText, setAlertText] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState('');

  const [history, setHistory] = useState([]);
  const [severityFilter, setSeverityFilter] = useState('ALL');

  const canAnalyze = alertText.trim().length > 0 && !isAnalyzing;

  useEffect(() => {
    try {
      const raw = localStorage.getItem('alert_history_v1');
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) setHistory(parsed);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem('alert_history_v1', JSON.stringify(history.slice(0, 50)));
    } catch {
      // ignore
    }
  }, [history]);

  const historyItems = useMemo(() => history.slice(0, 50), [history]);

  async function handleAnalyze() {
    const text = alertText.trim();
    if (!text) return;

    setIsAnalyzing(true);
    setAnalysisError('');

    try {
      const data = await analyzeAlert(text);
      const normalized = normalizeAnalysisResponse(data, text);
      setAnalysis(normalized);

      setHistory((prev) => {
        const next = [
          {
            id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
            text,
            severity: normalized.severity,
            department: normalized.department,
            keywords: normalized.keywords,
            timestamp: new Date().toISOString(),
          },
          ...prev,
        ];
        return next.slice(0, 50);
      });
    } catch (e) {
      setAnalysis(null);
      setAnalysisError('Analysis failed. Ensure the API is running and reachable.');
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <div className="border-b border-slate-200/70 dark:border-slate-800 px-5 py-4">
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Main Dashboard</h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Paste an alert and get severity, routing, keywords, and similarity matches.
              </p>
            </div>

            <div className="px-5 py-4 space-y-3">
              <div>
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                    Alert text
                  </label>
                  <span className="text-xs text-slate-500 dark:text-slate-400">{alertText.length} chars</span>
                </div>
                <Textarea
                  rows={6}
                  value={alertText}
                  onChange={(e) => setAlertText(e.target.value)}
                  placeholder="Example: Gas sensor spike detected at Unit 4. Ventilation fan failed to start."
                  className="mt-2"
                />
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <Button onClick={handleAnalyze} disabled={!canAnalyze}>
                  {isAnalyzing ? (
                    <span className="inline-flex items-center gap-2">
                      <Spinner className="border-white/70" size={16} />
                      Analyzing
                    </span>
                  ) : (
                    'Analyze Alert'
                  )}
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => {
                    setAlertText('');
                    setAnalysis(null);
                    setAnalysisError('');
                  }}
                  disabled={isAnalyzing}
                >
                  Clear
                </Button>
              </div>

              {analysisError ? (
                <div className="rounded-lg border border-rose-200/70 dark:border-rose-900/60 bg-rose-50 dark:bg-rose-950/30 px-4 py-3 text-sm text-rose-700 dark:text-rose-200">
                  {analysisError}
                </div>
              ) : null}
            </div>
          </Card>

          {isAnalyzing ? (
            <Card>
              <div className="px-5 py-8 flex items-center justify-center gap-3 text-sm text-slate-600 dark:text-slate-300">
                <Spinner />
                Running model inference…
              </div>
            </Card>
          ) : null}

          {analysis && !isAnalyzing ? <AlertResult result={analysis} /> : null}
        </div>

        <div className="space-y-6">
          <SemanticSearch />
          <AlertHistory
            items={historyItems}
            severityFilter={severityFilter}
            onChangeSeverityFilter={setSeverityFilter}
          />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;