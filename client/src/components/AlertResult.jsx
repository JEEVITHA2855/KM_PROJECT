import React from 'react';
import Card from './ui/Card';
import Badge from './ui/Badge';
import ProgressBar from './ui/ProgressBar';
import Tag from './ui/Tag';

const AlertResult = ({ result }) => {
  const { severity, department, confidence, keywords, immediateAction, semanticSimilar } = result;

  const showImmediate = immediateAction?.required;
  const immediateMessage = immediateAction?.message;

  return (
    <Card>
      <div className="px-5 py-4 border-b border-slate-200/70 dark:border-slate-800">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Results</h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">Structured output from the ML model</p>
          </div>
          <div className="flex items-center gap-2">
            <Tag>{department}</Tag>
            <Badge severity={severity}>{severity}</Badge>
          </div>
        </div>
      </div>

      <div className="px-5 py-5 space-y-5">
        {showImmediate ? (
          <div className="rounded-lg border border-amber-200/70 dark:border-amber-900/60 bg-amber-50 dark:bg-amber-950/30 px-4 py-3">
            <div className="flex items-start gap-3">
              <div className="mt-0.5 h-2.5 w-2.5 rounded-full bg-amber-500" />
              <div>
                <div className="text-sm font-semibold text-amber-900 dark:text-amber-200">Immediate action required</div>
                {immediateMessage ? (
                  <div className="mt-1 text-sm text-amber-800 dark:text-amber-300">{immediateMessage}</div>
                ) : (
                  <div className="mt-1 text-sm text-amber-800 dark:text-amber-300">
                    Escalate and follow your operational runbook.
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="rounded-lg border border-slate-200/70 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/40 px-4 py-3">
            <div className="text-sm font-semibold text-slate-900 dark:text-slate-100">No immediate action flagged</div>
            <div className="mt-1 text-sm text-slate-600 dark:text-slate-300">Monitor and route to the owning team.</div>
          </div>
        )}

        <ProgressBar progress={confidence === null ? 0 : confidence * 100} />

        <div>
          <div className="flex items-center justify-between">
            <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">Keywords</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">{keywords.length ? `${keywords.length} extracted` : 'None'}</div>
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {keywords.length ? keywords.map((keyword) => <Tag key={keyword}>{keyword}</Tag>) : null}
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between">
            <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
              Semantic similarity
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400">Top matches</div>
          </div>

          {semanticSimilar?.length ? (
            <div className="mt-3 space-y-2">
              {semanticSimilar.map((r, idx) => (
                <div
                  key={`${idx}-${r.text}`}
                  className="rounded-lg border border-slate-200/70 dark:border-slate-800 bg-white/60 dark:bg-slate-900/30 px-4 py-3"
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
              ))}
            </div>
          ) : (
            <div className="mt-3 rounded-lg border border-dashed border-slate-200 dark:border-slate-800 px-4 py-6 text-sm text-slate-500 dark:text-slate-400">
              No similarity results returned.
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};

export default AlertResult;