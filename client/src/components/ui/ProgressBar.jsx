import React from 'react';

export default function ProgressBar({ progress }) {
  const clamped = Math.max(0, Math.min(100, Number(progress) || 0));

  return (
    <div className="w-full">
      <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
        <span>Confidence</span>
        <span className="tabular-nums">{Math.round(clamped)}%</span>
      </div>
      <div className="mt-2 h-2 w-full rounded-full bg-slate-200 dark:bg-slate-800">
        <div
          className="h-2 rounded-full bg-indigo-600"
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}