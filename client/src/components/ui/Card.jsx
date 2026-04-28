import React from 'react';

export default function Card({ children, className = '' }) {
  return (
    <div
      className={
        'rounded-xl border border-slate-200/70 dark:border-slate-800 bg-white dark:bg-slate-950/40 shadow-sm hover:shadow-md transition-shadow overflow-hidden ' +
        className
      }
    >
      {children}
    </div>
  );
}