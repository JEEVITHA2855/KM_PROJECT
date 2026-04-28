import React from 'react';

export default function Select({ className = '', children, ...props }) {
  return (
    <select
      className={
        'rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 px-3 py-2 text-sm text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/30 ' +
        className
      }
      {...props}
    >
      {children}
    </select>
  );
}
