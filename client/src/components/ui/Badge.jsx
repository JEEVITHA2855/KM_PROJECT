import React from 'react';

const Badge = ({ children, severity }) => {
  const severityClasses = {
    CRITICAL: 'bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-200 border-rose-200/70 dark:border-rose-900/70',
    HIGH: 'bg-orange-50 text-orange-700 dark:bg-orange-950/60 dark:text-orange-200 border-orange-200/70 dark:border-orange-900/70',
    MEDIUM: 'bg-amber-50 text-amber-800 dark:bg-amber-950/60 dark:text-amber-200 border-amber-200/70 dark:border-amber-900/70',
    LOW: 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-200 border-emerald-200/70 dark:border-emerald-900/70',
  };

  const dotClasses = {
    CRITICAL: 'bg-rose-500',
    HIGH: 'bg-orange-500',
    MEDIUM: 'bg-amber-500',
    LOW: 'bg-emerald-500',
  };

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs font-semibold ${
        severityClasses[severity] || 'bg-slate-50 text-slate-700 dark:bg-slate-900/50 dark:text-slate-200 border-slate-200/70 dark:border-slate-800'
      }`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${dotClasses[severity] || 'bg-slate-400'}`} />
      {children}
    </span>
  );
};

export default Badge;