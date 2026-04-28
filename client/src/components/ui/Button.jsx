import React from 'react';

export default function Button({ className = '', variant = 'primary', ...props }) {
  const base =
    'inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500/40 disabled:opacity-50 disabled:cursor-not-allowed';

  const variants = {
    primary: 'bg-indigo-600 text-white hover:bg-indigo-700',
    secondary:
      'bg-white dark:bg-slate-900/40 text-slate-900 dark:text-slate-100 border border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-900/60',
    danger: 'bg-rose-600 text-white hover:bg-rose-700',
  };

  return <button className={`${base} ${variants[variant] || variants.primary} ${className}`} {...props} />;
}
