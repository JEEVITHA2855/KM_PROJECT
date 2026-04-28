import React from 'react';

const Tag = ({ children }) => (
  <span className="inline-flex items-center rounded-full border border-slate-200/70 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/40 px-2.5 py-1 text-xs font-semibold text-slate-700 dark:text-slate-200">
    {children}
  </span>
);

export default Tag;