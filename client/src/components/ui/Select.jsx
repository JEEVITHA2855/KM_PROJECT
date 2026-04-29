import React from 'react';
export default function Select({ className = "", children, ...props }) {
  return (
    <select
      className={`
        bg-white text-slate-900
        dark:bg-slate-800 dark:text-white
        
        border border-slate-300 dark:border-slate-700
        rounded-md px-3 py-2 text-sm
        
        focus:outline-none focus:ring-2 focus:ring-blue-500
        
        cursor-pointer
        
        ${className}
      `}
      {...props}
    >
      {children}
    </select>
  );
}