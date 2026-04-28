import React from 'react';

const Header = () => {
  return (
    <header className="sticky top-0 z-10 border-b border-slate-200/70 dark:border-slate-800 bg-white/80 dark:bg-slate-950/50 backdrop-blur">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between py-4">
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-slate-900 dark:text-slate-100">
              Alert Intelligence
            </h1>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              ML-based triage for operational alerts
            </p>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;