import React, { useMemo } from 'react';
import Badge from './ui/Badge';
import Tag from './ui/Tag';
import Card from './ui/Card';
import Select from './ui/Select';
import { formatTimestamp } from '../lib/analysis';

const SEVERITY_OPTIONS = ['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];

export default function AlertHistory({ items, severityFilter, onChangeSeverityFilter }) {
  const filtered = useMemo(() => {
    if (severityFilter === 'ALL') return items;
    return items.filter((i) => i.severity === severityFilter);
  }, [items, severityFilter]);

  return (
    <Card>
      <div className="flex items-center justify-between gap-3 border-b border-slate-200/70 dark:border-slate-800 px-5 py-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Alert History</h3>
          <p className="text-xs text-slate-500 dark:text-slate-400">Recent analyses in this session</p>
        </div>
        <Select
          value={severityFilter}
          onChange={(e) => onChangeSeverityFilter(e.target.value)}
          aria-label="Filter by severity"
        >
          {SEVERITY_OPTIONS.map((opt) => (
            <option key={opt} value={opt}>
              {opt === 'ALL' ? 'All severities' : opt}
            </option>
          ))}
        </Select>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-blue-50 dark:bg-slate-900/40 text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
            <tr>
              <th className="px-5 py-3 font-medium">Alert</th>
              <th className="px-5 py-3 font-medium">Severity</th>
              <th className="px-5 py-3 font-medium">Department</th>
              <th className="px-5 py-3 font-medium">Timestamp</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200/70 dark:divide-slate-800">
            {filtered.length === 0 ? (
              <tr>
                <td className="px-5 py-6 text-slate-500 dark:text-slate-400" colSpan={4}>
                  No alerts match this filter.
                </td>
              </tr>
            ) : (
              filtered.map((row) => (
                <tr key={row.id} className="hover:bg-slate-50/60 dark:hover:bg-slate-900/40 transition-colors">
                  <td className="px-5 py-4 max-w-[420px]">
                    <div className="truncate text-slate-900 dark:text-slate-100">{row.text}</div>
                    {row.keywords?.length ? (
                      <div className="mt-1 flex flex-wrap gap-1">
                        {row.keywords.slice(0, 3).map((k) => (
                          <span
                            key={k}
                            className="rounded-full bg-slate-100 dark:bg-slate-800 px-2 py-0.5 text-[11px] text-slate-600 dark:text-slate-300"
                          >
                            {k}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </td>
                  <td className="px-5 py-4">
                    <Badge severity={row.severity}>{row.severity}</Badge>
                  </td>
                  <td className="px-5 py-4">
                    <Tag>{row.department}</Tag>
                  </td>
                  <td className="px-5 py-4 text-slate-600 dark:text-slate-300 whitespace-nowrap">
                    {formatTimestamp(new Date(row.timestamp))}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
