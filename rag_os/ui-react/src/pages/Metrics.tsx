import React, { useState, useEffect } from 'react';
import {
  ChartBarIcon,
  ClockIcon,
  CpuChipIcon,
  CircleStackIcon,
  ArrowPathIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline';
import { apiClient } from '../api/client';

interface MetricCard {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ElementType;
  color: string;
}

const Metrics: React.FC = () => {
  const [metrics, setMetrics] = useState<Record<string, unknown>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [apiStatus, setApiStatus] = useState<'healthy' | 'unhealthy' | 'unknown'>('unknown');
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  useEffect(() => {
    loadMetrics();
    const interval = setInterval(loadMetrics, 120000);
    return () => clearInterval(interval);
  }, []);

  const loadMetrics = async () => {
    try {
      const health = await apiClient.healthCheck();
      setApiStatus(health.status === 'healthy' ? 'healthy' : 'unhealthy');

      try {
        const data = await apiClient.getMetrics();
        setMetrics(data);
      } catch {
        setMetrics({});
      }

      setLastUpdated(new Date());
    } catch (error) {
      setApiStatus('unhealthy');
    } finally {
      setIsLoading(false);
    }
  };

  const cards: MetricCard[] = [
    {
      title: 'API Status',
      value: apiStatus === 'healthy' ? 'Healthy' : 'Unhealthy',
      subtitle: 'Backend connection',
      icon: apiStatus === 'healthy' ? CheckCircleIcon : ExclamationCircleIcon,
      color: apiStatus === 'healthy' ? 'emerald' : 'red',
    },
    {
      title: 'Total Queries',
      value: (metrics.total_queries as number) || 0,
      subtitle: 'All time',
      icon: ChartBarIcon,
      color: 'blue',
    },
    {
      title: 'Avg Response Time',
      value: metrics.avg_response_time ? `${(metrics.avg_response_time as number).toFixed(0)}ms` : 'N/A',
      subtitle: 'Last 24 hours',
      icon: ClockIcon,
      color: 'amber',
    },
    {
      title: 'Active Pipelines',
      value: (metrics.active_pipelines as number) || 0,
      subtitle: 'Currently loaded',
      icon: CpuChipIcon,
      color: 'violet',
    },
    {
      title: 'Documents Indexed',
      value: (metrics.documents_indexed as number) || 0,
      subtitle: 'Total in vector store',
      icon: CircleStackIcon,
      color: 'cyan',
    },
    {
      title: 'Cache Hit Rate',
      value: metrics.cache_hit_rate ? `${((metrics.cache_hit_rate as number) * 100).toFixed(1)}%` : 'N/A',
      subtitle: 'Response cache',
      icon: ArrowPathIcon,
      color: 'pink',
    },
  ];

  const colorClasses: Record<string, { bg: string; border: string; text: string; icon: string }> = {
    emerald: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', icon: 'text-emerald-500' },
    red: { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', icon: 'text-red-500' },
    blue: { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400', icon: 'text-blue-500' },
    amber: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', icon: 'text-amber-500' },
    violet: { bg: 'bg-violet-500/10', border: 'border-violet-500/30', text: 'text-violet-400', icon: 'text-violet-500' },
    cyan: { bg: 'bg-cyan-500/10', border: 'border-cyan-500/30', text: 'text-cyan-400', icon: 'text-cyan-500' },
    pink: { bg: 'bg-pink-500/10', border: 'border-pink-500/30', text: 'text-pink-400', icon: 'text-pink-500' },
  };

  return (
    <div className="h-full flex flex-col bg-[#0a0a0b]">
      {/* Header */}
      <div className="bg-[#111113] border-b border-zinc-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-zinc-100">System Metrics</h1>
            <p className="text-sm text-zinc-500 mt-0.5">
              {lastUpdated
                ? `Last updated: ${lastUpdated.toLocaleTimeString()}`
                : 'Loading...'}
            </p>
          </div>
          <button
            onClick={loadMetrics}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-zinc-800 text-zinc-300 text-sm font-medium hover:text-zinc-100 hover:bg-zinc-700 disabled:opacity-50 transition-colors"
          >
            <ArrowPathIcon className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Metric Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          {cards.map((card) => {
            const colors = colorClasses[card.color];
            return (
              <div
                key={card.title}
                className={`p-5 rounded-xl border ${colors.bg} ${colors.border}`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <p className={`text-sm font-medium ${colors.text} opacity-80`}>{card.title}</p>
                    <p className={`text-2xl font-bold mt-1 ${colors.text}`}>{card.value}</p>
                    {card.subtitle && (
                      <p className="text-xs text-zinc-500 mt-1">{card.subtitle}</p>
                    )}
                  </div>
                  <card.icon className={`w-7 h-7 ${colors.icon} opacity-50`} />
                </div>
              </div>
            );
          })}
        </div>

        {/* System Info */}
        <div className="rounded-xl bg-[#111113] border border-zinc-800 p-5 mb-6">
          <h2 className="text-sm font-medium text-zinc-300 mb-4">System Information</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-zinc-500">Version</p>
              <p className="text-zinc-200 font-mono text-sm">0.1.0</p>
            </div>
            <div>
              <p className="text-xs text-zinc-500">Environment</p>
              <p className="text-zinc-200 text-sm">Production</p>
            </div>
            <div>
              <p className="text-xs text-zinc-500">API Endpoint</p>
              <p className="text-zinc-200 font-mono text-sm truncate">
                {process.env.REACT_APP_API_URL || 'http://localhost:8000'}
              </p>
            </div>
            <div>
              <p className="text-xs text-zinc-500">Uptime</p>
              <p className="text-zinc-200 text-sm">
                {metrics.uptime_seconds
                  ? `${Math.floor((metrics.uptime_seconds as number) / 3600)}h ${Math.floor(
                      ((metrics.uptime_seconds as number) % 3600) / 60
                    )}m`
                  : 'N/A'}
              </p>
            </div>
          </div>
        </div>

        {/* Raw Metrics */}
        {Object.keys(metrics).length > 0 && (
          <div className="rounded-xl bg-[#111113] border border-zinc-800 p-5">
            <h2 className="text-sm font-medium text-zinc-300 mb-4">Raw Metrics</h2>
            <pre className="text-xs font-mono text-zinc-400 overflow-auto max-h-64">
              {JSON.stringify(metrics, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default Metrics;
