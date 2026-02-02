import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import {
  Squares2X2Icon,
  BoltIcon,
  ChatBubbleLeftRightIcon,
  ChartBarIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline';

const navigation = [
  { name: 'Pipelines', href: '/', icon: Squares2X2Icon },
  { name: 'Builder', href: '/builder', icon: BoltIcon },
  { name: 'Query', href: '/query', icon: ChatBubbleLeftRightIcon },
  { name: 'Metrics', href: '/metrics', icon: ChartBarIcon },
];

const Layout: React.FC = () => {
  return (
    <div className="flex h-screen bg-[#0a0a0b]">
      {/* Sidebar */}
      <aside className="w-64 bg-[#111113] border-r border-zinc-800 flex flex-col">
        {/* Logo */}
        <div className="h-16 flex items-center px-6 border-b border-zinc-800">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <BoltIcon className="w-5 h-5 text-white" />
            </div>
            <span className="text-lg font-semibold text-zinc-100">RAG OS</span>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1">
          {navigation.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-zinc-800 text-zinc-100'
                    : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'
                }`
              }
            >
              <item.icon className="w-5 h-5" />
              {item.name}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-zinc-800">
          <button className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200 w-full transition-colors">
            <Cog6ToothIcon className="w-5 h-5" />
            Settings
          </button>
          <div className="mt-3 px-3 text-xs text-zinc-600">
            Version 1.0.0
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;
