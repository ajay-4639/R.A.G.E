import React, { useState, useEffect, useRef } from 'react';
import toast from 'react-hot-toast';
import {
  PaperAirplaneIcon,
  ChatBubbleLeftIcon,
  ClockIcon,
  DocumentTextIcon,
  ChevronDownIcon,
} from '@heroicons/react/24/outline';
import { apiClient } from '../api/client';
import { QueryResult } from '../types';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  sources?: QueryResult['sources'];
  executionTime?: number;
  timestamp: Date;
}

const QueryInterface: React.FC = () => {
  const [pipelines, setPipelines] = useState<string[]>([]);
  const [selectedPipeline, setSelectedPipeline] = useState<string>('');
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showPipelineDropdown, setShowPipelineDropdown] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadPipelines();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadPipelines = async () => {
    try {
      const list = await apiClient.listPipelines();
      setPipelines(list);
      if (list.length > 0 && !selectedPipeline) {
        setSelectedPipeline(list[0]);
      }
    } catch (error) {
      console.error('Failed to load pipelines:', error);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !selectedPipeline) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: query,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuery('');
    setIsLoading(true);

    try {
      const result = await apiClient.queryPipeline(selectedPipeline, query);
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: result.answer,
        sources: result.sources,
        executionTime: result.execution_time_ms,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to get response');
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, I encountered an error processing your request.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-[#0a0a0b]">
      {/* Header */}
      <div className="bg-[#111113] border-b border-zinc-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-zinc-100">Query Interface</h1>
            <p className="text-sm text-zinc-500 mt-0.5">Test your RAG pipelines</p>
          </div>

          {/* Pipeline Selector */}
          <div className="relative">
            <button
              onClick={() => setShowPipelineDropdown(!showPipelineDropdown)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg border border-zinc-700 bg-zinc-800/50 text-sm font-medium text-zinc-200 hover:bg-zinc-800 transition-colors"
            >
              {selectedPipeline || 'Select Pipeline'}
              <ChevronDownIcon className="w-4 h-4 text-zinc-500" />
            </button>

            {showPipelineDropdown && (
              <div className="absolute right-0 mt-2 w-64 rounded-lg bg-[#111113] border border-zinc-800 shadow-lg z-50 overflow-hidden">
                {pipelines.length === 0 ? (
                  <div className="p-4 text-center text-sm text-zinc-500">
                    No pipelines available
                  </div>
                ) : (
                  pipelines.map((pipeline) => (
                    <button
                      key={pipeline}
                      onClick={() => {
                        setSelectedPipeline(pipeline);
                        setShowPipelineDropdown(false);
                      }}
                      className={`w-full px-4 py-2.5 text-left text-sm hover:bg-zinc-800 transition-colors ${
                        selectedPipeline === pipeline
                          ? 'bg-blue-500/20 text-blue-400'
                          : 'text-zinc-300'
                      }`}
                    >
                      {pipeline}
                    </button>
                  ))
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-zinc-500">
            <div className="w-16 h-16 rounded-xl bg-zinc-800 flex items-center justify-center mb-4">
              <ChatBubbleLeftIcon className="w-8 h-8 text-zinc-600" />
            </div>
            <p className="font-medium text-zinc-400">Start a conversation</p>
            <p className="text-sm mt-1 text-zinc-600">
              {selectedPipeline
                ? `Ask questions using "${selectedPipeline}"`
                : 'Select a pipeline to begin'}
            </p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-2xl rounded-lg px-4 py-3 ${
                  message.type === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-[#111113] border border-zinc-800 text-zinc-200'
                }`}
              >
                <p className="whitespace-pre-wrap text-sm">{message.content}</p>

                {/* Sources */}
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-zinc-700">
                    <p className="text-xs font-medium text-zinc-400 mb-2 flex items-center gap-1">
                      <DocumentTextIcon className="w-3 h-3" />
                      Sources
                    </p>
                    <div className="space-y-2">
                      {message.sources.map((source, i) => (
                        <div
                          key={i}
                          className="text-xs bg-zinc-800/50 rounded p-2 border border-zinc-700"
                        >
                          <p className="text-zinc-400 line-clamp-2">
                            {source.content}
                          </p>
                          {source.score && (
                            <p className="text-zinc-500 mt-1">
                              Score: {(source.score * 100).toFixed(1)}%
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Execution Time */}
                {message.executionTime && (
                  <div className="mt-2 flex items-center gap-1 text-xs text-zinc-500">
                    <ClockIcon className="w-3 h-3" />
                    {message.executionTime.toFixed(0)}ms
                  </div>
                )}
              </div>
            </div>
          ))
        )}

        {/* Loading Indicator */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-[#111113] border border-zinc-800 rounded-lg px-4 py-3">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-zinc-500 rounded-full animate-pulse" />
                <div className="w-2 h-2 bg-zinc-500 rounded-full animate-pulse" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-zinc-500 rounded-full animate-pulse" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="bg-[#111113] border-t border-zinc-800 p-4">
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={
              selectedPipeline
                ? 'Type your question...'
                : 'Select a pipeline first...'
            }
            disabled={!selectedPipeline || isLoading}
            className="flex-1 px-4 py-2.5 rounded-lg border border-zinc-700 bg-zinc-800/50 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-zinc-900 disabled:text-zinc-600 transition-colors"
          />
          <button
            type="submit"
            disabled={!query.trim() || !selectedPipeline || isLoading}
            className="px-4 py-2.5 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            <PaperAirplaneIcon className="w-4 h-4" />
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

export default QueryInterface;
