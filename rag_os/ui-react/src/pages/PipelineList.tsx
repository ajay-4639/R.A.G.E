import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import {
  TrashIcon,
  PlayIcon,
  DocumentDuplicateIcon,
  EyeIcon,
  FolderIcon,
  PlusIcon,
} from '@heroicons/react/24/outline';
import { apiClient } from '../api/client';
import { PipelineSpec } from '../types';

const PipelineList: React.FC = () => {
  const [pipelines, setPipelines] = useState<string[]>([]);
  const [selectedPipeline, setSelectedPipeline] = useState<string | null>(null);
  const [pipelineDetails, setPipelineDetails] = useState<PipelineSpec | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    loadPipelines();
  }, []);

  const loadPipelines = async () => {
    setIsLoading(true);
    try {
      const list = await apiClient.listPipelines();
      setPipelines(list);
    } catch (error) {
      toast.error('Failed to load pipelines');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectPipeline = async (name: string) => {
    setSelectedPipeline(name);
    try {
      const details = await apiClient.getPipeline(name);
      setPipelineDetails(details);
    } catch (error) {
      toast.error('Failed to load pipeline details');
    }
  };

  const handleDeletePipeline = async (name: string) => {
    if (!window.confirm(`Are you sure you want to delete "${name}"?`)) return;

    try {
      await apiClient.deletePipeline(name);
      toast.success(`Pipeline "${name}" deleted`);
      setPipelines((prev) => prev.filter((p) => p !== name));
      if (selectedPipeline === name) {
        setSelectedPipeline(null);
        setPipelineDetails(null);
      }
    } catch (error) {
      toast.error('Failed to delete pipeline');
    }
  };

  const handleCopyJson = () => {
    if (pipelineDetails) {
      navigator.clipboard.writeText(JSON.stringify(pipelineDetails, null, 2));
      toast.success('Pipeline JSON copied to clipboard');
    }
  };

  return (
    <div className="h-full flex flex-col bg-[#0a0a0b]">
      {/* Header */}
      <div className="bg-[#111113] border-b border-zinc-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-zinc-100">Saved Pipelines</h1>
            <p className="text-sm text-zinc-500 mt-0.5">
              {pipelines.length} pipeline{pipelines.length !== 1 ? 's' : ''} available
            </p>
          </div>
          <button
            onClick={() => navigate('/builder')}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 transition-colors"
          >
            <PlusIcon className="w-4 h-4" />
            New Pipeline
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Pipeline List */}
        <div className="w-80 border-r border-zinc-800 overflow-y-auto bg-[#111113]">
          {isLoading ? (
            <div className="p-4 space-y-3">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="h-14 rounded-lg bg-zinc-800 animate-pulse"
                />
              ))}
            </div>
          ) : pipelines.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-zinc-500 p-6">
              <div className="w-12 h-12 rounded-xl bg-zinc-800 flex items-center justify-center mb-3">
                <FolderIcon className="w-6 h-6 text-zinc-600" />
              </div>
              <p className="text-sm text-center text-zinc-500">No pipelines yet</p>
              <button
                onClick={() => navigate('/builder')}
                className="mt-4 px-4 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-sm transition-colors"
              >
                Create your first pipeline
              </button>
            </div>
          ) : (
            <div className="p-3 space-y-2">
              {pipelines.map((pipeline) => (
                <button
                  key={pipeline}
                  onClick={() => handleSelectPipeline(pipeline)}
                  className={`w-full p-4 rounded-lg border text-left transition-colors ${
                    selectedPipeline === pipeline
                      ? 'bg-blue-500/20 border-blue-500/50 text-blue-400'
                      : 'bg-zinc-800/50 border-zinc-700/50 text-zinc-200 hover:bg-zinc-800 hover:border-zinc-600'
                  }`}
                >
                  <div className="font-medium truncate text-sm">{pipeline}</div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Pipeline Details */}
        <div className="flex-1 overflow-y-auto">
          {selectedPipeline && pipelineDetails ? (
            <div className="p-6">
              {/* Header */}
              <div className="flex items-start justify-between mb-6">
                <div>
                  <h2 className="text-xl font-semibold text-zinc-100">
                    {pipelineDetails.name}
                  </h2>
                  {pipelineDetails.description && (
                    <p className="text-zinc-400 mt-1 text-sm">
                      {pipelineDetails.description}
                    </p>
                  )}
                  <p className="text-xs text-zinc-500 mt-2">
                    Version {pipelineDetails.version} • {pipelineDetails.steps.length} steps
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => navigate(`/query`)}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 transition-colors"
                  >
                    <PlayIcon className="w-4 h-4" />
                    Run
                  </button>
                  <button
                    onClick={handleCopyJson}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-800 text-zinc-300 text-sm hover:text-zinc-100 hover:bg-zinc-700 transition-colors"
                  >
                    <DocumentDuplicateIcon className="w-4 h-4" />
                    Copy JSON
                  </button>
                  <button
                    onClick={() => handleDeletePipeline(selectedPipeline)}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-800 text-red-400 text-sm hover:bg-red-500/10 transition-colors"
                  >
                    <TrashIcon className="w-4 h-4" />
                    Delete
                  </button>
                </div>
              </div>

              {/* Steps */}
              <div className="mb-6">
                <h3 className="text-sm font-medium text-zinc-300 mb-4">Pipeline Steps</h3>
                <div className="space-y-3">
                  {pipelineDetails.steps.map((step, index) => (
                    <div
                      key={step.step_id}
                      className="p-4 rounded-xl bg-[#111113] border border-zinc-800"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-zinc-800 flex items-center justify-center text-zinc-400 font-mono text-sm">
                          {index + 1}
                        </div>
                        <div>
                          <div className="font-medium text-zinc-100 text-sm">
                            {step.step_id}
                          </div>
                          <div className="text-xs text-zinc-500">
                            {step.step_type} • {step.step_class}
                          </div>
                        </div>
                      </div>
                      {Object.keys(step.config).length > 0 && (
                        <div className="mt-3 pt-3 border-t border-zinc-800">
                          <pre className="text-xs font-mono text-zinc-400 overflow-x-auto">
                            {JSON.stringify(step.config, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Raw JSON */}
              <div>
                <h3 className="text-sm font-medium text-zinc-300 mb-4">Raw JSON</h3>
                <pre className="p-4 rounded-xl bg-[#111113] border border-zinc-800 text-sm font-mono text-zinc-400 overflow-auto max-h-96">
                  {JSON.stringify(pipelineDetails, null, 2)}
                </pre>
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-zinc-500">
              <div className="w-12 h-12 rounded-xl bg-zinc-800 flex items-center justify-center mb-3">
                <EyeIcon className="w-6 h-6 text-zinc-600" />
              </div>
              <p className="text-sm">Select a pipeline to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PipelineList;
