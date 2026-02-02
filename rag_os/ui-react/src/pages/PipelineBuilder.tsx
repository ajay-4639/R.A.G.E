import React, { useState } from 'react';
import toast from 'react-hot-toast';
import {
  ArrowDownTrayIcon,
  CloudArrowUpIcon,
  TrashIcon,
  CodeBracketIcon,
} from '@heroicons/react/24/outline';
import StepLibrary from '../components/StepLibrary';
import PipelineCanvas from '../components/PipelineCanvas';
import StepConfig from '../components/StepConfig';
import { usePipelineStore } from '../store/pipelineStore';
import { apiClient } from '../api/client';
import { PipelineSpec } from '../types';

const PipelineBuilder: React.FC = () => {
  const {
    steps,
    pipelineName,
    pipelineDescription,
    setPipelineDescription,
    clearPipeline,
    isDirty,
    setDirty,
  } = usePipelineStore();
  const [showJson, setShowJson] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const generatePipelineSpec = (): PipelineSpec => {
    return {
      name: pipelineName.toLowerCase().replace(/\s+/g, '-'),
      version: '1.0.0',
      description: pipelineDescription,
      steps: steps.map((step, index) => ({
        step_id: `step_${index + 1}_${step.stepType}`,
        step_type: step.stepType,
        step_class: `rag_os.steps.${step.stepType}.${step.implementation}`,
        config: step.config,
      })),
    };
  };

  const handleSave = async () => {
    if (steps.length === 0) {
      toast.error('Add at least one step to save the pipeline');
      return;
    }

    setIsSaving(true);
    try {
      const spec = generatePipelineSpec();
      await apiClient.createPipeline(spec);
      toast.success(`Pipeline "${pipelineName}" saved successfully!`);
      setDirty(false);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to save pipeline');
    } finally {
      setIsSaving(false);
    }
  };

  const handleDownload = () => {
    const spec = generatePipelineSpec();
    const blob = new Blob([JSON.stringify(spec, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${spec.name}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Pipeline downloaded!');
  };

  const handleClear = () => {
    if (steps.length > 0 && isDirty) {
      if (window.confirm('Are you sure you want to clear the pipeline? Unsaved changes will be lost.')) {
        clearPipeline();
        toast.success('Pipeline cleared');
      }
    } else {
      clearPipeline();
    }
  };

  return (
    <div className="h-full flex flex-col bg-[#0a0a0b]">
      {/* Toolbar */}
      <div className="bg-[#111113] border-b border-zinc-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-lg font-semibold text-zinc-100">Pipeline Builder</h1>
            {isDirty && (
              <span className="px-2 py-0.5 text-xs font-medium bg-amber-500/20 text-amber-400 rounded-full">
                Unsaved
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowJson(!showJson)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                showJson
                  ? 'bg-blue-500/20 text-blue-400'
                  : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
              }`}
            >
              <CodeBracketIcon className="w-4 h-4" />
              JSON
            </button>

            <button
              onClick={handleClear}
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-zinc-400 hover:bg-zinc-800 hover:text-red-400 transition-colors"
            >
              <TrashIcon className="w-4 h-4" />
              Clear
            </button>

            <button
              onClick={handleDownload}
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition-colors"
            >
              <ArrowDownTrayIcon className="w-4 h-4" />
              Download
            </button>

            <button
              onClick={handleSave}
              disabled={isSaving || steps.length === 0}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isSaving ? (
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <CloudArrowUpIcon className="w-4 h-4" />
              )}
              Save
            </button>
          </div>
        </div>

        {/* Description Input */}
        <div className="mt-3">
          <input
            type="text"
            value={pipelineDescription}
            onChange={(e) => setPipelineDescription(e.target.value)}
            placeholder="Add a description for your pipeline..."
            className="w-full px-3 py-2 rounded-lg border border-zinc-700 bg-zinc-800/50 text-sm text-zinc-200 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Step Library */}
        <div className="w-72 border-r border-zinc-800 flex-shrink-0">
          <StepLibrary />
        </div>

        {/* Pipeline Canvas */}
        <div className="flex-1">
          {showJson ? (
            <div className="h-full p-6 overflow-auto bg-[#0a0a0b]">
              <pre className="p-4 rounded-lg bg-[#111113] border border-zinc-800 text-sm font-mono text-zinc-300 overflow-auto">
                {JSON.stringify(generatePipelineSpec(), null, 2)}
              </pre>
            </div>
          ) : (
            <PipelineCanvas />
          )}
        </div>

        {/* Step Config */}
        <div className="w-80 border-l border-zinc-800 flex-shrink-0">
          <StepConfig />
        </div>
      </div>
    </div>
  );
};

export default PipelineBuilder;
