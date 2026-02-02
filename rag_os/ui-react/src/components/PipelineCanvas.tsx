import React, { useCallback } from 'react';
import {
  TrashIcon,
  Cog6ToothIcon,
  PlusIcon,
} from '@heroicons/react/24/outline';
import { usePipelineStore } from '../store/pipelineStore';
import { STEP_TYPES } from '../data/stepTypes';
import { PipelineStep } from '../types';

interface NodeProps {
  step: PipelineStep;
  index: number;
  isSelected: boolean;
  isLast: boolean;
}

const stepColors: Record<string, { bg: string; border: string; text: string }> = {
  ingestion: { bg: 'bg-blue-500/10', border: 'border-blue-500/50', text: 'text-blue-400' },
  chunking: { bg: 'bg-amber-500/10', border: 'border-amber-500/50', text: 'text-amber-400' },
  embedding: { bg: 'bg-violet-500/10', border: 'border-violet-500/50', text: 'text-violet-400' },
  retrieval: { bg: 'bg-cyan-500/10', border: 'border-cyan-500/50', text: 'text-cyan-400' },
  reranking: { bg: 'bg-pink-500/10', border: 'border-pink-500/50', text: 'text-pink-400' },
  prompt_assembly: { bg: 'bg-teal-500/10', border: 'border-teal-500/50', text: 'text-teal-400' },
  llm_execution: { bg: 'bg-orange-500/10', border: 'border-orange-500/50', text: 'text-orange-400' },
  post_processing: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/50', text: 'text-emerald-400' },
};

const FlowNode: React.FC<NodeProps> = ({ step, index, isSelected, isLast }) => {
  const { selectStep, removeStep } = usePipelineStore();
  const stepInfo = STEP_TYPES[step.stepType];
  const colors = stepColors[step.stepType] || { bg: 'bg-zinc-500/10', border: 'border-zinc-500/50', text: 'text-zinc-400' };

  return (
    <div className="flex flex-col items-center">
      {/* Node */}
      <div
        onClick={() => selectStep(step.id)}
        className={`
          relative w-64 rounded-xl border-2 cursor-pointer transition-all
          ${colors.bg} ${colors.border}
          ${isSelected ? 'ring-2 ring-blue-500 ring-offset-2 ring-offset-[#0a0a0b]' : 'hover:border-opacity-100'}
        `}
      >
        {/* Header */}
        <div className={`px-4 py-3 border-b ${colors.border} flex items-center gap-3`}>
          <span className="text-2xl">{stepInfo?.icon}</span>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-zinc-100 truncate">{step.name}</div>
            <div className={`text-xs ${colors.text}`}>{stepInfo?.name}</div>
          </div>
        </div>

        {/* Body - Config preview */}
        <div className="px-4 py-3">
          {Object.keys(step.config).length > 0 ? (
            <div className="space-y-1">
              {Object.entries(step.config).slice(0, 2).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between text-xs">
                  <span className="text-zinc-500 truncate">{key}</span>
                  <span className="text-zinc-400 truncate ml-2">
                    {typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value).slice(0, 12)}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-zinc-600 text-center py-1">No configuration</div>
          )}
        </div>

        {/* Actions - show on hover */}
        <div className="absolute -right-2 top-1/2 -translate-y-1/2 flex flex-col gap-1 opacity-0 hover:opacity-100 transition-opacity">
          <button
            onClick={(e) => {
              e.stopPropagation();
              selectStep(step.id);
            }}
            className="p-1.5 rounded-lg bg-zinc-800 border border-zinc-700 hover:bg-zinc-700 transition-colors"
          >
            <Cog6ToothIcon className="w-3.5 h-3.5 text-zinc-400" />
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              removeStep(step.id);
            }}
            className="p-1.5 rounded-lg bg-zinc-800 border border-zinc-700 hover:bg-red-900/50 hover:border-red-500/50 transition-colors"
          >
            <TrashIcon className="w-3.5 h-3.5 text-zinc-400 hover:text-red-400" />
          </button>
        </div>

        {/* Connection points */}
        {index > 0 && (
          <div className="absolute -top-1.5 left-1/2 -translate-x-1/2 w-3 h-3 rounded-full bg-zinc-700 border-2 border-zinc-600" />
        )}
        {!isLast && (
          <div className="absolute -bottom-1.5 left-1/2 -translate-x-1/2 w-3 h-3 rounded-full bg-zinc-700 border-2 border-zinc-600" />
        )}
      </div>

      {/* Connector line */}
      {!isLast && (
        <div className="flex flex-col items-center py-2">
          <div className="w-0.5 h-8 bg-gradient-to-b from-zinc-600 to-zinc-700" />
          <svg className="w-3 h-2 text-zinc-600" viewBox="0 0 12 8" fill="currentColor">
            <path d="M6 8L0 0H12L6 8Z" />
          </svg>
        </div>
      )}
    </div>
  );
};

const PipelineCanvas: React.FC = () => {
  const { steps, selectedStepId, pipelineName, setPipelineName } = usePipelineStore();

  return (
    <div className="h-full flex flex-col bg-[#0a0a0b]">
      {/* Header */}
      <div className="px-6 py-4 bg-[#111113] border-b border-zinc-800">
        <input
          type="text"
          value={pipelineName}
          onChange={(e) => setPipelineName(e.target.value)}
          className="text-lg font-semibold bg-transparent border-none outline-none text-zinc-100 w-full placeholder-zinc-600"
          placeholder="Pipeline Name..."
        />
        <p className="text-sm text-zinc-500 mt-1">
          {steps.length} step{steps.length !== 1 ? 's' : ''} in pipeline
        </p>
      </div>

      {/* Canvas with grid background */}
      <div
        className="flex-1 overflow-auto"
        style={{
          backgroundImage: `
            linear-gradient(rgba(39, 39, 42, 0.3) 1px, transparent 1px),
            linear-gradient(90deg, rgba(39, 39, 42, 0.3) 1px, transparent 1px)
          `,
          backgroundSize: '20px 20px',
        }}
      >
        {steps.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-zinc-500">
            <div className="w-20 h-20 rounded-2xl bg-zinc-800/50 border-2 border-dashed border-zinc-700 flex items-center justify-center mb-4">
              <PlusIcon className="w-8 h-8 text-zinc-600" />
            </div>
            <p className="font-medium text-zinc-400">No steps yet</p>
            <p className="text-sm mt-1 text-zinc-600">Add steps from the library to build your pipeline</p>
          </div>
        ) : (
          <div className="flex flex-col items-center py-8 min-h-full">
            {/* Start node */}
            <div className="flex flex-col items-center mb-2">
              <div className="w-12 h-12 rounded-full bg-emerald-500/20 border-2 border-emerald-500/50 flex items-center justify-center">
                <div className="w-3 h-3 rounded-full bg-emerald-500" />
              </div>
              <span className="text-xs text-zinc-500 mt-2">Start</span>
              <div className="flex flex-col items-center py-2">
                <div className="w-0.5 h-6 bg-zinc-700" />
                <svg className="w-3 h-2 text-zinc-600" viewBox="0 0 12 8" fill="currentColor">
                  <path d="M6 8L0 0H12L6 8Z" />
                </svg>
              </div>
            </div>

            {/* Steps */}
            {steps.map((step, index) => (
              <FlowNode
                key={step.id}
                step={step}
                index={index}
                isSelected={selectedStepId === step.id}
                isLast={index === steps.length - 1}
              />
            ))}

            {/* End node */}
            <div className="flex flex-col items-center mt-2">
              <div className="flex flex-col items-center py-2">
                <div className="w-0.5 h-6 bg-zinc-700" />
                <svg className="w-3 h-2 text-zinc-600" viewBox="0 0 12 8" fill="currentColor">
                  <path d="M6 8L0 0H12L6 8Z" />
                </svg>
              </div>
              <div className="w-12 h-12 rounded-full bg-red-500/20 border-2 border-red-500/50 flex items-center justify-center">
                <div className="w-3 h-3 rounded-sm bg-red-500" />
              </div>
              <span className="text-xs text-zinc-500 mt-2">End</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PipelineCanvas;
