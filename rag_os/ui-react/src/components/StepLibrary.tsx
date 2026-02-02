import React, { useState } from 'react';
import { ChevronDownIcon, ChevronRightIcon, PlusIcon } from '@heroicons/react/24/outline';
import { STEP_TYPES, STEP_ORDER } from '../data/stepTypes';
import { usePipelineStore } from '../store/pipelineStore';

const StepLibrary: React.FC = () => {
  const [expandedTypes, setExpandedTypes] = useState<Set<string>>(new Set(['ingestion']));
  const addStep = usePipelineStore((state) => state.addStep);

  const toggleExpand = (stepType: string) => {
    setExpandedTypes((prev) => {
      const next = new Set(prev);
      if (next.has(stepType)) {
        next.delete(stepType);
      } else {
        next.add(stepType);
      }
      return next;
    });
  };

  const handleAddStep = (stepType: string, implementation: string) => {
    addStep(stepType, implementation);
  };

  return (
    <div className="h-full flex flex-col bg-[#111113]">
      {/* Header */}
      <div className="px-4 py-4 border-b border-zinc-800">
        <h2 className="text-sm font-semibold text-zinc-100">Step Library</h2>
        <p className="text-xs text-zinc-500 mt-0.5">Click to add steps to pipeline</p>
      </div>

      {/* Step List */}
      <div className="flex-1 overflow-y-auto">
        {STEP_ORDER.map((stepType) => {
          const typeInfo = STEP_TYPES[stepType];
          const isExpanded = expandedTypes.has(stepType);

          return (
            <div key={stepType} className="border-b border-zinc-800/50">
              {/* Step Type Header */}
              <button
                onClick={() => toggleExpand(stepType)}
                className="w-full flex items-center gap-3 px-4 py-3 hover:bg-zinc-800/30 transition-colors"
              >
                <span className="text-xl">{typeInfo.icon}</span>
                <div className="flex-1 text-left">
                  <div className="text-sm font-medium text-zinc-200">{typeInfo.name}</div>
                  <div className="text-xs text-zinc-500">{typeInfo.description}</div>
                </div>
                <span className="text-xs text-zinc-600 mr-2">
                  {Object.keys(typeInfo.implementations).length}
                </span>
                {isExpanded ? (
                  <ChevronDownIcon className="w-4 h-4 text-zinc-500" />
                ) : (
                  <ChevronRightIcon className="w-4 h-4 text-zinc-500" />
                )}
              </button>

              {/* Implementations */}
              {isExpanded && (
                <div className="bg-zinc-900/50 py-1">
                  {Object.entries(typeInfo.implementations).map(([implKey, impl]) => (
                    <button
                      key={implKey}
                      onClick={() => handleAddStep(stepType, implKey)}
                      className="w-full flex items-center gap-3 px-4 py-2.5 pl-12 hover:bg-zinc-800/50 transition-colors group"
                    >
                      <div className="flex-1 text-left">
                        <div className="text-sm text-zinc-300 group-hover:text-zinc-100">
                          {impl.name}
                        </div>
                        <div className="text-xs text-zinc-600">
                          {impl.description}
                        </div>
                      </div>
                      <PlusIcon className="w-4 h-4 text-zinc-600 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default StepLibrary;
