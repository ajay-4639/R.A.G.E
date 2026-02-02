import React from 'react';
import { XMarkIcon, DocumentDuplicateIcon, Cog6ToothIcon } from '@heroicons/react/24/outline';
import { usePipelineStore } from '../store/pipelineStore';
import { STEP_TYPES } from '../data/stepTypes';
import { ConfigField } from '../types';

type ConfigValue = string | number | boolean | string[] | Record<string, unknown>;

const ConfigInput: React.FC<{
  field: ConfigField;
  value: unknown;
  onChange: (value: ConfigValue) => void;
}> = ({ field, value, onChange }) => {
  const baseInputClass = 'w-full px-3 py-2 rounded-lg border border-zinc-700 bg-zinc-800/50 text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm';

  switch (field.type) {
    case 'text':
      return (
        <input
          type="text"
          value={String(value || '')}
          onChange={(e) => onChange(e.target.value)}
          placeholder={field.placeholder}
          className={baseInputClass}
        />
      );

    case 'number':
      return (
        <input
          type="number"
          value={Number(value) || 0}
          onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
          min={field.min}
          max={field.max}
          step={field.step}
          className={baseInputClass}
        />
      );

    case 'boolean':
      return (
        <label className="flex items-center gap-3 cursor-pointer">
          <div className="relative">
            <input
              type="checkbox"
              checked={Boolean(value)}
              onChange={(e) => onChange(e.target.checked)}
              className="sr-only"
            />
            <div className={`w-10 h-6 rounded-full transition-colors ${value ? 'bg-blue-600' : 'bg-zinc-700'}`}>
              <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${value ? 'translate-x-5' : 'translate-x-1'}`} />
            </div>
          </div>
          <span className="text-sm text-zinc-400">{value ? 'Enabled' : 'Disabled'}</span>
        </label>
      );

    case 'select':
      return (
        <select
          value={String(value || '')}
          onChange={(e) => onChange(e.target.value)}
          className={baseInputClass}
        >
          {field.options?.map((opt) => (
            <option key={opt.value} value={opt.value} className="bg-zinc-800">
              {opt.label}
            </option>
          ))}
        </select>
      );

    case 'textarea':
      return (
        <textarea
          value={String(value || '')}
          onChange={(e) => onChange(e.target.value)}
          placeholder={field.placeholder}
          rows={4}
          className={`${baseInputClass} resize-none`}
        />
      );

    default:
      return (
        <input
          type="text"
          value={String(value || '')}
          onChange={(e) => onChange(e.target.value)}
          className={baseInputClass}
        />
      );
  }
};

const StepConfig: React.FC = () => {
  const { steps, selectedStepId, selectStep, updateStepConfig, updateStepName } = usePipelineStore();

  const selectedStep = steps.find((s) => s.id === selectedStepId);
  const stepInfo = selectedStep ? STEP_TYPES[selectedStep.stepType] : null;
  const implInfo = stepInfo?.implementations[selectedStep?.implementation || ''];

  const handleConfigChange = (key: string, value: ConfigValue) => {
    if (selectedStepId) {
      updateStepConfig(selectedStepId, { [key]: value });
    }
  };

  const copyConfig = () => {
    if (selectedStep) {
      navigator.clipboard.writeText(JSON.stringify(selectedStep.config, null, 2));
    }
  };

  return (
    <div className="h-full flex flex-col bg-[#111113]">
      {selectedStep ? (
        <>
          {/* Header */}
          <div className="px-4 py-4 border-b border-zinc-800">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-2xl">{stepInfo?.icon}</span>
                <div>
                  <input
                    type="text"
                    value={selectedStep.name}
                    onChange={(e) => updateStepName(selectedStep.id, e.target.value)}
                    className="font-medium text-zinc-100 bg-transparent border-none outline-none"
                  />
                  <p className="text-xs text-zinc-500">{stepInfo?.name}</p>
                </div>
              </div>
              <button
                onClick={() => selectStep(null)}
                className="p-2 rounded-lg hover:bg-zinc-800 transition-colors"
              >
                <XMarkIcon className="w-5 h-5 text-zinc-500" />
              </button>
            </div>
          </div>

          {/* Config Fields */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {implInfo?.configSchema.map((field) => (
              <div key={field.key}>
                <label className="block text-sm font-medium text-zinc-300 mb-1.5">
                  {field.label}
                  {field.required && <span className="text-red-400 ml-1">*</span>}
                </label>
                <ConfigInput
                  field={field}
                  value={selectedStep.config[field.key]}
                  onChange={(value) => handleConfigChange(field.key, value)}
                />
              </div>
            ))}
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-zinc-800">
            <button
              onClick={copyConfig}
              className="flex items-center justify-center gap-2 w-full px-4 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-sm font-medium transition-colors"
            >
              <DocumentDuplicateIcon className="w-4 h-4" />
              Copy Config JSON
            </button>
          </div>
        </>
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center text-zinc-500 p-4">
          <div className="w-12 h-12 rounded-xl bg-zinc-800 flex items-center justify-center mb-3">
            <Cog6ToothIcon className="w-6 h-6 text-zinc-600" />
          </div>
          <p className="text-sm text-zinc-500 text-center">
            Select a step from the pipeline to configure it
          </p>
        </div>
      )}
    </div>
  );
};

export default StepConfig;
