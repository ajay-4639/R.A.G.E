import { create } from 'zustand';
import { PipelineStep, StepConfig } from '../types';
import { STEP_TYPES } from '../data/stepTypes';

interface PipelineState {
  steps: PipelineStep[];
  selectedStepId: string | null;
  pipelineName: string;
  pipelineDescription: string;
  isDirty: boolean;

  // Actions
  addStep: (stepType: string, implementation: string) => void;
  removeStep: (id: string) => void;
  moveStep: (fromIndex: number, toIndex: number) => void;
  selectStep: (id: string | null) => void;
  updateStepConfig: (id: string, config: StepConfig) => void;
  updateStepName: (id: string, name: string) => void;
  setPipelineName: (name: string) => void;
  setPipelineDescription: (description: string) => void;
  clearPipeline: () => void;
  loadPipeline: (steps: PipelineStep[], name: string, description: string) => void;
  setDirty: (dirty: boolean) => void;
}

let stepCounter = 0;

const generateStepId = (stepType: string): string => {
  stepCounter++;
  return `${stepType}_${stepCounter}_${Date.now()}`;
};

export const usePipelineStore = create<PipelineState>((set, get) => ({
  steps: [],
  selectedStepId: null,
  pipelineName: 'New Pipeline',
  pipelineDescription: '',
  isDirty: false,

  addStep: (stepType: string, implementation: string) => {
    const stepTypeInfo = STEP_TYPES[stepType];
    if (!stepTypeInfo) return;

    const implInfo = stepTypeInfo.implementations[implementation];
    if (!implInfo) return;

    const newStep: PipelineStep = {
      id: generateStepId(stepType),
      stepType: stepType as PipelineStep['stepType'],
      implementation,
      name: implInfo.name,
      config: { ...implInfo.defaultConfig },
    };

    set((state) => ({
      steps: [...state.steps, newStep],
      selectedStepId: newStep.id,
      isDirty: true,
    }));
  },

  removeStep: (id: string) => {
    set((state) => ({
      steps: state.steps.filter((s) => s.id !== id),
      selectedStepId: state.selectedStepId === id ? null : state.selectedStepId,
      isDirty: true,
    }));
  },

  moveStep: (fromIndex: number, toIndex: number) => {
    set((state) => {
      const newSteps = [...state.steps];
      const [removed] = newSteps.splice(fromIndex, 1);
      newSteps.splice(toIndex, 0, removed);
      return { steps: newSteps, isDirty: true };
    });
  },

  selectStep: (id: string | null) => {
    set({ selectedStepId: id });
  },

  updateStepConfig: (id: string, config: StepConfig) => {
    set((state) => ({
      steps: state.steps.map((s) =>
        s.id === id ? { ...s, config: { ...s.config, ...config } } : s
      ),
      isDirty: true,
    }));
  },

  updateStepName: (id: string, name: string) => {
    set((state) => ({
      steps: state.steps.map((s) => (s.id === id ? { ...s, name } : s)),
      isDirty: true,
    }));
  },

  setPipelineName: (name: string) => {
    set({ pipelineName: name, isDirty: true });
  },

  setPipelineDescription: (description: string) => {
    set({ pipelineDescription: description, isDirty: true });
  },

  clearPipeline: () => {
    set({
      steps: [],
      selectedStepId: null,
      pipelineName: 'New Pipeline',
      pipelineDescription: '',
      isDirty: false,
    });
  },

  loadPipeline: (steps: PipelineStep[], name: string, description: string) => {
    set({
      steps,
      selectedStepId: null,
      pipelineName: name,
      pipelineDescription: description,
      isDirty: false,
    });
  },

  setDirty: (dirty: boolean) => {
    set({ isDirty: dirty });
  },
}));
