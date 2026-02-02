export interface StepConfig {
  [key: string]: string | number | boolean | string[] | Record<string, unknown>;
}

export interface PipelineStep {
  id: string;
  stepType: StepType;
  implementation: string;
  name: string;
  config: StepConfig;
}

export type StepType =
  | 'ingestion'
  | 'chunking'
  | 'embedding'
  | 'retrieval'
  | 'reranking'
  | 'prompt_assembly'
  | 'llm_execution'
  | 'post_processing';

export interface StepTypeInfo {
  name: string;
  icon: string;
  description: string;
  color: string;
  implementations: Record<string, ImplementationInfo>;
}

export interface ImplementationInfo {
  name: string;
  description: string;
  defaultConfig: StepConfig;
  configSchema: ConfigField[];
}

export interface ConfigField {
  key: string;
  label: string;
  type: 'text' | 'number' | 'boolean' | 'select' | 'textarea' | 'array';
  required?: boolean;
  placeholder?: string;
  options?: { value: string; label: string }[];
  min?: number;
  max?: number;
  step?: number;
}

export interface Pipeline {
  name: string;
  version: string;
  description?: string;
  steps: PipelineStep[];
  metadata?: Record<string, unknown>;
}

export interface PipelineSpec {
  name: string;
  version: string;
  description?: string;
  steps: {
    step_id: string;
    step_type: string;
    step_class: string;
    config: StepConfig;
  }[];
  metadata?: Record<string, unknown>;
}

export interface QueryResult {
  answer: string;
  sources?: Source[];
  metadata?: Record<string, unknown>;
  execution_time_ms?: number;
}

export interface Source {
  content: string;
  metadata?: Record<string, unknown>;
  score?: number;
}
