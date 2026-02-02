import axios, { AxiosInstance } from 'axios';
import { PipelineSpec, QueryResult } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class APIClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        throw error;
      }
    );
  }

  async healthCheck(): Promise<{ status: string; version: string }> {
    const response = await this.client.get('/health');
    return response.data;
  }

  async listPipelines(): Promise<string[]> {
    const response = await this.client.get('/pipelines');
    return response.data;
  }

  async getPipeline(name: string): Promise<PipelineSpec | null> {
    try {
      const response = await this.client.get(`/pipelines/${name}`);
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 404) {
        return null;
      }
      throw error;
    }
  }

  async createPipeline(spec: PipelineSpec): Promise<PipelineSpec> {
    const response = await this.client.post('/pipelines', spec);
    return response.data;
  }

  async deletePipeline(name: string): Promise<void> {
    await this.client.delete(`/pipelines/${name}`);
  }

  async queryPipeline(
    pipelineName: string,
    query: string,
    context?: Record<string, unknown>
  ): Promise<QueryResult> {
    const response = await this.client.post(`/pipelines/${pipelineName}/query`, {
      query,
      context,
    });
    return response.data;
  }

  async getMetrics(): Promise<Record<string, unknown>> {
    const response = await this.client.get('/metrics');
    return response.data;
  }
}

export const apiClient = new APIClient();
export default apiClient;
