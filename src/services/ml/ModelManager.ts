import { onnxInferenceService } from './OnnxInferenceService';

interface ModelConfiguration {
  name: string;
  version: string;
  modelPath: string;
  inputShape: number[];
  outputShape: number[];
  scalerParams?: {
    mean: number[];
    std: number[];
    min?: number[];
    max?: number[];
  };
  metadata?: {
    description: string;
    author: string;
    created: string;
    accuracy: number;
    trainingData: string;
  };
}

interface DeploymentConfig {
  modelName: string;
  targetEnvironment: 'development' | 'staging' | 'production';
  rolloutPercentage: number;
  autoRollback: boolean;
  healthCheckInterval: number;
}

export class ModelManager {
  private modelConfigs: Map<string, ModelConfiguration> = new Map();
  private deploymentHistory: Array<{
    timestamp: number;
    modelName: string;
    version: string;
    action: 'deploy' | 'rollback' | 'remove';
    success: boolean;
    error?: string;
  }> = [];

  constructor() {
    this.initializeDefaultModels();
  }

  private initializeDefaultModels() {
    // Add default model configurations
    const defaultConfigs: ModelConfiguration[] = [
      {
        name: 'binary-classifier-v1',
        version: '1.0.0',
        modelPath: '/models/binary-classifier-v1.onnx',
        inputShape: [1, 15],
        outputShape: [1, 2],
        scalerParams: {
          mean: [
            50000, 50500, 49500, 50250, 1000000, // OHLCV
            1.005, 1.02, // Price ratios
            50100, 50150, 50200, // SMAs
            1.003, 1.001, 0.998, // Price/SMA ratios
            0.02, 0.05 // Volatility, Momentum
          ],
          std: [
            5000, 5000, 5000, 5000, 500000, // OHLCV
            0.1, 0.1, // Price ratios
            5000, 5000, 5000, // SMAs
            0.1, 0.1, 0.1, // Price/SMA ratios
            0.01, 0.1 // Volatility, Momentum
          ]
        },
        metadata: {
          description: 'Binary classification model for CALL/PUT predictions',
          author: 'Trading Team',
          created: '2024-01-15',
          accuracy: 0.67,
          trainingData: 'EUR/USD 5M candles, 6 months'
        }
      },
      {
        name: 'ensemble-predictor-v2',
        version: '2.1.0',
        modelPath: '/models/ensemble-predictor-v2.onnx',
        inputShape: [1, 20],
        outputShape: [1, 3], // UP, DOWN, SIDEWAYS
        scalerParams: {
          mean: Array(20).fill(0),
          std: Array(20).fill(1)
        },
        metadata: {
          description: 'Advanced ensemble model with pattern recognition',
          author: 'ML Team',
          created: '2024-02-01',
          accuracy: 0.73,
          trainingData: 'Multi-pair data with technical indicators'
        }
      }
    ];

    defaultConfigs.forEach(config => {
      this.modelConfigs.set(config.name, config);
    });
  }

  async deployModel(
    modelName: string,
    config?: DeploymentConfig
  ): Promise<{ success: boolean; error?: string }> {
    const startTime = Date.now();
    
    try {
      const modelConfig = this.modelConfigs.get(modelName);
      if (!modelConfig) {
        throw new Error(`Model configuration not found: ${modelName}`);
      }

      console.log(`Deploying model: ${modelName} v${modelConfig.version}`);

      // Validate model file exists (in production, this would check actual file)
      if (!modelConfig.modelPath) {
        throw new Error(`Model path not specified for ${modelName}`);
      }

      // Load model with inference service
      await onnxInferenceService.loadModel(modelConfig);

      // Log successful deployment
      this.deploymentHistory.push({
        timestamp: startTime,
        modelName,
        version: modelConfig.version,
        action: 'deploy',
        success: true
      });

      console.log(`Successfully deployed ${modelName} v${modelConfig.version}`);
      
      return { success: true };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      // Log failed deployment
      this.deploymentHistory.push({
        timestamp: startTime,
        modelName,
        version: this.modelConfigs.get(modelName)?.version || 'unknown',
        action: 'deploy',
        success: false,
        error: errorMessage
      });

      console.error(`Failed to deploy ${modelName}:`, error);
      
      return { success: false, error: errorMessage };
    }
  }

  async hotSwapModel(
    currentModelName: string,
    newModelName: string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const newModelConfig = this.modelConfigs.get(newModelName);
      if (!newModelConfig) {
        throw new Error(`New model configuration not found: ${newModelName}`);
      }

      console.log(`Hot-swapping ${currentModelName} -> ${newModelName}`);

      // Perform hot swap with inference service
      await onnxInferenceService.hotSwapModel(currentModelName, newModelConfig);

      // Log successful hot swap
      this.deploymentHistory.push({
        timestamp: Date.now(),
        modelName: newModelName,
        version: newModelConfig.version,
        action: 'deploy',
        success: true
      });

      console.log(`Successfully hot-swapped to ${newModelName}`);
      
      return { success: true };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('Hot swap failed:', error);
      
      return { success: false, error: errorMessage };
    }
  }

  async rollbackModel(modelName: string): Promise<{ success: boolean; error?: string }> {
    try {
      // Find the previous successful deployment
      const deployments = this.deploymentHistory
        .filter(d => d.modelName === modelName && d.success)
        .sort((a, b) => b.timestamp - a.timestamp);

      if (deployments.length < 2) {
        throw new Error('No previous version available for rollback');
      }

      const previousDeployment = deployments[1];
      const previousConfig = this.modelConfigs.get(previousDeployment.modelName);
      
      if (!previousConfig) {
        throw new Error('Previous model configuration not found');
      }

      console.log(`Rolling back ${modelName} to version ${previousConfig.version}`);

      // Perform rollback
      await onnxInferenceService.loadModel(previousConfig);

      // Log rollback
      this.deploymentHistory.push({
        timestamp: Date.now(),
        modelName: previousDeployment.modelName,
        version: previousConfig.version,
        action: 'rollback',
        success: true
      });

      console.log(`Successfully rolled back to ${previousDeployment.modelName} v${previousConfig.version}`);
      
      return { success: true };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('Rollback failed:', error);
      
      return { success: false, error: errorMessage };
    }
  }

  addModelConfiguration(config: ModelConfiguration): void {
    this.modelConfigs.set(config.name, config);
    console.log(`Added model configuration: ${config.name} v${config.version}`);
  }

  removeModelConfiguration(modelName: string): boolean {
    const removed = this.modelConfigs.delete(modelName);
    if (removed) {
      console.log(`Removed model configuration: ${modelName}`);
    }
    return removed;
  }

  getModelConfiguration(modelName: string): ModelConfiguration | undefined {
    return this.modelConfigs.get(modelName);
  }

  getAllModelConfigurations(): ModelConfiguration[] {
    return Array.from(this.modelConfigs.values());
  }

  getDeploymentHistory(): Array<{
    timestamp: number;
    modelName: string;
    version: string;
    action: 'deploy' | 'rollback' | 'remove';
    success: boolean;
    error?: string;
  }> {
    return [...this.deploymentHistory].sort((a, b) => b.timestamp - a.timestamp);
  }

  async validateModelHealth(modelName: string): Promise<{
    healthy: boolean;
    latency?: number;
    accuracy?: number;
    errors?: string[];
  }> {
    try {
      const config = this.modelConfigs.get(modelName);
      if (!config) {
        return { healthy: false, errors: ['Model configuration not found'] };
      }

      // Perform health check with test prediction
      const startTime = performance.now();
      
      // Create test features
      const testFeatures = config.inputShape.slice(1).reduce((acc, dim) => {
        acc.push(...Array(dim).fill(0.5));
        return acc;
      }, [] as number[]);

      // Test prediction
      const result = await onnxInferenceService.predict('TEST', testFeatures, modelName);
      const latency = performance.now() - startTime;

      const healthy = latency < 50 && result.confidence > 0;
      
      return {
        healthy,
        latency,
        accuracy: config.metadata?.accuracy,
        errors: healthy ? [] : ['High latency or low confidence']
      };

    } catch (error) {
      return {
        healthy: false,
        errors: [error instanceof Error ? error.message : 'Unknown error']
      };
    }
  }

  async performCanaryDeployment(
    modelName: string,
    canaryPercentage: number = 10
  ): Promise<{ success: boolean; metrics?: any; error?: string }> {
    try {
      console.log(`Starting canary deployment for ${modelName} (${canaryPercentage}%)`);

      // In a real implementation, this would:
      // 1. Deploy to a subset of traffic
      // 2. Monitor metrics
      // 3. Gradually increase traffic
      // 4. Rollback if metrics degrade

      const modelConfig = this.modelConfigs.get(modelName);
      if (!modelConfig) {
        throw new Error(`Model configuration not found: ${modelName}`);
      }

      // Simulate canary deployment
      await this.deployModel(modelName);
      
      // Collect metrics after deployment
      const health = await onnxInferenceService.healthCheck();
      const metrics = onnxInferenceService.getMetrics();

      return {
        success: true,
        metrics: {
          health: health.status,
          latency: metrics.averageLatency,
          errorRate: metrics.errorRate,
          canaryPercentage
        }
      };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('Canary deployment failed:', error);
      
      return { success: false, error: errorMessage };
    }
  }

  exportModelRegistry(): string {
    const registry = {
      models: Array.from(this.modelConfigs.entries()).map(([name, config]) => ({
        name,
        ...config
      })),
      deploymentHistory: this.deploymentHistory,
      exportedAt: new Date().toISOString()
    };

    return JSON.stringify(registry, null, 2);
  }

  importModelRegistry(registryJson: string): { success: boolean; error?: string } {
    try {
      const registry = JSON.parse(registryJson);
      
      if (!registry.models || !Array.isArray(registry.models)) {
        throw new Error('Invalid registry format');
      }

      // Clear existing configurations
      this.modelConfigs.clear();

      // Import model configurations
      for (const model of registry.models) {
        const { name, ...config } = model;
        this.modelConfigs.set(name, config);
      }

      // Import deployment history if available
      if (registry.deploymentHistory && Array.isArray(registry.deploymentHistory)) {
        this.deploymentHistory = registry.deploymentHistory;
      }

      console.log(`Imported ${registry.models.length} model configurations`);
      
      return { success: true };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('Failed to import model registry:', error);
      
      return { success: false, error: errorMessage };
    }
  }
}

// Singleton instance
export const modelManager = new ModelManager();