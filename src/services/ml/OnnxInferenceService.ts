import * as ort from 'onnxruntime-web';
import { CandleData } from '@/types/session';
import { PredictionResult } from '@/types/trading';

interface ModelConfig {
  modelPath: string;
  name: string;
  version: string;
  inputShape: number[];
  outputShape: number[];
  scalerParams?: {
    mean: number[];
    std: number[];
    min?: number[];
    max?: number[];
  };
}

interface PredictionRequest {
  symbol: string;
  features: number[];
  requestId: string;
  timestamp: number;
}

interface PredictionResponse {
  symbol: string;
  prediction: PredictionResult;
  confidence: number;
  uncertainty: number;
  latency: number;
  requestId: string;
  modelVersion: string;
}

interface InferenceMetrics {
  totalRequests: number;
  averageLatency: number;
  errorRate: number;
  successfulPredictions: number;
  modelLoadTime: number;
  memoryUsage: number;
  batchSize: number;
  throughput: number;
}

export class OnnxInferenceService {
  private models: Map<string, ort.InferenceSession> = new Map();
  private modelConfigs: Map<string, ModelConfig> = new Map();
  private metrics: InferenceMetrics = {
    totalRequests: 0,
    averageLatency: 0,
    errorRate: 0,
    successfulPredictions: 0,
    modelLoadTime: 0,
    memoryUsage: 0,
    batchSize: 1,
    throughput: 0
  };
  private requestQueue: PredictionRequest[] = [];
  private isProcessing = false;
  private warmupComplete = false;

  constructor() {
    this.initializeOnnxRuntime();
  }

  private async initializeOnnxRuntime() {
    try {
      // Configure ONNX Runtime for optimal performance
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/';
      ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
      ort.env.webgl.contextId = 'webgl2';
      
      console.log('ONNX Runtime initialized successfully');
    } catch (error) {
      console.error('Failed to initialize ONNX Runtime:', error);
      throw error;
    }
  }

  async loadModel(config: ModelConfig): Promise<void> {
    const startTime = performance.now();
    
    try {
      console.log(`Loading ONNX model: ${config.name} v${config.version}`);
      
      // Create inference session with optimized options
      const sessionOptions: ort.InferenceSession.SessionOptions = {
        executionProviders: ['webgl', 'wasm'],
        enableCpuMemArena: true,
        enableMemPattern: true,
        executionMode: 'parallel',
        interOpNumThreads: 2,
        intraOpNumThreads: 4
      };

      const session = await ort.InferenceSession.create(config.modelPath, sessionOptions);
      
      this.models.set(config.name, session);
      this.modelConfigs.set(config.name, config);
      
      const loadTime = performance.now() - startTime;
      this.metrics.modelLoadTime = loadTime;
      
      console.log(`Model ${config.name} loaded successfully in ${loadTime.toFixed(2)}ms`);
      
      // Warm up the model
      await this.warmupModel(config.name);
      
    } catch (error) {
      console.error(`Failed to load model ${config.name}:`, error);
      throw error;
    }
  }

  private async warmupModel(modelName: string): Promise<void> {
    try {
      const config = this.modelConfigs.get(modelName);
      if (!config) return;

      // Create dummy input data for warmup
      const dummyInput = new Float32Array(config.inputShape.reduce((a, b) => a * b, 1));
      dummyInput.fill(0.5);

      await this.runInference(modelName, dummyInput);
      
      console.log(`Model ${modelName} warmed up successfully`);
      this.warmupComplete = true;
    } catch (error) {
      console.error(`Failed to warm up model ${modelName}:`, error);
    }
  }

  async hotSwapModel(oldModelName: string, newConfig: ModelConfig): Promise<void> {
    try {
      // Load new model first
      await this.loadModel(newConfig);
      
      // Remove old model
      const oldSession = this.models.get(oldModelName);
      if (oldSession) {
        oldSession.release();
        this.models.delete(oldModelName);
        this.modelConfigs.delete(oldModelName);
      }
      
      console.log(`Hot-swapped model ${oldModelName} -> ${newConfig.name}`);
    } catch (error) {
      console.error('Model hot-swap failed:', error);
      throw error;
    }
  }

  private preprocessFeatures(features: number[], modelName: string): Float32Array {
    const config = this.modelConfigs.get(modelName);
    if (!config?.scalerParams) {
      return new Float32Array(features);
    }

    const { mean, std, min, max } = config.scalerParams;
    const processed = features.map((value, index) => {
      let normalized = value;
      
      // Standard scaling
      if (mean && std) {
        normalized = (value - mean[index]) / std[index];
      }
      
      // Min-max scaling
      if (min && max) {
        normalized = (value - min[index]) / (max[index] - min[index]);
      }
      
      return normalized;
    });

    return new Float32Array(processed);
  }

  private async runInference(modelName: string, inputData: Float32Array): Promise<ort.InferenceSession.OnnxValueMapType> {
    const session = this.models.get(modelName);
    if (!session) {
      throw new Error(`Model ${modelName} not found`);
    }

    const config = this.modelConfigs.get(modelName);
    if (!config) {
      throw new Error(`Model config for ${modelName} not found`);
    }

    // Create input tensor
    const inputTensor = new ort.Tensor('float32', inputData, config.inputShape);
    const feeds: Record<string, ort.Tensor> = { input: inputTensor };

    // Run inference
    const results = await session.run(feeds);
    
    return results;
  }

  private calculateUncertainty(predictions: Float32Array): number {
    // Calculate prediction uncertainty using entropy
    const probs = Array.from(predictions);
    const entropy = -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
    return Math.min(entropy / Math.log(probs.length), 1.0);
  }

  private convertToPredictionResult(
    output: ort.InferenceSession.OnnxValueMapType,
    symbol: string,
    modelVersion: string
  ): PredictionResult {
    const outputTensor = output.output as ort.Tensor;
    const predictions = outputTensor.data as Float32Array;

    // Assuming binary classification output [call_prob, put_prob]
    const callProb = predictions[0];
    const putProb = predictions[1];
    
    const direction = callProb > putProb ? 'UP' : 'DOWN';
    const probability = Math.max(callProb, putProb);
    const confidence = Math.abs(callProb - putProb);

    return {
      direction,
      probability,
      confidence,
      interval: 300, // 5 minutes
      factors: {
        technical: probability * 0.3,
        volume: probability * 0.15,
        momentum: probability * 0.2,
        volatility: probability * 0.1,
        pattern: probability * 0.15,
        trend: probability * 0.1
      },
      recommendation: direction === 'UP' ? 'CALL' : 'PUT',
      metadata: {
        modelAgreement: confidence,
        riskScore: 1 - confidence,
        marketCondition: 'NORMAL',
        modelBreakdown: [
          { model: modelVersion, prediction: direction, confidence }
        ]
      }
    };
  }

  async predict(
    symbol: string,
    features: number[],
    modelName = 'default'
  ): Promise<PredictionResponse> {
    const startTime = performance.now();
    const requestId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    try {
      this.metrics.totalRequests++;

      if (!this.warmupComplete) {
        throw new Error('Model not ready - warmup in progress');
      }

      const session = this.models.get(modelName);
      const config = this.modelConfigs.get(modelName);
      
      if (!session || !config) {
        throw new Error(`Model ${modelName} not available`);
      }

      // Preprocess features
      const processedFeatures = this.preprocessFeatures(features, modelName);

      // Run inference
      const results = await this.runInference(modelName, processedFeatures);

      // Convert to prediction result
      const prediction = this.convertToPredictionResult(results, symbol, config.version);

      // Calculate uncertainty
      const outputTensor = results.output as ort.Tensor;
      const uncertainty = this.calculateUncertainty(outputTensor.data as Float32Array);

      const latency = performance.now() - startTime;
      
      // Update metrics
      this.metrics.successfulPredictions++;
      this.metrics.averageLatency = 
        (this.metrics.averageLatency * (this.metrics.successfulPredictions - 1) + latency) / 
        this.metrics.successfulPredictions;

      return {
        symbol,
        prediction,
        confidence: prediction.confidence,
        uncertainty,
        latency,
        requestId,
        modelVersion: config.version
      };

    } catch (error) {
      this.metrics.errorRate = (this.metrics.totalRequests - this.metrics.successfulPredictions) / this.metrics.totalRequests;
      console.error('Prediction failed:', error);
      throw error;
    }
  }

  async batchPredict(
    requests: Array<{ symbol: string; features: number[] }>,
    modelName = 'default'
  ): Promise<PredictionResponse[]> {
    const batchStartTime = performance.now();
    
    try {
      // Process requests in parallel with controlled concurrency
      const BATCH_SIZE = 8;
      const results: PredictionResponse[] = [];
      
      for (let i = 0; i < requests.length; i += BATCH_SIZE) {
        const batch = requests.slice(i, i + BATCH_SIZE);
        const batchPromises = batch.map(req => 
          this.predict(req.symbol, req.features, modelName)
        );
        
        const batchResults = await Promise.all(batchPromises);
        results.push(...batchResults);
      }

      const totalLatency = performance.now() - batchStartTime;
      this.metrics.batchSize = requests.length;
      this.metrics.throughput = requests.length / (totalLatency / 1000);

      return results;
    } catch (error) {
      console.error('Batch prediction failed:', error);
      throw error;
    }
  }

  async extractFeaturesFromCandles(candles: CandleData[]): Promise<number[]> {
    if (candles.length < 20) {
      throw new Error('Insufficient candle data for feature extraction');
    }

    const features: number[] = [];
    const latest = candles[candles.length - 1];
    
    // Price features
    features.push(latest.open, latest.high, latest.low, latest.close);
    
    // Volume features
    features.push(latest.volume);
    
    // Price ratios
    features.push(latest.close / latest.open);
    features.push(latest.high / latest.low);
    
    // Moving averages
    const prices = candles.map(c => c.close);
    const sma5 = prices.slice(-5).reduce((a, b) => a + b) / 5;
    const sma10 = prices.slice(-10).reduce((a, b) => a + b) / 10;
    const sma20 = prices.slice(-20).reduce((a, b) => a + b) / 20;
    
    features.push(sma5, sma10, sma20);
    features.push(latest.close / sma5, latest.close / sma10, latest.close / sma20);
    
    // Volatility
    const returns = prices.slice(1).map((price, i) => Math.log(price / prices[i]));
    const volatility = Math.sqrt(returns.reduce((sum, ret) => sum + ret * ret, 0) / returns.length);
    features.push(volatility);
    
    // Momentum
    const momentum = (latest.close - candles[candles.length - 5].close) / candles[candles.length - 5].close;
    features.push(momentum);
    
    return features;
  }

  getMetrics(): InferenceMetrics {
    return { ...this.metrics };
  }

  getModelInfo(): Array<{ name: string; version: string; loaded: boolean }> {
    return Array.from(this.modelConfigs.entries()).map(([name, config]) => ({
      name,
      version: config.version,
      loaded: this.models.has(name)
    }));
  }

  async healthCheck(): Promise<{ status: 'healthy' | 'degraded' | 'unhealthy'; details: any }> {
    try {
      const modelCount = this.models.size;
      const memoryUsage = (performance as any).memory?.usedJSHeapSize || 0;
      const avgLatency = this.metrics.averageLatency;
      const errorRate = this.metrics.errorRate;

      let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
      
      if (modelCount === 0) status = 'unhealthy';
      else if (avgLatency > 50 || errorRate > 0.1) status = 'degraded';

      return {
        status,
        details: {
          modelsLoaded: modelCount,
          memoryUsage: Math.round(memoryUsage / 1024 / 1024), // MB
          averageLatency: Math.round(avgLatency),
          errorRate: Math.round(errorRate * 100),
          warmupComplete: this.warmupComplete,
          totalRequests: this.metrics.totalRequests
        }
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        details: { error: error.message }
      };
    }
  }

  dispose(): void {
    // Release all model sessions
    for (const [name, session] of this.models.entries()) {
      try {
        session.release();
        console.log(`Released model: ${name}`);
      } catch (error) {
        console.error(`Error releasing model ${name}:`, error);
      }
    }
    
    this.models.clear();
    this.modelConfigs.clear();
    console.log('ONNX Inference Service disposed');
  }
}

// Singleton instance
export const onnxInferenceService = new OnnxInferenceService();