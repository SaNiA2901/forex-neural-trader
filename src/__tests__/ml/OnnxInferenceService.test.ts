import { OnnxInferenceService } from '@/services/ml/OnnxInferenceService';
import { CandleData } from '@/types/session';

// Mock ONNX Runtime
jest.mock('onnxruntime-web', () => ({
  InferenceSession: {
    create: jest.fn().mockResolvedValue({
      run: jest.fn().mockResolvedValue({
        predictions: {
          data: new Float32Array([0.7, 0.3, 0.8])
        }
      })
    })
  },
  Tensor: jest.fn().mockImplementation((type, data, dims) => ({
    type,
    data,
    dims
  }))
}));

describe('OnnxInferenceService', () => {
  let service: OnnxInferenceService;
  let mockCandles: CandleData[];

  beforeEach(() => {
    service = new OnnxInferenceService();
    jest.clearAllMocks();
    
    mockCandles = Array.from({ length: 20 }, (_, i) => ({
      id: `candle_${i}`,
      timestamp: Date.now() + i * 60000,
      open: 100 + Math.random() * 10,
      high: 105 + Math.random() * 10,
      low: 95 + Math.random() * 10,
      close: 100 + Math.random() * 10,
      volume: 1000 + Math.random() * 500,
      sessionId: 'test_session'
    } as CandleData));
  });

  describe('Model Management', () => {
    const mockModelConfig = {
      name: 'test-model',
      version: '1.0.0',
      modelPath: '/models/test-model.onnx',
      scalerParams: {
        mean: [100, 102, 98, 101, 1000],
        std: [5, 6, 4, 5, 200]
      },
      inputShape: [1, 5],
      outputClasses: ['up', 'down', 'sideways'],
      confidenceThreshold: 0.6
    };

    it('should load model successfully', async () => {
      await service.loadModel(mockModelConfig);
      
      const modelInfo = service.getModelInfo();
      expect(modelInfo.name).toBe(mockModelConfig.name);
      expect(modelInfo.isLoaded).toBe(true);
    });

    it('should handle model loading errors gracefully', async () => {
      const invalidConfig = { ...mockModelConfig, modelPath: '/invalid/path.onnx' };
      
      await expect(service.loadModel(invalidConfig)).rejects.toThrow();
    });

    it('should perform hot swap correctly', async () => {
      await service.loadModel(mockModelConfig);
      
      const newConfig = { ...mockModelConfig, version: '2.0.0' };
      await service.hotSwapModel('test-model', newConfig);
      
      const modelInfo = service.getModelInfo();
      expect(modelInfo.version).toBe('2.0.0');
    });
  });

  describe('Feature Engineering', () => {
    beforeEach(async () => {
      const mockModelConfig = {
        name: 'test-model',
        version: '1.0.0',
        modelPath: '/models/test-model.onnx',
        scalerParams: {
          mean: [100, 102, 98, 101, 1000],
          std: [5, 6, 4, 5, 200]
        },
        inputShape: [1, 5],
        outputClasses: ['up', 'down', 'sideways'],
        confidenceThreshold: 0.6
      };
      
      await service.loadModel(mockModelConfig);
    });

    it('should extract features from candles', async () => {
      const features = await service.extractFeaturesFromCandles(mockCandles);
      
      expect(features).toHaveLength(mockCandles.length);
      expect(features[0]).toHaveLength(5); // OHLCV
    });

    it('should normalize features correctly', async () => {
      const features = await service.extractFeaturesFromCandles(mockCandles);
      const normalized = await service['normalizeFeatures'](features[0]);
      
      expect(normalized).toHaveLength(5);
      // Check if normalization is applied (values should be different from original)
      expect(normalized).not.toEqual(features[0]);
    });
  });

  describe('Prediction', () => {
    beforeEach(async () => {
      const mockModelConfig = {
        name: 'test-model',
        version: '1.0.0',
        modelPath: '/models/test-model.onnx',
        scalerParams: {
          mean: [100, 102, 98, 101, 1000],
          std: [5, 6, 4, 5, 200]
        },
        inputShape: [1, 5],
        outputClasses: ['up', 'down', 'sideways'],
        confidenceThreshold: 0.6
      };
      
      await service.loadModel(mockModelConfig);
    });

    it('should make single prediction', async () => {
      const features = [100, 102, 98, 101, 1000];
      const result = await service.predict('EURUSD', features);
      
      expect(result).toHaveProperty('symbol', 'EURUSD');
      expect(result).toHaveProperty('prediction');
      expect(result).toHaveProperty('confidence');
      expect(result).toHaveProperty('probabilities');
      expect(result.confidence).toBeGreaterThanOrEqual(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
    });

    it('should handle batch predictions', async () => {
      const batchRequest = [
        { symbol: 'EURUSD', features: [100, 102, 98, 101, 1000] },
        { symbol: 'GBPUSD', features: [130, 132, 128, 131, 800] }
      ];
      
      const results = await service.batchPredict(batchRequest);
      
      expect(results).toHaveLength(2);
      expect(results[0].symbol).toBe('EURUSD');
      expect(results[1].symbol).toBe('GBPUSD');
    });

    it('should calculate uncertainty correctly', async () => {
      const features = [100, 102, 98, 101, 1000];
      const result = await service.predict('EURUSD', features);
      
      expect(result.uncertainty).toBeDefined();
      expect(result.uncertainty).toBeGreaterThanOrEqual(0);
      expect(result.uncertainty).toBeLessThanOrEqual(1);
    });
  });

  describe('Performance Monitoring', () => {
    it('should track metrics correctly', async () => {
      const initialMetrics = service.getMetrics();
      
      expect(initialMetrics).toHaveProperty('totalPredictions', 0);
      expect(initialMetrics).toHaveProperty('averageLatency', 0);
      expect(initialMetrics).toHaveProperty('errorRate', 0);
    });

    it('should update latency metrics', async () => {
      await service.loadModel({
        name: 'test-model',
        version: '1.0.0',
        modelPath: '/models/test-model.onnx',
        scalerParams: { mean: [100], std: [5] },
        inputShape: [1, 1],
        outputClasses: ['up', 'down'],
        confidenceThreshold: 0.6
      });

      const features = [100];
      await service.predict('EURUSD', features);
      
      const metrics = service.getMetrics();
      expect(metrics.totalPredictions).toBe(1);
      expect(metrics.averageLatency).toBeGreaterThan(0);
    });
  });

  describe('Health Checks', () => {
    it('should report healthy status when model is loaded', async () => {
      await service.loadModel({
        name: 'test-model',
        version: '1.0.0',
        modelPath: '/models/test-model.onnx',
        scalerParams: { mean: [100], std: [5] },
        inputShape: [1, 1],
        outputClasses: ['up', 'down'],
        confidenceThreshold: 0.6
      });

      const health = await service.healthCheck();
      expect(health.status).toBe('healthy');
    });

    it('should report unhealthy status when no model is loaded', async () => {
      const health = await service.healthCheck();
      expect(health.status).toBe('unhealthy');
    });
  });
});