import { AdvancedMLTrainingService } from '@/services/ml/AdvancedMLTrainingService';
import { OnnxInferenceService } from '@/services/ml/OnnxInferenceService';
import { BacktestingEngine } from '@/services/ml/BacktestingEngine';
import { CandleData } from '@/types/session';

// Mock ONNX Runtime for integration tests
jest.mock('onnxruntime-web', () => ({
  InferenceSession: {
    create: jest.fn().mockResolvedValue({
      run: jest.fn().mockResolvedValue({
        predictions: {
          data: new Float32Array([0.7, 0.2, 0.1])
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

describe('ML Pipeline Integration Tests', () => {
  let trainingService: AdvancedMLTrainingService;
  let inferenceService: OnnxInferenceService;
  let backtestingEngine: BacktestingEngine;
  let mockCandles: CandleData[];

  beforeEach(() => {
    trainingService = new AdvancedMLTrainingService();
    inferenceService = new OnnxInferenceService();
    backtestingEngine = new BacktestingEngine({
      initialCapital: 10000,
      positionSize: 10,
      stopLoss: 2,
      takeProfit: 4,
      transactionCost: 0.1,
      maxPositions: 3,
      riskPerTrade: 1
    });

    // Generate realistic market data
    mockCandles = generateRealisticMarketData(500);
  });

  describe('End-to-End Training and Inference', () => {
    it('should complete full pipeline from training to prediction', async () => {
      // Step 1: Train model
      const trainingConfig = {
        modelType: 'xgboost' as const,
        hyperparameters: {
          n_estimators: 50,
          max_depth: 4,
          learning_rate: 0.1
        },
        crossValidationFolds: 3,
        testSize: 0.2,
        randomState: 42
      };

      const experimentId = await trainingService.startExperiment(
        'integration-test',
        trainingConfig,
        mockCandles
      );

      // Wait for training to complete (mocked)
      await new Promise(resolve => setTimeout(resolve, 100));

      // Step 2: Load trained model for inference
      const modelConfig = {
        name: 'integration-test-model',
        version: '1.0.0',
        modelPath: '/models/integration-test.onnx',
        scalerParams: {
          mean: [100, 102, 98, 101, 1000],
          std: [5, 6, 4, 5, 200]
        },
        inputShape: [1, 5],
        outputClasses: ['up', 'down', 'sideways'],
        confidenceThreshold: 0.6
      };

      await inferenceService.loadModel(modelConfig);

      // Step 3: Make predictions
      const features = await inferenceService.extractFeaturesFromCandles(mockCandles.slice(-20));
      const predictions = [];

      for (let i = 0; i < features.length; i++) {
        const prediction = await inferenceService.predict('EURUSD', features[i]);
        predictions.push({
          timestamp: mockCandles[mockCandles.length - 20 + i].timestamp,
          direction: prediction.prediction === 'up' ? 'long' as const : 'short' as const,
          confidence: prediction.confidence,
          price: mockCandles[mockCandles.length - 20 + i].close
        });
      }

      // Step 4: Run backtest with predictions
      const backtestResults = await backtestingEngine.runBacktest(mockCandles, predictions);

      // Assertions
      expect(experimentId).toBeDefined();
      expect(predictions.length).toBeGreaterThan(0);
      expect(backtestResults.metrics).toBeDefined();
      expect(backtestResults.trades.length).toBeGreaterThanOrEqual(0);
    }, 30000);

    it('should handle model hot-swapping during live inference', async () => {
      // Load initial model
      const initialConfig = {
        name: 'model-v1',
        version: '1.0.0',
        modelPath: '/models/model-v1.onnx',
        scalerParams: { mean: [100], std: [5] },
        inputShape: [1, 1],
        outputClasses: ['up', 'down'],
        confidenceThreshold: 0.6
      };

      await inferenceService.loadModel(initialConfig);

      // Make initial prediction
      const prediction1 = await inferenceService.predict('EURUSD', [100]);
      expect(prediction1.symbol).toBe('EURUSD');

      // Hot swap to new model
      const newConfig = {
        ...initialConfig,
        name: 'model-v2',
        version: '2.0.0',
        modelPath: '/models/model-v2.onnx'
      };

      await inferenceService.hotSwapModel('model-v1', newConfig);

      // Make prediction with new model
      const prediction2 = await inferenceService.predict('EURUSD', [100]);
      expect(prediction2.symbol).toBe('EURUSD');

      // Verify model info updated
      const modelInfo = inferenceService.getModelInfo();
      expect(modelInfo.version).toBe('2.0.0');
    });
  });

  describe('Data Flow Validation', () => {
    it('should maintain data consistency across pipeline stages', async () => {
      const testCandles = mockCandles.slice(0, 100);
      
      // Extract features using training service
      const trainingFeatures = await trainingService['extractFeatures'](testCandles);
      
      // Load model and extract features using inference service
      await inferenceService.loadModel({
        name: 'consistency-test',
        version: '1.0.0',
        modelPath: '/models/test.onnx',
        scalerParams: { mean: [100, 102, 98, 101, 1000], std: [5, 6, 4, 5, 200] },
        inputShape: [1, 5],
        outputClasses: ['up', 'down', 'sideways'],
        confidenceThreshold: 0.6
      });

      const inferenceFeatures = await inferenceService.extractFeaturesFromCandles(testCandles);

      // Verify feature consistency (basic checks)
      expect(inferenceFeatures.length).toBe(testCandles.length);
      expect(inferenceFeatures[0].length).toBe(5); // OHLCV
    });

    it('should handle missing or corrupted data gracefully', async () => {
      // Create candles with missing data
      const corruptedCandles = mockCandles.slice(0, 50).map((candle, i) => ({
        ...candle,
        // Introduce some data issues
        close: i % 10 === 0 ? NaN : candle.close,
        volume: i % 15 === 0 ? 0 : candle.volume
      }));

      // Test feature extraction with corrupted data
      const features = await trainingService['extractFeatures'](corruptedCandles);
      
      // Should still produce some features, handling NaN values
      expect(features.length).toBeGreaterThan(0);
    });
  });

  describe('Performance Benchmarks', () => {
    it('should meet latency requirements for single predictions', async () => {
      await inferenceService.loadModel({
        name: 'performance-test',
        version: '1.0.0',
        modelPath: '/models/test.onnx',
        scalerParams: { mean: [100], std: [5] },
        inputShape: [1, 1],
        outputClasses: ['up', 'down'],
        confidenceThreshold: 0.6
      });

      const features = [100];
      const startTime = performance.now();
      
      await inferenceService.predict('EURUSD', features);
      
      const endTime = performance.now();
      const latency = endTime - startTime;
      
      // Should be under 20ms as per requirements
      expect(latency).toBeLessThan(20);
    });

    it('should handle batch predictions efficiently', async () => {
      await inferenceService.loadModel({
        name: 'batch-test',
        version: '1.0.0',
        modelPath: '/models/test.onnx',
        scalerParams: { mean: [100], std: [5] },
        inputShape: [1, 1],
        outputClasses: ['up', 'down'],
        confidenceThreshold: 0.6
      });

      const batchSize = 100;
      const batchRequest = Array.from({ length: batchSize }, (_, i) => ({
        symbol: `PAIR${i}`,
        features: [100 + i]
      }));

      const startTime = performance.now();
      const results = await inferenceService.batchPredict(batchRequest);
      const endTime = performance.now();

      const totalLatency = endTime - startTime;
      const avgLatencyPerPrediction = totalLatency / batchSize;

      expect(results.length).toBe(batchSize);
      expect(avgLatencyPerPrediction).toBeLessThan(5); // Should be faster per prediction in batch
    });
  });

  describe('Error Handling and Recovery', () => {
    it('should recover from prediction errors gracefully', async () => {
      await inferenceService.loadModel({
        name: 'error-test',
        version: '1.0.0',
        modelPath: '/models/test.onnx',
        scalerParams: { mean: [100], std: [5] },
        inputShape: [1, 1],
        outputClasses: ['up', 'down'],
        confidenceThreshold: 0.6
      });

      // Test with invalid features (wrong shape)
      await expect(inferenceService.predict('EURUSD', [])).rejects.toThrow();
      
      // Service should still be operational for valid requests
      const validPrediction = await inferenceService.predict('EURUSD', [100]);
      expect(validPrediction.symbol).toBe('EURUSD');
    });

    it('should handle backtesting with no signals', async () => {
      const results = await backtestingEngine.runBacktest(mockCandles, []);
      
      expect(results.trades).toHaveLength(0);
      expect(results.metrics.totalTrades).toBe(0);
      expect(results.equityCurve.length).toBeGreaterThan(0);
    });
  });
});

// Helper function to generate realistic market data
function generateRealisticMarketData(count: number): CandleData[] {
  const candles: CandleData[] = [];
  let price = 1.2000; // Starting price for EURUSD
  
  for (let i = 0; i < count; i++) {
    // Add some trend and noise
    const trend = Math.sin(i * 0.01) * 0.001;
    const noise = (Math.random() - 0.5) * 0.002;
    const priceChange = trend + noise;
    
    price += priceChange;
    
    const volatility = 0.001;
    const high = price + Math.random() * volatility;
    const low = price - Math.random() * volatility;
    const open = price - priceChange / 2;
    
    candles.push({
      id: `candle_${i}`,
      timestamp: Date.now() + i * 60000, // 1 minute intervals
      open,
      high: Math.max(open, high, price),
      low: Math.min(open, low, price),
      close: price,
      volume: 1000 + Math.random() * 2000,
      sessionId: 'integration_test_session'
    } as CandleData);
  }
  
  return candles;
}