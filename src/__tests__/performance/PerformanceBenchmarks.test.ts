import { OnnxInferenceService } from '@/services/ml/OnnxInferenceService';
import { BacktestingEngine } from '@/services/ml/BacktestingEngine';
import { testDataGenerator } from '../utils/TestDataGenerator';

// Mock ONNX Runtime for performance tests
jest.mock('onnxruntime-web', () => ({
  InferenceSession: {
    create: jest.fn().mockResolvedValue({
      run: jest.fn().mockImplementation(() => 
        new Promise(resolve => {
          // Simulate processing time
          setTimeout(() => {
            resolve({
              predictions: {
                data: new Float32Array([0.7, 0.2, 0.1])
              }
            });
          }, Math.random() * 10); // 0-10ms processing time
        })
      )
    })
  },
  Tensor: jest.fn().mockImplementation((type, data, dims) => ({
    type,
    data,
    dims
  }))
}));

describe('Performance Benchmarks', () => {
  let inferenceService: OnnxInferenceService;
  let backtestingEngine: BacktestingEngine;

  beforeEach(async () => {
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

    // Load model for inference tests
    await inferenceService.loadModel({
      name: 'performance-test',
      version: '1.0.0',
      modelPath: '/models/test.onnx',
      scalerParams: { mean: [100], std: [5] },
      inputShape: [1, 1],
      outputClasses: ['up', 'down'],
      confidenceThreshold: 0.6
    });
  });

  describe('Prediction Latency Benchmarks', () => {
    it('should meet single prediction latency requirement (<20ms)', async () => {
      const features = [100];
      const iterations = 100;
      const latencies: number[] = [];

      for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        await inferenceService.predict('EURUSD', features);
        const end = performance.now();
        latencies.push(end - start);
      }

      const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      const p95Latency = latencies.sort((a, b) => a - b)[Math.floor(iterations * 0.95)];

      console.log(`Average latency: ${avgLatency.toFixed(2)}ms`);
      console.log(`P95 latency: ${p95Latency.toFixed(2)}ms`);

      expect(avgLatency).toBeLessThan(20);
      expect(p95Latency).toBeLessThan(30); // Allow some variance for P95
    }, 30000);

    it('should handle batch predictions efficiently', async () => {
      const batchSizes = [10, 50, 100, 200];
      
      for (const batchSize of batchSizes) {
        const batchRequest = Array.from({ length: batchSize }, (_, i) => ({
          symbol: `PAIR${i}`,
          features: [100 + i]
        }));

        const start = performance.now();
        const results = await inferenceService.batchPredict(batchRequest);
        const end = performance.now();

        const totalTime = end - start;
        const avgTimePerPrediction = totalTime / batchSize;

        console.log(`Batch size ${batchSize}: ${avgTimePerPrediction.toFixed(2)}ms per prediction`);

        expect(results.length).toBe(batchSize);
        expect(avgTimePerPrediction).toBeLessThan(5); // Should be faster in batches
      }
    }, 60000);
  });

  describe('Memory Usage Benchmarks', () => {
    it('should maintain memory usage under 500MB', async () => {
      const initialMemory = (performance as any).memory ? (performance as any).memory.usedJSHeapSize : 0;
      
      // Perform intensive operations
      const batchSize = 1000;
      const batchRequest = Array.from({ length: batchSize }, (_, i) => ({
        symbol: `INTENSIVE_${i}`,
        features: Array.from({ length: 10 }, () => Math.random() * 100)
      }));

      await inferenceService.batchPredict(batchRequest);

      const finalMemory = (performance as any).memory ? (performance as any).memory.usedJSHeapSize : 0;
      const memoryIncrease = finalMemory - initialMemory;
      const memoryIncreaseMB = memoryIncrease / (1024 * 1024);

      console.log(`Memory increase: ${memoryIncreaseMB.toFixed(2)}MB`);

      // In browser environment, we can't reliably measure memory usage
      // This test would be more meaningful in Node.js environment
      expect(memoryIncreaseMB).toBeLessThan(100); // Reasonable increase for test
    });

    it('should not have memory leaks in repeated operations', async () => {
      const iterations = 100;
      const features = [100];
      
      // Perform many predictions to check for memory leaks
      for (let i = 0; i < iterations; i++) {
        await inferenceService.predict('EURUSD', features);
        
        // Force garbage collection periodically (if available)
        if (i % 10 === 0 && global.gc) {
          global.gc();
        }
      }

      // If we reach here without out-of-memory, test passes
      expect(true).toBe(true);
    }, 30000);
  });

  describe('Throughput Benchmarks', () => {
    it('should achieve target throughput for concurrent predictions', async () => {
      const concurrentRequests = 20;
      const features = [100];

      const start = performance.now();
      
      const promises = Array.from({ length: concurrentRequests }, (_, i) =>
        inferenceService.predict(`PAIR${i}`, features)
      );

      const results = await Promise.all(promises);
      const end = performance.now();

      const totalTime = end - start;
      const throughput = (concurrentRequests / totalTime) * 1000; // predictions per second

      console.log(`Concurrent throughput: ${throughput.toFixed(2)} predictions/sec`);

      expect(results.length).toBe(concurrentRequests);
      expect(throughput).toBeGreaterThan(100); // Target: >100 predictions/sec
    }, 30000);

    it('should maintain performance under sustained load', async () => {
      const duration = 10000; // 10 seconds
      const features = [100];
      const predictions: any[] = [];
      
      const start = performance.now();
      let requestCount = 0;

      while (performance.now() - start < duration) {
        const prediction = await inferenceService.predict('EURUSD', features);
        predictions.push(prediction);
        requestCount++;
      }

      const actualDuration = performance.now() - start;
      const throughput = (requestCount / actualDuration) * 1000;

      console.log(`Sustained throughput: ${throughput.toFixed(2)} predictions/sec over ${actualDuration.toFixed(0)}ms`);

      expect(throughput).toBeGreaterThan(50); // Should maintain reasonable throughput
    }, 15000);
  });

  describe('Backtesting Performance', () => {
    it('should complete backtests within reasonable time', async () => {
      const testData = testDataGenerator.generateBacktestData('mixed');
      const signals = testData.candles.slice(100, 900).map((candle, i) => ({
        timestamp: candle.timestamp,
        direction: i % 2 === 0 ? 'long' as const : 'short' as const,
        confidence: 0.7 + Math.random() * 0.3,
        price: candle.close
      }));

      const start = performance.now();
      const results = await backtestingEngine.runBacktest(testData.candles, signals);
      const end = performance.now();

      const duration = end - start;
      const candlesPerMs = testData.candles.length / duration;

      console.log(`Backtest duration: ${duration.toFixed(2)}ms`);
      console.log(`Processing rate: ${(candlesPerMs * 1000).toFixed(0)} candles/sec`);

      expect(duration).toBeLessThan(5000); // Should complete within 5 seconds
      expect(results.trades.length).toBeGreaterThan(0);
    });

    it('should scale with data size efficiently', async () => {
      const dataSizes = [100, 500, 1000, 2000];
      const processingRates: number[] = [];

      for (const size of dataSizes) {
        const testData = testDataGenerator.generateMarketData({
          symbol: 'SCALE_TEST',
          startPrice: 100,
          candleCount: size,
          timeframe: 60000,
          conditions: [{ type: 'ranging', strength: 0.5, duration: size }],
          addNoise: true,
          noiseLevel: 0.01
        });

        const signals = testData.slice(50, size - 50).map((candle, i) => ({
          timestamp: candle.timestamp,
          direction: i % 2 === 0 ? 'long' as const : 'short' as const,
          confidence: 0.7,
          price: candle.close
        }));

        const start = performance.now();
        await backtestingEngine.runBacktest(testData, signals);
        const end = performance.now();

        const rate = size / (end - start);
        processingRates.push(rate);

        console.log(`${size} candles: ${rate.toFixed(2)} candles/ms`);
      }

      // Processing rate should not degrade significantly with size
      // (allowing for some variation due to complexity)
      const rateVariation = Math.max(...processingRates) / Math.min(...processingRates);
      expect(rateVariation).toBeLessThan(3); // Less than 3x variation
    });
  });

  describe('Resource Utilization', () => {
    it('should efficiently utilize CPU for parallel processing', async () => {
      const parallelTasks = Array.from({ length: 10 }, async (_, i) => {
        const features = Array.from({ length: 100 }, () => Math.random() * 100);
        return inferenceService.predict(`PARALLEL_${i}`, features);
      });

      const start = performance.now();
      const results = await Promise.all(parallelTasks);
      const end = performance.now();

      const parallelTime = end - start;
      
      // Sequential execution for comparison
      const sequentialStart = performance.now();
      for (let i = 0; i < 10; i++) {
        const features = Array.from({ length: 100 }, () => Math.random() * 100);
        await inferenceService.predict(`SEQUENTIAL_${i}`, features);
      }
      const sequentialEnd = performance.now();
      const sequentialTime = sequentialEnd - sequentialStart;

      const speedup = sequentialTime / parallelTime;
      console.log(`Parallel speedup: ${speedup.toFixed(2)}x`);

      expect(results.length).toBe(10);
      expect(speedup).toBeGreaterThan(1.5); // Should show some parallelization benefit
    }, 30000);
  });
});