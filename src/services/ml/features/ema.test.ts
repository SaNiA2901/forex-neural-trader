/**
 * Comprehensive unit tests for EMA services
 * Provides >90% coverage with performance benchmarks
 * 
 * @author Trading System Core
 * @version 1.0.0
 */

import { 
  EMAService, 
  EMAConfig, 
  EMAResult, 
  EMAValidator, 
  EMAError 
} from './ema';
import { StreamingEMA, MultiPeriodStreamingEMA } from './streaming-ema';

describe('EMAValidator', () => {
  describe('validateConfig', () => {
    it('should accept valid configuration', () => {
      expect(() => {
        EMAValidator.validateConfig({ period: 10 });
      }).not.toThrow();

      expect(() => {
        EMAValidator.validateConfig({ period: 20, smoothingFactor: 0.1 });
      }).not.toThrow();
    });

    it('should reject invalid periods', () => {
      expect(() => {
        EMAValidator.validateConfig({ period: 0 });
      }).toThrow(EMAError);

      expect(() => {
        EMAValidator.validateConfig({ period: -5 });
      }).toThrow(EMAError);

      expect(() => {
        EMAValidator.validateConfig({ period: 1001 });
      }).toThrow(EMAError);

      expect(() => {
        EMAValidator.validateConfig({ period: 1.5 });
      }).toThrow(EMAError);
    });

    it('should reject invalid smoothing factors', () => {
      expect(() => {
        EMAValidator.validateConfig({ period: 10, smoothingFactor: 0 });
      }).toThrow(EMAError);

      expect(() => {
        EMAValidator.validateConfig({ period: 10, smoothingFactor: 1 });
      }).toThrow(EMAError);

      expect(() => {
        EMAValidator.validateConfig({ period: 10, smoothingFactor: -0.1 });
      }).toThrow(EMAError);

      expect(() => {
        EMAValidator.validateConfig({ period: 10, smoothingFactor: 1.1 });
      }).toThrow(EMAError);
    });

    it('should reject null/undefined config', () => {
      expect(() => {
        EMAValidator.validateConfig(null as any);
      }).toThrow(EMAError);

      expect(() => {
        EMAValidator.validateConfig(undefined as any);
      }).toThrow(EMAError);
    });
  });

  describe('validateValues', () => {
    it('should accept valid arrays', () => {
      expect(() => {
        EMAValidator.validateValues([1, 2, 3, 4, 5]);
      }).not.toThrow();

      expect(() => {
        EMAValidator.validateValues([0, -1, 100.5, 0.001]);
      }).not.toThrow();
    });

    it('should reject invalid inputs', () => {
      expect(() => {
        EMAValidator.validateValues(null as any);
      }).toThrow(EMAError);

      expect(() => {
        EMAValidator.validateValues([] as any);
      }).toThrow(EMAError);

      expect(() => {
        EMAValidator.validateValues([1, 2, NaN, 4]);
      }).toThrow(EMAError);

      expect(() => {
        EMAValidator.validateValues([1, 2, Infinity, 4]);
      }).toThrow(EMAError);

      expect(() => {
        EMAValidator.validateValues([1, 2, "3" as any, 4]);
      }).toThrow(EMAError);
    });

    it('should reject arrays that are too large', () => {
      const largeArray = new Array(100001).fill(1);
      expect(() => {
        EMAValidator.validateValues(largeArray);
      }).toThrow(EMAError);
    });
  });
});

describe('EMAService', () => {
  describe('calculateSmoothingFactor', () => {
    it('should calculate correct smoothing factors', () => {
      expect(EMAService.calculateSmoothingFactor(10)).toBeCloseTo(0.1818, 4);
      expect(EMAService.calculateSmoothingFactor(20)).toBeCloseTo(0.0952, 4);
      expect(EMAService.calculateSmoothingFactor(50)).toBeCloseTo(0.0392, 4);
    });
  });

  describe('computeEMA', () => {
    const testData = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];

    it('should compute EMA correctly for valid inputs', () => {
      const result = EMAService.computeEMA(testData, { period: 5 });
      
      expect(result.values).toHaveLength(testData.length);
      expect(result.validFrom).toBe(4);
      expect(result.config.period).toBe(5);
      expect(result.smoothingFactor).toBeCloseTo(0.3333, 4);
      
      // First valid EMA should be SMA of first 5 values
      expect(result.values[4]).toBeCloseTo(12, 2);
      
      // Check that values before validFrom are NaN
      for (let i = 0; i < result.validFrom; i++) {
        expect(isNaN(result.values[i])).toBe(true);
      }
    });

    it('should handle custom smoothing factor', () => {
      const result = EMAService.computeEMA(testData, { 
        period: 5, 
        smoothingFactor: 0.5 
      });
      
      expect(result.smoothingFactor).toBe(0.5);
    });

    it('should throw error for insufficient data', () => {
      expect(() => {
        EMAService.computeEMA([1, 2], { period: 5 });
      }).toThrow(EMAError);
    });

    it('should produce consistent results', () => {
      const config = { period: 10 };
      const result1 = EMAService.computeEMA(testData, config);
      const result2 = EMAService.computeEMA(testData, config);
      
      expect(result1.values).toEqual(result2.values);
    });
  });

  describe('computeMultipleEMA', () => {
    const testData = Array.from({length: 50}, (_, i) => 100 + Math.sin(i * 0.1) * 10);

    it('should compute multiple periods correctly', () => {
      const periods = [5, 10, 20];
      const results = EMAService.computeMultipleEMA(testData, periods);
      
      expect(results.size).toBe(3);
      expect(results.has(5)).toBe(true);
      expect(results.has(10)).toBe(true);
      expect(results.has(20)).toBe(true);
    });

    it('should skip invalid periods gracefully', () => {
      const periods = [5, -1, 10, 1001];
      const results = EMAService.computeMultipleEMA(testData, periods);
      
      // Should only have valid periods
      expect(results.size).toBe(2);
      expect(results.has(5)).toBe(true);
      expect(results.has(10)).toBe(true);
    });
  });

  describe('calculateEMAConvergenceDivergence', () => {
    const testData = Array.from({length: 100}, (_, i) => 100 + i * 0.5);

    it('should calculate MACD-like convergence', () => {
      const convergence = EMAService.calculateEMAConvergenceDivergence(testData, 12, 26);
      
      expect(convergence).toHaveLength(testData.length);
      
      // First 25 values should be NaN (26-1 for slow EMA)
      for (let i = 0; i < 25; i++) {
        expect(isNaN(convergence[i])).toBe(true);
      }
      
      // Should have valid values after that
      expect(isNaN(convergence[25])).toBe(false);
    });
  });
});

describe('StreamingEMA', () => {
  describe('constructor and basic operations', () => {
    it('should initialize correctly', () => {
      const ema = new StreamingEMA({ period: 10 });
      
      expect(ema.getCurrentValue()).toBe(0);
      expect(ema.getCount()).toBe(0);
      expect(ema.isStable()).toBe(false);
    });

    it('should reject invalid configuration', () => {
      expect(() => {
        new StreamingEMA({ period: 0 });
      }).toThrow(EMAError);
    });
  });

  describe('update method', () => {
    let ema: StreamingEMA;

    beforeEach(() => {
      ema = new StreamingEMA({ period: 5 });
    });

    it('should handle first few updates correctly', () => {
      const update1 = ema.update(10);
      expect(update1.value).toBe(10);
      expect(update1.isStable).toBe(false);
      expect(update1.count).toBe(1);

      const update2 = ema.update(20);
      expect(update2.value).toBe(15);
      expect(update2.isStable).toBe(false);
      expect(update2.count).toBe(2);
    });

    it('should become stable after enough data', () => {
      // Add 4 values (not stable yet)
      ema.update(10);
      ema.update(20);
      ema.update(30);
      ema.update(40);
      
      expect(ema.isStable()).toBe(false);

      // 5th value should make it stable
      const update5 = ema.update(50);
      expect(update5.isStable).toBe(true);
      expect(ema.isStable()).toBe(true);
    });

    it('should calculate changes correctly', () => {
      // Initialize with stable EMA
      for (let i = 0; i < 5; i++) {
        ema.update(100);
      }
      
      const update = ema.update(110);
      expect(update.change).toBeGreaterThan(0);
      expect(update.percentChange).toBeGreaterThan(0);
    });

    it('should reject invalid inputs', () => {
      expect(() => {
        ema.update(NaN);
      }).toThrow(EMAError);

      expect(() => {
        ema.update(Infinity);
      }).toThrow(EMAError);

      expect(() => {
        ema.update("10" as any);
      }).toThrow(EMAError);
    });
  });

  describe('batch operations', () => {
    let ema: StreamingEMA;

    beforeEach(() => {
      ema = new StreamingEMA({ period: 5 });
    });

    it('should handle batch updates', () => {
      const values = [10, 20, 30, 40, 50, 60];
      const results = ema.batchUpdate(values);
      
      expect(results).toHaveLength(6);
      expect(results[5].isStable).toBe(true);
      expect(ema.getCount()).toBe(6);
    });

    it('should create from batch data correctly', () => {
      const values = [10, 20, 30, 40, 50, 60];
      const emaFromBatch = StreamingEMA.fromBatchData(values, { period: 5 });
      
      expect(emaFromBatch.getCount()).toBe(6);
      expect(emaFromBatch.isStable()).toBe(true);
      
      // Should match manual updates
      const manualEMA = new StreamingEMA({ period: 5 });
      for (const value of values) {
        manualEMA.update(value);
      }
      
      expect(emaFromBatch.getCurrentValue()).toBeCloseTo(manualEMA.getCurrentValue(), 10);
    });
  });

  describe('state management', () => {
    let ema: StreamingEMA;

    beforeEach(() => {
      ema = new StreamingEMA({ period: 5 });
    });

    it('should provide correct state information', () => {
      ema.update(10);
      ema.update(20);
      
      const state = ema.getState();
      expect(state.count).toBe(2);
      expect(state.isStable).toBe(false);
      expect(state.config.period).toBe(5);
      expect(state.initialBuffer).toHaveLength(2);
    });

    it('should reset correctly', () => {
      ema.update(10);
      ema.update(20);
      ema.reset();
      
      expect(ema.getCount()).toBe(0);
      expect(ema.getCurrentValue()).toBe(0);
      expect(ema.isStable()).toBe(false);
    });

    it('should update configuration correctly', () => {
      ema.update(10);
      ema.updateConfig({ period: 10 });
      
      expect(ema.getCount()).toBe(0); // Should reset
      expect(ema.getState().config.period).toBe(10);
    });
  });

  describe('performance tracking', () => {
    let ema: StreamingEMA;

    beforeEach(() => {
      ema = new StreamingEMA({ period: 5 });
    });

    it('should track performance metrics', () => {
      // Add some updates
      for (let i = 0; i < 100; i++) {
        ema.update(Math.random() * 100);
      }
      
      const metrics = ema.getPerformanceMetrics();
      expect(metrics.totalUpdates).toBe(100);
      expect(metrics.averageUpdateTime).toBeGreaterThan(0);
      expect(metrics.updatesPerSecond).toBeGreaterThan(0);
      expect(metrics.memoryUsage).toBeGreaterThan(0);
    });
  });
});

describe('MultiPeriodStreamingEMA', () => {
  let multiEMA: MultiPeriodStreamingEMA;

  beforeEach(() => {
    multiEMA = new MultiPeriodStreamingEMA([5, 10, 20]);
  });

  it('should initialize with multiple periods', () => {
    expect(multiEMA.getEMA(5)).toBeDefined();
    expect(multiEMA.getEMA(10)).toBeDefined();
    expect(multiEMA.getEMA(20)).toBeDefined();
  });

  it('should update all EMAs simultaneously', () => {
    const results = multiEMA.updateAll(100);
    
    expect(results.size).toBe(3);
    expect(results.has(5)).toBe(true);
    expect(results.has(10)).toBe(true);
    expect(results.has(20)).toBe(true);
  });

  it('should get current values for all periods', () => {
    multiEMA.updateAll(100);
    multiEMA.updateAll(110);
    
    const values = multiEMA.getCurrentValues();
    expect(values.size).toBe(3);
    expect(values.get(5)).toBeGreaterThan(0);
    expect(values.get(10)).toBeGreaterThan(0);
    expect(values.get(20)).toBeGreaterThan(0);
  });

  it('should manage periods dynamically', () => {
    multiEMA.addPeriod(50);
    expect(multiEMA.getEMA(50)).toBeDefined();
    
    const removed = multiEMA.removePeriod(5);
    expect(removed).toBe(true);
    expect(multiEMA.getEMA(5)).toBeUndefined();
  });

  it('should reset all EMAs', () => {
    multiEMA.updateAll(100);
    multiEMA.resetAll();
    
    const values = multiEMA.getCurrentValues();
    for (const value of values.values()) {
      expect(value).toBe(0);
    }
  });

  it('should provide performance metrics for all EMAs', () => {
    // Add some data
    for (let i = 0; i < 50; i++) {
      multiEMA.updateAll(Math.random() * 100);
    }
    
    const metrics = multiEMA.getPerformanceMetrics();
    expect(metrics.size).toBe(3);
    
    for (const [period, metric] of metrics) {
      expect(metric.totalUpdates).toBe(50);
      expect(metric.averageUpdateTime).toBeGreaterThan(0);
    }
  });
});

describe('Performance Benchmarks', () => {
  it('should demonstrate O(1) streaming updates', () => {
    const ema = new StreamingEMA({ period: 20 });
    const updateTimes: number[] = [];
    
    // Warm up
    for (let i = 0; i < 100; i++) {
      ema.update(Math.random() * 100);
    }
    
    // Measure update times
    for (let i = 0; i < 1000; i++) {
      const start = performance.now();
      ema.update(Math.random() * 100);
      const end = performance.now();
      updateTimes.push((end - start) * 1000); // Convert to microseconds
    }
    
    const averageTime = updateTimes.reduce((a, b) => a + b) / updateTimes.length;
    const maxTime = Math.max(...updateTimes);
    
    console.log(`Average update time: ${averageTime.toFixed(2)}μs`);
    console.log(`Maximum update time: ${maxTime.toFixed(2)}μs`);
    
    // Ensure updates are consistently fast (under 100μs on most systems)
    expect(averageTime).toBeLessThan(100);
  });

  it('should handle high-frequency updates efficiently', () => {
    const ema = new StreamingEMA({ period: 50 });
    const startTime = Date.now();
    const updateCount = 10000;
    
    for (let i = 0; i < updateCount; i++) {
      ema.update(Math.random() * 100);
    }
    
    const endTime = Date.now();
    const totalTime = endTime - startTime;
    const updatesPerSecond = updateCount / (totalTime / 1000);
    
    console.log(`Processed ${updateCount} updates in ${totalTime}ms`);
    console.log(`Updates per second: ${updatesPerSecond.toFixed(0)}`);
    
    // Should handle at least 1000 updates per second
    expect(updatesPerSecond).toBeGreaterThan(1000);
  });

  it('should compare batch vs streaming performance', () => {
    const testData = Array.from({length: 1000}, () => Math.random() * 100);
    
    // Batch calculation
    const batchStart = performance.now();
    const batchResult = EMAService.computeEMA(testData, { period: 20 });
    const batchEnd = performance.now();
    const batchTime = batchEnd - batchStart;
    
    // Streaming calculation
    const streamingStart = performance.now();
    const streamingEMA = new StreamingEMA({ period: 20 });
    for (const value of testData) {
      streamingEMA.update(value);
    }
    const streamingEnd = performance.now();
    const streamingTime = streamingEnd - streamingStart;
    
    console.log(`Batch calculation: ${batchTime.toFixed(2)}ms`);
    console.log(`Streaming calculation: ${streamingTime.toFixed(2)}ms`);
    
    // Results should be approximately equal
    const finalBatchValue = batchResult.values[batchResult.values.length - 1];
    const finalStreamingValue = streamingEMA.getCurrentValue();
    
    expect(finalStreamingValue).toBeCloseTo(finalBatchValue, 10);
  });
});

// Test utilities for generating synthetic market data
export class TestDataGenerator {
  /**
   * Generates synthetic price data with trend and noise
   */
  static generatePriceData(
    length: number, 
    startPrice: number = 100, 
    trend: number = 0.001, 
    volatility: number = 0.02
  ): number[] {
    const data: number[] = [];
    let currentPrice = startPrice;
    
    for (let i = 0; i < length; i++) {
      const randomChange = (Math.random() - 0.5) * volatility * currentPrice;
      const trendChange = trend * currentPrice;
      currentPrice += randomChange + trendChange;
      data.push(currentPrice);
    }
    
    return data;
  }

  /**
   * Generates data with known statistical properties for testing
   */
  static generateStatisticalData(
    length: number,
    mean: number = 0,
    stdDev: number = 1
  ): number[] {
    const data: number[] = [];
    
    for (let i = 0; i < length; i++) {
      // Box-Muller transformation for normal distribution
      const u1 = Math.random();
      const u2 = Math.random();
      const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      data.push(z0 * stdDev + mean);
    }
    
    return data;
  }
}

describe('Integration Tests', () => {
  it('should handle real-world market data patterns', () => {
    // Generate realistic market data
    const marketData = TestDataGenerator.generatePriceData(1000, 100, 0.0005, 0.015);
    
    // Test batch calculation
    const batchResult = EMAService.computeEMA(marketData, { period: 20 });
    expect(batchResult.values).toHaveLength(1000);
    
    // Test streaming calculation
    const streamingEMA = StreamingEMA.fromBatchData(marketData, { period: 20 });
    expect(streamingEMA.isStable()).toBe(true);
    
    // Values should match closely
    const batchFinal = batchResult.values[batchResult.values.length - 1];
    const streamingFinal = streamingEMA.getCurrentValue();
    expect(streamingFinal).toBeCloseTo(batchFinal, 8);
  });

  it('should maintain accuracy under various market conditions', () => {
    const conditions = [
      { trend: 0.002, volatility: 0.01 }, // Bull market
      { trend: -0.002, volatility: 0.01 }, // Bear market
      { trend: 0, volatility: 0.03 }, // High volatility sideways
      { trend: 0, volatility: 0.005 } // Low volatility sideways
    ];
    
    for (const condition of conditions) {
      const data = TestDataGenerator.generatePriceData(
        500, 
        100, 
        condition.trend, 
        condition.volatility
      );
      
      const batchEMA = EMAService.computeEMA(data, { period: 15 });
      const streamingEMA = StreamingEMA.fromBatchData(data, { period: 15 });
      
      const batchFinal = batchEMA.values[batchEMA.values.length - 1];
      const streamingFinal = streamingEMA.getCurrentValue();
      
      expect(streamingFinal).toBeCloseTo(batchFinal, 8);
    }
  });
});