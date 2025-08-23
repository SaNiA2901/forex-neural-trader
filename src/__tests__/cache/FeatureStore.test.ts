/**
 * Feature Store Integration Tests
 * Tests Redis caching functionality with mock Redis
 */

import { FeatureStore, FeatureVector } from '../../services/cache/FeatureStore';
import { RedisClientManager } from '../../services/cache/RedisClient';

// Mock Redis client
const mockRedisClient = {
  connect: jest.fn().mockResolvedValue(undefined),
  quit: jest.fn().mockResolvedValue(undefined),
  ping: jest.fn().mockResolvedValue('PONG'),
  get: jest.fn(),
  setEx: jest.fn(),
  multi: jest.fn(),
  keys: jest.fn(),
  del: jest.fn(),
  info: jest.fn()
};

// Mock pipeline
const mockPipeline = {
  get: jest.fn().mockReturnThis(),
  setEx: jest.fn().mockReturnThis(),
  exec: jest.fn()
};

mockRedisClient.multi.mockReturnValue(mockPipeline);

// Mock Redis client manager
class MockRedisManager extends RedisClientManager {
  constructor() {
    super();
  }

  getClient(): any {
    return mockRedisClient;
  }

  async executeCommand<T>(operation: () => Promise<T>): Promise<T | null> {
    try {
      return await operation();
    } catch (error) {
      return null;
    }
  }

  async healthCheck(): Promise<boolean> {
    return true;
  }

  isConnected(): boolean {
    return true;
  }

  getMetrics() {
    return {
      connectionStatus: 'connected' as const,
      totalCommands: 0,
      successfulCommands: 0,
      failedCommands: 0,
      averageResponseTime: 0,
      uptime: 0
    };
  }
}

describe('FeatureStore', () => {
  let featureStore: FeatureStore;
  let mockRedis: MockRedisManager;

  beforeEach(() => {
    mockRedis = new MockRedisManager();
    featureStore = new FeatureStore(mockRedis);
    jest.clearAllMocks();
  });

  const createTestFeatureVector = (symbol: string = 'EURUSD', timestamp: number = Date.now()): FeatureVector => ({
    symbol,
    timestamp,
    indicators: {
      rsi: 65.5,
      macd: { line: 0.0012, signal: 0.0008, histogram: 0.0004 },
      bollingerBands: { upper: 1.1050, middle: 1.1000, lower: 1.0950 },
      ema: { ema12: 1.1020, ema26: 1.1010 },
      stochastic: { k: 75.2, d: 72.8 },
      atr: 0.0025,
      adx: 45.3
    },
    patterns: [],
    metadata: {
      version: '1.0.0',
      computedAt: Date.now(),
      dataPoints: 100,
      confidence: 0.95
    }
  });

  describe('Basic Operations', () => {
    test('should store and retrieve feature vector', async () => {
      const features = createTestFeatureVector();
      const serialized = JSON.stringify(features);

      // Mock successful set
      mockRedisClient.setEx.mockResolvedValue('OK');
      
      // Mock successful get
      mockRedisClient.get.mockResolvedValue(serialized);

      // Test set operation
      const setResult = await featureStore.set(features);
      expect(setResult).toBe(true);
      expect(mockRedisClient.setEx).toHaveBeenCalled();

      // Test get operation
      const retrieved = await featureStore.get(features.symbol, features.timestamp);
      expect(retrieved).toEqual(features);
      expect(mockRedisClient.get).toHaveBeenCalled();
    });

    test('should return null for non-existent key', async () => {
      mockRedisClient.get.mockResolvedValue(null);

      const result = await featureStore.get('NONEXISTENT', Date.now());
      expect(result).toBeNull();
    });

    test('should handle serialization errors gracefully', async () => {
      mockRedisClient.get.mockResolvedValue('invalid json');

      const result = await featureStore.get('EURUSD', Date.now());
      expect(result).toBeNull();
    });
  });

  describe('Batch Operations', () => {
    test('should handle batch retrieval', async () => {
      const timestamps = [1000, 2000, 3000];
      const features = timestamps.map(ts => createTestFeatureVector('EURUSD', ts));
      
      // Mock pipeline responses
      mockPipeline.exec.mockResolvedValue([
        JSON.stringify(features[0]),
        null,
        JSON.stringify(features[2])
      ]);

      const results = await featureStore.getBatch([{
        symbol: 'EURUSD',
        timestamps
      }]);

      expect(results).toHaveLength(1);
      const result = results[0];
      expect(result.cached).toEqual([1000, 3000]);
      expect(result.computed).toEqual([2000]);
      expect(result.features.size).toBe(2);
    });

    test('should handle batch storage', async () => {
      const features = [
        createTestFeatureVector('EURUSD', 1000),
        createTestFeatureVector('EURUSD', 2000)
      ];

      mockPipeline.exec.mockResolvedValue(['OK', 'OK']);

      const result = await featureStore.setBatch(features);
      expect(result.successful).toBe(2);
      expect(result.failed).toBe(0);
      expect(result.errors).toHaveLength(0);
    });
  });

  describe('Cache Management', () => {
    test('should invalidate keys for symbol', async () => {
      mockRedisClient.keys.mockResolvedValue([
        'trading:features:EURUSD:1000:v1.0.0',
        'trading:features:EURUSD:2000:v1.0.0'
      ]);
      mockRedisClient.del.mockResolvedValue(2);

      const deleted = await featureStore.invalidate('EURUSD');
      expect(deleted).toBe(2);
      expect(mockRedisClient.keys).toHaveBeenCalledWith('trading:features:EURUSD:*');
    });

    test('should return cache statistics', async () => {
      mockRedisClient.keys.mockResolvedValue(['key1', 'key2', 'key3']);
      mockRedisClient.info.mockResolvedValue('used_memory_human:2.50M\r\nused_memory:2621440\r\n');

      const stats = await featureStore.getCacheStats();
      expect(stats.totalKeys).toBe(3);
      expect(stats.memoryUsage).toBe('2.50M');
      expect(stats.metrics).toBeDefined();
    });
  });

  describe('TTL Management', () => {
    test('should use different TTL for real-time vs historical data', async () => {
      const now = Date.now();
      const realtimeFeatures = createTestFeatureVector('EURUSD', now);
      const historicalFeatures = createTestFeatureVector('EURUSD', now - 2 * 60 * 60 * 1000); // 2 hours ago

      mockRedisClient.setEx.mockResolvedValue('OK');

      await featureStore.set(realtimeFeatures);
      await featureStore.set(historicalFeatures);

      // Should be called twice with different TTL values
      expect(mockRedisClient.setEx).toHaveBeenCalledTimes(2);
      
      const calls = mockRedisClient.setEx.mock.calls;
      expect(calls[0][1]).toBe(5 * 60); // 5 minutes for real-time
      expect(calls[1][1]).toBe(60 * 60); // 1 hour for historical
    });
  });

  describe('Error Handling', () => {
    test('should handle Redis connection errors', async () => {
      mockRedisClient.get.mockRejectedValue(new Error('Connection failed'));

      const result = await featureStore.get('EURUSD', Date.now());
      expect(result).toBeNull();
      
      const metrics = featureStore.getMetrics();
      expect(metrics.totalErrors).toBeGreaterThan(0);
    });

    test('should handle batch operation failures', async () => {
      mockPipeline.exec.mockRejectedValue(new Error('Pipeline failed'));

      const results = await featureStore.getBatch([{
        symbol: 'EURUSD',
        timestamps: [1000, 2000]
      }]);

      expect(results[0].errors.length).toBeGreaterThan(0);
      expect(results[0].computed).toEqual([1000, 2000]);
    });
  });

  describe('Health Monitoring', () => {
    test('should perform health check', async () => {
      mockRedisClient.setEx.mockResolvedValue('OK');
      mockRedisClient.get.mockResolvedValue('{"test":true}');
      mockRedisClient.del.mockResolvedValue(1);

      const health = await featureStore.healthCheck();
      expect(health.status).toBe('healthy');
      expect(health.redis).toBe(true);
      expect(health.cache).toBe(true);
    });

    test('should detect unhealthy cache', async () => {
      mockRedisClient.setEx.mockRejectedValue(new Error('Cache write failed'));

      const health = await featureStore.healthCheck();
      expect(health.status).not.toBe('healthy');
      expect(health.cache).toBe(false);
    });
  });

  describe('Metrics Tracking', () => {
    test('should track cache hit/miss rates', async () => {
      // Setup cache hits
      mockRedisClient.get
        .mockResolvedValueOnce(JSON.stringify(createTestFeatureVector()))
        .mockResolvedValueOnce(null)
        .mockResolvedValueOnce(JSON.stringify(createTestFeatureVector()));

      await featureStore.get('EURUSD', 1000);
      await featureStore.get('EURUSD', 2000);
      await featureStore.get('EURUSD', 3000);

      const metrics = featureStore.getMetrics();
      expect(metrics.totalRequests).toBe(3);
      expect(metrics.cacheHits).toBe(2);
      expect(metrics.cacheMisses).toBe(1);
      expect(metrics.hitRate).toBeCloseTo(66.67, 1);
    });
  });
});