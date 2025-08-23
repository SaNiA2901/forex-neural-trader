/**
 * Feature Store Service with Redis Caching
 * High-performance caching for technical indicators and feature vectors
 */

import { redisClient, RedisClientManager } from './RedisClient';
import { TechnicalIndicators } from '@/services/indicators/TechnicalIndicators';
import { CandleData } from '@/types/session';

export interface FeatureVector {
  symbol: string;
  timestamp: number;
  indicators: TechnicalIndicators;
  patterns: any[];
  metadata: {
    version: string;
    computedAt: number;
    dataPoints: number;
    confidence: number;
  };
}

export interface CacheOptions {
  ttl?: number;
  version?: string;
  compress?: boolean;
  namespace?: string;
}

export interface FeatureStoreMetrics {
  totalRequests: number;
  cacheHits: number;
  cacheMisses: number;
  hitRate: number;
  averageComputeTime: number;
  averageCacheTime: number;
  totalErrors: number;
  lastError?: string;
}

export interface BatchRequest {
  symbol: string;
  timestamps: number[];
  options?: CacheOptions;
}

export interface BatchResult {
  symbol: string;
  features: Map<number, FeatureVector>;
  cached: number[];
  computed: number[];
  errors: Array<{ timestamp: number; error: string }>;
}

export class FeatureStore {
  private redis: RedisClientManager;
  private metrics: FeatureStoreMetrics;
  private readonly VERSION = '1.0.0';
  
  // Cache TTL configurations (in seconds)
  private readonly TTL_CONFIGS = {
    REAL_TIME: 5 * 60,      // 5 minutes for real-time data
    HISTORICAL: 60 * 60,     // 1 hour for historical data
    PATTERNS: 30 * 60,       // 30 minutes for pattern analysis
    INDICATORS: 15 * 60      // 15 minutes for technical indicators
  };

  constructor(redisManager?: RedisClientManager) {
    this.redis = redisManager || redisClient;
    this.metrics = {
      totalRequests: 0,
      cacheHits: 0,
      cacheMisses: 0,
      hitRate: 0,
      averageComputeTime: 0,
      averageCacheTime: 0,
      totalErrors: 0
    };
  }

  /**
   * Generate cache key with versioning and namespacing
   */
  private generateKey(
    symbol: string, 
    timestamp: number, 
    type: string = 'features',
    namespace: string = 'trading'
  ): string {
    return `${namespace}:${type}:${symbol}:${timestamp}:v${this.VERSION}`;
  }

  /**
   * Serialize feature vector for Redis storage
   */
  private serialize(data: FeatureVector): string {
    return JSON.stringify(data);
  }

  /**
   * Deserialize feature vector from Redis
   */
  private deserialize(data: string): FeatureVector | null {
    try {
      return JSON.parse(data);
    } catch (error) {
      console.error('Failed to deserialize feature vector:', error);
      return null;
    }
  }

  /**
   * Determine appropriate TTL based on data age and type
   */
  private getTTL(timestamp: number, options?: CacheOptions): number {
    if (options?.ttl) return options.ttl;

    const now = Date.now();
    const ageMinutes = (now - timestamp) / (1000 * 60);

    if (ageMinutes < 5) return this.TTL_CONFIGS.REAL_TIME;
    if (ageMinutes < 60) return this.TTL_CONFIGS.INDICATORS;
    return this.TTL_CONFIGS.HISTORICAL;
  }

  /**
   * Get single feature vector from cache
   */
  async get(
    symbol: string, 
    timestamp: number, 
    options?: CacheOptions
  ): Promise<FeatureVector | null> {
    const startTime = performance.now();
    this.metrics.totalRequests++;

    try {
      const key = this.generateKey(symbol, timestamp, 'features', options?.namespace);
      
      const cached = await this.redis.executeCommand(
        () => this.redis.getClient().get(key),
        'GET'
      );

      const cacheTime = performance.now() - startTime;

      if (cached && typeof cached === 'string') {
        this.metrics.cacheHits++;
        this.updateAverageCacheTime(cacheTime);
        
        const features = this.deserialize(cached);
        if (features && this.isValidFeatureVector(features)) {
          return features;
        }
      }

      this.metrics.cacheMisses++;
      return null;

    } catch (error) {
      this.handleError('Cache get failed', error);
      return null;
    } finally {
      this.updateHitRate();
    }
  }

  /**
   * Store feature vector in cache
   */
  async set(
    featureVector: FeatureVector,
    options?: CacheOptions
  ): Promise<boolean> {
    try {
      const key = this.generateKey(
        featureVector.symbol, 
        featureVector.timestamp, 
        'features',
        options?.namespace
      );
      
      const ttl = this.getTTL(featureVector.timestamp, options);
      const serialized = this.serialize(featureVector);

      const success = await this.redis.executeCommand(
        () => this.redis.getClient().setEx(key, ttl, serialized),
        'SETEX'
      );

      return success === 'OK';

    } catch (error) {
      this.handleError('Cache set failed', error);
      return false;
    }
  }

  /**
   * Batch retrieve multiple feature vectors
   */
  async getBatch(requests: BatchRequest[]): Promise<BatchResult[]> {
    const results: BatchResult[] = [];

    for (const request of requests) {
      const result: BatchResult = {
        symbol: request.symbol,
        features: new Map(),
        cached: [],
        computed: [],
        errors: []
      };

      // Prepare Redis pipeline for batch operation
      const pipeline = this.redis.getClient().multi();
      const keys = request.timestamps.map(ts => 
        this.generateKey(request.symbol, ts, 'features', request.options?.namespace)
      );

      // Add all GET operations to pipeline
      keys.forEach(key => pipeline.get(key));

      try {
        const responses = await this.redis.executeCommand(
          () => pipeline.exec(),
          'PIPELINE_GET_BATCH'
        );

        if (responses) {
          responses.forEach((response: any, index: number) => {
            const timestamp = request.timestamps[index];
            
            if (response && typeof response === 'string') {
              const features = this.deserialize(response);
              if (features && this.isValidFeatureVector(features)) {
                result.features.set(timestamp, features);
                result.cached.push(timestamp);
                this.metrics.cacheHits++;
              } else {
                result.computed.push(timestamp);
                this.metrics.cacheMisses++;
              }
            } else {
              result.computed.push(timestamp);
              this.metrics.cacheMisses++;
            }
          });
        }

      } catch (error) {
        // If batch fails, mark all as needing computation
        request.timestamps.forEach(timestamp => {
          result.computed.push(timestamp);
          result.errors.push({
            timestamp,
            error: `Batch cache retrieval failed: ${error}`
          });
        });
        this.handleError('Batch get failed', error);
      }

      results.push(result);
      this.metrics.totalRequests += request.timestamps.length;
    }

    this.updateHitRate();
    return results;
  }

  /**
   * Batch store multiple feature vectors
   */
  async setBatch(
    featureVectors: FeatureVector[],
    options?: CacheOptions
  ): Promise<{ successful: number; failed: number; errors: string[] }> {
    const pipeline = this.redis.getClient().multi();
    const errors: string[] = [];
    let successful = 0;

    try {
      // Add all SET operations to pipeline
      featureVectors.forEach(features => {
        try {
          const key = this.generateKey(
            features.symbol, 
            features.timestamp, 
            'features',
            options?.namespace
          );
          const ttl = this.getTTL(features.timestamp, options);
          const serialized = this.serialize(features);
          
          pipeline.setEx(key, ttl, serialized);
        } catch (error) {
          errors.push(`Failed to prepare ${features.symbol}@${features.timestamp}: ${error}`);
        }
      });

      const responses = await this.redis.executeCommand(
        () => pipeline.exec(),
        'PIPELINE_SET_BATCH'
      );

      if (responses) {
        responses.forEach((response: any, index: number) => {
          if (response === 'OK') {
            successful++;
          } else {
            const features = featureVectors[index];
            errors.push(`Failed to cache ${features.symbol}@${features.timestamp}`);
          }
        });
      }

    } catch (error) {
      this.handleError('Batch set failed', error);
      errors.push(`Pipeline execution failed: ${error}`);
    }

    return {
      successful,
      failed: featureVectors.length - successful,
      errors
    };
  }

  /**
   * Invalidate cached features for a symbol
   */
  async invalidate(symbol: string, namespace: string = 'trading'): Promise<number> {
    try {
      const pattern = `${namespace}:features:${symbol}:*`;
      
      const keys = await this.redis.executeCommand(
        () => this.redis.getClient().keys(pattern),
        'KEYS'
      );

      if (keys && keys.length > 0) {
        const deleted = await this.redis.executeCommand(
          () => this.redis.getClient().del(keys),
          'DEL_BATCH'
        );
        return deleted || 0;
      }

      return 0;

    } catch (error) {
      this.handleError('Cache invalidation failed', error);
      return 0;
    }
  }

  /**
   * Get cache statistics for monitoring
   */
  async getCacheStats(namespace: string = 'trading'): Promise<{
    totalKeys: number;
    memoryUsage: string;
    metrics: FeatureStoreMetrics;
    redisMetrics: any;
  }> {
    try {
      const pattern = `${namespace}:features:*`;
      
      const [keys, info] = await Promise.all([
        this.redis.executeCommand(
          () => this.redis.getClient().keys(pattern),
          'KEYS'
        ),
        this.redis.executeCommand(
          () => this.redis.getClient().info('memory'),
          'INFO'
        )
      ]);

      const memoryInfo = info ? this.parseRedisInfo(info) : {};

      return {
        totalKeys: keys ? keys.length : 0,
        memoryUsage: memoryInfo.used_memory_human || 'Unknown',
        metrics: this.metrics,
        redisMetrics: this.redis.getMetrics()
      };

    } catch (error) {
      this.handleError('Stats retrieval failed', error);
      return {
        totalKeys: 0,
        memoryUsage: 'Unknown',
        metrics: this.metrics,
        redisMetrics: this.redis.getMetrics()
      };
    }
  }

  /**
   * Health check for the feature store
   */
  async healthCheck(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    redis: boolean;
    cache: boolean;
    details: any;
  }> {
    const redisHealthy = await this.redis.healthCheck();
    
    // Test cache operations
    let cacheHealthy = false;
    try {
      const testKey = `health:test:${Date.now()}`;
      const testValue = { test: true, timestamp: Date.now() };
      
      await this.redis.executeCommand(
        () => this.redis.getClient().setEx(testKey, 10, JSON.stringify(testValue)),
        'HEALTH_SET'
      );
      
      const retrieved = await this.redis.executeCommand(
        () => this.redis.getClient().get(testKey),
        'HEALTH_GET'
      );
      
      cacheHealthy = retrieved !== null;
      
      // Cleanup
      await this.redis.executeCommand(
        () => this.redis.getClient().del(testKey),
        'HEALTH_CLEANUP'
      );
      
    } catch (error) {
      console.error('Cache health check failed:', error);
    }

    const overall = redisHealthy && cacheHealthy ? 'healthy' : 
                   redisHealthy || cacheHealthy ? 'degraded' : 'unhealthy';

    return {
      status: overall,
      redis: redisHealthy,
      cache: cacheHealthy,
      details: {
        metrics: this.metrics,
        redisMetrics: this.redis.getMetrics()
      }
    };
  }

  // Private helper methods
  private isValidFeatureVector(features: any): features is FeatureVector {
    return features && 
           typeof features.symbol === 'string' &&
           typeof features.timestamp === 'number' &&
           features.indicators &&
           features.metadata;
  }

  private updateHitRate(): void {
    const total = this.metrics.cacheHits + this.metrics.cacheMisses;
    this.metrics.hitRate = total > 0 ? (this.metrics.cacheHits / total) * 100 : 0;
  }

  private updateAverageCacheTime(duration: number): void {
    const samples = 100; // Keep rolling average of last 100 samples
    this.metrics.averageCacheTime = 
      (this.metrics.averageCacheTime * (samples - 1) + duration) / samples;
  }

  private handleError(context: string, error: any): void {
    this.metrics.totalErrors++;
    this.metrics.lastError = `${context}: ${error.message || error}`;
    console.error(context, error);
  }

  private parseRedisInfo(info: string): Record<string, string> {
    const result: Record<string, string> = {};
    info.split('\r\n').forEach(line => {
      const [key, value] = line.split(':');
      if (key && value) {
        result[key] = value;
      }
    });
    return result;
  }

  // Getter methods
  getMetrics(): FeatureStoreMetrics {
    return { ...this.metrics };
  }

  getVersion(): string {
    return this.VERSION;
  }
}

// Singleton instance
export const featureStore = new FeatureStore();