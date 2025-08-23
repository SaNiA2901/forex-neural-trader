/**
 * Feature Store Integration Service
 * Connects technical indicators with Redis caching
 */

import { TechnicalIndicatorService, TechnicalIndicators } from '@/services/indicators/TechnicalIndicators';
import { featureStore, FeatureVector, CacheOptions } from './FeatureStore';
import { CandleData } from '@/types/session';

export interface ComputeOptions extends CacheOptions {
  forceRecompute?: boolean;
  includePatterns?: boolean;
  confidenceThreshold?: number;
}

export interface FeatureComputeResult {
  features: FeatureVector;
  source: 'cache' | 'computed';
  computeTime?: number;
  cacheTime?: number;
}

export interface BatchComputeRequest {
  symbol: string;
  candles: CandleData[];
  indices: number[];
  options?: ComputeOptions;
}

export interface BatchComputeResult {
  symbol: string;
  results: Map<number, FeatureComputeResult>;
  totalCached: number;
  totalComputed: number;
  totalTime: number;
  errors: Array<{ index: number; error: string }>;
}

export class FeatureIntegrationService {
  private readonly VERSION = '1.0.0';

  /**
   * Get or compute feature vector for a specific candle
   */
  async getFeatures(
    symbol: string,
    candles: CandleData[],
    currentIndex: number,
    options?: ComputeOptions
  ): Promise<FeatureComputeResult> {
    const startTime = performance.now();
    const timestamp = candles[currentIndex]?.timestamp;

    if (!timestamp) {
      throw new Error(`Invalid candle index: ${currentIndex}`);
    }

    try {
      // Check cache first (unless force recompute)
      if (!options?.forceRecompute) {
        const cacheStartTime = performance.now();
        const cached = await featureStore.get(symbol, new Date(timestamp).getTime(), options);
        const cacheTime = performance.now() - cacheStartTime;

        if (cached && this.isValidCachedFeatures(cached, options)) {
          return {
            features: cached,
            source: 'cache',
            cacheTime
          };
        }
      }

      // Compute features
      const computeStartTime = performance.now();
      const features = await this.computeFeatures(
        symbol, 
        candles, 
        currentIndex, 
        options
      );
      const computeTime = performance.now() - computeStartTime;

      // Cache the result
      await featureStore.set(features, options);

      return {
        features,
        source: 'computed',
        computeTime
      };

    } catch (error) {
      throw new Error(`Failed to get features for ${symbol}@${currentIndex}: ${error}`);
    }
  }

  /**
   * Batch compute features for multiple indices
   */
  async getBatchFeatures(
    requests: BatchComputeRequest[]
  ): Promise<BatchComputeResult[]> {
    const results: BatchComputeResult[] = [];

    for (const request of requests) {
      const batchResult: BatchComputeResult = {
        symbol: request.symbol,
        results: new Map(),
        totalCached: 0,
        totalComputed: 0,
        totalTime: 0,
        errors: []
      };

      const batchStartTime = performance.now();

      try {
        // Prepare batch cache request
        const timestamps = request.indices.map(index => 
          new Date(request.candles[index]?.timestamp).getTime()
        ).filter(ts => !isNaN(ts));

        // Get cached features
        const cacheResults = await featureStore.getBatch([{
          symbol: request.symbol,
          timestamps,
          options: request.options
        }]);

        const cacheResult = cacheResults[0];
        if (cacheResult) {
          batchResult.totalCached = cacheResult.cached.length;
        }

        // Process each index
        for (const index of request.indices) {
          try {
            const timestamp = new Date(request.candles[index]?.timestamp).getTime();
            if (!timestamp || isNaN(timestamp)) {
              batchResult.errors.push({
                index,
                error: 'Invalid timestamp'
              });
              continue;
            }

            // Check if we have cached result
            const cachedFeature = cacheResult?.features.get(timestamp);
            if (cachedFeature && this.isValidCachedFeatures(cachedFeature, request.options)) {
              batchResult.results.set(index, {
                features: cachedFeature,
                source: 'cache'
              });
              continue;
            }

            // Compute if not cached or force recompute
            const computeStartTime = performance.now();
            const features = await this.computeFeatures(
              request.symbol,
              request.candles,
              index,
              request.options
            );
            const computeTime = performance.now() - computeStartTime;

            batchResult.results.set(index, {
              features,
              source: 'computed',
              computeTime
            });

            batchResult.totalComputed++;

            // Cache the computed result
            await featureStore.set(features, request.options);

          } catch (error) {
            batchResult.errors.push({
              index,
              error: error instanceof Error ? error.message : String(error)
            });
          }
        }

      } catch (error) {
        // If batch processing fails completely
        request.indices.forEach(index => {
          batchResult.errors.push({
            index,
            error: `Batch processing failed: ${error}`
          });
        });
      }

      batchResult.totalTime = performance.now() - batchStartTime;
      results.push(batchResult);
    }

    return results;
  }

  /**
   * Precompute and cache features for a range of candles
   */
  async precomputeFeatures(
    symbol: string,
    candles: CandleData[],
    startIndex: number = 0,
    endIndex?: number,
    options?: ComputeOptions
  ): Promise<{
    processed: number;
    cached: number;
    errors: number;
    totalTime: number;
  }> {
    const finalEndIndex = endIndex || candles.length - 1;
    const startTime = performance.now();
    let processed = 0;
    let cached = 0;
    let errors = 0;

    const batchSize = 50; // Process in batches to avoid memory issues

    for (let i = startIndex; i <= finalEndIndex; i += batchSize) {
      const batchEnd = Math.min(i + batchSize - 1, finalEndIndex);
      const indices = Array.from(
        { length: batchEnd - i + 1 }, 
        (_, idx) => i + idx
      );

      try {
        const batchResults = await this.getBatchFeatures([{
          symbol,
          candles,
          indices,
          options: { ...options, forceRecompute: false } // Use cache when precomputing
        }]);

        const result = batchResults[0];
        if (result) {
          processed += result.results.size;
          cached += result.totalComputed; // New computations that were cached
          errors += result.errors.length;
        }

      } catch (error) {
        console.error(`Precompute batch ${i}-${batchEnd} failed:`, error);
        errors += batchEnd - i + 1;
      }
    }

    return {
      processed,
      cached,
      errors,
      totalTime: performance.now() - startTime
    };
  }

  /**
   * Invalidate cached features for symbol
   */
  async invalidateSymbol(symbol: string): Promise<number> {
    return await featureStore.invalidate(symbol);
  }

  /**
   * Get performance statistics
   */
  async getPerformanceStats(): Promise<{
    featureStore: any;
    integration: {
      version: string;
      uptime: number;
    };
  }> {
    const stats = await featureStore.getCacheStats();
    
    return {
      featureStore: stats,
      integration: {
        version: this.VERSION,
        uptime: process.uptime() * 1000
      }
    };
  }

  /**
   * Health check for the integration service
   */
  async healthCheck(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    components: {
      featureStore: any;
      indicators: boolean;
    };
  }> {
    const featureStoreHealth = await featureStore.healthCheck();
    
    // Test indicators computation
    let indicatorsHealthy = false;
    try {
      // Create minimal test data
      const testCandles: CandleData[] = Array.from({ length: 30 }, (_, i) => ({
        candle_index: i,
        timestamp: new Date(Date.now() - (30 - i) * 60000).toISOString(),
        open: 100 + Math.random() * 10,
        high: 105 + Math.random() * 10,
        low: 95 + Math.random() * 10,
        close: 100 + Math.random() * 10,
        volume: 1000 + Math.random() * 500,
        session_id: 'test-session',
        candle_datetime: new Date(Date.now() - (30 - i) * 60000).toISOString()
      }));

      const indicators = await TechnicalIndicatorService.calculateAll(testCandles, 29);
      indicatorsHealthy = typeof indicators.rsi === 'number';
      
    } catch (error) {
      console.error('Indicators health check failed:', error);
    }

    const overallStatus = 
      featureStoreHealth.status === 'healthy' && indicatorsHealthy ? 'healthy' :
      featureStoreHealth.status !== 'unhealthy' || indicatorsHealthy ? 'degraded' :
      'unhealthy';

    return {
      status: overallStatus,
      components: {
        featureStore: featureStoreHealth,
        indicators: indicatorsHealthy
      }
    };
  }

  // Private methods
  private async computeFeatures(
    symbol: string,
    candles: CandleData[],
    currentIndex: number,
    options?: ComputeOptions
  ): Promise<FeatureVector> {
    const candle = candles[currentIndex];
    if (!candle) {
      throw new Error(`No candle at index ${currentIndex}`);
    }

    // Ensure minimum data for indicators
    if (currentIndex < 20) {
      throw new Error(`Insufficient data: need at least 20 candles, got ${currentIndex + 1}`);
    }

    // Calculate technical indicators
    const indicators = await TechnicalIndicatorService.calculateAll(candles, currentIndex);
    
    // Calculate patterns if requested
    let patterns: any[] = [];
    if (options?.includePatterns) {
      // TODO: Integrate pattern detection when available
      patterns = [];
    }

    // Calculate confidence score
    const confidence = this.calculateConfidence(indicators, candles.length, currentIndex);
    
    // Validate confidence threshold
    if (options?.confidenceThreshold && confidence < options.confidenceThreshold) {
      throw new Error(`Confidence ${confidence} below threshold ${options.confidenceThreshold}`);
    }

    return {
      symbol,
      timestamp: new Date(candle.timestamp).getTime(),
      indicators,
      patterns,
      metadata: {
        version: this.VERSION,
        computedAt: Date.now(),
        dataPoints: currentIndex + 1,
        confidence
      }
    };
  }

  private isValidCachedFeatures(
    features: FeatureVector, 
    options?: ComputeOptions
  ): boolean {
    // Check version compatibility
    if (features.metadata.version !== this.VERSION) {
      return false;
    }

    // Check confidence threshold
    if (options?.confidenceThreshold && 
        features.metadata.confidence < options.confidenceThreshold) {
      return false;
    }

    // Check data freshness (max 1 hour old for real-time data)
    const now = Date.now();
    const age = now - features.metadata.computedAt;
    const maxAge = 60 * 60 * 1000; // 1 hour

    if (age > maxAge && (now - features.timestamp) < 5 * 60 * 1000) {
      // Real-time data older than 1 hour should be recomputed
      return false;
    }

    return true;
  }

  private calculateConfidence(
    indicators: TechnicalIndicators,
    totalCandles: number,
    currentIndex: number
  ): number {
    let confidence = 0;

    // Base confidence on data sufficiency
    const dataRatio = Math.min(currentIndex / 50, 1); // Optimal at 50+ candles
    confidence += dataRatio * 0.4;

    // Add confidence based on indicator validity
    const validIndicators = [
      !isNaN(indicators.rsi) && indicators.rsi >= 0 && indicators.rsi <= 100,
      !isNaN(indicators.macd.line) && isFinite(indicators.macd.line),
      !isNaN(indicators.stochastic.k) && indicators.stochastic.k >= 0 && indicators.stochastic.k <= 100,
      !isNaN(indicators.atr) && indicators.atr > 0,
      !isNaN(indicators.adx) && indicators.adx >= 0 && indicators.adx <= 100
    ].filter(Boolean).length;

    confidence += (validIndicators / 5) * 0.4;

    // Add confidence based on data consistency
    if (totalCandles > currentIndex) {
      confidence += 0.2; // Complete dataset
    }

    return Math.min(Math.max(confidence, 0), 1);
  }
}

// Singleton instance
export const featureIntegration = new FeatureIntegrationService();