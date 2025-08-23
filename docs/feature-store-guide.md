# Feature Store with Redis Caching - Usage Guide

## Overview

The Feature Store service provides high-performance caching for technical indicators and feature vectors using Redis. It's designed to reduce computation time, improve system responsiveness, and provide reliable access to trading data.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Trading App   │────│ FeatureIntegration │────│   FeatureStore  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          ▼
                                               ┌─────────────────┐
                                               │   Redis Cache   │
                                               └─────────────────┘
```

## Core Components

### 1. RedisClientManager
Manages Redis connections with health monitoring, error recovery, and performance metrics.

### 2. FeatureStore
Core caching service with support for:
- Single and batch operations
- TTL management
- Version control
- Performance monitoring

### 3. FeatureIntegration
High-level service that integrates technical indicators with caching.

## Quick Start

### Basic Usage

```typescript
import { featureIntegration } from '@/services/cache/FeatureIntegration';
import { CandleData } from '@/types/session';

// Get cached or computed features for a single candle
const result = await featureIntegration.getFeatures(
  'EURUSD',
  candles,
  currentIndex,
  {
    forceRecompute: false,
    includePatterns: true,
    confidenceThreshold: 0.8
  }
);

console.log('Features:', result.features);
console.log('Source:', result.source); // 'cache' or 'computed'
console.log('Compute time:', result.computeTime);
```

### Batch Processing

```typescript
// Process multiple candles efficiently
const batchResults = await featureIntegration.getBatchFeatures([{
  symbol: 'EURUSD',
  candles: historicalCandles,
  indices: [100, 150, 200, 250],
  options: {
    includePatterns: true,
    namespace: 'trading'
  }
}]);

const result = batchResults[0];
console.log(`Cached: ${result.totalCached}, Computed: ${result.totalComputed}`);
```

### Precomputation

```typescript
// Precompute and cache features for a range
const precomputeResult = await featureIntegration.precomputeFeatures(
  'EURUSD',
  candles,
  0,        // start index
  500,      // end index
  { includePatterns: false }
);

console.log(`Processed ${precomputeResult.processed} candles in ${precomputeResult.totalTime}ms`);
```

## Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_password
REDIS_DB=0

# Optional: Custom cache settings
CACHE_TTL_REALTIME=300    # 5 minutes
CACHE_TTL_HISTORICAL=3600 # 1 hour
```

### TTL Management

The system automatically chooses appropriate TTL based on data age:

- **Real-time data** (< 5 minutes old): 5 minutes TTL
- **Recent data** (< 1 hour old): 15 minutes TTL  
- **Historical data** (> 1 hour old): 1 hour TTL

### Custom TTL

```typescript
await featureStore.set(features, {
  ttl: 1800, // 30 minutes custom TTL
  namespace: 'custom'
});
```

## Advanced Features

### Feature Versioning

The system automatically versions cached features to handle indicator updates:

```typescript
// Features are automatically versioned
const features = {
  // ... feature data
  metadata: {
    version: '1.0.0',  // Automatic versioning
    computedAt: Date.now(),
    confidence: 0.95
  }
};
```

### Health Monitoring

```typescript
// Check system health
const health = await featureIntegration.healthCheck();

console.log('Overall status:', health.status);
console.log('Redis healthy:', health.components.featureStore.redis);
console.log('Cache healthy:', health.components.featureStore.cache);
console.log('Indicators healthy:', health.components.indicators);
```

### Performance Metrics

```typescript
// Get detailed performance statistics
const stats = await featureIntegration.getPerformanceStats();

console.log('Cache hit rate:', stats.featureStore.metrics.hitRate);
console.log('Average compute time:', stats.featureStore.metrics.averageComputeTime);
console.log('Total cached keys:', stats.featureStore.totalKeys);
console.log('Memory usage:', stats.featureStore.memoryUsage);
```

## Cache Strategies

### 1. Lazy Loading (Default)
Features are computed and cached on first request.

```typescript
const result = await featureIntegration.getFeatures('EURUSD', candles, index);
// First call: computes and caches
// Subsequent calls: serves from cache
```

### 2. Precomputation
Compute and cache features ahead of time.

```typescript
// Precompute during off-peak hours
await featureIntegration.precomputeFeatures('EURUSD', candles, 0, 1000);
```

### 3. Force Refresh
Bypass cache and force recomputation.

```typescript
const result = await featureIntegration.getFeatures(
  'EURUSD', 
  candles, 
  index,
  { forceRecompute: true }
);
```

## Error Handling

### Graceful Degradation

The system automatically falls back to direct computation if Redis is unavailable:

```typescript
try {
  const result = await featureIntegration.getFeatures('EURUSD', candles, index);
  // Will work even if Redis is down
} catch (error) {
  console.error('Feature computation failed:', error);
}
```

### Error Recovery

```typescript
// The system automatically retries failed Redis operations
const redisConfig = {
  maxRetries: 3,
  retryDelayOnFailover: 100,
  connectTimeout: 10000
};
```

## Monitoring Dashboard

Use the built-in monitoring dashboard:

```tsx
import CacheMonitoringDashboard from '@/components/ui/monitoring/CacheMonitoringDashboard';

function MonitoringPage() {
  return (
    <div>
      <h1>Cache Performance</h1>
      <CacheMonitoringDashboard />
    </div>
  );
}
```

The dashboard provides:
- Real-time health status
- Cache hit/miss rates
- Performance metrics
- Error tracking
- Optimization recommendations

## Best Practices

### 1. Batch Operations
Use batch operations for multiple requests:

```typescript
// ✅ Efficient - batch request
const batchResults = await featureIntegration.getBatchFeatures([
  { symbol: 'EURUSD', candles, indices: [100, 101, 102] }
]);

// ❌ Inefficient - individual requests
for (const index of [100, 101, 102]) {
  await featureIntegration.getFeatures('EURUSD', candles, index);
}
```

### 2. Confidence Thresholds
Set appropriate confidence thresholds:

```typescript
const result = await featureIntegration.getFeatures(
  'EURUSD', 
  candles, 
  index,
  { 
    confidenceThreshold: 0.8,  // Require high confidence
    includePatterns: true 
  }
);
```

### 3. Namespace Organization
Use namespaces to organize cache keys:

```typescript
// Separate by trading session or strategy
await featureStore.set(features, { namespace: 'session_2024_01' });
await featureStore.set(features, { namespace: 'strategy_scalping' });
```

### 4. Memory Management
Monitor and clean up stale cache entries:

```typescript
// Invalidate old data for a symbol
await featureIntegration.invalidateSymbol('EURUSD');

// Check cache statistics
const stats = await featureStore.getCacheStats();
if (stats.totalKeys > 10000) {
  console.warn('Cache size is growing large');
}
```

## Performance Optimization

### Expected Performance Metrics

- **Cache Hit Rate**: >85% for production workloads
- **Cache Response Time**: <5ms average
- **Compute Time**: <100ms for full indicator set
- **Memory Usage**: <500MB for 10,000 cached features

### Optimization Tips

1. **Increase Cache Hit Rate**:
   - Use longer TTL for historical data
   - Precompute frequently accessed features
   - Optimize cache key structure

2. **Reduce Compute Time**:
   - Optimize technical indicator algorithms
   - Use batch processing for multiple symbols
   - Consider parallel computation

3. **Memory Efficiency**:
   - Use appropriate TTL values
   - Implement cache size limits
   - Monitor memory usage regularly

## Troubleshooting

### Common Issues

1. **Low Cache Hit Rate**
   ```
   Problem: Hit rate below 70%
   Solution: Check TTL settings, verify cache keys, monitor invalidation patterns
   ```

2. **High Computation Time**
   ```
   Problem: Compute time >200ms
   Solution: Optimize indicator calculations, check data quality, consider caching intermediate results
   ```

3. **Redis Connection Issues**
   ```
   Problem: Connection failures
   Solution: Check network connectivity, verify Redis configuration, monitor Redis logs
   ```

### Debug Mode

Enable debug logging:

```typescript
// Add to your environment
DEBUG=feature-store,redis-client

// Or programmatically
process.env.DEBUG = 'feature-store,redis-client';
```

## Testing

### Unit Tests

Run the comprehensive test suite:

```bash
npm test -- src/__tests__/cache/FeatureStore.test.ts
```

### Integration Tests

Test with real Redis instance:

```bash
# Start Redis locally
docker run -d -p 6379:6379 redis:alpine

# Run integration tests
npm run test:integration
```

### Load Testing

Test cache performance under load:

```typescript
// Example load test
const symbols = ['EURUSD', 'GBPUSD', 'USDJPY'];
const concurrentRequests = 100;

const promises = Array.from({ length: concurrentRequests }, () =>
  featureIntegration.getFeatures(
    symbols[Math.floor(Math.random() * symbols.length)],
    candles,
    Math.floor(Math.random() * candles.length)
  )
);

const startTime = performance.now();
await Promise.all(promises);
const endTime = performance.now();

console.log(`Processed ${concurrentRequests} requests in ${endTime - startTime}ms`);
```

## Migration Guide

### From Direct Computation

Replace direct indicator calculations:

```typescript
// Before
const indicators = await TechnicalIndicatorService.calculateAll(candles, index);

// After
const result = await featureIntegration.getFeatures('EURUSD', candles, index);
const indicators = result.features.indicators;
```

### From Simple Caching

Upgrade from basic caching:

```typescript
// Before - simple Map-based cache
const cache = new Map();
const key = `${symbol}_${index}`;
let indicators = cache.get(key);
if (!indicators) {
  indicators = await calculateIndicators(candles, index);
  cache.set(key, indicators);
}

// After - Feature Store
const result = await featureIntegration.getFeatures(symbol, candles, index);
const indicators = result.features.indicators;
```

## API Reference

### FeatureIntegrationService

```typescript
class FeatureIntegrationService {
  // Get single feature vector
  async getFeatures(
    symbol: string,
    candles: CandleData[],
    currentIndex: number,
    options?: ComputeOptions
  ): Promise<FeatureComputeResult>

  // Batch processing
  async getBatchFeatures(
    requests: BatchComputeRequest[]
  ): Promise<BatchComputeResult[]>

  // Precomputation
  async precomputeFeatures(
    symbol: string,
    candles: CandleData[],
    startIndex?: number,
    endIndex?: number,
    options?: ComputeOptions
  ): Promise<PrecomputeResult>

  // Cache management
  async invalidateSymbol(symbol: string): Promise<number>
  
  // Monitoring
  async getPerformanceStats(): Promise<PerformanceStats>
  async healthCheck(): Promise<HealthStatus>
}
```

### FeatureStore

```typescript
class FeatureStore {
  // Single operations
  async get(symbol: string, timestamp: number, options?: CacheOptions): Promise<FeatureVector | null>
  async set(featureVector: FeatureVector, options?: CacheOptions): Promise<boolean>

  // Batch operations
  async getBatch(requests: BatchRequest[]): Promise<BatchResult[]>
  async setBatch(featureVectors: FeatureVector[], options?: CacheOptions): Promise<BatchSetResult>

  // Management
  async invalidate(symbol: string, namespace?: string): Promise<number>
  async getCacheStats(namespace?: string): Promise<CacheStats>
  async healthCheck(): Promise<HealthResult>

  // Metrics
  getMetrics(): FeatureStoreMetrics
}
```

## Support

For questions or issues:

1. Check the troubleshooting section above
2. Review Redis logs and cache metrics
3. Enable debug logging for detailed information
4. Monitor the dashboard for performance insights

The Feature Store is designed to be robust and self-healing, but monitoring and proper configuration are key to optimal performance.