/**
 * Core Types and Interfaces for Technical Indicators Library
 * 
 * Provides consistent interfaces for all technical indicators with
 * support for streaming updates, caching, and performance optimization.
 * 
 * @author Trading System Core
 * @version 1.0.0
 */

/**
 * Market data point with OHLCV structure
 */
export interface MarketDataPoint {
  /** Timestamp of the data point */
  timestamp: number;
  /** Opening price */
  open: number;
  /** Highest price */
  high: number;
  /** Lowest price */
  low: number;
  /** Closing price */
  close: number;
  /** Trading volume */
  volume: number;
}

/**
 * Configuration for indicator calculation
 */
export interface IndicatorConfig {
  /** Primary period for calculation (e.g., 14 for RSI) */
  period: number;
  /** Additional parameters specific to each indicator */
  [key: string]: any;
}

/**
 * Metadata about indicator calculation
 */
export interface IndicatorMetadata {
  /** Configuration used for calculation */
  config: IndicatorConfig;
  /** Minimum number of data points needed for valid calculation */
  minDataPoints: number;
  /** Index from which values are considered valid */
  validFrom: number;
  /** Total computation time in milliseconds */
  computationTime: number;
  /** Whether result is from cache */
  cached: boolean;
  /** Last updated timestamp */
  lastUpdated: number;
  /** Data quality score (0-1) */
  dataQuality: number;
}

/**
 * Result of indicator calculation
 */
export interface IndicatorResult<T = number> {
  /** Calculated indicator values */
  values: T[];
  /** Metadata about the calculation */
  metadata: IndicatorMetadata;
  /** Validity flags for each value */
  validity: boolean[];
  /** Error information if calculation failed */
  error?: IndicatorError;
}

/**
 * Single indicator value with timestamp
 */
export interface IndicatorValue<T = number> {
  /** Timestamp of the value */
  timestamp: number;
  /** Indicator value */
  value: T;
  /** Whether this value is valid */
  isValid: boolean;
  /** Confidence level (0-1) */
  confidence: number;
}

/**
 * Streaming update result
 */
export interface StreamingUpdate<T = number> {
  /** New indicator value */
  value: IndicatorValue<T>;
  /** Change from previous value */
  change: number;
  /** Percentage change from previous value */
  percentChange: number;
  /** Whether indicator crossed a significant level */
  levelCross?: {
    level: number;
    direction: 'up' | 'down';
  };
}

/**
 * Error information for indicator calculations
 */
export interface IndicatorError {
  /** Error code */
  code: string;
  /** Human-readable error message */
  message: string;
  /** Detailed error context */
  context?: Record<string, any>;
  /** Suggestions for fixing the error */
  suggestions?: string[];
}

/**
 * Cache entry for indicator results
 */
export interface CacheEntry<T = any> {
  /** Cached result */
  result: T;
  /** Timestamp when cached */
  timestamp: number;
  /** Time to live in milliseconds */
  ttl: number;
  /** Cache key hash */
  key: string;
  /** Size in bytes (estimated) */
  size: number;
}

/**
 * Performance metrics for indicators
 */
export interface PerformanceMetrics {
  /** Average computation time in milliseconds */
  averageComputationTime: number;
  /** Maximum computation time in milliseconds */
  maxComputationTime: number;
  /** Minimum computation time in milliseconds */
  minComputationTime: number;
  /** Total number of calculations performed */
  totalCalculations: number;
  /** Cache hit ratio (0-1) */
  cacheHitRatio: number;
  /** Memory usage in bytes */
  memoryUsage: number;
  /** Updates per second for streaming */
  updatesPerSecond: number;
}

/**
 * RSI-specific result type
 */
export interface RSIResult extends IndicatorResult<number> {
  /** Current RSI level classification */
  level: 'oversold' | 'neutral' | 'overbought';
  /** Divergence signals if detected */
  divergence?: {
    type: 'bullish' | 'bearish';
    strength: number;
  };
}

/**
 * MACD-specific result type
 */
export interface MACDValue {
  /** MACD line value */
  macd: number;
  /** Signal line value */
  signal: number;
  /** Histogram value */
  histogram: number;
}

export interface MACDResult extends IndicatorResult<MACDValue> {
  /** Signal crossovers detected */
  crossovers: Array<{
    timestamp: number;
    type: 'bullish' | 'bearish';
    strength: number;
  }>;
}

/**
 * Bollinger Bands specific result type
 */
export interface BollingerBandsValue {
  /** Upper band value */
  upper: number;
  /** Middle line (SMA) value */
  middle: number;
  /** Lower band value */
  lower: number;
  /** Bandwidth (volatility measure) */
  bandwidth: number;
  /** %B indicator */
  percentB: number;
}

export interface BollingerBandsResult extends IndicatorResult<BollingerBandsValue> {
  /** Squeeze periods detected */
  squeezes: Array<{
    startTimestamp: number;
    endTimestamp?: number;
    intensity: number;
  }>;
}

/**
 * Stochastic Oscillator specific result type
 */
export interface StochasticValue {
  /** %K line value */
  k: number;
  /** %D line value */
  d: number;
}

export interface StochasticResult extends IndicatorResult<StochasticValue> {
  /** Overbought/oversold signals */
  signals: Array<{
    timestamp: number;
    type: 'overbought' | 'oversold';
    strength: number;
  }>;
}

/**
 * VWAP specific result type
 */
export interface VWAPResult extends IndicatorResult<number> {
  /** Standard deviation bands */
  bands: Array<{
    upper: number;
    lower: number;
    deviation: number;
  }>;
  /** Volume profile data */
  volumeProfile?: Array<{
    price: number;
    volume: number;
    percentage: number;
  }>;
}

/**
 * Supported indicator types
 */
export type IndicatorType = 'rsi' | 'macd' | 'bollinger' | 'stochastic' | 'vwap' | 'ema' | 'sma';

/**
 * Base configuration for all indicators
 */
export interface BaseIndicatorConfig extends IndicatorConfig {
  /** Enable caching for this indicator */
  enableCaching?: boolean;
  /** Cache TTL in milliseconds */
  cacheTTL?: number;
  /** Enable streaming updates */
  enableStreaming?: boolean;
  /** Validation level (strict, normal, relaxed) */
  validationLevel?: 'strict' | 'normal' | 'relaxed';
}

/**
 * Indicator factory configuration
 */
export interface FactoryConfig {
  /** Default cache settings */
  cacheConfig?: {
    maxSize: number;
    defaultTTL: number;
    cleanupInterval: number;
  };
  /** Performance monitoring settings */
  performanceConfig?: {
    enableMetrics: boolean;
    metricsInterval: number;
    alertThresholds: {
      computationTime: number;
      memoryUsage: number;
    };
  };
  /** Streaming settings */
  streamingConfig?: {
    bufferSize: number;
    updateThrottle: number;
  };
}

/**
 * Market data validation result
 */
export interface DataValidationResult {
  /** Whether data is valid */
  isValid: boolean;
  /** Validation errors found */
  errors: string[];
  /** Data quality score (0-1) */
  qualityScore: number;
  /** Suggestions for data improvement */
  suggestions: string[];
  /** Statistics about the data */
  statistics: {
    dataPoints: number;
    completeness: number;
    duplicates: number;
    outliers: number;
  };
}

/**
 * Indicator comparison result
 */
export interface IndicatorComparison {
  /** Correlation coefficient between indicators */
  correlation: number;
  /** Divergence points detected */
  divergences: Array<{
    timestamp: number;
    indicator1Value: number;
    indicator2Value: number;
    divergenceStrength: number;
  }>;
  /** Statistical analysis */
  statistics: {
    mean1: number;
    mean2: number;
    standardDeviation1: number;
    standardDeviation2: number;
  };
}

/**
 * Real-time market data stream interface
 */
export interface MarketDataStream {
  /** Subscribe to real-time updates */
  subscribe(symbol: string, callback: (data: MarketDataPoint) => void): void;
  /** Unsubscribe from updates */
  unsubscribe(symbol: string): void;
  /** Get current market status */
  getMarketStatus(symbol: string): 'open' | 'closed' | 'pre_market' | 'after_hours';
}

/**
 * Batch calculation request
 */
export interface BatchCalculationRequest {
  /** Indicators to calculate */
  indicators: Array<{
    type: IndicatorType;
    config: IndicatorConfig;
  }>;
  /** Market data to use */
  data: MarketDataPoint[];
  /** Parallel processing options */
  parallel?: boolean;
  /** Progress callback */
  onProgress?: (completed: number, total: number) => void;
}

/**
 * Batch calculation result
 */
export interface BatchCalculationResult {
  /** Results for each indicator */
  results: Map<IndicatorType, IndicatorResult>;
  /** Overall computation time */
  totalTime: number;
  /** Memory usage during calculation */
  memoryUsage: number;
  /** Any errors encountered */
  errors: IndicatorError[];
}