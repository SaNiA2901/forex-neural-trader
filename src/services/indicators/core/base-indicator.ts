/**
 * Base Indicator Class
 * 
 * Abstract base class for all technical indicators providing common
 * functionality like caching, validation, streaming, and performance monitoring.
 * 
 * @author Trading System Core
 * @version 1.0.0
 */

import {
  MarketDataPoint,
  IndicatorConfig,
  IndicatorResult,
  IndicatorMetadata,
  IndicatorError,
  PerformanceMetrics,
  StreamingUpdate,
  IndicatorValue,
  DataValidationResult,
  BaseIndicatorConfig
} from './types';
import { CacheManager } from './cache-manager';
import { DataValidator } from './data-validator';

/**
 * Abstract base class for all technical indicators
 */
export abstract class BaseIndicator<T = number, C extends BaseIndicatorConfig = BaseIndicatorConfig> {
  protected config: C;
  protected cache: CacheManager;
  protected validator: DataValidator;
  protected performanceMetrics: PerformanceMetrics;
  private calculationCount: number = 0;
  private computationTimes: number[] = [];

  /**
   * Creates a new indicator instance
   * @param config - Indicator configuration
   */
  constructor(config: C) {
    this.config = { ...this.getDefaultConfig(), ...config };
    this.cache = new CacheManager({
      maxSize: 1000,
      defaultTTL: this.config.cacheTTL || 300000, // 5 minutes default
      enabled: this.config.enableCaching !== false
    });
    this.validator = new DataValidator(this.config.validationLevel || 'normal');
    
    this.performanceMetrics = {
      averageComputationTime: 0,
      maxComputationTime: 0,
      minComputationTime: Infinity,
      totalCalculations: 0,
      cacheHitRatio: 0,
      memoryUsage: 0,
      updatesPerSecond: 0
    };
  }

  /**
   * Calculate indicator values for the given data
   * @param data - Market data points
   * @returns Indicator result with values and metadata
   */
  async calculate(data: MarketDataPoint[]): Promise<IndicatorResult<T>> {
    const startTime = performance.now();
    
    try {
      // Validate input data
      const validationResult = await this.validateData(data);
      if (!validationResult.isValid) {
        throw new Error(`Data validation failed: ${validationResult.errors.join(', ')}`);
      }

      // Check cache first
      const cacheKey = this.generateCacheKey(data, this.config);
      const cachedResult = this.cache.get<IndicatorResult<T>>(cacheKey);
      
      if (cachedResult) {
        this.updateCacheHitRatio(true);
        return {
          ...cachedResult,
          metadata: {
            ...cachedResult.metadata,
            cached: true,
            lastUpdated: Date.now()
          }
        };
      }

      // Perform calculation
      const result = await this.performCalculation(data);
      
      // Update performance metrics
      const computationTime = performance.now() - startTime;
      this.updatePerformanceMetrics(computationTime);
      
      // Create metadata
      const metadata: IndicatorMetadata = {
        config: this.config,
        minDataPoints: this.getMinDataPoints(),
        validFrom: Math.max(0, this.getMinDataPoints() - 1),
        computationTime,
        cached: false,
        lastUpdated: Date.now(),
        dataQuality: validationResult.qualityScore
      };

      // Prepare final result
      const finalResult: IndicatorResult<T> = {
        values: result.values,
        metadata,
        validity: this.calculateValidity(result.values, data),
        error: result.error
      };

      // Cache the result
      this.cache.set(cacheKey, finalResult);
      this.updateCacheHitRatio(false);

      return finalResult;
      
    } catch (error) {
      const computationTime = performance.now() - startTime;
      this.updatePerformanceMetrics(computationTime);
      
      return {
        values: [],
        metadata: {
          config: this.config,
          minDataPoints: this.getMinDataPoints(),
          validFrom: 0,
          computationTime,
          cached: false,
          lastUpdated: Date.now(),
          dataQuality: 0
        },
        validity: [],
        error: {
          code: 'CALCULATION_ERROR',
          message: error instanceof Error ? error.message : 'Unknown calculation error',
          context: { config: this.config },
          suggestions: ['Check input data quality', 'Verify configuration parameters']
        }
      };
    }
  }

  /**
   * Update indicator with new data point (streaming)
   * @param newDataPoint - New market data point
   * @param existingData - Existing data points
   * @returns Streaming update result
   */
  async update(
    newDataPoint: MarketDataPoint, 
    existingData: MarketDataPoint[]
  ): Promise<StreamingUpdate<T>> {
    if (!this.config.enableStreaming) {
      throw new Error('Streaming is not enabled for this indicator');
    }

    const allData = [...existingData, newDataPoint];
    const result = await this.calculate(allData);
    
    if (result.values.length === 0) {
      throw new Error('Failed to calculate indicator value for streaming update');
    }

    const newValue = result.values[result.values.length - 1];
    const previousValue = result.values.length > 1 ? result.values[result.values.length - 2] : newValue;
    
    const change = this.calculateChange(newValue, previousValue);
    const percentChange = this.calculatePercentChange(newValue, previousValue);

    return {
      value: {
        timestamp: newDataPoint.timestamp,
        value: newValue,
        isValid: result.validity[result.validity.length - 1],
        confidence: this.calculateConfidence(result, allData)
      },
      change,
      percentChange,
      levelCross: this.detectLevelCross(newValue, previousValue)
    };
  }

  /**
   * Get current performance metrics
   */
  getPerformanceMetrics(): PerformanceMetrics {
    return { ...this.performanceMetrics };
  }

  /**
   * Reset performance metrics
   */
  resetPerformanceMetrics(): void {
    this.performanceMetrics = {
      averageComputationTime: 0,
      maxComputationTime: 0,
      minComputationTime: Infinity,
      totalCalculations: 0,
      cacheHitRatio: 0,
      memoryUsage: 0,
      updatesPerSecond: 0
    };
    this.calculationCount = 0;
    this.computationTimes = [];
  }

  /**
   * Clear indicator cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Get indicator configuration
   */
  getConfig(): C {
    return { ...this.config };
  }

  /**
   * Update indicator configuration
   * @param newConfig - New configuration parameters
   */
  updateConfig(newConfig: Partial<C>): void {
    this.config = { ...this.config, ...newConfig };
    this.clearCache(); // Clear cache when config changes
  }

  // Abstract methods that must be implemented by concrete indicators

  /**
   * Get default configuration for this indicator
   */
  protected abstract getDefaultConfig(): C;

  /**
   * Perform the actual indicator calculation
   * @param data - Validated market data
   * @returns Calculation result
   */
  protected abstract performCalculation(data: MarketDataPoint[]): Promise<{
    values: T[];
    error?: IndicatorError;
  }>;

  /**
   * Get minimum number of data points required for calculation
   */
  protected abstract getMinDataPoints(): number;

  /**
   * Get indicator name for caching and identification
   */
  protected abstract getIndicatorName(): string;

  // Protected helper methods

  /**
   * Validate input data
   * @param data - Market data to validate
   */
  protected async validateData(data: MarketDataPoint[]): Promise<DataValidationResult> {
    return this.validator.validate(data, this.getMinDataPoints());
  }

  /**
   * Generate cache key for the given data and config
   * @param data - Market data
   * @param config - Indicator configuration
   */
  protected generateCacheKey(data: MarketDataPoint[], config: C): string {
    const dataHash = this.hashData(data);
    const configHash = this.hashConfig(config);
    return `${this.getIndicatorName()}:${dataHash}:${configHash}`;
  }

  /**
   * Calculate validity flags for indicator values
   * @param values - Calculated indicator values
   * @param data - Original market data
   */
  protected calculateValidity(values: T[], data: MarketDataPoint[]): boolean[] {
    const minData = this.getMinDataPoints();
    return values.map((_, index) => {
      return index >= minData - 1 && data.length >= minData;
    });
  }

  /**
   * Calculate confidence level for indicator value
   * @param result - Indicator result
   * @param data - Market data used
   */
  protected calculateConfidence(result: IndicatorResult<T>, data: MarketDataPoint[]): number {
    const dataQuality = result.metadata.dataQuality;
    const dataAge = Date.now() - data[data.length - 1].timestamp;
    const ageFactor = Math.max(0, 1 - (dataAge / (5 * 60 * 1000))); // Decrease confidence with age
    
    return Math.min(1, dataQuality * ageFactor);
  }

  /**
   * Calculate change between two values
   * @param newValue - New indicator value
   * @param previousValue - Previous indicator value
   */
  protected calculateChange(newValue: T, previousValue: T): number {
    if (typeof newValue === 'number' && typeof previousValue === 'number') {
      return newValue - previousValue;
    }
    return 0; // For complex types, override in subclass
  }

  /**
   * Calculate percentage change between two values
   * @param newValue - New indicator value
   * @param previousValue - Previous indicator value
   */
  protected calculatePercentChange(newValue: T, previousValue: T): number {
    if (typeof newValue === 'number' && typeof previousValue === 'number') {
      return previousValue !== 0 ? ((newValue - previousValue) / previousValue) * 100 : 0;
    }
    return 0; // For complex types, override in subclass
  }

  /**
   * Detect level crossovers (override in subclasses for specific logic)
   * @param newValue - New indicator value
   * @param previousValue - Previous indicator value
   */
  protected detectLevelCross(newValue: T, previousValue: T): 
    { level: number; direction: 'up' | 'down' } | undefined {
    // Base implementation - override in subclasses for specific level detection
    return undefined;
  }

  // Private helper methods

  /**
   * Update performance metrics with new computation time
   * @param computationTime - Time taken for calculation
   */
  private updatePerformanceMetrics(computationTime: number): void {
    this.calculationCount++;
    this.computationTimes.push(computationTime);
    
    // Keep only last 1000 computation times for performance
    if (this.computationTimes.length > 1000) {
      this.computationTimes.shift();
    }

    this.performanceMetrics.totalCalculations = this.calculationCount;
    this.performanceMetrics.maxComputationTime = Math.max(
      this.performanceMetrics.maxComputationTime, 
      computationTime
    );
    this.performanceMetrics.minComputationTime = Math.min(
      this.performanceMetrics.minComputationTime, 
      computationTime
    );
    this.performanceMetrics.averageComputationTime = 
      this.computationTimes.reduce((sum, time) => sum + time, 0) / this.computationTimes.length;
  }

  /**
   * Update cache hit ratio
   * @param wasHit - Whether the cache was hit
   */
  private updateCacheHitRatio(wasHit: boolean): void {
    const currentHits = this.performanceMetrics.cacheHitRatio * this.performanceMetrics.totalCalculations;
    const newHits = wasHit ? currentHits + 1 : currentHits;
    const newTotal = this.performanceMetrics.totalCalculations + 1;
    
    this.performanceMetrics.cacheHitRatio = newHits / newTotal;
  }

  /**
   * Generate hash for market data
   * @param data - Market data to hash
   */
  private hashData(data: MarketDataPoint[]): string {
    if (data.length === 0) return 'empty';
    
    const firstPoint = data[0];
    const lastPoint = data[data.length - 1];
    const checksum = data.reduce((sum, point) => sum + point.close + point.volume, 0);
    
    return `${data.length}-${firstPoint.timestamp}-${lastPoint.timestamp}-${checksum.toFixed(2)}`;
  }

  /**
   * Generate hash for configuration
   * @param config - Configuration to hash
   */
  private hashConfig(config: C): string {
    return JSON.stringify(config, Object.keys(config).sort());
  }
}
