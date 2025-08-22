/**
 * Streaming Exponential Moving Average (EMA) Implementation
 * 
 * Provides O(1) real-time EMA updates for streaming data scenarios.
 * Maintains state between updates for efficient continuous calculation.
 * 
 * @author Trading System Core
 * @version 1.0.0
 */

import { EMAConfig, EMAValidator, EMAError, EMAService } from './ema';

/**
 * State information for streaming EMA calculation
 */
export interface StreamingEMAState {
  /** Current EMA value */
  currentValue: number;
  /** Number of data points processed */
  count: number;
  /** Whether EMA has reached stability (enough data points) */
  isStable: boolean;
  /** Configuration being used */
  config: EMAConfig;
  /** Smoothing factor being used */
  smoothingFactor: number;
  /** Buffer for initial SMA calculation */
  initialBuffer: number[];
  /** Timestamp of last update */
  lastUpdated: number;
}

/**
 * Update result from streaming EMA
 */
export interface StreamingEMAUpdate {
  /** New EMA value */
  value: number;
  /** Whether this value is considered stable */
  isStable: boolean;
  /** Number of values processed so far */
  count: number;
  /** Change from previous EMA value */
  change: number;
  /** Percentage change from previous EMA value */
  percentChange: number;
  /** Current state snapshot */
  state: StreamingEMAState;
}

/**
 * Performance metrics for streaming EMA
 */
export interface StreamingEMAMetrics {
  /** Total number of updates processed */
  totalUpdates: number;
  /** Average update time in microseconds */
  averageUpdateTime: number;
  /** Maximum update time in microseconds */
  maxUpdateTime: number;
  /** Minimum update time in microseconds */
  minUpdateTime: number;
  /** Updates per second */
  updatesPerSecond: number;
  /** Memory usage estimate in bytes */
  memoryUsage: number;
}

/**
 * Streaming EMA Calculator
 * 
 * Efficiently calculates EMA in real-time with O(1) complexity per update.
 * Maintains internal state and provides detailed update information.
 */
export class StreamingEMA {
  private currentEMA: number = 0;
  private count: number = 0;
  private smoothingFactor: number;
  private initialBuffer: number[] = [];
  private isInitialized: boolean = false;
  private lastValue: number = 0;
  
  // Performance tracking
  private updateTimes: number[] = [];
  private totalUpdates: number = 0;
  private startTime: number = Date.now();

  /**
   * Creates a new streaming EMA calculator
   * 
   * @param config - EMA configuration
   * 
   * @example
   * ```typescript
   * const streamingEMA = new StreamingEMA({ period: 20 });
   * 
   * // Process real-time data
   * const update1 = streamingEMA.update(100);
   * const update2 = streamingEMA.update(105);
   * const update3 = streamingEMA.update(102);
   * 
   * console.log(update3.value); // Current EMA value
   * console.log(update3.isStable); // Whether EMA is stable
   * ```
   */
  constructor(private config: EMAConfig) {
    EMAValidator.validateConfig(config);
    
    this.smoothingFactor = config.smoothingFactor ?? 
      EMAService.calculateSmoothingFactor(config.period);
  }

  /**
   * Updates the EMA with a new value
   * 
   * @param newValue - New data point
   * @returns Update result with new EMA and metadata
   * @throws {EMAError} If input value is invalid
   */
  update(newValue: number): StreamingEMAUpdate {
    const startTime = performance.now();
    
    // Validate input
    if (typeof newValue !== 'number' || !Number.isFinite(newValue)) {
      throw new EMAError('Invalid input value: must be a finite number', 'INVALID_INPUT');
    }

    let emaValue: number;
    let isStable: boolean;
    const previousValue = this.currentEMA;

    if (!this.isInitialized) {
      // Collect initial values for SMA calculation
      this.initialBuffer.push(newValue);
      
      if (this.initialBuffer.length < this.config.period) {
        // Not enough data yet, return simple average
        emaValue = this.initialBuffer.reduce((sum, val) => sum + val, 0) / this.initialBuffer.length;
        isStable = false;
      } else {
        // Calculate initial SMA
        emaValue = this.initialBuffer.reduce((sum, val) => sum + val, 0) / this.config.period;
        this.isInitialized = true;
        isStable = true;
      }
    } else {
      // Standard EMA calculation: EMA = α × Current + (1-α) × Previous_EMA
      emaValue = (this.smoothingFactor * newValue) + 
                 ((1 - this.smoothingFactor) * this.currentEMA);
      isStable = true;
    }

    // Update internal state
    this.currentEMA = emaValue;
    this.count++;
    this.lastValue = newValue;
    this.totalUpdates++;

    // Calculate changes
    const change = emaValue - previousValue;
    const percentChange = previousValue !== 0 ? (change / previousValue) * 100 : 0;

    // Record performance
    const endTime = performance.now();
    const updateTime = (endTime - startTime) * 1000; // Convert to microseconds
    this.updateTimes.push(updateTime);
    
    // Keep only last 1000 update times for performance calculation
    if (this.updateTimes.length > 1000) {
      this.updateTimes.shift();
    }

    return {
      value: emaValue,
      isStable,
      count: this.count,
      change,
      percentChange,
      state: this.getState()
    };
  }

  /**
   * Gets the current state of the streaming EMA
   */
  getState(): StreamingEMAState {
    return {
      currentValue: this.currentEMA,
      count: this.count,
      isStable: this.isInitialized,
      config: { ...this.config },
      smoothingFactor: this.smoothingFactor,
      initialBuffer: [...this.initialBuffer],
      lastUpdated: Date.now()
    };
  }

  /**
   * Resets the streaming EMA to initial state
   */
  reset(): void {
    this.currentEMA = 0;
    this.count = 0;
    this.initialBuffer = [];
    this.isInitialized = false;
    this.lastValue = 0;
    this.updateTimes = [];
    this.totalUpdates = 0;
    this.startTime = Date.now();
  }

  /**
   * Updates configuration (resets internal state)
   * 
   * @param newConfig - New EMA configuration
   */
  updateConfig(newConfig: EMAConfig): void {
    EMAValidator.validateConfig(newConfig);
    this.config = { ...newConfig };
    this.smoothingFactor = newConfig.smoothingFactor ?? 
      EMAService.calculateSmoothingFactor(newConfig.period);
    this.reset();
  }

  /**
   * Gets performance metrics for this streaming EMA instance
   */
  getPerformanceMetrics(): StreamingEMAMetrics {
    const now = Date.now();
    const totalTime = (now - this.startTime) / 1000; // seconds
    const avgUpdateTime = this.updateTimes.length > 0 ? 
      this.updateTimes.reduce((sum, time) => sum + time, 0) / this.updateTimes.length : 0;
    
    return {
      totalUpdates: this.totalUpdates,
      averageUpdateTime: avgUpdateTime,
      maxUpdateTime: Math.max(...this.updateTimes, 0),
      minUpdateTime: Math.min(...this.updateTimes, 0),
      updatesPerSecond: totalTime > 0 ? this.totalUpdates / totalTime : 0,
      memoryUsage: this.estimateMemoryUsage()
    };
  }

  /**
   * Estimates memory usage in bytes
   */
  private estimateMemoryUsage(): number {
    const numberSize = 8; // 8 bytes per number in JavaScript
    const baseSize = numberSize * 5; // Basic numeric properties
    const bufferSize = this.initialBuffer.length * numberSize;
    const updateTimesSize = this.updateTimes.length * numberSize;
    const configSize = 24; // Estimated config object size
    
    return baseSize + bufferSize + updateTimesSize + configSize;
  }

  /**
   * Creates a streaming EMA from existing batch data
   * Useful for warm-starting with historical data
   * 
   * @param values - Historical values to initialize with
   * @param config - EMA configuration
   * @returns New StreamingEMA instance initialized with data
   */
  static fromBatchData(values: number[], config: EMAConfig): StreamingEMA {
    EMAValidator.validateValues(values);
    EMAValidator.validateConfig(config);
    
    const streamingEMA = new StreamingEMA(config);
    
    // Process all historical values
    for (const value of values) {
      streamingEMA.update(value);
    }
    
    return streamingEMA;
  }

  /**
   * Batch update multiple values efficiently
   * 
   * @param values - Array of values to process
   * @returns Array of update results
   */
  batchUpdate(values: number[]): StreamingEMAUpdate[] {
    EMAValidator.validateValues(values);
    
    const results: StreamingEMAUpdate[] = [];
    
    for (const value of values) {
      results.push(this.update(value));
    }
    
    return results;
  }

  /**
   * Gets the current EMA value without updating
   */
  getCurrentValue(): number {
    return this.currentEMA;
  }

  /**
   * Checks if the EMA is stable (has enough data points)
   */
  isStable(): boolean {
    return this.isInitialized;
  }

  /**
   * Gets the number of values processed
   */
  getCount(): number {
    return this.count;
  }

  /**
   * Gets the smoothing factor being used
   */
  getSmoothingFactor(): number {
    return this.smoothingFactor;
  }
}

/**
 * Multi-period streaming EMA manager
 * Manages multiple streaming EMAs with different periods efficiently
 */
export class MultiPeriodStreamingEMA {
  private emas = new Map<number, StreamingEMA>();

  constructor(periods: number[], baseConfig?: Partial<EMAConfig>) {
    for (const period of periods) {
      const config: EMAConfig = { period, ...baseConfig };
      this.emas.set(period, new StreamingEMA(config));
    }
  }

  /**
   * Updates all EMAs with new value
   */
  updateAll(value: number): Map<number, StreamingEMAUpdate> {
    const results = new Map<number, StreamingEMAUpdate>();
    
    for (const [period, ema] of this.emas) {
      results.set(period, ema.update(value));
    }
    
    return results;
  }

  /**
   * Gets current values for all periods
   */
  getCurrentValues(): Map<number, number> {
    const values = new Map<number, number>();
    
    for (const [period, ema] of this.emas) {
      values.set(period, ema.getCurrentValue());
    }
    
    return values;
  }

  /**
   * Gets EMA for specific period
   */
  getEMA(period: number): StreamingEMA | undefined {
    return this.emas.get(period);
  }

  /**
   * Adds new period
   */
  addPeriod(period: number, config?: EMAConfig): void {
    const emaConfig = config ?? { period };
    this.emas.set(period, new StreamingEMA(emaConfig));
  }

  /**
   * Removes period
   */
  removePeriod(period: number): boolean {
    return this.emas.delete(period);
  }

  /**
   * Resets all EMAs
   */
  resetAll(): void {
    for (const ema of this.emas.values()) {
      ema.reset();
    }
  }

  /**
   * Gets performance metrics for all EMAs
   */
  getPerformanceMetrics(): Map<number, StreamingEMAMetrics> {
    const metrics = new Map<number, StreamingEMAMetrics>();
    
    for (const [period, ema] of this.emas) {
      metrics.set(period, ema.getPerformanceMetrics());
    }
    
    return metrics;
  }
}
