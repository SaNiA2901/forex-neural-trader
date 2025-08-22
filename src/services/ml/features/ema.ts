/**
 * Exponential Moving Average (EMA) Service
 * 
 * Provides both batch computation and streaming EMA calculations
 * with configurable periods and smoothing factors.
 * 
 * @author Trading System Core
 * @version 1.0.0
 */

/**
 * Configuration for EMA calculation
 */
export interface EMAConfig {
  /** Number of periods for EMA calculation (must be > 0) */
  period: number;
  /** Custom smoothing factor (optional, calculated from period if not provided) */
  smoothingFactor?: number;
}

/**
 * Result of EMA calculation with metadata
 */
export interface EMAResult {
  /** Array of EMA values */
  values: number[];
  /** Configuration used for calculation */
  config: EMAConfig;
  /** Index from which EMA values are considered valid */
  validFrom: number;
  /** Actual smoothing factor used */
  smoothingFactor: number;
}

/**
 * EMA calculation errors
 */
export class EMAError extends Error {
  constructor(message: string, public code: string) {
    super(message);
    this.name = 'EMAError';
  }
}

/**
 * Validation utilities for EMA inputs
 */
export class EMAValidator {
  /**
   * Validates EMA configuration
   * @param config - EMA configuration to validate
   * @throws {EMAError} If configuration is invalid
   */
  static validateConfig(config: EMAConfig): void {
    if (!config) {
      throw new EMAError('EMA config is required', 'INVALID_CONFIG');
    }

    if (!Number.isInteger(config.period) || config.period <= 0) {
      throw new EMAError('Period must be a positive integer', 'INVALID_PERIOD');
    }

    if (config.period > 1000) {
      throw new EMAError('Period cannot exceed 1000', 'PERIOD_TOO_LARGE');
    }

    if (config.smoothingFactor !== undefined) {
      if (typeof config.smoothingFactor !== 'number' || 
          config.smoothingFactor <= 0 || 
          config.smoothingFactor > 1) {
        throw new EMAError(
          'Smoothing factor must be between 0 and 1 (exclusive)', 
          'INVALID_SMOOTHING_FACTOR'
        );
      }
    }
  }

  /**
   * Validates input values array
   * @param values - Array of values to validate
   * @throws {EMAError} If values are invalid
   */
  static validateValues(values: number[]): void {
    if (!Array.isArray(values)) {
      throw new EMAError('Values must be an array', 'INVALID_VALUES');
    }

    if (values.length === 0) {
      throw new EMAError('Values array cannot be empty', 'EMPTY_VALUES');
    }

    if (values.length > 100000) {
      throw new EMAError('Values array too large (max 100,000)', 'VALUES_TOO_LARGE');
    }

    for (let i = 0; i < values.length; i++) {
      if (typeof values[i] !== 'number' || !Number.isFinite(values[i])) {
        throw new EMAError(
          `Invalid value at index ${i}: must be a finite number`, 
          'INVALID_VALUE'
        );
      }
    }
  }
}

/**
 * Main EMA calculation service
 */
export class EMAService {
  /**
   * Calculates the smoothing factor from period using standard formula
   * α = 2 / (n + 1)
   * 
   * @param period - Number of periods
   * @returns Smoothing factor (alpha)
   */
  static calculateSmoothingFactor(period: number): number {
    return 2 / (period + 1);
  }

  /**
   * Computes EMA for a series of values
   * 
   * @param values - Array of numeric values
   * @param config - EMA configuration
   * @returns EMA result with values and metadata
   * 
   * @example
   * ```typescript
   * const prices = [10, 11, 12, 11, 10, 9, 8, 9, 10];
   * const result = EMAService.computeEMA(prices, { period: 5 });
   * console.log(result.values); // EMA values
   * console.log(result.validFrom); // Index where EMA becomes reliable
   * ```
   */
  static computeEMA(values: number[], config: EMAConfig): EMAResult {
    // Validate inputs
    EMAValidator.validateConfig(config);
    EMAValidator.validateValues(values);

    if (values.length < config.period) {
      throw new EMAError(
        `Insufficient data: need at least ${config.period} values, got ${values.length}`,
        'INSUFFICIENT_DATA'
      );
    }

    // Calculate or use provided smoothing factor
    const smoothingFactor = config.smoothingFactor ?? 
      this.calculateSmoothingFactor(config.period);

    const emaValues: number[] = [];
    
    // Initialize first EMA value as Simple Moving Average of first 'period' values
    let sum = 0;
    for (let i = 0; i < config.period; i++) {
      sum += values[i];
    }
    const initialSMA = sum / config.period;
    
    // Fill initial values with null/undefined equivalent (using NaN for clarity)
    for (let i = 0; i < config.period - 1; i++) {
      emaValues.push(NaN);
    }
    
    // Set first valid EMA value
    emaValues.push(initialSMA);
    let currentEMA = initialSMA;

    // Calculate subsequent EMA values
    for (let i = config.period; i < values.length; i++) {
      currentEMA = (values[i] * smoothingFactor) + (currentEMA * (1 - smoothingFactor));
      emaValues.push(currentEMA);
    }

    return {
      values: emaValues,
      config: { ...config, smoothingFactor },
      validFrom: config.period - 1,
      smoothingFactor
    };
  }

  /**
   * Computes multiple EMA periods at once for efficiency
   * 
   * @param values - Array of numeric values
   * @param periods - Array of periods to calculate
   * @returns Map of period to EMA result
   */
  static computeMultipleEMA(
    values: number[], 
    periods: number[]
  ): Map<number, EMAResult> {
    EMAValidator.validateValues(values);
    
    const results = new Map<number, EMAResult>();
    
    for (const period of periods) {
      try {
        const result = this.computeEMA(values, { period });
        results.set(period, result);
      } catch (error) {
        // Skip invalid periods but continue with others
        console.warn(`Skipping EMA calculation for period ${period}:`, error);
      }
    }
    
    return results;
  }

  /**
   * Calculates EMA convergence between two different periods
   * Useful for MACD-like indicators
   * 
   * @param values - Array of numeric values
   * @param fastPeriod - Fast EMA period
   * @param slowPeriod - Slow EMA period
   * @returns Difference between fast and slow EMA
   */
  static calculateEMAConvergenceDivergence(
    values: number[],
    fastPeriod: number,
    slowPeriod: number
  ): number[] {
    const fastEMA = this.computeEMA(values, { period: fastPeriod });
    const slowEMA = this.computeEMA(values, { period: slowPeriod });
    
    const convergence: number[] = [];
    const validFrom = Math.max(fastEMA.validFrom, slowEMA.validFrom);
    
    for (let i = 0; i < values.length; i++) {
      if (i < validFrom) {
        convergence.push(NaN);
      } else {
        convergence.push(fastEMA.values[i] - slowEMA.values[i]);
      }
    }
    
    return convergence;
  }
}