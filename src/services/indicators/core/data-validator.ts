/**
 * Data Validator for Technical Indicators
 * 
 * Comprehensive validation system for market data quality assessment,
 * outlier detection, and data integrity checks.
 * 
 * @author Trading System Core
 * @version 1.0.0
 */

import { MarketDataPoint, DataValidationResult } from './types';

/**
 * Validation levels
 */
export type ValidationLevel = 'strict' | 'normal' | 'relaxed';

/**
 * Outlier detection methods
 */
export type OutlierMethod = 'iqr' | 'zscore' | 'modified_zscore';

/**
 * Data validation configuration
 */
export interface ValidationConfig {
  /** Validation strictness level */
  level: ValidationLevel;
  /** Maximum allowed gap between timestamps (in milliseconds) */
  maxTimeGap?: number;
  /** Maximum allowed price change percentage */
  maxPriceChange?: number;
  /** Minimum required data completeness ratio */
  minCompleteness?: number;
  /** Outlier detection method */
  outlierMethod?: OutlierMethod;
  /** Z-score threshold for outlier detection */
  outlierThreshold?: number;
}

/**
 * Data quality metrics
 */
interface DataQualityMetrics {
  completeness: number;
  consistency: number;
  accuracy: number;
  timeliness: number;
  uniqueness: number;
}

/**
 * Comprehensive data validator for market data
 */
export class DataValidator {
  private config: Required<ValidationConfig>;

  constructor(level: ValidationLevel = 'normal') {
    this.config = this.getDefaultConfig(level);
  }

  /**
   * Validate market data array
   * @param data - Market data to validate
   * @param minDataPoints - Minimum required data points
   */
  async validate(data: MarketDataPoint[], minDataPoints: number = 1): Promise<DataValidationResult> {
    const errors: string[] = [];
    const suggestions: string[] = [];
    
    // Basic validations
    if (!Array.isArray(data)) {
      errors.push('Data must be an array');
      return this.createFailResult(errors, suggestions);
    }

    if (data.length === 0) {
      errors.push('Data array cannot be empty');
      suggestions.push('Provide at least one data point');
      return this.createFailResult(errors, suggestions);
    }

    if (data.length < minDataPoints) {
      errors.push(`Insufficient data: need at least ${minDataPoints} points, got ${data.length}`);
      suggestions.push(`Collect more historical data (minimum ${minDataPoints} points required)`);
      return this.createFailResult(errors, suggestions);
    }

    // Detailed validations
    const structureErrors = this.validateStructure(data);
    const timeErrors = this.validateTimestamps(data);
    const priceErrors = this.validatePrices(data);
    const volumeErrors = this.validateVolumes(data);
    const outliers = this.detectOutliers(data);
    const duplicates = this.detectDuplicates(data);

    errors.push(...structureErrors, ...timeErrors, ...priceErrors, ...volumeErrors);

    // Generate suggestions based on findings
    if (outliers.length > 0) {
      suggestions.push(`Found ${outliers.length} potential outliers - consider data smoothing`);
    }
    if (duplicates.length > 0) {
      suggestions.push(`Found ${duplicates.length} duplicate entries - remove duplicates`);
    }
    if (timeErrors.length > 0) {
      suggestions.push('Ensure data is properly sorted by timestamp');
    }

    // Calculate quality metrics
    const qualityMetrics = this.calculateQualityMetrics(data, {
      outliers: outliers.length,
      duplicates: duplicates.length,
      errors: errors.length
    });

    const qualityScore = this.calculateOverallQuality(qualityMetrics);
    
    // Statistics
    const statistics = {
      dataPoints: data.length,
      completeness: qualityMetrics.completeness,
      duplicates: duplicates.length,
      outliers: outliers.length
    };

    return {
      isValid: errors.length === 0 || this.config.level === 'relaxed',
      errors,
      qualityScore,
      suggestions,
      statistics
    };
  }

  /**
   * Validate real-time data point
   * @param dataPoint - Single data point to validate
   * @param previousPoint - Previous data point for comparison
   */
  validateRealTimePoint(
    dataPoint: MarketDataPoint, 
    previousPoint?: MarketDataPoint
  ): { isValid: boolean; errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Structure validation
    if (!this.isValidDataPoint(dataPoint)) {
      errors.push('Invalid data point structure');
      return { isValid: false, errors, warnings };
    }

    // OHLC consistency
    if (dataPoint.high < dataPoint.low) {
      errors.push('High price cannot be lower than low price');
    }

    if (dataPoint.high < Math.max(dataPoint.open, dataPoint.close)) {
      errors.push('High price must be greater than or equal to open and close');
    }

    if (dataPoint.low > Math.min(dataPoint.open, dataPoint.close)) {
      errors.push('Low price must be less than or equal to open and close');
    }

    // Comparison with previous point
    if (previousPoint) {
      const priceChange = Math.abs(dataPoint.close - previousPoint.close) / previousPoint.close;
      if (priceChange > this.config.maxPriceChange) {
        if (this.config.level === 'strict') {
          errors.push(`Price change exceeds threshold: ${(priceChange * 100).toFixed(2)}%`);
        } else {
          warnings.push(`Large price change detected: ${(priceChange * 100).toFixed(2)}%`);
        }
      }

      const timeGap = dataPoint.timestamp - previousPoint.timestamp;
      if (timeGap > this.config.maxTimeGap) {
        warnings.push(`Large time gap detected: ${timeGap}ms`);
      }

      if (timeGap <= 0) {
        errors.push('Data points must be in chronological order');
      }
    }

    // Volume validation
    if (dataPoint.volume < 0) {
      errors.push('Volume cannot be negative');
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Update validation configuration
   * @param newConfig - New configuration options
   */
  updateConfig(newConfig: Partial<ValidationConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Get current validation configuration
   */
  getConfig(): ValidationConfig {
    return { ...this.config };
  }

  // Private validation methods

  /**
   * Validate data structure
   */
  private validateStructure(data: MarketDataPoint[]): string[] {
    const errors: string[] = [];
    
    for (let i = 0; i < data.length; i++) {
      if (!this.isValidDataPoint(data[i])) {
        errors.push(`Invalid data structure at index ${i}`);
        if (errors.length >= 10) break; // Limit error count
      }
    }
    
    return errors;
  }

  /**
   * Validate timestamps
   */
  private validateTimestamps(data: MarketDataPoint[]): string[] {
    const errors: string[] = [];
    
    for (let i = 1; i < data.length; i++) {
      const current = data[i];
      const previous = data[i - 1];
      
      if (current.timestamp <= previous.timestamp) {
        errors.push(`Non-chronological data at index ${i}`);
      }
      
      const timeGap = current.timestamp - previous.timestamp;
      if (timeGap > this.config.maxTimeGap && this.config.level === 'strict') {
        errors.push(`Large time gap at index ${i}: ${timeGap}ms`);
      }
    }
    
    return errors;
  }

  /**
   * Validate price data
   */
  private validatePrices(data: MarketDataPoint[]): string[] {
    const errors: string[] = [];
    
    for (let i = 0; i < data.length; i++) {
      const point = data[i];
      
      // OHLC consistency
      if (point.high < point.low) {
        errors.push(`High < Low at index ${i}`);
      }
      
      if (point.high < Math.max(point.open, point.close)) {
        errors.push(`High price inconsistent at index ${i}`);
      }
      
      if (point.low > Math.min(point.open, point.close)) {
        errors.push(`Low price inconsistent at index ${i}`);
      }
      
      // Price reasonableness
      if (point.open <= 0 || point.high <= 0 || point.low <= 0 || point.close <= 0) {
        errors.push(`Non-positive prices at index ${i}`);
      }
      
      // Price change validation
      if (i > 0) {
        const priceChange = Math.abs(point.close - data[i - 1].close) / data[i - 1].close;
        if (priceChange > this.config.maxPriceChange && this.config.level === 'strict') {
          errors.push(`Excessive price change at index ${i}: ${(priceChange * 100).toFixed(2)}%`);
        }
      }
    }
    
    return errors;
  }

  /**
   * Validate volume data
   */
  private validateVolumes(data: MarketDataPoint[]): string[] {
    const errors: string[] = [];
    
    for (let i = 0; i < data.length; i++) {
      if (data[i].volume < 0) {
        errors.push(`Negative volume at index ${i}`);
      }
      
      if (!Number.isFinite(data[i].volume)) {
        errors.push(`Invalid volume at index ${i}`);
      }
    }
    
    return errors;
  }

  /**
   * Detect outliers in the data
   */
  private detectOutliers(data: MarketDataPoint[]): number[] {
    if (data.length < 10) return []; // Need enough data for outlier detection
    
    const closes = data.map(d => d.close);
    const outlierIndices: number[] = [];
    
    switch (this.config.outlierMethod) {
      case 'iqr':
        outlierIndices.push(...this.detectOutliersIQR(closes));
        break;
      case 'zscore':
        outlierIndices.push(...this.detectOutliersZScore(closes));
        break;
      case 'modified_zscore':
        outlierIndices.push(...this.detectOutliersModifiedZScore(closes));
        break;
    }
    
    return outlierIndices;
  }

  /**
   * Detect outliers using IQR method
   */
  private detectOutliersIQR(values: number[]): number[] {
    const sorted = [...values].sort((a, b) => a - b);
    const q1Index = Math.floor(sorted.length * 0.25);
    const q3Index = Math.floor(sorted.length * 0.75);
    
    const q1 = sorted[q1Index];
    const q3 = sorted[q3Index];
    const iqr = q3 - q1;
    
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;
    
    const outliers: number[] = [];
    values.forEach((value, index) => {
      if (value < lowerBound || value > upperBound) {
        outliers.push(index);
      }
    });
    
    return outliers;
  }

  /**
   * Detect outliers using Z-score method
   */
  private detectOutliersZScore(values: number[]): number[] {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);
    
    const outliers: number[] = [];
    values.forEach((value, index) => {
      const zScore = Math.abs((value - mean) / stdDev);
      if (zScore > this.config.outlierThreshold) {
        outliers.push(index);
      }
    });
    
    return outliers;
  }

  /**
   * Detect outliers using Modified Z-score method
   */
  private detectOutliersModifiedZScore(values: number[]): number[] {
    const median = this.calculateMedian(values);
    const medianDeviations = values.map(val => Math.abs(val - median));
    const mad = this.calculateMedian(medianDeviations);
    
    const outliers: number[] = [];
    values.forEach((value, index) => {
      const modifiedZScore = 0.6745 * (value - median) / mad;
      if (Math.abs(modifiedZScore) > this.config.outlierThreshold) {
        outliers.push(index);
      }
    });
    
    return outliers;
  }

  /**
   * Detect duplicate entries
   */
  private detectDuplicates(data: MarketDataPoint[]): number[] {
    const seen = new Set<string>();
    const duplicates: number[] = [];
    
    data.forEach((point, index) => {
      const key = `${point.timestamp}-${point.open}-${point.high}-${point.low}-${point.close}-${point.volume}`;
      if (seen.has(key)) {
        duplicates.push(index);
      } else {
        seen.add(key);
      }
    });
    
    return duplicates;
  }

  /**
   * Calculate quality metrics
   */
  private calculateQualityMetrics(
    data: MarketDataPoint[], 
    issues: { outliers: number; duplicates: number; errors: number }
  ): DataQualityMetrics {
    const totalPoints = data.length;
    
    // Completeness: ratio of non-null/valid data points
    const completeness = Math.max(0, 1 - (issues.errors / totalPoints));
    
    // Consistency: based on OHLC relationships and logical consistency
    let consistentPoints = 0;
    data.forEach(point => {
      if (point.high >= point.low && 
          point.high >= Math.max(point.open, point.close) &&
          point.low <= Math.min(point.open, point.close)) {
        consistentPoints++;
      }
    });
    const consistency = consistentPoints / totalPoints;
    
    // Accuracy: inverse relationship with outliers
    const accuracy = Math.max(0, 1 - (issues.outliers / totalPoints));
    
    // Timeliness: based on timestamp consistency
    let timelyPoints = 0;
    for (let i = 1; i < data.length; i++) {
      if (data[i].timestamp > data[i - 1].timestamp) {
        timelyPoints++;
      }
    }
    const timeliness = totalPoints > 1 ? timelyPoints / (totalPoints - 1) : 1;
    
    // Uniqueness: inverse relationship with duplicates
    const uniqueness = Math.max(0, 1 - (issues.duplicates / totalPoints));
    
    return {
      completeness,
      consistency,
      accuracy,
      timeliness,
      uniqueness
    };
  }

  /**
   * Calculate overall quality score
   */
  private calculateOverallQuality(metrics: DataQualityMetrics): number {
    // Weighted average of quality metrics
    const weights = {
      completeness: 0.25,
      consistency: 0.25,
      accuracy: 0.20,
      timeliness: 0.15,
      uniqueness: 0.15
    };
    
    return (
      metrics.completeness * weights.completeness +
      metrics.consistency * weights.consistency +
      metrics.accuracy * weights.accuracy +
      metrics.timeliness * weights.timeliness +
      metrics.uniqueness * weights.uniqueness
    );
  }

  /**
   * Check if data point has valid structure
   */
  private isValidDataPoint(point: any): boolean {
    return point &&
           typeof point.timestamp === 'number' &&
           typeof point.open === 'number' &&
           typeof point.high === 'number' &&
           typeof point.low === 'number' &&
           typeof point.close === 'number' &&
           typeof point.volume === 'number' &&
           Number.isFinite(point.timestamp) &&
           Number.isFinite(point.open) &&
           Number.isFinite(point.high) &&
           Number.isFinite(point.low) &&
           Number.isFinite(point.close) &&
           Number.isFinite(point.volume);
  }

  /**
   * Calculate median of array
   */
  private calculateMedian(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  /**
   * Create failed validation result
   */
  private createFailResult(errors: string[], suggestions: string[]): DataValidationResult {
    return {
      isValid: false,
      errors,
      qualityScore: 0,
      suggestions,
      statistics: {
        dataPoints: 0,
        completeness: 0,
        duplicates: 0,
        outliers: 0
      }
    };
  }

  /**
   * Get default configuration for validation level
   */
  private getDefaultConfig(level: ValidationLevel): Required<ValidationConfig> {
    const baseConfig = {
      level,
      maxTimeGap: 24 * 60 * 60 * 1000, // 24 hours
      maxPriceChange: 0.2, // 20%
      minCompleteness: 0.95, // 95%
      outlierMethod: 'iqr' as OutlierMethod,
      outlierThreshold: 3
    };

    switch (level) {
      case 'strict':
        return {
          ...baseConfig,
          maxTimeGap: 2 * 60 * 60 * 1000, // 2 hours
          maxPriceChange: 0.1, // 10%
          minCompleteness: 0.98, // 98%
          outlierThreshold: 2.5
        };
      
      case 'relaxed':
        return {
          ...baseConfig,
          maxTimeGap: 7 * 24 * 60 * 60 * 1000, // 7 days
          maxPriceChange: 0.5, // 50%
          minCompleteness: 0.8, // 80%
          outlierThreshold: 4
        };
      
      default: // normal
        return baseConfig;
    }
  }
}