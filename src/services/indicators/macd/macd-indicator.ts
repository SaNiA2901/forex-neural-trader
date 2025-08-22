/**
 * MACD (Moving Average Convergence Divergence) Indicator
 * 
 * Professional implementation with signal line, histogram, and crossover detection.
 * Supports multiple EMA calculation methods and advanced signal analysis.
 * 
 * @author Trading System Core
 * @version 1.0.0
 */

import { BaseIndicator } from '../core/base-indicator';
import { 
  MarketDataPoint, 
  MACDResult, 
  MACDValue, 
  BaseIndicatorConfig, 
  IndicatorError,
  StreamingUpdate
} from '../core/types';

/**
 * MACD-specific configuration
 */
export interface MACDConfig extends BaseIndicatorConfig {
  /** Fast EMA period (default: 12) */
  fastPeriod: number;
  /** Slow EMA period (default: 26) */
  slowPeriod: number;
  /** Signal line EMA period (default: 9) */
  signalPeriod: number;
  /** Price field to use for calculation */
  priceField?: 'close' | 'open' | 'high' | 'low' | 'hl2' | 'hlc3' | 'ohlc4';
  /** Enable crossover detection */
  enableCrossoverDetection?: boolean;
  /** Minimum crossover strength threshold */
  crossoverThreshold?: number;
}

/**
 * MACD calculation state for streaming
 */
interface MACDState {
  fastEMA: number;
  slowEMA: number;
  signalEMA: number;
  isInitialized: boolean;
  dataCount: number;
}

/**
 * Professional MACD indicator implementation
 */
export class MACDIndicator extends BaseIndicator<MACDValue, MACDConfig> {
  private streamingState: MACDState = {
    fastEMA: 0,
    slowEMA: 0,
    signalEMA: 0,
    isInitialized: false,
    dataCount: 0
  };

  /**
   * Get default MACD configuration
   */
  protected getDefaultConfig(): MACDConfig {
    return {
      period: 26, // Use slow period as primary period
      fastPeriod: 12,
      slowPeriod: 26,
      signalPeriod: 9,
      priceField: 'close',
      enableCrossoverDetection: true,
      crossoverThreshold: 0.01,
      enableCaching: true,
      enableStreaming: true,
      validationLevel: 'normal'
    };
  }

  /**
   * Get minimum data points required
   */
  protected getMinDataPoints(): number {
    return this.config.slowPeriod + this.config.signalPeriod;
  }

  /**
   * Get indicator name
   */
  protected getIndicatorName(): string {
    return 'MACD';
  }

  /**
   * Perform MACD calculation
   */
  protected async performCalculation(data: MarketDataPoint[]): Promise<{
    values: MACDValue[];
    error?: IndicatorError;
  }> {
    try {
      const prices = this.extractPrices(data);
      const result = this.calculateMACD(prices);
      
      return {
        values: result.values,
        error: result.error
      };
    } catch (error) {
      return {
        values: [],
        error: {
          code: 'MACD_CALCULATION_ERROR',
          message: error instanceof Error ? error.message : 'Unknown MACD calculation error',
          context: { 
            fastPeriod: this.config.fastPeriod,
            slowPeriod: this.config.slowPeriod,
            signalPeriod: this.config.signalPeriod,
            dataLength: data.length 
          },
          suggestions: [
            'Ensure sufficient historical data',
            'Check EMA period configurations',
            'Verify price data quality'
          ]
        }
      };
    }
  }

  /**
   * Calculate MACD with enhanced features
   */
  private calculateMACD(prices: number[]): {
    values: MACDValue[];
    fastEMA: number[];
    slowEMA: number[];
    signalEMA: number[];
    error?: IndicatorError;
  } {
    if (prices.length < this.getMinDataPoints()) {
      throw new Error(`Insufficient data: need ${this.getMinDataPoints()} points, got ${prices.length}`);
    }

    // Calculate fast and slow EMAs
    const fastEMA = this.calculateEMA(prices, this.config.fastPeriod);
    const slowEMA = this.calculateEMA(prices, this.config.slowPeriod);

    // Calculate MACD line
    const macdLine: number[] = [];
    const validFrom = Math.max(this.config.fastPeriod, this.config.slowPeriod) - 1;
    
    for (let i = 0; i < prices.length; i++) {
      if (i >= validFrom) {
        macdLine.push(fastEMA[i] - slowEMA[i]);
      } else {
        macdLine.push(NaN);
      }
    }

    // Calculate signal line (EMA of MACD line)
    const validMACDValues = macdLine.filter(v => !isNaN(v));
    const signalEMA = this.calculateEMA(validMACDValues, this.config.signalPeriod);

    // Prepare final MACD values
    const macdValues: MACDValue[] = [];
    
    for (let i = 0; i < prices.length; i++) {
      const signalIndex = i - validFrom;
      const signal = signalIndex >= 0 && signalIndex < signalEMA.length ? 
        signalEMA[signalIndex] : NaN;
      
      const macd = macdLine[i];
      const histogram = !isNaN(macd) && !isNaN(signal) ? macd - signal : NaN;

      macdValues.push({
        macd,
        signal,
        histogram
      });
    }

    return {
      values: macdValues,
      fastEMA,
      slowEMA,
      signalEMA
    };
  }

  /**
   * Calculate EMA with proper initialization
   */
  private calculateEMA(prices: number[], period: number): number[] {
    if (prices.length === 0) return [];

    const ema: number[] = [];
    const multiplier = 2 / (period + 1);

    // Initialize with first price
    ema[0] = prices[0];

    // Calculate EMA for subsequent values
    for (let i = 1; i < prices.length; i++) {
      ema[i] = (prices[i] * multiplier) + (ema[i - 1] * (1 - multiplier));
    }

    return ema;
  }

  /**
   * Enhanced calculate method with crossover detection
   */
  async calculate(data: MarketDataPoint[]): Promise<MACDResult> {
    const baseResult = await super.calculate(data);
    
    if (baseResult.error || baseResult.values.length === 0) {
      return {
        ...baseResult,
        crossovers: []
      } as MACDResult;
    }

    // Detect crossovers if enabled
    let crossovers: MACDResult['crossovers'] = [];
    if (this.config.enableCrossoverDetection) {
      crossovers = this.detectCrossovers(data, baseResult.values);
    }

    return {
      ...baseResult,
      crossovers
    };
  }

  /**
   * Streaming update with MACD-specific optimizations
   */
  async update(
    newDataPoint: MarketDataPoint, 
    existingData: MarketDataPoint[]
  ): Promise<StreamingUpdate<MACDValue>> {
    const newPrice = this.extractPrice(newDataPoint);
    
    if (!this.streamingState.isInitialized) {
      // Initialize streaming state with existing data
      await this.initializeStreaming(existingData);
    }

    // Calculate new MACD value using streaming EMAs
    const newMACDValue = this.calculateStreamingMACD(newPrice);
    
    // Get previous value for comparison
    const allData = [...existingData, newDataPoint];
    const result = await this.calculate(allData);
    const previousValue = result.values.length > 1 ? 
      result.values[result.values.length - 2] : newMACDValue;

    const change = this.calculateMACDChange(newMACDValue, previousValue);
    const percentChange = this.calculateMACDPercentChange(newMACDValue, previousValue);

    return {
      value: {
        timestamp: newDataPoint.timestamp,
        value: newMACDValue,
        isValid: !isNaN(newMACDValue.macd) && !isNaN(newMACDValue.signal),
        confidence: this.calculateMACDConfidence(newMACDValue)
      },
      change,
      percentChange,
      levelCross: this.detectMACDCross(newMACDValue, previousValue)
    };
  }

  /**
   * Detect MACD crossovers
   */
  private detectCrossovers(
    data: MarketDataPoint[], 
    macdValues: MACDValue[]
  ): MACDResult['crossovers'] {
    const crossovers: MACDResult['crossovers'] = [];
    const threshold = this.config.crossoverThreshold || 0.01;

    for (let i = 1; i < macdValues.length; i++) {
      const current = macdValues[i];
      const previous = macdValues[i - 1];

      if (isNaN(current.macd) || isNaN(current.signal) || 
          isNaN(previous.macd) || isNaN(previous.signal)) {
        continue;
      }

      // Bullish crossover: MACD crosses above signal
      if (previous.macd <= previous.signal && current.macd > current.signal) {
        const strength = Math.abs(current.macd - current.signal);
        if (strength >= threshold) {
          crossovers.push({
            timestamp: data[i].timestamp,
            type: 'bullish',
            strength: Math.min(1, strength / threshold)
          });
        }
      }

      // Bearish crossover: MACD crosses below signal
      if (previous.macd >= previous.signal && current.macd < current.signal) {
        const strength = Math.abs(current.macd - current.signal);
        if (strength >= threshold) {
          crossovers.push({
            timestamp: data[i].timestamp,
            type: 'bearish',
            strength: Math.min(1, strength / threshold)
          });
        }
      }
    }

    return crossovers;
  }

  /**
   * Initialize streaming state
   */
  private async initializeStreaming(data: MarketDataPoint[]): Promise<void> {
    if (data.length < this.getMinDataPoints()) {
      throw new Error('Insufficient data to initialize MACD streaming');
    }

    const prices = this.extractPrices(data);
    const fastEMA = this.calculateEMA(prices, this.config.fastPeriod);
    const slowEMA = this.calculateEMA(prices, this.config.slowPeriod);
    
    // Calculate MACD line
    const macdLine: number[] = [];
    for (let i = 0; i < prices.length; i++) {
      macdLine.push(fastEMA[i] - slowEMA[i]);
    }
    
    // Calculate signal line
    const signalEMA = this.calculateEMA(macdLine, this.config.signalPeriod);

    this.streamingState = {
      fastEMA: fastEMA[fastEMA.length - 1],
      slowEMA: slowEMA[slowEMA.length - 1],
      signalEMA: signalEMA[signalEMA.length - 1],
      isInitialized: true,
      dataCount: data.length
    };
  }

  /**
   * Calculate streaming MACD value
   */
  private calculateStreamingMACD(newPrice: number): MACDValue {
    if (!this.streamingState.isInitialized) {
      throw new Error('MACD streaming not initialized');
    }

    // Update EMAs
    const fastMultiplier = 2 / (this.config.fastPeriod + 1);
    const slowMultiplier = 2 / (this.config.slowPeriod + 1);
    
    this.streamingState.fastEMA = (newPrice * fastMultiplier) + 
      (this.streamingState.fastEMA * (1 - fastMultiplier));
    
    this.streamingState.slowEMA = (newPrice * slowMultiplier) + 
      (this.streamingState.slowEMA * (1 - slowMultiplier));

    // Calculate MACD line
    const macd = this.streamingState.fastEMA - this.streamingState.slowEMA;

    // Update signal line
    const signalMultiplier = 2 / (this.config.signalPeriod + 1);
    this.streamingState.signalEMA = (macd * signalMultiplier) + 
      (this.streamingState.signalEMA * (1 - signalMultiplier));

    // Calculate histogram
    const histogram = macd - this.streamingState.signalEMA;

    this.streamingState.dataCount++;

    return {
      macd,
      signal: this.streamingState.signalEMA,
      histogram
    };
  }

  /**
   * Extract prices from market data based on price field configuration
   */
  private extractPrices(data: MarketDataPoint[]): number[] {
    return data.map(point => this.extractPrice(point));
  }

  /**
   * Extract single price from market data point
   */
  private extractPrice(point: MarketDataPoint): number {
    switch (this.config.priceField) {
      case 'open': return point.open;
      case 'high': return point.high;
      case 'low': return point.low;
      case 'close': return point.close;
      case 'hl2': return (point.high + point.low) / 2;
      case 'hlc3': return (point.high + point.low + point.close) / 3;
      case 'ohlc4': return (point.open + point.high + point.low + point.close) / 4;
      default: return point.close;
    }
  }

  /**
   * Calculate change between MACD values
   */
  protected calculateChange(newValue: MACDValue, previousValue: MACDValue): number {
    return this.calculateMACDChange(newValue, previousValue);
  }

  private calculateMACDChange(newValue: MACDValue, previousValue: MACDValue): number {
    if (isNaN(newValue.macd) || isNaN(previousValue.macd)) return 0;
    return newValue.macd - previousValue.macd;
  }

  /**
   * Calculate percentage change between MACD values
   */
  protected calculatePercentChange(newValue: MACDValue, previousValue: MACDValue): number {
    return this.calculateMACDPercentChange(newValue, previousValue);
  }

  private calculateMACDPercentChange(newValue: MACDValue, previousValue: MACDValue): number {
    if (isNaN(newValue.macd) || isNaN(previousValue.macd) || previousValue.macd === 0) return 0;
    return ((newValue.macd - previousValue.macd) / Math.abs(previousValue.macd)) * 100;
  }

  /**
   * Detect MACD level crossovers
   */
  private detectMACDCross(
    newValue: MACDValue, 
    previousValue: MACDValue
  ): { level: number; direction: 'up' | 'down' } | undefined {
    // Zero line crossover
    if (previousValue.macd <= 0 && newValue.macd > 0) {
      return { level: 0, direction: 'up' };
    }
    if (previousValue.macd >= 0 && newValue.macd < 0) {
      return { level: 0, direction: 'down' };
    }

    // Signal line crossover
    if (previousValue.macd <= previousValue.signal && newValue.macd > newValue.signal) {
      return { level: newValue.signal, direction: 'up' };
    }
    if (previousValue.macd >= previousValue.signal && newValue.macd < newValue.signal) {
      return { level: newValue.signal, direction: 'down' };
    }

    return undefined;
  }

  /**
   * Calculate confidence for MACD value
   */
  private calculateMACDConfidence(value: MACDValue): number {
    if (isNaN(value.macd) || isNaN(value.signal)) return 0;
    
    // Higher confidence when MACD and signal are clearly separated
    const separation = Math.abs(value.macd - value.signal);
    const confidence = Math.min(1, separation / 0.1); // Normalize to 0-1
    
    return confidence;
  }

  /**
   * Reset streaming state
   */
  resetStreaming(): void {
    this.streamingState = {
      fastEMA: 0,
      slowEMA: 0,
      signalEMA: 0,
      isInitialized: false,
      dataCount: 0
    };
  }

  /**
   * Get current streaming state
   */
  getStreamingState(): MACDState {
    return { ...this.streamingState };
  }
}