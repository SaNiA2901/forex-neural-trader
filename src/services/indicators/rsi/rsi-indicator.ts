/**
 * RSI (Relative Strength Index) Indicator
 * 
 * Professional implementation of RSI with configurable periods,
 * divergence detection, and overbought/oversold level analysis.
 * 
 * @author Trading System Core
 * @version 1.0.0
 */

import { BaseIndicator } from '../core/base-indicator';
import { 
  MarketDataPoint, 
  RSIResult, 
  BaseIndicatorConfig, 
  IndicatorError,
  StreamingUpdate
} from '../core/types';

/**
 * RSI-specific configuration
 */
export interface RSIConfig extends BaseIndicatorConfig {
  /** RSI calculation period (default: 14) */
  period: number;
  /** Overbought level threshold (default: 70) */
  overboughtLevel?: number;
  /** Oversold level threshold (default: 30) */
  oversoldLevel?: number;
  /** Smoothing method for RSI calculation */
  smoothingMethod?: 'wilder' | 'ema' | 'sma';
  /** Enable divergence detection */
  enableDivergenceDetection?: boolean;
}

/**
 * RSI calculation result with additional metadata
 */
interface RSICalculationResult {
  values: number[];
  gains: number[];
  losses: number[];
  avgGains: number[];
  avgLosses: number[];
  error?: IndicatorError;
}

/**
 * Professional RSI indicator implementation
 */
export class RSIIndicator extends BaseIndicator<number, RSIConfig> {
  private previousGain: number = 0;
  private previousLoss: number = 0;
  private isInitialized: boolean = false;

  /**
   * Get default RSI configuration
   */
  protected getDefaultConfig(): RSIConfig {
    return {
      period: 14,
      overboughtLevel: 70,
      oversoldLevel: 30,
      smoothingMethod: 'wilder',
      enableDivergenceDetection: true,
      enableCaching: true,
      enableStreaming: true,
      validationLevel: 'normal'
    };
  }

  /**
   * Get minimum data points required for RSI calculation
   */
  protected getMinDataPoints(): number {
    return this.config.period + 1;
  }

  /**
   * Get indicator name for identification
   */
  protected getIndicatorName(): string {
    return 'RSI';
  }

  /**
   * Perform RSI calculation
   * @param data - Validated market data
   */
  protected async performCalculation(data: MarketDataPoint[]): Promise<{
    values: number[];
    error?: IndicatorError;
  }> {
    try {
      const result = this.calculateRSI(data);
      return {
        values: result.values,
        error: result.error
      };
    } catch (error) {
      return {
        values: [],
        error: {
          code: 'RSI_CALCULATION_ERROR',
          message: error instanceof Error ? error.message : 'Unknown RSI calculation error',
          context: { 
            period: this.config.period,
            dataLength: data.length 
          },
          suggestions: [
            'Ensure sufficient historical data',
            'Check for valid price data',
            'Verify period configuration'
          ]
        }
      };
    }
  }

  /**
   * Calculate RSI with enhanced features
   * @param data - Market data points
   */
  private calculateRSI(data: MarketDataPoint[]): RSICalculationResult {
    const closes = data.map(point => point.close);
    const period = this.config.period;
    
    if (closes.length < period + 1) {
      throw new Error(`Insufficient data: need ${period + 1} points, got ${closes.length}`);
    }

    const gains: number[] = [];
    const losses: number[] = [];
    const rsiValues: number[] = [];
    const avgGains: number[] = [];
    const avgLosses: number[] = [];

    // Calculate price changes
    for (let i = 1; i < closes.length; i++) {
      const change = closes[i] - closes[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? -change : 0);
    }

    // Calculate initial averages
    let sumGains = 0;
    let sumLosses = 0;
    
    for (let i = 0; i < period; i++) {
      sumGains += gains[i];
      sumLosses += losses[i];
    }

    let avgGain = sumGains / period;
    let avgLoss = sumLosses / period;
    
    avgGains.push(avgGain);
    avgLosses.push(avgLoss);

    // Calculate first RSI value
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    const rsi = 100 - (100 / (1 + rs));
    rsiValues.push(rsi);

    // Calculate subsequent RSI values using selected smoothing method
    for (let i = period; i < gains.length; i++) {
      switch (this.config.smoothingMethod) {
        case 'wilder':
          avgGain = this.wilderSmoothing(avgGain, gains[i], period);
          avgLoss = this.wilderSmoothing(avgLoss, losses[i], period);
          break;
        case 'ema':
          avgGain = this.emaSmoothing(avgGain, gains[i], period);
          avgLoss = this.emaSmoothing(avgLoss, losses[i], period);
          break;
        case 'sma':
          avgGain = this.smaSmoothing(gains, i, period);
          avgLoss = this.smaSmoothing(losses, i, period);
          break;
        default:
          avgGain = this.wilderSmoothing(avgGain, gains[i], period);
          avgLoss = this.wilderSmoothing(avgLoss, losses[i], period);
      }

      avgGains.push(avgGain);
      avgLosses.push(avgLoss);

      const currentRS = avgLoss === 0 ? 100 : avgGain / avgLoss;
      const currentRSI = 100 - (100 / (1 + currentRS));
      rsiValues.push(currentRSI);
    }

    // Pad with NaN for initial values
    const paddedRSI: number[] = [];
    for (let i = 0; i < period; i++) {
      paddedRSI.push(NaN);
    }
    paddedRSI.push(...rsiValues);

    return {
      values: paddedRSI,
      gains,
      losses,
      avgGains,
      avgLosses
    };
  }

  /**
   * Calculate RSI for the complete dataset and return enhanced result
   * @param data - Market data points
   */
  async calculate(data: MarketDataPoint[]): Promise<RSIResult> {
    const baseResult = await super.calculate(data);
    
    if (baseResult.error || baseResult.values.length === 0) {
      return {
        ...baseResult,
        level: 'neutral',
        divergence: undefined
      } as RSIResult;
    }

    // Analyze RSI levels
    const currentRSI = baseResult.values[baseResult.values.length - 1];
    const level = this.classifyRSILevel(currentRSI);

    // Detect divergences if enabled
    let divergence: RSIResult['divergence'];
    if (this.config.enableDivergenceDetection) {
      divergence = this.detectDivergence(data, baseResult.values);
    }

    return {
      ...baseResult,
      level,
      divergence
    };
  }

  /**
   * Streaming update with RSI-specific level crossing detection
   */
  async update(
    newDataPoint: MarketDataPoint, 
    existingData: MarketDataPoint[]
  ): Promise<StreamingUpdate<number>> {
    const baseUpdate = await super.update(newDataPoint, existingData);
    
    // Add RSI-specific level crossing detection
    const levelCross = this.detectRSILevelCross(
      baseUpdate.value.value, 
      existingData.length > 0 ? 
        (await this.calculate(existingData)).values[existingData.length - 1] : 
        baseUpdate.value.value
    );

    return {
      ...baseUpdate,
      levelCross
    };
  }

  /**
   * Wilder's smoothing method (original RSI)
   */
  private wilderSmoothing(previousAvg: number, newValue: number, period: number): number {
    return (previousAvg * (period - 1) + newValue) / period;
  }

  /**
   * Exponential moving average smoothing
   */
  private emaSmoothing(previousAvg: number, newValue: number, period: number): number {
    const alpha = 2 / (period + 1);
    return alpha * newValue + (1 - alpha) * previousAvg;
  }

  /**
   * Simple moving average smoothing
   */
  private smaSmoothing(values: number[], currentIndex: number, period: number): number {
    const start = Math.max(0, currentIndex - period + 1);
    const end = currentIndex + 1;
    const slice = values.slice(start, end);
    return slice.reduce((sum, val) => sum + val, 0) / slice.length;
  }

  /**
   * Classify RSI level
   */
  private classifyRSILevel(rsi: number): 'oversold' | 'neutral' | 'overbought' {
    if (rsi >= (this.config.overboughtLevel || 70)) {
      return 'overbought';
    } else if (rsi <= (this.config.oversoldLevel || 30)) {
      return 'oversold';
    }
    return 'neutral';
  }

  /**
   * Detect RSI divergences
   */
  private detectDivergence(
    priceData: MarketDataPoint[], 
    rsiValues: number[]
  ): RSIResult['divergence'] {
    const minPeriod = 10; // Minimum periods to look for divergence
    if (priceData.length < minPeriod || rsiValues.length < minPeriod) {
      return undefined;
    }

    const recentPrices = priceData.slice(-minPeriod).map(d => d.close);
    const recentRSI = rsiValues.slice(-minPeriod).filter(v => !isNaN(v));

    if (recentRSI.length < minPeriod) return undefined;

    // Simple divergence detection
    const priceSlope = this.calculateSlope(recentPrices);
    const rsiSlope = this.calculateSlope(recentRSI);

    // Bullish divergence: price falling, RSI rising
    if (priceSlope < -0.001 && rsiSlope > 0.1) {
      return {
        type: 'bullish',
        strength: Math.min(1, Math.abs(priceSlope) + rsiSlope)
      };
    }

    // Bearish divergence: price rising, RSI falling
    if (priceSlope > 0.001 && rsiSlope < -0.1) {
      return {
        type: 'bearish',
        strength: Math.min(1, priceSlope + Math.abs(rsiSlope))
      };
    }

    return undefined;
  }

  /**
   * Calculate slope of a data series
   */
  private calculateSlope(values: number[]): number {
    const n = values.length;
    if (n < 2) return 0;

    const x = Array.from({ length: n }, (_, i) => i);
    const sumX = x.reduce((sum, val) => sum + val, 0);
    const sumY = values.reduce((sum, val) => sum + val, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * values[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    return isNaN(slope) ? 0 : slope;
  }

  /**
   * Detect RSI level crossovers
   */
  private detectRSILevelCross(
    currentRSI: number, 
    previousRSI: number
  ): { level: number; direction: 'up' | 'down' } | undefined {
    const overbought = this.config.overboughtLevel || 70;
    const oversold = this.config.oversoldLevel || 30;

    // Crossing overbought level
    if (previousRSI < overbought && currentRSI >= overbought) {
      return { level: overbought, direction: 'up' };
    }
    if (previousRSI > overbought && currentRSI <= overbought) {
      return { level: overbought, direction: 'down' };
    }

    // Crossing oversold level
    if (previousRSI > oversold && currentRSI <= oversold) {
      return { level: oversold, direction: 'down' };
    }
    if (previousRSI < oversold && currentRSI >= oversold) {
      return { level: oversold, direction: 'up' };
    }

    return undefined;
  }

  /**
   * Get current RSI level classification
   */
  getCurrentLevel(rsi: number): 'oversold' | 'neutral' | 'overbought' {
    return this.classifyRSILevel(rsi);
  }

  /**
   * Calculate RSI for a single new value (optimized for streaming)
   */
  calculateStreamingRSI(newPrice: number, previousPrice: number): number {
    if (!this.isInitialized) {
      throw new Error('RSI not initialized for streaming. Call calculate() first.');
    }

    const change = newPrice - previousPrice;
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? -change : 0;

    // Update averages using Wilder's smoothing
    this.previousGain = this.wilderSmoothing(this.previousGain, gain, this.config.period);
    this.previousLoss = this.wilderSmoothing(this.previousLoss, loss, this.config.period);

    const rs = this.previousLoss === 0 ? 100 : this.previousGain / this.previousLoss;
    return 100 - (100 / (1 + rs));
  }

  /**
   * Initialize streaming RSI with historical data
   */
  initializeStreaming(data: MarketDataPoint[]): void {
    const result = this.calculateRSI(data);
    if (result.avgGains.length > 0 && result.avgLosses.length > 0) {
      this.previousGain = result.avgGains[result.avgGains.length - 1];
      this.previousLoss = result.avgLosses[result.avgLosses.length - 1];
      this.isInitialized = true;
    }
  }

  /**
   * Reset streaming state
   */
  resetStreaming(): void {
    this.previousGain = 0;
    this.previousLoss = 0;
    this.isInitialized = false;
  }
}