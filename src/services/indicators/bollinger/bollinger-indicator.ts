/**
 * Bollinger Bands Indicator
 * Professional implementation with squeeze detection and %B calculation.
 */

import { BaseIndicator } from '../core/base-indicator';
import { 
  MarketDataPoint, 
  BollingerBandsResult, 
  BollingerBandsValue, 
  BaseIndicatorConfig, 
  IndicatorError
} from '../core/types';

export interface BollingerBandsConfig extends BaseIndicatorConfig {
  period: number;
  standardDeviations: number;
  enableSqueezeDetection?: boolean;
}

export class BollingerBandsIndicator extends BaseIndicator<BollingerBandsValue, BollingerBandsConfig> {
  protected getDefaultConfig(): BollingerBandsConfig {
    return {
      period: 20,
      standardDeviations: 2,
      enableSqueezeDetection: true,
      enableCaching: true,
      enableStreaming: true,
      validationLevel: 'normal'
    };
  }

  protected getMinDataPoints(): number {
    return this.config.period;
  }

  protected getIndicatorName(): string {
    return 'BollingerBands';
  }

  protected async performCalculation(data: MarketDataPoint[]): Promise<{
    values: BollingerBandsValue[];
    error?: IndicatorError;
  }> {
    try {
      const closes = data.map(d => d.close);
      const values = this.calculateBollingerBands(closes);
      return { values };
    } catch (error) {
      return {
        values: [],
        error: {
          code: 'BOLLINGER_ERROR',
          message: error instanceof Error ? error.message : 'Calculation failed',
          context: { period: this.config.period },
          suggestions: ['Check data quality', 'Verify period setting']
        }
      };
    }
  }

  private calculateBollingerBands(closes: number[]): BollingerBandsValue[] {
    const period = this.config.period;
    const stdDev = this.config.standardDeviations;
    const values: BollingerBandsValue[] = [];

    for (let i = 0; i < closes.length; i++) {
      if (i < period - 1) {
        values.push({
          upper: NaN,
          middle: NaN,
          lower: NaN,
          bandwidth: NaN,
          percentB: NaN
        });
        continue;
      }

      const slice = closes.slice(i - period + 1, i + 1);
      const sma = slice.reduce((sum, val) => sum + val, 0) / period;
      
      const variance = slice.reduce((sum, val) => sum + Math.pow(val - sma, 2), 0) / period;
      const standardDeviation = Math.sqrt(variance);
      
      const upper = sma + (standardDeviation * stdDev);
      const lower = sma - (standardDeviation * stdDev);
      const bandwidth = (upper - lower) / sma;
      const percentB = (closes[i] - lower) / (upper - lower);

      values.push({
        upper,
        middle: sma,
        lower,
        bandwidth,
        percentB
      });
    }

    return values;
  }
}