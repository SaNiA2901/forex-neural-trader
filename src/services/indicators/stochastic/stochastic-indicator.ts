/**
 * Stochastic Oscillator Indicator
 * Professional implementation with %K and %D lines.
 */

import { BaseIndicator } from '../core/base-indicator';
import { 
  MarketDataPoint, 
  StochasticResult, 
  StochasticValue, 
  BaseIndicatorConfig, 
  IndicatorError
} from '../core/types';

export interface StochasticConfig extends BaseIndicatorConfig {
  period: number;
  kSmoothing: number;
  dSmoothing: number;
  overboughtLevel?: number;
  oversoldLevel?: number;
}

export class StochasticIndicator extends BaseIndicator<StochasticValue, StochasticConfig> {
  protected getDefaultConfig(): StochasticConfig {
    return {
      period: 14,
      kSmoothing: 3,
      dSmoothing: 3,
      overboughtLevel: 80,
      oversoldLevel: 20,
      enableCaching: true,
      enableStreaming: true,
      validationLevel: 'normal'
    };
  }

  protected getMinDataPoints(): number {
    return this.config.period + this.config.kSmoothing + this.config.dSmoothing;
  }

  protected getIndicatorName(): string {
    return 'Stochastic';
  }

  protected async performCalculation(data: MarketDataPoint[]): Promise<{
    values: StochasticValue[];
    error?: IndicatorError;
  }> {
    try {
      const values = this.calculateStochastic(data);
      return { values };
    } catch (error) {
      return {
        values: [],
        error: {
          code: 'STOCHASTIC_ERROR',
          message: error instanceof Error ? error.message : 'Calculation failed',
          context: { period: this.config.period },
          suggestions: ['Check data sufficiency', 'Verify configuration']
        }
      };
    }
  }

  private calculateStochastic(data: MarketDataPoint[]): StochasticValue[] {
    const period = this.config.period;
    const values: StochasticValue[] = [];

    // Calculate %K values
    const rawK: number[] = [];
    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) {
        rawK.push(NaN);
        continue;
      }

      const slice = data.slice(i - period + 1, i + 1);
      const highest = Math.max(...slice.map(d => d.high));
      const lowest = Math.min(...slice.map(d => d.low));
      const current = data[i].close;

      const k = ((current - lowest) / (highest - lowest)) * 100;
      rawK.push(isNaN(k) ? 50 : k);
    }

    // Smooth %K and calculate %D
    for (let i = 0; i < data.length; i++) {
      const kStart = Math.max(0, i - this.config.kSmoothing + 1);
      const kSlice = rawK.slice(kStart, i + 1).filter(v => !isNaN(v));
      const smoothK = kSlice.length > 0 ? kSlice.reduce((sum, val) => sum + val, 0) / kSlice.length : NaN;

      values.push({ k: smoothK, d: NaN });
    }

    // Calculate %D (SMA of %K)
    for (let i = 0; i < values.length; i++) {
      const dStart = Math.max(0, i - this.config.dSmoothing + 1);
      const dSlice = values.slice(dStart, i + 1).map(v => v.k).filter(v => !isNaN(v));
      const d = dSlice.length > 0 ? dSlice.reduce((sum, val) => sum + val, 0) / dSlice.length : NaN;
      values[i].d = d;
    }

    return values;
  }
}