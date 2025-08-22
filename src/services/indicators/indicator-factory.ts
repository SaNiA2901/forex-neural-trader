/**
 * Technical Indicators Factory
 * Central factory for creating and managing technical indicators.
 */

import { RSIIndicator, RSIConfig } from './rsi/rsi-indicator';
import { MACDIndicator, MACDConfig } from './macd/macd-indicator';
import { BollingerBandsIndicator, BollingerBandsConfig } from './bollinger/bollinger-indicator';
import { StochasticIndicator, StochasticConfig } from './stochastic/stochastic-indicator';
import { 
  IndicatorType, 
  FactoryConfig, 
  BatchCalculationRequest, 
  BatchCalculationResult,
  MarketDataPoint
} from './core/types';

export class IndicatorFactory {
  private config: FactoryConfig;

  constructor(config?: FactoryConfig) {
    this.config = config || {
      cacheConfig: { maxSize: 1000, defaultTTL: 300000, cleanupInterval: 60000 },
      performanceConfig: { enableMetrics: true, metricsInterval: 30000, alertThresholds: { computationTime: 100, memoryUsage: 50 * 1024 * 1024 } }
    };
  }

  createRSI(config: Partial<RSIConfig> = {}): RSIIndicator {
    return new RSIIndicator(config as RSIConfig);
  }

  createMACD(config: Partial<MACDConfig> = {}): MACDIndicator {
    return new MACDIndicator(config as MACDConfig);
  }

  createBollingerBands(config: Partial<BollingerBandsConfig> = {}): BollingerBandsIndicator {
    return new BollingerBandsIndicator(config as BollingerBandsConfig);
  }

  createStochastic(config: Partial<StochasticConfig> = {}): StochasticIndicator {
    return new StochasticIndicator(config as StochasticConfig);
  }

  async calculateBatch(request: BatchCalculationRequest): Promise<BatchCalculationResult> {
    const startTime = performance.now();
    const results = new Map();
    const errors: any[] = [];

    for (const indicatorRequest of request.indicators) {
      try {
        let indicator;
        switch (indicatorRequest.type) {
          case 'rsi':
            indicator = this.createRSI(indicatorRequest.config as RSIConfig);
            break;
          case 'macd':
            indicator = this.createMACD(indicatorRequest.config as MACDConfig);
            break;
          case 'bollinger':
            indicator = this.createBollingerBands(indicatorRequest.config as BollingerBandsConfig);
            break;
          case 'stochastic':
            indicator = this.createStochastic(indicatorRequest.config as StochasticConfig);
            break;
          default:
            throw new Error(`Unsupported indicator type: ${indicatorRequest.type}`);
        }

        const result = await indicator.calculate(request.data);
        results.set(indicatorRequest.type, result);
      } catch (error) {
        errors.push({ type: indicatorRequest.type, error });
      }
    }

    return {
      results,
      totalTime: performance.now() - startTime,
      memoryUsage: process.memoryUsage().heapUsed,
      errors
    };
  }
}

export const indicatorFactory = new IndicatorFactory();