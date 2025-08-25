import { CandleData } from '@/types/session';

export interface MarketCondition {
  type: 'trending' | 'ranging' | 'volatile' | 'breakout';
  strength: number; // 0-1
  duration: number; // in candles
}

export interface TestDataConfig {
  symbol: string;
  startPrice: number;
  candleCount: number;
  timeframe: number; // milliseconds between candles
  conditions: MarketCondition[];
  addNoise: boolean;
  noiseLevel: number;
}

export class TestDataGenerator {
  private static instance: TestDataGenerator;
  
  static getInstance(): TestDataGenerator {
    if (!TestDataGenerator.instance) {
      TestDataGenerator.instance = new TestDataGenerator();
    }
    return TestDataGenerator.instance;
  }

  /**
   * Generate realistic market data with various conditions
   */
  generateMarketData(config: TestDataConfig): CandleData[] {
    const candles: CandleData[] = [];
    let currentPrice = config.startPrice;
    let currentTime = Date.now();
    
    let conditionIndex = 0;
    let conditionProgress = 0;
    let activeCondition = config.conditions[0] || { type: 'ranging', strength: 0.5, duration: config.candleCount };

    for (let i = 0; i < config.candleCount; i++) {
      // Switch market condition if needed
      if (conditionProgress >= activeCondition.duration && conditionIndex < config.conditions.length - 1) {
        conditionIndex++;
        activeCondition = config.conditions[conditionIndex];
        conditionProgress = 0;
      }

      // Generate price movement based on current condition
      const movement = this.generatePriceMovement(activeCondition, conditionProgress);
      const baseChange = currentPrice * movement;
      
      // Add noise if configured
      const noise = config.addNoise ? (Math.random() - 0.5) * config.noiseLevel * currentPrice : 0;
      const priceChange = baseChange + noise;
      
      // Calculate OHLC
      const open = currentPrice;
      const close = currentPrice + priceChange;
      const spread = Math.abs(priceChange) * (0.5 + Math.random() * 0.5);
      const high = Math.max(open, close) + spread * Math.random();
      const low = Math.min(open, close) - spread * Math.random();
      
      // Generate volume (more volume during volatile periods)
      const volatilityFactor = Math.abs(priceChange / currentPrice) * 100;
      const baseVolume = 1000 + Math.random() * 1000;
      const volume = baseVolume * (1 + volatilityFactor);

      candles.push({
        id: `${config.symbol}_${i}`,
        timestamp: currentTime + i * config.timeframe,
        open,
        high,
        low,
        close,
        volume,
        sessionId: `test_session_${config.symbol}`
      } as CandleData);

      currentPrice = close;
      conditionProgress++;
    }

    return candles;
  }

  /**
   * Generate candles with specific patterns for testing
   */
  generatePatternData(pattern: 'bullish_engulfing' | 'doji' | 'hammer' | 'shooting_star', count: number = 50): CandleData[] {
    const candles: CandleData[] = [];
    let price = 1.2000;
    const baseTime = Date.now();

    for (let i = 0; i < count; i++) {
      let candle: Partial<CandleData>;

      if (i === Math.floor(count / 2)) {
        // Insert the pattern in the middle
        candle = this.generateSpecificPattern(pattern, price, i);
      } else {
        // Generate normal candle
        const change = (Math.random() - 0.5) * 0.002;
        const open = price;
        const close = price + change;
        const spread = Math.abs(change) * 0.5;
        
        candle = {
          open,
          close,
          high: Math.max(open, close) + spread * Math.random(),
          low: Math.min(open, close) - spread * Math.random(),
          volume: 1000 + Math.random() * 500
        };
      }

      candles.push({
        id: `pattern_${i}`,
        timestamp: baseTime + i * 60000,
        sessionId: 'pattern_test',
        ...candle
      } as CandleData);

      price = candle.close!;
    }

    return candles;
  }

  /**
   * Generate data for backtesting with known outcomes
   */
  generateBacktestData(scenarios: 'bull_market' | 'bear_market' | 'sideways' | 'mixed'): {
    candles: CandleData[];
    expectedOutcome: {
      trend: string;
      volatility: 'low' | 'medium' | 'high';
      expectedProfitability: 'positive' | 'negative' | 'neutral';
    };
  } {
    let config: TestDataConfig;
    let expectedOutcome: any;

    switch (scenarios) {
      case 'bull_market':
        config = {
          symbol: 'BULL',
          startPrice: 1.2000,
          candleCount: 1000,
          timeframe: 60000,
          conditions: [
            { type: 'trending', strength: 0.8, duration: 800 },
            { type: 'ranging', strength: 0.3, duration: 200 }
          ],
          addNoise: true,
          noiseLevel: 0.001
        };
        expectedOutcome = {
          trend: 'bullish',
          volatility: 'medium',
          expectedProfitability: 'positive'
        };
        break;
        
      case 'bear_market':
        config = {
          symbol: 'BEAR',
          startPrice: 1.2000,
          candleCount: 1000,
          timeframe: 60000,
          conditions: [
            { type: 'trending', strength: -0.8, duration: 800 },
            { type: 'ranging', strength: 0.3, duration: 200 }
          ],
          addNoise: true,
          noiseLevel: 0.001
        };
        expectedOutcome = {
          trend: 'bearish',
          volatility: 'medium',
          expectedProfitability: 'negative'
        };
        break;
        
      case 'sideways':
        config = {
          symbol: 'SIDE',
          startPrice: 1.2000,
          candleCount: 1000,
          timeframe: 60000,
          conditions: [
            { type: 'ranging', strength: 0.5, duration: 1000 }
          ],
          addNoise: true,
          noiseLevel: 0.0005
        };
        expectedOutcome = {
          trend: 'sideways',
          volatility: 'low',
          expectedProfitability: 'neutral'
        };
        break;
        
      default:
        config = {
          symbol: 'MIXED',
          startPrice: 1.2000,
          candleCount: 1000,
          timeframe: 60000,
          conditions: [
            { type: 'trending', strength: 0.6, duration: 250 },
            { type: 'ranging', strength: 0.4, duration: 200 },
            { type: 'volatile', strength: 0.8, duration: 150 },
            { type: 'trending', strength: -0.5, duration: 250 },
            { type: 'breakout', strength: 0.7, duration: 150 }
          ],
          addNoise: true,
          noiseLevel: 0.002
        };
        expectedOutcome = {
          trend: 'mixed',
          volatility: 'high',
          expectedProfitability: 'neutral'
        };
    }

    return {
      candles: this.generateMarketData(config),
      expectedOutcome
    };
  }

  /**
   * Generate reference data for validation against known indicators
   */
  generateReferenceData(): {
    candles: CandleData[];
    expectedIndicators: {
      sma20: number[];
      rsi14: number[];
      macd: { line: number[]; signal: number[]; histogram: number[] };
    };
  } {
    // Generate simple data where we can calculate expected indicators manually
    const candles: CandleData[] = [];
    const closes: number[] = [];
    
    // Create a simple uptrend for predictable indicators
    for (let i = 0; i < 100; i++) {
      const close = 100 + i * 0.1; // Simple linear increase
      closes.push(close);
      
      candles.push({
        id: `ref_${i}`,
        timestamp: Date.now() + i * 60000,
        open: close - 0.05,
        high: close + 0.05,
        low: close - 0.05,
        close,
        volume: 1000,
        sessionId: 'reference_session'
      } as CandleData);
    }

    // Calculate expected indicators manually for validation
    const expectedIndicators = {
      sma20: this.calculateSMA(closes, 20),
      rsi14: this.calculateRSI(closes, 14),
      macd: this.calculateMACD(closes, 12, 26, 9)
    };

    return { candles, expectedIndicators };
  }

  private generatePriceMovement(condition: MarketCondition, progress: number): number {
    const baseMovement = 0.001; // 0.1% base movement
    
    switch (condition.type) {
      case 'trending':
        return baseMovement * condition.strength * (condition.strength > 0 ? 1 : -1);
        
      case 'ranging':
        const cycle = Math.sin(progress * 0.1) * baseMovement * condition.strength;
        return cycle;
        
      case 'volatile':
        return (Math.random() - 0.5) * baseMovement * condition.strength * 4;
        
      case 'breakout':
        if (progress < condition.duration * 0.8) {
          return (Math.random() - 0.5) * baseMovement * 0.2; // Low volatility before breakout
        } else {
          return baseMovement * condition.strength * (Math.random() > 0.5 ? 1 : -1); // Sharp move
        }
        
      default:
        return (Math.random() - 0.5) * baseMovement;
    }
  }

  private generateSpecificPattern(pattern: string, price: number, index: number): Partial<CandleData> {
    const baseSpread = price * 0.0005;
    
    switch (pattern) {
      case 'bullish_engulfing':
        return {
          open: price - baseSpread,
          close: price + baseSpread * 3,
          high: price + baseSpread * 3.2,
          low: price - baseSpread * 1.2,
          volume: 1500
        };
        
      case 'doji':
        return {
          open: price,
          close: price + baseSpread * 0.1,
          high: price + baseSpread * 2,
          low: price - baseSpread * 2,
          volume: 800
        };
        
      case 'hammer':
        return {
          open: price - baseSpread * 0.5,
          close: price,
          high: price + baseSpread * 0.5,
          low: price - baseSpread * 4,
          volume: 1200
        };
        
      case 'shooting_star':
        return {
          open: price + baseSpread * 0.5,
          close: price,
          high: price + baseSpread * 4,
          low: price - baseSpread * 0.5,
          volume: 1200
        };
        
      default:
        return {
          open: price,
          close: price,
          high: price + baseSpread,
          low: price - baseSpread,
          volume: 1000
        };
    }
  }

  // Simple indicator calculations for reference validation
  private calculateSMA(values: number[], period: number): number[] {
    const sma: number[] = [];
    for (let i = 0; i < values.length; i++) {
      if (i < period - 1) {
        sma.push(NaN);
      } else {
        const sum = values.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
        sma.push(sum / period);
      }
    }
    return sma;
  }

  private calculateRSI(values: number[], period: number): number[] {
    const rsi: number[] = [];
    const gains: number[] = [];
    const losses: number[] = [];

    for (let i = 1; i < values.length; i++) {
      const change = values[i] - values[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }

    for (let i = 0; i < values.length; i++) {
      if (i < period) {
        rsi.push(NaN);
      } else {
        const avgGain = gains.slice(i - period, i).reduce((a, b) => a + b, 0) / period;
        const avgLoss = losses.slice(i - period, i).reduce((a, b) => a + b, 0) / period;
        const rs = avgGain / avgLoss;
        rsi.push(100 - (100 / (1 + rs)));
      }
    }

    return rsi;
  }

  private calculateMACD(values: number[], fastPeriod: number, slowPeriod: number, signalPeriod: number) {
    const emaFast = this.calculateEMA(values, fastPeriod);
    const emaSlow = this.calculateEMA(values, slowPeriod);
    const line = emaFast.map((fast, i) => fast - emaSlow[i]);
    const signal = this.calculateEMA(line.filter(v => !isNaN(v)), signalPeriod);
    const histogram = line.map((l, i) => l - (signal[i] || 0));

    return { line, signal, histogram };
  }

  private calculateEMA(values: number[], period: number): number[] {
    const ema: number[] = [];
    const multiplier = 2 / (period + 1);

    for (let i = 0; i < values.length; i++) {
      if (i === 0) {
        ema.push(values[i]);
      } else {
        ema.push((values[i] - ema[i - 1]) * multiplier + ema[i - 1]);
      }
    }

    return ema;
  }
}

export const testDataGenerator = TestDataGenerator.getInstance();