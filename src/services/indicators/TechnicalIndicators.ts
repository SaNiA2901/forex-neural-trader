import { CandleData } from '@/types/session';
import { indicatorFactory } from './indicator-factory';
import { MarketDataPoint } from './core/types';

export interface TechnicalIndicators {
  rsi: number;
  macd: { line: number; signal: number; histogram: number };
  bollingerBands: { upper: number; middle: number; lower: number };
  ema: { ema12: number; ema26: number };
  stochastic: { k: number; d: number };
  atr: number;
  adx: number;
}

// Legacy compatibility - converts CandleData to MarketDataPoint
function convertToMarketData(candles: CandleData[]): MarketDataPoint[] {
  return candles.map(candle => ({
    timestamp: new Date(candle.timestamp).getTime(),
    open: candle.open,
    high: candle.high,
    low: candle.low,
    close: candle.close,
    volume: candle.volume
  }));
}

export class TechnicalIndicatorService {
  static async calculateAll(candles: CandleData[], currentIndex: number): Promise<TechnicalIndicators> {
    const lookback = Math.min(50, currentIndex + 1);
    const recentCandles = candles.slice(Math.max(0, currentIndex - lookback + 1), currentIndex + 1);
    const marketData = convertToMarketData(recentCandles);
    
    try {
      // Use new modular indicators for core calculations
      const rsiIndicator = indicatorFactory.createRSI({ period: 14 });
      const macdIndicator = indicatorFactory.createMACD({ fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 });
      const bbIndicator = indicatorFactory.createBollingerBands({ period: 20 });
      const stochIndicator = indicatorFactory.createStochastic({ period: 14 });

      const [rsiResult, macdResult, bbResult, stochResult] = await Promise.all([
        rsiIndicator.calculate(marketData),
        macdIndicator.calculate(marketData),
        bbIndicator.calculate(marketData),
        stochIndicator.calculate(marketData)
      ]);

      // Get latest values or fallback to legacy calculations
      const rsi = rsiResult.values.length > 0 ? 
        rsiResult.values[rsiResult.values.length - 1] : 
        this.calculateRSI(recentCandles);

      const macdLatest = macdResult.values.length > 0 ? 
        macdResult.values[macdResult.values.length - 1] :
        this.calculateMACD(recentCandles);

      const bbLatest = bbResult.values.length > 0 ?
        bbResult.values[bbResult.values.length - 1] :
        this.calculateBollingerBands(recentCandles);

      const stochLatest = stochResult.values.length > 0 ?
        stochResult.values[stochResult.values.length - 1] :
        this.calculateStochastic(recentCandles);

      return {
        rsi: typeof rsi === 'number' ? rsi : 50,
        macd: {
          line: typeof macdLatest === 'object' && 'macd' in macdLatest ? macdLatest.macd : (typeof macdLatest === 'object' && 'line' in macdLatest ? macdLatest.line : 0),
          signal: typeof macdLatest === 'object' && 'signal' in macdLatest ? macdLatest.signal : 0,
          histogram: typeof macdLatest === 'object' && 'histogram' in macdLatest ? macdLatest.histogram : 0
        },
        bollingerBands: {
          upper: bbLatest.upper || 0,
          middle: bbLatest.middle || 0,
          lower: bbLatest.lower || 0
        },
        ema: this.calculateEMA(recentCandles),
        stochastic: {
          k: stochLatest.k || 50,
          d: stochLatest.d || 50
        },
        atr: this.calculateATR(recentCandles),
        adx: this.calculateADX(recentCandles)
      };
    } catch (error) {
      console.warn('New indicators failed, falling back to legacy:', error);
      // Fallback to legacy calculations
      return {
        rsi: this.calculateRSI(recentCandles),
        macd: this.calculateMACD(recentCandles),
        bollingerBands: this.calculateBollingerBands(recentCandles),
        ema: this.calculateEMA(recentCandles),
        stochastic: this.calculateStochastic(recentCandles),
        atr: this.calculateATR(recentCandles),
        adx: this.calculateADX(recentCandles)
      };
    }
  }

  static calculateRSI(candles: CandleData[], period: number = 14): number {
    if (candles.length < period + 1) return 50;
    
    const closes = candles.map(c => c.close);
    let avgGain = 0;
    let avgLoss = 0;

    // Первый расчет на period значений
    for (let i = 1; i <= period; i++) {
      const change = closes[i] - closes[i - 1];
      if (change > 0) {
        avgGain += change;
      } else {
        avgLoss -= change;
      }
    }

    avgGain /= period;
    avgLoss /= period;

    // Сглаженный RSI для оставшихся значений (Wilder's smoothing)
    for (let i = period + 1; i < closes.length; i++) {
      const change = closes[i] - closes[i - 1];
      if (change > 0) {
        avgGain = (avgGain * (period - 1) + change) / period;
        avgLoss = (avgLoss * (period - 1)) / period;
      } else {
        avgGain = (avgGain * (period - 1)) / period;
        avgLoss = (avgLoss * (period - 1) - change) / period;
      }
    }

    if (avgLoss === 0) return 100;
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  static calculateMACD(candles: CandleData[], fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9): { line: number; signal: number; histogram: number } {
    const closes = candles.map(c => c.close);
    const ema12 = this.calculateEMAArray(closes, fastPeriod);
    const ema26 = this.calculateEMAArray(closes, slowPeriod);
    
    if (ema12.length === 0 || ema26.length === 0) {
      return { line: 0, signal: 0, histogram: 0 };
    }

    // MACD Line
    const macdLine = ema12[ema12.length - 1] - ema26[ema26.length - 1];
    
    // Создаем историю MACD для расчета сигнальной линии
    const macdHistory: number[] = [];
    const minLength = Math.min(ema12.length, ema26.length);
    for (let i = 0; i < minLength; i++) {
      macdHistory.push(ema12[i] - ema26[i]);
    }
    
    // Signal Line (EMA от MACD)
    const signalEMA = this.calculateEMAArray(macdHistory, signalPeriod);
    const signalLine = signalEMA.length > 0 ? signalEMA[signalEMA.length - 1] : 0;
    
    // Histogram
    const histogram = macdLine - signalLine;
    
    return {
      line: macdLine,
      signal: signalLine,
      histogram
    };
  }

  static calculateBollingerBands(candles: CandleData[], period: number = 20, stdDev: number = 2): 
    { upper: number; middle: number; lower: number } {
    const closes = candles.map(c => c.close);
    const sma = closes.reduce((sum, price) => sum + price, 0) / closes.length;
    
    const variance = closes.reduce((sum, price) => sum + Math.pow(price - sma, 2), 0) / closes.length;
    const standardDeviation = Math.sqrt(variance);
    
    return {
      upper: sma + (standardDeviation * stdDev),
      middle: sma,
      lower: sma - (standardDeviation * stdDev)
    };
  }

  static calculateEMA(candles: CandleData[]): { ema12: number; ema26: number } {
    return {
      ema12: this.calculateSimpleEMA(candles, 12),
      ema26: this.calculateSimpleEMA(candles, 26)
    };
  }

  private static calculateSimpleEMA(candles: CandleData[], period: number): number {
    if (candles.length === 0) return 0;
    
    const multiplier = 2 / (period + 1);
    let ema = candles[0].close;
    
    for (let i = 1; i < candles.length; i++) {
      ema = (candles[i].close * multiplier) + (ema * (1 - multiplier));
    }
    
    return ema;
  }

  private static calculateEMAArray(prices: number[], period: number): number[] {
    if (prices.length === 0) return [];
    
    const ema: number[] = [];
    const multiplier = 2 / (period + 1);
    
    ema[0] = prices[0];
    for (let i = 1; i < prices.length; i++) {
      ema[i] = (prices[i] * multiplier) + (ema[i - 1] * (1 - multiplier));
    }
    
    return ema;
  }

  static calculateStochastic(candles: CandleData[], period: number = 14, kSmoothing: number = 3): { k: number; d: number } {
    if (candles.length < period) return { k: 50, d: 50 };
    
    const kValues: number[] = [];
    
    // Рассчитываем %K для каждого периода
    for (let i = period - 1; i < candles.length; i++) {
      const recentCandles = candles.slice(i - period + 1, i + 1);
      const currentClose = candles[i].close;
      const highestHigh = Math.max(...recentCandles.map(c => c.high));
      const lowestLow = Math.min(...recentCandles.map(c => c.low));
      
      if (highestHigh === lowestLow) {
        kValues.push(50); // Избегаем деления на ноль
      } else {
        const k = ((currentClose - lowestLow) / (highestHigh - lowestLow)) * 100;
        kValues.push(k);
      }
    }
    
    // %K (последнее значение)
    const currentK = kValues[kValues.length - 1];
    
    // %D (SMA от %K)
    const dPeriod = Math.min(kSmoothing, kValues.length);
    const recentK = kValues.slice(-dPeriod);
    const d = recentK.reduce((sum, val) => sum + val, 0) / recentK.length;
    
    return { k: currentK, d };
  }

  static calculateATR(candles: CandleData[], period: number = 14): number {
    if (candles.length < 2) return 0;
    
    let trueRanges = [];
    
    for (let i = 1; i < candles.length; i++) {
      const high = candles[i].high;
      const low = candles[i].low;
      const prevClose = candles[i - 1].close;
      
      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );
      
      trueRanges.push(tr);
    }
    
    return trueRanges.reduce((sum, tr) => sum + tr, 0) / trueRanges.length;
  }

  private static calculateADX(candles: CandleData[], period: number = 14): number {
    if (candles.length < period + 1) return 25;
    
    const dmPlus: number[] = [];
    const dmMinus: number[] = [];
    const tr: number[] = [];

    // Расчет DM+ DM- и TR
    for (let i = 1; i < candles.length; i++) {
      const high = candles[i].high;
      const low = candles[i].low;
      const prevHigh = candles[i - 1].high;
      const prevLow = candles[i - 1].low;
      const prevClose = candles[i - 1].close;

      const upMove = high - prevHigh;
      const downMove = prevLow - low;

      dmPlus.push(upMove > downMove && upMove > 0 ? upMove : 0);
      dmMinus.push(downMove > upMove && downMove > 0 ? downMove : 0);

      const trueRange = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );
      tr.push(trueRange);
    }

    // Сглаженные значения DM и TR
    const smoothDMPlus = this.smoothArray(dmPlus, period);
    const smoothDMMinus = this.smoothArray(dmMinus, period);
    const smoothTR = this.smoothArray(tr, period);

    if (smoothTR === 0) return 25;

    // DI+ и DI-
    const diPlus = (smoothDMPlus / smoothTR) * 100;
    const diMinus = (smoothDMMinus / smoothTR) * 100;

    // DX
    const dx = Math.abs(diPlus - diMinus) / (diPlus + diMinus) * 100;
    
    return isNaN(dx) ? 25 : Math.min(100, Math.max(0, dx));
  }

  private static smoothArray(values: number[], period: number): number {
    if (values.length === 0) return 0;
    if (values.length < period) return values.reduce((sum, val) => sum + val, 0) / values.length;
    
    // Wilder's smoothing
    let sum = values.slice(0, period).reduce((s, v) => s + v, 0);
    let smoothed = sum / period;
    
    for (let i = period; i < values.length; i++) {
      smoothed = (smoothed * (period - 1) + values[i]) / period;
    }
    
    return smoothed;
  }
}