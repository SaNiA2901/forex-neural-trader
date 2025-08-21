import { CandleData } from '@/types/session';

export interface TechnicalFeatures {
  // Базовые индикаторы
  sma: number[];
  ema: number[];
  rsi: number;
  macd: number;
  macdSignal: number;
  bollingerUpper: number;
  bollingerLower: number;
  bollingerPosition: number;
  
  // Моментум индикаторы
  stochastic: number;
  williamsR: number;
  cciIndicator: number;
  
  // Волатильность
  atr: number;
  volatility: number;
  
  // Объемные индикаторы
  volumeRatio: number;
  obv: number;
  
  // Паттерны
  doji: boolean;
  hammer: boolean;
  engulfing: boolean;
  
  // Уровни поддержки/сопротивления
  supportLevel: number;
  resistanceLevel: number;
  
  // Рыночная структура
  trendDirection: number; // -1, 0, 1
  trendStrength: number;
  marketRegime: 'trending' | 'ranging' | 'volatile';
}

export class FeatureEngineeringService {
  /**
   * Создает полный набор технических признаков для свечи
   */
  extractFeatures(candles: CandleData[], currentIndex: number, lookbackPeriod: number = 50): TechnicalFeatures {
    const startIndex = Math.max(0, currentIndex - lookbackPeriod);
    const historicalCandles = candles.slice(startIndex, currentIndex + 1);
    
    if (historicalCandles.length < 10) {
      return this.getDefaultFeatures();
    }

    return {
      // Базовые индикаторы
      sma: this.calculateSMAArray(historicalCandles, [5, 10, 20, 50]),
      ema: this.calculateEMAArray(historicalCandles, [5, 10, 20, 50]),
      rsi: this.calculateRSI(historicalCandles, 14),
      macd: this.calculateMACD(historicalCandles).macd,
      macdSignal: this.calculateMACD(historicalCandles).signal,
      bollingerUpper: this.calculateBollingerBands(historicalCandles).upper,
      bollingerLower: this.calculateBollingerBands(historicalCandles).lower,
      bollingerPosition: this.calculateBollingerBands(historicalCandles).position,
      
      // Моментум индикаторы
      stochastic: this.calculateStochastic(historicalCandles),
      williamsR: this.calculateWilliamsR(historicalCandles),
      cciIndicator: this.calculateCCI(historicalCandles),
      
      // Волатильность
      atr: this.calculateATR(historicalCandles),
      volatility: this.calculateVolatility(historicalCandles),
      
      // Объемные индикаторы
      volumeRatio: this.calculateVolumeRatio(historicalCandles),
      obv: this.calculateOBV(historicalCandles),
      
      // Паттерны
      doji: this.isDoji(historicalCandles[historicalCandles.length - 1]),
      hammer: this.isHammer(historicalCandles[historicalCandles.length - 1]),
      engulfing: this.isEngulfing(historicalCandles),
      
      // Уровни поддержки/сопротивления
      supportLevel: this.findSupportLevel(historicalCandles),
      resistanceLevel: this.findResistanceLevel(historicalCandles),
      
      // Рыночная структура
      trendDirection: this.determineTrendDirection(historicalCandles),
      trendStrength: this.calculateTrendStrength(historicalCandles),
      marketRegime: this.classifyMarketRegime(historicalCandles)
    };
  }

  private getDefaultFeatures(): TechnicalFeatures {
    return {
      sma: [0, 0, 0, 0],
      ema: [0, 0, 0, 0],
      rsi: 50,
      macd: 0,
      macdSignal: 0,
      bollingerUpper: 0,
      bollingerLower: 0,
      bollingerPosition: 0.5,
      stochastic: 50,
      williamsR: -50,
      cciIndicator: 0,
      atr: 0,
      volatility: 0,
      volumeRatio: 1,
      obv: 0,
      doji: false,
      hammer: false,
      engulfing: false,
      supportLevel: 0,
      resistanceLevel: 0,
      trendDirection: 0,
      trendStrength: 0,
      marketRegime: 'ranging' as const
    };
  }

  private calculateSMAArray(candles: CandleData[], periods: number[]): number[] {
    return periods.map(period => {
      if (candles.length < period) return 0;
      
      const sum = candles.slice(-period).reduce((acc, candle) => acc + candle.close, 0);
      return sum / period;
    });
  }

  private calculateEMAArray(candles: CandleData[], periods: number[]): number[] {
    return periods.map(period => {
      if (candles.length < period) return 0;
      
      const multiplier = 2 / (period + 1);
      let ema = candles[0].close;
      
      for (let i = 1; i < candles.length; i++) {
        ema = (candles[i].close - ema) * multiplier + ema;
      }
      
      return ema;
    });
  }

  private calculateRSI(candles: CandleData[], period: number = 14): number {
    if (candles.length < period + 1) return 50;

    let gains = 0;
    let losses = 0;

    for (let i = candles.length - period; i < candles.length; i++) {
      const change = candles[i].close - candles[i - 1].close;
      if (change > 0) {
        gains += change;
      } else {
        losses -= change;
      }
    }

    const avgGain = gains / period;
    const avgLoss = losses / period;
    
    if (avgLoss === 0) return 100;
    
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  private calculateMACD(candles: CandleData[]): { macd: number; signal: number } {
    if (candles.length < 26) return { macd: 0, signal: 0 };

    const ema12 = this.calculateEMAArray(candles, [12])[0];
    const ema26 = this.calculateEMAArray(candles, [26])[0];
    const macd = ema12 - ema26;
    
    // Упрощенный расчет сигнальной линии
    const signal = macd * 0.9; // Заглушка
    
    return { macd, signal };
  }

  private calculateBollingerBands(candles: CandleData[], period: number = 20): { upper: number; lower: number; position: number } {
    if (candles.length < period) return { upper: 0, lower: 0, position: 0.5 };

    const recentCandles = candles.slice(-period);
    const closes = recentCandles.map(c => c.close);
    const sma = closes.reduce((sum, close) => sum + close, 0) / period;
    
    const variance = closes.reduce((sum, close) => sum + Math.pow(close - sma, 2), 0) / period;
    const stdDev = Math.sqrt(variance);
    
    const upper = sma + (2 * stdDev);
    const lower = sma - (2 * stdDev);
    const currentPrice = candles[candles.length - 1].close;
    
    let position = 0.5;
    if (upper !== lower) {
      position = (currentPrice - lower) / (upper - lower);
    }
    
    return { upper, lower, position };
  }

  private calculateStochastic(candles: CandleData[], period: number = 14): number {
    if (candles.length < period) return 50;

    const recentCandles = candles.slice(-period);
    const highs = recentCandles.map(c => c.high);
    const lows = recentCandles.map(c => c.low);
    
    const highestHigh = Math.max(...highs);
    const lowestLow = Math.min(...lows);
    const currentClose = candles[candles.length - 1].close;
    
    if (highestHigh === lowestLow) return 50;
    
    return ((currentClose - lowestLow) / (highestHigh - lowestLow)) * 100;
  }

  private calculateWilliamsR(candles: CandleData[], period: number = 14): number {
    const stochastic = this.calculateStochastic(candles, period);
    return stochastic - 100;
  }

  private calculateCCI(candles: CandleData[], period: number = 20): number {
    if (candles.length < period) return 0;

    const recentCandles = candles.slice(-period);
    const typicalPrices = recentCandles.map(c => (c.high + c.low + c.close) / 3);
    const sma = typicalPrices.reduce((sum, tp) => sum + tp, 0) / period;
    
    const meanDeviation = typicalPrices.reduce((sum, tp) => sum + Math.abs(tp - sma), 0) / period;
    const currentTypicalPrice = typicalPrices[typicalPrices.length - 1];
    
    if (meanDeviation === 0) return 0;
    
    return (currentTypicalPrice - sma) / (0.015 * meanDeviation);
  }

  private calculateATR(candles: CandleData[], period: number = 14): number {
    if (candles.length < period + 1) return 0;

    const trueRanges = [];
    
    for (let i = 1; i < candles.length; i++) {
      const current = candles[i];
      const previous = candles[i - 1];
      
      const tr = Math.max(
        current.high - current.low,
        Math.abs(current.high - previous.close),
        Math.abs(current.low - previous.close)
      );
      
      trueRanges.push(tr);
    }
    
    const recentTR = trueRanges.slice(-period);
    return recentTR.reduce((sum, tr) => sum + tr, 0) / period;
  }

  private calculateVolatility(candles: CandleData[], period: number = 20): number {
    if (candles.length < period) return 0;

    const returns = [];
    for (let i = 1; i < candles.length; i++) {
      const ret = Math.log(candles[i].close / candles[i - 1].close);
      returns.push(ret);
    }
    
    const recentReturns = returns.slice(-period);
    const mean = recentReturns.reduce((sum, ret) => sum + ret, 0) / period;
    const variance = recentReturns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / period;
    
    return Math.sqrt(variance) * Math.sqrt(252); // Аннуализированная волатильность
  }

  private calculateVolumeRatio(candles: CandleData[], period: number = 10): number {
    if (candles.length < period) return 1;

    const recentVolumes = candles.slice(-period).map(c => c.volume);
    const currentVolume = candles[candles.length - 1].volume;
    const avgVolume = recentVolumes.reduce((sum, vol) => sum + vol, 0) / period;
    
    return avgVolume === 0 ? 1 : currentVolume / avgVolume;
  }

  private calculateOBV(candles: CandleData[]): number {
    let obv = 0;
    
    for (let i = 1; i < candles.length; i++) {
      if (candles[i].close > candles[i - 1].close) {
        obv += candles[i].volume;
      } else if (candles[i].close < candles[i - 1].close) {
        obv -= candles[i].volume;
      }
    }
    
    return obv;
  }

  private isDoji(candle: CandleData): boolean {
    const bodySize = Math.abs(candle.close - candle.open);
    const totalRange = candle.high - candle.low;
    
    return totalRange > 0 && (bodySize / totalRange) < 0.1;
  }

  private isHammer(candle: CandleData): boolean {
    const bodySize = Math.abs(candle.close - candle.open);
    const lowerShadow = Math.min(candle.open, candle.close) - candle.low;
    const upperShadow = candle.high - Math.max(candle.open, candle.close);
    
    return lowerShadow > bodySize * 2 && upperShadow < bodySize * 0.5;
  }

  private isEngulfing(candles: CandleData[]): boolean {
    if (candles.length < 2) return false;
    
    const prev = candles[candles.length - 2];
    const current = candles[candles.length - 1];
    
    const prevBodySize = Math.abs(prev.close - prev.open);
    const currentBodySize = Math.abs(current.close - current.open);
    
    return currentBodySize > prevBodySize * 1.5 &&
           ((prev.close < prev.open && current.close > current.open) ||
            (prev.close > prev.open && current.close < current.open));
  }

  private findSupportLevel(candles: CandleData[]): number {
    const lows = candles.map(c => c.low);
    const sortedLows = [...lows].sort((a, b) => a - b);
    
    // Упрощенный поиск уровня поддержки
    return sortedLows[Math.floor(sortedLows.length * 0.2)];
  }

  private findResistanceLevel(candles: CandleData[]): number {
    const highs = candles.map(c => c.high);
    const sortedHighs = [...highs].sort((a, b) => b - a);
    
    // Упрощенный поиск уровня сопротивления
    return sortedHighs[Math.floor(sortedHighs.length * 0.2)];
  }

  private determineTrendDirection(candles: CandleData[]): number {
    if (candles.length < 20) return 0;
    
    const sma20 = this.calculateSMAArray(candles, [20])[0];
    const sma50 = this.calculateSMAArray(candles, [50])[0];
    const currentPrice = candles[candles.length - 1].close;
    
    if (currentPrice > sma20 && sma20 > sma50) return 1; // Восходящий тренд
    if (currentPrice < sma20 && sma20 < sma50) return -1; // Нисходящий тренд
    return 0; // Боковик
  }

  private calculateTrendStrength(candles: CandleData[]): number {
    if (candles.length < 20) return 0;
    
    const closes = candles.slice(-20).map(c => c.close);
    const firstPrice = closes[0];
    const lastPrice = closes[closes.length - 1];
    
    const totalChange = Math.abs(lastPrice - firstPrice);
    const maxPrice = Math.max(...closes);
    const minPrice = Math.min(...closes);
    const range = maxPrice - minPrice;
    
    return range === 0 ? 0 : totalChange / range;
  }

  private classifyMarketRegime(candles: CandleData[]): 'trending' | 'ranging' | 'volatile' {
    const volatility = this.calculateVolatility(candles);
    const trendStrength = this.calculateTrendStrength(candles);
    
    if (volatility > 0.3) return 'volatile';
    if (trendStrength > 0.6) return 'trending';
    return 'ranging';
  }
}

export const featureEngineeringService = new FeatureEngineeringService();