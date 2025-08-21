import { CandleData } from '@/types/session';

export interface PatternSignals {
  candlestickPattern: string | null;
  strength: number;
  isReversal: boolean;
  isContinuation: boolean;
}

export interface VolumeAnalysis {
  volumeTrend: 'increasing' | 'decreasing' | 'stable';
  volumeOscillator: number;
  onBalanceVolume: number;
  volumeWeightedPrice: number;
}

export class PatternAnalysisService {
  static analyzePatterns(candles: CandleData[], currentIndex: number): PatternSignals {
    const current = candles[currentIndex];
    if (!current) return { candlestickPattern: null, strength: 0, isReversal: false, isContinuation: false };
    
    // Адаптивные пороги на основе волатильности
    const volatility = this.calculateVolatility(candles, currentIndex);
    const adaptiveThreshold = Math.max(0.05, Math.min(0.2, volatility * 2));
    
    const bodySize = Math.abs(current.close - current.open);
    const candleRange = current.high - current.low;
    const upperShadow = current.high - Math.max(current.open, current.close);
    const lowerShadow = Math.min(current.open, current.close) - current.low;
    
    // Избегаем false positives на малых свечах
    if (candleRange < this.getAverageRange(candles, currentIndex) * 0.3) {
      return { candlestickPattern: null, strength: 0, isReversal: false, isContinuation: false };
    }
    
    // Multi-candle patterns
    if (currentIndex >= 2) {
      const multiPattern = this.detectMultiCandlePatterns(candles, currentIndex);
      if (multiPattern.candlestickPattern) return multiPattern;
    }
    
    // Single candle patterns with adaptive thresholds
    // Doji pattern
    if (bodySize < candleRange * adaptiveThreshold) {
      const strength = this.calculatePatternStrength(current, candles, currentIndex, 'doji');
      return {
        candlestickPattern: 'Doji',
        strength,
        isReversal: true,
        isContinuation: false
      };
    }
    
    // Hammer pattern
    if (lowerShadow > bodySize * 2 && upperShadow < bodySize * 0.5 && this.isAtBottom(candles, currentIndex)) {
      const strength = this.calculatePatternStrength(current, candles, currentIndex, 'hammer');
      return {
        candlestickPattern: 'Hammer',
        strength,
        isReversal: true,
        isContinuation: false
      };
    }
    
    // Shooting Star
    if (upperShadow > bodySize * 2 && lowerShadow < bodySize * 0.5 && this.isAtTop(candles, currentIndex)) {
      const strength = this.calculatePatternStrength(current, candles, currentIndex, 'shooting_star');
      return {
        candlestickPattern: 'Shooting Star',
        strength,
        isReversal: true,
        isContinuation: false
      };
    }

    // Engulfing patterns
    if (currentIndex >= 1) {
      const prev = candles[currentIndex - 1];
      if (this.isBullishEngulfing(prev, current)) {
        return {
          candlestickPattern: 'Bullish Engulfing',
          strength: this.calculatePatternStrength(current, candles, currentIndex, 'bullish_engulfing'),
          isReversal: true,
          isContinuation: false
        };
      }
      
      if (this.isBearishEngulfing(prev, current)) {
        return {
          candlestickPattern: 'Bearish Engulfing',
          strength: this.calculatePatternStrength(current, candles, currentIndex, 'bearish_engulfing'),
          isReversal: true,
          isContinuation: false
        };
      }
    }
    
    return { candlestickPattern: null, strength: 0, isReversal: false, isContinuation: false };
  }

  static analyzeVolume(candles: CandleData[], currentIndex: number): VolumeAnalysis {
    const lookback = Math.min(20, currentIndex + 1);
    const recentCandles = candles.slice(Math.max(0, currentIndex - lookback + 1), currentIndex + 1);
    
    if (recentCandles.length < 3) {
      return {
        volumeTrend: 'stable',
        volumeOscillator: 0,
        onBalanceVolume: 0,
        volumeWeightedPrice: recentCandles[0]?.close || 0
      };
    }
    
    // Улучшенный анализ тренда объема
    const periods = [3, 5, 10];
    let trendScore = 0;
    
    periods.forEach(period => {
      if (recentCandles.length >= period * 2) {
        const earlyPeriod = recentCandles.slice(0, period);
        const latePeriod = recentCandles.slice(-period);
        
        const earlyAvg = earlyPeriod.reduce((sum, c) => sum + c.volume, 0) / period;
        const lateAvg = latePeriod.reduce((sum, c) => sum + c.volume, 0) / period;
        
        if (lateAvg > earlyAvg * 1.2) trendScore += 1;
        else if (lateAvg < earlyAvg * 0.8) trendScore -= 1;
      }
    });
    
    let volumeTrend: 'increasing' | 'decreasing' | 'stable';
    if (trendScore >= 2) volumeTrend = 'increasing';
    else if (trendScore <= -2) volumeTrend = 'decreasing';
    else volumeTrend = 'stable';
    
    // Расчет On-Balance Volume (OBV)
    let obv = 0;
    for (let i = 1; i < recentCandles.length; i++) {
      if (recentCandles[i].close > recentCandles[i - 1].close) {
        obv += recentCandles[i].volume;
      } else if (recentCandles[i].close < recentCandles[i - 1].close) {
        obv -= recentCandles[i].volume;
      }
    }
    
    // VWAP (Volume Weighted Average Price)
    const totalVolume = recentCandles.reduce((sum, c) => sum + c.volume, 0);
    const vwap = totalVolume > 0 ? 
      recentCandles.reduce((sum, c) => sum + ((c.high + c.low + c.close) / 3 * c.volume), 0) / totalVolume :
      recentCandles[recentCandles.length - 1].close;
    
    // Volume Oscillator
    const shortVolumeAvg = recentCandles.slice(-5).reduce((sum, c) => sum + c.volume, 0) / 5;
    const longVolumeAvg = recentCandles.slice(-10).reduce((sum, c) => sum + c.volume, 0) / 10;
    const volumeOscillator = longVolumeAvg > 0 ? ((shortVolumeAvg - longVolumeAvg) / longVolumeAvg) * 100 : 0;
    
    return {
      volumeTrend,
      volumeOscillator,
      onBalanceVolume: obv,
      volumeWeightedPrice: vwap
    };
  }

  // Вспомогательные методы для улучшенного анализа паттернов
  private static calculateVolatility(candles: CandleData[], currentIndex: number): number {
    const lookback = Math.min(10, currentIndex + 1);
    const recentCandles = candles.slice(Math.max(0, currentIndex - lookback + 1), currentIndex + 1);
    
    if (recentCandles.length < 2) return 0.02;
    
    const returns = [];
    for (let i = 1; i < recentCandles.length; i++) {
      returns.push((recentCandles[i].close - recentCandles[i - 1].close) / recentCandles[i - 1].close);
    }
    
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    return Math.sqrt(variance);
  }

  private static getAverageRange(candles: CandleData[], currentIndex: number): number {
    const lookback = Math.min(20, currentIndex + 1);
    const recentCandles = candles.slice(Math.max(0, currentIndex - lookback + 1), currentIndex + 1);
    
    const avgRange = recentCandles.reduce((sum, c) => sum + (c.high - c.low), 0) / recentCandles.length;
    return avgRange;
  }

  private static isAtBottom(candles: CandleData[], currentIndex: number): boolean {
    const lookback = Math.min(5, currentIndex + 1);
    const recentCandles = candles.slice(Math.max(0, currentIndex - lookback + 1), currentIndex + 1);
    const current = candles[currentIndex];
    
    const lowestLow = Math.min(...recentCandles.map(c => c.low));
    return current.low <= lowestLow * 1.01; // В пределах 1% от минимума
  }

  private static isAtTop(candles: CandleData[], currentIndex: number): boolean {
    const lookback = Math.min(5, currentIndex + 1);
    const recentCandles = candles.slice(Math.max(0, currentIndex - lookback + 1), currentIndex + 1);
    const current = candles[currentIndex];
    
    const highestHigh = Math.max(...recentCandles.map(c => c.high));
    return current.high >= highestHigh * 0.99; // В пределах 1% от максимума
  }

  private static isBullishEngulfing(prev: CandleData, current: CandleData): boolean {
    return prev.close < prev.open && // Предыдущая медвежья
           current.close > current.open && // Текущая бычья
           current.open < prev.close && // Открытие ниже закрытия предыдущей
           current.close > prev.open && // Закрытие выше открытия предыдущей
           (current.close - current.open) > (prev.open - prev.close) * 1.1; // Тело больше на 10%
  }

  private static isBearishEngulfing(prev: CandleData, current: CandleData): boolean {
    return prev.close > prev.open && // Предыдущая бычья
           current.close < current.open && // Текущая медвежья
           current.open > prev.close && // Открытие выше закрытия предыдущей
           current.close < prev.open && // Закрытие ниже открытия предыдущей
           (current.open - current.close) > (prev.close - prev.open) * 1.1; // Тело больше на 10%
  }

  private static detectMultiCandlePatterns(candles: CandleData[], currentIndex: number): PatternSignals {
    const current = candles[currentIndex];
    const prev = candles[currentIndex - 1];
    const prev2 = candles[currentIndex - 2];

    // Three White Soldiers
    if (current.close > current.open &&
        prev.close > prev.open &&
        prev2.close > prev2.open &&
        current.close > prev.close &&
        prev.close > prev2.close &&
        current.high > prev.high &&
        prev.high > prev2.high) {
      return {
        candlestickPattern: 'Three White Soldiers',
        strength: 0.85,
        isReversal: false,
        isContinuation: true
      };
    }

    // Three Black Crows
    if (current.close < current.open &&
        prev.close < prev.open &&
        prev2.close < prev2.open &&
        current.close < prev.close &&
        prev.close < prev2.close &&
        current.low < prev.low &&
        prev.low < prev2.low) {
      return {
        candlestickPattern: 'Three Black Crows',
        strength: 0.85,
        isReversal: false,
        isContinuation: true
      };
    }

    return { candlestickPattern: null, strength: 0, isReversal: false, isContinuation: false };
  }

  private static calculatePatternStrength(
    current: CandleData, 
    candles: CandleData[], 
    currentIndex: number, 
    patternType: string
  ): number {
    let baseStrength = 0.7;
    
    // Учитываем объем
    const avgVolume = this.calculateAverageVolume(candles, currentIndex);
    if (current.volume > avgVolume * 1.5) baseStrength += 0.1;
    
    // Учитываем положение в тренде
    const trendStrength = this.calculateTrendContext(candles, currentIndex);
    baseStrength += trendStrength * 0.1;
    
    // Учитываем размер свечи
    const avgRange = this.getAverageRange(candles, currentIndex);
    const currentRange = current.high - current.low;
    if (currentRange > avgRange * 1.2) baseStrength += 0.05;
    
    return Math.min(0.95, Math.max(0.5, baseStrength));
  }

  private static calculateAverageVolume(candles: CandleData[], currentIndex: number): number {
    const lookback = Math.min(10, currentIndex + 1);
    const recentCandles = candles.slice(Math.max(0, currentIndex - lookback + 1), currentIndex + 1);
    return recentCandles.reduce((sum, c) => sum + c.volume, 0) / recentCandles.length;
  }

  private static calculateTrendContext(candles: CandleData[], currentIndex: number): number {
    const lookback = Math.min(10, currentIndex + 1);
    const recentCandles = candles.slice(Math.max(0, currentIndex - lookback + 1), currentIndex + 1);
    
    if (recentCandles.length < 3) return 0;
    
    const first = recentCandles[0].close;
    const last = recentCandles[recentCandles.length - 1].close;
    const trendDirection = (last - first) / first;
    
    return Math.tanh(trendDirection * 10); // Нормализуем к [-1, 1]
  }
}