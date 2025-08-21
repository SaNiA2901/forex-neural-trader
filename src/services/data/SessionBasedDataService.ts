import { CandleData } from '@/types/session';

interface VolumeAnalysis {
  currentVolume: number;
  averageVolume: number;
  volumeRatio: number;
  volumeTrend: 'INCREASING' | 'DECREASING' | 'STABLE';
  significantVolumeCandles: number[];
}

interface RealTimeMetrics {
  priceChangePercent: number;
  volatilityIndex: number;
  momentumScore: number;
  supportLevel: number;
  resistanceLevel: number;
  trendStrength: number;
}

export class SessionBasedDataService {
  private static instance: SessionBasedDataService;
  private sessionCache: Map<string, CandleData[]> = new Map();
  private metricsCache: Map<string, RealTimeMetrics> = new Map();
  private lastUpdateTimes: Map<string, number> = new Map();

  static getInstance(): SessionBasedDataService {
    if (!SessionBasedDataService.instance) {
      SessionBasedDataService.instance = new SessionBasedDataService();
    }
    return SessionBasedDataService.instance;
  }

  getSessionCandles(sessionId: string): CandleData[] {
    const cached = this.sessionCache.get(sessionId);
    if (cached) {
      return cached;
    }
    
    // В реальном приложении здесь будет запрос к базе данных
    // Пока возвращаем пустой массив
    return [];
  }

  getCurrentPrice(sessionId: string): number {
    const candles = this.getSessionCandles(sessionId);
    if (candles.length === 0) return 0;
    
    return candles[candles.length - 1].close;
  }

  calculateVolatility(sessionId: string): number {
    const candles = this.getSessionCandles(sessionId);
    if (candles.length < 2) return 0;
    
    const returns = [];
    for (let i = 1; i < candles.length; i++) {
      const returnRate = (candles[i].close - candles[i - 1].close) / candles[i - 1].close;
      returns.push(returnRate);
    }
    
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance) * 100; // В процентах
  }

  getTrendDirection(sessionId: string): 'UP' | 'DOWN' | 'SIDEWAYS' {
    const candles = this.getSessionCandles(sessionId);
    if (candles.length < 5) return 'SIDEWAYS';
    
    const recentCandles = candles.slice(-10);
    const firstPrice = recentCandles[0].close;
    const lastPrice = recentCandles[recentCandles.length - 1].close;
    const changePercent = ((lastPrice - firstPrice) / firstPrice) * 100;
    
    if (changePercent > 0.5) return 'UP';
    if (changePercent < -0.5) return 'DOWN';
    return 'SIDEWAYS';
  }

  getVolumeAnalysis(sessionId: string): VolumeAnalysis {
    const candles = this.getSessionCandles(sessionId);
    if (candles.length === 0) {
      return {
        currentVolume: 0,
        averageVolume: 0,
        volumeRatio: 1,
        volumeTrend: 'STABLE',
        significantVolumeCandles: []
      };
    }
    
    const currentVolume = candles[candles.length - 1].volume;
    const averageVolume = candles.reduce((sum, c) => sum + c.volume, 0) / candles.length;
    const volumeRatio = currentVolume / (averageVolume || 1);
    
    // Анализ тренда объема (последние 5 свечей)
    const recentVolumes = candles.slice(-5).map(c => c.volume);
    let increasingCount = 0;
    let decreasingCount = 0;
    
    for (let i = 1; i < recentVolumes.length; i++) {
      if (recentVolumes[i] > recentVolumes[i - 1]) increasingCount++;
      if (recentVolumes[i] < recentVolumes[i - 1]) decreasingCount++;
    }
    
    let volumeTrend: 'INCREASING' | 'DECREASING' | 'STABLE';
    if (increasingCount > decreasingCount) {
      volumeTrend = 'INCREASING';
    } else if (decreasingCount > increasingCount) {
      volumeTrend = 'DECREASING';
    } else {
      volumeTrend = 'STABLE';
    }
    
    // Поиск свечей с значимым объемом (выше среднего в 1.5 раза)
    const significantVolumeCandles = candles
      .map((candle, index) => ({ index, volume: candle.volume }))
      .filter(item => item.volume > averageVolume * 1.5)
      .map(item => item.index);
    
    return {
      currentVolume,
      averageVolume,
      volumeRatio,
      volumeTrend,
      significantVolumeCandles
    };
  }

  getRealTimeMetrics(sessionId: string): RealTimeMetrics {
    const cacheKey = sessionId;
    const lastUpdate = this.lastUpdateTimes.get(cacheKey) || 0;
    const now = Date.now();
    
    // Кэшируем метрики на 30 секунд
    if (now - lastUpdate < 30000 && this.metricsCache.has(cacheKey)) {
      return this.metricsCache.get(cacheKey)!;
    }
    
    const candles = this.getSessionCandles(sessionId);
    if (candles.length < 5) {
      const defaultMetrics: RealTimeMetrics = {
        priceChangePercent: 0,
        volatilityIndex: 0,
        momentumScore: 0,
        supportLevel: 0,
        resistanceLevel: 0,
        trendStrength: 0
      };
      return defaultMetrics;
    }
    
    const metrics = this.calculateRealTimeMetrics(candles);
    
    // Обновляем кэш
    this.metricsCache.set(cacheKey, metrics);
    this.lastUpdateTimes.set(cacheKey, now);
    
    return metrics;
  }

  private calculateRealTimeMetrics(candles: CandleData[]): RealTimeMetrics {
    const currentPrice = candles[candles.length - 1].close;
    const previousPrice = candles[candles.length - 2].close;
    const firstPrice = candles[0].close;
    
    // Процентное изменение цены
    const priceChangePercent = ((currentPrice - previousPrice) / previousPrice) * 100;
    
    // Индекс волатильности
    const volatilityIndex = this.calculateVolatilityIndex(candles);
    
    // Оценка моментума
    const momentumScore = this.calculateMomentumScore(candles);
    
    // Уровни поддержки и сопротивления
    const { supportLevel, resistanceLevel } = this.calculateSupportResistance(candles);
    
    // Сила тренда
    const trendStrength = this.calculateTrendStrength(candles);
    
    return {
      priceChangePercent,
      volatilityIndex,
      momentumScore,
      supportLevel,
      resistanceLevel,
      trendStrength
    };
  }

  private calculateVolatilityIndex(candles: CandleData[]): number {
    const ranges = candles.map(c => c.high - c.low);
    const avgRange = ranges.reduce((sum, r) => sum + r, 0) / ranges.length;
    const currentRange = ranges[ranges.length - 1];
    
    return (currentRange / avgRange) * 100;
  }

  private calculateMomentumScore(candles: CandleData[]): number {
    if (candles.length < 10) return 0;
    
    const shortMA = this.calculateSMA(candles.slice(-5).map(c => c.close));
    const longMA = this.calculateSMA(candles.slice(-10).map(c => c.close));
    
    if (longMA === 0) return 0;
    
    return ((shortMA - longMA) / longMA) * 100;
  }

  private calculateSMA(prices: number[]): number {
    if (prices.length === 0) return 0;
    return prices.reduce((sum, price) => sum + price, 0) / prices.length;
  }

  private calculateSupportResistance(candles: CandleData[]): { supportLevel: number; resistanceLevel: number } {
    const lows = candles.map(c => c.low);
    const highs = candles.map(c => c.high);
    
    // Простой алгоритм: находим минимумы и максимумы
    const supportLevel = Math.min(...lows);
    const resistanceLevel = Math.max(...highs);
    
    return { supportLevel, resistanceLevel };
  }

  private calculateTrendStrength(candles: CandleData[]): number {
    if (candles.length < 5) return 0;
    
    const prices = candles.map(c => c.close);
    const firstPrice = prices[0];
    const lastPrice = prices[prices.length - 1];
    const totalChange = (lastPrice - firstPrice) / firstPrice;
    
    // Подсчитываем количество движений в направлении тренда
    let trendMoves = 0;
    for (let i = 1; i < prices.length; i++) {
      const move = prices[i] - prices[i - 1];
      if ((totalChange > 0 && move > 0) || (totalChange < 0 && move < 0)) {
        trendMoves++;
      }
    }
    
    const consistency = trendMoves / (prices.length - 1);
    return Math.abs(totalChange) * consistency * 100;
  }

  updateSessionData(sessionId: string, candles: CandleData[]): void {
    this.sessionCache.set(sessionId, candles);
    
    // Очищаем кэш метрик при обновлении данных
    this.metricsCache.delete(sessionId);
    this.lastUpdateTimes.delete(sessionId);
  }

  clearCache(): void {
    this.sessionCache.clear();
    this.metricsCache.clear();
    this.lastUpdateTimes.clear();
  }
}

export const sessionBasedDataService = SessionBasedDataService.getInstance();