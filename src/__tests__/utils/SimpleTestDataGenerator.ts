import { CandleData } from '@/types/session';

export class SimpleTestDataGenerator {
  /**
   * Generate basic candle data for testing
   */
  static generateCandles(count: number, symbol: string = 'EURUSD'): CandleData[] {
    const candles: CandleData[] = [];
    let currentPrice = 1.2000;

    for (let i = 0; i < count; i++) {
      // Simple random walk
      const change = (Math.random() - 0.5) * 0.01;
      currentPrice += change;

      const volatility = 0.005;
      const high = currentPrice + Math.random() * volatility;
      const low = currentPrice - Math.random() * volatility;
      const open = currentPrice - change / 2;

      candles.push({
        id: `${symbol}_${i}`,
        timestamp: new Date(Date.now() + i * 60000).toISOString(),
        open,
        high: Math.max(open, currentPrice, high),
        low: Math.min(open, currentPrice, low),
        close: currentPrice,
        volume: Math.floor(1000 + Math.random() * 2000),
        session_id: 'test_session',
        candle_index: i,
        candle_datetime: new Date(Date.now() + i * 60000).toISOString()
      } as CandleData);
    }

    return candles;
  }

  /**
   * Generate trending data
   */
  static generateTrendingData(count: number, direction: 'up' | 'down' = 'up'): CandleData[] {
    const candles: CandleData[] = [];
    let currentPrice = 100;
    const trendStrength = direction === 'up' ? 0.001 : -0.001;

    for (let i = 0; i < count; i++) {
      const trend = trendStrength * i;
      const noise = (Math.random() - 0.5) * 0.01;
      currentPrice = 100 + trend + noise;

      const volatility = 0.005;
      const high = currentPrice + Math.random() * volatility;
      const low = currentPrice - Math.random() * volatility;

      candles.push({
        id: `trend_${i}`,
        timestamp: new Date(Date.now() + i * 60000).toISOString(),
        open: currentPrice - noise / 2,
        high: Math.max(currentPrice, high),
        low: Math.min(currentPrice, low),
        close: currentPrice,
        volume: Math.floor(1000 + Math.random() * 1000),
        session_id: 'trend_session',
        candle_index: i,
        candle_datetime: new Date(Date.now() + i * 60000).toISOString()
      } as CandleData);
    }

    return candles;
  }

  /**
   * Generate volatile market data
   */
  static generateVolatileData(count: number): CandleData[] {
    const candles: CandleData[] = [];
    let currentPrice = 1.1000;

    for (let i = 0; i < count; i++) {
      // High volatility simulation
      const volatility = 0.02;
      const change = (Math.random() - 0.5) * volatility;
      currentPrice += change;

      const intrabarVol = volatility * 0.5;
      const high = currentPrice + Math.random() * intrabarVol;
      const low = currentPrice - Math.random() * intrabarVol;

      candles.push({
        id: `volatile_${i}`,
        timestamp: new Date(Date.now() + i * 60000).toISOString(),
        open: currentPrice - change,
        high: Math.max(currentPrice - change, currentPrice, high),
        low: Math.min(currentPrice - change, currentPrice, low),
        close: currentPrice,
        volume: Math.floor(2000 + Math.random() * 5000), // Higher volume for volatile periods
        session_id: 'volatile_session',
        candle_index: i,
        candle_datetime: new Date(Date.now() + i * 60000).toISOString()
      } as CandleData);
    }

    return candles;
  }
}

export default SimpleTestDataGenerator;