
import { useMemo } from 'react';
import { TradingSession, CandleData, SessionStats } from '@/types/session';

export const useSessionStats = (currentSession: TradingSession | null, candles: CandleData[]) => {
  const sessionStats = useMemo((): SessionStats => {
    if (!currentSession || candles.length === 0) {
      return { 
        totalCandles: 0, 
        lastPrice: null, 
        priceChange: 0,
        highestPrice: null,
        lowestPrice: null,
        averageVolume: 0
      };
    }

    const sortedCandles = [...candles].sort((a, b) => a.candle_index - b.candle_index);
    const firstCandle = sortedCandles[0];
    const lastCandle = sortedCandles[sortedCandles.length - 1];
    
    let highestPrice = -Infinity;
    let lowestPrice = Infinity;
    let totalVolume = 0;
    
    for (const candle of candles) {
      const prices = [candle.open, candle.high, candle.low, candle.close];
      for (const price of prices) {
        if (price > highestPrice) highestPrice = price;
        if (price < lowestPrice) lowestPrice = price;
      }
      totalVolume += candle.volume;
    }
    
    return {
      totalCandles: candles.length,
      lastPrice: lastCandle?.close || null,
      priceChange: lastCandle && firstCandle 
        ? ((lastCandle.close - firstCandle.open) / firstCandle.open) * 100 
        : 0,
      highestPrice: highestPrice === -Infinity ? null : highestPrice,
      lowestPrice: lowestPrice === Infinity ? null : lowestPrice,
      averageVolume: candles.length > 0 ? totalVolume / candles.length : 0
    };
  }, [currentSession?.id, candles]);

  const nextCandleIndex = useMemo(() => {
    if (!currentSession) return 0;
    return Math.max(currentSession.current_candle_index + 1, candles.length);
  }, [currentSession?.current_candle_index, candles.length]);

  return { sessionStats, nextCandleIndex };
};
