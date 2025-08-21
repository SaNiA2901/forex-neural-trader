import { CandleData } from '@/types/session';
import { PatternResult } from './types';

export const detectDoji = (candles: CandleData[], index: number): PatternResult | null => {
  if (index >= candles.length) return null;
  
  const candle = candles[index];
  const bodySize = Math.abs(candle.close - candle.open);
  const totalSize = candle.high - candle.low;
  
  if (bodySize / totalSize < 0.1) {
    return {
      name: 'Doji',
      type: 'reversal',
      confidence: 70,
      index,
      strength: 0.7,
      description: 'Doji pattern detected'
    };
  }
  
  return null;
};

export const detectHammer = (candles: CandleData[], index: number): PatternResult | null => {
  if (index >= candles.length) return null;
  
  const candle = candles[index];
  const bodySize = Math.abs(candle.close - candle.open);
  const lowerWick = Math.min(candle.open, candle.close) - candle.low;
  const upperWick = candle.high - Math.max(candle.open, candle.close);
  
  if (lowerWick > bodySize * 2 && upperWick < bodySize * 0.5) {
    return {
      name: 'Hammer',
      type: 'reversal',
      confidence: 80,
      index,
      strength: 0.8,
      description: 'Hammer pattern detected'
    };
  }
  
  return null;
};