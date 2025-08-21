import { CandleData } from '@/types/session';
import { PatternResult } from './types';

export const detectEngulfing = (candles: CandleData[], index: number): PatternResult | null => {
  if (index < 1 || index >= candles.length) return null;
  
  const prev = candles[index - 1];
  const curr = candles[index];
  
  const prevBullish = prev.close > prev.open;
  const currBullish = curr.close > curr.open;
  
  // Bullish engulfing
  if (!prevBullish && currBullish && 
      curr.open < prev.close && curr.close > prev.open) {
    return {
      name: 'Bullish Engulfing',
      type: 'reversal',
      confidence: 85,
      index,
      strength: 0.85,
      description: 'Bullish engulfing pattern detected'
    };
  }
  
  // Bearish engulfing
  if (prevBullish && !currBullish && 
      curr.open > prev.close && curr.close < prev.open) {
    return {
      name: 'Bearish Engulfing',
      type: 'reversal',
      confidence: 85,
      index,
      strength: 0.85,
      description: 'Bearish engulfing pattern detected'
    };
  }
  
  return null;
};