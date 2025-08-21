import { CandleData } from '@/types/session';
import { PatternResult } from './types';

export const detectThreeWhiteSoldiers = (candles: CandleData[], index: number): PatternResult | null => {
  if (index < 2 || index >= candles.length) return null;
  
  const c1 = candles[index - 2];
  const c2 = candles[index - 1];
  const c3 = candles[index];
  
  if (c1.close > c1.open && c2.close > c2.open && c3.close > c3.open &&
      c2.close > c1.close && c3.close > c2.close) {
    return {
      name: 'Three White Soldiers',
      type: 'continuation',
      confidence: 75,
      index,
      strength: 0.75,
      description: 'Three white soldiers pattern detected'
    };
  }
  
  return null;
};

export const detectThreeBlackCrows = (candles: CandleData[], index: number): PatternResult | null => {
  if (index < 2 || index >= candles.length) return null;
  
  const c1 = candles[index - 2];
  const c2 = candles[index - 1];
  const c3 = candles[index];
  
  if (c1.close < c1.open && c2.close < c2.open && c3.close < c3.open &&
      c2.close < c1.close && c3.close < c2.close) {
    return {
      name: 'Three Black Crows',
      type: 'continuation',
      confidence: 75,
      index,
      strength: 0.75,
      description: 'Three black crows pattern detected'
    };
  }
  
  return null;
};

export const detectMorningStar = (candles: CandleData[], index: number): PatternResult | null => {
  if (index < 2 || index >= candles.length) return null;
  
  const c1 = candles[index - 2];
  const c2 = candles[index - 1];
  const c3 = candles[index];
  
  if (c1.close < c1.open && // First candle bearish
      Math.abs(c2.close - c2.open) < (c2.high - c2.low) * 0.3 && // Second candle small body
      c3.close > c3.open && // Third candle bullish
      c3.close > (c1.open + c1.close) / 2) { // Third closes above midpoint of first
    return {
      name: 'Morning Star',
      type: 'reversal',
      confidence: 80,
      index,
      strength: 0.8,
      description: 'Morning star pattern detected'
    };
  }
  
  return null;
};

export const detectEveningStar = (candles: CandleData[], index: number): PatternResult | null => {
  if (index < 2 || index >= candles.length) return null;
  
  const c1 = candles[index - 2];
  const c2 = candles[index - 1];
  const c3 = candles[index];
  
  if (c1.close > c1.open && // First candle bullish
      Math.abs(c2.close - c2.open) < (c2.high - c2.low) * 0.3 && // Second candle small body
      c3.close < c3.open && // Third candle bearish
      c3.close < (c1.open + c1.close) / 2) { // Third closes below midpoint of first
    return {
      name: 'Evening Star',
      type: 'reversal',
      confidence: 80,
      index,
      strength: 0.8,
      description: 'Evening star pattern detected'
    };
  }
  
  return null;
};