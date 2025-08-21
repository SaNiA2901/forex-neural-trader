import { CandleData } from '@/types/session';
import { PatternResult } from './types';

export const detectDoubleTop = (candles: CandleData[], index: number): PatternResult | null => {
  if (index < 10 || index >= candles.length) return null;
  
  // Simplified double top detection
  const lookback = 10;
  const recentCandles = candles.slice(index - lookback, index + 1);
  const highs = recentCandles.map(c => c.high);
  const maxHigh = Math.max(...highs);
  
  let peaks = 0;
  for (let i = 1; i < highs.length - 1; i++) {
    if (highs[i] > highs[i-1] && highs[i] > highs[i+1] && highs[i] > maxHigh * 0.98) {
      peaks++;
    }
  }
  
  if (peaks >= 2) {
    return {
      name: 'Double Top',
      type: 'reversal',
      confidence: 70,
      index,
      strength: 0.7,
      description: 'Double top pattern detected'
    };
  }
  
  return null;
};

export const detectHeadAndShoulders = (candles: CandleData[], index: number): PatternResult | null => {
  if (index < 15 || index >= candles.length) return null;
  
  // Simplified head and shoulders detection
  const lookback = 15;
  const recentCandles = candles.slice(index - lookback, index + 1);
  const highs = recentCandles.map(c => c.high);
  
  // Find three distinct peaks
  let peaks = [];
  for (let i = 2; i < highs.length - 2; i++) {
    if (highs[i] > highs[i-1] && highs[i] > highs[i+1] && 
        highs[i] > highs[i-2] && highs[i] > highs[i+2]) {
      peaks.push({ index: i, value: highs[i] });
    }
  }
  
  if (peaks.length >= 3) {
    const sorted = peaks.sort((a, b) => b.value - a.value);
    const head = sorted[0];
    const leftShoulder = sorted[1];
    const rightShoulder = sorted[2];
    
    if (leftShoulder.index < head.index && head.index < rightShoulder.index) {
      return {
        name: 'Head and Shoulders',
        type: 'reversal',
        confidence: 75,
        index,
        strength: 0.75,
        description: 'Head and shoulders pattern detected'
      };
    }
  }
  
  return null;
};

export const detectFlagPattern = (candles: CandleData[], index: number): PatternResult | null => {
  if (index < 8 || index >= candles.length) return null;
  
  // Simplified flag pattern detection
  const lookback = 8;
  const recentCandles = candles.slice(index - lookback, index + 1);
  
  // Check for consolidation after a strong move
  const firstHalf = recentCandles.slice(0, 4);
  const secondHalf = recentCandles.slice(4);
  
  const firstRange = Math.max(...firstHalf.map(c => c.high)) - Math.min(...firstHalf.map(c => c.low));
  const secondRange = Math.max(...secondHalf.map(c => c.high)) - Math.min(...secondHalf.map(c => c.low));
  
  if (secondRange < firstRange * 0.5) {
    return {
      name: 'Flag Pattern',
      type: 'continuation',
      confidence: 65,
      index,
      strength: 0.65,
      description: 'Flag pattern detected'
    };
  }
  
  return null;
};