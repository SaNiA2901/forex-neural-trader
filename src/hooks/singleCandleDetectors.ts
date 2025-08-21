
import { CandleData } from '@/types/session';
import { PatternResult } from './types';

export const detectDoji = (candles: CandleData[], index: number): PatternResult | null => {
  const candle = candles[index];
  const bodySize = Math.abs(candle.close - candle.open);
  const totalRange = candle.high - candle.low;
  
  if (bodySize / totalRange < 0.1 && totalRange > 0) {
    return {
      name: 'Дожи',
      type: 'NEUTRAL',
      confidence: 75,
      description: 'Свеча с очень маленьким телом, указывает на неопределенность рынка',
      requiredCandles: 1,
      candleRange: `${index + 1}`
    };
  }
  return null;
};

export const detectHammer = (candles: CandleData[], index: number): PatternResult | null => {
  const candle = candles[index];
  const bodySize = Math.abs(candle.close - candle.open);
  const lowerShadow = Math.min(candle.open, candle.close) - candle.low;
  const upperShadow = candle.high - Math.max(candle.open, candle.close);
  
  if (lowerShadow > bodySize * 2 && upperShadow < bodySize * 0.5 && bodySize > 0) {
    return {
      name: 'Молот',
      type: 'BULLISH',
      confidence: 80,
      description: 'Разворотный паттерн с длинной нижней тенью',
      requiredCandles: 1,
      candleRange: `${index + 1}`
    };
  }
  return null;
};
