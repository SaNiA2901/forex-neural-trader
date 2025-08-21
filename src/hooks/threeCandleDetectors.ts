
import { CandleData } from '@/types/session';
import { PatternResult } from './types';

export const detectThreeWhiteSoldiers = (candles: CandleData[], index: number): PatternResult | null => {
  if (index < 2) return null;
  
  const candle1 = candles[index - 2];
  const candle2 = candles[index - 1];
  const candle3 = candles[index];
  
  if (candle1.close > candle1.open && 
      candle2.close > candle2.open && 
      candle3.close > candle3.open &&
      candle2.close > candle1.close &&
      candle3.close > candle2.close) {
    return {
      name: 'Три белых солдата',
      type: 'BULLISH',
      confidence: 90,
      description: 'Три подряд идущие бычьи свечи с возрастающими максимумами',
      requiredCandles: 3,
      candleRange: `${index - 1}-${index + 1}`
    };
  }
  
  return null;
};

export const detectThreeBlackCrows = (candles: CandleData[], index: number): PatternResult | null => {
  if (index < 2) return null;
  
  const candle1 = candles[index - 2];
  const candle2 = candles[index - 1];
  const candle3 = candles[index];
  
  if (candle1.close < candle1.open && 
      candle2.close < candle2.open && 
      candle3.close < candle3.open &&
      candle2.close < candle1.close &&
      candle3.close < candle2.close) {
    return {
      name: 'Три черные вороны',
      type: 'BEARISH',
      confidence: 90,
      description: 'Три подряд идущие медвежьи свечи с убывающими минимумами',
      requiredCandles: 3,
      candleRange: `${index - 1}-${index + 1}`
    };
  }
  
  return null;
};

export const detectMorningStar = (candles: CandleData[], index: number): PatternResult | null => {
  if (index < 2) return null;
  
  const candle1 = candles[index - 2];
  const candle2 = candles[index - 1];
  const candle3 = candles[index];
  
  const body1 = Math.abs(candle1.close - candle1.open);
  const body2 = Math.abs(candle2.close - candle2.open);
  const body3 = Math.abs(candle3.close - candle3.open);
  
  if (candle1.close < candle1.open && // Первая свеча медвежья
      body2 < body1 * 0.3 && // Вторая свеча маленькая
      candle3.close > candle3.open && // Третья свеча бычья
      candle3.close > (candle1.open + candle1.close) / 2) { // Закрытие выше середины первой свечи
    return {
      name: 'Утренняя звезда',
      type: 'BULLISH',
      confidence: 85,
      description: 'Трёхсвечной разворотный паттерн снизу вверх',
      requiredCandles: 3,
      candleRange: `${index - 1}-${index + 1}`
    };
  }
  
  return null;
};

export const detectEveningStar = (candles: CandleData[], index: number): PatternResult | null => {
  if (index < 2) return null;
  
  const candle1 = candles[index - 2];
  const candle2 = candles[index - 1];
  const candle3 = candles[index];
  
  const body1 = Math.abs(candle1.close - candle1.open);
  const body2 = Math.abs(candle2.close - candle2.open);
  const body3 = Math.abs(candle3.close - candle3.open);
  
  if (candle1.close > candle1.open && // Первая свеча бычья
      body2 < body1 * 0.3 && // Вторая свеча маленькая
      candle3.close < candle3.open && // Третья свеча медвежья
      candle3.close < (candle1.open + candle1.close) / 2) { // Закрытие ниже середины первой свечи
    return {
      name: 'Вечерняя звезда',
      type: 'BEARISH',
      confidence: 85,
      description: 'Трёхсвечной разворотный паттерн сверху вниз',
      requiredCandles: 3,
      candleRange: `${index - 1}-${index + 1}`
    };
  }
  
  return null;
};
