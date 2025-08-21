
import { CandleData } from '@/types/session';
import { PatternResult } from './types';

export const detectDoubleTop = (candles: CandleData[], index: number): PatternResult | null => {
  if (index < 10) return null;
  
  const recent = candles.slice(index - 10, index + 1);
  
  let peaks = [];
  for (let i = 1; i < recent.length - 1; i++) {
    if (recent[i].high > recent[i-1].high && recent[i].high > recent[i+1].high) {
      peaks.push({ index: i, high: recent[i].high });
    }
  }
  
  if (peaks.length >= 2) {
    const lastTwoPeaks = peaks.slice(-2);
    const [peak1, peak2] = lastTwoPeaks;
    
    if (Math.abs(peak1.high - peak2.high) / peak1.high < 0.02) {
      return {
        name: 'Двойная вершина',
        type: 'BEARISH',
        confidence: 75,
        description: 'Формирование двух примерно равных максимумов',
        requiredCandles: 10,
        candleRange: `${index - 9}-${index + 1}`
      };
    }
  }
  
  return null;
};

export const detectHeadAndShoulders = (candles: CandleData[], index: number): PatternResult | null => {
  if (index < 15) return null;
  
  const recent = candles.slice(index - 15, index + 1);
  let peaks = [];
  
  for (let i = 2; i < recent.length - 2; i++) {
    if (recent[i].high > recent[i-1].high && 
        recent[i].high > recent[i+1].high &&
        recent[i].high > recent[i-2].high && 
        recent[i].high > recent[i+2].high) {
      peaks.push({ index: i, high: recent[i].high });
    }
  }
  
  if (peaks.length >= 3) {
    const lastThreePeaks = peaks.slice(-3);
    const [left, head, right] = lastThreePeaks;
    
    if (head.high > left.high && head.high > right.high &&
        Math.abs(left.high - right.high) / left.high < 0.05) {
      return {
        name: 'Голова и плечи',
        type: 'BEARISH',
        confidence: 80,
        description: 'Классический разворотный паттерн с тремя пиками',
        requiredCandles: 15,
        candleRange: `${index - 14}-${index + 1}`
      };
    }
  }
  
  return null;
};

export const detectFlagPattern = (candles: CandleData[], index: number): PatternResult | null => {
  if (index < 8) return null;
  
  const recent = candles.slice(index - 8, index + 1);
  
  // Ищем сильное движение в начале
  const firstThree = recent.slice(0, 3);
  const strongMove = firstThree.every(c => c.close > c.open) || 
                    firstThree.every(c => c.close < c.open);
  
  if (!strongMove) return null;
  
  // Проверяем консолидацию после
  const consolidation = recent.slice(3);
  const priceRange = Math.max(...consolidation.map(c => c.high)) - 
                     Math.min(...consolidation.map(c => c.low));
  const avgPrice = consolidation.reduce((sum, c) => sum + (c.high + c.low) / 2, 0) / consolidation.length;
  
  if (priceRange / avgPrice < 0.03) { // Узкий диапазон
    const isBullish = firstThree[0].close < firstThree[2].close;
    return {
      name: isBullish ? 'Бычий флаг' : 'Медвежий флаг',
      type: isBullish ? 'BULLISH' : 'BEARISH',
      confidence: 70,
      description: 'Паттерн продолжения тренда после сильного движения',
      requiredCandles: 8,
      candleRange: `${index - 7}-${index + 1}`
    };
  }
  
  return null;
};
