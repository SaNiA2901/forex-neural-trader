
import { useMemo } from 'react';
import { CandleData } from '@/types/session';
import { PatternResult, PatternDetectionConfig } from './patterns/types';
import { detectDoji, detectHammer } from './patterns/singleCandleDetectors';
import { detectEngulfing } from './patterns/twoCandleDetectors';
import { 
  detectThreeWhiteSoldiers, 
  detectThreeBlackCrows, 
  detectMorningStar, 
  detectEveningStar 
} from './patterns/threeCandleDetectors';
import { 
  detectDoubleTop, 
  detectHeadAndShoulders, 
  detectFlagPattern 
} from './patterns/complexPatternDetectors';
import { 
  isDuplicatePattern, 
  sortPatternsByConfidence, 
  filterPatternsByConfidence,
  executeDetectorSafely 
} from './patterns/patternUtils';

const DEFAULT_CONFIG: PatternDetectionConfig = {
  maxPatterns: 10,
  minConfidence: 60
};

// Мемоизируем массив детекторов для предотвращения пересоздания
const PATTERN_DETECTORS = [
  { fn: detectDoji, name: 'detectDoji' },
  { fn: detectHammer, name: 'detectHammer' },
  { fn: detectEngulfing, name: 'detectEngulfing' },
  { fn: detectThreeWhiteSoldiers, name: 'detectThreeWhiteSoldiers' },
  { fn: detectThreeBlackCrows, name: 'detectThreeBlackCrows' },
  { fn: detectMorningStar, name: 'detectMorningStar' },
  { fn: detectEveningStar, name: 'detectEveningStar' },
  { fn: detectDoubleTop, name: 'detectDoubleTop' },
  { fn: detectHeadAndShoulders, name: 'detectHeadAndShoulders' },
  { fn: detectFlagPattern, name: 'detectFlagPattern' }
];

export const usePatternDetection = (
  candles: CandleData[], 
  config: Partial<PatternDetectionConfig> = {}
) => {
  const finalConfig = useMemo(() => ({ ...DEFAULT_CONFIG, ...config }), [config]);

  const detectedPatterns = useMemo(() => {
    if (candles.length < 5) return [];
    
    const patterns: PatternResult[] = [];
    
    // Оптимизация: обрабатываем только последние N свечей для лучшей производительности
    const maxCandlesToAnalyze = Math.min(candles.length, 100);
    const startIndex = Math.max(0, candles.length - maxCandlesToAnalyze);
    
    // Проходим по свечам и ищем паттерны
    for (let i = startIndex; i < candles.length; i++) {
      for (const detector of PATTERN_DETECTORS) {
        const pattern = executeDetectorSafely(detector.fn, candles, i, detector.name);
        
        if (pattern && !isDuplicatePattern(patterns, pattern, i)) {
          patterns.push(pattern);
        }
      }
    }
    
    // Фильтруем и сортируем
    const filteredPatterns = filterPatternsByConfidence(patterns, finalConfig.minConfidence);
    const sortedPatterns = sortPatternsByConfidence(filteredPatterns);
    
    return sortedPatterns.slice(0, finalConfig.maxPatterns);
  }, [candles, finalConfig.maxPatterns, finalConfig.minConfidence]);

  const hasPatterns = detectedPatterns.length > 0;

  return {
    detectedPatterns,
    hasPatterns
  };
};

// Экспортируем типы для использования в других файлах
export type { PatternResult } from './patterns/types';
