
import { PatternResult, PatternDetector } from './types';

export const isDuplicatePattern = (
  patterns: PatternResult[], 
  newPattern: PatternResult, 
  currentIndex: number
): boolean => {
  return patterns.some(p => 
    p.name === newPattern.name && 
    Math.abs(currentIndex - parseInt(newPattern.candleRange.split('-')[0] || '0')) < newPattern.requiredCandles
  );
};

export const sortPatternsByConfidence = (patterns: PatternResult[]): PatternResult[] => {
  return patterns.sort((a, b) => b.confidence - a.confidence);
};

export const filterPatternsByConfidence = (
  patterns: PatternResult[], 
  minConfidence: number
): PatternResult[] => {
  return patterns.filter(p => p.confidence >= minConfidence);
};

export const executeDetectorSafely = (
  detector: PatternDetector,
  candles: any[],
  index: number,
  detectorName: string
): PatternResult | null => {
  try {
    return detector(candles, index);
  } catch (error) {
    console.warn(`Error detecting pattern with ${detectorName}:`, error);
    return null;
  }
};
