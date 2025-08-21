import { PatternResult } from './types';

export const isDuplicatePattern = (
  patterns: PatternResult[], 
  newPattern: PatternResult, 
  currentIndex: number
): boolean => {
  return patterns.some(p => 
    p.name === newPattern.name && 
    Math.abs(p.index - currentIndex) < 3
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
  detector: Function, 
  candles: any[], 
  index: number, 
  detectorName: string
): PatternResult | null => {
  try {
    return detector(candles, index);
  } catch (error) {
    console.warn(`Error in ${detectorName}:`, error);
    return null;
  }
};