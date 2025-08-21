
import { CandleData } from '@/types/session';

export interface PatternResult {
  name: string;
  type: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  confidence: number;
  description: string;
  requiredCandles: number;
  candleRange: string;
}

export type PatternDetector = (candles: CandleData[], index: number) => PatternResult | null;

export interface PatternDetectionConfig {
  maxPatterns: number;
  minConfidence: number;
}
