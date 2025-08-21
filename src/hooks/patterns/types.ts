export interface PatternResult {
  name: string;
  type: string;
  confidence: number;
  index: number;
  strength: number;
  description?: string;
}

export interface PatternDetectionConfig {
  maxPatterns: number;
  minConfidence: number;
}