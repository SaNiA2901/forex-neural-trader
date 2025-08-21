
export interface ManualDataInputs {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  date: string;
  time: string;
}

export interface PredictionConfig {
  predictionInterval: number;
  analysisMode: 'session';
}

export interface PredictionResult {
  direction: 'UP' | 'DOWN';
  probability: number;
  confidence: number;
  interval: number;
  factors: {
    technical: number;
    volume: number;
    momentum: number;
    volatility: number;
    pattern: number;
    trend: number;
  };
  recommendation: string;
  metadata?: {
    modelAgreement: number;
    riskScore: number;
    marketCondition: string;
    modelBreakdown: any[];
    patternDetected?: string;
    volumeAnalysis?: string;
    riskLevel?: string;
  };
}

export interface ModelStatistics {
  totalPredictions: number;
  accurateCount: number;
  overallAccuracy: number;
  callAccuracy: number;
  putAccuracy: number;
  currentWeights: {
    technical: number;
    volume: number;
    momentum: number;
    volatility: number;
    pattern: number;
    trend: number;
  };
}

export interface TechnicalIndicators {
  rsi: number;
  macd: { line: number; signal: number; histogram: number };
  bollingerBands: { upper: number; middle: number; lower: number };
  ema: { ema12: number; ema26: number };
  stochastic: { k: number; d: number };
  atr: number;
  adx: number;
}

export interface PatternSignal {
  name: string;
  strength: number;
  isReversal: boolean;
  isContinuation: boolean;
}

export interface TechnicalIndicator {
  name: string;
  value: number;
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
  strength: number;
  description?: string;
}

export interface PatternDetection {
  pattern: string;
  confidence: number;
  timeframe: string;
  name: string;
  type: string;
  index?: number;
  strength: number;
  description?: string;
}

export interface SupportResistanceLevel {
  level: number;
  strength: number;
  type: 'support' | 'resistance';
}

export interface VolumeProfile {
  price: number;
  volume: number;
  percentage: number;
}

export interface MarketStructure {
  trend: 'UPTREND' | 'DOWNTREND' | 'SIDEWAYS';
  strength: number;
  breakoutLevel?: number;
}

export interface MarketData {
  symbol: string;
  price: number;
  volume: number;
  timestamp: number;
  change: number;
  changePercent: number;
  volatility: number;
}

export interface StreamConfig {
  symbols: string[];
  interval: number;
  autoReconnect: boolean;
  enableRealTime?: boolean;
  enableVolume?: boolean;
  updateInterval?: number;
  maxHistoryLength?: number;
  enableOrderBook?: boolean;
}

export interface RiskMetrics {
  var95: number;
  var99: number;
  expectedShortfall: number;
  sharpeRatio: number;
  riskLevel: number;
  volatility: number;
  maxLoss: number;
  riskAdjustedReturn: number;
}


export interface VaRCalculation {
  historical: number;
  parametric: number;
  monteCarlo: number;
  var95: number;
  expectedShortfall: number;
  confidenceLevel: number;
  historicalVaR: number[];
}

export interface DrawdownAnalysis {
  currentDrawdown: number;
  maxDrawdown: number;
  drawdownDuration: number;
  avgDrawdown: number;
  maxDrawdownDuration: number;
  recoveryTime: number;
  drawdownHistory: Array<{date: string; value: number}>;
}

export interface PortfolioRisk {
  totalExposure: number;
  concentrationRisk: number;
  correlationRisk: number;
  beta: number;
  alpha: number;
  correlationWithMarket: number;
  assetAllocation: Array<{asset: string; percentage: number}>;
}
