import { useState, useEffect, useCallback, useMemo } from 'react';
import { CandleData } from '@/types/session';

export interface AnalyticsData {
  marketDepth: any[];
  orderFlow: any[];
  sentimentData: any[];
  correlationMatrix: any[];
  volatilityData: any[];
}

export interface MLEngineData {
  ensembleResult: any | null;
  featureImportance: any[];
  modelPerformance: any[];
  isTraining: boolean;
  isGeneratingPrediction: boolean;
  modelStats: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  };
}

export interface TechnicalAnalysisData {
  indicators: any[];
  patterns: any[];
  supportResistance: any[];
  chartData: any[];
}

/**
 * Unified hook for analytics data management
 * Optimizes data generation and reduces unnecessary re-renders
 */
export const useAnalyticsData = (pair: string, timeframe: string, candles?: CandleData[]) => {
  // Analytics state
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData>({
    marketDepth: [],
    orderFlow: [],
    sentimentData: [],
    correlationMatrix: [],
    volatilityData: []
  });

  // ML Engine state
  const [mlEngineData, setMlEngineData] = useState<MLEngineData>({
    ensembleResult: null,
    featureImportance: [],
    modelPerformance: [],
    isTraining: false,
    isGeneratingPrediction: false,
    modelStats: {
      accuracy: 0,
      precision: 0,
      recall: 0,
      f1Score: 0
    }
  });

  // Technical Analysis state
  const [technicalData, setTechnicalData] = useState<TechnicalAnalysisData>({
    indicators: [],
    patterns: [],
    supportResistance: [],
    chartData: []
  });

  // Memoized data generation functions
  const generateAdvancedAnalytics = useCallback(() => {
    // Market Depth Data
    const marketDepth = Array.from({ length: 20 }, (_, i) => ({
      price: 1.1000 + (i - 10) * 0.0001,
      bids: Math.floor(Math.random() * 1000000) + 100000,
      asks: Math.floor(Math.random() * 1000000) + 100000,
      orders: Math.floor(Math.random() * 50) + 10
    }));

    // Order Flow Data
    const orderFlow = Array.from({ length: 24 }, (_, i) => ({
      hour: i,
      buyVolume: Math.floor(Math.random() * 5000000) + 1000000,
      sellVolume: Math.floor(Math.random() * 5000000) + 1000000,
      netFlow: Math.floor(Math.random() * 2000000) - 1000000
    }));

    // Sentiment Data
    const sentimentData = Array.from({ length: 30 }, (_, i) => ({
      date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      bullish: Math.floor(Math.random() * 40) + 30,
      bearish: Math.floor(Math.random() * 40) + 30,
      neutral: Math.floor(Math.random() * 20) + 10
    }));

    // Correlation Matrix
    const assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'];
    const correlationMatrix = assets.map(asset1 => 
      assets.map(asset2 => ({
        asset1,
        asset2,
        correlation: asset1 === asset2 ? 1 : (Math.random() * 2 - 1).toFixed(3)
      }))
    ).flat();

    // Volatility Data
    const volatilityData = Array.from({ length: 100 }, (_, i) => ({
      period: i + 1,
      impliedVol: Math.random() * 0.3 + 0.1,
      realizedVol: Math.random() * 0.25 + 0.08,
      volOfVol: Math.random() * 0.1 + 0.02
    }));

    setAnalyticsData({
      marketDepth,
      orderFlow,
      sentimentData,
      correlationMatrix,
      volatilityData
    });
  }, []);

  const generateMLData = useCallback(() => {
    if (!candles || candles.length < 10) return;

    // Feature Importance
    const features = [
      'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'EMA_12', 'EMA_26', 
      'Volume', 'Price_Change', 'Volatility', 'Momentum'
    ];
    
    const featureImportance = features.map(feature => ({
      feature,
      importance: Math.random(),
      rank: 0
    })).sort((a, b) => b.importance - a.importance)
      .map((item, index) => ({ ...item, rank: index + 1 }));

    // Model Performance
    const modelPerformance = Array.from({ length: 50 }, (_, i) => ({
      epoch: i + 1,
      accuracy: Math.min(0.95, 0.5 + (i / 50) * 0.4 + Math.random() * 0.1),
      loss: Math.max(0.05, 1 - (i / 50) * 0.8 + Math.random() * 0.2),
      valAccuracy: Math.min(0.92, 0.48 + (i / 50) * 0.38 + Math.random() * 0.12),
      valLoss: Math.max(0.08, 1.05 - (i / 50) * 0.75 + Math.random() * 0.25)
    }));

    // Ensemble Result (mock)
    const ensembleResult = {
      prediction: Math.random() > 0.5 ? 'UP' : 'DOWN',
      confidence: Math.random() * 0.3 + 0.7,
      models: [
        { name: 'LSTM', prediction: 'UP', confidence: 0.82 },
        { name: 'Random Forest', prediction: 'UP', confidence: 0.78 },
        { name: 'SVM', prediction: 'DOWN', confidence: 0.75 },
        { name: 'Neural Network', prediction: 'UP', confidence: 0.85 }
      ]
    };

    setMlEngineData(prev => ({
      ...prev,
      featureImportance,
      modelPerformance,
      ensembleResult,
      modelStats: {
        accuracy: 0.84,
        precision: 0.82,
        recall: 0.86,
        f1Score: 0.84
      }
    }));
  }, [candles]);

  const generateTechnicalAnalysis = useCallback(() => {
    if (!candles || candles.length < 20) return;

    // Technical Indicators
    const indicators = [
      { name: 'RSI', value: Math.random() * 100, signal: 'NEUTRAL' },
      { name: 'MACD', value: Math.random() * 2 - 1, signal: 'BUY' },
      { name: 'Stochastic', value: Math.random() * 100, signal: 'SELL' },
      { name: 'Williams %R', value: Math.random() * 100, signal: 'BUY' }
    ];

    // Pattern Detection
    const patterns = [
      { name: 'Double Top', probability: 0.75, timeframe: '4H' },
      { name: 'Head and Shoulders', probability: 0.68, timeframe: '1D' },
      { name: 'Bull Flag', probability: 0.82, timeframe: '1H' }
    ];

    // Support/Resistance Levels
    const currentPrice = candles[candles.length - 1]?.close || 1.1000;
    const supportResistance = [
      { level: currentPrice + 0.001, type: 'resistance', strength: 0.8 },
      { level: currentPrice - 0.0015, type: 'support', strength: 0.9 },
      { level: currentPrice + 0.002, type: 'resistance', strength: 0.6 }
    ];

    // Chart Data
    const chartData = candles.slice(-50).map((candle, index) => ({
      index,
      price: candle.close,
      volume: candle.volume,
      rsi: Math.random() * 100,
      macd: Math.random() * 2 - 1
    }));

    setTechnicalData({
      indicators,
      patterns,
      supportResistance,
      chartData
    });
  }, [candles]);

  // Data update functions
  const updateMLEngineState = useCallback((updates: Partial<MLEngineData>) => {
    setMlEngineData(prev => ({ ...prev, ...updates }));
  }, []);

  // Auto-refresh data
  useEffect(() => {
    generateAdvancedAnalytics();
    generateMLData();
    generateTechnicalAnalysis();

    const interval = setInterval(() => {
      generateAdvancedAnalytics();
    }, 5000); // Refresh every 5 seconds

    return () => clearInterval(interval);
  }, [generateAdvancedAnalytics, generateMLData, generateTechnicalAnalysis]);

  // Memoized return value to prevent unnecessary re-renders
  return useMemo(() => ({
    // Data
    analyticsData,
    mlEngineData,
    technicalData,
    
    // Update functions
    updateMLEngineState,
    generateMLData,
    generateTechnicalAnalysis,
    
    // Status
    isDataReady: candles && candles.length > 0
  }), [
    analyticsData,
    mlEngineData,
    technicalData,
    updateMLEngineState,
    generateMLData,
    generateTechnicalAnalysis,
    candles
  ]);
};