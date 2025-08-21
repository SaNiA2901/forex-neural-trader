import { CandleData } from '@/types/session';
import { TechnicalIndicators } from '../indicators/TechnicalIndicators';
import { PatternSignals, VolumeAnalysis } from '../patterns/PatternAnalysis';

export interface ModelWeights {
  technical: number;
  volume: number;
  momentum: number;
  volatility: number;
  pattern: number;
  trend: number;
}

export interface AdvancedFactors {
  technical: number;
  volume: number;
  momentum: number;
  volatility: number;
  pattern: number;
  trend: number;
}

export class AdvancedFactorsService {
  private static readonly adaptiveWeights: ModelWeights = {
    technical: 0.30,
    volume: 0.15,
    momentum: 0.25,
    volatility: 0.15,
    pattern: 0.10,
    trend: 0.05
  };

  static calculateAdvancedFactors(
    current: CandleData,
    technical: TechnicalIndicators,
    patterns: PatternSignals,
    volume: VolumeAnalysis
  ): AdvancedFactors {
    // Technical factor (RSI + MACD + Bollinger Bands)
    let technicalFactor = 50;
    
    // RSI analysis
    if (technical.rsi > 70) technicalFactor -= 20; // Перекупленность
    else if (technical.rsi < 30) technicalFactor += 20; // Перепроданность
    else technicalFactor += (50 - technical.rsi) * 0.4;
    
    // MACD analysis
    if (technical.macd.histogram > 0) technicalFactor += 10;
    else technicalFactor -= 10;
    
    // Bollinger Bands analysis
    if (current.close > technical.bollingerBands.upper) technicalFactor -= 15;
    else if (current.close < technical.bollingerBands.lower) technicalFactor += 15;
    
    // Volume factor
    let volumeFactor = 50;
    switch (volume.volumeTrend) {
      case 'increasing': volumeFactor += 20; break;
      case 'decreasing': volumeFactor -= 10; break;
      default: volumeFactor += 0;
    }
    
    // Momentum factor (based on EMA and Stochastic)
    let momentumFactor = 50;
    if (technical.ema.ema12 > technical.ema.ema26) momentumFactor += 15;
    else momentumFactor -= 15;
    
    if (technical.stochastic.k > 80) momentumFactor -= 10;
    else if (technical.stochastic.k < 20) momentumFactor += 10;
    
    // Volatility factor (based on ATR and ADX)
    let volatilityFactor = 50;
    if (technical.atr > 0) {
      const volatilityRatio = technical.atr / current.close;
      if (volatilityRatio > 0.02) volatilityFactor += 20; // Высокая волатильность
      else if (volatilityRatio < 0.005) volatilityFactor -= 10; // Низкая волатильность
    }
    
    // Pattern factor
    let patternFactor = 50;
    if (patterns.candlestickPattern) {
      if (patterns.isReversal) {
        patternFactor += patterns.strength * 30;
      }
    }
    
    // Trend factor (based on ADX)
    let trendFactor = 50;
    if (technical.adx > 60) trendFactor += 20; // Сильный тренд
    else if (technical.adx < 25) trendFactor -= 10; // Слабый тренд
    
    return {
      technical: Math.max(0, Math.min(100, technicalFactor)),
      volume: Math.max(0, Math.min(100, volumeFactor)),
      momentum: Math.max(0, Math.min(100, momentumFactor)),
      volatility: Math.max(0, Math.min(100, volatilityFactor)),
      pattern: Math.max(0, Math.min(100, patternFactor)),
      trend: Math.max(0, Math.min(100, trendFactor))
    };
  }

  static calculateWeightedScore(factors: AdvancedFactors, weights: ModelWeights = this.adaptiveWeights): number {
    return (
      factors.technical * weights.technical +
      factors.volume * weights.volume +
      factors.momentum * weights.momentum +
      factors.volatility * weights.volatility +
      factors.pattern * weights.pattern +
      factors.trend * weights.trend
    );
  }

  static calculateConfidenceModifiers(technical: TechnicalIndicators, patterns: PatternSignals): number {
    let modifier = 1.0;
    
    // Высокий ADX увеличивает уверенность
    if (technical.adx > 60) modifier += 0.2;
    
    // Паттерны увеличивают уверенность
    if (patterns.candlestickPattern) {
      modifier += patterns.strength * 0.3;
    }
    
    // Экстремальные значения RSI увеличивают уверенность
    if (technical.rsi > 80 || technical.rsi < 20) {
      modifier += 0.15;
    }
    
    return Math.min(1.5, modifier);
  }

  static getModelWeights(): ModelWeights {
    return { ...this.adaptiveWeights };
  }

  static updateModelWeights(predictionHistory: any[]): void {
    // Простая адаптация весов на основе производительности
    if (predictionHistory.length < 10) return;
    
    const recentResults = predictionHistory.slice(-20);
    const technicalSuccessRate = this.calculateFactorSuccessRate(recentResults, 'technical');
    const volumeSuccessRate = this.calculateFactorSuccessRate(recentResults, 'volume');
    
    // Адаптируем веса (упрощенная версия)
    if (technicalSuccessRate > 0.7) this.adaptiveWeights.technical = Math.min(0.4, this.adaptiveWeights.technical + 0.05);
    if (volumeSuccessRate > 0.7) this.adaptiveWeights.volume = Math.min(0.3, this.adaptiveWeights.volume + 0.02);
  }

  private static calculateFactorSuccessRate(results: any[], factor: string): number {
    const relevantResults = results.filter(r => r.actualOutcome);
    if (relevantResults.length === 0) return 0.5;
    
    const successCount = relevantResults.filter(r => r.result).length;
    return successCount / relevantResults.length;
  }
}