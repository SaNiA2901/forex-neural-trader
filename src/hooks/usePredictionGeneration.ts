
import { useState, useCallback } from 'react';
import { PredictionResult, PredictionConfig } from '@/types/trading';
import { CandleData } from '@/types/session';
import { predictionService } from '@/services/predictionService';

// Хранилище результатов для обучения модели
const predictionHistory: Array<{ 
  factors: any; 
  result: boolean; 
  timestamp: number;
  prediction: PredictionResult;
  actualOutcome?: 'UP' | 'DOWN';
}> = [];

export const usePredictionGeneration = () => {
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [modelAccuracy, setModelAccuracy] = useState<number>(0);

  // Основная функция генерации прогноза
  const generatePrediction = useCallback(async (
    candleData: CandleData, 
    predictionConfig: PredictionConfig,
    historicalCandles?: CandleData[]
  ): Promise<PredictionResult | null> => {
    if (
      candleData == null ||
      !Number.isFinite(candleData.open) ||
      !Number.isFinite(candleData.high) ||
      !Number.isFinite(candleData.low) ||
      !Number.isFinite(candleData.close) ||
      !Number.isFinite(candleData.volume)
    ) {
      console.error('Invalid candle data for prediction:', candleData);
      return null;
    }

    setIsGenerating(true);
    
    try {
      // Если есть исторические данные, используем продвинутую модель
      if (historicalCandles && historicalCandles.length > 0) {
        const currentIndex = historicalCandles.findIndex(c => c.candle_index === candleData.candle_index);
        
        if (currentIndex >= 0 && currentIndex < historicalCandles.length - 1) {
          const prediction = await predictionService.generateAdvancedPrediction(
            historicalCandles,
            currentIndex,
            predictionConfig
          );
          
          if (prediction) {
            setPredictionResult(prediction);
            
            // Сохраняем для обучения модели
            predictionHistory.push({
              factors: prediction.factors,
              result: false, // Будет обновлено позже
              timestamp: Date.now(),
              prediction
            });
            
            return prediction;
          }
        }
      }
      
      // Fallback на простую модель
      const simplePrediction = await generateSimplePrediction(candleData, predictionConfig);
      setPredictionResult(simplePrediction);
      return simplePrediction;
      
    } catch (error) {
      console.error('Error generating prediction:', error);
      return null;
    } finally {
      setIsGenerating(false);
    }
  }, []);

  // Усиленная модель для fallback с глубоким анализом
  const generateSimplePrediction = async (
    candleData: CandleData, 
    predictionConfig: PredictionConfig
  ): Promise<PredictionResult> => {
    // Реалистичное время обработки для сложных вычислений
    await new Promise(resolve => setTimeout(resolve, 1200 + Math.random() * 800));
    
    const { open, high, low, close, volume } = candleData;
    
    // Расширенные расчеты с множественными факторами
    const priceRange = high - low;
    const bodySize = Math.abs(close - open);
    const upperShadow = high - Math.max(open, close);
    const lowerShadow = Math.min(open, close) - low;
    const middlePoint = (high + low) / 2;
    const closePosition = (close - low) / (priceRange || 0.0001);
    
    // Паттерн анализ
    const isBullish = close > open;
    const isHammer = bodySize < priceRange * 0.3 && lowerShadow > bodySize * 2;
    const isShootingStar = bodySize < priceRange * 0.3 && upperShadow > bodySize * 2;
    const isDoji = bodySize < priceRange * 0.1;
    const isMarubozu = (upperShadow + lowerShadow) < bodySize * 0.1;
    const isSpinningTop = bodySize < priceRange * 0.3 && upperShadow > bodySize && lowerShadow > bodySize;
    
    // Сложные технические факторы
    const volumeRelative = volume / 100000; // Нормализация объема
    const priceActionScore = calculatePriceActionScore(open, high, low, close, closePosition);
    const volatilityScore = calculateVolatilityScore(priceRange, close);
    const momentumScore = calculateMomentumScore(open, close, bodySize, priceRange);
    
    // Расчет весовых коэффициентов
    const volumeFactor = Math.min(95, Math.max(20, 
      40 + Math.log10(volumeRelative + 1) * 15 + (volume > 500000 ? 10 : 0)
    ));
    
    const technicalFactor = calculateTechnicalFactor(
      isBullish, isHammer, isShootingStar, isDoji, isMarubozu, isSpinningTop
    );
    
    const patternFactor = calculatePatternFactor(
      isHammer, isShootingStar, isDoji, isMarubozu, isSpinningTop, closePosition
    );
    
    // Комплексный анализ тренда
    const trendFactor = calculateTrendFactor(close, middlePoint, closePosition, isBullish);
    
    // Взвешенная оценка с динамическими весами
    const baseScore = (
      technicalFactor * 0.35 +
      volumeFactor * 0.15 +
      momentumScore * 0.20 +
      volatilityScore * 0.15 +
      patternFactor * 0.10 +
      trendFactor * 0.05
    );
    
    // Применяем коррекцию на основе рыночных условий
    const marketConditionAdjustment = calculateMarketConditionAdjustment(
      volume, priceRange, bodySize, close
    );
    
    const adjustedScore = Math.min(95, Math.max(5, baseScore + marketConditionAdjustment));
    
    const direction = adjustedScore > 50 ? 'UP' : 'DOWN';
    const rawProbability = Math.abs(adjustedScore - 50) * 2;
    const probability = Math.min(95, Math.max(55, rawProbability));
    
    // Динамическая уверенность
    const confidenceModifiers = calculateConfidenceModifiers(
      volume, priceRange, bodySize, isHammer, isDoji, isMarubozu
    );
    const confidence = Math.min(90, Math.max(60, probability - 5 + confidenceModifiers));
    
    // Расширенные факторы для отображения
    const factors = {
      technical: technicalFactor,
      volume: volumeFactor,
      momentum: momentumScore,
      volatility: volatilityScore,
      pattern: patternFactor,
      trend: trendFactor,
      priceAction: priceActionScore,
      marketCondition: 50 + marketConditionAdjustment
    };
    
    // Продвинутая рекомендация
    const recommendation = generateAdvancedRecommendation(
      direction, probability, predictionConfig.predictionInterval,
      { isHammer, isShootingStar, isDoji, isMarubozu }, volume
    );

    return {
      direction,
      probability: Number(probability.toFixed(1)),
      confidence: Number(confidence.toFixed(1)),
      interval: predictionConfig.predictionInterval,
      factors,
      recommendation,
      metadata: {
        modelAgreement: probability,
        riskScore: 100 - confidence,
        marketCondition: determineMarketCondition(priceRange, volume, bodySize),
        modelBreakdown: [{ name: 'Enhanced Analysis', confidence: probability, weight: 1.0 }],
        patternDetected: getDetectedPattern({ isHammer, isShootingStar, isDoji, isMarubozu, isSpinningTop }),
        volumeAnalysis: analyzeVolume(volume, volumeRelative),
        riskLevel: calculateRiskLevel(volatilityScore, confidence)
      }
    };
  };

  // Вспомогательные функции для глубокого анализа
  const calculatePriceActionScore = (open: number, high: number, low: number, close: number, closePosition: number): number => {
    const openPosition = (open - low) / ((high - low) || 0.0001);
    const positionScore = closePosition > 0.7 ? 75 : closePosition < 0.3 ? 25 : 50;
    const gapScore = Math.abs(close - open) / ((high - low) || 0.0001) * 100;
    return Math.min(95, Math.max(5, (positionScore + gapScore) / 2));
  };

  const calculateVolatilityScore = (priceRange: number, close: number): number => {
    const volatilityRatio = priceRange / (close || 0.0001);
    if (volatilityRatio > 0.02) return 85; // Высокая волатильность
    if (volatilityRatio > 0.01) return 65; // Умеренная волатильность
    if (volatilityRatio > 0.005) return 45; // Низкая волатильность
    return 25; // Очень низкая волатильность
  };

  const calculateMomentumScore = (open: number, close: number, bodySize: number, priceRange: number): number => {
    const direction = close > open ? 1 : -1;
    const strength = bodySize / (priceRange || 0.0001);
    const momentum = direction * strength * 100;
    return Math.min(95, Math.max(5, 50 + momentum * 0.5));
  };

  const calculateTechnicalFactor = (isBullish: boolean, isHammer: boolean, isShootingStar: boolean, 
                                  isDoji: boolean, isMarubozu: boolean, isSpinningTop: boolean): number => {
    let score = isBullish ? 65 : 35;
    
    if (isHammer && !isBullish) score += 25; // Потенциальный разворот
    if (isShootingStar && isBullish) score -= 25; // Потенциальный разворот
    if (isDoji) score = 50; // Неопределенность
    if (isMarubozu) score += isBullish ? 15 : -15; // Сильный тренд
    if (isSpinningTop) score += (Math.random() - 0.5) * 10; // Неопределенность с флуктуацией
    
    return Math.min(95, Math.max(5, score));
  };

  const calculatePatternFactor = (isHammer: boolean, isShootingStar: boolean, isDoji: boolean,
                                isMarubozu: boolean, isSpinningTop: boolean, closePosition: number): number => {
    if (isHammer) return 80;
    if (isShootingStar) return 20;
    if (isDoji) return 50;
    if (isMarubozu) return closePosition > 0.5 ? 75 : 25;
    if (isSpinningTop) return 45;
    return closePosition * 100;
  };

  const calculateTrendFactor = (close: number, middlePoint: number, closePosition: number, isBullish: boolean): number => {
    const positionStrength = Math.abs(closePosition - 0.5) * 2; // 0-1
    const trendAlignment = close > middlePoint ? (isBullish ? 1 : 0.3) : (isBullish ? 0.3 : 1);
    return Math.min(95, Math.max(5, 50 + (positionStrength * trendAlignment * 45)));
  };

  const calculateMarketConditionAdjustment = (volume: number, priceRange: number, bodySize: number, close: number): number => {
    let adjustment = 0;
    
    // Объемная коррекция
    if (volume > 1000000) adjustment += 5;
    if (volume < 100000) adjustment -= 3;
    
    // Волатильность коррекция
    const volatility = priceRange / (close || 0.0001);
    if (volatility > 0.02) adjustment += 3;
    if (volatility < 0.005) adjustment -= 2;
    
    // Активность коррекция
    const bodyRatio = bodySize / (priceRange || 0.0001);
    if (bodyRatio > 0.7) adjustment += 4; // Сильное движение
    if (bodyRatio < 0.2) adjustment -= 2; // Слабое движение
    
    return adjustment;
  };

  const calculateConfidenceModifiers = (volume: number, priceRange: number, bodySize: number,
                                      isHammer: boolean, isDoji: boolean, isMarubozu: boolean): number => {
    let modifier = 0;
    
    if (volume > 500000) modifier += 3;
    if (isHammer || isMarubozu) modifier += 5;
    if (isDoji) modifier -= 3;
    if (bodySize / (priceRange || 0.0001) > 0.8) modifier += 4;
    
    return modifier;
  };

  const generateAdvancedRecommendation = (direction: string, probability: number, interval: number,
                                        patterns: any, volume: number): string => {
    const strength = probability > 80 ? 'Очень сильный' : probability > 70 ? 'Сильный' : probability > 60 ? 'Умеренный' : 'Слабый';
    const action = direction === 'UP' ? 'CALL' : 'PUT';
    const volumeNote = volume > 500000 ? ' при высоком объеме' : volume < 100000 ? ' при низком объеме' : '';
    const patternNote = patterns.isHammer ? ' (паттерн "Молот")' : 
                       patterns.isShootingStar ? ' (паттерн "Падающая звезда")' :
                       patterns.isDoji ? ' (паттерн "Доджи")' :
                       patterns.isMarubozu ? ' (паттерн "Марубозу")' : '';
    
    return `${strength} сигнал для ${action} опциона на ${interval} мин. Вероятность: ${probability.toFixed(1)}%${volumeNote}${patternNote}`;
  };

  const getDetectedPattern = (patterns: any): string => {
    if (patterns.isHammer) return 'Молот';
    if (patterns.isShootingStar) return 'Падающая звезда';
    if (patterns.isDoji) return 'Доджи';
    if (patterns.isMarubozu) return 'Марубозу';
    if (patterns.isSpinningTop) return 'Волчок';
    return 'Стандартная свеча';
  };

  const analyzeVolume = (volume: number, volumeRelative: number): string => {
    if (volume > 1000000) return 'Очень высокий объем';
    if (volume > 500000) return 'Высокий объем';
    if (volume > 200000) return 'Нормальный объем';
    if (volume > 100000) return 'Низкий объем';
    return 'Очень низкий объем';
  };

  const calculateRiskLevel = (volatilityScore: number, confidence: number): string => {
    const riskScore = (100 - confidence) + (volatilityScore > 80 ? 20 : 0);
    if (riskScore > 50) return 'Высокий';
    if (riskScore > 30) return 'Умеренный';
    return 'Низкий';
  };

  const determineMarketCondition = (priceRange: number, volume: number, bodySize: number): string => {
    const volatility = priceRange / bodySize;
    if (volume > 800000 && volatility > 2) return 'Активный волатильный рынок';
    if (volume > 500000) return 'Активный рынок';
    if (volatility > 3) return 'Волатильный рынок';
    if (volume < 200000 && volatility < 1.5) return 'Спокойный рынок';
    return 'Нормальные условия';
  };

  // Обновление результата прогноза (для обучения модели)
  const updatePredictionResult = useCallback((
    predictionId: number, 
    actualOutcome: 'UP' | 'DOWN'
  ) => {
    if (predictionHistory[predictionId]) {
      const prediction = predictionHistory[predictionId];
      prediction.actualOutcome = actualOutcome;
      prediction.result = prediction.prediction.direction === actualOutcome;
      
      // Обновляем модель
      const recentPredictions = predictionHistory.slice(-50); // Последние 50 прогнозов
      predictionService.updateModelWeights(recentPredictions);
      
      // Обновляем точность модели
      const accuratePredictions = recentPredictions.filter(p => p.result && p.actualOutcome).length;
      const totalPredictions = recentPredictions.filter(p => p.actualOutcome).length;
      setModelAccuracy(totalPredictions > 0 ? (accuratePredictions / totalPredictions) * 100 : 0);
    }
  }, []);

  // Получение статистики модели
  const getModelStats = useCallback(() => {
    const completedPredictions = predictionHistory.filter(p => p.actualOutcome);
    const accurateCount = completedPredictions.filter(p => p.result).length;
    
    const callPredictions = completedPredictions.filter(p => p.prediction.direction === 'UP');
    const putPredictions = completedPredictions.filter(p => p.prediction.direction === 'DOWN');
    
    const callAccuracy = callPredictions.length > 0 ? 
      (callPredictions.filter(p => p.result).length / callPredictions.length) * 100 : 0;
    const putAccuracy = putPredictions.length > 0 ? 
      (putPredictions.filter(p => p.result).length / putPredictions.length) * 100 : 0;
    
    return {
      totalPredictions: completedPredictions.length,
      accurateCount,
      overallAccuracy: completedPredictions.length > 0 ? (accurateCount / completedPredictions.length) * 100 : 0,
      callAccuracy,
      putAccuracy,
      currentWeights: predictionService.getModelWeights()
    };
  }, []);

  // Batch генерация прогнозов
  const generateBatchPredictions = useCallback(async (
    candles: CandleData[],
    config: PredictionConfig
  ): Promise<PredictionResult[]> => {
    const results: PredictionResult[] = [];
    
    for (let i = 5; i < candles.length; i++) { // Начинаем с 5-й свечи для истории
      const prediction = await predictionService.generateAdvancedPrediction(
        candles,
        i,
        config
      );
      if (prediction) {
        results.push(prediction);
      }
    }
    
    return results;
  }, []);

  return {
    predictionResult,
    setPredictionResult,
    isGenerating,
    modelAccuracy,
    generatePrediction,
    updatePredictionResult,
    getModelStats,
    generateBatchPredictions
  };
};
