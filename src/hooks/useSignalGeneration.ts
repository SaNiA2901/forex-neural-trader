
import { useState, useEffect } from "react";
import { useNewApplicationState } from "./useNewApplicationState";
import { predictionService } from "@/services/predictionService";
import { TechnicalIndicatorService } from "@/services/indicators/TechnicalIndicators";

interface Signal {
  id: string;
  type: 'CALL' | 'PUT';
  strength: number;
  timeLeft: number;
  reason: string;
  probability: number;
  entry: number;
  technicalBasis: string;
}

export const useSignalGeneration = (pair: string, timeframe: string) => {
  const { currentSession, candles } = useNewApplicationState();
  const [signals, setSignals] = useState<Signal[]>([]);
  const [overallSentiment, setOverallSentiment] = useState<'BULLISH' | 'BEARISH' | 'NEUTRAL'>('NEUTRAL');
  const [confidence, setConfidence] = useState(0);

  useEffect(() => {
    const generateRealSignals = async () => {
      if (!currentSession || candles.length < 10) {
        setSignals([]);
        setOverallSentiment('NEUTRAL');
        setConfidence(0);
        return;
      }

      try {
        const recentCandles = candles.slice(-20); // Последние 20 свечей
        const currentCandle = recentCandles[recentCandles.length - 1];
        const newSignals: Signal[] = [];

        // Анализируем множественные таймфреймы
        const timeframes = [5, 15, 30, 60];
        const predictions = [];

        for (const interval of timeframes) {
          const prediction = await predictionService.generateAdvancedPrediction(
            recentCandles,
            recentCandles.length - 1,
            { predictionInterval: interval, analysisMode: 'session' as const }
          );
          
          if (prediction && prediction.probability > 65) {
            predictions.push({ ...prediction, interval });
          }
        }

        // Технический анализ
        const technicalIndicators = TechnicalIndicatorService.calculateAll(recentCandles, recentCandles.length - 1);
        
        // Генерируем сигналы на основе реального анализа
        predictions.forEach((prediction, index) => {
          const signal: Signal = {
            id: `signal-${index}-${Date.now()}`,
            type: prediction.direction === 'UP' ? 'CALL' : 'PUT',
            strength: prediction.confidence,
            timeLeft: prediction.interval,
            reason: generateTechnicalReason(prediction, technicalIndicators),
            probability: prediction.probability,
            entry: currentCandle.close,
            technicalBasis: describeTechnicalSetup(prediction, technicalIndicators)
          };
          
          newSignals.push(signal);
        });

        // Дополнительные сигналы на основе технических индикаторов
        const additionalSignals = generateIndicatorBasedSignals(recentCandles, technicalIndicators, currentCandle);
        newSignals.push(...additionalSignals);

        // Фильтруем дублирующиеся сигналы
        const uniqueSignals = filterDuplicateSignals(newSignals);
        
        setSignals(uniqueSignals.slice(0, 5)); // Максимум 5 сигналов

        // Рассчитываем общий sentiment на основе реальных данных
        const { sentiment, confidenceLevel } = calculateMarketSentiment(uniqueSignals, technicalIndicators);
        setOverallSentiment(sentiment);
        setConfidence(confidenceLevel);

      } catch (error) {
        console.error('Error generating real signals:', error);
        setSignals([]);
        setOverallSentiment('NEUTRAL');
        setConfidence(0);
      }
    };

    generateRealSignals();
    const interval = setInterval(generateRealSignals, 30000); // Обновляем каждые 30 секунд
    return () => clearInterval(interval);
  }, [currentSession, candles, pair, timeframe]);

  const generateTechnicalReason = (prediction: any, indicators: any): string => {
    const reasons = [];
    
    if (indicators.rsi < 30) reasons.push('RSI показывает перепроданность');
    if (indicators.rsi > 70) reasons.push('RSI показывает перекупленность');
    if (indicators.macd.histogram > 0 && prediction.direction === 'UP') reasons.push('MACD бычий кроссовер');
    if (indicators.macd.histogram < 0 && prediction.direction === 'DOWN') reasons.push('MACD медвежий кроссовер');
    if (indicators.bollingerBands && prediction.direction === 'UP') reasons.push('Отскок от нижней полосы Боллинжера');
    if (indicators.bollingerBands && prediction.direction === 'DOWN') reasons.push('Отскок от верхней полосы Боллинжера');
    
    return reasons.length > 0 ? reasons[Math.floor(Math.random() * reasons.length)] : 
           'Анализ ценового действия и объемов';
  };

  const describeTechnicalSetup = (prediction: any, indicators: any): string => {
    const setups = [];
    
    if (indicators.rsi) setups.push(`RSI: ${indicators.rsi.toFixed(1)}`);
    if (indicators.macd) setups.push(`MACD: ${indicators.macd.histogram.toFixed(4)}`);
    if (indicators.stochastic) setups.push(`Stoch: ${indicators.stochastic.k.toFixed(1)}`);
    
    return setups.join(' | ');
  };

  const generateIndicatorBasedSignals = (candles: any[], indicators: any, currentCandle: any): Signal[] => {
    const signals: Signal[] = [];
    
    // RSI экстремальные значения
    if (indicators.rsi < 25 || indicators.rsi > 75) {
      signals.push({
        id: `rsi-signal-${Date.now()}`,
        type: indicators.rsi < 25 ? 'CALL' : 'PUT',
        strength: Math.abs(50 - indicators.rsi) + 50,
        timeLeft: 15,
        reason: indicators.rsi < 25 ? 'Крайняя перепроданность по RSI' : 'Крайняя перекупленность по RSI',
        probability: Math.min(90, Math.abs(50 - indicators.rsi) + 65),
        entry: currentCandle.close,
        technicalBasis: `RSI: ${indicators.rsi.toFixed(1)}`
      });
    }

    // Стохастик сигналы
    if (indicators.stochastic) {
      const { k, d } = indicators.stochastic;
      if ((k < 20 && k > d) || (k > 80 && k < d)) {
        signals.push({
          id: `stoch-signal-${Date.now()}`,
          type: k < 20 ? 'CALL' : 'PUT',
          strength: Math.abs(k - d) * 2 + 60,
          timeLeft: 30,
          reason: k < 20 ? 'Бычий кроссовер в зоне перепроданности' : 'Медвежий кроссовер в зоне перекупленности',
          probability: Math.min(85, Math.abs(k - d) * 2 + 70),
          entry: currentCandle.close,
          technicalBasis: `Stoch K: ${k.toFixed(1)}, D: ${d.toFixed(1)}`
        });
      }
    }

    return signals;
  };

  const filterDuplicateSignals = (signals: Signal[]): Signal[] => {
    const seen = new Set();
    return signals.filter(signal => {
      const key = `${signal.type}-${signal.timeLeft}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  };

  const calculateMarketSentiment = (signals: Signal[], indicators: any) => {
    if (signals.length === 0) {
      return { sentiment: 'NEUTRAL' as const, confidenceLevel: 0 };
    }

    const callSignals = signals.filter(s => s.type === 'CALL');
    const putSignals = signals.filter(s => s.type === 'PUT');
    
    const callStrength = callSignals.reduce((sum, s) => sum + s.strength, 0);
    const putStrength = putSignals.reduce((sum, s) => sum + s.strength, 0);
    
    let sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
    if (callStrength > putStrength * 1.2) {
      sentiment = 'BULLISH';
    } else if (putStrength > callStrength * 1.2) {
      sentiment = 'BEARISH';
    } else {
      sentiment = 'NEUTRAL';
    }

    // Базовая уверенность на основе силы сигналов
    const avgStrength = signals.reduce((sum, s) => sum + s.strength, 0) / signals.length;
    let confidenceLevel = Math.round(avgStrength);

    // Корректировки на основе технических индикаторов
    if (indicators.rsi < 30 || indicators.rsi > 70) confidenceLevel += 10;
    if (indicators.macd && Math.abs(indicators.macd.histogram) > 0.001) confidenceLevel += 5;

    return { sentiment, confidenceLevel: Math.min(95, Math.max(50, confidenceLevel)) };
  };

  return { signals, overallSentiment, confidence };
};
