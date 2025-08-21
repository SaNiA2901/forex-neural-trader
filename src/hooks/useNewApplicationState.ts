import { useCallback } from 'react';
import { useSessionSelector, useCandleSelector, usePredictionSelector, useUISelector } from '@/store/TradingStore';
import { useSessionActions } from './store/useSessionActions';
import { useCandleActions } from './store/useCandleActions';
import { useStoreEventHandlers } from './store/useStoreEventHandlers';

/**
 * Новый централизованный хук состояния приложения
 * Заменяет старый useApplicationState с улучшенной архитектурой
 */
export const useNewApplicationState = () => {
  // Инициализация обработчиков событий
  useStoreEventHandlers();

  // Селекторы состояния
  const { sessions, currentSession, sessionStats } = useSessionSelector();
  const { candles, nextCandleIndex } = useCandleSelector();
  const { lastPrediction, predictionHistory, performance } = usePredictionSelector();
  const { isLoading, errors, isConnected } = useUISelector();

  // Actions
  const sessionActions = useSessionActions();
  const candleActions = useCandleActions();

  // Дополнительные удобные методы
  const getLastCandle = useCallback(() => {
    if (candles.length === 0) return null;
    return candles.reduce((latest, current) => 
      current.candle_index > latest.candle_index ? current : latest
    );
  }, [candles]);

  const getCandleByIndex = useCallback((index: number) => {
    return candles.find(candle => candle.candle_index === index) || null;
  }, [candles]);

  const getRecentCandles = useCallback((count: number = 10) => {
    return candles.slice(-count);
  }, [candles]);

  const getCandlesInRange = useCallback((startIndex: number, endIndex: number) => {
    return candles.filter(candle => 
      candle.candle_index >= startIndex && candle.candle_index <= endIndex
    );
  }, [candles]);

  // Статистика и аналитика
  const getSessionAnalytics = useCallback(() => {
    if (!currentSession) return null;

    const sessionCandles = candles.filter(c => c.session_id === currentSession.id);
    
    return {
      totalCandles: sessionCandles.length,
      predictionsCount: sessionCandles.filter(c => c.prediction_direction).length,
      averageAccuracy: performance.accuracy,
      lastPrediction,
      predictionHistory: predictionHistory.slice(0, 10) // Последние 10
    };
  }, [currentSession, candles, performance, lastPrediction, predictionHistory]);

  return {
    // Состояние
    sessions,
    currentSession,
    sessionStats,
    candles,
    nextCandleIndex,
    lastPrediction,
    predictionHistory,
    performance,
    isLoading,
    errors,
    isConnected,

    // Операции с сессиями
    ...sessionActions,

    // Операции со свечами
    ...candleActions,

    // Удобные методы
    getLastCandle,
    getCandleByIndex,
    getRecentCandles,
    getCandlesInRange,
    getSessionAnalytics
  };
};