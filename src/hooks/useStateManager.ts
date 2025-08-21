import { useCallback } from 'react';
import { useNewApplicationState } from './useNewApplicationState';

/**
 * Унифицированный хук для управления состоянием приложения
 * Заменяет старые useApplicationState и useNewApplicationState
 */
export const useStateManager = () => {
  const appState = useNewApplicationState();

  // Дополнительные утилиты для работы с состоянием
  const getSessionById = useCallback((sessionId: string) => {
    return appState.sessions.find(session => session.id === sessionId) || null;
  }, [appState.sessions]);

  const hasUnsavedChanges = useCallback(() => {
    // Логика проверки несохраненных изменений
    return false; // Заглушка
  }, []);

  const resetAppState = useCallback(async () => {
    try {
      await appState.resetSessionState();
      console.log('Состояние приложения сброшено');
    } catch (error) {
      console.error('Ошибка сброса состояния:', error);
    }
  }, [appState]);

  // Статистика и метрики
  const getAppStatistics = useCallback(() => {
    return {
      totalSessions: appState.sessions.length,
      totalCandles: appState.candles.length,
      currentSessionCandles: appState.currentSession 
        ? appState.candles.filter(c => c.session_id === appState.currentSession!.id).length
        : 0,
      predictionsCount: appState.candles.filter(c => c.prediction_direction).length,
      lastActivity: appState.currentSession?.updated_at || null
    };
  }, [appState.sessions, appState.candles, appState.currentSession]);

  return {
    // Передаем все методы из useNewApplicationState
    ...appState,
    
    // Дополнительные утилиты
    getSessionById,
    hasUnsavedChanges,
    resetAppState,
    getAppStatistics
  };
};