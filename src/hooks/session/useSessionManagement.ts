import { useCallback } from 'react';
import { useTradingStore } from '@/store/TradingStore';
import { sessionService } from '@/services/sessionService';
import { TradingSession, CandleData } from '@/types/session';
import { useErrorHandler } from '@/hooks/useErrorHandler';

interface SessionCreationData {
  session_name: string;
  pair: string;
  timeframe: string;
  start_date: string;
  start_time: string;
}

/**
 * Unified hook for session management operations
 * Combines initialization, operations, and state management
 */
export const useSessionManagement = () => {
  const { state, dispatch } = useTradingStore();
  const { addError } = useErrorHandler();

  // State setters through TradingStore
  const setSessions = useCallback((sessions: TradingSession[]) => {
    dispatch({ type: 'SET_SESSIONS', payload: sessions });
  }, [dispatch]);

  const setCurrentSession = useCallback((session: TradingSession | null) => {
    dispatch({ type: 'SET_CURRENT_SESSION', payload: session });
  }, [dispatch]);

  const setIsLoading = useCallback((loading: boolean) => {
    dispatch({ type: 'SET_LOADING', payload: loading });
  }, [dispatch]);

  const updateCandles = useCallback((updater: (prev: CandleData[]) => CandleData[]) => {
    const newCandles = updater(state.candles);
    dispatch({ type: 'SET_CANDLES', payload: newCandles });
  }, [dispatch, state.candles]);

  const resetSessionState = useCallback(() => {
    dispatch({ type: 'SET_CURRENT_SESSION', payload: null });
    dispatch({ type: 'CLEAR_CANDLES' });
    dispatch({ type: 'CLEAR_PREDICTIONS' });
  }, [dispatch]);

  // Session operations
  const loadSessions = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await sessionService.loadSessions();
      setSessions(data);
      console.log('Sessions loaded:', data.length);
      return data;
    } catch (error) {
      console.error('Failed to load sessions:', error);
      addError('Ошибка загрузки сессий', error instanceof Error ? error.message : 'Unknown error');
      setSessions([]);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, [setSessions, setIsLoading, addError]);

  const createSession = useCallback(async (sessionData: SessionCreationData) => {
    setIsLoading(true);
    try {
      // Валидация данных перед отправкой
      if (!sessionData.session_name?.trim()) {
        throw new Error('Название сессии обязательно');
      }
      if (!sessionData.pair?.trim()) {
        throw new Error('Валютная пара обязательна');
      }
      if (!sessionData.timeframe?.trim()) {
        throw new Error('Таймфрейм обязателен');
      }
      if (!sessionData.start_date?.trim()) {
        throw new Error('Дата начала обязательна');
      }
      if (!sessionData.start_time?.trim()) {
        throw new Error('Время начала обязательно');
      }

      const session = await sessionService.createSession(sessionData);
      setCurrentSession(session);
      updateCandles(() => []);
      console.log('✅ Session created successfully:', session.id);
      
      // Refresh sessions list
      await loadSessions();
      return session;
    } catch (error) {
      console.error('❌ Failed to create session:', error);
      addError('Ошибка создания сессии', error instanceof Error ? error.message : 'Unknown error');
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [setIsLoading, setCurrentSession, updateCandles, addError, loadSessions]);

  const loadSession = useCallback(async (sessionId: string) => {
    setIsLoading(true);
    try {
      const result = await sessionService.loadSessionWithCandles(sessionId);
      updateCandles(() => result.candles);
      setCurrentSession(result.session);
      console.log('Session loaded:', sessionId, 'with', result.candles.length, 'candles');
      return result.session;
    } catch (error) {
      console.error('Failed to load session:', error);
      addError('Ошибка загрузки сессии', error instanceof Error ? error.message : 'Unknown error');
      setCurrentSession(null);
      updateCandles(() => []);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [setIsLoading, setCurrentSession, updateCandles, addError]);

  const deleteSession = useCallback(async (sessionId: string) => {
    setIsLoading(true);
    try {
      await sessionService.deleteSession(sessionId);
      setCurrentSession(null);
      console.log('Session deleted:', sessionId);
      
      // Refresh sessions list
      await loadSessions();
    } catch (error) {
      console.error('Failed to delete session:', error);
      addError('Ошибка удаления сессии', error instanceof Error ? error.message : 'Unknown error');
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [setIsLoading, setCurrentSession, addError, loadSessions]);

  // Data integrity validation
  const validateDataIntegrity = useCallback(() => {
    if (!state.currentSession) return null;

    const actualCandleCount = state.candles.length;
    const sessionIndex = state.currentSession.current_candle_index;
    const expectedMaxIndex = actualCandleCount > 0 
      ? Math.max(...state.candles.map((c: any) => c.candle_index))
      : 0;

    const isConsistent = sessionIndex >= expectedMaxIndex;

    return {
      isConsistent,
      actualCandleCount,
      sessionIndex,
      expectedMaxIndex,
      discrepancy: sessionIndex - expectedMaxIndex
    };
  }, [state.currentSession, state.candles]);

  // Sync state with database
  const syncStateWithDB = useCallback(async () => {
    if (!state.currentSession) return;

    try {
      console.log('🔄 Синхронизация состояния с БД...');
      setIsLoading(true);

      await loadSession(state.currentSession.id);
      
      console.log('✅ Синхронизация завершена');
    } catch (error) {
      console.error('Ошибка синхронизации:', error);
      dispatch({ type: 'ADD_ERROR', payload: 'Ошибка синхронизации с базой данных' });
    } finally {
      setIsLoading(false);
    }
  }, [state.currentSession, setIsLoading, loadSession, dispatch]);

  return {
    // State from TradingStore
    sessions: state.sessions,
    currentSession: state.currentSession,
    candles: state.candles,
    isLoading: state.isLoading,
    sessionStats: state.sessionStats,
    nextCandleIndex: state.nextCandleIndex,
    
    // Operations
    loadSessions,
    createSession,
    loadSession,
    deleteSession,
    resetSessionState,
    updateCandles,
    setCurrentSession,
    
    // Advanced operations
    syncStateWithDB,
    validateDataIntegrity
  };
};