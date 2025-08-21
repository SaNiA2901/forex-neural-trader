import { useCallback } from 'react';
import { useTradingStore } from '@/store/TradingStore';
import { sessionController } from '@/controllers/SessionController';
import { TradingSession, CandleData } from '@/types/session';

export const useSessionActions = () => {
  const { dispatch } = useTradingStore();

  // Load all sessions
  const loadSessions = useCallback(async () => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'CLEAR_ERRORS' });
      
      const sessions = await sessionController.loadSessions();
      dispatch({ type: 'SET_SESSIONS', payload: sessions });
      
      return sessions;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load sessions';
      dispatch({ type: 'ADD_ERROR', payload: errorMessage });
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [dispatch]);

  // Create new session
  const createSession = useCallback(async (sessionData: {
    session_name: string;
    pair: string;
    timeframe: string;
    start_date: string;
    start_time: string;
  }) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'CLEAR_ERRORS' });
      
      const session = await sessionController.createSession(sessionData);
      dispatch({ type: 'ADD_SESSION', payload: session });
      dispatch({ type: 'SET_CURRENT_SESSION', payload: session });
      dispatch({ type: 'CLEAR_CANDLES' });
      
      return session;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to create session';
      dispatch({ type: 'ADD_ERROR', payload: errorMessage });
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [dispatch]);

  // Load session with candles
  const loadSession = useCallback(async (sessionId: string) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'CLEAR_ERRORS' });
      
      const result = await sessionController.loadSession(sessionId);
      dispatch({ type: 'SET_CURRENT_SESSION', payload: result.session });
      dispatch({ type: 'SET_CANDLES', payload: result.candles });
      
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load session';
      dispatch({ type: 'ADD_ERROR', payload: errorMessage });
      dispatch({ type: 'SET_CURRENT_SESSION', payload: null });
      dispatch({ type: 'CLEAR_CANDLES' });
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [dispatch]);

  // Delete session
  const deleteSession = useCallback(async (sessionId: string) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'CLEAR_ERRORS' });
      
      await sessionController.deleteSession(sessionId);
      dispatch({ type: 'DELETE_SESSION', payload: sessionId });
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to delete session';
      dispatch({ type: 'ADD_ERROR', payload: errorMessage });
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [dispatch]);

  // Duplicate session
  const duplicateSession = useCallback(async (sessionId: string, newName: string) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'CLEAR_ERRORS' });
      
      const newSession = await sessionController.duplicateSession(sessionId, newName);
      dispatch({ type: 'ADD_SESSION', payload: newSession });
      
      return newSession;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to duplicate session';
      dispatch({ type: 'ADD_ERROR', payload: errorMessage });
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [dispatch]);

  // Set current session
  const setCurrentSession = useCallback((session: TradingSession | null) => {
    sessionController.setCurrentSession(session);
    dispatch({ type: 'SET_CURRENT_SESSION', payload: session });
    
    if (!session) {
      dispatch({ type: 'CLEAR_CANDLES' });
    }
  }, [dispatch]);

  // Reset session state
  const resetSessionState = useCallback(() => {
    sessionController.setCurrentSession(null);
    dispatch({ type: 'SET_CURRENT_SESSION', payload: null });
    dispatch({ type: 'CLEAR_CANDLES' });
    dispatch({ type: 'CLEAR_PREDICTIONS' });
  }, [dispatch]);

  return {
    loadSessions,
    createSession,
    loadSession,
    deleteSession,
    duplicateSession,
    setCurrentSession,
    resetSessionState
  };
};