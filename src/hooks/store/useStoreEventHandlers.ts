import { useEffect } from 'react';
import { useTradingStore } from '@/store/TradingStore';
import { sessionController } from '@/controllers/SessionController';

export const useStoreEventHandlers = () => {
  const { dispatch } = useTradingStore();

  useEffect(() => {
    // Session Events
    const handleSessionCreated = (session: any) => {
      console.log('📝 Session created:', session.session_name);
    };

    const handleSessionLoaded = (data: any) => {
      console.log('📂 Session loaded:', data.session.session_name, 'with', data.candles.length, 'candles');
    };

    const handleSessionDeleted = (sessionId: string) => {
      console.log('🗑️ Session deleted:', sessionId);
    };

    const handleSessionError = (error: any) => {
      console.error('❌ Session error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown session error';
      dispatch({ type: 'ADD_ERROR', payload: errorMessage });
    };

    // Candle Events
    const handleCandleSaved = (candle: any) => {
      console.log('💡 Candle saved:', candle.candle_index);
    };

    const handleCandleUpdated = (candle: any) => {
      console.log('✏️ Candle updated:', candle.candle_index);
    };

    const handleCandleDeleted = (candleIndex: number) => {
      console.log('🗑️ Candle deleted:', candleIndex);
    };

    const handleCandleError = (error: any) => {
      console.error('❌ Candle error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown candle error';
      dispatch({ type: 'ADD_ERROR', payload: errorMessage });
    };

    // Loading Events
    const handleLoadingStart = (isLoading: boolean) => {
      dispatch({ type: 'SET_LOADING', payload: isLoading });
    };

    // Batch Events
    const handleBatchSaved = (candles: any[]) => {
      console.log('📦 Batch saved:', candles.length, 'candles');
    };

    const handleBatchError = (error: any) => {
      console.error('❌ Batch error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Batch operation failed';
      dispatch({ type: 'ADD_ERROR', payload: errorMessage });
    };

    // Cache Events
    const handleCacheCleared = () => {
      console.log('🧹 Cache cleared');
    };

    // Auto-save Events
    const handleAutoSave = (sessionId: string) => {
      console.log('💾 Auto-save triggered for session:', sessionId);
    };

    // Connection Events
    const handleConnectionLost = () => {
      dispatch({ type: 'SET_CONNECTION_STATUS', payload: false });
      dispatch({ type: 'ADD_ERROR', payload: 'Connection lost. Please check your internet connection.' });
    };

    const handleConnectionRestored = () => {
      dispatch({ type: 'SET_CONNECTION_STATUS', payload: true });
      console.log('🔗 Connection restored');
    };

    // Register event listeners
    sessionController.on('session.created', handleSessionCreated);
    sessionController.on('session.loaded', handleSessionLoaded);
    sessionController.on('session.deleted', handleSessionDeleted);
    sessionController.on('session.error', handleSessionError);
    sessionController.on('sessions.loading', handleLoadingStart);

    sessionController.on('candle.saved', handleCandleSaved);
    sessionController.on('candle.updated', handleCandleUpdated);
    sessionController.on('candle.deleted', handleCandleDeleted);
    sessionController.on('candle.error', handleCandleError);

    sessionController.on('candles.batch.saved', handleBatchSaved);
    sessionController.on('candles.batch.error', handleBatchError);

    sessionController.on('cache.cleared', handleCacheCleared);
    sessionController.on('session.autosave', handleAutoSave);

    // Connection monitoring
    const handleOnline = () => handleConnectionRestored();
    const handleOffline = () => handleConnectionLost();

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Cleanup
    return () => {
      sessionController.off('session.created', handleSessionCreated);
      sessionController.off('session.loaded', handleSessionLoaded);
      sessionController.off('session.deleted', handleSessionDeleted);
      sessionController.off('session.error', handleSessionError);
      sessionController.off('sessions.loading', handleLoadingStart);

      sessionController.off('candle.saved', handleCandleSaved);
      sessionController.off('candle.updated', handleCandleUpdated);
      sessionController.off('candle.deleted', handleCandleDeleted);
      sessionController.off('candle.error', handleCandleError);

      sessionController.off('candles.batch.saved', handleBatchSaved);
      sessionController.off('candles.batch.error', handleBatchError);

      sessionController.off('cache.cleared', handleCacheCleared);
      sessionController.off('session.autosave', handleAutoSave);

      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [dispatch]);

  // Performance monitoring
  useEffect(() => {
    const startTime = performance.now();
    
    return () => {
      const endTime = performance.now();
      console.log(`⏱️ Store event handlers active for ${(endTime - startTime).toFixed(2)}ms`);
    };
  }, []);
};