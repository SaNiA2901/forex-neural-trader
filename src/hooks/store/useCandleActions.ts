import { useCallback } from 'react';
import { useTradingStore } from '@/store/TradingStore';
import { sessionController } from '@/controllers/SessionController';
import { CandleData } from '@/types/session';

export const useCandleActions = () => {
  const { dispatch } = useTradingStore();

  // Save single candle
  const saveCandle = useCallback(async (candleData: CandleData) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'CLEAR_ERRORS' });
      
      const savedCandle = await sessionController.saveCandle(candleData);
      dispatch({ type: 'ADD_CANDLE', payload: savedCandle });
      
      return savedCandle;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to save candle';
      dispatch({ type: 'ADD_ERROR', payload: errorMessage });
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [dispatch]);

  // Update candle
  const updateCandle = useCallback(async (candleIndex: number, updates: Partial<CandleData>) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'CLEAR_ERRORS' });
      
      const updatedCandle = await sessionController.updateCandle(candleIndex, updates);
      dispatch({ type: 'UPDATE_CANDLE', payload: { index: candleIndex, updates } });
      
      return updatedCandle;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to update candle';
      dispatch({ type: 'ADD_ERROR', payload: errorMessage });
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [dispatch]);

  // Delete candle
  const deleteCandle = useCallback(async (candleIndex: number) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'CLEAR_ERRORS' });
      
      await sessionController.deleteCandle(candleIndex);
      dispatch({ type: 'DELETE_CANDLE', payload: candleIndex });
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to delete candle';
      dispatch({ type: 'ADD_ERROR', payload: errorMessage });
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [dispatch]);

  // Save multiple candles (batch operation)
  const saveCandlesBatch = useCallback(async (candles: CandleData[]) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'CLEAR_ERRORS' });
      
      const savedCandles = await sessionController.saveCandlesBatch(candles);
      
      // Добавляем все свечи в состояние
      savedCandles.forEach(candle => {
        dispatch({ type: 'ADD_CANDLE', payload: candle });
      });
      
      return savedCandles;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to save candles batch';
      dispatch({ type: 'ADD_ERROR', payload: errorMessage });
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [dispatch]);

  // Delete last candle (convenience method)
  const deleteLastCandle = useCallback(async () => {
    const { state } = useTradingStore();
    
    if (state.candles.length === 0) {
      throw new Error('No candles to delete');
    }

    const lastCandle = state.candles.reduce((latest, current) => 
      current.candle_index > latest.candle_index ? current : latest
    );

    return await deleteCandle(lastCandle.candle_index);
  }, [deleteCandle]);

  // Clear all candles (for current session)
  const clearCandles = useCallback(() => {
    dispatch({ type: 'CLEAR_CANDLES' });
  }, [dispatch]);

  return {
    saveCandle,
    updateCandle,
    deleteCandle,
    saveCandlesBatch,
    deleteLastCandle,
    clearCandles
  };
};