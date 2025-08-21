
import { useEffect, useMemo, useCallback } from 'react';
import { TradingSession, CandleData } from '@/types/session';
import { useCandleManagement } from './useCandleManagement';
import { useNewApplicationState } from '@/hooks/useNewApplicationState';

interface UseCandleInputLogicProps {
  currentSession: TradingSession | null;
  onCandleSaved?: (candleData: CandleData) => Promise<void>;
}

export const useCandleInputLogic = ({
  currentSession,
  onCandleSaved
}: UseCandleInputLogicProps) => {
  const { candles, nextCandleIndex, saveCandle, deleteLastCandle } = useNewApplicationState();
  
  const {
    formData,
    formErrors: errors,
    isFormValid: isValid,
    isSubmitting: formSubmitting,
    handleInputChange,
    handleFormSubmit,
    resetForm: reset,
    validateForm
  } = useCandleManagement(currentSession, () => {}, () => {});

  const isSubmitting = formSubmitting;

  const lastCandle = useMemo(() => {
    if (candles.length === 0) return null;
    return candles.reduce((latest, current) => 
      current.candle_index > latest.candle_index ? current : latest
    );
  }, [candles]);

  // Мемоизируем обработчики для предотвращения повторных рендеров
  const handleSave = useCallback(async () => {
    if (!currentSession) {
      console.error('Нет активной сессии для сохранения данных');
      return;
    }
    
    try {
      const candleData = await handleFormSubmit(nextCandleIndex);
      if (candleData) {
        console.log('Saving candle with data:', candleData);
        const savedCandle = await saveCandle(candleData);
        console.log('Candle saved successfully:', savedCandle);
        
        // Вызываем колбэк после успешного сохранения
        if (onCandleSaved) {
          await onCandleSaved(savedCandle);
        }
        
        reset();
      }
    } catch (error) {
      console.error('Error saving candle:', error);
      throw error;
    }
  }, [handleFormSubmit, saveCandle, reset, currentSession, onCandleSaved, nextCandleIndex]);

  const handleDeleteLast = useCallback(async () => {
    if (lastCandle && window.confirm('Удалить последнюю свечу?')) {
      try {
        await deleteLastCandle();
        console.log('Last candle deleted successfully');
      } catch (error) {
        console.error('Error deleting last candle:', error);
        throw error;
      }
    }
  }, [lastCandle, deleteLastCandle]);

  // Автозаполнение цены открытия
  useEffect(() => {
    if (lastCandle && !formData.open && !isSubmitting) {
      handleInputChange('open', lastCandle.close.toString());
    }
  }, [lastCandle?.close, formData.open, handleInputChange, isSubmitting]);

  return {
    formData,
    errors,
    isSubmitting,
    isFormValid: isValid,
    lastCandle,
    updateField: handleInputChange,
    handleSave,
    handleDeleteLast,
    reset,
    nextCandleIndex,
    candles
  };
};
