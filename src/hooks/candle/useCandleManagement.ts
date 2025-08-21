import { useState, useCallback, useMemo } from 'react';
import { CandleData, TradingSession } from '@/types/session';
import { candleService } from '@/services/candleService';
import { sessionService } from '@/services/sessionService';
import { calculateCandleDateTime } from '@/utils/dateTimeUtils';
import { validateCandleData } from '@/utils/candleValidation';
import { useErrorHandler } from '@/hooks/useErrorHandler';

interface CandleFormData {
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
  timestamp: string;
  spread?: string;
}

interface CandleFormState {
  formData: CandleFormData;
  errors: Partial<Record<keyof CandleFormData, string>>;
  isValid: boolean;
  isSubmitting: boolean;
}

/**
 * Unified hook for candle management operations
 * Combines form handling, validation, and CRUD operations
 */
export const useCandleManagement = (
  currentSession: TradingSession | null,
  updateCandles: (updater: (prev: CandleData[]) => CandleData[]) => void,
  setCurrentSession: (session: TradingSession | null) => void
) => {
  const { addError } = useErrorHandler();
  
  // Form state
  const [formState, setFormState] = useState<CandleFormState>({
    formData: {
      open: '',
      high: '',
      low: '',
      close: '',
      volume: '',
      timestamp: new Date().toISOString().slice(0, 16),
      spread: ''
    },
    errors: {},
    isValid: false,
    isSubmitting: false
  });

  // Form validation
  const validateField = useCallback((field: keyof CandleFormData, value: string) => {
    const errors: string[] = [];

    if (!value && ['open', 'high', 'low', 'close', 'volume'].includes(field)) {
      errors.push('Обязательное поле');
    }

    if (value && ['open', 'high', 'low', 'close', 'volume', 'spread'].includes(field)) {
      const num = Number(value);
      if (isNaN(num) || num < 0) {
        errors.push('Должно быть положительное число');
      }
    }

    return errors[0] || '';
  }, []);

  const validateForm = useCallback((formData: CandleFormData) => {
    const errors: Partial<Record<keyof CandleFormData, string>> = {};

    // Individual field validation
    Object.keys(formData).forEach(key => {
      const field = key as keyof CandleFormData;
      const error = validateField(field, formData[field] || '');
      if (error) errors[field] = error;
    });

    // OHLC logic validation
    if (formData.open && formData.high && formData.low && formData.close) {
      const o = Number(formData.open);
      const h = Number(formData.high);
      const l = Number(formData.low);
      const c = Number(formData.close);

      if (h < Math.max(o, c)) {
        errors.high = 'High должен быть >= max(Open, Close)';
      }
      if (l > Math.min(o, c)) {
        errors.low = 'Low должен быть <= min(Open, Close)';
      }
      if (h < l) {
        errors.high = 'High должен быть >= Low';
      }
    }

    const isValid = Object.keys(errors).length === 0 && 
                   !!formData.open && !!formData.high && !!formData.low && 
                   !!formData.close && !!formData.volume;

    return { errors, isValid };
  }, [validateField]);

  // Form input handler
  const handleInputChange = useCallback((field: keyof CandleFormData, value: string) => {
    setFormState(prev => {
      const newFormData = { ...prev.formData, [field]: value };
      const { errors, isValid } = validateForm(newFormData);
      
      return {
        ...prev,
        formData: newFormData,
        errors,
        isValid
      };
    });
  }, [validateForm]);

  // CRUD Operations
  const saveCandle = useCallback(async (candleData: Omit<CandleData, 'id' | 'candle_datetime'>) => {
    if (!currentSession) {
      throw new Error('Нет активной сессии для сохранения данных');
    }

    const validation = validateCandleData(candleData);
    if (!validation.isValid) {
      throw new Error(`Ошибки валидации: ${validation.errors.join(', ')}`);
    }

    try {
      const candleDateTime = calculateCandleDateTime(
        currentSession.start_date,
        currentSession.start_time,
        currentSession.timeframe,
        candleData.candle_index
      );

      const fullCandleData = {
        ...candleData,
        candle_datetime: candleDateTime
      };

      console.log('💾 Сохраняем свечу в БД:', fullCandleData);
      const savedCandle = await candleService.saveCandle(fullCandleData);

      if (!savedCandle) {
        throw new Error('Не удалось сохранить свечу');
      }

      // Update local candles state
      updateCandles(prev => {
        const filtered = prev.filter(c => c.candle_index !== candleData.candle_index);
        const newCandles = [...filtered, savedCandle].sort((a, b) => a.candle_index - b.candle_index);
        
        console.log(`🕯️ Локальное обновление: ${prev.length} -> ${newCandles.length} свечей`);
        return newCandles;
      });

      // Update session index
      const newCandleIndex = Math.max(
        currentSession.current_candle_index, 
        candleData.candle_index
      );
      
      console.log(`📈 Обновляем индекс сессии: ${currentSession.current_candle_index} -> ${newCandleIndex}`);
      
      await sessionService.updateSessionCandleIndex(currentSession.id, newCandleIndex);

      // Full sync after save
      try {
        const syncResult = await sessionService.loadSessionWithCandles(currentSession.id);
        console.log(`📊 Полная синхронизация: сессия загружена с ${syncResult.candles.length} свечами`);
        
        setCurrentSession(syncResult.session);
        updateCandles(() => syncResult.candles);
        
      } catch (syncError) {
        console.warn('⚠️ Ошибка полной синхронизации, используем fallback:', syncError);
        setCurrentSession({
          ...currentSession,
          current_candle_index: newCandleIndex,
          updated_at: new Date().toISOString()
        });
      }

      return savedCandle;
    } catch (error) {
      console.error('❌ Ошибка в saveCandle:', error);
      addError('Ошибка сохранения свечи', error instanceof Error ? error.message : 'Unknown error');
      throw error;
    }
  }, [currentSession, updateCandles, setCurrentSession, addError]);

  const deleteCandle = useCallback(async (candleIndex: number) => {
    if (!currentSession) {
      throw new Error('Нет активной сессии');
    }
    
    try {
      console.log(`🗑️ Удаляем свечу ${candleIndex} из БД...`);
      await candleService.deleteCandle(currentSession.id, candleIndex);
      
      updateCandles(prev => {
        const filtered = prev.filter(c => c.candle_index !== candleIndex);
        console.log(`🗑️ Локальное удаление: ${prev.length} -> ${filtered.length} свечей`);
        return filtered;
      });

      // Full sync after deletion
      try {
        const syncResult = await sessionService.loadSessionWithCandles(currentSession.id);
        console.log(`📊 Синхронизация после удаления: ${syncResult.candles.length} свечей`);
        
        setCurrentSession(syncResult.session);
        updateCandles(() => syncResult.candles);
        
      } catch (syncError) {
        console.warn('⚠️ Ошибка синхронизации после удаления:', syncError);
      }
      
      console.log('✅ Свеча удалена успешно:', candleIndex);
    } catch (error) {
      console.error('❌ Ошибка удаления свечи:', error);
      addError('Ошибка удаления свечи', error instanceof Error ? error.message : 'Unknown error');
      throw error;
    }
  }, [currentSession, updateCandles, setCurrentSession, addError]);

  const updateCandle = useCallback(async (candleIndex: number, updatedData: Partial<CandleData>) => {
    if (!currentSession) {
      throw new Error('Нет активной сессии');
    }
    
    try {
      console.log(`✏️ Обновляем свечу ${candleIndex}:`, updatedData);
      const updatedCandle = await candleService.updateCandle(currentSession.id, candleIndex, updatedData);

      if (updatedCandle) {
        updateCandles(prev => {
          const newCandles = prev.map(c => 
            c.candle_index === candleIndex ? updatedCandle : c
          );
          console.log(`✏️ Локальное обновление свечи ${candleIndex}`);
          return newCandles;
        });
        
        console.log('✅ Свеча обновлена успешно:', candleIndex);
        return updatedCandle;
      }
    } catch (error) {
      console.error('❌ Ошибка обновления свечи:', error);
      addError('Ошибка обновления свечи', error instanceof Error ? error.message : 'Unknown error');
      throw error;
    }
  }, [currentSession, updateCandles, addError]);

  // Form submit handler
  const handleFormSubmit = useCallback(async (candleIndex: number): Promise<CandleData | null> => {
    if (!currentSession) {
      throw new Error('Нет активной сессии');
    }

    const { errors, isValid } = validateForm(formState.formData);
    
    setFormState(prev => ({ ...prev, errors, isValid }));
    
    if (!isValid) {
      return null;
    }

    setFormState(prev => ({ ...prev, isSubmitting: true }));

    try {
      const { formData } = formState;
      
      const candleData: Omit<CandleData, 'id'> = {
        session_id: currentSession.id,
        candle_index: candleIndex,
        open: Number(formData.open),
        high: Number(formData.high),
        low: Number(formData.low),
        close: Number(formData.close),
        volume: Number(formData.volume),
        candle_datetime: new Date().toISOString(),
        spread: formData.spread ? Number(formData.spread) : undefined
      };

      return await saveCandle(candleData);
    } finally {
      setFormState(prev => ({ ...prev, isSubmitting: false }));
    }
  }, [currentSession, formState, validateForm, saveCandle]);

  // Reset form
  const resetForm = useCallback(() => {
    setFormState({
      formData: {
        open: '',
        high: '',
        low: '',
        close: '',
        volume: '',
        timestamp: new Date().toISOString().slice(0, 16),
        spread: ''
      },
      errors: {},
      isValid: false,
      isSubmitting: false
    });
  }, []);

  // Auto-calculation helper
  const calculateFromOHLC = useCallback((open: number, high: number, low: number) => {
    const range = high - low;
    const close = open + (Math.random() - 0.5) * range * 0.8;
    const volume = Math.floor(Math.random() * 10000) + 1000;

    return {
      close: Math.max(low, Math.min(high, close)),
      volume
    };
  }, []);

  return {
    // Form state
    formData: formState.formData,
    formErrors: formState.errors,
    isFormValid: formState.isValid,
    isSubmitting: formState.isSubmitting,
    
    // Form operations
    handleInputChange,
    handleFormSubmit,
    resetForm,
    calculateFromOHLC,
    validateForm,
    
    // CRUD operations
    saveCandle,
    deleteCandle,
    updateCandle
  };
};