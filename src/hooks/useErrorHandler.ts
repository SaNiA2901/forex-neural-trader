import { useState, useCallback, useRef, useEffect } from 'react';
import { toast } from '@/hooks/use-toast';

interface ErrorState {
  message: string;
  type: 'error' | 'warning' | 'info';
  timestamp: number;
  context?: any;
}

interface UseErrorHandlerOptions {
  enableRetry?: boolean;
  maxRetries?: number;
  retryDelay?: number;
  logErrors?: boolean;
  showToast?: boolean;
}

export const useErrorHandler = (options: UseErrorHandlerOptions = {}) => {
  const {
    enableRetry = true,
    maxRetries = 3,
    retryDelay = 1000,
    logErrors = true,
    showToast = true
  } = options;

  const [errors, setErrors] = useState<ErrorState[]>([]);
  const [isRetrying, setIsRetrying] = useState(false);
  const retryCountRef = useRef(new Map<string, number>());

  // Обработка ошибок
  const handleError = useCallback((
    error: Error | string,
    context?: any,
    errorType: 'error' | 'warning' | 'info' = 'error'
  ) => {
    const message = typeof error === 'string' ? error : error.message;
    const errorState: ErrorState = {
      message,
      type: errorType,
      timestamp: Date.now(),
      context
    };

    setErrors(prev => [...prev.slice(-9), errorState]); // Храним последние 10 ошибок

    if (logErrors) {
      console.error(`[${errorType.toUpperCase()}]`, message, context);
    }

    if (showToast) {
      toast({
        title: errorType === 'error' ? 'Ошибка' : errorType === 'warning' ? 'Предупреждение' : 'Информация',
        description: message,
        variant: errorType === 'error' ? 'destructive' : 'default'
      });
    }

    return errorState;
  }, [logErrors, showToast]);

  // Выполнение операции с повторными попытками
  const executeWithRetry = useCallback(async <T>(
    operation: () => Promise<T>,
    operationId: string,
    customOptions?: Partial<UseErrorHandlerOptions>
  ): Promise<T | null> => {
    const opts = { ...options, ...customOptions };
    const currentRetries = retryCountRef.current.get(operationId) || 0;

    try {
      const result = await operation();
      retryCountRef.current.delete(operationId); // Сброс счетчика при успехе
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Неизвестная ошибка';
      
      if (enableRetry && currentRetries < maxRetries) {
        retryCountRef.current.set(operationId, currentRetries + 1);
        setIsRetrying(true);
        
        handleError(
          `Попытка ${currentRetries + 1}/${maxRetries}: ${errorMessage}`,
          { operationId, attempt: currentRetries + 1 },
          'warning'
        );

        // Задержка перед повторной попыткой
        await new Promise(resolve => setTimeout(resolve, retryDelay * (currentRetries + 1)));
        
        setIsRetrying(false);
        return executeWithRetry(operation, operationId, customOptions);
      } else {
        handleError(
          `Операция завершилась неудачей после ${currentRetries + 1} попыток: ${errorMessage}`,
          { operationId, finalAttempt: true },
          'error'
        );
        retryCountRef.current.delete(operationId);
        return null;
      }
    }
  }, [enableRetry, maxRetries, retryDelay, handleError]);

  // Безопасное выполнение операции
  const safeExecute = useCallback(async <T>(
    operation: () => Promise<T> | T,
    fallback?: T,
    errorContext?: string
  ): Promise<T | undefined> => {
    try {
      const result = await operation();
      return result;
    } catch (error) {
      handleError(
        error instanceof Error ? error : new Error('Неизвестная ошибка'),
        errorContext
      );
      return fallback;
    }
  }, [handleError]);

  // Очистка ошибок
  const clearErrors = useCallback(() => {
    setErrors([]);
    retryCountRef.current.clear();
  }, []);

  // Получение статистики ошибок
  const getErrorStats = useCallback(() => {
    const now = Date.now();
    const recentErrors = errors.filter(error => now - error.timestamp < 300000); // За последние 5 минут
    
    return {
      total: errors.length,
      recent: recentErrors.length,
      byType: {
        errors: errors.filter(e => e.type === 'error').length,
        warnings: errors.filter(e => e.type === 'warning').length,
        info: errors.filter(e => e.type === 'info').length
      },
      lastError: errors.length > 0 ? errors[errors.length - 1] : null
    };
  }, [errors]);

  // Удаление конкретной ошибки
  const dismissError = useCallback((timestamp: number) => {
    setErrors(prev => prev.filter(error => error.timestamp !== timestamp));
  }, []);

  return {
    errors,
    isRetrying,
    handleError,
    addError: handleError, // Алиас для обратной совместимости
    executeWithRetry,
    safeExecute,
    clearErrors,
    dismissError,
    getErrorStats
  };
};