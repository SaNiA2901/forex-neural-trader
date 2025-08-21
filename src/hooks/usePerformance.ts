import { useMemo, useCallback, useRef } from 'react';
import { debounce } from '@/lib/utils';

interface UsePerformanceOptions {
  debounceMs?: number;
  memoizationKey?: string;
}

export const usePerformance = (options: UsePerformanceOptions = {}) => {
  const { debounceMs = 300, memoizationKey } = options;
  const cacheRef = useRef(new Map());
  const performanceRef = useRef(new Map());

  // Дебаунсированная функция для операций ввода
  const debouncedCallback = useCallback(
    debounce((callback: Function, ...args: any[]) => {
      callback(...args);
    }, debounceMs),
    [debounceMs]
  );

  // Мемоизация с ключом
  const memoizedValue = useCallback(<T>(
    computeFn: () => T,
    deps: any[],
    cacheKey?: string
  ): T => {
    const key = cacheKey || memoizationKey || JSON.stringify(deps);
    
    if (cacheRef.current.has(key)) {
      const cached = cacheRef.current.get(key);
      if (JSON.stringify(cached.deps) === JSON.stringify(deps)) {
        return cached.value;
      }
    }

    const value = computeFn();
    cacheRef.current.set(key, { value, deps });
    
    // Ограничиваем размер кэша
    if (cacheRef.current.size > 100) {
      const firstKey = cacheRef.current.keys().next().value;
      cacheRef.current.delete(firstKey);
    }
    
    return value;
  }, [memoizationKey]);

  // Трекинг производительности
  const trackPerformance = useCallback((operationName: string, operation: () => Promise<any> | any) => {
    const start = performance.now();
    
    const result = operation();
    
    if (result instanceof Promise) {
      return result.finally(() => {
        const end = performance.now();
        performanceRef.current.set(operationName, end - start);
        console.debug(`Performance: ${operationName} took ${(end - start).toFixed(2)}ms`);
      });
    } else {
      const end = performance.now();
      performanceRef.current.set(operationName, end - start);
      console.debug(`Performance: ${operationName} took ${(end - start).toFixed(2)}ms`);
      return result;
    }
  }, []);

  // Получение статистики производительности
  const getPerformanceStats = useCallback(() => {
    return Array.from(performanceRef.current.entries()).map(([name, time]) => ({
      operation: name,
      time: Math.round(time * 100) / 100,
      status: time < 100 ? 'fast' : time < 500 ? 'moderate' : 'slow'
    }));
  }, []);

  // Очистка кэша
  const clearCache = useCallback(() => {
    cacheRef.current.clear();
    performanceRef.current.clear();
  }, []);

  return {
    debouncedCallback,
    memoizedValue,
    trackPerformance,
    getPerformanceStats,
    clearCache
  };
};