
import { supabase } from '@/integrations/supabase/client';
import { CandleData } from '@/types/session';

// Кэш для свечей по сессиям
const candleCache = new Map<string, { data: CandleData[], timestamp: number, lastIndex: number }>();
const CACHE_TTL = 3 * 60 * 1000; // 3 минуты для свечей

// Batch операции
interface BatchOperation {
  type: 'insert' | 'update' | 'delete';
  data: any;
  sessionId: string;
  candleIndex?: number;
}

let batchQueue: BatchOperation[] = [];
let batchTimeout: NodeJS.Timeout | null = null;
const BATCH_DELAY = 1000; // 1 секунда

// Utility функции
const isCacheValid = (timestamp: number): boolean => {
  return Date.now() - timestamp < CACHE_TTL;
};

const getCacheKey = (sessionId: string): string => sessionId;

const retryOperation = async <T>(
  operation: () => Promise<T>, 
  maxRetries: number = 3,
  delayMs: number = 500
): Promise<T> => {
  let lastError: Error;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error('Unknown error');
      
      if (attempt === maxRetries) break;
      
      // Exponential backoff
      await new Promise(resolve => setTimeout(resolve, delayMs * Math.pow(1.5, attempt - 1)));
    }
  }
  
  throw lastError!;
};

// Batch processing
const processBatch = async () => {
  if (batchQueue.length === 0) return;
  
  const operations = [...batchQueue];
  batchQueue = [];
  
  // Группируем операции по сессиям для оптимизации
  const sessionOperations = new Map<string, BatchOperation[]>();
  
  operations.forEach(op => {
    if (!sessionOperations.has(op.sessionId)) {
      sessionOperations.set(op.sessionId, []);
    }
    sessionOperations.get(op.sessionId)!.push(op);
  });
  
  // Выполняем операции для каждой сессии
  for (const [sessionId, ops] of sessionOperations) {
    try {
      await processSessionBatch(sessionId, ops);
    } catch (error) {
      console.error(`Batch processing failed for session ${sessionId}:`, error);
    }
  }
};

const processSessionBatch = async (sessionId: string, operations: BatchOperation[]) => {
  // Здесь можно добавить более сложную логику батчинга
  // Пока выполняем операции последовательно
  for (const op of operations) {
    try {
      switch (op.type) {
        case 'insert':
          await supabase.from('candle_data').upsert(op.data);
          break;
        case 'update':
          await supabase
            .from('candle_data')
            .update(op.data)
            .eq('session_id', sessionId)
            .eq('candle_index', op.candleIndex!);
          break;
        case 'delete':
          await supabase
            .from('candle_data')
            .delete()
            .eq('session_id', sessionId)
            .eq('candle_index', op.candleIndex!);
          break;
      }
    } catch (error) {
      console.error('Batch operation failed:', op, error);
    }
  }
  
  // Инвалидируем кэш для данной сессии
  candleCache.delete(getCacheKey(sessionId));
};

const addToBatch = (operation: BatchOperation) => {
  batchQueue.push(operation);
  
  if (batchTimeout) {
    clearTimeout(batchTimeout);
  }
  
  batchTimeout = setTimeout(processBatch, BATCH_DELAY);
};

export const candleService = {
  async saveCandle(candleData: Omit<CandleData, 'id'>): Promise<CandleData> {
    // Проверяем обязательные поля
    if (!candleData.session_id) {
      throw new Error('Session ID is required for saving candle data');
    }

    if (typeof candleData.candle_index !== 'number' || candleData.candle_index < 0) {
      throw new Error('Valid candle index is required');
    }

    return retryOperation(async () => {
      const { data, error } = await supabase
        .from('candle_data')
        .upsert(candleData, { 
          onConflict: 'session_id,candle_index',
          ignoreDuplicates: false 
        })
        .select()
        .single();

      if (error) {
        console.error('Error saving candle:', error);
        throw new Error(`Failed to save candle: ${error.message}`);
      }
      
      if (!data) {
        throw new Error('No data returned from candle save operation');
      }
      
      // Обновляем кэш
      const cacheKey = getCacheKey(candleData.session_id);
      const cached = candleCache.get(cacheKey);
      
      if (cached && isCacheValid(cached.timestamp)) {
        // Обновляем или добавляем свечу в кэш
        const existingIndex = cached.data.findIndex(c => c.candle_index === data.candle_index);
        if (existingIndex >= 0) {
          cached.data[existingIndex] = data;
        } else {
          cached.data.push(data);
          cached.data.sort((a, b) => a.candle_index - b.candle_index);
        }
        cached.lastIndex = Math.max(cached.lastIndex, data.candle_index);
      }
      
      return data;
    });
  },

  async getCandlesFromCache(sessionId: string): Promise<CandleData[] | null> {
    const cached = candleCache.get(getCacheKey(sessionId));
    if (cached && isCacheValid(cached.timestamp)) {
      return [...cached.data]; // Возвращаем копию
    }
    return null;
  },

  async getCandlesForSession(sessionId: string): Promise<CandleData[]> {
    if (!sessionId?.trim()) {
      throw new Error('Session ID is required and cannot be empty');
    }

    // Проверяем кэш
    const cachedCandles = await this.getCandlesFromCache(sessionId);
    if (cachedCandles) {
      return cachedCandles;
    }

    return retryOperation(async () => {
      const { data, error } = await supabase
        .from('candle_data')
        .select('*')
        .eq('session_id', sessionId)
        .order('candle_index', { ascending: true });

      if (error) {
        console.error('Error fetching candles:', error);
        throw new Error(`Failed to fetch candles: ${error.message}`);
      }
      
      const candles = data || [];
      
      // Кэшируем результат
      const lastIndex = candles.length > 0 ? Math.max(...candles.map(c => c.candle_index)) : -1;
      candleCache.set(getCacheKey(sessionId), {
        data: candles,
        timestamp: Date.now(),
        lastIndex
      });
      
      return candles;
    });
  },

  async deleteCandle(sessionId: string, candleIndex: number): Promise<void> {
    if (!sessionId?.trim()) {
      throw new Error('Session ID is required and cannot be empty');
    }
    
    if (typeof candleIndex !== 'number' || candleIndex < 0) {
      throw new Error('Valid candle index is required');
    }

    return retryOperation(async () => {
      const { error } = await supabase
        .from('candle_data')
        .delete()
        .eq('session_id', sessionId)
        .eq('candle_index', candleIndex);

      if (error) {
        console.error('Error deleting candle:', error);
        throw new Error(`Failed to delete candle: ${error.message}`);
      }
      
      // Обновляем кэш
      const cacheKey = getCacheKey(sessionId);
      const cached = candleCache.get(cacheKey);
      
      if (cached) {
        cached.data = cached.data.filter(c => c.candle_index !== candleIndex);
        cached.timestamp = Date.now();
      }
    });
  },

  async updateCandle(sessionId: string, candleIndex: number, updatedData: Partial<CandleData>): Promise<CandleData> {
    if (!sessionId?.trim()) {
      throw new Error('Session ID is required and cannot be empty');
    }
    
    if (typeof candleIndex !== 'number' || candleIndex < 0) {
      throw new Error('Valid candle index is required');
    }

    if (!updatedData || Object.keys(updatedData).length === 0) {
      throw new Error('Updated data is required');
    }

    return retryOperation(async () => {
      const { data, error } = await supabase
        .from('candle_data')
        .update(updatedData)
        .eq('session_id', sessionId)
        .eq('candle_index', candleIndex)
        .select()
        .single();

      if (error) {
        console.error('Error updating candle:', error);
        throw new Error(`Failed to update candle: ${error.message}`);
      }
      
      if (!data) {
        throw new Error('No data returned from candle update operation');
      }
      
      // Обновляем кэш
      const cacheKey = getCacheKey(sessionId);
      const cached = candleCache.get(cacheKey);
      
      if (cached) {
        const candleIndex = cached.data.findIndex(c => c.candle_index === data.candle_index);
        if (candleIndex >= 0) {
          cached.data[candleIndex] = data;
          cached.timestamp = Date.now();
        }
      }
      
      return data;
    });
  },

  async getCandleByIndex(sessionId: string, candleIndex: number): Promise<CandleData | null> {
    if (!sessionId?.trim()) {
      throw new Error('Session ID is required and cannot be empty');
    }
    
    if (typeof candleIndex !== 'number' || candleIndex < 0) {
      throw new Error('Valid candle index is required');
    }

    // Проверяем кэш сначала
    const cached = candleCache.get(getCacheKey(sessionId));
    if (cached && isCacheValid(cached.timestamp)) {
      const candleFromCache = cached.data.find(c => c.candle_index === candleIndex);
      if (candleFromCache) {
        return candleFromCache;
      }
    }

    return retryOperation(async () => {
      const { data, error } = await supabase
        .from('candle_data')
        .select('*')
        .eq('session_id', sessionId)
        .eq('candle_index', candleIndex)
        .maybeSingle();

      if (error) {
        console.error('Error fetching candle by index:', error);
        throw new Error(`Failed to fetch candle: ${error.message}`);
      }
      
      return data;
    });
  },

  async getLatestCandles(sessionId: string, limit: number = 10): Promise<CandleData[]> {
    if (!sessionId?.trim()) {
      throw new Error('Session ID is required and cannot be empty');
    }
    
    if (typeof limit !== 'number' || limit <= 0) {
      throw new Error('Valid limit is required');
    }

    // Проверяем кэш
    const cached = candleCache.get(getCacheKey(sessionId));
    if (cached && isCacheValid(cached.timestamp)) {
      return cached.data.slice(-limit);
    }

    return retryOperation(async () => {
      const { data, error } = await supabase
        .from('candle_data')
        .select('*')
        .eq('session_id', sessionId)
        .order('candle_index', { ascending: false })
        .limit(limit);

      if (error) {
        console.error('Error fetching latest candles:', error);
        throw new Error(`Failed to fetch latest candles: ${error.message}`);
      }
      
      return (data || []).reverse(); // Возвращаем в правильном порядке
    });
  },

  // Batch операции
  async saveCandleBatch(candles: CandleData[]): Promise<CandleData[]> {
    if (candles.length === 0) return [];
    
    return retryOperation(async () => {
      const { data, error } = await supabase
        .from('candle_data')
        .upsert(candles, { 
          onConflict: 'session_id,candle_index',
          ignoreDuplicates: false 
        })
        .select();

      if (error) {
        console.error('Error saving candle batch:', error);
        throw new Error(`Failed to save candle batch: ${error.message}`);
      }
      
      if (!data) {
        throw new Error('No data returned from batch save operation');
      }
      
      // Обновляем кэш для каждой сессии
      const sessionGroups = new Map<string, CandleData[]>();
      data.forEach(candle => {
        if (!sessionGroups.has(candle.session_id)) {
          sessionGroups.set(candle.session_id, []);
        }
        sessionGroups.get(candle.session_id)!.push(candle);
      });
      
      sessionGroups.forEach((sessionCandles, sessionId) => {
        const cacheKey = getCacheKey(sessionId);
        const cached = candleCache.get(cacheKey);
        
        if (cached && isCacheValid(cached.timestamp)) {
          sessionCandles.forEach(candle => {
            const existingIndex = cached.data.findIndex(c => c.candle_index === candle.candle_index);
            if (existingIndex >= 0) {
              cached.data[existingIndex] = candle;
            } else {
              cached.data.push(candle);
            }
          });
          cached.data.sort((a, b) => a.candle_index - b.candle_index);
          cached.lastIndex = Math.max(cached.lastIndex, ...sessionCandles.map(c => c.candle_index));
          cached.timestamp = Date.now();
        }
      });
      
      return data;
    });
  },

  // Принудительная обработка батча
  async flushBatch(): Promise<void> {
    if (batchTimeout) {
      clearTimeout(batchTimeout);
      batchTimeout = null;
    }
    await processBatch();
  },

  // Управление кэшем
  clearCache(sessionId?: string): void {
    if (sessionId) {
      candleCache.delete(getCacheKey(sessionId));
    } else {
      candleCache.clear();
    }
  },

  getCacheStats() {
    const now = Date.now();
    const stats = Array.from(candleCache.entries()).map(([sessionId, cache]) => ({
      sessionId,
      candleCount: cache.data.length,
      lastIndex: cache.lastIndex,
      isValid: isCacheValid(cache.timestamp),
      age: now - cache.timestamp
    }));
    
    return {
      totalSessions: candleCache.size,
      validSessions: stats.filter(s => s.isValid).length,
      totalCandles: stats.reduce((sum, s) => sum + s.candleCount, 0),
      batchQueueSize: batchQueue.length,
      sessions: stats
    };
  }
};
