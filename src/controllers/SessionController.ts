import { sessionService } from '@/services/sessionService';
import { candleService } from '@/services/candleService';
import { TradingSession, CandleData } from '@/types/session';
import { validateSessionData } from '@/utils/candleValidation';

// Event Emitter для real-time обновлений
class SessionEventEmitter extends EventTarget {
  emit(eventType: string, data: any) {
    this.dispatchEvent(new CustomEvent(eventType, { detail: data }));
  }

  on(eventType: string, callback: (event: CustomEvent) => void) {
    this.addEventListener(eventType, callback);
  }

  off(eventType: string, callback: (event: CustomEvent) => void) {
    this.removeEventListener(eventType, callback);
  }
}

export class SessionController {
  private static instance: SessionController;
  private eventEmitter = new SessionEventEmitter();
  private currentSession: TradingSession | null = null;
  private sessionCache = new Map<string, { session: TradingSession; candles: CandleData[]; timestamp: number }>();
  private readonly CACHE_TTL = 5 * 60 * 1000; // 5 минут
  private autoSaveInterval: NodeJS.Timeout | null = null;

  private constructor() {
    this.setupAutoSave();
  }

  static getInstance(): SessionController {
    if (!SessionController.instance) {
      SessionController.instance = new SessionController();
    }
    return SessionController.instance;
  }

  // Event Management
  on(event: string, callback: (data: any) => void) {
    this.eventEmitter.on(event, (e) => callback(e.detail));
  }

  off(event: string, callback: (data: any) => void) {
    this.eventEmitter.off(event, callback);
  }

  private emit(event: string, data: any) {
    this.eventEmitter.emit(event, data);
  }

  // Session Lifecycle Management
  async loadSessions(): Promise<TradingSession[]> {
    try {
      this.emit('sessions.loading', true);
      const sessions = await sessionService.loadSessions();
      this.emit('sessions.loaded', sessions);
      return sessions;
    } catch (error) {
      this.emit('sessions.error', error);
      throw error;
    } finally {
      this.emit('sessions.loading', false);
    }
  }

  async createSession(sessionData: {
    session_name: string;
    pair: string;
    timeframe: string;
    start_date: string;
    start_time: string;
  }): Promise<TradingSession> {
    try {
      // Продвинутая валидация
      const validation = validateSessionData(sessionData);
      if (!validation.isValid) {
        throw new Error(`Validation failed: ${validation.errors.join(', ')}`);
      }

      this.emit('session.creating', sessionData);
      const session = await sessionService.createSession(sessionData);
      
      // Кэшируем новую сессию
      this.sessionCache.set(session.id, {
        session,
        candles: [],
        timestamp: Date.now()
      });

      this.emit('session.created', session);
      return session;
    } catch (error) {
      this.emit('session.error', error);
      throw error;
    }
  }

  async loadSession(sessionId: string): Promise<{ session: TradingSession; candles: CandleData[] }> {
    try {
      // Проверяем кэш
      const cached = this.sessionCache.get(sessionId);
      if (cached && this.isCacheValid(cached.timestamp)) {
        this.currentSession = cached.session;
        this.emit('session.loaded', { session: cached.session, candles: cached.candles });
        return { session: cached.session, candles: cached.candles };
      }

      this.emit('session.loading', sessionId);
      const result = await sessionService.loadSessionWithCandles(sessionId);
      
      // Обновляем кэш
      this.sessionCache.set(sessionId, {
        session: result.session,
        candles: result.candles,
        timestamp: Date.now()
      });

      this.currentSession = result.session;
      this.emit('session.loaded', result);
      
      return result;
    } catch (error) {
      this.emit('session.error', error);
      throw error;
    }
  }

  async deleteSession(sessionId: string): Promise<void> {
    try {
      this.emit('session.deleting', sessionId);
      await sessionService.deleteSession(sessionId);
      
      // Очищаем кэш
      this.sessionCache.delete(sessionId);
      
      // Если удаляем текущую сессию
      if (this.currentSession?.id === sessionId) {
        this.currentSession = null;
        this.emit('session.current.changed', null);
      }

      this.emit('session.deleted', sessionId);
    } catch (error) {
      this.emit('session.error', error);
      throw error;
    }
  }

  async duplicateSession(sessionId: string, newName: string): Promise<TradingSession> {
    try {
      this.emit('session.duplicating', { sessionId, newName });
      const newSession = await sessionService.duplicateSession(sessionId, newName);
      this.emit('session.duplicated', newSession);
      return newSession;
    } catch (error) {
      this.emit('session.error', error);
      throw error;
    }
  }

  // Candle Management
  async saveCandle(candleData: CandleData): Promise<CandleData> {
    try {
      if (!this.currentSession) {
        throw new Error('No active session');
      }

      this.emit('candle.saving', candleData);
      const savedCandle = await candleService.saveCandle(candleData);
      
      // Обновляем кэш
      const cached = this.sessionCache.get(this.currentSession.id);
      if (cached) {
        cached.candles = [...cached.candles, savedCandle].sort((a, b) => a.candle_index - b.candle_index);
        cached.timestamp = Date.now();
      }

      // Обновляем индекс сессии
      await this.updateSessionCandleIndex(this.currentSession.id, savedCandle.candle_index);

      this.emit('candle.saved', savedCandle);
      return savedCandle;
    } catch (error) {
      this.emit('candle.error', error);
      throw error;
    }
  }

  async updateCandle(candleIndex: number, updates: Partial<CandleData>): Promise<CandleData> {
    try {
      if (!this.currentSession) {
        throw new Error('No active session');
      }

      this.emit('candle.updating', { candleIndex, updates });
      const updatedCandle = await candleService.updateCandle(this.currentSession.id, candleIndex, updates);
      
      // Обновляем кэш
      const cached = this.sessionCache.get(this.currentSession.id);
      if (cached) {
        cached.candles = cached.candles.map(c => 
          c.candle_index === candleIndex ? updatedCandle : c
        );
        cached.timestamp = Date.now();
      }

      this.emit('candle.updated', updatedCandle);
      return updatedCandle;
    } catch (error) {
      this.emit('candle.error', error);
      throw error;
    }
  }

  async deleteCandle(candleIndex: number): Promise<void> {
    try {
      if (!this.currentSession) {
        throw new Error('No active session');
      }

      this.emit('candle.deleting', candleIndex);
      await candleService.deleteCandle(this.currentSession.id, candleIndex);
      
      // Обновляем кэш
      const cached = this.sessionCache.get(this.currentSession.id);
      if (cached) {
        cached.candles = cached.candles.filter(c => c.candle_index !== candleIndex);
        cached.timestamp = Date.now();
      }

      this.emit('candle.deleted', candleIndex);
    } catch (error) {
      this.emit('candle.error', error);
      throw error;
    }
  }

  // Batch Operations
  async saveCandlesBatch(candles: CandleData[]): Promise<CandleData[]> {
    try {
      if (!this.currentSession) {
        throw new Error('No active session');
      }

      this.emit('candles.batch.saving', candles);
      const savedCandles = await candleService.saveCandleBatch(candles);
      
      // Обновляем кэш
      const cached = this.sessionCache.get(this.currentSession.id);
      if (cached) {
        cached.candles = [...cached.candles, ...savedCandles].sort((a, b) => a.candle_index - b.candle_index);
        cached.timestamp = Date.now();
      }

      this.emit('candles.batch.saved', savedCandles);
      return savedCandles;
    } catch (error) {
      this.emit('candles.batch.error', error);
      throw error;
    }
  }

  // Session State Management
  getCurrentSession(): TradingSession | null {
    return this.currentSession;
  }

  setCurrentSession(session: TradingSession | null): void {
    const previousSession = this.currentSession;
    this.currentSession = session;
    
    if (previousSession?.id !== session?.id) {
      this.emit('session.current.changed', { previous: previousSession, current: session });
    }
  }

  getSessionCandles(sessionId: string): CandleData[] {
    const cached = this.sessionCache.get(sessionId);
    return cached?.candles || [];
  }

  // Cache Management
  private isCacheValid(timestamp: number): boolean {
    return Date.now() - timestamp < this.CACHE_TTL;
  }

  clearCache(): void {
    this.sessionCache.clear();
    this.emit('cache.cleared', null);
  }

  getCacheStats() {
    const validEntries = Array.from(this.sessionCache.values())
      .filter(entry => this.isCacheValid(entry.timestamp));
    
    return {
      totalEntries: this.sessionCache.size,
      validEntries: validEntries.length,
      invalidEntries: this.sessionCache.size - validEntries.length
    };
  }

  // Auto-save functionality
  private setupAutoSave(): void {
    this.autoSaveInterval = setInterval(() => {
      if (this.currentSession) {
        this.emit('session.autosave', this.currentSession.id);
      }
    }, 30000); // Каждые 30 секунд
  }

  enableAutoSave(enabled: boolean): void {
    if (enabled && !this.autoSaveInterval) {
      this.setupAutoSave();
    } else if (!enabled && this.autoSaveInterval) {
      clearInterval(this.autoSaveInterval);
      this.autoSaveInterval = null;
    }
  }

  // Private helpers
  private async updateSessionCandleIndex(sessionId: string, candleIndex: number): Promise<void> {
    await sessionService.updateSessionCandleIndex(sessionId, candleIndex);
    
    // Обновляем текущую сессию
    if (this.currentSession && this.currentSession.id === sessionId) {
      this.currentSession.current_candle_index = candleIndex;
      this.emit('session.index.updated', { sessionId, candleIndex });
    }
  }

  // Cleanup
  destroy(): void {
    if (this.autoSaveInterval) {
      clearInterval(this.autoSaveInterval);
    }
    this.clearCache();
    this.currentSession = null;
  }
}

// Singleton export
export const sessionController = SessionController.getInstance();