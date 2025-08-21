import React, { createContext, useContext, useReducer, useCallback, useEffect } from 'react';
import { TradingSession, CandleData, SessionStats } from '@/types/session';
import { PredictionResult } from '@/types/trading';

// Состояние приложения
interface TradingState {
  // Сессии
  sessions: TradingSession[];
  currentSession: TradingSession | null;
  sessionStats: SessionStats | null;
  
  // Свечи
  candles: CandleData[];
  nextCandleIndex: number;
  
  // Прогнозы
  lastPrediction: PredictionResult | null;
  predictionHistory: PredictionResult[];
  
  // UI состояние
  isLoading: boolean;
  errors: string[];
  isConnected: boolean;
  
  // Аналитика
  performance: {
    totalPredictions: number;
    accuratePredictions: number;
    accuracy: number;
  };
}

// Типы действий
type TradingAction =
  // Сессии
  | { type: 'SET_SESSIONS'; payload: TradingSession[] }
  | { type: 'SET_CURRENT_SESSION'; payload: TradingSession | null }
  | { type: 'ADD_SESSION'; payload: TradingSession }
  | { type: 'UPDATE_SESSION'; payload: { id: string; updates: Partial<TradingSession> } }
  | { type: 'DELETE_SESSION'; payload: string }
  
  // Свечи
  | { type: 'SET_CANDLES'; payload: CandleData[] }
  | { type: 'ADD_CANDLE'; payload: CandleData }
  | { type: 'UPDATE_CANDLE'; payload: { index: number; updates: Partial<CandleData> } }
  | { type: 'DELETE_CANDLE'; payload: number }
  | { type: 'CLEAR_CANDLES' }
  
  // Прогнозы
  | { type: 'SET_PREDICTION'; payload: PredictionResult }
  | { type: 'ADD_PREDICTION_HISTORY'; payload: PredictionResult }
  | { type: 'CLEAR_PREDICTIONS' }
  
  // UI состояние
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'ADD_ERROR'; payload: string }
  | { type: 'CLEAR_ERRORS' }
  | { type: 'SET_CONNECTION_STATUS'; payload: boolean }
  
  // Аналитика
  | { type: 'UPDATE_PERFORMANCE'; payload: Partial<TradingState['performance']> };

// Начальное состояние
const initialState: TradingState = {
  sessions: [],
  currentSession: null,
  sessionStats: null,
  candles: [],
  nextCandleIndex: 0,
  lastPrediction: null,
  predictionHistory: [],
  isLoading: false,
  errors: [],
  isConnected: true,
  performance: {
    totalPredictions: 0,
    accuratePredictions: 0,
    accuracy: 0
  }
};

// Reducer с логикой состояния
function tradingReducer(state: TradingState, action: TradingAction): TradingState {
  switch (action.type) {
    case 'SET_SESSIONS':
      return { ...state, sessions: action.payload };

    case 'SET_CURRENT_SESSION':
      return { 
        ...state, 
        currentSession: action.payload,
        sessionStats: action.payload ? calculateSessionStats(state.candles) : null
      };

    case 'ADD_SESSION':
      return { 
        ...state, 
        sessions: [action.payload, ...state.sessions] 
      };

    case 'UPDATE_SESSION':
      return {
        ...state,
        sessions: state.sessions.map(session =>
          session.id === action.payload.id
            ? { ...session, ...action.payload.updates }
            : session
        ),
        currentSession: state.currentSession?.id === action.payload.id
          ? { ...state.currentSession, ...action.payload.updates }
          : state.currentSession
      };

    case 'DELETE_SESSION':
      return {
        ...state,
        sessions: state.sessions.filter(session => session.id !== action.payload),
        currentSession: state.currentSession?.id === action.payload ? null : state.currentSession
      };

    case 'SET_CANDLES':
      const sessionStats = calculateSessionStats(action.payload);
      const nextIndex = action.payload.length > 0 
        ? Math.max(...action.payload.map(c => c.candle_index)) + 1 
        : (state.currentSession?.current_candle_index || 0) + 1;
      
      return { 
        ...state, 
        candles: action.payload,
        sessionStats,
        nextCandleIndex: nextIndex
      };

    case 'ADD_CANDLE':
      const newCandles = [...state.candles, action.payload]
        .sort((a, b) => a.candle_index - b.candle_index);
      
      return { 
        ...state, 
        candles: newCandles,
        sessionStats: calculateSessionStats(newCandles),
        nextCandleIndex: action.payload.candle_index + 1
      };

    case 'UPDATE_CANDLE':
      return {
        ...state,
        candles: state.candles.map(candle =>
          candle.candle_index === action.payload.index
            ? { ...candle, ...action.payload.updates }
            : candle
        )
      };

    case 'DELETE_CANDLE':
      const filteredCandles = state.candles.filter(c => c.candle_index !== action.payload);
      return {
        ...state,
        candles: filteredCandles,
        sessionStats: calculateSessionStats(filteredCandles)
      };

    case 'CLEAR_CANDLES':
      return { 
        ...state, 
        candles: [], 
        sessionStats: null,
        nextCandleIndex: (state.currentSession?.current_candle_index || 0) + 1
      };

    case 'SET_PREDICTION':
      return { 
        ...state, 
        lastPrediction: action.payload 
      };

    case 'ADD_PREDICTION_HISTORY':
      return {
        ...state,
        predictionHistory: [action.payload, ...state.predictionHistory].slice(0, 100), // Последние 100
        performance: updatePerformanceMetrics(state.performance, action.payload)
      };

    case 'CLEAR_PREDICTIONS':
      return { 
        ...state, 
        lastPrediction: null, 
        predictionHistory: [] 
      };

    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };

    case 'ADD_ERROR':
      return { 
        ...state, 
        errors: [action.payload, ...state.errors].slice(0, 10) // Последние 10 ошибок
      };

    case 'CLEAR_ERRORS':
      return { ...state, errors: [] };

    case 'SET_CONNECTION_STATUS':
      return { ...state, isConnected: action.payload };

    case 'UPDATE_PERFORMANCE':
      return {
        ...state,
        performance: { ...state.performance, ...action.payload }
      };

    default:
      return state;
  }
}

// Вспомогательные функции
function calculateSessionStats(candles: CandleData[]): SessionStats {
  if (candles.length === 0) {
    return {
      totalCandles: 0,
      lastPrice: null,
      priceChange: 0,
      highestPrice: null,
      lowestPrice: null,
      averageVolume: 0
    };
  }

  const prices = candles.map(c => c.close);
  const volumes = candles.map(c => c.volume);
  
  return {
    totalCandles: candles.length,
    lastPrice: prices[prices.length - 1],
    priceChange: prices.length > 1 ? prices[prices.length - 1] - prices[0] : 0,
    highestPrice: Math.max(...prices),
    lowestPrice: Math.min(...prices),
    averageVolume: volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length
  };
}

function updatePerformanceMetrics(
  current: TradingState['performance'], 
  prediction: PredictionResult
): TradingState['performance'] {
  const newTotal = current.totalPredictions + 1;
  // Здесь должна быть логика проверки точности предсказания
  // Пока используем заглушку
  const newAccurate = current.accuratePredictions + (Math.random() > 0.5 ? 1 : 0);
  
  return {
    totalPredictions: newTotal,
    accuratePredictions: newAccurate,
    accuracy: newTotal > 0 ? (newAccurate / newTotal) * 100 : 0
  };
}

// Context
const TradingContext = createContext<{
  state: TradingState;
  dispatch: React.Dispatch<TradingAction>;
} | null>(null);

// Provider компонент
export const TradingStoreProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(tradingReducer, initialState);

  // Middleware для логирования
  const middlewareDispatch = useCallback((action: TradingAction) => {
    console.log('🔄 Trading Store Action:', action.type, 'payload' in action ? action.payload : 'no payload');
    dispatch(action);
  }, []);

  // Автосохранение в localStorage
  useEffect(() => {
    const stateToSave = {
      sessions: state.sessions,
      currentSession: state.currentSession,
      performance: state.performance
    };
    localStorage.setItem('trading-store', JSON.stringify(stateToSave));
  }, [state.sessions, state.currentSession, state.performance]);

  // Восстановление из localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem('trading-store');
      if (saved) {
        const { sessions, currentSession, performance } = JSON.parse(saved);
        dispatch({ type: 'SET_SESSIONS', payload: sessions || [] });
        if (currentSession) {
          dispatch({ type: 'SET_CURRENT_SESSION', payload: currentSession });
        }
        if (performance) {
          dispatch({ type: 'UPDATE_PERFORMANCE', payload: performance });
        }
      }
    } catch (error) {
      console.error('Error restoring trading store:', error);
    }
  }, []);

  return (
    <TradingContext.Provider value={{ state, dispatch: middlewareDispatch }}>
      {children}
    </TradingContext.Provider>
  );
};

// Hook для использования store
export const useTradingStore = () => {
  const context = useContext(TradingContext);
  if (!context) {
    throw new Error('useTradingStore must be used within TradingStoreProvider');
  }
  return context;
};

// Селекторы для оптимизации
export const useSessionSelector = () => {
  const { state } = useTradingStore();
  return {
    sessions: state.sessions,
    currentSession: state.currentSession,
    sessionStats: state.sessionStats
  };
};

export const useCandleSelector = () => {
  const { state } = useTradingStore();
  return {
    candles: state.candles,
    nextCandleIndex: state.nextCandleIndex
  };
};

export const usePredictionSelector = () => {
  const { state } = useTradingStore();
  return {
    lastPrediction: state.lastPrediction,
    predictionHistory: state.predictionHistory,
    performance: state.performance
  };
};

export const useUISelector = () => {
  const { state } = useTradingStore();
  return {
    isLoading: state.isLoading,
    errors: state.errors,
    isConnected: state.isConnected
  };
};