
import { useCallback } from 'react';
import { TradingSession, CandleData } from '@/types/session';
import { usePredictionGeneration } from './usePredictionGeneration';
import { useNewApplicationState } from './useNewApplicationState';

interface UsePredictionLogicProps {
  currentSession: TradingSession | null;
  pair: string;
  timeframe: string;
}

export const usePredictionLogic = ({ 
  currentSession, 
  pair, 
  timeframe 
}: UsePredictionLogicProps) => {
  const { updateCandle, candles } = useNewApplicationState();
  const { generatePrediction } = usePredictionGeneration();

  const handleCandleSaved = useCallback(async (candleData: CandleData) => {
    if (!currentSession || typeof candleData.candle_index !== 'number') return;

    try {
      console.log('PredictionLogic: Generating advanced prediction for candle:', candleData.candle_index);
      
      const predictionConfig = {
        predictionInterval: 5,
        analysisMode: 'session' as const
      };

      // Получаем исторические данные из состояния приложения
      const allCandles = [...candles, candleData].sort((a, b) => a.candle_index - b.candle_index);

      // Используем продвинутую модель с историческими данными
      const prediction = await generatePrediction(candleData, predictionConfig, allCandles);
      
      if (prediction) {
        const updatedCandle: CandleData = {
          ...candleData,
          prediction_direction: prediction.direction,
          prediction_probability: prediction.probability,
          prediction_confidence: prediction.confidence
        };

        await updateCandle(candleData.candle_index, updatedCandle);
        console.log('PredictionLogic: Advanced prediction saved to candle');
      }
    } catch (error) {
      console.error('PredictionLogic: Error generating prediction:', error);
    }
  }, [currentSession, generatePrediction, updateCandle, candles]);

  return {
    handleCandleSaved
  };
};
