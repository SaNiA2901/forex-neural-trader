
import { useMemo, memo } from 'react';
import { Card } from '@/components/ui/card';
import { TradingSession, CandleData } from '@/types/session';
import { calculateCandleDateTime } from '@/utils/dateTimeUtils';
import { useStateManager } from '@/hooks/useStateManager';
import { usePerformance } from '@/hooks/usePerformance';
import { useErrorHandler } from '@/hooks/useErrorHandler';
import CandleInputHeader from './candle-input/CandleInputHeader';
import { CandleInputForm } from './candle-input/CandleInputForm';
import CandleInputValidation from './candle-input/CandleInputValidation';
import { CandleInputActions } from './candle-input/CandleInputActions';
import { CandleInputStats } from './candle-input/CandleInputStats';
import { useCandleInputLogic } from '@/hooks/candle/useCandleInputLogic';

interface CandleInputProps {
  currentSession: TradingSession;
  candles: CandleData[];
  pair: string;
  onCandleSaved: (candleData: CandleData) => Promise<void>;
}

const CandleInput = memo(({ 
  currentSession,
  candles,
  pair,
  onCandleSaved
}: CandleInputProps) => {
  const { saveCandle, deleteLastCandle } = useStateManager();
  const { trackPerformance, memoizedValue } = usePerformance();
  const { safeExecute } = useErrorHandler();
  
  const nextCandleIndex = memoizedValue(
    () => Math.max(currentSession.current_candle_index + 1, candles.length),
    [currentSession.current_candle_index, candles.length],
    'nextCandleIndex'
  );

  const nextCandleTime = memoizedValue(() => {
    try {
      return calculateCandleDateTime(
        currentSession.start_date,
        currentSession.start_time,
        currentSession.timeframe,
        nextCandleIndex
      );
    } catch (error) {
      console.error('Error calculating next candle time:', error);
      return '';
    }
  }, [currentSession, nextCandleIndex], 'nextCandleTime');

  const {
    formData,
    errors,
    isSubmitting,
    isFormValid,
    lastCandle,
    updateField,
    handleSave,
    handleDeleteLast,
    reset
  } = useCandleInputLogic({
    currentSession,
    onCandleSaved: async (candleData) => {
      return trackPerformance('saveCandleOperation', async () => {
        await safeExecute(
          () => onCandleSaved(candleData),
          undefined,
          'Обработка сохраненной свечи'
        );
      });
    }
  });

  return (
    <Card className="p-6 bg-gradient-to-br from-card/80 to-card/50 border-border/50 backdrop-blur-sm animate-fade-in">
      <CandleInputHeader
        currentSession={currentSession}
        nextCandleIndex={nextCandleIndex}
        pair={pair}
        nextCandleTime={nextCandleTime}
      />

      <CandleInputForm
        currentSession={currentSession}
        pair={pair}
        formData={formData}
        errors={errors}
        isValid={isFormValid}
        isSubmitting={isSubmitting}
        onInputChange={updateField}
        onSubmit={handleSave}
        onReset={reset}
      />

      <CandleInputValidation
        errors={Object.values(errors).filter(Boolean) as string[]}
        isFormValid={isFormValid}
      />

      <CandleInputActions />

      <CandleInputStats />
    </Card>
  );
});

CandleInput.displayName = 'CandleInput';

export default CandleInput;
