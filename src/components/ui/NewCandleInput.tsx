import { memo, useMemo, useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { TradingSession, CandleData } from '@/types/session';
import { calculateCandleDateTime } from '@/utils/dateTimeUtils';
import { useNewApplicationState } from '@/hooks/useNewApplicationState';
import { useCandleInputLogic } from '@/hooks/candle/useCandleInputLogic';
import { CandleInputForm } from './candle-input/CandleInputForm';
import CandleInputValidation from './candle-input/CandleInputValidation';
import { CandleInputStats } from './candle-input/CandleInputStats';
import { Clock, TrendingUp, BarChart3, Save, Trash2 } from 'lucide-react';
import { toast } from 'sonner';

interface NewCandleInputProps {
  currentSession: TradingSession | null;
  pair: string;
  onCandleSaved?: (candleData: CandleData) => Promise<void>;
}

const NewCandleInput = memo(({ 
  currentSession,
  pair,
  onCandleSaved
}: NewCandleInputProps) => {
  // Если нет активной сессии, не показываем компонент
  if (!currentSession) {
    return null;
  }
  
  const { 
    candles, 
    nextCandleIndex, 
    saveCandle, 
    deleteLastCandle,
    getLastCandle,
    isLoading
  } = useNewApplicationState();

  // Вычисляем время следующей свечи
  const nextCandleTime = useMemo(() => {
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
  }, [currentSession, nextCandleIndex]);

  // Логика формы ввода свечи
  const {
    formData,
    errors,
    isSubmitting,
    isFormValid,
    lastCandle,
    updateField,
    handleSave,
    handleDeleteLast,
    reset,
    nextCandleIndex: calculatedNextIndex,
    candles: candlesFromLogic
  } = useCandleInputLogic({
    currentSession,
    onCandleSaved: async (candleData) => {
      try {
        toast.success(`Свеча №${candleData.candle_index} сохранена`);
        
        // Вызываем callback если предоставлен
        if (onCandleSaved) {
          await onCandleSaved(candleData);
        }
      } catch (error) {
        toast.error('Ошибка при сохранении свечи');
        throw error;
      }
    }
  });

  // Быстрое заполнение данными предыдущей свечи
  const handleFillPreviousData = useCallback(() => {
    const lastCandleData = getLastCandle();
    if (!lastCandleData) return;

    // Заполняем форму данными последней свечи с небольшими вариациями
    const variance = 0.001; // 0.1% вариация
    const basePrice = lastCandleData.close;
    
    updateField('open', String(basePrice));
    updateField('high', String(basePrice * (1 + Math.random() * variance)));
    updateField('low', String(basePrice * (1 - Math.random() * variance)));
    updateField('close', String(basePrice * (1 + (Math.random() - 0.5) * variance * 2)));
    updateField('volume', String(Math.round(lastCandleData.volume * (0.8 + Math.random() * 0.4))));
    
    toast.info('Форма заполнена данными предыдущей свечи');
  }, [getLastCandle, updateField]);

  // Статистика сессии
  const sessionStats = useMemo(() => {
    const sessionCandles = candles.filter(c => c.session_id === currentSession.id);
    
    return {
      totalCandles: sessionCandles.length,
      withPredictions: sessionCandles.filter(c => c.prediction_direction).length,
      avgVolume: sessionCandles.length > 0 
        ? sessionCandles.reduce((sum, c) => sum + c.volume, 0) / sessionCandles.length 
        : 0
    };
  }, [candles, currentSession.id]);

  return (
    <Card className="p-6 bg-gradient-to-br from-card/90 to-card/70 border-border/50 backdrop-blur-sm">
      {/* Заголовок с информацией о сессии */}
      <div className="flex items-center justify-between mb-6">
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <BarChart3 className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-semibold">
              Ввод данных - {currentSession.session_name}
            </h3>
          </div>
          
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Badge variant="outline" className="text-xs">
                {pair}
              </Badge>
              <span>{currentSession.timeframe}</span>
            </div>
            
            <div className="flex items-center gap-1">
              <Clock className="h-4 w-4" />
              <span>Свеча №{calculatedNextIndex}</span>
            </div>
            
            {nextCandleTime && (
              <div className="flex items-center gap-1">
                <TrendingUp className="h-4 w-4" />
                <span>{nextCandleTime}</span>
              </div>
            )}
          </div>
        </div>

        {/* Статистика */}
        <div className="text-right space-y-1">
          <div className="text-sm font-medium">
            Свечей в сессии: {sessionStats.totalCandles}
          </div>
          <div className="text-xs text-muted-foreground">
            С прогнозами: {sessionStats.withPredictions}
          </div>
        </div>
      </div>

      {/* Форма ввода данных */}
      <div className="space-y-6">
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

        {/* Валидация */}
        <CandleInputValidation
          errors={Object.values(errors).filter(Boolean) as string[]}
          isFormValid={isFormValid}
        />

        {/* Действия */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {lastCandle && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleFillPreviousData}
                  disabled={isSubmitting || isLoading}
                >
                  Заполнить как предыдущую
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDeleteLast}
                  disabled={isSubmitting || isLoading}
                  className="text-destructive hover:text-destructive"
                >
                  <Trash2 className="h-4 w-4 mr-1" />
                  Удалить последнюю
                </Button>
              </>
            )}
          </div>

          <Button
            onClick={handleSave}
            disabled={!isFormValid || isSubmitting || isLoading}
            className="bg-primary hover:bg-primary/90"
          >
            <Save className="h-4 w-4 mr-2" />
            {isSubmitting ? 'Сохранение...' : 'Сохранить свечу'}
          </Button>
        </div>

        {/* Статистика свечей */}
        <CandleInputStats />
      </div>
    </Card>
  );
});

NewCandleInput.displayName = 'NewCandleInput';

export default NewCandleInput;