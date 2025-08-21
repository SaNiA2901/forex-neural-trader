import { useCallback } from 'react';
import { TradingSession, CandleData } from '@/types/session';
import { useToast } from '@/hooks/use-toast';

export const useDataExport = () => {
  const { toast } = useToast();

  const exportSessionData = useCallback(async (
    session: TradingSession,
    candles: CandleData[]
  ) => {
    try {
      const exportData = {
        session: {
          ...session,
          exported_at: new Date().toISOString(),
          version: '1.0'
        },
        candles: candles.map(candle => ({
          ...candle,
          // Убираем служебные поля для экспорта
          id: undefined,
          session_id: undefined
        })),
        metadata: {
          total_candles: candles.length,
          date_range: {
            start: candles[0]?.candle_datetime,
            end: candles[candles.length - 1]?.candle_datetime
          },
          statistics: calculateSessionStats(candles)
        }
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json'
      });

      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `session_${session.session_name}_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      toast({
        title: "Экспорт выполнен",
        description: `Данные сессии "${session.session_name}" экспортированы`
      });

      return true;
    } catch (error) {
      console.error('Export error:', error);
      toast({
        title: "Ошибка экспорта",
        description: "Не удалось экспортировать данные",
        variant: "destructive"
      });
      return false;
    }
  }, [toast]);

  const exportToCsv = useCallback(async (
    session: TradingSession,
    candles: CandleData[]
  ) => {
    try {
      const headers = [
        'Дата и время',
        'Индекс свечи',
        'Открытие',
        'Максимум',
        'Минимум',
        'Закрытие',
        'Объем',
        'Прогноз направления',
        'Вероятность прогноза',
        'Уверенность модели'
      ];

      const csvData = candles.map(candle => [
        candle.candle_datetime,
        candle.candle_index,
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.volume,
        candle.prediction_direction || '',
        candle.prediction_probability || '',
        candle.prediction_confidence || ''
      ]);

      const csvContent = [
        headers.join(','),
        ...csvData.map(row => row.join(','))
      ].join('\n');

      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `session_${session.session_name}_${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      toast({
        title: "CSV экспорт выполнен",
        description: `Данные экспортированы в формате CSV`
      });

      return true;
    } catch (error) {
      console.error('CSV export error:', error);
      toast({
        title: "Ошибка CSV экспорта",
        description: "Не удалось экспортировать в CSV",
        variant: "destructive"
      });
      return false;
    }
  }, [toast]);

  const importSessionData = useCallback(async (
    file: File,
    onSessionImported: (session: TradingSession, candles: CandleData[]) => Promise<void>
  ) => {
    try {
      const text = await file.text();
      const data = JSON.parse(text);

      if (!data.session || !data.candles) {
        throw new Error('Неверный формат файла');
      }

      const session: TradingSession = {
        ...data.session,
        id: crypto.randomUUID(),
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };

      const candles: CandleData[] = data.candles.map((candle: any) => ({
        ...candle,
        id: crypto.randomUUID(),
        session_id: session.id
      }));

      await onSessionImported(session, candles);

      toast({
        title: "Импорт выполнен",
        description: `Сессия "${session.session_name}" импортирована (${candles.length} свечей)`
      });

      return true;
    } catch (error) {
      console.error('Import error:', error);
      toast({
        title: "Ошибка импорта",
        description: "Не удалось импортировать данные",
        variant: "destructive"
      });
      return false;
    }
  }, [toast]);

  return {
    exportSessionData,
    exportToCsv,
    importSessionData
  };
};

const calculateSessionStats = (candles: CandleData[]) => {
  if (candles.length === 0) return {};

  const prices = candles.map(c => c.close);
  const volumes = candles.map(c => c.volume);

  return {
    total_candles: candles.length,
    price_range: {
      min: Math.min(...prices),
      max: Math.max(...prices),
      avg: prices.reduce((a, b) => a + b, 0) / prices.length
    },
    volume_stats: {
      total: volumes.reduce((a, b) => a + b, 0),
      avg: volumes.reduce((a, b) => a + b, 0) / volumes.length
    },
    predictions: {
      total: candles.filter(c => c.prediction_direction).length,
      accuracy: calculatePredictionAccuracy(candles)
    }
  };
};

const calculatePredictionAccuracy = (candles: CandleData[]): number => {
  const predictionsWithOutcome = [];
  
  for (let i = 0; i < candles.length - 1; i++) {
    const current = candles[i];
    const next = candles[i + 1];
    
    if (current.prediction_direction && next) {
      const actualDirection = next.close > current.close ? 'UP' : 'DOWN';
      const correct = current.prediction_direction === actualDirection;
      predictionsWithOutcome.push(correct);
    }
  }
  
  if (predictionsWithOutcome.length === 0) return 0;
  
  const correctPredictions = predictionsWithOutcome.filter(Boolean).length;
  return (correctPredictions / predictionsWithOutcome.length) * 100;
};