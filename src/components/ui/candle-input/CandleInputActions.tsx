import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { 
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import { 
  Trash2, 
  RotateCcw, 
  Download, 
  Upload,
  History,
  Settings,
  BarChart3
} from 'lucide-react';
import { useTradingStore } from '@/store/TradingStore';
import { useCandleActions } from '@/hooks/store/useCandleActions';
import { useToast } from '@/hooks/use-toast';

export const CandleInputActions: React.FC = () => {
  const { state } = useTradingStore();
  const { deleteCandle, clearCandles } = useCandleActions();
  const { toast } = useToast();
  const [isDeleting, setIsDeleting] = useState(false);

  const handleDeleteLastCandle = async () => {
    if (state.candles.length === 0) return;
    
    setIsDeleting(true);
    try {
      const lastCandle = state.candles[state.candles.length - 1];
      await deleteCandle(lastCandle.candle_index);
      
      toast({
        title: "Успех",
        description: `Свеча #${lastCandle.candle_index} удалена`
      });
    } catch (error) {
      toast({
        title: "Ошибка",
        description: "Не удалось удалить свечу",
        variant: "destructive"
      });
    } finally {
      setIsDeleting(false);
    }
  };

  const handleClearAllCandles = async () => {
    try {
      await clearCandles();
      toast({
        title: "Успех",
        description: "Все свечи очищены"
      });
    } catch (error) {
      toast({
        title: "Ошибка",
        description: "Не удалось очистить свечи",
        variant: "destructive"
      });
    }
  };

  const handleExportCandles = () => {
    if (state.candles.length === 0) {
      toast({
        title: "Предупреждение",
        description: "Нет данных для экспорта",
        variant: "destructive"
      });
      return;
    }

    const dataStr = JSON.stringify({
      session: state.currentSession,
      candles: state.candles,
      exportDate: new Date().toISOString()
    }, null, 2);
    
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `candles-${state.currentSession?.session_name}-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    
    URL.revokeObjectURL(url);
    
    toast({
      title: "Успех",
      description: "Данные экспортированы"
    });
  };

  const lastCandle = state.candles[state.candles.length - 1];

  return (
    <div className="flex flex-wrap items-center gap-2">
      {/* Удалить последнюю свечу */}
      <AlertDialog>
        <AlertDialogTrigger asChild>
          <Button
            variant="outline"
            size="sm"
            disabled={state.candles.length === 0 || isDeleting}
            className="bg-background/50"
          >
            <Trash2 className="h-4 w-4 mr-1" />
            Удалить последнюю
          </Button>
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Удалить последнюю свечу?</AlertDialogTitle>
            <AlertDialogDescription>
              {lastCandle ? (
                <>
                  Вы действительно хотите удалить свечу #{lastCandle.candle_index}?
                  <br />
                  O: {lastCandle.open}, H: {lastCandle.high}, L: {lastCandle.low}, C: {lastCandle.close}
                  <br />
                  Это действие нельзя отменить.
                </>
              ) : (
                "Нет свечей для удаления."
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Отмена</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteLastCandle}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Удалить
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Очистить все свечи */}
      <AlertDialog>
        <AlertDialogTrigger asChild>
          <Button
            variant="outline"
            size="sm"
            disabled={state.candles.length === 0}
            className="bg-background/50"
          >
            <RotateCcw className="h-4 w-4 mr-1" />
            Очистить все
          </Button>
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Очистить все свечи?</AlertDialogTitle>
            <AlertDialogDescription>
              Вы действительно хотите удалить все {state.candles.length} свечей из текущей сессии?
              <br />
              Это действие нельзя отменить.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Отмена</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleClearAllCandles}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Очистить все
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Экспорт данных */}
      <Button
        variant="outline"
        size="sm"
        onClick={handleExportCandles}
        disabled={state.candles.length === 0}
        className="bg-background/50"
      >
        <Download className="h-4 w-4 mr-1" />
        Экспорт
      </Button>

      {/* История изменений */}
      <Button
        variant="outline"
        size="sm"
        disabled
        className="bg-background/50"
      >
        <History className="h-4 w-4 mr-1" />
        История
      </Button>

      {/* Статистика */}
      <div className="flex items-center gap-2 ml-auto">
        <div className="text-sm text-muted-foreground">
          Свечей: {state.candles.length}
        </div>
        {state.sessionStats && (
          <div className="text-sm text-muted-foreground">
            Последняя цена: {state.sessionStats.lastPrice?.toFixed(5)}
          </div>
        )}
      </div>
    </div>
  );
};