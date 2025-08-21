
import { memo, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Settings, Download } from 'lucide-react';
import { NewSessionManager } from './session/NewSessionManager';
import NewCandleInput from '@/components/ui/NewCandleInput';
import PredictionHeader from './predictor/PredictionHeader';
import PredictionTabs from './predictor/PredictionTabs';
import PerformanceMonitor from './common/PerformanceMonitor';
import { ExportDialog } from './common/ExportDialog';
import { useTradingStore } from '@/store/TradingStore';
import { usePredictionLogic } from '@/hooks/usePredictionLogic';
import { usePredictionGeneration } from '@/hooks/usePredictionGeneration';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';
import { useErrorHandler } from '@/hooks/useErrorHandler';

interface BinaryOptionsPredictorProps {
  pair: string;
  timeframe: string;
}

const BinaryOptionsPredictor = memo(({ pair, timeframe }: BinaryOptionsPredictorProps) => {
  const [showPerformanceMonitor, setShowPerformanceMonitor] = useState(false);
  const { state } = useTradingStore();

  const { executeWithRetry } = useErrorHandler();

  const { handleCandleSaved } = usePredictionLogic({
    currentSession: state.currentSession,
    pair,
    timeframe
  });

  const { predictionResult, isGenerating } = usePredictionGeneration();

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Управление сессиями */}
      {!state.currentSession && (
        <Card className="p-6 bg-gradient-to-br from-card/80 to-card/50 border-border/50 backdrop-blur-lg shadow-2xl">
          <NewSessionManager />
        </Card>
      )}

      {/* Ввод данных свечей - показывается автоматически когда есть активная сессия */}
      {state.currentSession && (
        <Card className="p-6 bg-gradient-to-br from-card/80 to-card/50 border-border/50 backdrop-blur-lg shadow-2xl">
          <div className="flex items-center justify-between mb-6">
            <PredictionHeader 
              pair={pair} 
              timeframe={timeframe} 
              isActive={!!state.currentSession && !isGenerating}
            />
            
            {/* Управление и экспорт */}
            <div className="flex items-center gap-2">
              <ExportDialog
                session={state.currentSession}
                candles={state.candles}
                onSessionImported={async (session, sessionCandles) => {
                  await executeWithRetry(
                    async () => {
                      console.log('Importing session:', session.session_name);
                    },
                    'import-session'
                  );
                }}
              >
                <Button variant="outline" size="sm" className="bg-background/50">
                  <Download className="h-4 w-4 mr-2" />
                  Экспорт/Импорт
                </Button>
              </ExportDialog>
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowPerformanceMonitor(!showPerformanceMonitor)}
                className="bg-background/50"
              >
                <Settings className="h-4 w-4 mr-2" />
                {showPerformanceMonitor ? 'Скрыть отладку' : 'Показать отладку'}
              </Button>
            </div>
          </div>

          
          <div className="mt-6">
            <PredictionTabs
              currentSession={state.currentSession}
              candles={state.candles}
              pair={pair}
              timeframe={timeframe}
              predictionResult={predictionResult}
              onCandleSaved={handleCandleSaved}
            />
          </div>
        </Card>
      )}

      {/* Performance Monitor - опциональный для отладки */}
      <PerformanceMonitor isVisible={showPerformanceMonitor} />
    </div>
  );
});

BinaryOptionsPredictor.displayName = 'BinaryOptionsPredictor';

export default BinaryOptionsPredictor;
