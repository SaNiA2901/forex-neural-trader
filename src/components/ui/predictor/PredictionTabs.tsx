import { memo } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card } from '@/components/ui/card';
import { TradingSession, CandleData } from '@/types/session';
import { PredictionResult } from '@/types/trading';
import SessionManager from '../session/SessionManager';
import NewCandleInput from '../NewCandleInput';
import PatternDetection from '../patterns/PatternDetection';
import PredictionDisplay from '../PredictionDisplay';
import WeightedPriceForecast from '../WeightedPriceForecast';
import TradingAnalytics from '../analytics/TradingAnalytics';

interface PredictionTabsProps {
  currentSession: TradingSession | null;
  candles: CandleData[];
  pair: string;
  timeframe: string;
  predictionResult: PredictionResult | null;
  onCandleSaved: (candleData: CandleData) => Promise<void>;
}

const PredictionTabs = memo(({ 
  currentSession, 
  candles, 
  pair, 
  timeframe, 
  predictionResult, 
  onCandleSaved 
}: PredictionTabsProps) => {
  return (
    <Tabs defaultValue="input" className="space-y-6">
      <TabsList className="grid w-full grid-cols-5 bg-background/95 backdrop-blur-sm border border-border/50 shadow-sm">
        <TabsTrigger 
          value="session" 
          className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground transition-all duration-200"
        >
          Управление сессией
        </TabsTrigger>
        <TabsTrigger 
          value="input" 
          className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground transition-all duration-200"
        >
          Ввод данных
        </TabsTrigger>
        <TabsTrigger 
          value="predictions" 
          className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground transition-all duration-200"
        >
          Прогнозы
        </TabsTrigger>
        <TabsTrigger 
          value="patterns" 
          className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground transition-all duration-200"
        >
          Паттерны
        </TabsTrigger>
        <TabsTrigger 
          value="analytics" 
          className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground transition-all duration-200"
        >
          Аналитика
        </TabsTrigger>
      </TabsList>

      <TabsContent value="session" className="animate-fade-in">
        <SessionManager pair={pair} />
      </TabsContent>

      <TabsContent value="input" className="animate-fade-in">
        {currentSession ? (
          <NewCandleInput 
            currentSession={currentSession}
            pair={pair}
            onCandleSaved={onCandleSaved}
          />
        ) : (
          <Card className="p-8 bg-card/30 border-border/50 text-center backdrop-blur-sm">
            <div className="text-muted-foreground">
              Создайте или загрузите сессию для начала ввода данных
            </div>
          </Card>
        )}
      </TabsContent>

      <TabsContent value="predictions" className="animate-fade-in">
        <div className="space-y-6">
          <PredictionDisplay candles={candles} currentSession={currentSession} />
          {predictionResult && (
            <div className="animate-scale-in">
              <WeightedPriceForecast 
                result={predictionResult} 
                pair={pair} 
                timeframe={timeframe} 
              />
            </div>
          )}
        </div>
      </TabsContent>

      <TabsContent value="patterns" className="animate-fade-in">
        <PatternDetection candles={candles} />
      </TabsContent>

      <TabsContent value="analytics" className="animate-fade-in">
        <TradingAnalytics candles={candles} />
      </TabsContent>
    </Tabs>
  );
});

PredictionTabs.displayName = 'PredictionTabs';

export default PredictionTabs;