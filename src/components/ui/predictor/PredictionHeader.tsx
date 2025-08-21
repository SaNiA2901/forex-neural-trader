import { memo } from 'react';
import { TrendingUp, Activity, Brain } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface PredictionHeaderProps {
  pair: string;
  timeframe: string;
  isActive?: boolean;
}

const PredictionHeader = memo(({ pair, timeframe, isActive = false }: PredictionHeaderProps) => {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center space-x-4">
        <div className="relative">
          <div className="p-3 bg-gradient-to-br from-primary/20 to-primary/10 rounded-xl border border-primary/20 backdrop-blur-sm">
            <TrendingUp className="h-6 w-6 text-primary" />
          </div>
          {isActive && (
            <div className="absolute -top-1 -right-1 h-3 w-3 bg-green-500 rounded-full animate-pulse" />
          )}
        </div>
        <div>
          <h2 className="text-2xl font-bold text-foreground bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text">
            Предиктор бинарных опционов
          </h2>
          <div className="flex items-center space-x-2 mt-1">
            <span className="text-muted-foreground">Анализ и прогнозирование для</span>
            <Badge variant="outline" className="bg-primary/10 border-primary/30">
              {pair}
            </Badge>
            <Badge variant="outline" className="bg-secondary/10 border-secondary/30">
              {timeframe}
            </Badge>
          </div>
        </div>
      </div>
      
      <div className="flex items-center space-x-2">
        <div className="flex items-center space-x-1 px-3 py-1 bg-green-500/10 rounded-full border border-green-500/20">
          <Activity className="h-4 w-4 text-green-500" />
          <span className="text-sm text-green-500 font-medium">AI Активен</span>
        </div>
        <div className="flex items-center space-x-1 px-3 py-1 bg-blue-500/10 rounded-full border border-blue-500/20">
          <Brain className="h-4 w-4 text-blue-500" />
          <span className="text-sm text-blue-500 font-medium">ML Модель</span>
        </div>
      </div>
    </div>
  );
});

PredictionHeader.displayName = 'PredictionHeader';

export default PredictionHeader;