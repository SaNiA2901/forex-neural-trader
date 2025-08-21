
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Activity, AlertTriangle } from 'lucide-react';
import { CandleData } from '@/types/session';
import { usePatternDetection } from '@/hooks/usePatternDetection';

interface PatternDetectionProps {
  candles: CandleData[];
}

const PatternDetection = ({ candles }: PatternDetectionProps) => {
  const { detectedPatterns, hasPatterns } = usePatternDetection(candles, {
    maxPatterns: 12,
    minConfidence: 65
  });

  const getPatternIcon = (type: string) => {
    switch (type) {
      case 'BULLISH': return <TrendingUp className="h-4 w-4 text-green-400" />;
      case 'BEARISH': return <TrendingDown className="h-4 w-4 text-red-400" />;
      default: return <Activity className="h-4 w-4 text-yellow-400" />;
    }
  };

  const getPatternColor = (type: string) => {
    switch (type) {
      case 'BULLISH': return 'bg-green-600';
      case 'BEARISH': return 'bg-red-600';  
      default: return 'bg-yellow-600';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-green-400';
    if (confidence >= 70) return 'text-yellow-400';
    return 'text-orange-400';
  };

  if (candles.length === 0) {
    return (
      <Card className="p-8 bg-slate-700/30 border-slate-600 text-center">
        <Activity className="h-12 w-12 mx-auto mb-4 text-slate-500" />
        <p className="text-slate-400">
          Нет данных для анализа паттернов
        </p>
        <p className="text-sm text-slate-500 mt-2">
          Добавьте свечи для обнаружения торговых паттернов
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card className="p-6 bg-slate-800/50 border-slate-700">
        <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
          <Activity className="h-5 w-5 mr-2" />
          Обнаружение паттернов
        </h3>
        
        {hasPatterns ? (
          <div className="space-y-3">
            {detectedPatterns.map((pattern, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-slate-700/50 rounded-lg border border-slate-600">
                <div className="flex items-center space-x-4">
                  {getPatternIcon(pattern.type)}
                  
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <span className="text-white font-medium">{pattern.name}</span>
                      {pattern.confidence >= 80 && (
                        <AlertTriangle className="h-3 w-3 text-amber-400" />
                      )}
                    </div>
                    <p className="text-sm text-slate-400">
                      Тип: {pattern.type}
                    </p>
                    {pattern.description && (
                      <p className="text-xs text-slate-500 mt-1 max-w-md">
                        {pattern.description}
                      </p>
                    )}
                  </div>
                </div>
                
                <div className="flex items-center space-x-3">
                  <div className="text-right">
                    <div className={`text-sm font-medium ${getConfidenceColor(pattern.confidence)}`}>
                      {pattern.confidence.toFixed(1)}%
                    </div>
                    <div className="text-xs text-slate-400">
                      Индекс: {pattern.index}
                    </div>
                  </div>
                  
                  <Badge className={getPatternColor(pattern.type)}>
                    {pattern.type}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-6">
            <p className="text-slate-400">Паттерны не обнаружены</p>
            <p className="text-sm text-slate-500 mt-1">
              Добавьте больше свечей для анализа
            </p>
          </div>
        )}
      </Card>

      <Card className="p-6 bg-slate-800/50 border-slate-700">
        <h4 className="text-lg font-medium text-white mb-4">Статистика анализа</h4>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-white">{candles.length}</div>
            <div className="text-sm text-slate-400">Проанализировано свечей</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">{detectedPatterns.length}</div>
            <div className="text-sm text-slate-400">Найдено паттернов</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">
              {detectedPatterns.filter(p => p.type === 'BULLISH').length}
            </div>
            <div className="text-sm text-slate-400">Бычьих сигналов</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-red-400">
              {detectedPatterns.filter(p => p.type === 'BEARISH').length}
            </div>
            <div className="text-sm text-slate-400">Медвежьих сигналов</div>
          </div>

          <div className="text-center">
            <div className="text-2xl font-bold text-amber-400">
              {detectedPatterns.filter(p => p.confidence >= 80).length}
            </div>
            <div className="text-sm text-slate-400">Высокой уверенности</div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default PatternDetection;
