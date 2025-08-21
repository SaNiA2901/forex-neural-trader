import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { ModelStatistics } from "@/types/trading";
import { TrendingUp, Target, BarChart3, Activity } from "lucide-react";

interface ModelPerformanceChartProps {
  stats: ModelStatistics;
}

const ModelPerformanceChart = ({ stats }: ModelPerformanceChartProps) => {
  const getPerformanceLevel = (accuracy: number) => {
    if (accuracy >= 80) return { label: 'Отличная', color: 'bg-green-600', textColor: 'text-green-400' };
    if (accuracy >= 70) return { label: 'Хорошая', color: 'bg-blue-600', textColor: 'text-blue-400' };
    if (accuracy >= 60) return { label: 'Средняя', color: 'bg-yellow-600', textColor: 'text-yellow-400' };
    return { label: 'Требует улучшения', color: 'bg-red-600', textColor: 'text-red-400' };
  };

  const overallPerformance = getPerformanceLevel(stats.overallAccuracy);
  const callPerformance = getPerformanceLevel(stats.callAccuracy);
  const putPerformance = getPerformanceLevel(stats.putAccuracy);

  return (
    <Card className="p-6 bg-slate-800/50 border-slate-700">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Activity className="h-6 w-6 text-purple-400" />
          <h3 className="text-xl font-semibold text-white">Производительность модели</h3>
        </div>
        <Badge className={overallPerformance.color}>
          {overallPerformance.label}
        </Badge>
      </div>

      {/* Основные метрики */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="text-center p-4 bg-slate-700/50 rounded-lg">
          <div className={`text-4xl font-bold mb-2 ${overallPerformance.textColor}`}>
            {stats.overallAccuracy.toFixed(1)}%
          </div>
          <Progress value={stats.overallAccuracy} className="mb-3 h-3" />
          <div className="text-white font-medium">Общая точность</div>
          <div className="text-slate-400 text-sm">
            {stats.accurateCount} из {stats.totalPredictions} прогнозов
          </div>
        </div>

        <div className="text-center p-4 bg-slate-700/50 rounded-lg">
          <div className={`text-4xl font-bold mb-2 ${callPerformance.textColor}`}>
            {stats.callAccuracy.toFixed(1)}%
          </div>
          <Progress value={stats.callAccuracy} className="mb-3 h-3" />
          <div className="text-white font-medium flex items-center justify-center space-x-1">
            <TrendingUp className="h-4 w-4" />
            <span>CALL точность</span>
          </div>
          <Badge className="mt-2 bg-green-600/20 text-green-300">
            Бычьи сигналы
          </Badge>
        </div>

        <div className="text-center p-4 bg-slate-700/50 rounded-lg">
          <div className={`text-4xl font-bold mb-2 ${putPerformance.textColor}`}>
            {stats.putAccuracy.toFixed(1)}%
          </div>
          <Progress value={stats.putAccuracy} className="mb-3 h-3" />
          <div className="text-white font-medium flex items-center justify-center space-x-1">
            <Target className="h-4 w-4" />
            <span>PUT точность</span>
          </div>
          <Badge className="mt-2 bg-red-600/20 text-red-300">
            Медвежьи сигналы
          </Badge>
        </div>
      </div>

      {/* Адаптивные веса */}
      <div>
        <div className="flex items-center space-x-2 mb-4">
          <BarChart3 className="h-5 w-5 text-blue-400" />
          <h4 className="text-lg font-medium text-white">Адаптивные веса факторов</h4>
          <Badge variant="outline" className="text-xs">
            Самообучение
          </Badge>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {Object.entries(stats.currentWeights).map(([factor, weight]) => {
            const percentage = (weight as number) * 100;
            const getFactorIcon = (factorName: string) => {
              switch (factorName) {
                case 'technical': return '📊';
                case 'volume': return '📈';
                case 'momentum': return '⚡';
                case 'volatility': return '🌊';
                case 'pattern': return '🕯️';
                case 'trend': return '📉';
                default: return '📋';
              }
            };

            const getFactorName = (factorName: string) => {
              switch (factorName) {
                case 'technical': return 'Технический';
                case 'volume': return 'Объем';
                case 'momentum': return 'Моментум';
                case 'volatility': return 'Волатильность';
                case 'pattern': return 'Паттерны';
                case 'trend': return 'Тренд';
                default: return factorName;
              }
            };

            return (
              <div key={factor} className="bg-slate-700/30 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-lg">{getFactorIcon(factor)}</span>
                    <span className="text-sm text-slate-300 font-medium">
                      {getFactorName(factor)}
                    </span>
                  </div>
                  <Badge variant="outline" className="text-xs">
                    {percentage.toFixed(1)}%
                  </Badge>
                </div>
                <Progress value={percentage} className="h-2 mb-2" />
                <div className="text-xs text-slate-400">
                  {percentage > 25 ? 'Высокое влияние' : percentage > 15 ? 'Среднее влияние' : 'Низкое влияние'}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Рекомендации по улучшению */}
      <div className="mt-6 p-4 bg-blue-600/10 border border-blue-600/30 rounded-lg">
        <h5 className="text-white font-medium mb-2">Рекомендации для улучшения:</h5>
        <ul className="text-blue-200 text-sm space-y-1">
          {stats.overallAccuracy < 70 && (
            <li>• Добавьте больше исторических данных для обучения модели</li>
          )}
          {Math.abs(stats.callAccuracy - stats.putAccuracy) > 20 && (
            <li>• Модель показывает дисбаланс между CALL и PUT сигналами</li>
          )}
          {stats.totalPredictions < 50 && (
            <li>• Недостаточно данных для точной оценки производительности</li>
          )}
          <li>• Модель автоматически адаптирует веса факторов для улучшения точности</li>
        </ul>
      </div>
    </Card>
  );
};

export default ModelPerformanceChart;