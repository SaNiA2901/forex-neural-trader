
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { PredictionResult } from "@/types/trading";
import { TrendingUp, TrendingDown, Target, Clock, BarChart } from "lucide-react";

interface PredictionResultsProps {
  result: PredictionResult;
  pair: string;
}

const PredictionResults = ({ result, pair }: PredictionResultsProps) => {
  const getDirectionIcon = () => {
    return result.direction === 'UP' ? 
      <TrendingUp className="h-6 w-6 text-green-400" /> : 
      <TrendingDown className="h-6 w-6 text-red-400" />;
  };

  const getDirectionColor = () => {
    return result.direction === 'UP' ? 'text-green-400' : 'text-red-400';
  };

  const getDirectionBadgeColor = () => {
    return result.direction === 'UP' ? 'bg-green-600' : 'bg-red-600';
  };

  return (
    <div className="space-y-6">
      {/* Основной результат */}
      <Card className="p-6 bg-slate-800/50 border-slate-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold text-white">Результат прогноза</h3>
          <Badge className="bg-blue-600 text-white">
            {pair} • {result.interval} мин
          </Badge>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          {/* Направление */}
          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <div className={`text-3xl font-bold mb-2 flex items-center justify-center space-x-2 ${getDirectionColor()}`}>
              {getDirectionIcon()}
              <span>{result.direction === 'UP' ? '▲' : '▼'}</span>
            </div>
            <p className="text-slate-400 text-sm">Направление</p>
            <Badge className={`mt-2 ${getDirectionBadgeColor()}`}>
              {result.direction === 'UP' ? 'CALL' : 'PUT'}
            </Badge>
          </div>

          {/* Вероятность */}
          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <div className="text-3xl font-bold text-white mb-2">
              {result.probability.toFixed(1)}%
            </div>
            <Progress value={result.probability} className="mb-2 h-2" />
            <p className="text-slate-400 text-sm">Вероятность</p>
          </div>

          {/* Уверенность */}
          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <div className="text-3xl font-bold text-blue-400 mb-2">
              {result.confidence.toFixed(1)}%
            </div>
            <Progress value={result.confidence} className="mb-2 h-2" />
            <p className="text-slate-400 text-sm">Уверенность</p>
          </div>

          {/* Время */}
          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <div className="text-3xl font-bold text-white mb-2 flex items-center justify-center space-x-1">
              <Clock className="h-6 w-6" />
              <span>{result.interval}</span>
            </div>
            <p className="text-slate-400 text-sm">Минут</p>
          </div>
        </div>

        {/* Рекомендация */}
        <div className="bg-blue-600/20 border border-blue-600/50 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Target className="h-4 w-4 text-blue-400" />
            <span className="text-white font-medium">Торговая рекомендация:</span>
          </div>
          <p className="text-blue-200">{result.recommendation}</p>
        </div>
      </Card>

      {/* Детализация факторов */}
      <Card className="p-6 bg-slate-800/50 border-slate-700">
        <div className="flex items-center space-x-3 mb-4">
          <BarChart className="h-5 w-5 text-blue-400" />
          <h3 className="text-lg font-semibold text-white">Анализ факторов влияния</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="space-y-3">
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-slate-400">Технический анализ</span>
                <span className="text-white">{result.factors.technical.toFixed(1)}/100</span>
              </div>
              <Progress value={result.factors.technical} className="h-2" />
              <p className="text-xs text-slate-400 mt-1">RSI, MACD, Bollinger Bands</p>
            </div>

            <div>
              <div className="flex justify-between mb-2">
                <span className="text-slate-400">Моментум</span>
                <span className="text-white">{result.factors.momentum.toFixed(1)}/100</span>
              </div>
              <Progress value={result.factors.momentum} className="h-2" />
              <p className="text-xs text-slate-400 mt-1">EMA, Stochastic</p>
            </div>
          </div>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-slate-400">Объем торгов</span>
                <span className="text-white">{result.factors.volume.toFixed(1)}/100</span>
              </div>
              <Progress value={result.factors.volume} className="h-2" />
              <p className="text-xs text-slate-400 mt-1">Анализ активности</p>
            </div>

            <div>
              <div className="flex justify-between mb-2">
                <span className="text-slate-400">Волатильность</span>
                <span className="text-white">{result.factors.volatility.toFixed(1)}/100</span>
              </div>
              <Progress value={result.factors.volatility} className="h-2" />
              <p className="text-xs text-slate-400 mt-1">ATR, изменчивость</p>
            </div>
          </div>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-slate-400">Паттерны</span>
                <span className="text-white">{result.factors.pattern?.toFixed(1) || '50.0'}/100</span>
              </div>
              <Progress value={result.factors.pattern || 50} className="h-2" />
              <p className="text-xs text-slate-400 mt-1">Свечные паттерны</p>
            </div>

            <div>
              <div className="flex justify-between mb-2">
                <span className="text-slate-400">Тренд</span>
                <span className="text-white">{result.factors.trend?.toFixed(1) || '50.0'}/100</span>
              </div>
              <Progress value={result.factors.trend || 50} className="h-2" />
              <p className="text-xs text-slate-400 mt-1">ADX, направление</p>
            </div>
          </div>
        </div>
      </Card>

      {/* Таблица результатов */}
      <Card className="p-6 bg-slate-800/50 border-slate-700">
        <h3 className="text-lg font-semibold text-white mb-4">Сводная таблица</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-600">
                <th className="text-left text-slate-400 py-2">Параметр</th>
                <th className="text-left text-slate-400 py-2">Значение</th>
                <th className="text-left text-slate-400 py-2">Описание</th>
              </tr>
            </thead>
            <tbody className="text-white">
              <tr className="border-b border-slate-700">
                <td className="py-2">Валютная пара</td>
                <td className="py-2 font-medium">{pair}</td>
                <td className="py-2 text-slate-400">Анализируемый актив</td>
              </tr>
              <tr className="border-b border-slate-700">
                <td className="py-2">Направление</td>
                <td className={`py-2 font-bold ${getDirectionColor()}`}>
                  {result.direction === 'UP' ? '▲ ВВЕРХ' : '▼ ВНИЗ'}
                </td>
                <td className="py-2 text-slate-400">Прогнозируемое движение</td>
              </tr>
              <tr className="border-b border-slate-700">
                <td className="py-2">Вероятность</td>
                <td className="py-2 font-medium">{result.probability.toFixed(1)}%</td>
                <td className="py-2 text-slate-400">Точность прогноза</td>
              </tr>
              <tr className="border-b border-slate-700">
                <td className="py-2">Временной интервал</td>
                <td className="py-2 font-medium">{result.interval} минут</td>
                <td className="py-2 text-slate-400">Период действия прогноза</td>
              </tr>
              <tr>
                <td className="py-2">Тип опциона</td>
                <td className="py-2 font-medium">
                  <Badge className={getDirectionBadgeColor()}>
                    {result.direction === 'UP' ? 'CALL' : 'PUT'}
                  </Badge>
                </td>
                <td className="py-2 text-slate-400">Рекомендуемый тип бинарного опциона</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
};

export default PredictionResults;
