import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { PredictionResult } from "@/types/trading";
import { TrendingUp, TrendingDown, BarChart3, Target, Brain } from "lucide-react";

interface WeightedPriceForecastProps {
  result: PredictionResult;
  pair: string;
  timeframe: string;
}

const WeightedPriceForecast = ({ result, pair, timeframe }: WeightedPriceForecastProps) => {
  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= 85) return { label: 'Очень высокая', color: 'text-green-400' };
    if (confidence >= 70) return { label: 'Высокая', color: 'text-blue-400' };
    if (confidence >= 60) return { label: 'Средняя', color: 'text-yellow-400' };
    return { label: 'Низкая', color: 'text-red-400' };
  };

  const confidenceLevel = getConfidenceLevel(result.confidence);

  const factorWeights = [
    { name: 'Технический', value: result.factors.technical, weight: 30, color: 'bg-blue-500' },
    { name: 'Моментум', value: result.factors.momentum, weight: 25, color: 'bg-green-500' },
    { name: 'Объем', value: result.factors.volume, weight: 20, color: 'bg-purple-500' },
    { name: 'Волатильность', value: result.factors.volatility, weight: 15, color: 'bg-orange-500' },
    { name: 'Паттерны', value: result.factors.pattern || 50, weight: 7, color: 'bg-pink-500' },
    { name: 'Тренд', value: result.factors.trend || 50, weight: 3, color: 'bg-indigo-500' },
  ];

  const weightedScore = factorWeights.reduce((sum, factor) => 
    sum + (factor.value * factor.weight / 100), 0
  );

  return (
    <Card className="p-6 bg-slate-800/50 border-slate-700">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Brain className="h-6 w-6 text-purple-400" />
          <h3 className="text-xl font-semibold text-white">Взвешенный прогноз</h3>
        </div>
        <Badge className="bg-purple-600 text-white">
          {pair} • {timeframe}
        </Badge>
      </div>

      {/* Основной результат */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="text-center p-4 bg-slate-700/50 rounded-lg">
          <div className={`text-4xl font-bold mb-2 flex items-center justify-center space-x-2 ${
            result.direction === 'UP' ? 'text-green-400' : 'text-red-400'
          }`}>
            {result.direction === 'UP' ? 
              <TrendingUp className="h-8 w-8" /> : 
              <TrendingDown className="h-8 w-8" />
            }
            <span>{result.direction}</span>
          </div>
          <p className="text-slate-400 text-sm">Направление</p>
          <Badge className={`mt-2 ${
            result.direction === 'UP' ? 'bg-green-600' : 'bg-red-600'
          }`}>
            {result.direction === 'UP' ? 'CALL опцион' : 'PUT опцион'}
          </Badge>
        </div>

        <div className="text-center p-4 bg-slate-700/50 rounded-lg">
          <div className="text-4xl font-bold text-white mb-2">
            {result.probability.toFixed(1)}%
          </div>
          <Progress value={result.probability} className="mb-2 h-3" />
          <p className="text-slate-400 text-sm">Вероятность</p>
        </div>

        <div className="text-center p-4 bg-slate-700/50 rounded-lg">
          <div className={`text-4xl font-bold mb-2 ${confidenceLevel.color}`}>
            {result.confidence.toFixed(1)}%
          </div>
          <Progress value={result.confidence} className="mb-2 h-3" />
          <p className="text-slate-400 text-sm">Уверенность - {confidenceLevel.label}</p>
        </div>
      </div>

      {/* Взвешенный анализ факторов */}
      <div className="mb-6">
        <div className="flex items-center space-x-2 mb-4">
          <BarChart3 className="h-5 w-5 text-blue-400" />
          <h4 className="text-lg font-medium text-white">Взвешенный анализ факторов</h4>
          <Badge variant="outline" className="text-xs">
            Итоговый балл: {weightedScore.toFixed(1)}
          </Badge>
        </div>

        <div className="space-y-3">
          {factorWeights.map((factor, index) => (
            <div key={index} className="bg-slate-700/30 rounded-lg p-3">
              <div className="flex justify-between items-center mb-2">
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${factor.color}`}></div>
                  <span className="text-slate-300 font-medium">{factor.name}</span>
                  <Badge variant="outline" className="text-xs">
                    Вес: {factor.weight}%
                  </Badge>
                </div>
                <div className="text-white font-bold">
                  {factor.value.toFixed(1)}/100
                </div>
              </div>
              <Progress value={factor.value} className="h-2 mb-1" />
              <div className="text-xs text-slate-400">
                Взвешенное влияние: {(factor.value * factor.weight / 100).toFixed(1)} баллов
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Рекомендация */}
      <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-blue-600/50 rounded-lg p-4">
        <div className="flex items-center space-x-2 mb-2">
          <Target className="h-5 w-5 text-blue-400" />
          <span className="text-white font-medium">Торговая рекомендация:</span>
        </div>
        <p className="text-blue-200 mb-3">{result.recommendation}</p>
        <div className="text-xs text-slate-400">
          * Рекомендация основана на взвешенном анализе {factorWeights.length} ключевых факторов
        </div>
      </div>
    </Card>
  );
};

export default WeightedPriceForecast;