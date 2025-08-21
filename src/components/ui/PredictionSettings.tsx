
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { PredictionConfig } from "@/types/trading";
import { Settings, Clock } from "lucide-react";

interface PredictionSettingsProps {
  config: PredictionConfig;
  onChange: (config: PredictionConfig) => void;
}

const PredictionSettings = ({ config, onChange }: PredictionSettingsProps) => {
  const predictionIntervals = [
    { value: 1, label: "1 минута" },
    { value: 3, label: "3 минуты" },
    { value: 5, label: "5 минут" },
    { value: 10, label: "10 минут" },
    { value: 15, label: "15 минут" },
    { value: 30, label: "30 минут" },
    { value: 60, label: "60 минут" }
  ];

  const updateConfig = (field: keyof PredictionConfig, value: any) => {
    onChange({
      ...config,
      [field]: value
    });
  };

  return (
    <Card className="p-6 bg-slate-800/50 border-slate-700">
      <div className="flex items-center space-x-3 mb-4">
        <Settings className="h-5 w-5 text-blue-400" />
        <h3 className="text-lg font-semibold text-white">Основные требования к прогнозам</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Целевая задача */}
        <div className="space-y-3">
          <h4 className="text-white font-medium">Целевая задача</h4>
          <div className="bg-slate-700/50 rounded-lg p-4">
            <Badge className="bg-orange-600 text-white mb-2">Бинарные опционы</Badge>
            <p className="text-slate-300 text-sm mb-2">
              Прогнозирование направления движения цены
            </p>
            <div className="flex items-center space-x-2 text-sm">
              <span className="text-green-400 font-bold">▲ Вверх</span>
              <span className="text-slate-400">или</span>
              <span className="text-red-400 font-bold">▼ Вниз</span>
            </div>
            <p className="text-slate-400 text-xs mt-2">
              *Критично: Определение направления, а не точного ценового диапазона
            </p>
          </div>
        </div>

        {/* Временные интервалы */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Clock className="h-4 w-4 text-blue-400" />
            <h4 className="text-white font-medium">Интервал прогноза</h4>
          </div>
          <Select 
            value={config.predictionInterval.toString()} 
            onValueChange={(value) => updateConfig('predictionInterval', parseInt(value))}
          >
            <SelectTrigger className="bg-slate-800 border-slate-600 text-white">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-slate-800 border-slate-600">
              {predictionIntervals.map(interval => (
                <SelectItem 
                  key={interval.value} 
                  value={interval.value.toString()}
                  className="text-white focus:bg-slate-700"
                >
                  {interval.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-slate-400 text-xs">
            Стандартные периоды для торговли бинарными опционами
          </p>
        </div>

        {/* Формат вывода */}
        <div className="space-y-3">
          <h4 className="text-white font-medium">Формат вывода</h4>
          <div className="bg-slate-700/50 rounded-lg p-4 space-y-2">
            <div className="flex items-center space-x-2">
              <div className="h-2 w-2 bg-blue-400 rounded-full"></div>
              <span className="text-slate-300 text-sm">Направление движения (▲/▼)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="h-2 w-2 bg-blue-400 rounded-full"></div>
              <span className="text-slate-300 text-sm">Вероятность в процентах</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="h-2 w-2 bg-blue-400 rounded-full"></div>
              <span className="text-slate-300 text-sm">Визуализация данных</span>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default PredictionSettings;
