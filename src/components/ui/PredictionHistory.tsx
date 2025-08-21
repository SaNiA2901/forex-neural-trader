import { useState, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { TrendingUp, TrendingDown, Search, Filter, Clock, Target } from 'lucide-react';
import { CandleData } from '@/types/session';

interface PredictionHistoryProps {
  candles: CandleData[];
}

const PredictionHistory = ({ candles }: PredictionHistoryProps) => {
  const [filter, setFilter] = useState<'all' | 'up' | 'down'>('all');
  const [sortBy, setSortBy] = useState<'date' | 'probability' | 'confidence'>('date');
  const [searchTerm, setSearchTerm] = useState('');

  const filteredCandles = useMemo(() => {
    let filtered = candles.filter(candle => 
      candle.prediction_direction && candle.prediction_probability
    );

    // Фильтрация по направлению
    if (filter !== 'all') {
      filtered = filtered.filter(candle => 
        candle.prediction_direction?.toLowerCase() === filter
      );
    }

    // Поиск по индексу свечи
    if (searchTerm) {
      filtered = filtered.filter(candle => 
        candle.candle_index.toString().includes(searchTerm)
      );
    }

    // Сортировка
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'probability':
          return (b.prediction_probability || 0) - (a.prediction_probability || 0);
        case 'confidence':
          return (b.prediction_confidence || 0) - (a.prediction_confidence || 0);
        default:
          return b.candle_index - a.candle_index;
      }
    });

    return filtered;
  }, [candles, filter, sortBy, searchTerm]);

  const stats = useMemo(() => {
    const total = filteredCandles.length;
    const upPredictions = filteredCandles.filter(c => c.prediction_direction === 'UP').length;
    const avgProbability = total > 0 
      ? filteredCandles.reduce((sum, c) => sum + (c.prediction_probability || 0), 0) / total
      : 0;
    const avgConfidence = total > 0
      ? filteredCandles.reduce((sum, c) => sum + (c.prediction_confidence || 0), 0) / total
      : 0;

    return { total, upPredictions, avgProbability, avgConfidence };
  }, [filteredCandles]);

  if (candles.length === 0) {
    return (
      <Card className="p-8 bg-card/30 border-border/50 text-center backdrop-blur-sm">
        <div className="text-muted-foreground">
          История прогнозов пуста. Добавьте свечи для получения прогнозов.
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Статистика */}
      <Card className="p-6 bg-gradient-to-r from-card/50 to-card/30 border-border/50 backdrop-blur-sm">
        <h3 className="text-lg font-semibold text-foreground mb-4">Статистика прогнозов</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-primary">{stats.total}</div>
            <div className="text-sm text-muted-foreground">Всего прогнозов</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-500">{stats.upPredictions}</div>
            <div className="text-sm text-muted-foreground">CALL сигналов</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-500">{stats.avgProbability.toFixed(1)}%</div>
            <div className="text-sm text-muted-foreground">Средняя вероятность</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-500">{stats.avgConfidence.toFixed(1)}%</div>
            <div className="text-sm text-muted-foreground">Средняя уверенность</div>
          </div>
        </div>
      </Card>

      {/* Фильтры */}
      <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center space-x-2">
            <Search className="h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Поиск по номеру свечи..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-48 bg-background/50"
            />
          </div>
          
          <div className="flex items-center space-x-2">
            <Filter className="h-4 w-4 text-muted-foreground" />
            <Select value={filter} onValueChange={(value: 'all' | 'up' | 'down') => setFilter(value)}>
              <SelectTrigger className="w-40 bg-background/50">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Все прогнозы</SelectItem>
                <SelectItem value="up">Только CALL</SelectItem>
                <SelectItem value="down">Только PUT</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Select value={sortBy} onValueChange={(value: 'date' | 'probability' | 'confidence') => setSortBy(value)}>
            <SelectTrigger className="w-48 bg-background/50">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="date">Сортировка по дате</SelectItem>
              <SelectItem value="probability">По вероятности</SelectItem>
              <SelectItem value="confidence">По уверенности</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </Card>

      {/* История */}
      <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
        <h3 className="text-lg font-semibold text-foreground mb-4">История прогнозов</h3>
        {filteredCandles.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            Нет прогнозов соответствующих выбранным фильтрам
          </div>
        ) : (
          <div className="grid gap-4">
            {filteredCandles.map((candle) => (
              <div key={candle.id} className="flex items-center justify-between p-4 bg-background/30 rounded-lg border border-border/50 hover:bg-background/50 transition-colors">
                <div className="flex items-center space-x-4">
                  <Badge variant="outline" className="bg-secondary/10">
                    Свеча #{candle.candle_index + 1}
                  </Badge>
                  <Badge className={
                    candle.prediction_direction === 'UP' 
                      ? 'bg-green-500/20 text-green-400 border-green-500/30' 
                      : 'bg-red-500/20 text-red-400 border-red-500/30'
                  }>
                    <div className="flex items-center space-x-1">
                      {candle.prediction_direction === 'UP' ? (
                        <TrendingUp className="h-3 w-3" />
                      ) : (
                        <TrendingDown className="h-3 w-3" />
                      )}
                      <span>{candle.prediction_direction}</span>
                    </div>
                  </Badge>
                </div>

                <div className="flex items-center space-x-6">
                  <div className="flex items-center space-x-2">
                    <Target className="h-4 w-4 text-blue-400" />
                    <span className="text-sm text-blue-400">
                      {candle.prediction_probability?.toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <Clock className="h-4 w-4 text-purple-400" />
                    <span className="text-sm text-purple-400">
                      {candle.prediction_confidence?.toFixed(1)}%
                    </span>
                  </div>

                  <div className="text-sm text-muted-foreground">
                    {candle.close}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
};

export default PredictionHistory;