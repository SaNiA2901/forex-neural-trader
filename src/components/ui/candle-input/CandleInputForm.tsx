import React, { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Card } from '@/components/ui/card';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertCircle, 
  CheckCircle,
  Calculator,
  Clock,
  BarChart3
} from 'lucide-react';
import { TradingSession } from '@/types/session';

interface CandleInputFormProps {
  currentSession: TradingSession;
  pair: string;
  formData: any;
  errors: any;
  isValid: boolean;
  isSubmitting: boolean;
  onInputChange: (field: string, value: string) => void;
  onSubmit: () => void;
  onReset: () => void;
}

export const CandleInputForm: React.FC<CandleInputFormProps> = ({
  currentSession,
  pair,
  formData,
  errors,
  isValid,
  isSubmitting,
  onInputChange,
  onSubmit,
  onReset
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Определение направления свечи
  const getCandleDirection = useCallback(() => {
    if (!formData.open || !formData.close) return 'neutral';
    const open = parseFloat(formData.open);
    const close = parseFloat(formData.close);
    
    if (close > open) return 'bullish';
    if (close < open) return 'bearish';
    return 'doji';
  }, [formData.open, formData.close]);

  // Расчет прогресса заполнения
  const getCompletionProgress = useCallback(() => {
    const requiredFields = ['open', 'high', 'low', 'close', 'volume'];
    const filledFields = requiredFields.filter(field => 
      formData[field] && formData[field].toString().trim() !== ''
    );
    return Math.round((filledFields.length / requiredFields.length) * 100);
  }, [formData]);

  const direction = getCandleDirection();
  const completionProgress = getCompletionProgress();

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    onSubmit();
  }, [onSubmit]);

  return (
    <Card className="p-6 bg-gradient-to-br from-background/95 to-background/80 border-border/50">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-lg">
            <BarChart3 className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">Ввод свечи</h3>
            <p className="text-sm text-muted-foreground">{pair} · {currentSession.timeframe}</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          {direction !== 'neutral' && (
            <Badge 
              variant={direction === 'bullish' ? 'default' : direction === 'bearish' ? 'destructive' : 'secondary'}
              className="flex items-center gap-1"
            >
              {direction === 'bullish' && <TrendingUp className="h-3 w-3" />}
              {direction === 'bearish' && <TrendingDown className="h-3 w-3" />}
              {direction === 'doji' && <AlertCircle className="h-3 w-3" />}
              {direction === 'bullish' ? 'Бычья' : direction === 'bearish' ? 'Медвежья' : 'Доджи'}
            </Badge>
          )}
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            <Calculator className="h-4 w-4 mr-1" />
            {showAdvanced ? 'Простой' : 'Расширенный'}
          </Button>
        </div>
      </div>

      {/* Прогресс заполнения */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <Label className="text-sm">Прогресс заполнения</Label>
          <span className="text-sm text-muted-foreground">{completionProgress}%</span>
        </div>
        <Progress value={completionProgress} className="h-2" />
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Основные поля OHLC */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="space-y-2">
            <Label htmlFor="open" className="flex items-center gap-1">
              Open
              {errors.open && <AlertCircle className="h-3 w-3 text-destructive" />}
            </Label>
            <Input
              id="open"
              type="number"
              step="0.00001"
              value={formData.open}
              onChange={(e) => onInputChange('open', e.target.value)}
              className={errors.open ? 'border-destructive' : ''}
              placeholder="Цена открытия"
            />
            {errors.open && (
              <p className="text-xs text-destructive">{errors.open}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="high" className="flex items-center gap-1">
              High
              {errors.high && <AlertCircle className="h-3 w-3 text-destructive" />}
            </Label>
            <Input
              id="high"
              type="number"
              step="0.00001"
              value={formData.high}
              onChange={(e) => onInputChange('high', e.target.value)}
              className={errors.high ? 'border-destructive' : ''}
              placeholder="Максимальная цена"
            />
            {errors.high && (
              <p className="text-xs text-destructive">{errors.high}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="low" className="flex items-center gap-1">
              Low
              {errors.low && <AlertCircle className="h-3 w-3 text-destructive" />}
            </Label>
            <Input
              id="low"
              type="number"
              step="0.00001"
              value={formData.low}
              onChange={(e) => onInputChange('low', e.target.value)}
              className={errors.low ? 'border-destructive' : ''}
              placeholder="Минимальная цена"
            />
            {errors.low && (
              <p className="text-xs text-destructive">{errors.low}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="close" className="flex items-center gap-1">
              Close
              {errors.close && <AlertCircle className="h-3 w-3 text-destructive" />}
            </Label>
            <Input
              id="close"
              type="number"
              step="0.00001"
              value={formData.close}
              onChange={(e) => onInputChange('close', e.target.value)}
              className={errors.close ? 'border-destructive' : ''}
              placeholder="Цена закрытия"
            />
            {errors.close && (
              <p className="text-xs text-destructive">{errors.close}</p>
            )}
          </div>
        </div>

        {/* Объем */}
        <div className="space-y-2">
          <Label htmlFor="volume" className="flex items-center gap-1">
            Объем
            {errors.volume && <AlertCircle className="h-3 w-3 text-destructive" />}
          </Label>
          <Input
            id="volume"
            type="number"
            step="1"
            value={formData.volume}
            onChange={(e) => onInputChange('volume', e.target.value)}
            className={errors.volume ? 'border-destructive' : ''}
            placeholder="Объем торгов"
          />
          {errors.volume && (
            <p className="text-xs text-destructive">{errors.volume}</p>
          )}
        </div>

        {/* Расширенные поля */}
        {showAdvanced && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 bg-muted/30 rounded-lg">
            <div className="space-y-2">
              <Label htmlFor="timestamp">Время свечи</Label>
              <Input
                id="timestamp"
                type="datetime-local"
                value={formData.timestamp}
                onChange={(e) => onInputChange('timestamp', e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="spread">Спред</Label>
              <Input
                id="spread"
                type="number"
                step="0.00001"
                value={formData.spread || ''}
                onChange={(e) => onInputChange('spread', e.target.value)}
                placeholder="Спред (опционально)"
              />
            </div>
          </div>
        )}

        {/* Кнопки действий */}
        <div className="flex items-center justify-between pt-4">
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={onReset}
            >
              Очистить
            </Button>
          </div>

          <Button
            type="submit"
            disabled={!isValid || isSubmitting}
            className="min-w-[120px]"
          >
            {isSubmitting ? (
              <>
                <Clock className="h-4 w-4 mr-2 animate-spin" />
                Сохранение...
              </>
            ) : (
              <>
                <CheckCircle className="h-4 w-4 mr-2" />
                Добавить свечу
              </>
            )}
          </Button>
        </div>
      </form>
    </Card>
  );
};