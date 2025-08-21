
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Play, Trash2, Copy, BarChart3 } from 'lucide-react';
import { TradingSession } from '@/types/session';
import { formatDistanceToNow } from 'date-fns';
import { ru } from 'date-fns/locale';

interface SessionCardProps {
  session: TradingSession;
  isActive?: boolean;
  onSelect: (session: TradingSession) => void;
  onDelete: (sessionId: string) => void;
  onDuplicate: (sessionId: string) => void;
  candles?: number;
  lastPrice?: number;
  className?: string;
}

export const SessionCard: React.FC<SessionCardProps> = ({
  session,
  isActive = false,
  onSelect,
  onDelete,
  onDuplicate,
  candles = 0,
  lastPrice,
  className = ''
}) => {
  const handleSelect = () => {
    onSelect(session);
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    onDelete(session.id);
  };

  const handleDuplicate = (e: React.MouseEvent) => {
    e.stopPropagation();
    onDuplicate(session.id);
  };

  const formatSessionTime = (date: string, time: string) => {
    try {
      const sessionDate = new Date(`${date}T${time}`);
      return formatDistanceToNow(sessionDate, { addSuffix: true, locale: ru });
    } catch {
      return 'Неизвестно';
    }
  };

  return (
    <Card 
      className={`cursor-pointer transition-all duration-200 hover:shadow-lg ${
        isActive 
          ? 'ring-2 ring-primary border-primary shadow-md' 
          : 'hover:border-primary/50'
      } ${className}`}
      onClick={handleSelect}
    >
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold truncate">
            {session.session_name}
          </CardTitle>
          {isActive && (
            <Badge variant="default" className="ml-2">
              Активна
            </Badge>
          )}
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Основная информация */}
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-muted-foreground">Пара:</span>
            <div className="font-medium">{session.pair}</div>
          </div>
          <div>
            <span className="text-muted-foreground">Таймфрейм:</span>
            <div className="font-medium">{session.timeframe}</div>
          </div>
        </div>

        {/* Статистика */}
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-muted-foreground">Свечей:</span>
            <div className="font-medium flex items-center">
              <BarChart3 className="h-3 w-3 mr-1" />
              {candles}
            </div>
          </div>
          <div>
            <span className="text-muted-foreground">Индекс:</span>
            <div className="font-medium">{session.current_candle_index}</div>
          </div>
        </div>

        {lastPrice && (
          <div className="text-sm">
            <span className="text-muted-foreground">Последняя цена:</span>
            <div className="font-medium text-lg">{lastPrice.toFixed(5)}</div>
          </div>
        )}

        {/* Время создания */}
        <div className="text-xs text-muted-foreground">
          Создана {formatSessionTime(session.start_date, session.start_time)}
        </div>

        {/* Действия */}
        <div className="flex gap-2 pt-2">
          <Button
            size="sm"
            onClick={handleSelect}
            className="flex-1"
            variant={isActive ? "secondary" : "default"}
          >
            <Play className="h-3 w-3 mr-1" />
            {isActive ? 'Открыта' : 'Открыть'}
          </Button>
          
          <Button
            size="sm"
            variant="outline"
            onClick={handleDuplicate}
          >
            <Copy className="h-3 w-3" />
          </Button>
          
          <Button
            size="sm"
            variant="destructive"
            onClick={handleDelete}
          >
            <Trash2 className="h-3 w-3" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};


