
import { Badge } from '@/components/ui/badge';
import { Calendar } from 'lucide-react';
import { TradingSession } from '@/types/session';

interface CandleInputHeaderProps {
  currentSession: TradingSession;
  nextCandleIndex: number;
  pair: string;
  nextCandleTime?: string;
}

const CandleInputHeader = ({ 
  currentSession, 
  nextCandleIndex, 
  pair, 
  nextCandleTime 
}: CandleInputHeaderProps) => {
  return (
    <>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <h4 className="text-lg font-medium text-white">
            Ввод свечи #{nextCandleIndex + 1}
          </h4>
          <Badge className="bg-blue-600 text-white">{currentSession.timeframe}</Badge>
          <Badge className="bg-green-600 text-white">{pair}</Badge>
        </div>
      </div>

      {nextCandleTime && (
        <div className="mb-4 p-3 bg-blue-600/20 border border-blue-600/50 rounded-lg">
          <div className="flex items-center space-x-2 text-blue-200">
            <Calendar className="h-4 w-4" />
            <span>Время свечи:</span>
            <span className="font-mono">
              {new Date(nextCandleTime).toLocaleString('ru-RU', {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit'
              })}
            </span>
          </div>
        </div>
      )}
    </>
  );
};

export default CandleInputHeader;
