
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CandleData } from "@/types/session";

interface CandleHistoryProps {
  candles: CandleData[];
}

const CandleHistory = ({ candles }: CandleHistoryProps) => {
  if (candles.length === 0) return null;

  return (
    <Card className="p-6 bg-slate-800/50 border-slate-700">
      <h4 className="text-lg font-medium text-white mb-4">История свечей текущей сессии</h4>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {candles.slice(-6).map((candle, index) => (
          <div key={candle.id || index} className="bg-slate-700/50 rounded-lg p-3">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-slate-400">Свеча #{candle.candle_index + 1}</span>
              {candle.prediction_direction && (
                <Badge className={candle.prediction_direction === 'UP' ? 'bg-green-600' : 'bg-red-600'}>
                  {candle.prediction_direction}
                </Badge>
              )}
            </div>
            <div className="text-xs text-slate-300 space-y-1">
              <div>O: {candle.open} | H: {candle.high}</div>
              <div>L: {candle.low} | C: {candle.close}</div>
              <div>V: {candle.volume}</div>
              {candle.prediction_probability && (
                <div className="text-blue-300">
                  Вероятность: {candle.prediction_probability.toFixed(1)}%
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};

export default CandleHistory;
