
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface MovingAveragesIndicatorProps {
  ma20: number;
  ma50: number;
}

const MovingAveragesIndicator = ({ ma20, ma50 }: MovingAveragesIndicatorProps) => {
  const getMAStatus = () => {
    if (ma20 > ma50) return { signal: 'BUY', color: 'text-green-400' };
    return { signal: 'SELL', color: 'text-red-400' };
  };

  const maStatus = getMAStatus();

  return (
    <Card className="p-6 bg-slate-800/50 border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Moving Averages</h3>
        <Badge className={`${maStatus.signal === 'BUY' ? 'bg-green-600' : 'bg-red-600'} text-white`}>
          {maStatus.signal}
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="text-center p-4 bg-slate-700/50 rounded-lg">
          <p className="text-slate-400 text-sm mb-1">MA 20</p>
          <p className="text-white font-semibold text-lg">
            {ma20.toFixed(4)}
          </p>
        </div>
        <div className="text-center p-4 bg-slate-700/50 rounded-lg">
          <p className="text-slate-400 text-sm mb-1">MA 50</p>
          <p className="text-white font-semibold text-lg">
            {ma50.toFixed(4)}
          </p>
        </div>
      </div>

      <div className="mt-4 text-sm text-slate-300">
        <p>
          Состояние: MA20 {ma20 > ma50 ? 'выше' : 'ниже'} MA50 - 
          <span className={maStatus.color}> {maStatus.signal === 'BUY' ? 'Бычий тренд' : 'Медвежий тренд'}</span>
        </p>
      </div>
    </Card>
  );
};

export default MovingAveragesIndicator;
