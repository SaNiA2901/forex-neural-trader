
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";

interface RSIIndicatorProps {
  rsi: number;
}

const RSIIndicator = ({ rsi }: RSIIndicatorProps) => {
  const getRSIStatus = (rsi: number) => {
    if (rsi > 70) return { status: 'Перекуплен', color: 'bg-red-600', signal: 'SELL' };
    if (rsi < 30) return { status: 'Перепродан', color: 'bg-green-600', signal: 'BUY' };
    return { status: 'Нейтральный', color: 'bg-blue-600', signal: 'HOLD' };
  };

  const rsiStatus = getRSIStatus(rsi);

  return (
    <Card className="p-6 bg-slate-800/50 border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">RSI (Relative Strength Index)</h3>
        <Badge className={`${rsiStatus.color} text-white`}>
          {rsiStatus.signal}
        </Badge>
      </div>
      
      <div className="space-y-3">
        <div className="flex justify-between text-sm">
          <span className="text-slate-400">RSI (14)</span>
          <span className="text-white font-medium">{rsi.toFixed(1)}</span>
        </div>
        <Progress 
          value={rsi} 
          className="h-3 bg-slate-700"
        />
        <div className="flex justify-between text-xs text-slate-400">
          <span>Перепродан (30)</span>
          <span>Нейтральный (50)</span>
          <span>Перекуплен (70)</span>
        </div>
        <p className="text-sm text-slate-300">Статус: <span className="text-white">{rsiStatus.status}</span></p>
      </div>
    </Card>
  );
};

export default RSIIndicator;
