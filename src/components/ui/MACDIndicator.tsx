
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { ChartDataPoint } from "@/hooks/useIndicatorData";

interface MACDData {
  value: number;
  signal: number;
  histogram: number;
}

interface MACDIndicatorProps {
  macd: MACDData;
  chartData: ChartDataPoint[];
}

const MACDIndicator = ({ macd, chartData }: MACDIndicatorProps) => {
  const getMACDStatus = (macd: MACDData) => {
    if (macd.value > macd.signal) return { signal: 'BUY', color: 'text-green-400' };
    return { signal: 'SELL', color: 'text-red-400' };
  };

  const macdStatus = getMACDStatus(macd);

  return (
    <Card className="p-6 bg-slate-800/50 border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">MACD</h3>
        <Badge className={`${macdStatus.signal === 'BUY' ? 'bg-green-600' : 'bg-red-600'} text-white`}>
          {macdStatus.signal}
        </Badge>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="text-center">
          <p className="text-slate-400 text-sm">MACD</p>
          <p className={`font-medium ${macdStatus.color}`}>
            {macd.value.toFixed(4)}
          </p>
        </div>
        <div className="text-center">
          <p className="text-slate-400 text-sm">Signal</p>
          <p className="text-white font-medium">
            {macd.signal.toFixed(4)}
          </p>
        </div>
        <div className="text-center">
          <p className="text-slate-400 text-sm">Histogram</p>
          <p className={`font-medium ${macd.histogram > 0 ? 'text-green-400' : 'text-red-400'}`}>
            {macd.histogram.toFixed(4)}
          </p>
        </div>
      </div>

      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" stroke="#9CA3AF" fontSize={10} />
            <YAxis stroke="#9CA3AF" fontSize={10} />
            <Tooltip 
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#F9FAFB'
              }}
            />
            <Line type="monotone" dataKey="macd" stroke="#3B82F6" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="signal" stroke="#EF4444" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
};

export default MACDIndicator;
