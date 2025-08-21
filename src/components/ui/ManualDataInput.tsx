import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ManualDataInputs } from "@/types/trading";

interface ManualDataInputProps {
  data: ManualDataInputs;
  onChange: (data: ManualDataInputs) => void;
  pair: string;
}

const ManualDataInput = ({ data, onChange, pair }: ManualDataInputProps) => {
  const updateField = (field: keyof ManualDataInputs, value: string | number) => {
    onChange({
      ...data,
      [field]: value
    });
  };

  const formatPlaceholder = (type: string) => {
    if (pair === "BTC/USD") {
      switch (type) {
        case "open": return "67500";
        case "high": return "68000";
        case "low": return "67000";
        case "close": return "67800";
        default: return "";
      }
    } else if (pair.includes("JPY")) {
      switch (type) {
        case "open": return "149.50";
        case "high": return "149.85";
        case "low": return "149.20";
        case "close": return "149.70";
        default: return "";
      }
    } else {
      switch (type) {
        case "open": return "1.0850";
        case "high": return "1.0875";
        case "low": return "1.0830";
        case "close": return "1.0860";
        default: return "";
      }
    }
  };

  return (
    <Card className="p-6 bg-slate-700/30 border-slate-600">
      <h4 className="text-lg font-medium text-white mb-4">Пользовательский ввод данных</h4>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* OHLC данные */}
        <div className="space-y-4">
          <h5 className="text-white font-medium">Данные OHLC</h5>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="open" className="text-slate-300">Open (Открытие)</Label>
              <Input
                id="open"
                type="number"
                step="any"
                placeholder={formatPlaceholder("open")}
                value={data.open || ''}
                onChange={(e) => updateField('open', parseFloat(e.target.value) || 0)}
                className="bg-slate-800 border-slate-600 text-white"
              />
            </div>
            
            <div>
              <Label htmlFor="high" className="text-slate-300">High (Максимум)</Label>
              <Input
                id="high"
                type="number"
                step="any"
                placeholder={formatPlaceholder("high")}
                value={data.high || ''}
                onChange={(e) => updateField('high', parseFloat(e.target.value) || 0)}
                className="bg-slate-800 border-slate-600 text-white"
              />
            </div>
            
            <div>
              <Label htmlFor="low" className="text-slate-300">Low (Минимум)</Label>
              <Input
                id="low"
                type="number"
                step="any"
                placeholder={formatPlaceholder("low")}
                value={data.low || ''}
                onChange={(e) => updateField('low', parseFloat(e.target.value) || 0)}
                className="bg-slate-800 border-slate-600 text-white"
              />
            </div>
            
            <div>
              <Label htmlFor="close" className="text-slate-300">Close (Закрытие)</Label>
              <Input
                id="close"
                type="number"
                step="any"
                placeholder={formatPlaceholder("close")}
                value={data.close || ''}
                onChange={(e) => updateField('close', parseFloat(e.target.value) || 0)}
                className="bg-slate-800 border-slate-600 text-white"
              />
            </div>
          </div>
        </div>

        {/* Объем и время */}
        <div className="space-y-4">
          <h5 className="text-white font-medium">Дополнительные данные</h5>
          
          <div>
            <Label htmlFor="volume" className="text-slate-300">Volume (Объем)</Label>
            <Input
              id="volume"
              type="number"
              placeholder={pair === "BTC/USD" ? "5000" : "1500"}
              value={data.volume || ''}
              onChange={(e) => updateField('volume', parseFloat(e.target.value) || 0)}
              className="bg-slate-800 border-slate-600 text-white"
            />
          </div>
          
          <div>
            <Label htmlFor="date" className="text-slate-300">Дата анализа</Label>
            <Input
              id="date"
              type="date"
              value={data.date}
              onChange={(e) => updateField('date', e.target.value)}
              className="bg-slate-800 border-slate-600 text-white"
            />
          </div>
          
          <div>
            <Label htmlFor="time" className="text-slate-300">Время анализа</Label>
            <Input
              id="time"
              type="time"
              value={data.time}
              onChange={(e) => updateField('time', e.target.value)}
              className="bg-slate-800 border-slate-600 text-white"
            />
          </div>
        </div>
      </div>

      <div className="mt-4 p-3 bg-blue-600/20 border border-blue-600/50 rounded-lg">
        <p className="text-blue-200 text-sm">
          <strong>Поддержка исторических данных:</strong> Вы можете указать любую дату и время для анализа прошлых рыночных ситуаций.
        </p>
      </div>
    </Card>
  );
};

export default ManualDataInput;
