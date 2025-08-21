
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface Stochastic {
  k: number;
  d: number;
}

interface StochasticIndicatorProps {
  stochastic: Stochastic;
}

const StochasticIndicator = ({ stochastic }: StochasticIndicatorProps) => {
  return (
    <Card className="p-6 bg-slate-800/50 border-slate-700">
      <h3 className="text-lg font-semibold text-white mb-4">Stochastic Oscillator</h3>
      
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-slate-400">%K</span>
            <span className="text-white">{stochastic.k.toFixed(1)}</span>
          </div>
          <Progress value={stochastic.k} className="h-2 bg-slate-700" />
        </div>
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-slate-400">%D</span>
            <span className="text-white">{stochastic.d.toFixed(1)}</span>
          </div>
          <Progress value={stochastic.d} className="h-2 bg-slate-700" />
        </div>
      </div>

      <div className="text-sm text-slate-300">
        <p>
          Состояние: {stochastic.k > 80 ? 'Перекуплен' : 
                     stochastic.k < 20 ? 'Перепродан' : 'Нейтральный'}
        </p>
      </div>
    </Card>
  );
};

export default StochasticIndicator;
