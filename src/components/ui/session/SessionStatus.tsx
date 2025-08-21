
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Clock } from "lucide-react";
import { TradingSession } from "@/types/session";

interface SessionStatusProps {
  currentSession: TradingSession | null;
}

const SessionStatus = ({ currentSession }: SessionStatusProps) => {
  if (currentSession) {
    return (
      <div className="mb-4 p-3 bg-green-600/20 border border-green-600/50 rounded-lg">
        <div className="text-green-200 text-sm">
          ✓ Поля ввода данных свечей активированы (Сессия: {currentSession.session_name})
        </div>
      </div>
    );
  }

  return (
    <Card className="p-6 bg-slate-700/30 border-slate-600">
      <div className="text-center text-slate-400">
        <Clock className="h-16 w-16 mx-auto mb-4 text-slate-500" />
        <p>Выберите или создайте сессию для начала ввода данных свечей</p>
      </div>
    </Card>
  );
};

export default SessionStatus;
