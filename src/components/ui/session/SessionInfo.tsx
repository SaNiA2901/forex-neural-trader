
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Database } from "lucide-react";

const SessionInfo = () => {
  return (
    <Card className="p-6 bg-slate-800/50 border-slate-700">
      <div className="flex items-center space-x-3 mb-4">
        <Database className="h-6 w-6 text-blue-400" />
        <h3 className="text-xl font-semibold text-white">Система торговых сессий</h3>
      </div>
      
      <div className="bg-slate-700/50 rounded-lg p-4">
        <ul className="text-slate-300 text-sm space-y-2">
          <li>• Создавайте сессии с автоматическим расчетом времени свечей</li>
          <li>• Каждая свеча сохраняется в базе данных в реальном времени</li>
          <li>• Возможность продолжить работу с любого места после сбоя</li>
          <li>• Автоматические прогнозы для каждой введенной свечи</li>
        </ul>
      </div>
    </Card>
  );
};

export default SessionInfo;
