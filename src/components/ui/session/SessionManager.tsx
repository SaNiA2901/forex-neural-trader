
import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Database, Plus } from 'lucide-react';
import { TradingSession } from '@/types/session';
import { useStateManager } from '@/hooks/useStateManager';
import { useErrorHandler } from '@/hooks/useErrorHandler';
import { SessionForm } from './SessionForm';
import { SessionCard } from './SessionCard';

interface SessionCreationData {
  session_name: string;
  pair: string;
  timeframe: string;
  start_date: string;
  start_time: string;
}

interface SessionManagerProps {
  pair: string;
}

const SessionManager = ({ pair }: SessionManagerProps) => {
  const [showCreateForm, setShowCreateForm] = useState(false);
  const { sessions, currentSession, isLoading, createSession, loadSession, deleteSession } = useStateManager();
  const { addError } = useErrorHandler();

  const handleSessionOperation = async <T,>(
    operation: () => Promise<T>,
    operationName: string,
    errorMessage: string,
    onSuccess?: (result: T) => void
  ): Promise<T | null> => {
    try {
      console.log(`SessionManager: ${operationName} started`);
      const result = await operation();
      console.log(`SessionManager: ${operationName} completed successfully`);
      onSuccess?.(result);
      return result;
    } catch (error) {
      console.error(`SessionManager: ${operationName} failed:`, error);
      addError(errorMessage, error instanceof Error ? error.message : 'Unknown error');
      return null;
    }
  };

  const handleCreateSession = async (sessionData: SessionCreationData) => {
    await handleSessionOperation(
      () => createSession(sessionData),
      'Creating session',
      'Ошибка создания сессии',
      () => setShowCreateForm(false)
    );
  };

  const handleLoadSession = async (sessionId: string) => {
    await handleSessionOperation(
      () => loadSession(sessionId),
      `Loading session ${sessionId}`,
      'Ошибка загрузки сессии'
    );
  };

  const handleDeleteSession = async (sessionId: string) => {
    await handleSessionOperation(
      () => deleteSession(sessionId),
      `Deleting session ${sessionId}`,
      'Ошибка удаления сессии'
    );
  };

  const filteredSessions = sessions.filter(s => s.pair === pair);

  return (
    <Card className="p-6 bg-slate-800/50 border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Database className="h-6 w-6 text-blue-400" />
          <h3 className="text-xl font-semibold text-white">Управление сессиями ({pair})</h3>
          {currentSession && (
            <Badge className="bg-green-600 text-white">
              Активна: {currentSession.session_name}
            </Badge>
          )}
        </div>
        <Button 
          onClick={() => setShowCreateForm(!showCreateForm)}
          className="bg-green-600 hover:bg-green-700"
          disabled={isLoading}
        >
          <Plus className="h-4 w-4 mr-2" />
          Новая сессия
        </Button>
      </div>

      <SessionForm
        isOpen={showCreateForm}
        onClose={() => setShowCreateForm(false)}
        onSubmit={handleCreateSession}
        isSubmitting={isLoading}
      />

      {filteredSessions.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-white font-medium">Доступные сессии</h4>
          
          {filteredSessions.map(session => (
            <SessionCard
              key={session.id}
              session={session}
              isActive={currentSession?.id === session.id}
              onSelect={(s) => handleLoadSession(s.id)}
              onDelete={handleDeleteSession}
              onDuplicate={(id) => console.log('Duplicate:', id)}
            />
          ))}
        </div>
      )}

      {filteredSessions.length === 0 && !isLoading && (
        <div className="text-center py-8 text-slate-400">
          <Database className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>Нет сессий для {pair}</p>
          <p className="text-sm">Создайте новую сессию для начала работы</p>
        </div>
      )}
    </Card>
  );
};

export default SessionManager;
