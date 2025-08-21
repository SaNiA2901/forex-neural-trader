import React, { useState, useCallback, useEffect } from 'react';
import { SessionList } from './SessionList';
import { SessionForm } from './SessionForm';
import { useTradingStore } from '@/store/TradingStore';
import { useSessionActions } from '@/hooks/store/useSessionActions';
import { useStoreEventHandlers } from '@/hooks/store/useStoreEventHandlers';
import { useToast } from '@/hooks/use-toast';
import { TradingSession } from '@/types/session';

interface SessionFormData {
  session_name: string;
  pair: string;
  timeframe: string;
  start_date: string;
  start_time: string;
}

export const NewSessionManager: React.FC = () => {
  const { state } = useTradingStore();
  const { createSession, loadSession, deleteSession, loadSessions } = useSessionActions();
  const { toast } = useToast();
  
  // UI состояние
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('date');
  const [filterBy, setFilterBy] = useState('all');

  useStoreEventHandlers();

  // Загрузка сессий при монтировании
  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Обработчики событий
  const handleCreateSession = useCallback(() => {
    setIsFormOpen(true);
  }, []);

  const handleFormSubmit = useCallback(async (data: SessionFormData) => {
    setIsSubmitting(true);
    try {
      await createSession(data);
      toast({
        title: "Успех",
        description: "Сессия создана успешно"
      });
      setIsFormOpen(false);
    } catch (error) {
      toast({
        title: "Ошибка",
        description: "Не удалось создать сессию",
        variant: "destructive"
      });
    } finally {
      setIsSubmitting(false);
    }
  }, [createSession, toast]);

  const handleSelectSession = useCallback(async (session: TradingSession) => {
    try {
      await loadSession(session.id);
      toast({
        title: "Успех",
        description: `Сессия "${session.session_name}" загружена`
      });
    } catch (error) {
      toast({
        title: "Ошибка",
        description: "Не удалось загрузить сессию",
        variant: "destructive"
      });
    }
  }, [loadSession, toast]);

  const handleDeleteSession = useCallback(async (sessionId: string) => {
    try {
      await deleteSession(sessionId);
      toast({
        title: "Успех",
        description: "Сессия удалена"
      });
    } catch (error) {
      toast({
        title: "Ошибка",
        description: "Не удалось удалить сессию",
        variant: "destructive"
      });
    }
  }, [deleteSession, toast]);

  const handleDuplicateSession = useCallback(async (sessionId: string) => {
    const originalSession = state.sessions.find(s => s.id === sessionId);
    if (!originalSession) return;

    const duplicatedData: SessionFormData = {
      session_name: `${originalSession.session_name} (копия)`,
      pair: originalSession.pair,
      timeframe: originalSession.timeframe,
      start_date: new Date().toISOString().split('T')[0],
      start_time: new Date().toTimeString().split(' ')[0].slice(0, 5)
    };

    try {
      await createSession(duplicatedData);
      toast({
        title: "Успех",
        description: "Сессия скопирована"
      });
    } catch (error) {
      toast({
        title: "Ошибка",
        description: "Не удалось скопировать сессию",
        variant: "destructive"
      });
    }
  }, [state.sessions, createSession, toast]);

  const handleRefresh = useCallback(async () => {
    await loadSessions();
  }, [loadSessions]);

  return (
    <div className="space-y-6">
      <SessionList
        sessions={state.sessions}
        currentSessionId={state.currentSession?.id}
        isLoading={state.isLoading}
        searchTerm={searchTerm}
        sortBy={sortBy}
        filterBy={filterBy}
        onSearchChange={setSearchTerm}
        onSortChange={setSortBy}
        onFilterChange={setFilterBy}
        onCreateSession={handleCreateSession}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
        onDuplicateSession={handleDuplicateSession}
        onRefresh={handleRefresh}
      />

      <SessionForm
        isOpen={isFormOpen}
        onClose={() => setIsFormOpen(false)}
        onSubmit={handleFormSubmit}
        isSubmitting={isSubmitting}
      />
    </div>
  );
};
