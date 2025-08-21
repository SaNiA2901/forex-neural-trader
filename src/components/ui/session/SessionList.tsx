import React, { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Search, Plus, Filter, RefreshCw } from 'lucide-react';
import { SessionCard } from './SessionCard';
import { TradingSession } from '@/types/session';
import { useTradingStore } from '@/store/TradingStore';

interface SessionListProps {
  sessions: TradingSession[];
  currentSessionId?: string;
  isLoading?: boolean;
  searchTerm: string;
  sortBy: string;
  filterBy: string;
  onSearchChange: (term: string) => void;
  onSortChange: (sort: string) => void;
  onFilterChange: (filter: string) => void;
  onCreateSession: () => void;
  onSelectSession: (session: TradingSession) => void;
  onDeleteSession: (sessionId: string) => void;
  onDuplicateSession: (sessionId: string) => void;
  onRefresh: () => void;
}

export const SessionList: React.FC<SessionListProps> = ({
  sessions,
  currentSessionId,
  isLoading = false,
  searchTerm,
  sortBy,
  filterBy,
  onSearchChange,
  onSortChange,
  onFilterChange,
  onCreateSession,
  onSelectSession,
  onDeleteSession,
  onDuplicateSession,
  onRefresh
}) => {
  const { state } = useTradingStore();

  // Фильтрация и сортировка сессий
  const filteredAndSortedSessions = useMemo(() => {
    let filtered = sessions.filter(session => {
      const matchesSearch = session.session_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          session.pair.toLowerCase().includes(searchTerm.toLowerCase());
      
      if (filterBy === 'all') return matchesSearch;
      if (filterBy === 'active') return matchesSearch && session.id === currentSessionId;
      if (filterBy === 'recent') {
        const sessionDate = new Date(`${session.start_date}T${session.start_time}`);
        const dayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
        return matchesSearch && sessionDate > dayAgo;
      }
      
      return matchesSearch;
    });

    // Сортировка
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.session_name.localeCompare(b.session_name);
        case 'pair':
          return a.pair.localeCompare(b.pair);
        case 'timeframe':
          return a.timeframe.localeCompare(b.timeframe);
        case 'candles':
          const aCandleCount = state.candles.filter(c => c.session_id === a.id).length;
          const bCandleCount = state.candles.filter(c => c.session_id === b.id).length;
          return bCandleCount - aCandleCount;
        case 'date':
        default:
          const aDate = new Date(`${a.start_date}T${a.start_time}`);
          const bDate = new Date(`${b.start_date}T${b.start_time}`);
          return bDate.getTime() - aDate.getTime();
      }
    });

    return filtered;
  }, [sessions, searchTerm, sortBy, filterBy, currentSessionId, state.candles]);

  const getSessionStats = (sessionId: string) => {
    const sessionCandles = state.candles.filter(c => c.session_id === sessionId);
    return {
      candles: sessionCandles.length,
      lastPrice: sessionCandles.length > 0 ? sessionCandles[sessionCandles.length - 1].close : undefined
    };
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5" />
            Торговые сессии
            {sessions.length > 0 && (
              <span className="text-sm font-normal text-muted-foreground">
                ({filteredAndSortedSessions.length} из {sessions.length})
              </span>
            )}
          </CardTitle>
          
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={onRefresh}
              disabled={isLoading}
            >
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
            
            <Button
              size="sm"
              onClick={onCreateSession}
            >
              <Plus className="h-4 w-4 mr-1" />
              Создать
            </Button>
          </div>
        </div>

        {/* Фильтры и поиск */}
        <div className="space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Поиск по названию или паре..."
              value={searchTerm}
              onChange={(e) => onSearchChange(e.target.value)}
              className="pl-10"
            />
          </div>

          <div className="flex gap-2">
            <Select value={filterBy} onValueChange={onFilterChange}>
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Все сессии</SelectItem>
                <SelectItem value="active">Активные</SelectItem>
                <SelectItem value="recent">За сутки</SelectItem>
              </SelectContent>
            </Select>

            <Select value={sortBy} onValueChange={onSortChange}>
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="date">По дате</SelectItem>
                <SelectItem value="name">По названию</SelectItem>
                <SelectItem value="pair">По паре</SelectItem>
                <SelectItem value="timeframe">По таймфрейму</SelectItem>
                <SelectItem value="candles">По свечам</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-0">
        <div className="max-h-[600px] overflow-y-auto">
          {isLoading ? (
            <div className="p-6 text-center text-muted-foreground">
              <RefreshCw className="h-6 w-6 animate-spin mx-auto mb-2" />
              Загрузка сессий...
            </div>
          ) : filteredAndSortedSessions.length === 0 ? (
            <div className="p-6 text-center text-muted-foreground">
              {sessions.length === 0 ? (
                <div>
                  <div className="text-lg font-medium mb-2">Нет сессий</div>
                  <div>Создайте первую торговую сессию для начала работы</div>
                </div>
              ) : (
                <div>Нет сессий, соответствующих критериям поиска</div>
              )}
            </div>
          ) : (
            <div className="p-4 space-y-3">
              {filteredAndSortedSessions.map((session) => {
                const stats = getSessionStats(session.id);
                return (
                  <SessionCard
                    key={session.id}
                    session={session}
                    isActive={session.id === currentSessionId}
                    onSelect={onSelectSession}
                    onDelete={onDeleteSession}
                    onDuplicate={onDuplicateSession}
                    candles={stats.candles}
                    lastPrice={stats.lastPrice}
                  />
                );
              })}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};