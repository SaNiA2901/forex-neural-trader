-- Исправление критических проблем безопасности

-- 1. Включаем RLS для таблиц с торговыми данными (КРИТИЧНО!)
ALTER TABLE IF EXISTS trading_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS candle_data ENABLE ROW LEVEL SECURITY;

-- 2. Создаем безопасные политики для торговых сессий
CREATE POLICY "Users can view their own trading sessions" 
ON trading_sessions 
FOR SELECT 
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own trading sessions" 
ON trading_sessions 
FOR INSERT 
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own trading sessions" 
ON trading_sessions 
FOR UPDATE 
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own trading sessions" 
ON trading_sessions 
FOR DELETE 
USING (auth.uid() = user_id);

-- 3. Создаем безопасные политики для данных свечей
CREATE POLICY "Users can view candle data for their sessions" 
ON candle_data 
FOR SELECT 
USING (
  EXISTS (
    SELECT 1 FROM trading_sessions 
    WHERE trading_sessions.id = candle_data.session_id 
    AND trading_sessions.user_id = auth.uid()
  )
);

CREATE POLICY "Users can insert candle data for their sessions" 
ON candle_data 
FOR INSERT 
WITH CHECK (
  EXISTS (
    SELECT 1 FROM trading_sessions 
    WHERE trading_sessions.id = candle_data.session_id 
    AND trading_sessions.user_id = auth.uid()
  )
);

CREATE POLICY "Users can update candle data for their sessions" 
ON candle_data 
FOR UPDATE 
USING (
  EXISTS (
    SELECT 1 FROM trading_sessions 
    WHERE trading_sessions.id = candle_data.session_id 
    AND trading_sessions.user_id = auth.uid()
  )
);

CREATE POLICY "Users can delete candle data for their sessions" 
ON candle_data 
FOR DELETE 
USING (
  EXISTS (
    SELECT 1 FROM trading_sessions 
    WHERE trading_sessions.id = candle_data.session_id 
    AND trading_sessions.user_id = auth.uid()
  )
);

-- 4. Исправляем функции с небезопасным search_path
CREATE OR REPLACE FUNCTION public.update_trading_session_timestamp()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $function$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$function$;

CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $function$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$function$;

-- 5. Создаем индексы для оптимизации производительности
CREATE INDEX IF NOT EXISTS idx_trading_sessions_user_id ON trading_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_trading_sessions_pair ON trading_sessions(pair);
CREATE INDEX IF NOT EXISTS idx_trading_sessions_created_at ON trading_sessions(created_at);

CREATE INDEX IF NOT EXISTS idx_candle_data_session_id ON candle_data(session_id);
CREATE INDEX IF NOT EXISTS idx_candle_data_timestamp ON candle_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_candle_data_session_timestamp ON candle_data(session_id, timestamp);

-- 6. Добавляем ограничения на размер данных для предотвращения DoS атак
ALTER TABLE trading_sessions ADD CONSTRAINT trading_sessions_name_length CHECK (char_length(name) <= 100);
ALTER TABLE trading_sessions ADD CONSTRAINT trading_sessions_pair_length CHECK (char_length(pair) <= 20);