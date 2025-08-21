-- Исправление схемы базы данных и настройка безопасности

-- 1. Добавляем отсутствующий столбец user_id в trading_sessions
ALTER TABLE trading_sessions ADD COLUMN user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;

-- 2. Обновляем существующие записи (устанавливаем временного пользователя)
-- В продакшене нужно будет назначить правильных пользователей
UPDATE trading_sessions SET user_id = (
  SELECT id FROM auth.users LIMIT 1
) WHERE user_id IS NULL;

-- 3. Делаем user_id обязательным после заполнения
ALTER TABLE trading_sessions ALTER COLUMN user_id SET NOT NULL;

-- 4. Удаляем старые небезопасные политики
DROP POLICY IF EXISTS "Allow all operations on trading_sessions" ON trading_sessions;
DROP POLICY IF EXISTS "Allow all operations on candle_data" ON candle_data;

-- 5. Включаем RLS (если еще не включен)
ALTER TABLE trading_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE candle_data ENABLE ROW LEVEL SECURITY;

-- 6. Создаем безопасные политики для торговых сессий
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

-- 7. Создаем безопасные политики для данных свечей
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

-- 8. Создаем индексы для производительности
CREATE INDEX IF NOT EXISTS idx_trading_sessions_user_id ON trading_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_trading_sessions_pair ON trading_sessions(pair);
CREATE INDEX IF NOT EXISTS idx_trading_sessions_created_at ON trading_sessions(created_at);

CREATE INDEX IF NOT EXISTS idx_candle_data_session_id ON candle_data(session_id);
CREATE INDEX IF NOT EXISTS idx_candle_data_datetime ON candle_data(candle_datetime);
CREATE INDEX IF NOT EXISTS idx_candle_data_session_datetime ON candle_data(session_id, candle_datetime);

-- 9. Добавляем ограничения безопасности
ALTER TABLE trading_sessions ADD CONSTRAINT IF NOT EXISTS session_name_length CHECK (char_length(session_name) <= 100);
ALTER TABLE trading_sessions ADD CONSTRAINT IF NOT EXISTS pair_length CHECK (char_length(pair) <= 20);
ALTER TABLE trading_sessions ADD CONSTRAINT IF NOT EXISTS timeframe_length CHECK (char_length(timeframe) <= 10);