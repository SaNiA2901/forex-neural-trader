export const parseTimeframe = (timeframe: string): number => {
  const timeframeMap: Record<string, number> = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440
  };
  
  const minutes = timeframeMap[timeframe];
  if (minutes === undefined) {
    throw new Error(`Неподдерживаемый таймфрейм: ${timeframe}`);
  }
  
  return minutes;
};

export const formatCandleDateTime = (dateTime: string): string => {
  if (!dateTime) {
    return 'Неизвестно';
  }
  
  try {
    const date = new Date(dateTime);
    if (isNaN(date.getTime())) {
      throw new Error('Невалидная дата');
    }
    
    return new Intl.DateTimeFormat('ru-RU', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      timeZone: 'UTC'
    }).format(date);
  } catch (error) {
    console.error('Error formatting date:', error);
    return dateTime;
  }
};

export const calculateCandleDateTime = (
  startDate: string,
  startTime: string,
  timeframe: string,
  candleIndex: number
): string => {
  if (!startDate?.trim()) {
    throw new Error('Дата начала не может быть пустой');
  }
  
  if (!startTime?.trim()) {
    throw new Error('Время начала не может быть пустым');
  }
  
  if (!timeframe?.trim()) {
    throw new Error('Таймфрейм не может быть пустым');
  }
  
  if (typeof candleIndex !== 'number' || candleIndex < 0 || !Number.isInteger(candleIndex)) {
    throw new Error('Индекс свечи должен быть неотрицательным целым числом');
  }
  
  try {
    // Проверяем формат даты
    if (!/^\d{4}-\d{2}-\d{2}$/.test(startDate)) {
      throw new Error('Дата должна быть в формате YYYY-MM-DD');
    }
    
    // Проверяем формат времени
    if (!/^\d{2}:\d{2}(:\d{2})?$/.test(startTime)) {
      throw new Error('Время должно быть в формате HH:MM или HH:MM:SS');
    }
    
    // Нормализуем время (добавляем секунды если их нет)
    const normalizedTime = startTime.length === 5 ? `${startTime}:00` : startTime;
    
    const startDateTime = new Date(`${startDate}T${normalizedTime}`);
    
    if (isNaN(startDateTime.getTime())) {
      throw new Error('Невалидная дата или время начала');
    }
    
    const minutes = parseTimeframe(timeframe) * candleIndex;
    const candleDateTime = new Date(startDateTime.getTime() + minutes * 60000);
    
    if (isNaN(candleDateTime.getTime())) {
      throw new Error('Ошибка при расчете времени свечи');
    }
    
    return candleDateTime.toISOString();
  } catch (error) {
    console.error('Error calculating candle time:', error);
    if (error instanceof Error) {
      throw error;
    }
    throw new Error(`Ошибка расчета времени свечи: Неизвестная ошибка`);
  }
};

export const validateTimeParameters = (startDate: string, startTime: string, timeframe: string): {
  isValid: boolean;
  errors: string[];
} => {
  const errors: string[] = [];
  
  if (!startDate?.trim()) {
    errors.push('Дата начала обязательна');
  } else if (!/^\d{4}-\d{2}-\d{2}$/.test(startDate)) {
    errors.push('Дата должна быть в формате YYYY-MM-DD');
  }
  
  if (!startTime?.trim()) {
    errors.push('Время начала обязательно');
  } else if (!/^\d{2}:\d{2}(:\d{2})?$/.test(startTime)) {
    errors.push('Время должно быть в формате HH:MM или HH:MM:SS');
  }
  
  if (!timeframe?.trim()) {
    errors.push('Таймфрейм обязателен');
  } else {
    try {
      parseTimeframe(timeframe);
    } catch (error) {
      errors.push('Неподдерживаемый таймфрейм');
    }
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};
