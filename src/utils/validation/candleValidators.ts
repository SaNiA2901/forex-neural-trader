
// Валидаторы для данных свечей
interface CandleFormData {
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
}

interface NumericCandleData {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export const validateRequiredFields = (data: CandleFormData): string[] => {
  const errors: string[] = [];
  
  if (!data.open?.trim()) {
    errors.push('Цена открытия обязательна');
  }
  
  if (!data.high?.trim()) {
    errors.push('Максимальная цена обязательна');
  }
  
  if (!data.low?.trim()) {
    errors.push('Минимальная цена обязательна');
  }
  
  if (!data.close?.trim()) {
    errors.push('Цена закрытия обязательна');
  }
  
  if (!data.volume?.trim()) {
    errors.push('Объем обязателен');
  }
  
  return errors;
};

export const validateNumericValues = (data: CandleFormData): string[] => {
  const errors: string[] = [];
  
  const fields = [
    { key: 'open', name: 'Цена открытия' },
    { key: 'high', name: 'Максимальная цена' },
    { key: 'low', name: 'Минимальная цена' },
    { key: 'close', name: 'Цена закрытия' },
    { key: 'volume', name: 'Объем' }
  ] as const;
  
  fields.forEach(({ key, name }) => {
    const value = data[key];
    if (value?.trim()) {
      const numValue = parseFloat(value);
      if (isNaN(numValue)) {
        errors.push(`${name} должна быть числом`);
      } else if (numValue < 0) {
        errors.push(`${name} должна быть положительным числом`);
      } else if (key === 'volume' && numValue === 0) {
        errors.push('Объем должен быть больше нуля');
      }
    }
  });
  
  return errors;
};

export const validatePriceLogic = (data: NumericCandleData): string[] => {
  const errors: string[] = [];
  
  // Проверяем, что high >= max(open, close) и high >= low
  if (data.high < Math.max(data.open, data.close)) {
    errors.push('Максимальная цена не может быть меньше цены открытия или закрытия');
  }
  
  if (data.high < data.low) {
    errors.push('Максимальная цена не может быть меньше минимальной');
  }
  
  // Проверяем, что low <= min(open, close) и low <= high
  if (data.low > Math.min(data.open, data.close)) {
    errors.push('Минимальная цена не может быть больше цены открытия или закрытия');
  }
  
  // Проверяем, что open и close находятся в диапазоне [low, high]
  if (data.open < data.low || data.open > data.high) {
    errors.push('Цена открытия должна быть в диапазоне между минимальной и максимальной ценой');
  }
  
  if (data.close < data.low || data.close > data.high) {
    errors.push('Цена закрытия должна быть в диапазоне между минимальной и максимальной ценой');
  }
  
  return errors;
};
