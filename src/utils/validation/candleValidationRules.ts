
interface CandleFormData {
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
}

interface CandleNumericData {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// Константы для валидации
const VALIDATION_CONSTANTS = {
  MIN_PRICE: 0.00001,
  MAX_PRICE: 999999,
  MIN_VOLUME: 0,
  MAX_VOLUME: 999999999,
  PRICE_TOLERANCE: 0.0001 // Допустимая погрешность для сравнения цен
} as const;

export const validateRequiredFields = (data: CandleFormData): string[] => {
  const errors: string[] = [];
  
  if (!data.open?.trim()) errors.push('Цена открытия обязательна');
  if (!data.high?.trim()) errors.push('Максимальная цена обязательна');
  if (!data.low?.trim()) errors.push('Минимальная цена обязательна');
  if (!data.close?.trim()) errors.push('Цена закрытия обязательна');
  if (!data.volume?.trim()) errors.push('Объем обязателен');
  
  return errors;
};

export const validateNumericValues = (data: CandleFormData): string[] => {
  const errors: string[] = [];
  
  const open = parseFloat(data.open);
  const high = parseFloat(data.high);
  const low = parseFloat(data.low);
  const close = parseFloat(data.close);
  const volume = parseFloat(data.volume);
  
  // Проверка на валидность чисел
  if (isNaN(open)) errors.push('Цена открытия должна быть числом');
  else if (open < VALIDATION_CONSTANTS.MIN_PRICE || open > VALIDATION_CONSTANTS.MAX_PRICE) {
    errors.push(`Цена открытия должна быть от ${VALIDATION_CONSTANTS.MIN_PRICE} до ${VALIDATION_CONSTANTS.MAX_PRICE}`);
  }
  
  if (isNaN(high)) errors.push('Максимальная цена должна быть числом');
  else if (high < VALIDATION_CONSTANTS.MIN_PRICE || high > VALIDATION_CONSTANTS.MAX_PRICE) {
    errors.push(`Максимальная цена должна быть от ${VALIDATION_CONSTANTS.MIN_PRICE} до ${VALIDATION_CONSTANTS.MAX_PRICE}`);
  }
  
  if (isNaN(low)) errors.push('Минимальная цена должна быть числом');
  else if (low < VALIDATION_CONSTANTS.MIN_PRICE || low > VALIDATION_CONSTANTS.MAX_PRICE) {
    errors.push(`Минимальная цена должна быть от ${VALIDATION_CONSTANTS.MIN_PRICE} до ${VALIDATION_CONSTANTS.MAX_PRICE}`);
  }
  
  if (isNaN(close)) errors.push('Цена закрытия должна быть числом');
  else if (close < VALIDATION_CONSTANTS.MIN_PRICE || close > VALIDATION_CONSTANTS.MAX_PRICE) {
    errors.push(`Цена закрытия должна быть от ${VALIDATION_CONSTANTS.MIN_PRICE} до ${VALIDATION_CONSTANTS.MAX_PRICE}`);
  }
  
  if (isNaN(volume)) errors.push('Объем должен быть числом');
  else if (volume < VALIDATION_CONSTANTS.MIN_VOLUME || volume > VALIDATION_CONSTANTS.MAX_VOLUME) {
    errors.push(`Объем должен быть от ${VALIDATION_CONSTANTS.MIN_VOLUME} до ${VALIDATION_CONSTANTS.MAX_VOLUME}`);
  }
  
  return errors;
};

export const validatePriceLogic = (data: CandleNumericData): string[] => {
  const errors: string[] = [];
  const { open, high, low, close } = data;
  
  // Проверяем, что high >= max(open, close) с учетом погрешности
  const maxOpenClose = Math.max(open, close);
  if (high < maxOpenClose - VALIDATION_CONSTANTS.PRICE_TOLERANCE) {
    errors.push('Максимальная цена не может быть меньше цен открытия или закрытия');
  }
  
  // Проверяем, что low <= min(open, close) с учетом погрешности
  const minOpenClose = Math.min(open, close);
  if (low > minOpenClose + VALIDATION_CONSTANTS.PRICE_TOLERANCE) {
    errors.push('Минимальная цена не может быть больше цен открытия или закрытия');
  }
  
  // Проверяем, что high >= low с учетом погрешности
  if (high < low - VALIDATION_CONSTANTS.PRICE_TOLERANCE) {
    errors.push('Максимальная цена не может быть меньше минимальной');
  }
  
  // Дополнительные проверки логики
  if (high < open - VALIDATION_CONSTANTS.PRICE_TOLERANCE) {
    errors.push('Максимальная цена не может быть меньше цены открытия');
  }
  
  if (high < close - VALIDATION_CONSTANTS.PRICE_TOLERANCE) {
    errors.push('Максимальная цена не может быть меньше цены закрытия');
  }
  
  if (low > open + VALIDATION_CONSTANTS.PRICE_TOLERANCE) {
    errors.push('Минимальная цена не может быть больше цены открытия');
  }
  
  if (low > close + VALIDATION_CONSTANTS.PRICE_TOLERANCE) {
    errors.push('Минимальная цена не может быть больше цены закрытия');
  }
  
  return errors;
};

export const validateVolumeLogic = (volume: number): string[] => {
  const warnings: string[] = [];
  
  if (volume === 0) {
    warnings.push('Объем равен нулю - возможно отсутствие торговой активности');
  }
  
  if (volume > 0 && volume < 1000) {
    warnings.push('Очень низкий объем торгов - проверьте корректность данных');
  }
  
  return warnings;
};

// Функция для проверки корректности спреда
export const validateSpread = (data: CandleNumericData): string[] => {
  const warnings: string[] = [];
  const spread = data.high - data.low;
  const price = (data.open + data.close) / 2;
  const spreadPercent = (spread / price) * 100;
  
  if (spreadPercent > 5) {
    warnings.push(`Большой спред (${spreadPercent.toFixed(2)}%) - проверьте корректность данных`);
  }
  
  if (spread < VALIDATION_CONSTANTS.PRICE_TOLERANCE) {
    warnings.push('Очень маленький спред - возможно некорректные данные');
  }
  
  return warnings;
};
