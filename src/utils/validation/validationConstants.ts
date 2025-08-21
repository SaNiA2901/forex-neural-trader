
// Константы для валидации данных свечей
export const VALIDATION_CONSTANTS = {
  MIN_PRICE: 0.00001,
  MAX_PRICE: 999999,
  MIN_VOLUME: 0,
  MAX_VOLUME: 999999999,
  PRICE_TOLERANCE: 0.0001,
  MAX_DECIMAL_PLACES: 8,
  MIN_SESSION_NAME_LENGTH: 2,
  MAX_SESSION_NAME_LENGTH: 100
} as const;

export const VALID_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'] as const;

export const CURRENCY_PAIR_REGEX = /^[A-Z]{3}\/[A-Z]{3}$/;
export const DATE_REGEX = /^\d{4}-\d{2}-\d{2}$/;
export const TIME_REGEX = /^\d{2}:\d{2}(:\d{2})?$/;
