
import { validateRequiredFields, validateNumericValues, validatePriceLogic } from './validation/candleValidators';
import { sanitizeNumericInput } from './validation/inputSanitization';
import { VALIDATION_CONSTANTS, VALID_TIMEFRAMES, CURRENCY_PAIR_REGEX, DATE_REGEX, TIME_REGEX } from './validation/validationConstants';

interface CandleFormData {
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
}

interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings?: string[];
}

export { sanitizeNumericInput };

export const validateFormData = (data: CandleFormData): ValidationResult => {
  let errors: string[] = [];

  // Проверка обязательных полей
  errors = errors.concat(validateRequiredFields(data));
  if (errors.length > 0) {
    return { isValid: false, errors };
  }

  // Проверка числовых значений
  errors = errors.concat(validateNumericValues(data));
  if (errors.length > 0) {
    return { isValid: false, errors };
  }

  // Проверка логики цен
  const numericData = {
    open: parseFloat(data.open),
    high: parseFloat(data.high),
    low: parseFloat(data.low),
    close: parseFloat(data.close),
    volume: parseFloat(data.volume)
  };

  errors = errors.concat(validatePriceLogic(numericData));

  return {
    isValid: errors.length === 0,
    errors
  };
};

export const validateCandleData = (data: any): ValidationResult => {
  const errors: string[] = [];

  if (!data.session_id?.trim()) {
    errors.push('ID сессии обязателен');
  }

  if (typeof data.candle_index !== 'number' || data.candle_index < 0) {
    errors.push('Индекс свечи должен быть неотрицательным числом');
  }

  const numericFields = ['open', 'high', 'low', 'close', 'volume'];
  numericFields.forEach(field => {
    if (typeof data[field] !== 'number' || isNaN(data[field]) || data[field] < 0) {
      errors.push(`${field} должно быть положительным числом`);
    }
  });

  return {
    isValid: errors.length === 0,
    errors
  };
};

export const validateSessionData = (sessionData: any): ValidationResult => {
  const errors: string[] = [];

  if (!sessionData.session_name?.trim()) {
    errors.push('Название сессии обязательно');
  } else if (sessionData.session_name.trim().length < VALIDATION_CONSTANTS.MIN_SESSION_NAME_LENGTH) {
    errors.push(`Название сессии должно содержать минимум ${VALIDATION_CONSTANTS.MIN_SESSION_NAME_LENGTH} символа`);
  }

  if (!sessionData.pair?.trim()) {
    errors.push('Валютная пара обязательна');
  } else if (!CURRENCY_PAIR_REGEX.test(sessionData.pair.trim())) {
    errors.push('Валютная пара должна быть в формате XXX/YYY');
  }

  if (!sessionData.timeframe?.trim()) {
    errors.push('Таймфрейм обязателен');
  } else if (!VALID_TIMEFRAMES.includes(sessionData.timeframe.trim() as any)) {
    errors.push(`Таймфрейм должен быть одним из: ${VALID_TIMEFRAMES.join(', ')}`);
  }

  if (!sessionData.start_date?.trim()) {
    errors.push('Дата начала обязательна');
  } else if (!DATE_REGEX.test(sessionData.start_date)) {
    errors.push('Дата должна быть в формате YYYY-MM-DD');
  }

  if (!sessionData.start_time?.trim()) {
    errors.push('Время начала обязательно');
  } else if (!TIME_REGEX.test(sessionData.start_time)) {
    errors.push('Время должно быть в формате HH:MM или HH:MM:SS');
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};
