
// Утилиты для санитизации пользовательского ввода
export const sanitizeNumericInput = (value: string): string => {
  if (!value) return '';
  
  // Убираем все символы кроме цифр, точки и минуса
  let sanitized = value.replace(/[^0-9.-]/g, '');
  
  // Обрабатываем множественные точки
  const parts = sanitized.split('.');
  if (parts.length > 2) {
    sanitized = parts[0] + '.' + parts.slice(1).join('');
  }
  
  // Обрабатываем множественные минусы (только в начале)
  const minusCount = (sanitized.match(/-/g) || []).length;
  if (minusCount > 1) {
    const hasLeadingMinus = sanitized.startsWith('-');
    sanitized = sanitized.replace(/-/g, '');
    if (hasLeadingMinus) {
      sanitized = '-' + sanitized;
    }
  }
  
  // Ограничиваем количество знаков после запятой
  if (sanitized.includes('.')) {
    const [integer, decimal] = sanitized.split('.');
    sanitized = integer + '.' + decimal.slice(0, 8);
  }
  
  return sanitized;
};

export const sanitizeSessionName = (name: string): string => {
  return name.trim().slice(0, 100);
};

export const sanitizeCurrencyPair = (pair: string): string => {
  return pair.trim().toUpperCase();
};
