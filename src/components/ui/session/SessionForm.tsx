import React from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Calendar, Clock, TrendingUp } from 'lucide-react';
import { format } from 'date-fns';

const sessionSchema = z.object({
  session_name: z.string()
    .min(1, 'Название сессии обязательно')
    .max(100, 'Название не может быть длиннее 100 символов')
    .regex(/^[a-zA-Zа-яА-Я0-9\s\-_]+$/, 'Недопустимые символы в названии'),
  
  pair: z.string()
    .min(1, 'Выберите валютную пару')
    .regex(/^[A-Z]{3}\/[A-Z]{3}$/, 'Неверный формат пары (например: EUR/USD)'),
  
  timeframe: z.string()
    .min(1, 'Выберите таймфрейм'),
  
  start_date: z.string()
    .min(1, 'Выберите дату начала')
    .refine((date) => {
      const selectedDate = new Date(date);
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      return selectedDate >= today;
    }, 'Дата не может быть в прошлом'),
  
  start_time: z.string()
    .min(1, 'Выберите время начала')
    .regex(/^([0-1][0-9]|2[0-3]):[0-5][0-9]$/, 'Неверный формат времени')
});

type SessionFormData = z.infer<typeof sessionSchema>;

interface SessionFormProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: SessionFormData) => Promise<void>;
  isSubmitting?: boolean;
  initialData?: Partial<SessionFormData>;
  title?: string;
}

const CURRENCY_PAIRS = [
  'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD',
  'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY', 'EUR/CHF', 'GBP/CHF',
  'AUD/JPY', 'CAD/JPY', 'CHF/JPY', 'EUR/AUD', 'EUR/CAD', 'EUR/NZD',
  'GBP/AUD', 'GBP/CAD', 'GBP/NZD', 'AUD/CAD', 'AUD/CHF', 'AUD/NZD',
  'CAD/CHF', 'NZD/CAD', 'NZD/CHF', 'NZD/JPY'
];

const TIMEFRAMES = [
  { value: '1m', label: '1 минута' },
  { value: '5m', label: '5 минут' },
  { value: '15m', label: '15 минут' },
  { value: '30m', label: '30 минут' },
  { value: '1h', label: '1 час' },
  { value: '4h', label: '4 часа' },
  { value: '1d', label: '1 день' },
  { value: '1w', label: '1 неделя' }
];

export const SessionForm: React.FC<SessionFormProps> = ({
  isOpen,
  onClose,
  onSubmit,
  isSubmitting = false,
  initialData,
  title = 'Создать торговую сессию'
}) => {
  const {
    register,
    handleSubmit,
    setValue,
    watch,
    formState: { errors, isValid },
    reset
  } = useForm<SessionFormData>({
    resolver: zodResolver(sessionSchema),
    defaultValues: {
      session_name: initialData?.session_name || '',
      pair: initialData?.pair || '',
      timeframe: initialData?.timeframe || '',
      start_date: initialData?.start_date || format(new Date(), 'yyyy-MM-dd'),
      start_time: initialData?.start_time || format(new Date(), 'HH:mm')
    }
  });

  const selectedPair = watch('pair');
  const selectedTimeframe = watch('timeframe');

  const handleClose = () => {
    reset();
    onClose();
  };

  const onFormSubmit = async (data: SessionFormData) => {
    try {
      await onSubmit(data);
      handleClose();
    } catch (error) {
      console.error('Error submitting session form:', error);
    }
  };

  const generateSessionName = () => {
    if (selectedPair && selectedTimeframe) {
      const timestamp = format(new Date(), 'dd.MM.yyyy HH:mm');
      const timeframeLabel = TIMEFRAMES.find(tf => tf.value === selectedTimeframe)?.label || selectedTimeframe;
      setValue('session_name', `${selectedPair} ${timeframeLabel} ${timestamp}`);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            {title}
          </DialogTitle>
        </DialogHeader>

        <form onSubmit={handleSubmit(onFormSubmit)} className="space-y-6">
          {/* Название сессии */}
          <div className="space-y-2">
            <Label htmlFor="session_name">Название сессии</Label>
            <div className="flex gap-2">
              <Input
                id="session_name"
                {...register('session_name')}
                placeholder="Введите название сессии"
                className={errors.session_name ? 'border-destructive' : ''}
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={generateSessionName}
                disabled={!selectedPair || !selectedTimeframe}
              >
                Авто
              </Button>
            </div>
            {errors.session_name && (
              <p className="text-sm text-destructive">{errors.session_name.message}</p>
            )}
          </div>

          {/* Валютная пара */}
          <div className="space-y-2">
            <Label htmlFor="pair">Валютная пара</Label>
            <Select
              value={watch('pair')}
              onValueChange={(value) => setValue('pair', value, { shouldValidate: true })}
            >
              <SelectTrigger className={errors.pair ? 'border-destructive' : ''}>
                <SelectValue placeholder="Выберите валютную пару" />
              </SelectTrigger>
              <SelectContent>
                {CURRENCY_PAIRS.map((pair) => (
                  <SelectItem key={pair} value={pair}>
                    {pair}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {errors.pair && (
              <p className="text-sm text-destructive">{errors.pair.message}</p>
            )}
          </div>

          {/* Таймфрейм */}
          <div className="space-y-2">
            <Label htmlFor="timeframe">Таймфрейм</Label>
            <Select
              value={watch('timeframe')}
              onValueChange={(value) => setValue('timeframe', value, { shouldValidate: true })}
            >
              <SelectTrigger className={errors.timeframe ? 'border-destructive' : ''}>
                <SelectValue placeholder="Выберите таймфрейм" />
              </SelectTrigger>
              <SelectContent>
                {TIMEFRAMES.map((tf) => (
                  <SelectItem key={tf.value} value={tf.value}>
                    {tf.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {errors.timeframe && (
              <p className="text-sm text-destructive">{errors.timeframe.message}</p>
            )}
          </div>

          {/* Дата и время */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="start_date" className="flex items-center gap-1">
                <Calendar className="h-4 w-4" />
                Дата начала
              </Label>
              <Input
                id="start_date"
                type="date"
                {...register('start_date')}
                className={errors.start_date ? 'border-destructive' : ''}
              />
              {errors.start_date && (
                <p className="text-sm text-destructive">{errors.start_date.message}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="start_time" className="flex items-center gap-1">
                <Clock className="h-4 w-4" />
                Время начала
              </Label>
              <Input
                id="start_time"
                type="time"
                {...register('start_time')}
                className={errors.start_time ? 'border-destructive' : ''}
              />
              {errors.start_time && (
                <p className="text-sm text-destructive">{errors.start_time.message}</p>
              )}
            </div>
          </div>

          {/* Кнопки */}
          <div className="flex justify-end gap-3 pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={handleClose}
              disabled={isSubmitting}
            >
              Отмена
            </Button>
            <Button
              type="submit"
              disabled={!isValid || isSubmitting}
            >
              {isSubmitting ? 'Создание...' : 'Создать сессию'}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};