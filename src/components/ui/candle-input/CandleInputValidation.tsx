
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertTriangle, CheckCircle, AlertCircle } from 'lucide-react';

interface CandleInputValidationProps {
  errors: string[];
  isFormValid: boolean;
}

const CandleInputValidation = ({ errors, isFormValid }: CandleInputValidationProps) => {
  return (
    <>
      {errors.length > 0 && (
        <Alert className="mb-4 bg-red-900/50 border-red-700">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            <ul className="list-disc list-inside">
              {errors.map((error, index) => (
                <li key={index} className="text-red-200">{error}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2 text-sm">
          {isFormValid ? (
            <>
              <CheckCircle className="h-4 w-4 text-green-400" />
              <span className="text-green-400">Данные корректны</span>
            </>
          ) : (
            <>
              <AlertCircle className="h-4 w-4 text-orange-400" />
              <span className="text-orange-400">Заполните все поля корректно</span>
            </>
          )}
        </div>
      </div>
    </>
  );
};

export default CandleInputValidation;
