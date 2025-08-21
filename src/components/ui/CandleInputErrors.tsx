
import React from 'react';

interface CandleInputErrorsProps {
  validationErrors: string[];
}

const CandleInputErrors = ({ validationErrors }: CandleInputErrorsProps) => {
  if (validationErrors.length === 0) {
    return null;
  }

  return (
    <div className="mb-4 p-3 bg-red-600/20 border border-red-600/50 rounded-lg">
      <ul className="text-red-200 text-sm space-y-1">
        {validationErrors.map((error, index) => (
          <li key={index}>• {error}</li>
        ))}
      </ul>
    </div>
  );
};

export default CandleInputErrors;
