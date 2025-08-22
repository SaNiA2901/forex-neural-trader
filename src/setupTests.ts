// Jest setup file
import '@testing-library/jest-dom';

// Mock performance.now for consistent timing in tests
Object.defineProperty(global, 'performance', {
  writable: true,
  value: {
    now: jest.fn(() => Date.now())
  }
});

// Mock console methods to reduce noise in tests
const originalConsole = { ...console };

beforeEach(() => {
  // Reset console methods before each test
  global.console = {
    ...originalConsole,
    warn: jest.fn(),
    log: jest.fn(),
    error: jest.fn()
  };
});

afterEach(() => {
  // Restore console after each test
  global.console = originalConsole;
});