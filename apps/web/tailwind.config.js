/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        ink: {
          50: '#f6f8fb',
          100: '#e6ecf2',
          200: '#c5d1e1',
          300: '#9aaac0',
          400: '#6c7e98',
          500: '#4b5d77',
          600: '#34465c',
          700: '#243246',
          800: '#172235',
          900: '#0c1424',
          950: '#060b18',
        },
        accent: {
          400: '#7ad9c4',
          500: '#3fc8a8',
          600: '#1faa8b',
        },
        danger: {
          400: '#f97366',
          500: '#ef5346',
        },
      },
      fontFamily: {
        mono: [
          'ui-monospace',
          'SFMono-Regular',
          'JetBrains Mono',
          'Menlo',
          'Consolas',
          'monospace',
        ],
      },
    },
  },
  plugins: [],
};
