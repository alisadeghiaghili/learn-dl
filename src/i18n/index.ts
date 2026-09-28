export type UiLocale = 'en_US' | 'fa';

const strings = {
  en_US: {
    appTitle: 'learn-dl',
    sandbox: 'Sandbox',
    levels: 'Levels',
    library: 'Library',
    goal: 'Goal',
    code: 'Code',
    builder: 'Builder',
    solved: 'solved',
  },
  fa: {
    appTitle: 'learn-dl',
    sandbox: 'سندباکس',
    levels: 'مرحله‌ها',
    library: 'کتابخانه',
    goal: 'هدف',
    code: 'کد',
    builder: 'سازنده',
    solved: 'حل‌شده',
  },
} as const;

export function t(key: keyof typeof strings.en_US, locale: UiLocale): string {
  return strings[locale][key] ?? strings.en_US[key];
}

export { strings };
