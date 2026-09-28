import { useEffect, useRef, useState } from 'react';

export interface TerminalLine {
  kind: 'in' | 'out' | 'err' | 'sys';
  text: string;
}

export interface TerminalProps {
  lines: TerminalLine[];
  onSubmit: (raw: string) => void;
  placeholder?: string;
  disabled?: boolean;
}

/**
 * LGB-style command terminal.
 */
export function Terminal({
  lines,
  onSubmit,
  placeholder = 'type help · net dense 8 relu · train 10',
  disabled = false,
}: TerminalProps) {
  const [value, setValue] = useState('');
  const [history, setHistory] = useState<string[]>([]);
  const [histIdx, setHistIdx] = useState(-1);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [lines.length]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === '`' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const handleKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      const raw = value.trim();
      if (!raw) return;
      setHistory((h) => [...h, raw]);
      setHistIdx(-1);
      onSubmit(raw);
      setValue('');
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (history.length === 0) return;
      const next = histIdx < 0 ? history.length - 1 : Math.max(0, histIdx - 1);
      setHistIdx(next);
      setValue(history[next] ?? '');
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (histIdx < 0) return;
      const next = histIdx + 1;
      if (next >= history.length) {
        setHistIdx(-1);
        setValue('');
      } else {
        setHistIdx(next);
        setValue(history[next] ?? '');
      }
    }
  };

  return (
    <div className="terminal" aria-label="Command terminal">
      <div className="terminal-log">
        {lines.map((line, i) => (
          <div key={i} className={`terminal-line terminal-${line.kind}`}>
            {line.kind === 'in' ? '› ' : ''}
            {line.text}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <div className="terminal-input-row">
        <span className="terminal-prompt">›</span>
        <input
          ref={inputRef}
          className="terminal-input"
          value={value}
          disabled={disabled}
          placeholder={placeholder}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKey}
          spellCheck={false}
          autoComplete="off"
        />
      </div>
    </div>
  );
}
