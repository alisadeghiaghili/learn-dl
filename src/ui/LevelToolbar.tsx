import type { Level } from '../levels/types';
import type { I18nText } from '../levels/types';

export interface LevelToolbarProps {
  level: Level;
  locale: 'en_US' | 'fa';
  solved: boolean;
  onGoal: () => void;
  onHint: () => void;
  onSolution: () => void;
  onExit: () => void;
  onNext: () => void;
}

function pick(t: I18nText, locale: 'en_US' | 'fa'): string {
  return t[locale] ?? t.en_US;
}

export function LevelToolbar({
  level,
  locale,
  solved,
  onGoal,
  onHint,
  onSolution,
  onExit,
  onNext,
}: LevelToolbarProps) {
  return (
    <header className="level-toolbar">
      <div>
        <div className="level-title">{pick(level.name, locale)}</div>
        {level.objective && (
          <div className="level-objective">{pick(level.objective, locale)}</div>
        )}
      </div>
      <div className="level-actions">
        {solved && <span className="badge success">solved</span>}
        <button className="btn ghost" onClick={onHint}>
          hint
        </button>
        <button className="btn ghost" onClick={onGoal}>
          goal
        </button>
        <button className="btn ghost" onClick={onSolution}>
          solution
        </button>
        <button className="btn ghost" onClick={onExit}>
          exit
        </button>
        <button className="btn accent" onClick={onNext}>
          next
        </button>
      </div>
    </header>
  );
}
