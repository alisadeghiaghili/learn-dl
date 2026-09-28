import { levelsBySequence, sequenceInfo } from '../levels';
import type { Level } from '../levels/types';

export interface LevelsBrowserProps {
  solved: Record<string, true>;
  onOpen: (level: Level) => void;
  onClose: () => void;
}

export function LevelsBrowser({ solved, onOpen, onClose }: LevelsBrowserProps) {
  const tabs = [
    { id: 'main', label: 'Main' },
    { id: 'libraries', label: 'Libraries' },
  ];
  return (
    <div className="dialog-backdrop" role="dialog" aria-modal="true">
      <div className="dialog-paper levels-browser">
        <div className="dialog-actions">
          <h2>Levels</h2>
          <button className="btn ghost" onClick={onClose}>
            close
          </button>
        </div>
        {tabs.map((tab) => (
          <section key={tab.id}>
            <h3 className="levels-tab-title">{tab.label}</h3>
            {sequenceInfo
              .filter((s) => s.tab === tab.id)
              .map((seq) => {
                const list = levelsBySequence(seq.id);
                const done = list.filter((l) => solved[l.id]).length;
                return (
                  <div key={seq.id} className="sequence-block">
                    <div className="sequence-head">
                      <strong>{seq.displayName.en_US}</strong>
                      <span className="muted">
                        {' '}
                        {seq.about.en_US} · {done}/{list.length}
                      </span>
                    </div>
                    <ul className="level-list">
                      {list.map((level) => (
                        <li key={level.id}>
                          <button
                            className={`level-item ${solved[level.id] ? 'done' : ''}`}
                            onClick={() => onOpen(level)}
                          >
                            {solved[level.id] ? '✓ ' : '○ '}
                            {level.name.en_US}
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                );
              })}
          </section>
        ))}
      </div>
    </div>
  );
}
