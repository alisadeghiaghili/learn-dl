import { useMemo, useState } from 'react';
import { generateAllCode } from '../codegen';
import type { ExperimentState, LibraryId } from '../state/types';

export interface CodePanelsProps {
  state: ExperimentState;
  onParse: (source: string) => void;
}

const LIB_META: Record<LibraryId, { label: string; cls: string }> = {
  pytorch: { label: 'PyTorch', cls: 'lib-pt' },
  tensorflow: { label: 'TensorFlow', cls: 'lib-tf' },
  keras: { label: 'Keras', cls: 'lib-keras' },
};

/**
 * Side-by-side generated code for the three libraries, with an editable
 * buffer that feeds the Python subset parser.
 */
export function CodePanels({ state, onParse }: CodePanelsProps) {
  const [tab, setTab] = useState<LibraryId>(state.library);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');

  const codes = useMemo(() => generateAllCode(state), [state]);
  const active = codes[tab];

  return (
    <div className="code-panels">
      <div className="code-tabs" role="tablist">
        {(Object.keys(LIB_META) as LibraryId[]).map((lib) => (
          <button
            key={lib}
            role="tab"
            aria-selected={tab === lib}
            className={`code-tab ${LIB_META[lib].cls} ${tab === lib ? 'active' : ''}`}
            onClick={() => {
              setTab(lib);
              setEditing(false);
            }}
          >
            {LIB_META[lib].label}
          </button>
        ))}
        <div className="code-actions">
          {!editing ? (
            <button className="btn ghost" onClick={() => { setDraft(active?.source ?? ''); setEditing(true); }}>
              Edit
            </button>
          ) : (
            <>
              <button
                className="btn accent"
                onClick={() => {
                  onParse(draft);
                  setEditing(false);
                }}
              >
                Apply
              </button>
              <button className="btn ghost" onClick={() => setEditing(false)}>
                Cancel
              </button>
            </>
          )}
        </div>
      </div>
      {editing ? (
        <textarea
          className="code-editor"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          spellCheck={false}
        />
      ) : (
        <pre className="code-view">
          <code>{active?.source ?? ''}</code>
        </pre>
      )}
    </div>
  );
}
