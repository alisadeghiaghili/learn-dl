import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { executeCommand } from './commands/dsl';
import { makeDataset, type FullData } from './engine/datasets';
import type { CompiledModel } from './engine/model';
import { train, evaluate } from './engine/trainer';
import {
  describeGoal,
  goalSatisfied,
  nextLevel,
} from './levels';
import type { Level } from './levels/types';
import { parsePythonSubset } from './codeparse/subset';
import {
  cloneExperiment,
  emptyExperiment,
  type ExperimentState,
  type LibraryId,
} from './state/types';
import { CodePanels } from './ui/CodePanels';
import { Dialog } from './ui/Dialog';
import { GoalPanel } from './ui/GoalPanel';
import { LevelsBrowser } from './ui/LevelsBrowser';
import { LevelToolbar } from './ui/LevelToolbar';
import { Terminal, type TerminalLine } from './ui/Terminal';
import { VisualBuilder } from './ui/VisualBuilder';
import { DecisionBoundary } from './viz/DecisionBoundary';
import { NetworkGraph } from './viz/NetworkGraph';
import { TrainCurves } from './viz/TrainCurves';
import { t, type UiLocale } from './i18n';

type Mode = 'sandbox' | 'level';
type RailTab = 'goal' | 'code' | 'builder';

const SOLVED_KEY = 'learn-dl.solved';

function loadSolved(): Record<string, true> {
  try {
    const raw = localStorage.getItem(SOLVED_KEY);
    return raw ? (JSON.parse(raw) as Record<string, true>) : {};
  } catch {
    return {};
  }
}

export function App() {
  const [state, setState] = useState<ExperimentState>(() => emptyExperiment());
  const [mode, setMode] = useState<Mode>('sandbox');
  const [level, setLevel] = useState<Level | null>(null);
  const [railTab, setRailTab] = useState<RailTab>('code');
  const [locale] = useState<UiLocale>('en_US');
  const [lines, setLines] = useState<TerminalLine[]>([
    { kind: 'sys', text: 'learn-dl · type help · levels to start learning' },
  ]);
  const [solvedMap, setSolvedMap] = useState<Record<string, true>>(() => loadSolved());
  const [showBrowser, setShowBrowser] = useState(false);
  const [showDialog, setShowDialog] = useState(false);
  const [showSolution, setShowSolution] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [trainedModel, setTrainedModel] = useState<CompiledModel | null>(null);
  const [data, setData] = useState<FullData | null>(null);
  const undoStack = useRef<ExperimentState[]>([]);
  const solvedRef = useRef(false);

  const pushLine = useCallback((line: TerminalLine) => {
    setLines((prev) => [...prev, line].slice(-200));
  }, []);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    window.setTimeout(() => setToast(null), 3200);
  }, []);

  // refresh dataset preview when dataset spec changes
  useEffect(() => {
    setData(makeDataset(state.dataset));
    setTrainedModel(null);
  }, [state.dataset.id, state.dataset.seed, state.dataset.nSamples, state.dataset.noise]);

  const maybeSolve = useCallback(
    (nextState: ExperimentState) => {
      if (mode !== 'level' || !level || solvedRef.current) return;
      if (goalSatisfied(level.goal, nextState)) {
        solvedRef.current = true;
        const updated = { ...solvedMap, [level.id]: true as const };
        setSolvedMap(updated);
        try {
          localStorage.setItem(SOLVED_KEY, JSON.stringify(updated));
        } catch {
          /* ignore */
        }
        pushLine({ kind: 'sys', text: `✔ level solved: ${level.name.en_US}` });
        showToast('Level solved. `next` continues.');
      }
    },
    [mode, level, solvedMap, pushLine, showToast],
  );

  const runTrain = useCallback(
    (epochs: number, overrideState?: ExperimentState) => {
      const st = overrideState ?? state;
      const ds = makeDataset(st.dataset);
      setData(ds);
      if (st.layers.length === 0) {
        pushLine({ kind: 'err', text: 'No layers — add one with `net dense 8 relu`' });
        return;
      }
      const inputDim = st.dataset.inputDim ?? ds.inputDim;
      const outputDim = st.dataset.nClasses ?? ds.nClasses;
      try {
        const { model, result } = train(
          ds,
          {
            layers: st.layers,
            compile: st.compile,
            batch: st.training.batch,
            epochs,
            inputDim,
            outputDim,
          },
          st.training.epochsTrained,
        );
        const history = [...st.training.history, ...result.history];
        const next: ExperimentState = {
          ...st,
          training: {
            epochsTrained: st.training.epochsTrained + epochs,
            batch: st.training.batch,
            history,
            weightNorm: result.weightNorm,
          },
        };
        setState(next);
        setTrainedModel(model);
        maybeSolve(next);
        const last = result.history[result.history.length - 1];
        pushLine({
          kind: 'out',
          text: `trained ${epochs} epoch(s) · loss=${last?.loss.toFixed(4)} · acc=${((last?.accuracy ?? 0) * 100).toFixed(1)}%`,
        });
      } catch (err) {
        pushLine({ kind: 'err', text: `train failed: ${(err as Error).message}` });
      }
    },
    [state, pushLine, maybeSolve],
  );

  const applyState = useCallback(
    (next: ExperimentState, recordUndo = true) => {
      if (recordUndo) undoStack.current.push(cloneExperiment(state));
      setState(next);
      maybeSolve(next);
    },
    [state, maybeSolve],
  );

  const openLevel = useCallback(
    (lv: Level) => {
      setMode('level');
      setLevel(lv);
      setState(cloneExperiment(lv.startState));
      undoStack.current = [];
      solvedRef.current = !!solvedMap[lv.id];
      setShowSolution(false);
      setShowBrowser(false);
      setTrainedModel(null);
      setLines([
        { kind: 'sys', text: `level · ${lv.name.en_US}` },
        { kind: 'sys', text: describeGoal(lv.goal) },
      ]);
      if (lv.startDialog?.length) setShowDialog(true);
    },
    [solvedMap],
  );

  const exitLevel = useCallback(() => {
    setMode('sandbox');
    setLevel(null);
    setLines([{ kind: 'sys', text: 'sandbox mode' }]);
  }, []);

  const handleCommand = useCallback(
    (raw: string) => {
      pushLine({ kind: 'in', text: raw });
      // side-channel: train is special-cased to run the engine
      const trimmed = raw.trim();
      const trainMatch = trimmed.match(/^train\s+(\d+)/i);
      if (trainMatch) {
        undoStack.current.push(cloneExperiment(state));
        runTrain(Number(trainMatch[1]));
        return;
      }

      const result = executeCommand(raw, {
        state,
        undoStack: undoStack.current,
        levelId: level?.id,
        disabled: level?.disabledCommands,
      });

      if (result.action === 'undo') {
        undoStack.current.pop();
      }

      if (result.message) {
        pushLine({ kind: result.ok ? 'out' : 'err', text: result.message });
      }

      switch (result.action) {
        case 'levels':
          setShowBrowser(true);
          break;
        case 'hint':
          if (level) {
            pushLine({ kind: 'sys', text: level.hint.en_US });
          } else {
            pushLine({ kind: 'sys', text: 'No active level. Type `levels`.' });
          }
          break;
        case 'goal':
          if (level) {
            setRailTab('goal');
            showToast(describeGoal(level.goal));
          }
          break;
        case 'solution':
          if (level) {
            setShowSolution(true);
            setRailTab('goal');
            setShowDialog(false);
            pushLine({ kind: 'sys', text: level.solutionCommand });
            undoStack.current.push(cloneExperiment(state));
            let acc = cloneExperiment(level.startState);
            let trainEpochs = 0;
            for (const part of level.solutionCommand.split(';')) {
              const r = executeCommand(part, { state: acc, undoStack: [] });
              acc = r.state;
              const payload = r.payload as { trainEpochs?: number } | undefined;
              if (payload?.trainEpochs) trainEpochs += payload.trainEpochs;
            }
            setState(acc);
            if (trainEpochs > 0) {
              runTrain(trainEpochs, acc);
            } else {
              maybeSolve(acc);
            }
          }
          break;
        case 'reset':
          setTrainedModel(null);
          undoStack.current = [];
          break;
        case 'undo':
          // state already swapped by command
          setTrainedModel(null);
          break;
        case 'build':
          setRailTab('builder');
          break;
        case 'demo': {
          const payload = result.payload as { trainEpochs?: number; demo?: string } | undefined;
          if (payload?.trainEpochs) {
            applyState(result.state);
            runTrain(payload.trainEpochs);
            return;
          }
          break;
        }
        case 'clear':
          setLines([{ kind: 'sys', text: 'cleared' }]);
          return;
        default:
          break;
      }

      // library switch also updates preferred tab
      if (result.state.library !== state.library) {
        setRailTab('code');
      }
      applyState(result.state);
    },
    [state, level, pushLine, runTrain, applyState, showToast, maybeSolve],
  );

  // train via `train N` already handled; also handle demo payload path from executeCommand
  // (executeCommand('train N') returns action demo — we intercept train before that)

  const onParse = useCallback(
    (source: string) => {
      const { state: next, warnings } = parsePythonSubset(source, state);
      applyState(next);
      warnings.forEach((w) => pushLine({ kind: 'sys', text: w }));
      pushLine({ kind: 'out', text: 'Parsed Python subset into experiment state.' });
    },
    [state, applyState, pushLine],
  );

  const evalStats = useMemo(() => {
    if (!trainedModel || !data) return null;
    return evaluate(trainedModel, data, state.compile.loss);
  }, [trainedModel, data, state.compile.loss]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="wordmark">
          learn<span>-dl</span>
        </div>
        <button
          className={`btn ghost ${mode === 'sandbox' ? 'active' : ''}`}
          onClick={exitLevel}
        >
          {t('sandbox', locale)}
        </button>
        <button
          className={`btn ghost ${mode === 'level' ? 'active' : ''}`}
          onClick={() => setShowBrowser(true)}
        >
          {t('levels', locale)}
        </button>
        <div className="topbar-spacer" />
        {level && (
          <span className="badge">
            {level.sequence}/{level.id}
          </span>
        )}
        {mode === 'level' && level && solvedMap[level.id] && (
          <span className="badge success">{t('solved', locale)}</span>
        )}
        <select
          value={state.library}
          onChange={(e) => {
            const lib = e.target.value as LibraryId;
            applyState({ ...cloneExperiment(state), library: lib }, false);
            setRailTab('code');
          }}
          aria-label="Preferred library"
        >
          <option value="pytorch">PyTorch</option>
          <option value="tensorflow">TensorFlow</option>
          <option value="keras">Keras</option>
        </select>
      </header>

      {mode === 'level' && level && (
        <LevelToolbar
          level={level}
          locale={locale}
          solved={!!solvedMap[level.id]}
          onHint={() => handleCommand('hint')}
          onGoal={() => handleCommand('show goal')}
          onSolution={() => handleCommand('show solution')}
          onExit={exitLevel}
          onNext={() => {
            const n = nextLevel(level.id);
            if (n) openLevel(n);
            else {
              exitLevel();
              pushLine({ kind: 'sys', text: 'All levels complete — sandbox unlocked. Type `levels` to replay.' });
            }
          }}
        />
      )}

      <div className="workspace">
        <main className="stage">
          <div className="stage-grid">
            <NetworkGraph state={state} width={560} height={Math.max(280, 72 + state.layers.length * 56)} />
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              <TrainCurves history={state.training.history} width={360} height={200} />
              <div>
                <DecisionBoundary model={trainedModel} data={data} width={360} height={200} />
                <p className="muted" style={{ marginTop: 8, fontFamily: 'var(--mono)', fontSize: 12 }}>
                  {evalStats
                    ? `eval loss=${evalStats.loss.toFixed(4)} acc=${((evalStats.accuracy ?? 0) * 100).toFixed(1)}%`
                    : 'train to paint the decision field'}
                </p>
              </div>
            </div>
          </div>
        </main>

        <aside className="rail">
          <div className="rail-tabs">
            {(
              [
                ['goal', t('goal', locale)],
                ['code', t('code', locale)],
                ['builder', t('builder', locale)],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                className={`btn ghost ${railTab === id ? 'active' : ''}`}
                onClick={() => setRailTab(id)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="rail-body">
            {railTab === 'goal' && <GoalPanel level={level} showSolution={showSolution} />}
            {railTab === 'code' && <CodePanels state={state} onParse={onParse} />}
            {railTab === 'builder' && <VisualBuilder state={state} onChange={(s) => applyState(s)} />}
          </div>
        </aside>

        <section className="dock">
          <Terminal lines={lines} onSubmit={handleCommand} />
        </section>
      </div>

      {showBrowser && (
        <LevelsBrowser
          solved={solvedMap}
          onOpen={openLevel}
          onClose={() => setShowBrowser(false)}
        />
      )}

      {showDialog && level?.startDialog && (
        <Dialog
          views={level.startDialog}
          onClose={() => setShowDialog(false)}
          onCommand={(cmd) => handleCommand(cmd)}
        />
      )}

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
