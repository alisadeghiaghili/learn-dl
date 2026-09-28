import type { ExperimentState, LayerSpec } from '../state/types';

export type LocaleId = 'en_US' | 'fa' | 'de_DE';

export type I18nText = Partial<Record<LocaleId, string>> & { en_US: string };

export type DialogView =
  | { type: 'ModalAlert'; options: { markdowns: string[] } }
  | {
      type: 'DemoView';
      options: {
        beforeMarkdowns: string[];
        afterMarkdowns: string[];
        command: string;
      };
    }
  | {
      type: 'CodeCompare';
      options: {
        markdowns: string[];
        highlight?: 'pytorch' | 'tensorflow' | 'keras';
      };
    };

export interface GoalMetrics {
  accuracyAtLeast?: number;
  lossAtMost?: number;
  epochsAtLeast?: number;
}

export interface GoalSpec {
  /** Exact architecture match (order-sensitive). */
  layers?: LayerSpec[];
  /** Custom checks. */
  asserts?: ((s: ExperimentState) => boolean)[];
  metrics?: GoalMetrics;
  /** Require compile.loss */
  loss?: ExperimentState['compile']['loss'];
  optimizer?: ExperimentState['compile']['optimizer'];
  dataset?: ExperimentState['dataset']['id'];
}

export interface Level {
  id: string;
  sequence: string;
  name: I18nText;
  hint: I18nText;
  /** Short objective for the toolbar. */
  objective?: I18nText;
  startState: ExperimentState;
  goal: GoalSpec;
  solutionCommand: string;
  solutionCode?: { pytorch: string; tensorflow: string; keras: string };
  startDialog?: DialogView[];
  disabledCommands?: Record<string, true>;
  golf?: { bestCommands: number };
  /** Skip live training checks (pure architecture/code levels). */
  pureConcept?: boolean;
}

export interface SequenceInfo {
  id: string;
  displayName: I18nText;
  about: I18nText;
  tab: 'main' | 'libraries';
}
